"""Search stronger offline ML families for the breakout trade log.

Research-only: this uses sklearn models directly in validation, not live/bot
inference. It tests whether tree/boosting models improve ranking/filtering over
the linear artifact before we consider exporting anything.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from train_model import build_xy
from walk_forward_ml_validate import FOLDS, _fit_scores, _run_variant, _load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ML family walk-forward search.")
    parser.add_argument("--input-csv", default="backtest_trades_massive_context_exhaustion.csv")
    parser.add_argument("--output", default="ml_family_search_summary.csv")
    parser.add_argument("--tail-r", type=float, default=1.5)
    parser.add_argument("--bad-r", type=float, default=-0.5)
    return parser.parse_args()


def _model_specs() -> list[tuple[str, object, object]]:
    return [
        ("linear", Ridge(alpha=1.0), LogisticRegression(max_iter=5000)),
        (
            "rf",
            RandomForestRegressor(n_estimators=300, min_samples_leaf=8, random_state=42, n_jobs=-1),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=8, random_state=42, n_jobs=-1),
        ),
        (
            "extra",
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=8, random_state=42, n_jobs=-1),
            ExtraTreesClassifier(n_estimators=300, min_samples_leaf=8, random_state=42, n_jobs=-1),
        ),
        (
            "gb",
            GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, random_state=42),
            GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.03, random_state=42),
        ),
        (
            "histgb",
            HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15, l2_regularization=0.1, random_state=42),
            HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, l2_regularization=0.1, random_state=42),
        ),
    ]


def _score_family(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    family: str,
    reg_model: object,
    clf_model: object,
    tail_r: float,
    bad_r: float,
) -> pd.DataFrame:
    X_train, _, _ = build_xy(train_df)
    X_test, _, _ = build_xy(test_df)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if family in {"rf", "extra", "gb", "histgb"}:
        X_train_fit = X_train
        X_test_fit = X_test
    else:
        X_train_fit = X_train_s
        X_test_fit = X_test_s

    scored = test_df.copy()
    r_train = train_df["r_multiple"].astype(float).values
    reg_model.fit(X_train_fit, np.clip(r_train, -1.0, 5.0))
    scored[f"{family}_expected_r"] = reg_model.predict(X_test_fit)

    y_tail = (r_train >= tail_r).astype(int)
    if len(set(y_tail.tolist())) > 1:
        clf_model.fit(X_train_fit, y_tail)
        scored[f"{family}_tail"] = clf_model.predict_proba(X_test_fit)[:, 1]
    else:
        scored[f"{family}_tail"] = 0.0

    y_bad = (r_train <= bad_r).astype(int)
    if len(set(y_bad.tolist())) > 1:
        # Use a separate fresh classifier of the same class for bad probability.
        bad_clf = clf_model.__class__(**clf_model.get_params())
        bad_clf.fit(X_train_fit, y_bad)
        scored[f"{family}_bad"] = bad_clf.predict_proba(X_test_fit)[:, 1]
    else:
        scored[f"{family}_bad"] = 0.0
    scored[f"{family}_not_bad"] = 1.0 - scored[f"{family}_bad"]
    scored[f"{family}_composite"] = (
        scored[f"{family}_expected_r"] + 0.35 * scored[f"{family}_tail"] - 0.35 * scored[f"{family}_bad"]
    )
    return scored


def main() -> int:
    args = _parse_args()
    df = _load(args.input_csv)
    rows: list[dict] = []
    for fold_index, (start, end) in enumerate(FOLDS, start=1):
        fold_start = pd.Timestamp(start, tz="UTC")
        fold_end = pd.Timestamp(end, tz="UTC")
        train_df = df[df["entry_ts"] < fold_start].copy()
        test_df = df[(df["entry_ts"] >= fold_start) & (df["entry_ts"] < fold_end)].copy()
        if len(train_df) < 30 or len(test_df) < 10:
            continue
        baseline_result = _run_variant(test_df, "baseline")
        baseline_result.update({"fold": fold_index, "fold_start": start, "fold_end": end, "family": "none"})
        rows.append(baseline_result)
        for family, reg_model, clf_model in _model_specs():
            scored = _score_family(train_df, test_df, family, reg_model, clf_model, args.tail_r, args.bad_r)
            variants = [
                (f"{family}-rank-tail", f"{family}_tail", None, True),
                (f"{family}-rank-composite", f"{family}_composite", None, True),
                (f"{family}-rank-expected-r", f"{family}_expected_r", None, True),
                (f"{family}-expected-r-20", f"{family}_expected_r", 0.20, False),
                (f"{family}-composite-10", f"{family}_composite", 0.10, False),
                (f"{family}-tail-05", f"{family}_tail", 0.05, False),
                (f"{family}-tail-10", f"{family}_tail", 0.10, False),
                (f"{family}-not-bad-55", f"{family}_not_bad", 0.55, False),
            ]
            for name, score_column, threshold, rank in variants:
                result = _run_variant(scored, name, score_column, threshold, rank)
                result.update({"fold": fold_index, "fold_start": start, "fold_end": end, "family": family})
                rows.append(result)

    fields = [
        "fold",
        "fold_start",
        "fold_end",
        "family",
        "config",
        "generated",
        "taken",
        "final_equity",
        "roi_pct",
        "win_rate",
        "expectancy_r",
        "profit_factor",
        "max_dd",
    ]
    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = pd.DataFrame(rows)
    baseline = summary[summary["config"] == "baseline"].set_index("fold")["final_equity"]
    agg = summary.groupby("config").agg(
        folds=("fold", "count"),
        avg_final_equity=("final_equity", "mean"),
        median_final_equity=("final_equity", "median"),
        avg_expectancy_r=("expectancy_r", "mean"),
        avg_profit_factor=("profit_factor", "mean"),
        avg_max_dd=("max_dd", "mean"),
    )
    wins_vs_baseline = {}
    for config, group in summary.groupby("config"):
        wins_vs_baseline[config] = sum(
            row.final_equity > baseline.get(row.fold, float("inf"))
            for row in group.itertuples()
        )
    agg["wins_vs_baseline"] = pd.Series(wins_vs_baseline)
    agg = agg.sort_values(["wins_vs_baseline", "avg_final_equity"], ascending=False)
    agg_out = out.with_name(out.stem + "_agg.csv")
    agg.to_csv(agg_out)
    print(agg.head(25).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {out} and {agg_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
