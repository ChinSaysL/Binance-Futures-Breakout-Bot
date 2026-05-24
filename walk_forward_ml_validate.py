"""Walk-forward validation for experimental ML ranking/filter ideas.

This is research-only. It reads a full baseline trade log with feature snapshots,
trains only on rows before each fold, scores the next fold, and replays the
portfolio simulator on that fold. It does not touch live defaults.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from screener.backtest import BacktestTrade, _simulate_portfolio
from train_model import FEATURES_NUMERIC, SIGNAL_FEATURES_NUMERIC, build_xy


FOLDS = [
    ("2025-09-01", "2025-10-15"),
    ("2025-10-15", "2025-12-01"),
    ("2025-12-01", "2026-01-15"),
    ("2026-01-15", "2026-03-01"),
    ("2026-03-01", "2026-04-01"),
    ("2026-04-01", "2026-05-21"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline walk-forward ML validation.")
    parser.add_argument("--input-csv", default="backtest_trades_massive_context_baseline.csv")
    parser.add_argument("--output", default="ml_walk_forward_summary.csv")
    parser.add_argument("--tail-r", type=float, default=1.5)
    parser.add_argument("--bad-r", type=float, default=-0.5)
    parser.add_argument("--r-clip-min", type=float, default=-1.0)
    parser.add_argument("--r-clip-max", type=float, default=5.0)
    parser.add_argument(
        "--feature-set",
        choices=["full", "live"],
        default="full",
        help="full uses offline context features; live uses only features available to the live scanner.",
    )
    return parser.parse_args()


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_ts"] = pd.to_datetime(df["entry_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").reset_index(drop=True)
    df["r_multiple"] = pd.to_numeric(df["r_multiple"], errors="coerce")
    df = df.dropna(subset=["r_multiple"]).copy()
    df["outcome"] = np.where(df["r_multiple"] > 0.0, "WIN", "LOSS")
    for column in FEATURES_NUMERIC:
        if column not in df.columns:
            df[column] = 0.0
    if "regime" not in df.columns:
        df["regime"] = "INSTANT"
    return df


def _fit_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tail_r: float,
    bad_r: float,
    r_clip_min: float,
    r_clip_max: float,
    features_numeric: list[str],
) -> pd.DataFrame:
    X_train, y_train, _ = build_xy(train_df, features_numeric)
    X_test, _, _ = build_xy(test_df, features_numeric)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    scored = test_df.copy()
    r_train_raw = train_df["r_multiple"].astype(float).values
    r_train = np.clip(r_train_raw, r_clip_min, r_clip_max)

    pwin = LogisticRegression(max_iter=5000)
    pwin.fit(X_train_s, y_train)
    scored["ml_pwin"] = pwin.predict_proba(X_test_s)[:, 1]

    expected_r = Ridge(alpha=1.0)
    expected_r.fit(X_train_s, r_train)
    scored["ml_expected_r"] = expected_r.predict(X_test_s)

    y_tail = (r_train_raw >= tail_r).astype(int)
    if len(set(y_tail.tolist())) > 1:
        tail = LogisticRegression(max_iter=5000)
        tail.fit(X_train_s, y_tail)
        scored["ml_tail"] = tail.predict_proba(X_test_s)[:, 1]
    else:
        scored["ml_tail"] = 0.0

    y_bad = (r_train_raw <= bad_r).astype(int)
    if len(set(y_bad.tolist())) > 1:
        bad = LogisticRegression(max_iter=5000)
        bad.fit(X_train_s, y_bad)
        scored["ml_bad"] = bad.predict_proba(X_test_s)[:, 1]
    else:
        scored["ml_bad"] = 0.0
    scored["ml_not_bad"] = 1.0 - scored["ml_bad"]
    scored["ml_composite"] = scored["ml_expected_r"] + 0.35 * scored["ml_tail"] - 0.35 * scored["ml_bad"]
    return scored


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _trades_from_rows(df: pd.DataFrame, score_column: str = "") -> list[BacktestTrade]:
    trades: list[BacktestTrade] = []
    for _, row in df.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        exit_ts = pd.to_datetime(row.get("exited_date"), utc=True, errors="coerce")
        detected_ts = pd.to_datetime(row.get("detected_date"), utc=True, errors="coerce")
        hold_candles = int(_as_float(row.get("hold_candles"), 0.0))
        score = _as_float(row.get(score_column), 0.0) if score_column else 0.0
        trades.append(
            BacktestTrade(
                symbol=str(row.get("symbol", "")),
                side=str(row.get("side", "LONG")),
                status=str(row.get("status", "BREAKOUT")),
                regime=str(row.get("regime", "INSTANT")),
                detected_time=_ms(detected_ts if not pd.isna(detected_ts) else entry_ts),
                entry_time=_ms(entry_ts),
                exit_time=_ms(exit_ts if not pd.isna(exit_ts) else entry_ts),
                entry_price=_as_float(row.get("entry")),
                stop_price=_as_float(row.get("stop")),
                target_price=_as_float(row.get("target")),
                avg_exit_price=_as_float(row.get("exit_price")),
                leverage=int(_as_float(row.get("leverage"), 10.0)),
                hold_candles=hold_candles,
                hold_hours=float(hold_candles),
                r_multiple=_as_float(row.get("r_multiple")),
                price_return=_as_float(row.get("price_return_pct")) / 100.0,
                momentum_score=_as_float(row.get("feat_momentum_score")),
                ml_score=score,
            )
        )
    return trades


def _portfolio_args(rank: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        capital=10.0,
        compound=True,
        position_pct=20.0,
        max_concurrent=2,
        order_margin=5.0,
        fee_pct=0.045,
        funding_pct_per_8h=0.01,
        interval="1h",
        loss_cooldown_after=1,
        loss_cooldown_candles=48,
        sizing_mode="moonshot",
        instant_size_multiplier=0.5,
        retest_size_multiplier=0.9,
        trailing_retest_size_multiplier=0.5,
        ml_rank_signals=rank,
        ml_filter_start_ms=None,
    )


def _fresh(trades: list[BacktestTrade]) -> list[BacktestTrade]:
    return [
        replace(
            trade,
            taken=False,
            margin=0.0,
            net_pnl=0.0,
            fees_usdt=0.0,
            funding_usdt=0.0,
            position_pct=0.0,
        )
        for trade in trades
    ]


def _run_variant(test_df: pd.DataFrame, name: str, score_column: str = "", threshold: float | None = None, rank: bool = False) -> dict:
    candidate_df = test_df
    if threshold is not None:
        candidate_df = candidate_df[candidate_df[score_column] >= threshold].copy()
    trades = _trades_from_rows(candidate_df, score_column=score_column)
    taken, final_equity, max_dd = _simulate_portfolio(_fresh(trades), _portfolio_args(rank=rank))
    wins = [t for t in taken if t.is_win]
    losses = [t for t in taken if not t.is_win]
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)
    return {
        "config": name,
        "generated": len(candidate_df),
        "taken": len(taken),
        "final_equity": final_equity,
        "roi_pct": (final_equity / 10.0 - 1.0) * 100.0,
        "win_rate": (len(wins) / len(taken) * 100.0) if taken else 0.0,
        "expectancy_r": (sum(t.r_multiple for t in taken) / len(taken)) if taken else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_dd": max_dd,
    }


def main() -> int:
    args = _parse_args()
    df = _load(args.input_csv)
    features_numeric = SIGNAL_FEATURES_NUMERIC if args.feature_set == "live" else FEATURES_NUMERIC
    rows: list[dict] = []
    variants = [
        ("baseline", "", None, False),
        ("pwin-50", "ml_pwin", 0.50, False),
        ("expected-r-00", "ml_expected_r", 0.00, False),
        ("expected-r-10", "ml_expected_r", 0.10, False),
        ("expected-r-20", "ml_expected_r", 0.20, False),
        ("not-bad-50", "ml_not_bad", 0.50, False),
        ("not-bad-55", "ml_not_bad", 0.55, False),
        ("tail-05", "ml_tail", 0.05, False),
        ("tail-10", "ml_tail", 0.10, False),
        ("tail-15", "ml_tail", 0.15, False),
        ("composite-00", "ml_composite", 0.00, False),
        ("composite-10", "ml_composite", 0.10, False),
        ("rank-expected-r", "ml_expected_r", None, True),
        ("rank-tail", "ml_tail", None, True),
        ("rank-not-bad", "ml_not_bad", None, True),
        ("rank-composite", "ml_composite", None, True),
    ]

    for fold_index, (start, end) in enumerate(FOLDS, start=1):
        fold_start = pd.Timestamp(start, tz="UTC")
        fold_end = pd.Timestamp(end, tz="UTC")
        train_df = df[df["entry_ts"] < fold_start].copy()
        test_df = df[(df["entry_ts"] >= fold_start) & (df["entry_ts"] < fold_end)].copy()
        if len(train_df) < 30 or len(test_df) < 10:
            continue
        scored = _fit_scores(
            train_df,
            test_df,
            args.tail_r,
            args.bad_r,
            args.r_clip_min,
            args.r_clip_max,
            features_numeric,
        )
        for name, score_column, threshold, rank in variants:
            result = _run_variant(scored, name, score_column, threshold, rank)
            result.update(
                {
                    "fold": fold_index,
                    "fold_start": start,
                    "fold_end": end,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                }
            )
            rows.append(result)

    output = Path(args.output)
    fields = [
        "fold",
        "fold_start",
        "fold_end",
        "n_train",
        "n_test",
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
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = pd.DataFrame(rows)
    agg = summary.groupby("config").agg(
        folds=("fold", "count"),
        avg_final_equity=("final_equity", "mean"),
        median_final_equity=("final_equity", "median"),
        avg_expectancy_r=("expectancy_r", "mean"),
        avg_profit_factor=("profit_factor", "mean"),
        avg_max_dd=("max_dd", "mean"),
        wins_vs_baseline=("final_equity", "count"),
    )
    baseline = summary[summary["config"] == "baseline"].set_index("fold")["final_equity"]
    wins_vs_baseline: dict[str, int] = {}
    for config, group in summary.groupby("config"):
        wins_vs_baseline[config] = sum(
            row.final_equity > baseline.get(row.fold, float("inf"))
            for row in group.itertuples()
        )
    agg["wins_vs_baseline"] = pd.Series(wins_vs_baseline)
    agg = agg.sort_values(["wins_vs_baseline", "avg_final_equity"], ascending=False)
    agg_path = output.with_name(output.stem + "_agg.csv")
    agg.to_csv(agg_path)
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {output} and {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
