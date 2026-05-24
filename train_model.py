"""Offline ML training for breakout-signal quality.

Default mode preserves the original W1+W2 -> W3 walk-forward test:

    python train_model.py

For larger experiments, generate one big backtest trade log and split it
chronologically:

    python train_model.py --input-csv backtest_trades_big.csv --split-date 2026-04-01

Or train on explicit historical logs and test on an untouched holdout:

    python train_model.py --train-csv trades_2025.csv --test-csv backtest_trades_W3.csv

The saved artifact is a pure-Python linear scorer for optional backtester
experiments and, when trained with --feature-set live, the live scanner's
ranking-only --ml-rank-model path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SIGNAL_FEATURES_NUMERIC = [
    "feat_score",
    "feat_breakout_pct",
    "feat_distance_to_trigger_pct",
    "feat_risk_pct",
    "feat_reward_pct",
    "feat_reward_risk",
    "feat_volume_ratio",
    "feat_avg_quote_volume",
    "feat_compression_pct",
    "feat_atr_pct",
    "feat_trend_score",
    "feat_close_position",
    "feat_quote_volume_24h",
    "feat_range_pct_24h",
    "feat_price_change_pct_24h",
    "feat_momentum_score",
]
CONTEXT_FEATURES_NUMERIC = [
    "feat_btc_momentum_pct",
    "feat_btc_ema_distance_pct",
    "feat_rel_momentum_pct",
    "feat_symbol_trades_30",
    "feat_symbol_win_rate_30",
    "feat_symbol_avg_r_30",
]
FEATURES_NUMERIC = SIGNAL_FEATURES_NUMERIC + CONTEXT_FEATURES_NUMERIC
# INSTANT is the implicit baseline; encode the other regimes.
REGIME_DUMMIES = ["RETEST", "STRICT_RETEST"]
DEFAULT_TRAIN_PATHS = ["backtest_trades_W1.csv", "backtest_trades_W2.csv"]
DEFAULT_TEST_PATHS = ["backtest_trades_W3.csv"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate an offline ML filter.")
    parser.add_argument(
        "--train-csv",
        action="append",
        help="Training CSV path. Repeat the flag or use comma-separated paths.",
    )
    parser.add_argument(
        "--test-csv",
        action="append",
        help="Holdout CSV path. Repeat the flag or use comma-separated paths.",
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        help="CSV path(s) to concatenate and split chronologically into train/test.",
    )
    parser.add_argument("--split-date", help="UTC date/time where --input-csv switches from train to test.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Tail fraction used as holdout when --input-csv is supplied without --split-date.",
    )
    parser.add_argument("--output", default="model.json", help="Output model artifact path.")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades required in train and test.")
    parser.add_argument("--tail-r", type=float, default=1.5, help="Tail target: classify trades with R >= this as right-tail winners.")
    parser.add_argument("--bad-r", type=float, default=-0.5, help="Bad-trade target: classify trades with R <= this as veto candidates.")
    parser.add_argument("--r-clip-min", type=float, default=-1.0, help="Clip R target below this value for expected-R regression.")
    parser.add_argument("--r-clip-max", type=float, default=5.0, help="Clip R target above this value for expected-R regression.")
    parser.add_argument(
        "--label-all-signals",
        action="store_true",
        help=(
            "Use every generated signal row and label by r_multiple > 0. "
            "Default keeps only portfolio-taken rows with WIN/LOSS outcomes."
        ),
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Optional logistic-regression class weighting for imbalanced big datasets.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["full", "live"],
        default="full",
        help="full uses signal plus offline context features; live uses only features available in the live scanner.",
    )
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.min_trades < 2:
        parser.error("--min-trades must be at least 2")
    if args.input_csv and (args.train_csv or args.test_csv):
        parser.error("--input-csv cannot be combined with --train-csv/--test-csv")
    if (args.train_csv and not args.test_csv) or (args.test_csv and not args.train_csv):
        parser.error("--train-csv and --test-csv must be supplied together")
    return args


def _expand_paths(values: list[str] | None, fallback: list[str] | None = None) -> list[str]:
    if not values:
        return list(fallback or [])
    paths: list[str] = []
    for value in values:
        paths.extend(part.strip() for part in value.split(",") if part.strip())
    return paths


def load_window(path: str, label_all_signals: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_all_signals:
        if "r_multiple" not in df.columns:
            raise ValueError(f"{path}: expected column 'r_multiple' for --label-all-signals")
        df = df.copy()
        df["r_multiple"] = pd.to_numeric(df["r_multiple"], errors="coerce")
        df = df.dropna(subset=["r_multiple"])
        df["outcome"] = np.where(df["r_multiple"] > 0.0, "WIN", "LOSS")
    else:
        if "taken" not in df.columns or "outcome" not in df.columns:
            raise ValueError(f"{path}: expected columns 'taken' and 'outcome'")
        df = df[df["taken"].astype(str).str.lower() == "yes"].copy()
        df = df[df["outcome"].isin(["WIN", "LOSS"])].copy()
    df["source_file"] = path
    for column in FEATURES_NUMERIC:
        if column not in df.columns:
            df[column] = 0.0
    if "regime" not in df.columns:
        df["regime"] = "INSTANT"
    if "r_multiple" not in df.columns:
        df["r_multiple"] = 0.0
    return df


def load_many(paths: list[str], label_all_signals: bool = False) -> pd.DataFrame:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing input CSV(s): {', '.join(missing)}")
    frames = [load_window(path, label_all_signals=label_all_signals) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _timestamp_column(df: pd.DataFrame) -> str:
    if "entry_date" in df.columns:
        return "entry_date"
    if "detected_date" in df.columns:
        return "detected_date"
    raise ValueError("input CSV needs entry_date or detected_date for chronological splitting")


def split_chronologically(
    df: pd.DataFrame,
    split_date: str | None,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ts_col = _timestamp_column(df)
    ordered = df.copy()
    ordered["_ts"] = pd.to_datetime(ordered[ts_col], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    if split_date:
        split_ts = pd.Timestamp(split_date, tz="UTC")
        train_df = ordered[ordered["_ts"] < split_ts].copy()
        test_df = ordered[ordered["_ts"] >= split_ts].copy()
        split_label = f"split-date {split_ts.isoformat()}"
    else:
        split_index = int(len(ordered) * (1.0 - test_fraction))
        split_index = min(max(split_index, 1), max(len(ordered) - 1, 1))
        train_df = ordered.iloc[:split_index].copy()
        test_df = ordered.iloc[split_index:].copy()
        split_label = f"tail test-fraction {test_fraction:.2f}"
    return train_df.drop(columns=["_ts"]), test_df.drop(columns=["_ts"]), split_label


def build_xy(
    df: pd.DataFrame,
    features_numeric: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features_numeric = features_numeric or FEATURES_NUMERIC
    X_num = df[features_numeric].astype(float).values
    X_regime = np.column_stack([
        (df["regime"] == r).astype(float).values for r in REGIME_DUMMIES
    ])
    X = np.hstack([X_num, X_regime])
    y = (df["outcome"] == "WIN").astype(int).values
    feature_names = features_numeric + [f"regime_{r}" for r in REGIME_DUMMIES]
    return X, y, feature_names


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _auc_label(auc: float) -> str:
    if math.isnan(auc):
        return "(one class only)"
    if auc < 0.55:
        return "(no signal)"
    if auc < 0.60:
        return "(weak signal)"
    return "(strong signal)"


def report_model(
    name: str,
    y_train: np.ndarray,
    p_train: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
) -> dict:
    pred_test = (p_test >= 0.5).astype(int)
    auc_train = _safe_auc(y_train, p_train)
    auc_test = _safe_auc(y_test, p_test)
    acc = accuracy_score(y_test, pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_test, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, pred_test)
    print(f"\n=== {name} ===")
    print(f"  Train AUC:        {auc_train:.3f} {_auc_label(auc_train)}")
    print(f"  Test AUC:         {auc_test:.3f} {_auc_label(auc_test)}")
    print(f"  Test accuracy:    {acc:.3f}")
    print(f"  Confusion matrix [[TN FP] [FN TP]]:\n    {cm.tolist()}")
    print(f"  WIN precision/recall/F1: {precision:.3f} / {recall:.3f} / {f1:.3f}")
    return {
        "train_auc": float(auc_train),
        "test_auc": float(auc_test),
        "test_accuracy": float(acc),
        "win_precision": float(precision),
        "win_recall": float(recall),
        "win_f1": float(f1),
    }


def _linear_head(kind: str, intercept: float, coef: np.ndarray, metrics: dict | None = None) -> dict:
    return {
        "kind": kind,
        "intercept": float(intercept),
        "coef": coef.tolist(),
        "metrics": metrics or {},
    }


def _fit_logistic_head(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_test_s: np.ndarray,
    y_test: np.ndarray,
    name: str,
    class_weight: str,
) -> tuple[dict | None, np.ndarray | None, dict | None]:
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        print(f"\n=== {name} ===")
        print("  skipped: train or test contains one class only")
        return None, None, None
    clf = LogisticRegression(
        max_iter=5000,
        C=1.0,
        class_weight="balanced" if class_weight == "balanced" else None,
    )
    clf.fit(X_train_s, y_train)
    p_train = clf.predict_proba(X_train_s)[:, 1]
    p_test = clf.predict_proba(X_test_s)[:, 1]
    metrics = report_model(name, y_train, p_train, y_test, p_test)
    head = _linear_head("logistic", float(clf.intercept_[0]), clf.coef_[0], metrics)
    return head, p_test, metrics


def _fit_ridge_head(
    X_train_s: np.ndarray,
    r_train: np.ndarray,
    X_test_s: np.ndarray,
    r_test: np.ndarray,
    name: str,
) -> tuple[dict, np.ndarray, dict]:
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_s, r_train)
    p_train = reg.predict(X_train_s)
    p_test = reg.predict(X_test_s)
    corr_train = float(np.corrcoef(r_train, p_train)[0, 1]) if len(r_train) > 1 else float("nan")
    corr_test = float(np.corrcoef(r_test, p_test)[0, 1]) if len(r_test) > 1 else float("nan")
    metrics = {
        "train_mae": float(mean_absolute_error(r_train, p_train)),
        "test_mae": float(mean_absolute_error(r_test, p_test)),
        "train_rmse": float(mean_squared_error(r_train, p_train) ** 0.5),
        "test_rmse": float(mean_squared_error(r_test, p_test) ** 0.5),
        "train_r2": float(r2_score(r_train, p_train)),
        "test_r2": float(r2_score(r_test, p_test)),
        "train_corr": corr_train,
        "test_corr": corr_test,
    }
    print(f"\n=== {name} ===")
    print(f"  Test MAE/RMSE:    {metrics['test_mae']:.3f} / {metrics['test_rmse']:.3f}")
    print(f"  Test R2/corr:     {metrics['test_r2']:.3f} / {metrics['test_corr']:.3f}")
    head = _linear_head("linear", float(reg.intercept_), reg.coef_, metrics)
    return head, p_test, metrics


def _market_regime_rows(df: pd.DataFrame) -> pd.Series:
    mom = pd.to_numeric(df.get("feat_btc_momentum_pct", 0.0), errors="coerce").fillna(0.0)
    ema = pd.to_numeric(df.get("feat_btc_ema_distance_pct", 0.0), errors="coerce").fillna(0.0)
    regime = pd.Series("FLAT", index=df.index, dtype="object")
    regime[(mom >= 2.0) & (ema >= 0.0)] = "UP"
    regime[(mom <= -2.0) & (ema <= 0.0)] = "DOWN"
    return regime


def _fit_regime_heads(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    r_train: np.ndarray,
    r_test: np.ndarray,
    class_weight: str,
    min_rows: int,
) -> dict:
    train_regime = _market_regime_rows(train_df).reset_index(drop=True)
    test_regime = _market_regime_rows(test_df).reset_index(drop=True)
    y_train_win = (train_df["outcome"].reset_index(drop=True) == "WIN").astype(int).values
    y_test_win = (test_df["outcome"].reset_index(drop=True) == "WIN").astype(int).values
    heads_by_regime: dict[str, dict] = {}
    for regime in ("UP", "FLAT", "DOWN"):
        tr_mask = (train_regime == regime).values
        te_mask = (test_regime == regime).values
        if tr_mask.sum() < min_rows or te_mask.sum() < max(10, min_rows // 3):
            continue
        print(f"\n--- REGIME HEAD {regime}: train={tr_mask.sum()} test={te_mask.sum()} ---")
        r_head, _, r_metrics = _fit_ridge_head(
            X_train_s[tr_mask],
            r_train[tr_mask],
            X_test_s[te_mask],
            r_test[te_mask],
            f"RIDGE EXPECTED-R ({regime})",
        )
        p_head, _, p_metrics = _fit_logistic_head(
            X_train_s[tr_mask],
            y_train_win[tr_mask],
            X_test_s[te_mask],
            y_test_win[te_mask],
            f"LOGISTIC P(WIN) ({regime})",
            class_weight,
        )
        heads_by_regime[regime] = {
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
            "expected_r": r_head,
            "pwin": p_head,
            "metrics_expected_r": r_metrics,
            "metrics_pwin": p_metrics,
        }
    return heads_by_regime


def report_thresholds(
    test_df: pd.DataFrame,
    p_test: np.ndarray,
    label: str = "P(WIN)",
    thresholds: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65),
) -> None:
    y = (test_df["outcome"] == "WIN").astype(int).values
    r = test_df["r_multiple"].astype(float).values
    print(f"\n=== TEST FILTER THRESHOLDS: {label} (trade-level, before portfolio slotting) ===")
    print("  thr   kept  kept_win  kept_avg_R   rejected  rejected_win  rejected_avg_R")
    for threshold in thresholds:
        kept = p_test >= threshold
        rejected = ~kept
        kept_win = y[kept].mean() * 100.0 if kept.any() else 0.0
        rej_win = y[rejected].mean() * 100.0 if rejected.any() else 0.0
        kept_r = r[kept].mean() if kept.any() else 0.0
        rej_r = r[rejected].mean() if rejected.any() else 0.0
        print(
            f"  {threshold:.2f}  {kept.sum():5d}  {kept_win:7.1f}%  {kept_r:+10.3f}"
            f"  {rejected.sum():9d}  {rej_win:10.1f}%  {rej_r:+13.3f}"
        )


def _date_span(df: pd.DataFrame) -> str:
    try:
        ts_col = _timestamp_column(df)
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
        if ts.empty:
            return "unknown dates"
        return f"{ts.min().strftime('%Y-%m-%d')}..{ts.max().strftime('%Y-%m-%d')}"
    except ValueError:
        return "unknown dates"


def _validate_dataset(name: str, df: pd.DataFrame, min_trades: int) -> None:
    if len(df) < min_trades:
        raise ValueError(f"{name}: only {len(df)} trades after filtering; need at least {min_trades}")
    outcomes = set(df["outcome"].astype(str))
    if not {"WIN", "LOSS"}.issubset(outcomes):
        raise ValueError(f"{name}: needs both WIN and LOSS rows, got {sorted(outcomes)}")


def main() -> int:
    args = _parse_args()
    features_numeric = SIGNAL_FEATURES_NUMERIC if args.feature_set == "live" else FEATURES_NUMERIC
    input_paths = _expand_paths(args.input_csv)
    train_paths = _expand_paths(args.train_csv, DEFAULT_TRAIN_PATHS)
    test_paths = _expand_paths(args.test_csv, DEFAULT_TEST_PATHS)

    try:
        if input_paths:
            full_df = load_many(input_paths, label_all_signals=args.label_all_signals)
            train_df, test_df, split_label = split_chronologically(
                full_df, args.split_date, args.test_fraction
            )
        else:
            train_df = load_many(train_paths, label_all_signals=args.label_all_signals)
            test_df = load_many(test_paths, label_all_signals=args.label_all_signals)
            split_label = "explicit train/test CSVs"
        _validate_dataset("train", train_df, args.min_trades)
        _validate_dataset("test", test_df, args.min_trades)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 2

    print(f"Split:         {split_label}")
    print(f"Train paths:   {input_paths or train_paths}")
    print(f"Test paths:    {input_paths if input_paths else test_paths}")
    print(f"Train trades:  {len(train_df)}  dates={_date_span(train_df)}  outcomes={train_df['outcome'].value_counts().to_dict()}")
    print(f"Test trades:   {len(test_df)}   dates={_date_span(test_df)}  outcomes={test_df['outcome'].value_counts().to_dict()}")

    X_train, y_train, feature_names = build_xy(train_df, features_numeric)
    X_test, y_test, _ = build_xy(test_df, features_numeric)
    r_train_raw = train_df["r_multiple"].astype(float).values
    r_test_raw = test_df["r_multiple"].astype(float).values
    r_train = np.clip(r_train_raw, args.r_clip_min, args.r_clip_max)
    r_test = np.clip(r_test_raw, args.r_clip_min, args.r_clip_max)

    print(f"Features ({len(feature_names)}): {feature_names}")

    majority = int(y_train.mean() >= 0.5)
    base_acc = accuracy_score(y_test, [majority] * len(y_test))
    base_win_rate = y_test.mean()
    print(f"\nBaseline majority-class accuracy: {base_acc:.3f}   (test win rate: {base_win_rate:.3f})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=5000,
        C=1.0,
        class_weight="balanced" if args.class_weight == "balanced" else None,
    )
    lr.fit(X_train_s, y_train)
    p_train_lr = lr.predict_proba(X_train_s)[:, 1]
    p_test_lr = lr.predict_proba(X_test_s)[:, 1]
    lr_metrics = report_model("LOGISTIC REGRESSION", y_train, p_train_lr, y_test, p_test_lr)

    print("\n  LR coefficients (standardized; magnitude ranks predictive strength):")
    coefs = sorted(zip(feature_names, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, c in coefs:
        print(f"    {name:<35s}  {c:+.4f}")
    print(f"    {'(intercept)':<35s}  {float(lr.intercept_[0]):+.4f}")

    gbt = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42, subsample=0.8
    )
    gbt.fit(X_train, y_train)
    p_train_gbt = gbt.predict_proba(X_train)[:, 1]
    p_test_gbt = gbt.predict_proba(X_test)[:, 1]
    gbt_metrics = report_model("GRADIENT BOOSTED TREES", y_train, p_train_gbt, y_test, p_test_gbt)

    print("\n  GBT feature importances (top 10):")
    importances = sorted(zip(feature_names, gbt.feature_importances_), key=lambda x: x[1], reverse=True)
    for name, imp in importances[:10]:
        print(f"    {name:<35s}  {imp:.4f}")

    expected_r_head, p_test_expected_r, expected_r_metrics = _fit_ridge_head(
        X_train_s,
        r_train,
        X_test_s,
        r_test,
        "RIDGE EXPECTED-R",
    )
    tail_y_train = (r_train_raw >= args.tail_r).astype(int)
    tail_y_test = (r_test_raw >= args.tail_r).astype(int)
    tail_head, p_test_tail, tail_metrics = _fit_logistic_head(
        X_train_s,
        tail_y_train,
        X_test_s,
        tail_y_test,
        f"LOGISTIC RIGHT-TAIL R>={args.tail_r:g}",
        args.class_weight,
    )
    bad_y_train = (r_train_raw <= args.bad_r).astype(int)
    bad_y_test = (r_test_raw <= args.bad_r).astype(int)
    bad_head, p_test_bad, bad_metrics = _fit_logistic_head(
        X_train_s,
        bad_y_train,
        X_test_s,
        bad_y_test,
        f"LOGISTIC BAD-TRADE R<={args.bad_r:g}",
        args.class_weight,
    )

    regime_heads = _fit_regime_heads(
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        X_train_s,
        X_test_s,
        r_train,
        r_test,
        args.class_weight,
        args.min_trades,
    )

    test_df_report = test_df.reset_index(drop=True)
    report_thresholds(test_df_report, p_test_lr, "P(WIN)")
    report_thresholds(test_df_report, p_test_expected_r, "EXPECTED-R", (-0.10, 0.00, 0.10, 0.20, 0.30))
    if p_test_tail is not None:
        report_thresholds(test_df_report, p_test_tail, f"P(R>={args.tail_r:g})")
    if p_test_bad is not None:
        report_thresholds(test_df_report, 1.0 - p_test_bad, f"1-P(R<={args.bad_r:g})")

    artifact = {
        "format": "linear-ml-experiment-v2",
        "feature_names_numeric": features_numeric,
        "feature_names_regime_dummies": REGIME_DUMMIES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "lr_intercept": float(lr.intercept_[0]),
        "lr_coef": lr.coef_[0].tolist(),
        "heads": {
            "pwin": _linear_head("logistic", float(lr.intercept_[0]), lr.coef_[0], lr_metrics),
            "expected_r": expected_r_head,
            "tail": tail_head,
            "bad": bad_head,
        },
        "regime_heads": regime_heads,
        "metrics_lr": lr_metrics,
        "metrics_gbt": gbt_metrics,
        "metrics_expected_r": expected_r_metrics,
        "metrics_tail": tail_metrics,
        "metrics_bad": bad_metrics,
        "metrics_baseline_accuracy": float(base_acc),
        "test_win_rate": float(base_win_rate),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "tail_r": float(args.tail_r),
        "bad_r": float(args.bad_r),
        "r_clip_min": float(args.r_clip_min),
        "r_clip_max": float(args.r_clip_max),
        "train_paths": train_paths if not input_paths else [],
        "test_paths": test_paths if not input_paths else [],
        "input_paths": input_paths,
        "split": split_label,
        "class_weight": args.class_weight,
        "label_all_signals": bool(args.label_all_signals),
        "feature_set": args.feature_set,
    }
    out = Path(args.output)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nLR artifact saved to {out}")

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    lr_auc = lr_metrics["test_auc"]
    gbt_auc = gbt_metrics["test_auc"]
    valid_aucs = [auc for auc in (lr_auc, gbt_auc) if not math.isnan(auc)]
    best_auc = max(valid_aucs) if valid_aucs else float("nan")
    if math.isnan(best_auc) or best_auc < 0.55:
        print("Neither model meaningfully beats random on the held-out window.")
        print("Recommendation: do NOT ship an ML filter; honest null result.")
    elif best_auc < 0.60:
        better = "LR" if lr_auc >= gbt_auc else "GBT"
        print(f"Weak signal only. Best model: {better} (test AUC {best_auc:.3f}).")
        print("Keep it experimental until a backtest with --ml-filter-model improves the holdout portfolio.")
    else:
        better = "LR" if lr_auc >= gbt_auc else "GBT"
        print(f"Promising classifier signal. Best model: {better} (test AUC {best_auc:.3f}).")
        print("Still require a clean holdout portfolio backtest before considering live use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
