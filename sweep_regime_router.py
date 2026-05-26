"""Regime-router sweep across W1/W2/W3.

Tests whether different parameter sets should be routed by BTC regime instead
of forcing one config to serve downtrend, chop, and uptrend.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


WINDOWS = [
    ("W1", "2025-12-20", "2026-02-10", "downtrend"),
    ("W2", "2026-02-10", "2026-04-01", "chop"),
    ("W3", "2026-04-01", "2026-05-21", "uptrend"),
]

BASE = [
    sys.executable,
    "-m",
    "screener.backtest",
    "--top",
    "40",
    "--interval",
    "1h",
    "--capital",
    "10",
    "--compound",
    "--sizing-mode",
    "auto",
    "--max-concurrent",
    "2",
    "--trailing-stop",
    "--adaptive-trailing-callback",
    "--dynamic-sl",
    "--sl-lookback",
    "20",
    "--breakeven-trigger-r",
    "1.5",
    "--breakeven-offset-pct",
    "0.1",
    "--stagnation-after-r",
    "0.5",
    "--stagnation-candles",
    "12",
    "--max-sl-loss-pct",
    "35",
    "--simulate-rotation",
]

CONFIGS: list[tuple[str, list[str]]] = [
    ("baseline_long", []),
    ("include_shorts", ["--include-shorts"]),
    ("shorts_only", ["--shorts-only"]),
    ("shorts_only_no_short_guard", ["--shorts-only", "--no-short-market-guard"]),
    (
        "shorts_only_relaxed_guard",
        ["--shorts-only", "--short-guard-momentum-pct", "1.0", "--short-guard-ema-slack-pct", "1.5"],
    ),
    ("no_dynamic_sl", []),
    ("rel_strength_4", ["--min-rel-strength-pct", "4"]),
    ("rel_strength_8", ["--min-rel-strength-pct", "8"]),
    (
        "btc_chop_rel4",
        [
            "--btc-chop-guards",
            "--btc-chop-momentum-abs-pct",
            "1.0",
            "--btc-chop-ema-abs-pct",
            "1.5",
            "--btc-chop-instant-min-rel-strength-pct",
            "4",
        ],
    ),
    (
        "btc_chop_skip_instant",
        [
            "--btc-chop-guards",
            "--btc-chop-momentum-abs-pct",
            "1.0",
            "--btc-chop-ema-abs-pct",
            "1.0",
            "--btc-chop-skip-entry-regimes",
            "INSTANT",
        ],
    ),
    (
        "btc_chop_skip_instant_retest",
        [
            "--btc-chop-guards",
            "--btc-chop-momentum-abs-pct",
            "1.0",
            "--btc-chop-ema-abs-pct",
            "1.0",
            "--btc-chop-skip-entry-regimes",
            "INSTANT,RETEST",
        ],
    ),
    ("trail_gate_050", ["--trail-activation-r", "0.5"]),
    ("mtf_4h_sma25", ["--mtf-alignment-tf", "4h", "--mtf-alignment-ma-period", "25"]),
    ("mtf_4h_ema25", ["--mtf-alignment-tf", "4h", "--mtf-alignment-ma-period", "25", "--mtf-alignment-ma-type", "ema"]),
]

MODEL = Path("model_live_context_tail_rank.json")
if MODEL.exists():
    CONFIGS.append(
        (
            "ml_tail_rank",
            [
                "--ml-filter-model",
                str(MODEL),
                "--ml-filter-score",
                "tail",
                "--ml-score-only",
                "--ml-rank-signals",
            ],
        )
    )


@dataclass
class Result:
    config: str
    window: str
    regime: str
    returncode: int
    generated: int = 0
    taken: int = 0
    equity: float = 0.0
    expectancy: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_dd: float = 0.0


PATTERNS = {
    "signals": re.compile(r"Signals\s+(\d+)\s+generated\s+->\s+(\d+)\s+taken"),
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "profit_factor": re.compile(r"Profit factor\s+([\d.]+)"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
}


def without_dynamic_sl(args: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for i, item in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if item == "--dynamic-sl":
            continue
        if item == "--sl-lookback":
            skip_next = True
            continue
        out.append(item)
    return out


def parse_result(config: str, window: str, regime: str, returncode: int, stdout: str) -> Result:
    result = Result(config=config, window=window, regime=regime, returncode=returncode)
    match = PATTERNS["signals"].search(stdout)
    if match:
        result.generated = int(match.group(1))
        result.taken = int(match.group(2))
    for attr, pattern in (
        ("equity", PATTERNS["equity"]),
        ("expectancy", PATTERNS["expectancy"]),
        ("win_rate", PATTERNS["win_rate"]),
        ("profit_factor", PATTERNS["profit_factor"]),
        ("max_dd", PATTERNS["max_dd"]),
    ):
        match = pattern.search(stdout)
        if match:
            setattr(result, attr, float(match.group(1)))
    return result


def run_one(config: str, extra: list[str], window: tuple[str, str, str, str]) -> Result:
    wid, start, end, regime = window
    base = without_dynamic_sl(BASE) if config == "no_dynamic_sl" else BASE
    log = Path("_sweep") / f"regime_router_{config}_{wid}.csv"
    cmd = base + [
        "--start-date",
        start,
        "--end-date",
        end,
        "--trade-log",
        str(log),
    ] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-500:]
        print(f"FAILED {config}/{wid}: {tail}", flush=True)
    return parse_result(config, wid, regime, proc.returncode, proc.stdout)


def main() -> int:
    Path("_sweep").mkdir(exist_ok=True)
    rows: list[Result] = []
    total = len(CONFIGS) * len(WINDOWS)
    idx = 0
    for config, extra in CONFIGS:
        for window in WINDOWS:
            idx += 1
            print(f"[{idx}/{total}] {config} {window[0]}...", flush=True)
            result = run_one(config, extra, window)
            rows.append(result)
            print(
                f"  {result.window}: eq={result.equity:.2f} R={result.expectancy:+.3f} "
                f"win={result.win_rate:.1f}% dd={result.max_dd:.1f}% taken={result.taken}",
                flush=True,
            )

    out_path = Path("_sweep") / "regime_router_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "window",
                "regime",
                "returncode",
                "generated",
                "taken",
                "equity",
                "expectancy_r",
                "win_rate",
                "profit_factor",
                "max_dd",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.config,
                    r.window,
                    r.regime,
                    r.returncode,
                    r.generated,
                    r.taken,
                    f"{r.equity:.2f}",
                    f"{r.expectancy:.3f}",
                    f"{r.win_rate:.2f}",
                    f"{r.profit_factor:.2f}",
                    f"{r.max_dd:.2f}",
                ]
            )

    print(f"\nWrote {out_path}")
    print("\nBest per window:")
    for wid, _, _, regime in WINDOWS:
        best = max((r for r in rows if r.window == wid and r.returncode == 0), key=lambda r: r.expectancy)
        print(
            f"  {wid} {regime}: {best.config} eq={best.equity:.2f} "
            f"R={best.expectancy:+.3f} dd={best.max_dd:.1f}% taken={best.taken}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
