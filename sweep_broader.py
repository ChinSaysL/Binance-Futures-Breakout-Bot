"""Broader exploration after the staged greedy failed.

The activation-r gate is counterproductive (gives up runner upside, sum_eq
collapses). Hold activation=0 and explore the parameters that could actually
narrow the wick-loss / catch-the-trend gap:
  - adaptive callback multiplier (trail tightness)
  - fixed trailing-callback-pct (alternative to adaptive)
  - runner-pct (how much of position uses trail)
  - stagnation knobs
  - no-trail mode (rely on TPs only)
  - profit-lock ladders

Metric: prefer configs that beat baseline on BOTH worst_R AND sum_eq.
Fallback: highest sum_eq while worst_R >= baseline (+0.224).
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from dataclasses import dataclass


WINDOWS = [
    ("W1", "2025-12-20", "2026-02-10", "downtrend"),
    ("W2", "2026-02-10", "2026-04-01", "chop"),
    ("W3", "2026-04-01", "2026-05-21", "uptrend"),
]

BASE = [
    sys.executable, "-m", "screener.backtest",
    "--top", "40", "--interval", "1h",
    "--capital", "10", "--compound",
    "--sizing-mode", "auto", "--max-concurrent", "2",
    "--simulate-rotation",
    "--btc-chop-guards",
]


PATTERNS = {
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "profit_factor": re.compile(r"Profit factor\s+([\d.]+)"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
}


def _parse(out: str) -> dict[str, float]:
    o: dict[str, float] = {}
    for k, p in PATTERNS.items():
        m = p.search(out)
        if m:
            o[k] = float(m.group(1))
    return o


@dataclass
class Result:
    name: str
    window: str
    final_equity: float
    expectancy_r: float
    win_rate: float
    profit_factor: float
    max_dd: float


def run_cfg(name: str, extra: list[str]) -> list[Result]:
    out: list[Result] = []
    for wid, start, end, _ in WINDOWS:
        cmd = BASE + [
            "--start-date", start, "--end-date", end,
            "--trade-log", f"sweep_logs/broader_{name}_{wid}.csv",
        ] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print(f"  FAIL {name}/{wid}: {proc.stderr[-300:]}")
            raise SystemExit(1)
        p = _parse(proc.stdout)
        out.append(Result(
            name=name, window=wid,
            final_equity=p.get("equity", 0.0),
            expectancy_r=p.get("expectancy", 0.0),
            win_rate=p.get("win_rate", 0.0),
            profit_factor=p.get("profit_factor", 0.0),
            max_dd=p.get("max_dd", 0.0),
        ))
    return out


def main() -> int:
    # Stagnation defaults from README baseline
    stagnation = ["--stagnation-after-r", "0.5", "--stagnation-candles", "12"]
    # Trail defaults: adaptive on, multiplier 0.3 (current), activation 0, runner 50, breakeven 1.5
    trail_default = [
        "--trailing-stop",
        "--adaptive-trailing-callback",
        "--adaptive-trailing-callback-multiplier", "0.3",
        "--trail-activation-r", "0.0",
        "--runner-pct", "50",
        "--breakeven-trigger-r", "1.5",
    ]
    configs: list[tuple[str, list[str]]] = []

    # Group A: baseline + adaptive multiplier sweep (tightness of trail)
    for mult in [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]:
        configs.append((f"A_mult{mult}", stagnation + [
            "--trailing-stop", "--adaptive-trailing-callback",
            "--adaptive-trailing-callback-multiplier", str(mult),
            "--trail-activation-r", "0.0",
            "--runner-pct", "50",
            "--breakeven-trigger-r", "1.5",
        ]))

    # Group B: runner-pct sweep at default mult
    for rp in [20, 30, 40, 50, 60, 70]:
        configs.append((f"B_runner{rp}", stagnation + [
            "--trailing-stop", "--adaptive-trailing-callback",
            "--adaptive-trailing-callback-multiplier", "0.3",
            "--trail-activation-r", "0.0",
            "--runner-pct", str(rp),
            "--breakeven-trigger-r", "1.5",
        ]))

    # Group C: breakeven trigger sweep
    for be in [0.0, 0.5, 1.0, 1.5, 2.0]:
        configs.append((f"C_be{be}", stagnation + [
            "--trailing-stop", "--adaptive-trailing-callback",
            "--adaptive-trailing-callback-multiplier", "0.3",
            "--trail-activation-r", "0.0",
            "--runner-pct", "50",
            "--breakeven-trigger-r", str(be),
        ]))

    # Group D: fixed callback (non-adaptive)
    for cb in [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        configs.append((f"D_fixed{cb}", stagnation + [
            "--trailing-stop",  # no --adaptive-trailing-callback
            "--trailing-callback-pct", str(cb),
            "--trail-activation-r", "0.0",
            "--runner-pct", "50",
            "--breakeven-trigger-r", "1.5",
        ]))

    # Group E: no trailing (rely on TPs only -- runner exits at TP target)
    # tp-count >1 splits exits between TPs, no trail
    for tpc in [1, 2, 3]:
        configs.append((f"E_no_trail_tp{tpc}", stagnation + [
            "--tp-count", str(tpc),
            "--trail-activation-r", "0.0",
            "--breakeven-trigger-r", "1.5",
        ]))

    # Group F: stagnation sweep
    for sr, sc in [(0.3, 8), (0.3, 12), (0.5, 8), (0.5, 12), (0.5, 16), (0.7, 12)]:
        configs.append((f"F_stag_r{sr}_c{sc}", [
            "--stagnation-after-r", str(sr), "--stagnation-candles", str(sc),
            "--trailing-stop", "--adaptive-trailing-callback",
            "--adaptive-trailing-callback-multiplier", "0.3",
            "--trail-activation-r", "0.0",
            "--runner-pct", "50",
            "--breakeven-trigger-r", "1.5",
        ]))

    # Group G: lookahead -- try tighter adaptive mult with larger runner
    for mult, rp in [(0.2, 30), (0.2, 50), (0.4, 30), (0.4, 70), (0.5, 30), (0.5, 70)]:
        configs.append((f"G_mult{mult}_runner{rp}", stagnation + [
            "--trailing-stop", "--adaptive-trailing-callback",
            "--adaptive-trailing-callback-multiplier", str(mult),
            "--trail-activation-r", "0.0",
            "--runner-pct", str(rp),
            "--breakeven-trigger-r", "1.5",
        ]))

    all_rows: list[Result] = []
    print(f"Running {len(configs)} configs x 3 windows = {len(configs)*3} backtests", flush=True)
    for i, (name, extra) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {name}", flush=True)
        triplet = run_cfg(name, extra)
        all_rows.extend(triplet)
        worst_r = min(r.expectancy_r for r in triplet)
        sum_eq = sum(r.final_equity for r in triplet)
        worst_eq = min(r.final_equity for r in triplet)
        print(f"   worst_R={worst_r:+.3f}  worst_eq={worst_eq:>7.2f}  sum_eq={sum_eq:>8.2f}", flush=True)

    # Save raw
    with open("sweep_broader_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "window", "final_equity", "expectancy_r", "win_rate", "profit_factor", "max_dd"])
        for r in all_rows:
            w.writerow([r.name, r.window, f"{r.final_equity:.2f}", f"{r.expectancy_r:.3f}",
                        f"{r.win_rate:.2f}", f"{r.profit_factor:.2f}", f"{r.max_dd:.2f}"])

    # Summarize
    by_cfg: dict[str, list[Result]] = {}
    for r in all_rows:
        by_cfg.setdefault(r.name, []).append(r)

    summary = []
    BASELINE_WORST_R = 0.224
    for name, triplet in by_cfg.items():
        triplet.sort(key=lambda x: x.window)
        worst_r = min(r.expectancy_r for r in triplet)
        sum_eq = sum(r.final_equity for r in triplet)
        worst_eq = min(r.final_equity for r in triplet)
        summary.append({
            "name": name,
            "worst_r": worst_r,
            "sum_eq": sum_eq,
            "worst_eq": worst_eq,
            "W1_eq": triplet[0].final_equity,
            "W1_r": triplet[0].expectancy_r,
            "W2_eq": triplet[1].final_equity,
            "W2_r": triplet[1].expectancy_r,
            "W3_eq": triplet[2].final_equity,
            "W3_r": triplet[2].expectancy_r,
            "beats_baseline_worst_r": worst_r >= BASELINE_WORST_R,
        })

    summary.sort(key=lambda s: (s["worst_r"] >= BASELINE_WORST_R, s["sum_eq"]), reverse=True)
    with open("sweep_broader_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)

    print("\n\n=== TOP 15 (worst_R >= baseline +0.224, by sum_eq) ===")
    print(f"{'name':<32} {'worst_R':>8} {'sum_eq':>9} {'W1':>9} {'W2':>9} {'W3':>9}")
    qualifying = [s for s in summary if s["beats_baseline_worst_r"]]
    qualifying.sort(key=lambda s: s["sum_eq"], reverse=True)
    for s in qualifying[:15]:
        print(f"{s['name']:<32} {s['worst_r']:+.3f}  {s['sum_eq']:>8.2f}  {s['W1_eq']:>7.2f}/{s['W1_r']:+.2f}  {s['W2_eq']:>7.2f}/{s['W2_r']:+.2f}  {s['W3_eq']:>7.2f}/{s['W3_r']:+.2f}")

    print("\n=== TOP 5 by absolute sum_eq (regardless of worst_R) ===")
    by_sum = sorted(summary, key=lambda s: s["sum_eq"], reverse=True)
    for s in by_sum[:5]:
        print(f"{s['name']:<32} {s['worst_r']:+.3f}  {s['sum_eq']:>8.2f}  {s['W1_eq']:>7.2f}/{s['W1_r']:+.2f}  {s['W2_eq']:>7.2f}/{s['W2_r']:+.2f}  {s['W3_eq']:>7.2f}/{s['W3_r']:+.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
