"""Final narrow grid around the sweet spot.

From refine: best Pareto-frontier candidates:
  R1  (mult 0.20, runner 50): worst_R +0.227 / sum $2641
  R10 (mult 0.18, runner 50): worst_R +0.229 / sum $2668
  R13 (mult 0.20, runner 60): worst_R +0.247 / sum $1961   <- big W2 jump
  R3  (mult 0.20, stag 0.7/16, runner 50): worst_R +0.283 / sum $1638

Narrow grid: mult x runner around (0.18..0.22) x (50..65) holding all else.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys


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
    "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
    "--trailing-stop", "--adaptive-trailing-callback",
    "--trail-activation-r", "0.5",
    "--breakeven-trigger-r", "1.5",
    "--btc-chop-guards",
    "--dynamic-sl",
]

PATTERNS = {
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
}


def run_one(name, mult, runner):
    triplet = []
    for wid, start, end, _ in WINDOWS:
        cmd = BASE + [
            "--start-date", start, "--end-date", end,
            "--adaptive-trailing-callback-multiplier", str(mult),
            "--runner-pct", str(runner),
            "--trade-log", f"sweep_logs/final_{name}_{wid}.csv",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise SystemExit(f"FAIL {name}/{wid}: {p.stderr[-200:]}")
        eq = float(PATTERNS["equity"].search(p.stdout).group(1))
        er = float(PATTERNS["expectancy"].search(p.stdout).group(1))
        triplet.append((wid, eq, er))
    worst_r = min(t[2] for t in triplet)
    sum_eq = sum(t[1] for t in triplet)
    return triplet, worst_r, sum_eq


def main():
    grid = []
    for mult in [0.18, 0.19, 0.20, 0.21, 0.22, 0.25]:
        for runner in [50, 55, 60, 65]:
            grid.append((f"m{mult}_r{runner}", mult, runner))

    rows = []
    print(f"Running {len(grid)} configs x 3 = {len(grid)*3} backtests", flush=True)
    for i, (name, mult, runner) in enumerate(grid, 1):
        triplet, wr, se = run_one(name, mult, runner)
        rows.append((name, mult, runner, wr, se, triplet))
        eqs = " ".join(f"{w}={e:>7.1f}/{r:+.2f}" for w, e, r in triplet)
        print(f"[{i:>2}/{len(grid)}] {name:<14} wr={wr:+.3f} sum={se:>8.2f}  {eqs}", flush=True)

    with open("sweep_final_results.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["config", "mult", "runner", "worst_r", "sum_eq", "W1_eq", "W1_r", "W2_eq", "W2_r", "W3_eq", "W3_r"])
        for name, mult, runner, wr, se, trip in rows:
            cw.writerow([name, mult, runner, f"{wr:.3f}", f"{se:.2f}",
                         f"{trip[0][1]:.2f}", f"{trip[0][2]:.3f}",
                         f"{trip[1][1]:.2f}", f"{trip[1][2]:.3f}",
                         f"{trip[2][1]:.2f}", f"{trip[2][2]:.3f}"])

    BASELINE_WR = 0.224
    BASELINE_SUM = 2230.67

    print("\n=== Configs that strictly improve over baseline ===")
    print(f"baseline                       wr=+{BASELINE_WR:.3f} sum={BASELINE_SUM:>8.2f}")
    strict = [r for r in rows if r[3] > BASELINE_WR and r[4] > BASELINE_SUM]
    strict.sort(key=lambda r: (r[3], r[4]), reverse=True)
    for name, mult, runner, wr, se, trip in strict:
        eqs = " ".join(f"{w}={e:>7.1f}/{r:+.2f}" for w, e, r in trip)
        print(f"{name:<14} mult={mult} runner={runner}  wr={wr:+.3f} sum={se:>8.2f}  {eqs}")

    print("\n=== Top 10 by worst_R (then sum_eq) ===")
    rows.sort(key=lambda r: (r[3], r[4]), reverse=True)
    for name, mult, runner, wr, se, trip in rows[:10]:
        eqs = " ".join(f"{w}={e:>7.1f}/{r:+.2f}" for w, e, r in trip)
        print(f"{name:<14} wr={wr:+.3f} sum={se:>8.2f}  {eqs}")


if __name__ == "__main__":
    main()
