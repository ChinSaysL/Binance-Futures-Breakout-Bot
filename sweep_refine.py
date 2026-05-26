"""Refinement: combine winners from broader sweep + retest chop guard variants.

Best from broader: A_mult0.2 (adaptive mult 0.2). Also strong: stag_r 0.7.
Combine them; vary chop guard on/off; try idx14/idx11 again under new sim.
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
    "--dynamic-sl",
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
        cmd = BASE + ["--start-date", start, "--end-date", end,
                      "--trade-log", f"sweep_logs/refine_{name}_{wid}.csv"] + extra
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


# Common config: trailing + adaptive callback + chop guard.
# Vary mult, stagnation, chop guard variant.
def make_cfg(mult, stag_r, stag_c, chop_args, runner=50, be=1.5):
    return [
        "--stagnation-after-r", str(stag_r),
        "--stagnation-candles", str(stag_c),
        "--trailing-stop",
        "--adaptive-trailing-callback",
        "--adaptive-trailing-callback-multiplier", str(mult),
        "--trail-activation-r", "0.0",
        "--runner-pct", str(runner),
        "--breakeven-trigger-r", str(be),
    ] + chop_args


CHOP_OFF: list[str] = []
CHOP_IDX18 = ["--btc-chop-guards"]  # idx18 is now the default values
CHOP_IDX14 = [
    "--btc-chop-guards",
    "--btc-chop-momentum-abs-pct", "1.0",
    "--btc-chop-ema-abs-pct", "1.0",
    "--btc-chop-skip-entry-regimes", "INSTANT,RETEST",
]
CHOP_IDX11 = [
    "--btc-chop-guards",
    "--btc-chop-momentum-abs-pct", "1.0",
    "--btc-chop-ema-abs-pct", "1.0",
    "--btc-chop-instant-min-rel-strength-pct", "4.0",
]


configs: list[tuple[str, list[str]]] = []

# Re-baseline
configs.append(("baseline_idx18", make_cfg(0.3, 0.5, 12, CHOP_IDX18)))

# Best from broader: mult=0.2 with idx18
configs.append(("R1_m0.2_idx18", make_cfg(0.2, 0.5, 12, CHOP_IDX18)))

# Combine: mult=0.2 + stag_r=0.7
configs.append(("R2_m0.2_stag0.7_idx18", make_cfg(0.2, 0.7, 12, CHOP_IDX18)))
configs.append(("R3_m0.2_stag0.7_c16_idx18", make_cfg(0.2, 0.7, 16, CHOP_IDX18)))

# Try with different chop guards
configs.append(("R4_m0.2_chop_off", make_cfg(0.2, 0.5, 12, CHOP_OFF)))
configs.append(("R5_m0.2_stag0.7_chop_off", make_cfg(0.2, 0.7, 12, CHOP_OFF)))
configs.append(("R6_m0.2_idx14", make_cfg(0.2, 0.5, 12, CHOP_IDX14)))
configs.append(("R7_m0.2_stag0.7_idx14", make_cfg(0.2, 0.7, 12, CHOP_IDX14)))
configs.append(("R8_m0.2_idx11", make_cfg(0.2, 0.5, 12, CHOP_IDX11)))

# Slightly different mults around 0.2
configs.append(("R9_m0.15_idx18", make_cfg(0.15, 0.5, 12, CHOP_IDX18)))
configs.append(("R10_m0.18_idx18", make_cfg(0.18, 0.5, 12, CHOP_IDX18)))
configs.append(("R11_m0.22_idx18", make_cfg(0.22, 0.5, 12, CHOP_IDX18)))

# Combine mult=0.2 with different runner pct
configs.append(("R12_m0.2_r40_idx18", make_cfg(0.2, 0.5, 12, CHOP_IDX18, runner=40)))
configs.append(("R13_m0.2_r60_idx18", make_cfg(0.2, 0.5, 12, CHOP_IDX18, runner=60)))
configs.append(("R14_m0.2_stag0.7_r60_idx18", make_cfg(0.2, 0.7, 12, CHOP_IDX18, runner=60)))

# Combine mult=0.2 with be variations
configs.append(("R15_m0.2_be0_idx18", make_cfg(0.2, 0.5, 12, CHOP_IDX18, be=0.0)))
configs.append(("R16_m0.2_be1.0_idx18", make_cfg(0.2, 0.5, 12, CHOP_IDX18, be=1.0)))
configs.append(("R17_m0.2_be2.0_idx18", make_cfg(0.2, 0.5, 12, CHOP_IDX18, be=2.0)))


def main() -> int:
    all_rows: list[Result] = []
    print(f"Running {len(configs)} configs x 3 windows = {len(configs)*3} backtests", flush=True)
    for i, (name, extra) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {name}", flush=True)
        triplet = run_cfg(name, extra)
        all_rows.extend(triplet)
        worst_r = min(r.expectancy_r for r in triplet)
        sum_eq = sum(r.final_equity for r in triplet)
        worst_eq = min(r.final_equity for r in triplet)
        eqs = {r.window: r.final_equity for r in triplet}
        rs = {r.window: r.expectancy_r for r in triplet}
        print(f"   worst_R={worst_r:+.3f}  sum_eq={sum_eq:>8.2f}  "
              f"W1={eqs.get('W1', 0):.1f}/{rs.get('W1', 0):+.2f}  "
              f"W2={eqs.get('W2', 0):.1f}/{rs.get('W2', 0):+.2f}  "
              f"W3={eqs.get('W3', 0):.1f}/{rs.get('W3', 0):+.2f}", flush=True)

    with open("sweep_refine_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "window", "final_equity", "expectancy_r", "win_rate", "profit_factor", "max_dd"])
        for r in all_rows:
            w.writerow([r.name, r.window, f"{r.final_equity:.2f}", f"{r.expectancy_r:.3f}",
                        f"{r.win_rate:.2f}", f"{r.profit_factor:.2f}", f"{r.max_dd:.2f}"])

    # Final ranking
    by_cfg: dict[str, list[Result]] = {}
    for r in all_rows:
        by_cfg.setdefault(r.name, []).append(r)

    summary = []
    for name, triplet in by_cfg.items():
        triplet.sort(key=lambda x: x.window)
        worst_r = min(r.expectancy_r for r in triplet)
        sum_eq = sum(r.final_equity for r in triplet)
        summary.append((name, worst_r, sum_eq, triplet))

    print("\n=== RANKED BY worst_R then sum_eq ===")
    summary.sort(key=lambda s: (s[1], s[2]), reverse=True)
    for name, wr, se, trip in summary:
        eqs = " ".join(f"{r.window}={r.final_equity:>7.2f}/{r.expectancy_r:+.2f}" for r in trip)
        print(f"  {name:<32} worst_R={wr:+.3f}  sum_eq={se:>8.2f}  {eqs}")

    print("\n=== RANKED BY sum_eq ===")
    summary.sort(key=lambda s: s[2], reverse=True)
    for name, wr, se, trip in summary:
        eqs = " ".join(f"{r.window}={r.final_equity:>7.2f}/{r.expectancy_r:+.2f}" for r in trip)
        print(f"  {name:<32} worst_R={wr:+.3f}  sum_eq={se:>8.2f}  {eqs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
