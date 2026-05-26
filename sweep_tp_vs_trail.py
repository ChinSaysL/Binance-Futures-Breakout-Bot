"""Multi-TP with-trail vs without-trail comparison.

User question: is trailing-take-profit actually the best exit, or is multi-TP
without trail competitive? Test cleanly across all 3 windows.

Holds: idx18 chop guard, stagnation 0.5/12, breakeven 1.5R, adaptive callback
mult 0.18 (the prior winner under realistic exit modeling). Varies tp-count
and runner-pct, including the no-trail (runner=0) configs.
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

COMMON = [
    sys.executable, "-m", "screener.backtest",
    "--top", "40", "--interval", "1h",
    "--capital", "10", "--compound",
    "--sizing-mode", "auto", "--max-concurrent", "2",
    "--simulate-rotation",
    "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
    "--breakeven-trigger-r", "1.5",
    "--btc-chop-guards",
]

PATTERNS = {
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
    "profit_factor": re.compile(r"Profit factor\s+([\d.]+)"),
}


def run_cfg(name, extra):
    triplet = []
    for wid, start, end, _ in WINDOWS:
        cmd = COMMON + [
            "--start-date", start, "--end-date", end,
            "--trade-log", f"sweep_logs/tpvtrail_{name}_{wid}.csv",
        ] + extra
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            print(f"  FAIL {name}/{wid}: {p.stderr[-200:]}")
            raise SystemExit(1)
        rec = {k: float(PATTERNS[k].search(p.stdout).group(1)) for k in PATTERNS}
        triplet.append((wid, rec))
    return triplet


def main():
    # --- TRAIL ON variants ---
    trail_extra = [
        "--trailing-stop",
        "--adaptive-trailing-callback",
        "--adaptive-trailing-callback-multiplier", "0.18",
        "--trail-activation-r", "0.0",
    ]

    configs: list[tuple[str, list[str]]] = []

    # With trailing-stop, vary tp-count and runner-pct
    for tp in [1, 2, 3]:
        for rp in [30, 50, 70]:
            configs.append((
                f"TRAIL_tp{tp}_r{rp}",
                trail_extra + ["--tp-count", str(tp), "--runner-pct", str(rp)],
            ))

    # No trailing-stop, vary tp-count only (runner is forced to 0)
    for tp in [1, 2, 3, 4]:
        configs.append((
            f"NOTRAIL_tp{tp}",
            ["--tp-count", str(tp)],
        ))

    # Smart-TP variants (adaptive splits + adaptive target)
    configs.append((
        "SMART_TP2_trail",
        trail_extra + ["--smart-tp", "--tp-count", "2", "--runner-pct", "50"],
    ))
    configs.append((
        "SMART_TP3_trail",
        trail_extra + ["--smart-tp", "--tp-count", "3", "--runner-pct", "50"],
    ))
    configs.append((
        "SMART_TP2_notrail",
        ["--smart-tp", "--tp-count", "2"],
    ))
    configs.append((
        "SMART_TP3_notrail",
        ["--smart-tp", "--tp-count", "3"],
    ))

    # Hybrid: tp1 + small trail (catches the tail without dominating)
    configs.append((
        "TRAIL_tp1_r20",
        trail_extra + ["--tp-count", "1", "--runner-pct", "20"],
    ))
    configs.append((
        "TRAIL_tp2_r20",
        trail_extra + ["--tp-count", "2", "--runner-pct", "20"],
    ))

    rows = []
    print(f"Running {len(configs)} configs x 3 windows = {len(configs)*3} backtests", flush=True)
    for i, (name, extra) in enumerate(configs, 1):
        triplet = run_cfg(name, extra)
        worst_r = min(t[1]["expectancy"] for t in triplet)
        sum_eq = sum(t[1]["equity"] for t in triplet)
        rows.append((name, worst_r, sum_eq, triplet))
        cells = " ".join(
            f"{w}={r['equity']:>7.1f}/{r['expectancy']:+.2f}/dd{r['max_dd']:.0f}" for w, r in triplet
        )
        print(f"[{i:>2}/{len(configs)}] {name:<22} wr={worst_r:+.3f} sum={sum_eq:>8.2f}  {cells}", flush=True)

    with open("sweep_tp_vs_trail_results.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow([
            "config", "window", "final_equity", "expectancy_r", "win_rate",
            "max_dd", "profit_factor",
        ])
        for name, _, _, triplet in rows:
            for wid, r in triplet:
                cw.writerow([
                    name, wid, f"{r['equity']:.2f}", f"{r['expectancy']:.3f}",
                    f"{r['win_rate']:.2f}", f"{r['max_dd']:.2f}", f"{r['profit_factor']:.2f}",
                ])

    BASELINE_WR = 0.229  # current winner m0.18_r50
    BASELINE_SUM = 2668.18

    print(f"\n=== Reference: prior winner (m0.18, runner 50, tp1, trail) wr=+0.229 sum=2668 ===\n")

    print("=== RANKED BY worst_R then sum_eq ===")
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for name, wr, se, trip in rows:
        eqs = " ".join(f"{w}={r['equity']:>7.1f}/{r['expectancy']:+.2f}" for w, r in trip)
        marker = "  <- beats prior" if wr > BASELINE_WR and se > BASELINE_SUM else ""
        print(f"  {name:<22} wr={wr:+.3f} sum={se:>8.2f}  {eqs}{marker}")

    print("\n=== RANKED BY sum_eq ===")
    rows.sort(key=lambda x: x[2], reverse=True)
    for name, wr, se, trip in rows:
        eqs = " ".join(f"{w}={r['equity']:>7.1f}/{r['expectancy']:+.2f}" for w, r in trip)
        marker = "  <- beats prior" if wr > BASELINE_WR and se > BASELINE_SUM else ""
        print(f"  {name:<22} wr={wr:+.3f} sum={se:>8.2f}  {eqs}{marker}")


if __name__ == "__main__":
    main()
