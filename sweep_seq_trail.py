"""Sequential trail (--trail-after-all-tps) vs parallel trail.

User wants: trail activates only AFTER every TP has filled. TPs first,
then the remaining runner trails. Tests various tp-count + runner-pct
combinations against the prior parallel-trail winner (m0.18, tp1, r50).
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
    "--trailing-stop",
    "--adaptive-trailing-callback",
    "--adaptive-trailing-callback-multiplier", "0.18",
    "--trail-activation-r", "0.0",
]

PATTERNS = {
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
    "profit_factor": re.compile(r"Profit factor\s+([\d.]+)"),
}


def run_cfg(name, extra):
    trip = []
    for wid, start, end, _ in WINDOWS:
        cmd = COMMON + [
            "--start-date", start, "--end-date", end,
            "--trade-log", f"sweep_logs/seqtrail_{name}_{wid}.csv",
        ] + extra
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            print(f"  FAIL {name}/{wid}: {p.stderr[-200:]}")
            raise SystemExit(1)
        rec = {k: float(PATTERNS[k].search(p.stdout).group(1)) for k in PATTERNS}
        trip.append((wid, rec))
    return trip


def main():
    configs = []

    # ---- Sequential trail (--trail-after-all-tps) ----
    for tp in [1, 2, 3]:
        for rp in [20, 30, 40, 50, 60, 70]:
            configs.append((
                f"SEQ_tp{tp}_r{rp}",
                ["--trail-after-all-tps", "--tp-count", str(tp), "--runner-pct", str(rp)],
            ))

    # ---- Parallel trail reference (prior winner config) ----
    configs.append((
        "PAR_tp1_r50_ref",
        ["--tp-count", "1", "--runner-pct", "50"],
    ))
    configs.append((
        "PAR_tp2_r50_ref",
        ["--tp-count", "2", "--runner-pct", "50"],
    ))
    configs.append((
        "PAR_tp3_r50_ref",
        ["--tp-count", "3", "--runner-pct", "50"],
    ))

    rows = []
    print(f"Running {len(configs)} configs x 3 windows = {len(configs)*3} backtests", flush=True)
    for i, (name, extra) in enumerate(configs, 1):
        trip = run_cfg(name, extra)
        wr = min(t[1]["expectancy"] for t in trip)
        se = sum(t[1]["equity"] for t in trip)
        rows.append((name, wr, se, trip))
        cells = " ".join(f"{w}={r['equity']:>7.1f}/{r['expectancy']:+.2f}" for w, r in trip)
        print(f"[{i:>2}/{len(configs)}] {name:<22} wr={wr:+.3f} sum={se:>8.2f}  {cells}", flush=True)

    with open("sweep_seq_trail_results.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["config", "window", "final_equity", "expectancy_r", "win_rate", "max_dd", "profit_factor"])
        for name, _, _, trip in rows:
            for wid, r in trip:
                cw.writerow([name, wid, f"{r['equity']:.2f}", f"{r['expectancy']:.3f}",
                             f"{r['win_rate']:.2f}", f"{r['max_dd']:.2f}", f"{r['profit_factor']:.2f}"])

    BASE_WR = 0.229
    BASE_SUM = 2668.18

    print(f"\n=== Reference: parallel m0.18 tp1 r50 wr=+{BASE_WR:.3f} sum={BASE_SUM:.2f} ===\n")

    print("=== Ranked by worst_R then sum_eq ===")
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for name, wr, se, trip in rows:
        eqs = " ".join(f"{w}={r['equity']:>7.1f}/{r['expectancy']:+.2f}" for w, r in trip)
        marker = "  <- beats parallel on BOTH" if wr > BASE_WR and se > BASE_SUM else ""
        print(f"  {name:<22} wr={wr:+.3f} sum={se:>8.2f}  {eqs}{marker}")

    print("\n=== Ranked by sum_eq ===")
    rows.sort(key=lambda x: x[2], reverse=True)
    for name, wr, se, trip in rows:
        eqs = " ".join(f"{w}={r['equity']:>7.1f}/{r['expectancy']:+.2f}" for w, r in trip)
        marker = "  <- beats parallel on BOTH" if wr > BASE_WR and se > BASE_SUM else ""
        print(f"  {name:<22} wr={wr:+.3f} sum={se:>8.2f}  {eqs}{marker}")


if __name__ == "__main__":
    main()
