"""Sweep trail-activation-r to find the optimal gate. Zero API calls — 
reads symbols from _kline_cache/ and klines from .backtest_kline_cache/.

Usage:
    python sweep_trail_gate.py              # full sweep
    python sweep_trail_gate.py --quick      # W2 only
"""

from __future__ import annotations

import argparse, csv, os, re, subprocess, sys, time

WINDOWS = [
    ("W1_down", "2025-12-20", "2026-02-10"),
    ("W2_chop", "2026-02-10", "2026-04-01"),
    ("W3_up",   "2026-04-01", "2026-05-21"),
]

# Trail gate values to test
GATES = [0.0, 0.15, 0.25, 0.35, 0.50]

PATTERNS = {
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
}


def run_one(gate: float, window: str, start: str, end: str) -> dict:
    name = f"gate{gate:.2f}_{window}"
    cmd = [
        sys.executable, "-m", "screener.backtest",
        "--interval", "1h",
        "--capital", "10", "--compound",
        "--sizing-mode", "auto", "--max-concurrent", "2",
        "--simulate-rotation",
        "--start-date", start, "--end-date", end,
        "--trade-log", f"_sweep/{name}.csv",
        "--rate-limit-rpm", "400",
        "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
        "--trailing-stop", "--adaptive-trailing-callback",
        "--adaptive-trailing-callback-multiplier", "0.18",
        "--trail-activation-r", str(gate),
        "--runner-pct", "50",
        "--breakeven-trigger-r", "1.5",
        "--btc-chop-guards",
        "--dynamic-sl",
        "--loss-cooldown-after", "1", "--loss-cooldown-candles", "48",
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    r = {"gate": gate, "window": window, "elapsed_s": elapsed}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        r["error"] = err[-300:] if err else f"exit {proc.returncode}"
        return r

    for key, pat in PATTERNS.items():
        m = pat.search(proc.stdout)
        if m:
            r[key] = float(m.group(1))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    os.makedirs("_sweep", exist_ok=True)
    windows = WINDOWS if not args.quick else [WINDOWS[1]]
    total = len(GATES) * len(windows)
    idx = 0
    all_results = []

    for gate in GATES:
        for wname, start, end in windows:
            idx += 1
            print(f"\n[{idx}/{total}] trail={gate:.2f}R  {wname}  ({start} -> {end})", flush=True)
            r = run_one(gate, wname, start, end)
            all_results.append(r)
            if "error" in r:
                print(f"  FAIL: {r['error'][:150]}")
            else:
                print(f"  eq=${r.get('equity',0):,.2f}  R={r.get('expectancy',0):+.3f}  WR={r.get('win_rate',0):.0f}%  DD={r.get('max_dd',0):.0f}%  ({r['elapsed_s']:.0f}s)")

    # ── summary ──
    print("\n" + "=" * 75)
    print("TRAIL GATE SWEEP RESULTS")
    print(f"{'Gate':<8} {'W1_eq':>10} {'W1_R':>8} {'W1_WR':>6} {'W2_eq':>10} {'W2_R':>8} {'W2_WR':>6} {'W3_eq':>10} {'W3_R':>8} {'W3_WR':>6} {'SUM':>10} {'WORST_R':>8}")
    print("-" * 75)

    best_sum = 0
    best_gate = 0.0
    for gate in GATES:
        gr = [r for r in all_results if r.get("gate") == gate and "equity" in r]
        if len(gr) < 3:
            continue
        by_win = {r["window"]: r for r in gr}
        w1 = by_win.get("W1_down", {})
        w2 = by_win.get("W2_chop", {})
        w3 = by_win.get("W3_up", {})
        seq = w1.get("equity",0) + w2.get("equity",0) + w3.get("equity",0)
        wr_r = min(w1.get("expectancy",0), w2.get("expectancy",0), w3.get("expectancy",0))
        print(f"{gate:<8.2f} ${w1.get('equity',0):>9,.2f} {w1.get('expectancy',0):>+7.3f} {w1.get('win_rate',0):>5.0f}% ${w2.get('equity',0):>9,.2f} {w2.get('expectancy',0):>+7.3f} {w2.get('win_rate',0):>5.0f}% ${w3.get('equity',0):>9,.2f} {w3.get('expectancy',0):>+7.3f} {w3.get('win_rate',0):>5.0f}% ${seq:>9,.2f} {wr_r:>+7.3f}")
        if seq > best_sum:
            best_sum = seq
            best_gate = gate

    print(f"\nBest gate: {best_gate:.2f}R  (sum_eq=${best_sum:,.2f})")

    # CSV
    with open("_sweep/trail_gate_sweep.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["gate", "window", "equity", "expectancy_r", "win_rate", "max_dd", "elapsed_s"])
        for r in all_results:
            cw.writerow([r.get("gate"), r["window"], r.get("equity",""), r.get("expectancy",""),
                         r.get("win_rate",""), r.get("max_dd",""), r.get("elapsed_s","")])

    print("Results in _sweep/trail_gate_sweep.csv")


if __name__ == "__main__":
    main()
