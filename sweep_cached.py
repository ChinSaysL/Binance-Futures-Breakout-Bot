"""Full-universe sweep: runs backtest directly — the backtest rate-limits
itself at 800 RPM, well under Binance's 2400/min cap. No pre-fetch needed.

Usage:
    python sweep_cached.py              # full OLD + NEW x W1/W2/W3
    python sweep_cached.py --quick      # just W2, one config
    python sweep_cached.py --old-only   # only OLD (verify Claude)
    python sweep_cached.py --new-only   # only NEW (verify fixes)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

PATTERNS = {
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
}

WINDOWS = [
    ("W1_down", "2025-12-20", "2026-02-10"),
    ("W2_chop", "2026-02-10", "2026-04-01"),
    ("W3_up",   "2026-04-01", "2026-05-21"),
]

# OLD config: Claude's sweep_final.py R10 (no dynamic-sl, trail always active)
OLD = [
    "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
    "--trailing-stop", "--adaptive-trailing-callback",
    "--adaptive-trailing-callback-multiplier", "0.18",
    "--trail-activation-r", "0.0",
    "--runner-pct", "50",
    "--breakeven-trigger-r", "1.5",
    "--btc-chop-guards",
]

# NEW config: all our fixes applied
NEW = [
    "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
    "--trailing-stop", "--adaptive-trailing-callback",
    "--adaptive-trailing-callback-multiplier", "0.18",
    "--trail-activation-r", "0.5",
    "--runner-pct", "50",
    "--breakeven-trigger-r", "1.5",
    "--btc-chop-guards",
    "--dynamic-sl",
    "--loss-cooldown-after", "1", "--loss-cooldown-candles", "48",
]


def run_backtest(name: str, window: str, start: str, end: str, extra: list[str]) -> dict:
    cmd = [
        sys.executable, "-m", "screener.backtest",
        "--interval", "1h",
        "--capital", "10", "--compound",
        "--sizing-mode", "auto", "--max-concurrent", "2",
        "--simulate-rotation",
        "--start-date", start, "--end-date", end,
        "--trade-log", f"_sweep/{name}_{window}.csv",
        "--rate-limit-rpm", "600",
    ] + extra

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    r = {"name": name, "window": window, "elapsed_s": elapsed, "returncode": proc.returncode}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        r["error"] = err[-400:] if err else f"exit {proc.returncode}"
        return r

    for key, pat in PATTERNS.items():
        m = pat.search(proc.stdout)
        if m:
            r[key] = float(m.group(1))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--old-only", action="store_true")
    p.add_argument("--new-only", action="store_true")
    args = p.parse_args()

    os.makedirs("_sweep", exist_ok=True)

    windows = WINDOWS if not args.quick else [WINDOWS[1]]
    configs = []
    if not args.new_only:
        configs.append(("OLD_no_dynsl", OLD))
    if not args.old_only:
        configs.append(("NEW_fixed", NEW))

    all_results = []
    total = len(configs) * len(windows)
    idx = 0
    t_start = time.time()

    for cfg_name, extra in configs:
        for win_name, start, end in windows:
            idx += 1
            name = f"{cfg_name}_{win_name}"
            eta = (time.time() - t_start) / max(idx - 1, 1) * (total - idx + 1) if idx > 1 else 0
            print(f"\n[{idx}/{total}] {name}  ({start} -> {end})  ETA {eta/60:.0f}min", flush=True)
            result = run_backtest(name, win_name, start, end, extra)
            all_results.append(result)

            if "error" in result:
                print(f"  FAIL: {result['error'][:200]}")
            else:
                print(f"  OK: eq=${result.get('equity',0):,.2f}  R={result.get('expectancy',0):+.3f}  WR={result.get('win_rate',0):.0f}%  DD={result.get('max_dd',0):.0f}%  ({result['elapsed_s']:.0f}s)")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    groups: dict[str, list[dict]] = {}
    for r in all_results:
        base = r["name"].rsplit("_", 1)[0]
        groups.setdefault(base, []).append(r)

    for cfg, results in groups.items():
        valid = [r for r in results if "equity" in r]
        if not valid:
            print(f"\n{cfg}: all failed")
            continue
        se = sum(r["equity"] for r in valid)
        wr = min(r["expectancy"] for r in valid)
        print(f"\n{cfg}:  sum_eq=${se:,.2f}  worst_R={wr:+.3f}")
        for r in sorted(valid, key=lambda x: x["window"]):
            print(f"  {r['window']:10s}  eq=${r['equity']:>9,.2f}  R={r['expectancy']:+.3f}  WR={r['win_rate']:.0f}%  DD={r['max_dd']:.0f}%")

    # CSV
    with open("_sweep/comparison.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f)
        cw.writerow(["config", "window", "equity", "expectancy_r", "win_rate", "max_dd", "elapsed_s"])
        for r in all_results:
            cw.writerow([r["name"], r["window"], r.get("equity",""), r.get("expectancy",""),
                         r.get("win_rate",""), r.get("max_dd",""), r.get("elapsed_s","")])

    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal: {elapsed:.0f} min. Results in _sweep/comparison.csv")


if __name__ == "__main__":
    main()
