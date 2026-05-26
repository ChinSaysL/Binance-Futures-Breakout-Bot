"""Quick validation: rerun the R10 config (mult 0.18, runner 50) WITH --dynamic-sl
to see if the backtest edge survives the dynamic-SL ratchet that live uses.
"""

from __future__ import annotations

import re
import subprocess
import sys

WINDOWS = [
    ("W1_down", "2025-12-20", "2026-02-10"),
    ("W2_chop", "2026-02-10", "2026-04-01"),
    ("W3_up",   "2026-04-01", "2026-05-21"),
]

# R10 config from sweep_final.py + --dynamic-sl
BASE = [
    sys.executable, "-m", "screener.backtest",
    "--top", "40", "--interval", "1h",
    "--capital", "10", "--compound",
    "--sizing-mode", "auto", "--max-concurrent", "2",
    "--simulate-rotation",
    "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
    "--trailing-stop", "--adaptive-trailing-callback",
    "--adaptive-trailing-callback-multiplier", "0.18",
    "--trail-activation-r", "0.0",
    "--runner-pct", "50",
    "--breakeven-trigger-r", "1.5",
    "--btc-chop-guards",
    "--dynamic-sl",
]

PATTERNS = {
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
}

# Reference results WITHOUT --dynamic-sl (from sweep_final R10)
REFERENCE = {
    "W1_down":  {"eq": 128.0, "r": 0.44},
    "W2_chop":  {"eq": 34.8,  "r": 0.23},
    "W3_up":    {"eq": 2505.4, "r": 0.86},
}

def main():
    print("=" * 70)
    print("VALIDATION: R10 config (mult=0.18, runner=50) WITH --dynamic-sl")
    print("Reference (no dynamic SL): sum_eq=$2668.18, worst_R=+0.229")
    print("=" * 70)

    results = []
    for name, start, end in WINDOWS:
        cmd = BASE + [
            "--start-date", start, "--end-date", end,
            "--trade-log", f"/tmp/validate_dynsl_{name}.csv",
        ]
        print(f"\n--- {name}: {start} → {end} ---")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr[-300:]}")
            continue

        parsed = {}
        for key, pat in PATTERNS.items():
            m = pat.search(proc.stdout)
            if m:
                parsed[key] = float(m.group(1))

        eq = parsed.get("equity", 0)
        er = parsed.get("expectancy", 0)
        wr = parsed.get("win_rate", 0)
        dd = parsed.get("max_dd", 0)

        ref = REFERENCE.get(name, {})
        ref_eq = ref.get("eq", 0)
        ref_r = ref.get("r", 0)
        eq_delta = eq - ref_eq
        r_delta = er - ref_r

        print(f"  Equity:     ${eq:,.2f}  (was ${ref_eq:,.2f}, Δ={eq_delta:+,.2f})")
        print(f"  Expectancy: {er:+.3f}R  (was {ref_r:+.3f}R, Δ={r_delta:+.3f})")
        print(f"  Win rate:   {wr:.1f}%")
        print(f"  Max DD:     {dd:.1f}%")
        results.append((name, eq, er, ref_eq, ref_r))

    if results:
        sum_eq = sum(r[1] for r in results)
        worst_r = min(r[2] for r in results)
        ref_sum = sum(r[3] for r in results)
        ref_worst = min(r[4] for r in results)
        print(f"\n{'=' * 70}")
        print(f"SUMMARY (with --dynamic-sl):")
        print(f"  Sum equity:  ${sum_eq:,.2f}  (was ${ref_sum:,.2f}, Δ={sum_eq - ref_sum:+,.2f})")
        print(f"  Worst R:     {worst_r:+.3f}R  (was {ref_worst:+.3f}R)")
        status = "✅ EDGE SURVIVES" if worst_r > 0 and sum_eq > ref_sum * 0.5 else "⚠️  DEGRADED"
        print(f"  Verdict:     {status}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
