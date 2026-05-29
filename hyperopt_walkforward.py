"""Walk-forward (cross-period) hyperopt — anti-overfit.

The old per-window hyperopt tuned each window on ITSELF, which overfit: params
that maxed the in-sample windows collapsed out-of-sample (2024: 77% DD).

This harness instead scores every param combo on TWO independent folds and
selects the combo that maximizes the WORST fold's robustness score. A combo that
only wins on one period is penalised by its poor score on the other, so the
winner is the one that generalises — the closest proxy we have to "works live".

Folds (independent halves of the OOS year):
  FOLD_A = 2024 H1   (the hard, losing half in the baseline)
  FOLD_B = 2024 H2

Objective per combo:  min(robust(A), robust(B))   where robust = equity*(1-DD).

After the search, the winner is reconfirmed on full-2024 + the 6 regime windows.

Run:   python hyperopt_walkforward.py
"""
from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

CAPITAL = "40"
LEVERAGE = "5"  # validated robustness sweet spot

BASE = [
    sys.executable, "-m", "screener.backtest",
    "--top", "0", "--intervals", "15m,1h,4h",
    "--btc-guard-interval", "1h", "--mtf-alignment-tf", "4h",
    "--capital", CAPITAL, "--compound", "--sizing-mode", "auto",
    "--simulate-rotation", "--trailing-stop", "--adaptive-trailing-callback",
    "--dynamic-sl", "--btc-chop-guards", "--stagnation-candles", "12",
    "--leverage", LEVERAGE, "--runner-pct", "50", "--bear-profile",
    "--offline-cache-dir", "_kline_cache",
    "--window-config", "window_config.json", "--workers", "16",
]

FOLDS = [("H1", "2024-01-01", "2024-06-30"), ("H2", "2024-07-01", "2024-12-31")]
CONFIRM = [
    ("2024", "2024-01-01", "2024-12-31"),
    ("W1", "2025-12-20", "2026-02-10"), ("W2", "2025-02-20", "2025-04-05"),
    ("W3", "2026-04-01", "2026-05-21"), ("W4", "2025-06-04", "2025-07-23"),
    ("W5", "2025-10-08", "2025-11-26"), ("W6", "2026-02-25", "2026-04-15"),
]

# Global knobs to tune. Small grid — few params, cross-validated, to avoid
# re-introducing overfit. These apply strategy-wide (not per-window).
GRID = {
    "instant_size_multiplier": [0.4, 0.5, 0.7],
    "max_sl_loss_pct": [35, 45, 50],
}

RE_EQ = re.compile(r"Final equity\s+([\d.]+)\s+USDT")
RE_DD = re.compile(r"Max drawdown\s+([\d.]+)%")


def run(start, end, combo):
    cmd = BASE + ["--start-date", start, "--end-date", end,
                  "--max-sl-loss-pct", str(combo["max_sl_loss_pct"]),
                  "--instant-size-multiplier", str(combo["instant_size_multiplier"]),
                  "--trade-log", "sweep_logs/wf_tmp.csv"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    eq = float(m.group(1)) if (m := RE_EQ.search(p.stdout)) else 0.0
    dd = float(m.group(1)) if (m := RE_DD.search(p.stdout)) else 100.0
    return eq, dd


def robust(eq, dd):
    return eq * (1.0 - dd / 100.0)


def main():
    Path("sweep_logs").mkdir(exist_ok=True)
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    print(f"Walk-forward: {len(combos)} combos x {len(FOLDS)} folds "
          f"(capital ${CAPITAL}, leverage {LEVERAGE}x)\n", flush=True)

    scored = []
    for c in combos:
        fold_scores = []
        parts = []
        for name, s, e in FOLDS:
            eq, dd = run(s, e, c)
            fold_scores.append(robust(eq, dd))
            parts.append(f"{name} ${eq:.0f}@{dd:.0f}%")
        worst = min(fold_scores)
        label = ", ".join(f"{k}={v}" for k, v in c.items())
        print(f"  {label:<48} worst_robust={worst:>8.1f}  ({'; '.join(parts)})", flush=True)
        scored.append((worst, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_worst, best = scored[0]
    print(f"\nBEST (max-min across folds): {best}  worst_robust={best_worst:.1f}", flush=True)

    print("\nConfirming winner on full-2024 + 6 windows:", flush=True)
    print(f"{'Period':<8}{'Equity':>11}{'MaxDD%':>9}{'Robust':>10}", flush=True)
    wsum = 0.0
    for name, s, e in CONFIRM:
        eq, dd = run(s, e, best)
        print(f"{name:<8}${eq:>9.2f}{dd:>8.1f}%{robust(eq, dd):>10.1f}", flush=True)
        if name != "2024":
            wsum += eq
    print(f"\nWINDOWS SUM ${wsum:.2f}", flush=True)
    print(f"\nApply by setting these as global defaults / window_config and re-running "
          f"eval. Winner params: {best}", flush=True)


if __name__ == "__main__":
    main()
