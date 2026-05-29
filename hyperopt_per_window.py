"""Per-window staged hyperopt.

Tunes parameters for each canonical window (W1-W6) INDEPENDENTLY so the
combo count stays bounded, then writes the best per-window params to
`hyperopt_per_window_best.json`.

Stages (per window):
  1. Detector quality  (HP_INST_REL, HP_INST_VOL)        8 combos
  2. Exit params       (breakeven_r, stagnation_after_r) 6 combos
  3. SL cap            (max_sl_loss_pct)                  4 combos
  4. MTF + concurrency (mtf_ma_period, max_concurrent)    8 combos

Each stage keeps the best result and feeds the params into the next stage.

Concurrency: each backtest uses 16 symbol workers. --combo-workers N runs
N parameter combos at once; keep it at 1 unless you intentionally want to
oversubscribe the machine.

Run:   python hyperopt_per_window.py --combo-workers 1
Output: hyperopt_per_window_best.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path


# ── Windows ───────────────────────────────────────────────────────────
WINDOWS: list[tuple[str, str, str, str]] = [
    ("W1", "2025-12-20", "2026-02-10", "bear crash+recovery -21.9%"),
    ("W2", "2025-02-20", "2025-04-05", "bear grinding -19.7%"),
    ("W3", "2025-04-02", "2025-05-21", "bull strong +25.4%"),
    ("W4", "2025-06-04", "2025-07-23", "bull chop +13.4%"),
    ("W5", "2025-10-08", "2025-11-26", "bear massive crash -28.2%"),
    ("W6", "2026-02-25", "2026-04-15", "bull recovery +16.1%"),
]


# ── Base backtest command (matches the rest of the codebase) ──────────
BASE_CMD: list[str] = [
    sys.executable, "-m", "screener.backtest",
    "--top", "0",
    "--interval", "1h",
    "--mtf-alignment-tf", "4h",
    "--capital", "10",
    "--compound",
    "--sizing-mode", "auto",
    "--simulate-rotation",
    "--trailing-stop",
    "--adaptive-trailing-callback",
    "--dynamic-sl",
    "--btc-chop-guards",
    "--leverage", "10",
    "--runner-pct", "50",
    "--bear-profile",
    "--offline-cache-dir", "_kline_cache",
    "--workers", "16",
]


# ── Stage grids — small, focused, narrow ──────────────────────────────
STAGES: list[tuple[str, dict[str, list]]] = [
    ("detector", {
        "HP_INST_REL": [3.0, 4.0, 5.0],
        "HP_INST_VOL": [3.0, 3.5],
    }),
    ("exits", {
        "breakeven_r": [0.0, 1.0],
        "stag_after_r": [0.5, 1.0],
    }),
    ("sl_cap", {
        "max_sl_loss_pct": [35, 40, 45, 50],
    }),
    ("mtf_conc", {
        "mtf_ma_period": [20, 25],
        "max_concurrent": [2, 3, 4, 5],
    }),
]


RE_EQUITY = re.compile(r"Final equity\s+([\d.]+)\s+USDT")
RE_EXP = re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R")


def _flag_for(param: str, value) -> list[str]:
    mapping = {
        "breakeven_r":      "--breakeven-trigger-r",
        "stag_after_r":     "--stagnation-after-r",
        "stag_candles":     "--stagnation-candles",
        "max_sl_loss_pct":  "--max-sl-loss-pct",
        "mtf_ma_period":    "--mtf-alignment-ma-period",
        "max_concurrent":   "--max-concurrent",
    }
    if param in mapping:
        return [mapping[param], str(value)]
    return []


def run_one(window_id: str, start: str, end: str, params: dict) -> dict:
    """Run a single backtest with `params` applied, return summary dict."""
    cmd = list(BASE_CMD) + [
        "--start-date", start,
        "--end-date", end,
        "--max-sl-loss-pct", str(params.get("max_sl_loss_pct", 50)),
        "--stagnation-candles", str(params.get("stag_candles", 12)),
    ]
    if "max_concurrent" in params:
        cmd += ["--max-concurrent", str(params["max_concurrent"])]
    else:
        cmd += ["--max-concurrent", "5"]
    for k in ("breakeven_r", "stag_after_r", "mtf_ma_period"):
        if k in params:
            cmd += _flag_for(k, params[k])

    env = os.environ.copy()
    for k in ("HP_INST_REL", "HP_INST_VOL", "HP_DV", "HP_BB_REL", "HP_BB_BTC",
              "HP_FR_VOL", "HP_FR_CP", "HP_FR_BODY",
              "HP_CRASH_BTC", "HP_CRASH_REL"):
        if k in params:
            env[k] = str(params[k])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=600, check=False)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"equity": 0.0, "expectancy": 0.0, "error": str(e)}
    if proc.returncode != 0:
        return {"equity": 0.0, "expectancy": 0.0, "error": proc.stderr[-300:]}
    eq_m = RE_EQUITY.search(proc.stdout)
    ex_m = RE_EXP.search(proc.stdout)
    return {
        "equity":     float(eq_m.group(1)) if eq_m else 0.0,
        "expectancy": float(ex_m.group(1)) if ex_m else 0.0,
    }


def _label(params: dict) -> str:
    return "_".join(f"{k}={v}" for k, v in sorted(params.items()))


def tune_window(window_id: str, start: str, end: str, desc: str,
                workers: int) -> tuple[dict, float, list[dict]]:
    """Run all stages for ONE window, return (best_params, best_equity, history)."""
    base_params: dict = {}
    history: list[dict] = []

    print(f"\n{'='*80}", flush=True)
    print(f"== {window_id}  ({desc})  {start} -> {end}", flush=True)
    print(f"{'='*80}", flush=True)

    for stage_name, grid in STAGES:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        combos = list(product(*values))
        print(f"\n[{window_id}] Stage {stage_name}: {len(combos)} combos × 1 window", flush=True)

        # Build per-combo param sets
        tasks: list[tuple[str, dict]] = []
        for combo_vals in combos:
            params = dict(base_params)
            params.update(dict(zip(keys, combo_vals)))
            tasks.append((_label(params), params))

        # Run in parallel
        best_eq = -1.0
        best_params = base_params
        results: list[tuple[dict, float, float]] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(run_one, window_id, start, end, params): (label, params)
                for label, params in tasks
            }
            for fut in as_completed(futures):
                label, params = futures[fut]
                r = fut.result()
                eq, expc = r.get("equity", 0.0), r.get("expectancy", 0.0)
                results.append((params, eq, expc))
                mark = " *" if eq > best_eq else ""
                print(f"  [{window_id}/{stage_name}] {label[:60]:<60} "
                      f"eq=${eq:>9.2f}  exp={expc:+.3f}R{mark}", flush=True)
                if eq > best_eq:
                    best_eq, best_params = eq, params

        history.append({
            "stage": stage_name,
            "best_equity": best_eq,
            "best_params": dict(best_params),
            "all": [
                {"params": p, "equity": eq, "expectancy": expc}
                for p, eq, expc in sorted(results, key=lambda x: x[1], reverse=True)
            ],
        })
        base_params = best_params
        print(f"[{window_id}/{stage_name}] BEST: ${best_eq:.2f}  params={base_params}", flush=True)

    return base_params, history[-1]["best_equity"], history


def main():
    parser = argparse.ArgumentParser(description="Per-window staged hyperopt")
    parser.add_argument("--combo-workers", type=int, default=1,
                        help="Parallel parameter combos. Each backtest uses --workers 16; default 1.")
    parser.add_argument("--windows", default="W1,W2,W3,W4,W5,W6",
                        help="Comma-separated subset of windows to tune.")
    parser.add_argument("--output", default="hyperopt_per_window_best.json")
    args = parser.parse_args()

    selected = set(args.windows.split(","))
    targets = [w for w in WINDOWS if w[0] in selected]

    print(f"Tuning {len(targets)} window(s): {[w[0] for w in targets]}")
    print(f"Combo workers per stage: {args.combo_workers}")
    n_combos = sum(len(list(product(*g.values()))) for _, g in STAGES)
    print(f"Per-window total combos across stages: {n_combos}")
    print(f"Estimated time: {n_combos * 60 * len(targets) / args.combo_workers / 60:.0f} min", flush=True)

    all_results: dict = {}
    t0 = time.time()
    for window_id, start, end, desc in targets:
        best_params, best_eq, history = tune_window(
            window_id, start, end, desc, workers=args.combo_workers
        )
        all_results[window_id] = {
            "best_params": best_params,
            "best_equity": best_eq,
            "history": history,
            "start": start, "end": end, "desc": desc,
        }
        # Persist after each window
        Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\n>>> {window_id} DONE: best_equity=${best_eq:.2f}  params={best_params}", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"\n\n{'='*80}")
    print(f"ALL DONE — {elapsed:.1f} min")
    print(f"{'='*80}")
    print("\nSummary (best per window):")
    for wid, r in all_results.items():
        print(f"  {wid}: ${r['best_equity']:.2f}   {r['best_params']}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
