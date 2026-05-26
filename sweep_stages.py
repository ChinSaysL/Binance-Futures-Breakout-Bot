"""Staged parameter sweep under path-conservative exit simulator.

Stages are sequential: each stage holds the previous stage's winner and
sweeps the next parameter. Winner = best worst-case R across W1/W2/W3.

Outputs:
- sweep_stages_results.csv  -- every row run
- sweep_stages_summary.csv  -- one row per (stage, config) with worst-case
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

BASE_ARGS = [
    sys.executable, "-m", "screener.backtest",
    "--top", "40", "--interval", "1h",
    "--capital", "10", "--compound",
    "--sizing-mode", "auto", "--max-concurrent", "2",
    "--trailing-stop",
    "--simulate-rotation",
    "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
    "--adaptive-trailing-callback",
    "--btc-chop-guards",
]


PATTERNS = {
    "signals": re.compile(r"Signals\s+(\d+)\s+generated\s+->\s+(\d+)\s+taken"),
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "profit_factor": re.compile(r"Profit factor\s+([\d.]+)"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
}


def _parse(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pat in PATTERNS.items():
        m = pat.search(stdout)
        if m:
            if key == "signals":
                out["generated"] = float(m.group(1))
                out["taken"] = float(m.group(2))
            else:
                out[key] = float(m.group(1))
    return out


@dataclass
class Run:
    stage: str
    config_name: str
    window: str
    regime: str
    extra_args: tuple[str, ...]
    final_equity: float
    expectancy_r: float
    win_rate: float
    profit_factor: float
    max_dd: float
    taken: int
    generated: int


def run_one(stage: str, config_name: str, window: tuple[str, str, str, str], extra_args: list[str]) -> Run:
    wid, start, end, regime = window
    cmd = BASE_ARGS + [
        "--start-date", start, "--end-date", end,
        "--runner-pct", "50",
    ] + extra_args + ["--trade-log", f"sweep_logs/stages_{stage}_{config_name}_{wid}.csv"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"  FAILED {stage}/{config_name}/{wid}: {proc.stderr[-300:]}", flush=True)
        raise SystemExit(1)
    p = _parse(proc.stdout)
    return Run(
        stage=stage, config_name=config_name, window=wid, regime=regime,
        extra_args=tuple(extra_args),
        final_equity=p.get("equity", 0.0),
        expectancy_r=p.get("expectancy", 0.0),
        win_rate=p.get("win_rate", 0.0),
        profit_factor=p.get("profit_factor", 0.0),
        max_dd=p.get("max_dd", 0.0),
        taken=int(p.get("taken", 0)),
        generated=int(p.get("generated", 0)),
    )


def sweep_stage(stage: str, configs: list[tuple[str, list[str]]]) -> list[Run]:
    print(f"\n========== Stage {stage}: {len(configs)} configs x 3 windows ==========", flush=True)
    rows: list[Run] = []
    for name, extra in configs:
        triplet: list[Run] = []
        for window in WINDOWS:
            print(f"  {stage} {name} {window[0]}", flush=True)
            triplet.append(run_one(stage, name, window, extra))
        rows.extend(triplet)
        worst_r = min(r.expectancy_r for r in triplet)
        worst_eq = min(r.final_equity for r in triplet)
        sum_eq = sum(r.final_equity for r in triplet)
        print(
            f"    -> worst_R={worst_r:+.3f}  worst_eq={worst_eq:>8.2f}  sum_eq={sum_eq:>9.2f}",
            flush=True,
        )
    return rows


def best_by_worst_r(rows: list[Run]) -> tuple[str, list[str]]:
    by_cfg: dict[str, list[Run]] = {}
    for r in rows:
        by_cfg.setdefault(r.config_name, []).append(r)
    ranked = []
    for name, triplet in by_cfg.items():
        worst_r = min(r.expectancy_r for r in triplet)
        sum_eq = sum(r.final_equity for r in triplet)
        ranked.append((worst_r, sum_eq, name, list(triplet[0].extra_args)))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print(f"\n  Stage winner: {ranked[0][2]}  worst_R={ranked[0][0]:+.3f}  sum_eq={ranked[0][1]:>9.2f}")
    for worst_r, sum_eq, name, _ in ranked:
        print(f"    {name:<40} worst_R={worst_r:+.3f}  sum_eq={sum_eq:>9.2f}")
    return ranked[0][2], ranked[0][3]


def main() -> int:
    all_rows: list[Run] = []

    # ---- Stage 1: --trail-activation-r ----
    stage1_configs = [
        ("act_0.00", ["--trail-activation-r", "0.0"]),
        ("act_0.25", ["--trail-activation-r", "0.25"]),
        ("act_0.50", ["--trail-activation-r", "0.5"]),
        ("act_0.75", ["--trail-activation-r", "0.75"]),
        ("act_1.00", ["--trail-activation-r", "1.0"]),
        ("act_1.50", ["--trail-activation-r", "1.5"]),
        ("act_2.00", ["--trail-activation-r", "2.0"]),
    ]
    s1_rows = sweep_stage("1", stage1_configs)
    all_rows.extend(s1_rows)
    s1_winner, s1_args = best_by_worst_r(s1_rows)

    # ---- Stage 2: --adaptive-trailing-callback-multiplier ----
    stage2_configs = [
        (f"{s1_winner}+mult_0.20", s1_args + ["--adaptive-trailing-callback-multiplier", "0.2"]),
        (f"{s1_winner}+mult_0.30", s1_args + ["--adaptive-trailing-callback-multiplier", "0.3"]),
        (f"{s1_winner}+mult_0.40", s1_args + ["--adaptive-trailing-callback-multiplier", "0.4"]),
        (f"{s1_winner}+mult_0.50", s1_args + ["--adaptive-trailing-callback-multiplier", "0.5"]),
        (f"{s1_winner}+mult_0.60", s1_args + ["--adaptive-trailing-callback-multiplier", "0.6"]),
    ]
    s2_rows = sweep_stage("2", stage2_configs)
    all_rows.extend(s2_rows)
    s2_winner, s2_args = best_by_worst_r(s2_rows)

    # ---- Stage 3: --runner-pct ---- (note: BASE_ARGS hardcodes --runner-pct 50,
    # so we need to OVERRIDE later by appending; argparse takes last value).
    stage3_configs = [
        (f"{s2_winner}+runner_30", s2_args + ["--runner-pct", "30"]),
        (f"{s2_winner}+runner_50", s2_args + ["--runner-pct", "50"]),
        (f"{s2_winner}+runner_70", s2_args + ["--runner-pct", "70"]),
    ]
    s3_rows = sweep_stage("3", stage3_configs)
    all_rows.extend(s3_rows)
    s3_winner, s3_args = best_by_worst_r(s3_rows)

    # ---- Stage 4: --breakeven-trigger-r ----
    stage4_configs = [
        (f"{s3_winner}+be_0.0", s3_args + ["--breakeven-trigger-r", "0.0"]),
        (f"{s3_winner}+be_1.0", s3_args + ["--breakeven-trigger-r", "1.0"]),
        (f"{s3_winner}+be_1.5", s3_args + ["--breakeven-trigger-r", "1.5"]),
        (f"{s3_winner}+be_2.0", s3_args + ["--breakeven-trigger-r", "2.0"]),
    ]
    s4_rows = sweep_stage("4", stage4_configs)
    all_rows.extend(s4_rows)
    s4_winner, s4_args = best_by_worst_r(s4_rows)

    # Persist
    with open("sweep_stages_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "config", "window", "regime",
            "final_equity", "expectancy_r", "win_rate", "profit_factor",
            "max_dd", "taken", "generated", "extra_args",
        ])
        for r in all_rows:
            w.writerow([
                r.stage, r.config_name, r.window, r.regime,
                f"{r.final_equity:.2f}", f"{r.expectancy_r:.3f}",
                f"{r.win_rate:.2f}", f"{r.profit_factor:.2f}",
                f"{r.max_dd:.2f}", r.taken, r.generated,
                " ".join(r.extra_args),
            ])

    print(f"\n\n=== FINAL WINNER ===")
    print(f"  config: {s4_winner}")
    print(f"  args:   {' '.join(s4_args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
