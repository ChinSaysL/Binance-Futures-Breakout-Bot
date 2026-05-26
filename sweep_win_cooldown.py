"""Win-cooldown sweep across 3 regime windows (W1/W2/W3).

Re-runs the backtester with --win-cooldown-candles in {0, 12, 24, 48, 96}
on each window using the README "standard 3-window sweep" config, parses
the printed summary, and reports the worst-case-across-windows expectancy
for each cooldown value.

Output: sweep_win_cooldown_results.csv
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

COOLDOWN_VALUES = [0, 12, 24, 48, 96]

BASE_ARGS = [
    sys.executable, "-m", "screener.backtest",
    "--top", "40", "--interval", "1h",
    "--capital", "10", "--compound",
    "--sizing-mode", "auto", "--max-concurrent", "2",
    "--trailing-stop", "--runner-pct", "50",
    "--simulate-rotation",
    "--stagnation-after-r", "0.5", "--stagnation-candles", "12",
]


@dataclass
class Result:
    window: str
    regime: str
    win_cooldown: int
    generated: int
    taken: int
    final_equity: float
    roi_pct: float
    win_rate: float
    expectancy_r: float
    profit_factor: float
    max_dd: float


SUMMARY_PATTERNS = {
    "signals": re.compile(r"Signals\s+(\d+)\s+generated\s+->\s+(\d+)\s+taken"),
    "equity": re.compile(r"Final equity\s+([\d.]+)\s+USDT\s+\(([+-]?[\d.]+)%"),
    "win_rate": re.compile(r"Win rate\s+([\d.]+)%"),
    "expectancy": re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R"),
    "profit_factor": re.compile(r"Profit factor\s+([\d.]+)"),
    "max_dd": re.compile(r"Max drawdown\s+([\d.]+)%"),
}


def _parse(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    m = SUMMARY_PATTERNS["signals"].search(stdout)
    if m:
        out["generated"] = float(m.group(1))
        out["taken"] = float(m.group(2))
    m = SUMMARY_PATTERNS["equity"].search(stdout)
    if m:
        out["final_equity"] = float(m.group(1))
        out["roi_pct"] = float(m.group(2))
    m = SUMMARY_PATTERNS["win_rate"].search(stdout)
    if m:
        out["win_rate"] = float(m.group(1))
    m = SUMMARY_PATTERNS["expectancy"].search(stdout)
    if m:
        out["expectancy_r"] = float(m.group(1))
    m = SUMMARY_PATTERNS["profit_factor"].search(stdout)
    if m:
        out["profit_factor"] = float(m.group(1))
    m = SUMMARY_PATTERNS["max_dd"].search(stdout)
    if m:
        out["max_dd"] = float(m.group(1))
    return out


def run_one(window: tuple[str, str, str, str], wc: int) -> Result:
    wid, start, end, regime = window
    cmd = BASE_ARGS + [
        "--start-date", start, "--end-date", end,
        "--win-cooldown-candles", str(wc),
        "--trade-log", f"sweep_{wid}_wc{wc}.csv",
    ]
    print(f"  running {wid} wc={wc}...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"  FAILED: {proc.stderr[-500:]}")
        raise SystemExit(1)
    parsed = _parse(proc.stdout)
    return Result(
        window=wid,
        regime=regime,
        win_cooldown=wc,
        generated=int(parsed.get("generated", 0)),
        taken=int(parsed.get("taken", 0)),
        final_equity=parsed.get("final_equity", 0.0),
        roi_pct=parsed.get("roi_pct", 0.0),
        win_rate=parsed.get("win_rate", 0.0),
        expectancy_r=parsed.get("expectancy_r", 0.0),
        profit_factor=parsed.get("profit_factor", 0.0),
        max_dd=parsed.get("max_dd", 0.0),
    )


def main() -> int:
    rows: list[Result] = []
    for window in WINDOWS:
        print(f"\n=== Window {window[0]} ({window[3]}): {window[1]} -> {window[2]} ===")
        for wc in COOLDOWN_VALUES:
            rows.append(run_one(window, wc))

    with open("sweep_win_cooldown_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "window", "regime", "win_cooldown",
            "generated", "taken",
            "final_equity", "roi_pct",
            "win_rate", "expectancy_r", "profit_factor", "max_dd",
        ])
        for r in rows:
            w.writerow([
                r.window, r.regime, r.win_cooldown,
                r.generated, r.taken,
                f"{r.final_equity:.2f}", f"{r.roi_pct:.2f}",
                f"{r.win_rate:.2f}", f"{r.expectancy_r:.3f}",
                f"{r.profit_factor:.2f}", f"{r.max_dd:.2f}",
            ])

    print("\n\n=== RESULTS ===")
    print(f"{'WC':>4} | {'W1 (down)':>22} | {'W2 (chop)':>22} | {'W3 (up)':>22} | worst R")
    print("-" * 110)
    by_wc: dict[int, list[Result]] = {}
    for r in rows:
        by_wc.setdefault(r.win_cooldown, []).append(r)
    for wc in COOLDOWN_VALUES:
        triplet = sorted(by_wc[wc], key=lambda r: r.window)
        worst_r = min(r.expectancy_r for r in triplet)
        cells = " | ".join(
            f"eq={r.final_equity:>8.1f} R={r.expectancy_r:+.3f}" for r in triplet
        )
        print(f"{wc:>4} | {cells} | {worst_r:+.3f}")

    print("\nWrote sweep_win_cooldown_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
