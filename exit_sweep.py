"""Exit-loosening sweep — test 'let winners run' against overfit.

Diagnosis (2024): 74% of winners exit under +1R; the 4 biggest runners are 34%
of all profit. The stagnation exit fires after only +0.5R then closes on a
12-candle stall, and the adaptive trailing callback (x0.3) is tight. Both cap
winners. This sweep loosens them and scores on 2024-OOS + key windows.

Each variant is scored by robust = equity*(1-DD). 2024 is the OOS acid test;
windows are sanity. We keep a change only if it lifts 2024 robust WITHOUT
wrecking the windows.
"""
from __future__ import annotations
import json, re, subprocess, sys, copy
from pathlib import Path

CAP, LEV = "40", "5"
BASE = [
    sys.executable, "-m", "screener.backtest", "--top", "0",
    "--intervals", "15m,1h,4h", "--btc-guard-interval", "1h",
    "--mtf-alignment-tf", "4h", "--capital", CAP, "--compound",
    "--sizing-mode", "auto", "--simulate-rotation", "--trailing-stop",
    "--adaptive-trailing-callback", "--dynamic-sl", "--btc-chop-guards",
    "--stagnation-candles", "12", "--leverage", LEV, "--max-sl-loss-pct", "50",
    "--runner-pct", "50", "--bear-profile", "--offline-cache-dir", "_kline_cache",
    "--workers", "16",
]
PERIODS = [
    ("2024", "2024-01-01", "2024-12-31"),
    ("W2grind", "2025-02-20", "2025-04-05"),
    ("W4chop", "2025-06-04", "2025-07-23"),
    ("W6recov", "2026-02-25", "2026-04-15"),
]
RE_EQ = re.compile(r"Final equity\s+([\d.]+)\s+USDT")
RE_DD = re.compile(r"Max drawdown\s+([\d.]+)%")

base_cfg = json.load(open("window_config.json"))


def make_cfg(stag_after_r):
    """Return a temp window_config path with stagnation_after_r overridden in
    every regime (None = leave as-is / baseline)."""
    if stag_after_r is None:
        return "window_config.json"
    cfg = copy.deepcopy(base_cfg)
    for w in cfg.values():
        w.setdefault("overrides", {})["stagnation_after_r"] = stag_after_r
    p = f"sweep_logs/wc_stag_{stag_after_r}.json"
    json.dump(cfg, open(p, "w"))
    return p


# (label, stagnation_after_r override, trailing-callback-mult, runner_pct)
VARIANTS = [
    ("baseline",          None, "0.3", "50"),
    ("stag_off",          0.0,  "0.3", "50"),
    ("stag_off+widetrl",  0.0,  "0.5", "65"),
    ("stag_2.0",          2.0,  "0.3", "50"),
    ("stag_2.0+widetrl",  2.0,  "0.5", "65"),
]


def run(cfg, mult, runner, s, e):
    cmd = BASE + ["--window-config", cfg,
                  "--adaptive-trailing-callback-multiplier", mult,
                  "--runner-pct", runner,
                  "--start-date", s, "--end-date", e,
                  "--trade-log", "sweep_logs/exit_tmp.csv"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    eq = float(m.group(1)) if (m := RE_EQ.search(p.stdout)) else 0.0
    dd = float(m.group(1)) if (m := RE_DD.search(p.stdout)) else 100.0
    return eq, dd


def main():
    Path("sweep_logs").mkdir(exist_ok=True)
    print(f"Exit sweep @ ${CAP} {LEV}x — robust = eq*(1-DD)\n", flush=True)
    for label, stag, mult, runner in VARIANTS:
        cfg = make_cfg(stag)
        parts, oos_robust = [], 0.0
        for name, s, e in PERIODS:
            eq, dd = run(cfg, mult, runner, s, e)
            r = eq * (1 - dd / 100)
            if name == "2024":
                oos_robust = r
            parts.append(f"{name} ${eq:.0f}@{dd:.0f}%(r{r:.0f})")
        print(f"{label:<18} 2024_robust={oos_robust:>7.0f} | {' '.join(parts)}", flush=True)


if __name__ == "__main__":
    main()
