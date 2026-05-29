"""Run all 6 windows with per-window hyperopt params and print a summary table."""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WINDOWS = [
    ("W1_bear_crash",    "W1", "2025-12-20", "2026-02-10", "Bear crash+recovery -21.9%"),
    ("W2_bear_grind",    "W2", "2025-02-20", "2025-04-05", "Bear grinding -19.7%"),
    ("W3_bull_strong",   "W3", "2026-04-01", "2026-05-21", "Bull strong +14.1%"),
    ("W4_bull_chop",     "W4", "2025-06-04", "2025-07-23", "Bull choppy +13.4%"),
    ("W5_bear_crash2",   "W5", "2025-10-08", "2025-11-26", "Bear massive -28.2%"),
    ("W6_bull_recovery", "W6", "2026-02-25", "2026-04-15", "Bull recovery +16.1%"),
]

HYPEROPT_FILE = "hyperopt_per_window_best.json"

BASE_CMD = [
    sys.executable, "-m", "screener.backtest",
    "--top", "0",
    "--intervals", "15m,1h,4h",
    "--btc-guard-interval", "1h",
    "--mtf-alignment-tf", "4h",
    "--capital", "10",
    "--compound",
    "--sizing-mode", "auto",
    "--simulate-rotation",
    "--trailing-stop",
    "--adaptive-trailing-callback",
    "--dynamic-sl",
    "--btc-chop-guards",
    "--stagnation-candles", "12",
    "--leverage", "10",
    "--max-sl-loss-pct", "50",
    "--runner-pct", "50",
    "--bear-profile",
    "--offline-cache-dir", "_kline_cache",
    "--window-config", "window_config.json",
    "--workers", "16",
]

RE_EQUITY      = re.compile(r"Final equity\s+([\d.]+)\s+USDT")
RE_EXPECTANCY  = re.compile(r"Expectancy\s+([+-]?[\d.]+)\s+R")
RE_WINRATE     = re.compile(r"Win rate\s+([\d.]+)%")
RE_TRADES      = re.compile(r"(?:Signals\s+\d+\s+generated\s+->\s+(\d+)\s+taken|(?:Taken trades|Trades taken|Total trades)\s+(\d+))")
RE_MAX_DD      = re.compile(r"Max drawdown\s+([\d.]+)%")

PARAM_TO_FLAG = {
    "breakeven_r":    "--breakeven-trigger-r",
    "breakeven_trigger_r": "--breakeven-trigger-r",
    "stag_after_r":   "--stagnation-after-r",
    "stagnation_after_r": "--stagnation-after-r",
    "stag_candles":   "--stagnation-candles",
    "stagnation_candles": "--stagnation-candles",
    "max_sl_loss_pct": "--max-sl-loss-pct",
    "mtf_ma_period":  "--mtf-alignment-ma-period",
    "max_concurrent": "--max-concurrent",
}

ENV_PARAMS = {"HP_INST_REL", "HP_INST_VOL", "HP_DV"}
HP_DV_DEFAULT = "3.0"


def load_hyperopt() -> dict:
    p = Path(HYPEROPT_FILE)
    if not p.exists():
        print(f"WARNING: {HYPEROPT_FILE} not found — using global defaults")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_cmd_and_env(base_cmd, start, end, wid_log, params: dict):
    cmd = list(base_cmd) + [
        "--start-date", start,
        "--end-date", end,
        "--trade-log", f"sweep_logs/6w_{wid_log}.csv",
    ]
    env = os.environ.copy()

    for k, v in params.items():
        if k in ENV_PARAMS:
            env[k] = str(v)
        elif k in PARAM_TO_FLAG:
            cmd += [PARAM_TO_FLAG[k], str(v)]

    env.setdefault("HP_DV", HP_DV_DEFAULT)

    return cmd, env


def run_window(wid_logical, wid_short, start, end, desc, hyperopt):
    generic_params = {"HP_DV": HP_DV_DEFAULT}

    if wid_short in hyperopt:
        best = hyperopt[wid_short]["best_params"]
        generic_params.update(best)
    else:
        print(f"  WARNING: {wid_short} not in hyperopt, using generic defaults")

    cmd, env = build_cmd_and_env(BASE_CMD, start, end, wid_logical, generic_params)
    print(f"  [{wid_logical}] params: HP_INST_REL={env.get('HP_INST_REL','?')}, "
          f"HP_INST_VOL={env.get('HP_INST_VOL','?')}, "
          f"breakeven_r={generic_params.get('breakeven_r','?')}, "
          f"stag_after_r={generic_params.get('stag_after_r','?')}, "
          f"mtf_ma_period={generic_params.get('mtf_ma_period','?')}, "
          f"max_concurrent={generic_params.get('max_concurrent','?')}", flush=True)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=900, check=False)
    except subprocess.TimeoutExpired:
        return wid_logical, desc, None
    except Exception as e:
        return wid_logical, desc, None

    if proc.returncode != 0:
        print(f"  FAIL {wid_logical}: {proc.stderr[-300:]}", flush=True)
        return wid_logical, desc, None

    out = proc.stdout
    equity     = float(m.group(1)) if (m := RE_EQUITY.search(out)) else 0.0
    expectancy = float(m.group(1)) if (m := RE_EXPECTANCY.search(out)) else 0.0
    win_rate   = float(m.group(1)) if (m := RE_WINRATE.search(out)) else 0.0
    trades     = int(next(g for g in m.groups() if g)) if (m := RE_TRADES.search(out)) else 0
    max_dd     = float(m.group(1)) if (m := RE_MAX_DD.search(out)) else 0.0
    return wid_logical, desc, dict(equity=equity, expectancy=expectancy,
                                   win_rate=win_rate, trades=trades, max_dd=max_dd)


def main():
    import pathlib
    pathlib.Path("sweep_logs").mkdir(exist_ok=True)

    hyperopt = load_hyperopt()
    print(f"Running {len(WINDOWS)} window backtests with per-window hyperopt params...\n")

    results = {}
    with ThreadPoolExecutor(max_workers=1) as ex:
        futs = {ex.submit(run_window, wid, wid_s, start, end, desc, hyperopt): wid
                for wid, wid_s, start, end, desc in WINDOWS}
        for fut in as_completed(futs):
            wid, desc, stats = fut.result()
            results[wid] = (desc, stats)
            if stats:
                print(f"  [{wid}] equity=${stats['equity']:.2f}  "
                      f"exp={stats['expectancy']:.3f}R  "
                      f"wr={stats['win_rate']:.1f}%  "
                      f"trades={stats['trades']}  "
                      f"maxDD={stats['max_dd']:.1f}%", flush=True)
            else:
                print(f"  [{wid}] FAILED", flush=True)

    print("\n" + "=" * 90)
    print(f"{'Window':<20} {'Period':<25} {'Equity':>9} {'Exp(R)':>8} {'WR%':>6} "
          f"{'Trades':>7} {'MaxDD%':>8}")
    print("-" * 90)
    total_eq = 0.0
    ok = 0
    for wid, wid_s, start, end, desc in WINDOWS:
        d, s = results.get(wid, (desc, None))
        period = f"{start} -> {end}"
        if s:
            print(f"{wid:<20} {period:<25} ${s['equity']:>8.2f} {s['expectancy']:>8.3f} "
                  f"{s['win_rate']:>5.1f}% {s['trades']:>7} {s['max_dd']:>7.1f}%")
            total_eq += s["equity"]
            ok += 1
        else:
            print(f"{wid:<20} {period:<25} {'FAILED':>9}")
    print("-" * 90)
    if ok:
        print(f"{'SUM / AVG':<46} ${total_eq:>8.2f}  (avg ${total_eq/ok:.2f} per window)")
    print("=" * 90)


if __name__ == "__main__":
    main()
