# Binance Futures Breakout Bot

A no-dependency Python bot that scans Binance USD-M futures perpetuals for compression-and-breakout setups, places managed entries with smart exits, and runs a continuous live auto-trader. Includes a deterministic backtester, an ML rank model, and a multi-window validation discipline that drives every default in the codebase.

> ⚠️ **This bot trades real money.** Use `--testnet` until you have validated the setup against your own risk tolerance. The numbers below come from backtests on a specific 3-window dataset and do not guarantee live results.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Quick start](#quick-start)
3. [Recommended live config](#recommended-live-config)
4. [Strategy design](#strategy-design)
5. [CLI reference](#cli-reference)
6. [Backtest workflow](#backtest-workflow)
7. [ML ranking](#ml-ranking)
8. [Project layout](#project-layout)
9. [Testing](#testing)
10. [Disclaimer](#disclaimer)

---

## What it does

- **Detects** compressed-then-expanding breakout patterns on Binance USD-M perpetuals across multiple timeframes (`15m`, `1h`, `4h` by default).
- **Arms** managed entries through a `SMART_RETEST` state machine: watches for breakout trigger → optional retest pullback → falls back to market with deviation guards.
- **Places** stop-loss, partial take-profits, and a trailing runner as Binance algo orders the moment a position fills.
- **Manages** open positions live: dynamic stop-loss ratchet, breakeven floor, stagnation exit, position rotation when stronger signals appear, dust-sweep on close.
- **Backtests** the entire pipeline on cached klines with the same code paths the live bot uses, including auto-sizing curves and concurrent-position limits.
- **Trains** a ranking ML model from backtest trade logs and applies it live to prioritize the most promising signals without hard-filtering them.

---

## Quick start

### Prerequisites

- Python 3.10+
- A Binance USD-M futures account (real or testnet)

### Install

```powershell
git clone <your-fork-url> screener
cd screener
pip install -e .          # registers the breakout-detector entry point (optional)
```

The project has **zero runtime dependencies** beyond the Python standard library. `pip install -e .` is only needed if you want the `breakout-detector` shell command.

### Credentials

Create a `.env` file in the project root:

```ini
BINANCE_API_KEY=your-real-key
BINANCE_API_SECRET=your-real-secret

# Optional: Binance demo / testnet keys, used with --testnet
BINANCE_TESTNET_API_KEY=your-testnet-key
BINANCE_TESTNET_API_SECRET=your-testnet-secret
```

`.env` is gitignored. Never commit it.

### Verify auth + permissions

```powershell
python .\breakout_detector.py --auth-check --auth-check-trade --order-notional 25
```

This calls Binance's signed account-info endpoint and runs a tiny `--test` order to confirm trade permission. It does **not** place a live order.

### First live launch (testnet)

```powershell
python -u .\breakout_detector.py `
  --testnet `
  --manage-exits `
  --sizing-mode auto --leverage 10 `
  --max-concurrent-orders 2 `
  --trailing-stop `
  --dynamic-sl --breakeven-trigger-r 1.5 `
  --stagnation-after-r 0.5 --stagnation-candles 12 `
  --scan-interval-minutes 3
```

You should see `Auto-trader started.` followed by a scan summary every 3 minutes.

---

## Recommended live config

This command was validated across 3 regime windows (downtrend / chop / uptrend) with the same code path the live bot uses. It reproduces the **+11127% W3 / +0.796 R baseline** in the backtester:

```powershell
python -u .\breakout_detector.py `
  --manage-exits `
  --sizing-mode auto --leverage 10 `
  --max-concurrent-orders 2 `
  --trailing-stop `
  --dynamic-sl --sl-update-interval-seconds 300 --sl-lookback 20 `
  --breakeven-trigger-r 1.5 --breakeven-offset-pct 0.1 `
  --stagnation-after-r 0.5 --stagnation-candles 12 `
  --max-sl-loss-pct 35 `
  --scan-interval-minutes 3
```

**What's deliberately NOT in the recommended config:**

- ❌ `--exhaustion-exit` — the hardcoded 4-candle stall cuts breakout consolidations short; tested net-negative on 4 windows.
- ❌ `--reserve-last-slot-s-tier` (backtest flag) — at `N=2` it costs ~95% of return by gating too aggressively.
- ❌ `--smart-tp` — not re-validated since recent exit-side changes; defaults stand.

---

## Strategy design

### Entry pipeline

```text
SymbolUniverse  ─►  Liquidity + 24h volume filters  ─►  Per-timeframe detector
                                                              │
                                                              ▼
                                                  BreakoutSignal candidates
                                                              │
                                ┌─────────────────────────────┴───────────────────┐
                                ▼                                                 ▼
                          ML rank (optional)                              BTC market guards
                                │                                                 │
                                └──────────────► Top N per symbol ◄───────────────┘
                                                              │
                                                              ▼
                                                Arm with SMART_RETEST manager
```

Every arming step persists to `.pending_entry_orders.json` so a restart resumes mid-trade.

### SMART_RETEST state machine

For each armed signal:

1. **WAIT_BREAKOUT** — mark price hasn't reached the trigger yet. Cheap to hold.
2. **WAIT_RETEST** — trigger fired. The bot enters at market immediately for every regime (a 27-run sweep showed trigger-entry beats waiting for a retest pullback in every regime). Deviation cap (`--max-market-deviation-pct`) abandons the trade if price ran too far.
3. **Structural-break guard** — if the planned SL is already breached by current mark, abandon entirely. Prevents the "10-hour wait then fill into a falling knife" pattern.
4. **ENTRY_ORDER_PLACED** — entry submitted to Binance.
5. **MONITORING** — position open. Exit plans attach, dynamic SL ratchets, stagnation/rotation gates run.

### Exit stack (live)

Each fill places, in this order:

1. **Stop-loss** — `closePosition: true` invalidation at the structure level (or leverage-capped, whichever is tighter).
2. **TP1** — partial take-profit at the consolidation-range target.
3. **Trailing runner** — `TRAILING_STOP_MARKET` at `--trailing-callback-pct` (default `1.2%`) on the **exact remainder** of the position (recent fix — used to leave lot-step dust).

Additional gates running every scan tick:

- **Dust sweep** — if the residual after partials drops below the exchange's min-notional, market-close it.
- **Stagnation exit** — once unrealized profit ≥ `+0.5 R`, market-close if no new favourable extreme in `--stagnation-candles` (default `12`) bars. Validated to beat plain-trailing alone in 3-window sweep.
- **Dynamic SL ratchet** — every `--sl-update-interval-seconds`, recompute swing-low / swing-high and ratchet the stop in the favourable direction only.
- **Breakeven floor** — once unrealized profit ≥ `+1.5 R`, lock the stop at `entry + 0.1%` (or below entry for shorts). Outranks the swing-based ratchet when applicable.
- **Position rotation** — if all slots are full and a fresh INSTANT exploder out-ranks an open trade by `momentum_score + 0.20`, close the weaker position to chase the explosive one (profitable rotations are automatic; loss-cuts prompt for approval).

### BTC market guards

Optional (`--btc-market-guards`). Read BTC's own EMA + momentum each scan, then:

- **INSTANT entries** require BTC momentum ≥ `--instant-guard-momentum-pct` (default `-2%`) AND close within `--instant-guard-ema-slack-pct` (default `1.5%`) of EMA.
- **Hostile BTC** (momentum < 0 or close below EMA) downgrades all entries to `STRICT_RETEST` only.
- **SHORT entries** require BTC momentum ≤ `--hostile-momentum-pct` (default `0%`).

### Auto-sizing curve

`--sizing-mode auto` ladders position size by absolute equity:

| Equity band | % of equity per trade |
|-------------|-----------------------|
| < $25       | 55%                   |
| $25–$100    | 45%                   |
| $100–$500   | 32%                   |
| $500–$2,500 | 22%                   |
| $2,500–$10k | 15%                   |
| > $10k      | 10%                   |

A drawdown haircut multiplies the tier % by `(1 - drawdown%)` when equity is below the running peak (`.equity_peak.json`). Designed to grow micro accounts aggressively while de-risking on drawdown.

---

## CLI reference

The bot is one long-running process: `python .\breakout_detector.py [flags]`. All scans, entries, and exits run inside that process.

### Core trading

| Flag | Default | Purpose |
|------|---------|---------|
| `--testnet` | off | Use Binance demo/testnet. Prefers testnet keys. |
| `--manage-exits` | off | Attach SL/TP/trailing as soon as entries fill. **Required for live trading.** |
| `--max-concurrent-orders` | 0 (∞) | Max simultaneous open positions or entry-order placements. |
| `--queue-size` | 0 (∞) | Max coins armed and watched at once. Independent of position cap. |
| `--scan-interval-minutes` | 3 | How often to re-scan for new opportunities. |
| `--leverage` | 0 | Set initial leverage per symbol. Auto-sized to 10 with `--sizing-mode auto`. |
| `--dynamic-leverage` | off | Scale leverage per coin from ATR. |
| `--margin-type ISOLATED\|CROSSED` | unset | Apply before placing entry. |
| `--hedge-mode` | off | Send `positionSide` for hedge accounts. |

### Position sizing

| Flag | Default | Purpose |
|------|---------|---------|
| `--sizing-mode fixed\|auto` | `fixed` | `auto` ladders by equity. |
| `--order-margin` | 0 | USDT margin per trade (`fixed` mode). |
| `--order-notional` | 0 | USDT notional per trade (`fixed` mode). |
| `--equity-peak-file` | `.equity_peak.json` | Drawdown-haircut state. Delete to reset. |

### Entry mode (`SMART_RETEST` family)

| Flag | Default | Purpose |
|------|---------|---------|
| `--entry-mode SMART_RETEST\|RETEST_LIMIT\|STOP_MARKET` | `SMART_RETEST` | Execution model. |
| `--retest-timeout-seconds` | 300 | How long `SMART_RETEST` waits for a retest before market fallback. |
| `--entry-pullback-pct` | 0.5 | Limit-entry distance for `RETEST_LIMIT`. |
| `--max-market-deviation-pct` | 1.5 | Skip market entry if price ran more than this past trigger. |
| `--no-market-fallback` | off | Never use market fallback (for `SMART_RETEST`). |
| `--adaptive-entry` | off | Auto-pick entry regime per coin from market conditions. |
| `--skip-entry-regimes INSTANT,RETEST,TRAILING_RETEST,STRICT_RETEST` | empty | Skip selected regimes. |

### Exit management

| Flag | Default | Purpose |
|------|---------|---------|
| `--trailing-stop` | off | Attach `TRAILING_STOP_MARKET` runner. |
| `--trailing-callback-pct` | 1.2 | Runner callback %. Tuned to match the validated backtest baseline. |
| `--trailing-quantity-pct` | 50 | Runner share of position (when `--tp-splits` not set). |
| `--tp-count` | 1 | Number of partial TP orders. |
| `--tp-splits 40,30,20` | unset | Explicit TP qty %s. Remainder becomes the trailing runner. |
| `--dynamic-sl` | off | Ratchet SL on new swing-low/high. |
| `--sl-update-interval-seconds` | 300 | Min seconds between SL re-evaluations. |
| `--sl-lookback` | 20 | Candles considered for swing-low/high. |
| `--breakeven-trigger-r` | 1.5 | R-multiple at which the SL ratchets to breakeven. `0` = off. |
| `--breakeven-offset-pct` | 0.1 | Breakeven SL sits this far past entry (profit lock + fees). |
| `--max-sl-loss-pct` | 35 | Cap leveraged loss at this % of margin. `0` = off. |
| `--stagnation-after-r` | 0 | Activate stagnation exit at this unrealized R. `0.5` is the validated default. |
| `--stagnation-candles` | 12 | Closed candles with no new favourable extreme before market-close. |
| `--stagnation-lookback` | 80 | Klines fetched per monitored position for the stagnation check. |
| `--exhaustion-exit` | off | **Discouraged**: hardcoded 4-candle stall cuts breakout consolidations short. Use stagnation instead. |
| `--smart-tp` | off | Adaptive target / TP splits from signal conviction. Not re-validated post exit-fix. |
| `--no-exits` | off | Place entries only, no managed exits (use only for diagnostics). |
| `--manage-exits` | required for live | Attach the exit stack after fills. |
| `--watch-exits` | off | Block until all pending exits attach (or timeout). |
| `--no-auto-manage-exits` | off | Skip the auto attach loop. |

### Detection filters

Each filter rejects setups that fail it. Defaults are tuned for liquid altcoin futures.

| Flag | Default | Purpose |
|------|---------|---------|
| `--quote` | `USDT` | Quote asset, or `ALL`. |
| `--interval` | unset | Single timeframe override. |
| `--timeframes` | `15m,1h,4h` | Multi-TF scan. |
| `--top` | 0 (all) | Top N by 24h quote volume. |
| `--symbols`, `--symbols-file` | unset | Restrict universe. |
| `--include-dead` | off | Disable liquidity filters. |
| `--skip-book-filter`, `--skip-oi-filter` | off | Disable specific filter. |
| `--min-quote-volume` | 1M | 24h quote volume floor. |
| `--min-trades` | 2000 | 24h trade-count floor. |
| `--min-24h-range-pct` | 2.0 | 24h H-L range floor. |
| `--min-book-depth` | 25k | Bid/ask depth near mid. |
| `--max-spread-bps` | 25 | Spread ceiling. |
| `--min-open-interest-notional` | 5M | OI floor. |
| `--max-extension-pct` | 3.5 | Reject moves already extended past resistance. |
| `--max-trigger-distance-pct` | 2.0 | Reject if mark is too far from trigger. |
| `--max-pre-trigger-move-pct` | 3.0 | Reject chases. |
| `--max-shakeout-distance-pct` | 4.0 | Spring/upthrust reclaim ceiling. |
| `--max-compression-pct` | 18.0 | Consolidation-width ceiling. |
| `--min-breakout-pct` | varies | Trigger-strength floor. |
| `--detector simple\|squeeze` | `simple` | High-recall long-only vs. original squeeze. |

### BTC market guards

| Flag | Default | Purpose |
|------|---------|---------|
| `--btc-market-guards` | off | Enable BTC-conditioned entry gating. |
| `--btc-guard-interval` | 1h | BTC timeframe. |
| `--btc-ema-candles` | 72 | EMA length. |
| `--btc-momentum-candles` | 72 | Momentum lookback. |
| `--instant-guard-momentum-pct` | -2.0 | Min BTC momentum for INSTANT entries. |
| `--instant-guard-ema-slack-pct` | 1.5 | Max BTC close-below-EMA for INSTANT entries. |
| `--hostile-momentum-pct` | 0.0 | Below = hostile (STRICT_RETEST only). |

### ML ranking (optional)

| Flag | Default | Purpose |
|------|---------|---------|
| `--ml-rank-model PATH` | unset | Load a model trained by `train_model.py`. |
| `--ml-rank-score pwin\|expected-r\|tail\|not-bad\|composite` | `tail` | Score head used for ranking. |
| `--ml-context-file` | `.ml_symbol_context.json` | Rolling per-symbol outcome state. |

### Auth / connectivity

| Flag | Default | Purpose |
|------|---------|---------|
| `--auth-check` | off | Validate signed account access + return early. |
| `--auth-check-trade` | off | Also run a `--test` order to verify trade permission. |
| `--env-file` | `.env` | Override credentials file. |
| `--base-url` | unset | Override Binance USD-M base URL. |
| `--timeout` | 10 | HTTP timeout (sec). |
| `--retries` | 2 | Retries for transient errors. |
| `--rate-limit-rpm` | 1100 | Stay under Binance's 2400/min IP cap. |
| `--workers` | 4 | Parallel kline requests per scan. |

---

## Backtest workflow

The backtester (`python -m screener.backtest`) replays the entire pipeline on cached klines. Same detector, same entry-mode classifier, same exit stack — only the live-order placement is simulated.

### Standard 3-window sweep (multi-window validation)

```powershell
foreach ($w in @(
    @{ id='W1'; start='2025-12-20'; end='2026-02-10' },   # downtrend
    @{ id='W2'; start='2026-02-10'; end='2026-04-01' },   # chop
    @{ id='W3'; start='2026-04-01'; end='2026-05-21' }    # uptrend
)) {
    python -m screener.backtest `
      --top 40 --interval 1h --capital 10 --compound `
      --sizing-mode auto --max-concurrent 2 `
      --trailing-stop --runner-pct 50 `
      --simulate-rotation `
      --stagnation-after-r 0.5 --stagnation-candles 12 `
      --start-date $w.start --end-date $w.end `
      --trade-log "backtest_W$($w.id).csv"
}
```

**Discipline**: never declare a winner from one window. Pick the config with the best **worst-case** R across all 3 windows, not the best average. A single window over-fits to that regime; the multi-window minimum is the honest number.

### Backtest-only flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--start-date YYYY-MM-DD` | unset | Window start. |
| `--end-date YYYY-MM-DD` | unset | Window end. |
| `--capital` | 100 | Starting USDT. |
| `--compound` | off | Compound equity across trades. |
| `--max-concurrent` | 2 | Same semantic as live `--max-concurrent-orders`. |
| `--runner-pct` | 25 | Trailing runner share. |
| `--simulate-rotation` | off | Apply the rotation simulator. |
| `--reserve-last-slot-s-tier` | off | Reserve final slot for momentum_score ≥ threshold. **Hurts at N=2; helps at N=3/4.** |
| `--s-tier-momentum-threshold` | 0.85 | Reservation gate. |
| `--ml-filter-model PATH` | unset | Use ML output as a hard filter (research only). |
| `--ml-filter-score` | `pwin` | Score head for the hard filter. |
| `--ml-rank-signals` | off | Rank rather than threshold. |
| `--trade-log PATH` | unset | Dump each trade as CSV for ML training. |
| `--sizing-mode fixed\|guarded\|moonshot\|aggressive\|auto` | `moonshot` | Sizing curve. Backtest has more options than live. |
| `--position-pct N` | 0 | Fixed % per trade (overrides sizing-mode). |

The backtester caches klines on disk under `.backtest_kline_cache/`. Repeated sweeps over the same window are near-instant. The cache is gitignored (it's regen-able and runs into the GB range).

---

## ML ranking

The bot can score and rank signals using a trained linear model. **Ranking only** — the model never hard-filters trades in live.

### Train a model

1. Generate a trade log:
   ```powershell
   python -m screener.backtest --top 40 --interval 1h --start-date 2025-05-01 --end-date 2026-05-21 --capital 10 --compound --trailing-stop --runner-pct 50 --simulate-rotation --stagnation-after-r 0.5 --stagnation-candles 12 --trade-log trades.csv
   ```
2. Train (with a forward-time split):
   ```powershell
   python train_model.py --input-csv trades.csv --split-date 2026-04-01 --label-all-signals --feature-set full --output my_model.json
   ```
3. Walk-forward validate:
   ```powershell
   python walk_forward_ml_validate.py --input-csv trades.csv --feature-set full --output ml_walk_forward.csv
   ```

### Use it live

```powershell
python .\breakout_detector.py --ml-rank-model my_model.json --ml-rank-score tail [other flags]
```

A trained example model is shipped: `model_live_context_tail_rank.json`.

---

## Project layout

```
screener/
├── breakout_detector.py        # Live entry point (thin wrapper -> screener.cli.main)
├── backtest.py                 # Backtest entry point (thin wrapper -> screener.backtest.main)
├── train_model.py              # ML training CLI
├── walk_forward_ml_validate.py # Walk-forward ML validation
├── ml_family_search.py         # Hyperparameter search for the ML head
├── model_live_context_tail_rank.json  # Validated ML model artifact
├── pyproject.toml
├── README.md
├── ML_FINAL_REPORT.md          # Research log on the ML rank experiment
├── screener/                   # Package
│   ├── __init__.py
│   ├── cli.py                  # Live auto-trader (largest module)
│   ├── backtest.py             # Backtest engine
│   ├── breakout.py             # Signal detector
│   ├── binance_client.py       # Signed REST client (stdlib only)
│   ├── orders.py               # Order plan builders
│   └── take_profit.py          # TP profile helpers
└── tests/                      # pytest / unittest suite (28 tests)
```

### State files (gitignored, written by the live bot)

- `.pending_entry_orders.json` — armed signals and their state machine
- `.pending_exit_orders.json` — deferred exit plans waiting on fills
- `.equity_peak.json` — running equity peak for the drawdown haircut
- `.ml_symbol_context.json` — rolling per-symbol outcome history for ML features

Delete any of these to reset that piece of state without recompiling.

---

## Testing

```powershell
python -m pytest tests/ -q
```

28 tests cover: signal detection, breakout filters, order plan construction (incl. TP+trail remainder math), trailing-stop callback bounds, take-profit splits, and the binance client signing path.

---

## Disclaimer

This is a research and engineering project, not financial advice. Crypto futures with leverage can drain an account quickly. Backtest results are a property of the dataset they were measured on; live results will diverge due to slippage, latency, funding rate variance, exchange downtime, and regime changes.

Validate on `--testnet` first. Start with the smallest sizing you can. Read the code before you trust it with money.
