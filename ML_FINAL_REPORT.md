# ML Final Report

Date: 2026-05-23

## Decision

The full context/exhaustion stack is now shipped to live as opt-in behavior:

- Live exhaustion monitoring: `--exhaustion-exit`
- Live context ML ranker: `--ml-rank-model model_live_context_tail_rank.json --ml-rank-score tail`
- ML still ranks candidates only. It does not hard-filter trades.

The shipped artifact is `model_live_context_tail_rank.json`.

## What Changed

Live now has the missing pieces that blocked the earlier `2768.23` result:

- Monitors open positions after fill.
- Tracks exhaustion state per pending entry.
- Market-closes on the same core backtest exhaustion rule: after +0.5R, close on peak rejection or 4 closed candles without a new favorable extreme.
- Computes live BTC context: BTC momentum and BTC EMA distance.
- Computes coin-vs-BTC relative momentum during scans.
- Stores rolling symbol outcome context in `.ml_symbol_context.json`.
- Accepts full-context artifacts with BTC, relative momentum, and rolling symbol features.

## Training

Dataset regeneration:

```powershell
python -m screener.backtest --top 40 --interval 1h --start-date 2025-05-01 --end-date 2026-05-21 --capital 10 --compound --trailing-stop --runner-pct 50 --simulate-rotation --exhaustion-exit --trade-log backtest_trades_massive_context_exhaustion.csv
```

Model training:

```powershell
python train_model.py --input-csv backtest_trades_massive_context_exhaustion.csv --split-date 2026-04-01 --label-all-signals --feature-set full --output model_live_context_tail_rank.json
```

Holdout model quality:

- Train: 1217 generated signals.
- Test: 275 generated signals.
- Plain win/loss AUC: `0.493`, useless.
- Right-tail AUC, `R >= 1.5`: `0.773`, strong enough for ranking.

That is why live uses `tail`, not `pwin`.

## Backtests

Current live-shaped sizing, no ML:

- Command adds `--exhaustion-exit`.
- Final equity: `422.09 USDT`.
- Win rate: `52.0%`.
- Expectancy: `+0.181 R`.
- Profit factor: `1.44`.
- Max drawdown: `39.7%`.

Current live-shaped sizing, context ML rank-tail:

```powershell
python -m screener.backtest --top 40 --interval 1h --start-date 2025-05-01 --end-date 2026-05-21 --capital 10 --compound --trailing-stop --runner-pct 50 --simulate-rotation --exhaustion-exit --ml-filter-model model_live_context_tail_rank.json --ml-filter-score tail --ml-score-only --ml-rank-signals --trade-log backtest_trades_context_ml_rank_tail.csv
```

- Final equity: `1059.36 USDT`.
- Win rate: `52.4%`.
- Expectancy: `+0.232 R`.
- Profit factor: `1.77`.
- Max drawdown: `38.4%`.

Fixed 20% compounding check:

- No ML: `1041.67 USDT`, max DD `70.0%`.
- Context ML rank-tail: `7737.62 USDT`, max DD `63.6%`.

This explains the old `2768.23` question: the huge final equity comes from high compounding exposure. The edge is real in this run, but the equity number is very sensitive to sizing.

## Walk-Forward

Command:

```powershell
python walk_forward_ml_validate.py --input-csv backtest_trades_massive_context_exhaustion.csv --feature-set full --output ml_walk_forward_context_exhaustion_summary.csv
```

Aggregate:

- Baseline avg fold final equity: `16.394`.
- `rank-tail` avg fold final equity: `24.439`.
- `rank-tail` wins versus baseline: `4 / 6` folds.
- Baseline avg max DD: `42.465%`.
- `rank-tail` avg max DD: `41.103%`.

Verdict: acceptable as ranking-only. Still not acceptable as a hard filter.

## Live Commands

Testnet first:

```powershell
python .\breakout_detector.py --testnet --timeframes 1h --order-margin 5 --leverage 10 --dynamic-leverage --max-concurrent-orders 2 --queue-size 8 --order-count 3 --adaptive-entry --skip-entry-regimes TRAILING_RETEST --btc-market-guards --trailing-stop --trailing-quantity-pct 50 --exhaustion-exit --ml-rank-model model_live_context_tail_rank.json --ml-rank-score tail
```

Mainnet live:

```powershell
python .\breakout_detector.py --timeframes 1h --order-margin 5 --leverage 10 --dynamic-leverage --max-concurrent-orders 2 --queue-size 8 --order-count 3 --adaptive-entry --skip-entry-regimes TRAILING_RETEST --btc-market-guards --trailing-stop --trailing-quantity-pct 50 --exhaustion-exit --ml-rank-model model_live_context_tail_rank.json --ml-rank-score tail
```

Notes:

- `--order-margin 5` is fixed margin per entry. The `7737` check used fixed `20%` compounding in the backtester, not this fixed live margin.
- `.ml_symbol_context.json` starts neutral and improves as the live bot records outcomes.
- Exhaustion closes are market orders and cancel leftover exit algos.

## Verification

```powershell
python -m pytest
```

Result: `28 passed`.

Live model load check:

- `model_live_context_tail_rank.json` loads with 22 full-context features.
