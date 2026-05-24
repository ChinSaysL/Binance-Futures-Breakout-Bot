"""Walk-forward backtester for the breakout strategy.

It runs the real ``evaluate_breakout`` detector over historical klines: for every
qualifying signal it simulates the breakout entry and the stop/take-profit/trailing
exits, then reports win rate, expectancy and drawdown.

Modeling choices (v1) - read these before trusting the numbers:
  * Entry: a market fill when price crosses the breakout trigger (gap-adjusted),
    plus slippage. The sub-candle SMART_RETEST discount is not modeled - the 300s
    retest window is shorter than one candle so most live entries fall back to
    market anyway. Slightly conservative on entry price.
  * Exits: a static stop loss, optional smart/even partial take-profits, and a
    trailing runner. Dynamic SL repositioning is NOT modeled.
  * Intrabar: if a candle's range covers both the stop and a take-profit, the
    stop is assumed to hit first (worst case).
  * One position per symbol at a time. Portfolio concurrency, the queue and
    position rotation are NOT modeled - they affect capital allocation, not the
    per-trade edge this measures.
  * Trading fees and a simple funding estimate are modeled.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from screener.binance_client import BinanceClient, BinanceClientError
from screener.breakout import (
    BreakoutSettings,
    BreakoutSignal,
    Candle,
    candles_from_klines,
    detect_long_breakout,
    detect_short_breakdown,
    evaluate_breakout,
    interval_to_ms,
)
from screener.cli import (
    _classify_entry_regime,
    _dead_coin_reason,
    _dynamic_leverage,
    _momentum_score,
)
from screener.cli import ROTATION_MIN_EDGE as _ROTATION_MIN_EDGE
from screener.cli import ROTATION_FEE_RATE as _ROTATION_FEE_RATE
from screener.cli import ROTATION_MIN_HOLD_SECONDS as _ROTATION_MIN_HOLD_SECONDS
from screener.cli import ROTATION_COOLDOWN_SECONDS as _ROTATION_COOLDOWN_SECONDS
from screener.take_profit import equal_take_profit_profile, smart_take_profit_profile

# Stop buffer below recent support (above resistance for shorts) for the dynamic SL,
# matching the live bot's stop_buffer_pct default.
_DYN_SL_BUFFER = 0.002


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    status: str
    regime: str
    detected_time: int
    entry_time: int
    exit_time: int
    entry_price: float
    stop_price: float
    target_price: float
    avg_exit_price: float
    leverage: int
    hold_candles: int
    hold_hours: float
    r_multiple: float
    price_return: float  # signed fractional P/L on notional
    # Filled by the portfolio simulation:
    taken: bool = False
    margin: float = 0.0
    net_pnl: float = 0.0
    fees_usdt: float = 0.0
    funding_usdt: float = 0.0
    position_pct: float = 0.0
    ml_score: float = 0.0
    ml_p_win: float = 0.0
    ml_expected_r: float = 0.0
    ml_tail_prob: float = 0.0
    ml_not_bad_prob: float = 0.0
    ml_market_regime: str = ""
    tp_runner_pct: float = 0.0
    tp_target_multiplier: float = 1.0
    tp_conviction: float = 0.0
    momentum_score: float = 0.0  # used by the rotation portfolio sim
    # Feature snapshot at signal-detection time, for offline ML training.
    # Populated by _simulate_trade from the BreakoutSignal; never read by the
    # backtester itself - only written to the CSV trade log for training.
    feat_score: float = 0.0
    feat_breakout_pct: float = 0.0
    feat_distance_to_trigger_pct: float = 0.0
    feat_risk_pct: float = 0.0
    feat_reward_pct: float = 0.0
    feat_reward_risk: float = 0.0
    feat_volume_ratio: float = 0.0
    feat_avg_quote_volume: float = 0.0
    feat_compression_pct: float = 0.0
    feat_atr_pct: float = 0.0
    feat_trend_score: float = 0.0
    feat_close_position: float = 0.0
    feat_quote_volume_24h: float = 0.0
    feat_range_pct_24h: float = 0.0
    feat_price_change_pct_24h: float = 0.0
    feat_btc_momentum_pct: float = 0.0
    feat_btc_ema_distance_pct: float = 0.0
    feat_rel_momentum_pct: float = 0.0
    feat_symbol_trades_30: float = 0.0
    feat_symbol_win_rate_30: float = 0.5
    feat_symbol_avg_r_30: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


@dataclass(frozen=True)
class MarketTrendPoint:
    close_time: int
    close: float
    ema: float
    momentum_pct: float


@dataclass(frozen=True)
class MarketTrend:
    points: list[MarketTrendPoint]
    times: list[int]

    def at_or_before(self, timestamp: int) -> MarketTrendPoint | None:
        index = bisect_right(self.times, timestamp) - 1
        if index < 0:
            return None
        return self.points[index]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rate_limit_rpm = args.rate_limit_rpm if args.rate_limit_rpm > 0 else None
    client = BinanceClient(
        market="futures",
        timeout=args.timeout,
        retries=args.retries,
        rate_limit_rpm=rate_limit_rpm,
    )
    try:
        return _run(client, args)
    except BinanceClientError as exc:
        print(f"Backtest failed: {exc}")
        return 2


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the breakout strategy.")
    parser.add_argument("--quote", default="USDT", help="Quote asset for the symbol universe.")
    parser.add_argument("--symbols", help="Explicit comma-separated symbols. Overrides --top.")
    parser.add_argument("--top", type=int, default=40, help="Backtest the top N symbols by 24h range. 0 = scan every coin.")
    parser.add_argument("--min-quote-volume", type=float, default=20_000_000.0, help="Liquidity floor: skip symbols below this 24h quote volume.")
    parser.add_argument("--detector", choices=["simple", "squeeze"], default="simple", help="simple = high-recall long-breakout detector; squeeze = the original pre-breakout detector.")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 15m, 1h, 4h.")
    parser.add_argument("--history", type=int, default=1500, help="Klines to pull per symbol (max 1500).")
    parser.add_argument("--start-date", help="UTC start date for the backtest window, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="UTC end date for the backtest window, YYYY-MM-DD.")
    parser.add_argument("--trade-log", default="backtest_trades.csv", help="CSV path for the per-signal trade log.")
    parser.add_argument("--order-margin", type=float, default=5.0, help="USDT margin committed per trade.")
    parser.add_argument("--leverage", type=int, default=10, help="Base leverage.")
    parser.add_argument("--dynamic-leverage", action="store_true", help="Scale leverage per coin from ATR.")
    parser.add_argument("--max-sl-loss-pct", type=float, default=35.0, help="Maximum leveraged loss percent on margin if the stop is hit. 0 disables stop tightening.")
    parser.add_argument("--dynamic-sl", action="store_true", help="Trail the stop up to recent support (down to resistance for shorts) as price moves - ports the live bot's dynamic stop loss.")
    parser.add_argument("--sl-lookback", type=int, default=20, help="Candles of recent support/resistance used to trail the dynamic stop.")
    parser.add_argument("--breakeven-trigger-r", type=float, default=1.5, help="Once unrealized profit reaches this R multiple, ratchet the stop to entry + a small profit lock. 0 = off. Default 1.5 was the worst-case-best in a 12-run sweep.")
    parser.add_argument("--breakeven-offset-pct", type=float, default=0.1, help="When the breakeven trigger fires, the new stop sits this percent past entry (locks tiny profit + covers fees).")
    parser.add_argument("--exhaustion-exit", action="store_true", help="Smart exit: once the trade has been at +0.5R profit, close on a bearish-rejection candle near the peak OR after 4 candles with no new high. Tested net-negative on a 6-run sweep - off by default.")
    parser.add_argument("--profit-lock-ladder", type=str, default="", help="Profit-lock ladder, comma-separated 'trigger:lock' pairs (R multiples). e.g. '1.5:0,3:1,5:2.5' = lock breakeven at +1.5R, +1R profit at +3R, +2.5R profit at +5R. Empty = single breakeven rung from --breakeven-trigger-r.")
    parser.add_argument("--stagnation-after-r", type=float, default=0.0, help="Stagnation exit: only active once the trade reaches this R multiple. 0 = off. Pairs with --stagnation-candles.")
    parser.add_argument("--stagnation-candles", type=int, default=8, help="Stagnation exit: candles of no new favorable extreme before closing. Only fires after --stagnation-after-r has been reached.")
    parser.add_argument("--no-overhead-guard", dest="overhead_guard", action="store_false", help="Disable skipping breakouts that fire straight into a long moving average.")
    parser.set_defaults(overhead_guard=True)
    parser.add_argument("--min-breakout-volume-ratio", type=float, default=2.5, help="Explosiveness floor: minimum breakout-candle volume vs the recent average (simple detector).")
    parser.add_argument("--min-breakout-atr-pct", type=float, default=1.0, help="Explosiveness floor: minimum ATR percent for a breakout to qualify (simple detector).")
    parser.add_argument("--breakout-min-candle-range-mult", type=float, default=1.5, help="Detection upgrade: require the breakout candle range >= this multiple of ATR. 0 = off.")
    parser.add_argument("--breakout-max-extension-pct", type=float, default=5.0, help="Detection upgrade: reject breakouts whose close is already more than this percent above the broken level. 0 = off.")
    parser.add_argument("--breakout-max-base-range-pct", type=float, default=0.0, help="Detection upgrade: reject breakouts from a base wider than this percent (tight-coil filter). 0 = off.")
    parser.add_argument("--min-rel-strength-pct", type=float, default=None, help="Detection upgrade: require the coin's momentum to beat BTC's by this percent. Unset = off.")
    parser.add_argument("--no-tp-cap-below-ma", dest="tp_cap_below_ma", action="store_false", help="Disable capping the take-profit just below an overhead MA99 wall.")
    parser.set_defaults(tp_cap_below_ma=True)
    parser.add_argument("--breakout-volume-trend-ratio", type=float, default=0.0, help="Detection upgrade: require base up-candle volume >= ratio x down-candle volume (LONG; mirrored for SHORT). 0 = off. 1.0 = simple net-accumulation, 1.5 = stronger demand bias.")
    parser.add_argument("--mtf-alignment-tf", type=str, default=None, help="Detection upgrade: require entry signals to align with a higher-timeframe trend (e.g. '4h'). Long signals only fire when the higher-TF close is above its MA; short signals require below. Unset = off.")
    parser.add_argument("--mtf-alignment-ma-period", type=int, default=25, help="Moving-average period on the higher TF used for --mtf-alignment-tf. Default 25.")
    parser.add_argument("--mtf-alignment-ma-type", choices=["sma", "ema"], default="sma", help="Moving-average type for the MTF alignment check. EMA reacts faster to recent price; SMA is smoother.")
    parser.add_argument("--simulate-rotation", action="store_true", help="Use the rotation-aware portfolio sim: when all slots are full and a higher-momentum INSTANT signal arrives, close a weaker profitable position to make room. Mirrors the live rotation logic (90-min min hold, 30-min cooldown, ROTATION_MIN_EDGE).")
    parser.add_argument("--reserve-last-slot-s-tier", action="store_true", help="When max-concurrent >= 2, the final slot is reserved for S-tier signals (momentum_score >= --s-tier-momentum-threshold). Other slots open to any qualifying signal.")
    parser.add_argument("--s-tier-momentum-threshold", type=float, default=0.85, help="Minimum momentum_score for a signal to qualify as S-tier. Default 0.85 = ~top 15-20%% of historical signals.")
    parser.add_argument("--target-intensity-max", type=float, default=4.0, help="Ceiling on the ATR intensity multiplier used to project the target (target = trigger + risk x 1.5 x intensity). 4.0 = current default; 2.0 caps high-ATR coins at 3R instead of 6R.")
    parser.add_argument("--tp-cap-recent-swing-high-candles", type=int, default=200, help="Cap the take-profit just past the last N candles' swing high (LONG) / swing low (SHORT), when the prior resistance is meaningfully above the trigger. 0 = off. Default 200 won a 12-run sweep (compounded 117x -> 203x).")
    parser.add_argument("--ml-filter-model", type=str, default=None, help="Path to a model.json (from train_model.py); skip signals with predicted P(WIN) below --ml-filter-threshold. Unset = off.")
    parser.add_argument("--ml-filter-threshold", type=float, default=0.5, help="Minimum predicted P(WIN) for a signal to be taken when --ml-filter-model is set. Default 0.5.")
    parser.add_argument("--ml-filter-score", choices=["pwin", "expected-r", "tail", "not-bad", "composite"], default="pwin", help="Experimental backtest-only ML score to threshold/rank. Higher is always better.")
    parser.add_argument("--ml-score-only", action="store_true", help="Load/write ML scores without applying the threshold filter. Use with --ml-rank-signals for ranking-only tests.")
    parser.add_argument("--ml-rank-signals", action="store_true", help="Experimental backtest-only: rank simultaneous entry candidates by the selected ML score before portfolio slotting.")
    parser.add_argument("--ml-regime-aware", action="store_true", help="Experimental backtest-only: use BTC-regime-specific ML heads when present in the artifact.")
    parser.add_argument("--ml-filter-start-date", help="Backtest-only UTC date, YYYY-MM-DD: load ML scores but apply threshold/ranking only on or after this date.")
    parser.add_argument("--tp-count", type=int, default=1, help="Number of partial take-profits.")
    parser.add_argument("--trailing-stop", action="store_true", help="Model a trailing-stop runner.")
    parser.add_argument("--trailing-callback-pct", type=float, default=1.2, help="Trailing-stop callback percent.")
    parser.add_argument("--runner-pct", type=float, default=50.0, help="Percent of position left for the trailing runner when --trailing-stop is used.")
    parser.add_argument("--smart-tp", action="store_true", help="Adapt the target, TP splits, and runner size from each signal's conviction.")
    parser.add_argument("--smart-tp-max-target-multiplier", type=float, default=1.0, help="Extra multiplier on the detector target for top-conviction signals. The detector target is already ATR-scaled, so keep this near 1.0; high values push the target out of reach and clog slots.")
    parser.add_argument("--smart-tp-min-runner-pct", type=float, default=20.0, help="Minimum trailing runner percent used by --smart-tp.")
    parser.add_argument("--smart-tp-max-runner-pct", type=float, default=55.0, help="Maximum trailing runner percent used by --smart-tp.")
    parser.add_argument("--target-range-multiple", type=float, default=1.5, help="Target as a multiple of the range.")
    parser.add_argument("--min-rr", type=float, default=1.2, help="Minimum reward/risk to take a signal.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum setup score to take a signal.")
    parser.add_argument("--trigger-wait-candles", type=int, default=8, help="Candles to wait for the breakout trigger.")
    parser.add_argument("--entry-on-signal-close", action="store_true", help="For confirmed breakouts, enter on the signal close instead of waiting for the next candle trigger.")
    parser.add_argument("--entry-style", choices=["trigger", "retest", "retest-confirmed"], default="trigger", help="trigger = enter when price crosses the breakout trigger; retest = limit fill at the broken level on a pullback; retest-confirmed = enter only after a candle closes back through the level, abort if a candle closes the wrong side.")
    parser.add_argument("--slippage-pct", type=float, default=0.03, help="Entry slippage percent.")
    parser.add_argument("--fee-pct", type=float, default=0.045, help="Taker fee percent, charged per side.")
    parser.add_argument("--funding-pct-per-8h", type=float, default=0.01, help="Assumed funding rate percent per 8h. Longs pay it, shorts receive it. Set 0 to disable.")
    parser.add_argument("--longs-only", dest="longs_only", action="store_true", default=True, help="Skip SHORT signals entirely.")
    parser.add_argument("--include-shorts", dest="longs_only", action="store_false", help="Also test SHORT breakdown signals.")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting account balance.")
    parser.add_argument("--compound", action="store_true", help="Scale each position's margin with the running equity (compounding).")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max positions open at once. 0 = dynamic cap by current equity.")
    parser.add_argument("--position-pct", type=float, default=0.0, help="Fixed margin percent of equity when --compound is set. 0 = use --sizing-mode.")
    parser.add_argument("--sizing-mode", choices=["guarded", "moonshot", "aggressive", "auto"], default="moonshot", help="guarded = fixed 10%% sizing; moonshot/aggressive = dynamic tiers vs starting-capital multiple; auto = absolute-equity tiers + drawdown haircut (designed to grow micro accounts fast while de-risking on drawdown).")
    parser.add_argument("--skip-entry-regimes", default="TRAILING_RETEST", help="Comma-separated adaptive entry regimes to exclude. Default: TRAILING_RETEST; pass none to include all.")
    parser.add_argument("--btc-trend-filter", action="store_true", help="Only take longs when BTC's rolling trend is not hostile; invert the rule for shorts.")
    parser.add_argument("--btc-ema-candles", type=int, default=72, help="EMA length for --btc-trend-filter.")
    parser.add_argument("--btc-momentum-candles", type=int, default=72, help="Momentum lookback for BTC trend and instant-entry guards.")
    parser.add_argument("--btc-ema-slack-pct", type=float, default=0.5, help="Allowed BTC distance below EMA for long signals, percent.")
    parser.add_argument("--btc-momentum-guard-pct", type=float, default=1.0, help="Allowed BTC adverse momentum for trend-filtered signals, percent.")
    parser.add_argument("--no-instant-market-guard", dest="instant_market_guard", action="store_false", help="Do not block INSTANT entries during hostile BTC trend.")
    parser.set_defaults(instant_market_guard=True)
    parser.add_argument("--instant-guard-momentum-pct", type=float, default=-2.0, help="Skip INSTANT entries when BTC momentum over --btc-momentum-candles is below this percent.")
    parser.add_argument("--instant-guard-ema-slack-pct", type=float, default=1.5, help="Skip INSTANT entries when BTC is this far below its EMA, percent.")
    parser.add_argument("--no-hostile-market-strict-only", dest="hostile_market_strict_only", action="store_false", help="Allow all non-skipped regimes even when BTC trend is hostile.")
    parser.set_defaults(hostile_market_strict_only=True)
    parser.add_argument("--hostile-momentum-pct", type=float, default=0.0, help="BTC momentum below this percent is hostile for regime selection.")
    parser.add_argument("--hostile-ema-slack-pct", type=float, default=0.0, help="BTC close this far below EMA is hostile for regime selection, percent.")
    parser.add_argument("--no-short-market-guard", dest="short_market_guard", action="store_false", help="Allow shorts even when BTC is not bearish.")
    parser.set_defaults(short_market_guard=True)
    parser.add_argument("--short-guard-momentum-pct", type=float, default=0.0, help="Allow SHORT signals only when BTC momentum is at or below this percent.")
    parser.add_argument("--short-guard-ema-slack-pct", type=float, default=0.0, help="Allow SHORT signals only when BTC is no more than this percent above EMA.")
    parser.add_argument("--loss-cooldown-after", type=int, default=1, help="With --loss-cooldown-candles, pause a symbol after this many consecutive realized losses on that symbol.")
    parser.add_argument("--loss-cooldown-candles", type=int, default=48, help="Pause new entries on the losing symbol for N candles after the loss threshold is hit. 0 disables.")
    parser.add_argument("--instant-size-multiplier", type=float, default=0.5, help="Moonshot sizing multiplier for INSTANT regimes.")
    parser.add_argument("--retest-size-multiplier", type=float, default=0.9, help="Moonshot sizing multiplier for RETEST regimes.")
    parser.add_argument("--trailing-retest-size-multiplier", type=float, default=0.5, help="Moonshot sizing multiplier for TRAILING_RETEST regimes.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry count.")
    parser.add_argument(
        "--rate-limit-rpm",
        type=float,
        default=1200.0,
        help="Max Binance requests per minute during backtests. 0 disables.",
    )
    args = parser.parse_args(argv)
    if args.tp_count <= 0:
        parser.error("--tp-count must be greater than 0")
    if not 0 <= args.runner_pct < 100:
        parser.error("--runner-pct must be from 0 to less than 100")
    if args.smart_tp_max_target_multiplier < 1:
        parser.error("--smart-tp-max-target-multiplier must be at least 1")
    if not 0 <= args.smart_tp_min_runner_pct < 100 or not 0 <= args.smart_tp_max_runner_pct < 100:
        parser.error("--smart-tp runner bounds must be from 0 to less than 100")
    if args.smart_tp_min_runner_pct > args.smart_tp_max_runner_pct:
        parser.error("--smart-tp-min-runner-pct must be <= --smart-tp-max-runner-pct")
    if args.history <= 100:
        parser.error("--history must be greater than 100")
    if args.max_sl_loss_pct < 0:
        parser.error("--max-sl-loss-pct must be zero or greater")
    if args.sl_lookback <= 1:
        parser.error("--sl-lookback must be greater than 1")
    if args.max_concurrent < 0:
        parser.error("--max-concurrent must be zero or greater")
    try:
        args.start_ms = _parse_date(args.start_date, end=False) if args.start_date else None
        args.end_ms = _parse_date(args.end_date, end=True) if args.end_date else None
        args.ml_filter_start_ms = _parse_date(args.ml_filter_start_date, end=False) if args.ml_filter_start_date else None
    except ValueError as exc:
        parser.error(str(exc))
    if args.start_ms is not None and args.end_ms is not None and args.start_ms > args.end_ms:
        parser.error("--start-date must be on or before --end-date")
    if args.btc_ema_candles <= 1:
        parser.error("--btc-ema-candles must be greater than 1")
    if args.btc_momentum_candles <= 0:
        parser.error("--btc-momentum-candles must be greater than 0")
    if args.loss_cooldown_after <= 0:
        parser.error("--loss-cooldown-after must be greater than 0")
    if args.loss_cooldown_candles < 0:
        parser.error("--loss-cooldown-candles must be zero or greater")
    if args.rate_limit_rpm < 0:
        parser.error("--rate-limit-rpm must be zero or greater")
    if args.instant_size_multiplier < 0 or args.retest_size_multiplier < 0 or args.trailing_retest_size_multiplier < 0:
        parser.error("size multipliers must be zero or greater")
    args.skip_entry_regimes = _parse_regime_set(args.skip_entry_regimes)
    args.profit_lock_pairs = _parse_profit_lock_ladder(args.profit_lock_ladder, parser)
    if args.stagnation_after_r < 0:
        parser.error("--stagnation-after-r must be zero or greater")
    if args.stagnation_candles < 1:
        parser.error("--stagnation-candles must be at least 1")
    return args


def _parse_date(raw: str, end: bool = False) -> int:
    try:
        day = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid date {raw!r}; expected YYYY-MM-DD") from exc
    if end:
        day = day + timedelta(days=1) - timedelta(milliseconds=1)
    return int(day.timestamp() * 1000)


def _parse_profit_lock_ladder(raw: str, parser: argparse.ArgumentParser) -> list[tuple[float, float]]:
    """Parse the --profit-lock-ladder string into sorted (trigger_R, lock_R) pairs."""
    if not raw or not raw.strip():
        return []
    pairs: list[tuple[float, float]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            parser.error(f"--profit-lock-ladder: invalid pair {chunk!r}; expected 'trigger:lock'")
        t_str, l_str = chunk.split(":", 1)
        try:
            trigger = float(t_str)
            lock = float(l_str)
        except ValueError:
            parser.error(f"--profit-lock-ladder: non-numeric pair {chunk!r}")
        if trigger <= 0 or lock < 0:
            parser.error(f"--profit-lock-ladder: trigger must be > 0 and lock >= 0 (got {chunk!r})")
        pairs.append((trigger, lock))
    return sorted(pairs, key=lambda p: p[0])


def _parse_regime_set(raw: str) -> set[str]:
    if raw.strip().lower() in {"none", "off"}:
        return set()
    return {part.strip().upper() for part in raw.split(",") if part.strip()}


def _run(client: BinanceClient, args: argparse.Namespace) -> int:
    if args.position_pct <= 0 and args.sizing_mode == "guarded":
        args.position_pct = _auto_position_pct(args.capital)
    settings = BreakoutSettings(
        target_range_multiple=args.target_range_multiple,
        overhead_ma_guard=args.overhead_guard,
        min_breakout_volume_ratio=args.min_breakout_volume_ratio,
        min_breakout_atr_pct=args.min_breakout_atr_pct / 100.0,
        breakout_min_candle_range_mult=args.breakout_min_candle_range_mult,
        breakout_max_extension_pct=args.breakout_max_extension_pct / 100.0,
        breakout_max_base_range_pct=args.breakout_max_base_range_pct / 100.0,
        tp_cap_below_ma=args.tp_cap_below_ma,
        breakout_min_up_down_volume_ratio=args.breakout_volume_trend_ratio,
        target_intensity_max=args.target_intensity_max,
        tp_cap_recent_swing_high_candles=args.tp_cap_recent_swing_high_candles,
    )
    interval_ms = interval_to_ms(args.interval)
    symbols = _resolve_symbols(client, args)
    if not symbols:
        print("No symbols resolved for the backtest.")
        return 1
    ml_model: dict | None = None
    if args.ml_filter_model:
        try:
            ml_model = json.loads(Path(args.ml_filter_model).read_text(encoding="utf-8"))
            test_auc = ml_model.get("metrics_lr", {}).get("test_auc", float("nan"))
            mode = "score-only" if args.ml_score_only else f"threshold={args.ml_filter_threshold:.3f}"
            rank = ", ranking" if args.ml_rank_signals else ""
            regime = ", regime-aware" if args.ml_regime_aware else ""
            print(
                f"ML filter: loaded {args.ml_filter_model} (LR test AUC {test_auc:.3f})  "
                f"score={args.ml_filter_score} {mode}{rank}{regime}"
            )
        except (OSError, ValueError, KeyError) as exc:
            print(f"ML filter disabled - could not load {args.ml_filter_model}: {exc}")
            ml_model = None
    btc_return, window_start, window_end = _market_context(client, args)
    market_trend = (
        _load_market_trend(client, args, interval_ms)
        if args.btc_trend_filter
        or args.instant_market_guard
        or args.hostile_market_strict_only
        or args.short_market_guard
        or args.min_rel_strength_pct is not None
        else None
    )
    window_size = "full date window" if args.start_ms is not None else f"{args.history} candles"
    print(f"Backtesting {len(symbols)} symbol(s) on {args.interval}, {window_size} each...")
    if args.start_ms is not None:
        print("Date-window run: scanning every eligible coin to avoid today's volatility-ranking bias.")
    if args.skip_entry_regimes:
        print(f"Entry-regime filter: skipping {', '.join(sorted(args.skip_entry_regimes))}.")
    if args.btc_trend_filter:
        print(
            "BTC trend filter: "
            f"EMA {args.btc_ema_candles}, momentum {args.btc_momentum_candles}, "
            f"slack {args.btc_ema_slack_pct:.2f}%, guard {args.btc_momentum_guard_pct:.2f}%."
        )
    if args.instant_market_guard:
        print(
            "Instant guard: "
            f"BTC momentum >= {args.instant_guard_momentum_pct:.2f}% and "
            f"close no more than {args.instant_guard_ema_slack_pct:.2f}% below EMA."
        )
    if args.hostile_market_strict_only:
        print(
            "Hostile market guard: only STRICT_RETEST when "
            f"BTC momentum < {args.hostile_momentum_pct:.2f}% or close is below EMA by "
            f"{args.hostile_ema_slack_pct:.2f}%."
        )
    if args.short_market_guard:
        print(
            "Short guard: BTC momentum <= "
            f"{args.short_guard_momentum_pct:.2f}% and close no more than "
            f"{args.short_guard_ema_slack_pct:.2f}% above EMA."
        )

    trades: list[BacktestTrade] = []
    candles_by_symbol: dict[str, list[Candle]] = {}
    for index, symbol in enumerate(symbols, start=1):
        try:
            raw = _fetch_klines(client, symbol, args, interval_ms)
        except BinanceClientError as exc:
            print(f"  [{index}/{len(symbols)}] {symbol}: klines failed ({exc})")
            continue
        candles = candles_from_klines(raw)
        if not _window_is_liquid(candles, args, interval_ms):
            print(f"  [{index}/{len(symbols)}] {symbol}: skipped (illiquid in window)")
            continue
        mtf_align: MtfAlignment | None = None
        if args.mtf_alignment_tf:
            try:
                mtf_raw = _fetch_klines_at(
                    client, symbol, args.mtf_alignment_tf,
                    interval_to_ms(args.mtf_alignment_tf), args.start_ms, args.end_ms,
                )
            except BinanceClientError:
                mtf_raw = []
            if mtf_raw:
                mtf_align = _build_mtf_alignment(
                    candles_from_klines(mtf_raw),
                    args.mtf_alignment_ma_period,
                    args.mtf_alignment_ma_type,
                )
        symbol_trades = _backtest_symbol(symbol, candles, args, settings, args.start_ms, args.end_ms, market_trend, mtf_align, ml_model)
        trades.extend(symbol_trades)
        if args.simulate_rotation and symbol_trades:
            candles_by_symbol[symbol] = candles
        print(f"  [{index}/{len(symbols)}] {symbol}: {len(symbol_trades)} trade(s)")

    if args.simulate_rotation:
        taken, final_equity, max_drawdown = _simulate_portfolio_with_rotation(trades, candles_by_symbol, args)
    else:
        taken, final_equity, max_drawdown = _simulate_portfolio(trades, args)
    trade_log = _write_trade_log(trades, path=args.trade_log, args=args)
    print(f"Trade log written: {trade_log} ({len(trades)} row(s))")
    _print_report(trades, taken, final_equity, max_drawdown, args, window_start, window_end, btc_return)
    return 0


def _auto_position_pct(capital: float) -> float:
    # Validation runs showed 20-50% compounding can dominate signal quality with
    # path risk; keep the guarded default at 10% unless the caller overrides it.
    return 10.0


def _kline_cache_path(symbol: str, interval: str, start_time: int, end_time: int | None, limit: int | str) -> Path | None:
    cache_dir = Path(".backtest_kline_cache")
    try:
        cache_dir.mkdir(exist_ok=True)
    except OSError:
        return None
    return cache_dir / f"{symbol}_{interval}_{start_time}_{end_time}_{limit}.json"


def _date_window_cache_limit(start_time: int | None, end_time: int | None, interval_ms: int, limit: int) -> int | str:
    if start_time is None or end_time is None:
        return limit
    expected = max(int((end_time - start_time) / max(interval_ms, 1)) + 1, 0)
    return limit if expected <= limit else "all"


def _fetch_klines_window(
    client: BinanceClient,
    symbol: str,
    interval: str,
    interval_ms: int,
    limit: int,
    start_time: int | None,
    end_time: int | None,
) -> list[list[object]]:
    """Fetch a historical kline window, paging past Binance's per-call limit.

    Short windows keep the old cache key so existing 1500-candle caches are reused.
    Longer windows use an ``all`` cache key to avoid accidentally trusting an older
    partial date-window response.
    """
    if start_time is None:
        return client.klines(symbol, interval, limit, end_time=end_time)

    cache_token = _date_window_cache_limit(start_time, end_time, interval_ms, limit)
    cache_path = _kline_cache_path(symbol, interval, start_time, end_time, cache_token)
    if cache_path is not None and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    rows: list[list[object]] = []
    current_start = int(start_time)
    last_open_time = -1
    while True:
        batch = client.klines(symbol, interval, limit, start_time=current_start, end_time=end_time)
        if not batch:
            break
        new_rows = [row for row in batch if int(row[0]) > last_open_time]
        if not new_rows:
            break
        rows.extend(new_rows)
        last_open_time = int(new_rows[-1][0])
        next_start = last_open_time + interval_ms
        if len(batch) < limit or (end_time is not None and next_start > end_time):
            break
        if next_start <= current_start:
            break
        current_start = next_start

    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps(rows), encoding="utf-8")
        except OSError:
            pass
    return rows


def _fetch_klines_at(
    client: BinanceClient,
    symbol: str,
    interval: str,
    interval_ms: int,
    start_ms: int | None,
    end_ms: int | None,
    history: int = 1500,
) -> list[list[object]]:
    """Fetch klines at an arbitrary interval (caches like _fetch_klines).
    Used for the MTF-alignment higher-timeframe lookup."""
    limit = min(history, 1500)
    start_time = start_ms
    end_time = end_ms
    if start_time is not None:
        start_time = max(int(start_time) - 150 * interval_ms, 0)
        limit = 1500
    return _fetch_klines_window(client, symbol, interval, interval_ms, limit, start_time, end_time)


@dataclass(frozen=True)
class MtfAlignment:
    """Higher-timeframe candle series with a trailing MA, for entry alignment lookups."""
    times: list[int]
    closes: list[float]
    ma: list[float]
    period: int


def _build_mtf_alignment(candles: list[Candle], ma_period: int, ma_type: str = "sma") -> MtfAlignment | None:
    if len(candles) < ma_period or ma_period <= 1:
        return None
    times = [c.close_time for c in candles]
    closes = [c.close for c in candles]
    ma: list[float] = [0.0] * len(candles)
    if ma_type == "ema":
        alpha = 2.0 / (ma_period + 1.0)
        ema_val = closes[0]
        for i, close in enumerate(closes):
            ema_val = close * alpha + ema_val * (1.0 - alpha)
            # Match SMA's warmup behaviour: first (period-1) bars stay 0 so the
            # alignment check returns True (insufficient history) early on.
            if i >= ma_period - 1:
                ma[i] = ema_val
    else:  # sma
        s = 0.0
        for i, close in enumerate(closes):
            s += close
            if i >= ma_period:
                s -= closes[i - ma_period]
            if i >= ma_period - 1:
                ma[i] = s / ma_period
    return MtfAlignment(times=times, closes=closes, ma=ma, period=ma_period)


def _mtf_aligned(mtf: MtfAlignment | None, signal_close_time: int, side: str) -> bool:
    """Return True if the higher-TF trend agrees with the signal side (or MTF off)."""
    if mtf is None:
        return True
    idx = bisect_right(mtf.times, signal_close_time) - 1
    if idx < mtf.period - 1:
        return True  # not enough higher-TF history yet - do not block
    ma_val = mtf.ma[idx]
    close_val = mtf.closes[idx]
    if ma_val <= 0:
        return True
    if side == "LONG":
        return close_val >= ma_val
    return close_val <= ma_val


def _fetch_klines(client: BinanceClient, symbol: str, args: argparse.Namespace, interval_ms: int) -> list[list[object]]:
    limit = min(args.history, 1500)
    start_time = args.start_ms
    end_time = args.end_ms
    if start_time is not None:
        start_time = max(int(start_time) - 150 * interval_ms, 0)
        limit = 1500
    return _fetch_klines_window(client, symbol, args.interval, interval_ms, limit, start_time, end_time)


def _load_market_trend(client: BinanceClient, args: argparse.Namespace, interval_ms: int) -> MarketTrend | None:
    try:
        raw = _fetch_klines(client, "BTCUSDT", args, interval_ms)
    except BinanceClientError as exc:
        print(f"BTC trend filter disabled: BTCUSDT klines failed ({exc})")
        return None
    candles = candles_from_klines(raw)
    trend = _build_market_trend(candles, args.btc_ema_candles, args.btc_momentum_candles)
    if not trend.points:
        print("BTC trend filter disabled: not enough BTCUSDT candles.")
        return None
    return trend


def _build_market_trend(candles: list[Candle], ema_candles: int, momentum_candles: int) -> MarketTrend:
    if not candles or ema_candles <= 1 or momentum_candles <= 0:
        return MarketTrend(points=[], times=[])
    alpha = 2.0 / (ema_candles + 1.0)
    ema = candles[0].close
    points: list[MarketTrendPoint] = []
    for index, candle in enumerate(candles):
        ema = candle.close * alpha + ema * (1.0 - alpha)
        if index < momentum_candles:
            continue
        previous = candles[index - momentum_candles].close
        momentum_pct = (candle.close / max(previous, 1e-9) - 1.0) * 100.0
        points.append(
            MarketTrendPoint(
                close_time=candle.close_time,
                close=candle.close,
                ema=ema,
                momentum_pct=momentum_pct,
            )
        )
    return MarketTrend(points=points, times=[point.close_time for point in points])


def _market_context(client: BinanceClient, args: argparse.Namespace) -> tuple[float | None, int, int]:
    """Return BTC's return over the window and the window's start/end timestamps."""
    try:
        raw = _fetch_klines(client, "BTCUSDT", args, interval_to_ms(args.interval))
        candles = candles_from_klines(raw)
    except BinanceClientError:
        return None, 0, 0
    candles = _candles_in_window(candles, args.start_ms, args.end_ms)
    if not candles:
        return None, 0, 0
    btc_return = (candles[-1].close / max(candles[0].open, 1e-9) - 1.0) * 100.0
    return btc_return, candles[0].open_time, candles[-1].close_time


def _window_is_liquid(candles: list[Candle], args: argparse.Namespace, interval_ms: int) -> bool:
    """Non-look-ahead liquidity gate.

    The bias fix made date-window runs scan every perpetual (no today's-ticker
    filter). That is correct for avoiding look-ahead, but it floods the universe
    with illiquid micro-caps. This gate keeps the bias fix and restores sanity:
    a symbol's MEDIAN per-candle turnover within the backtest window, annualised
    to 24h, must clear --min-quote-volume. It uses only in-window data.
    """
    if args.min_quote_volume <= 0:
        return True
    in_window = _candles_in_window(candles, args.start_ms, args.end_ms)
    candles_per_24h = max(int(86_400_000 / interval_ms), 1)
    if len(in_window) < candles_per_24h:
        return False
    volumes = sorted(candle.quote_volume for candle in in_window)
    median_qv = volumes[len(volumes) // 2]
    return median_qv * candles_per_24h >= args.min_quote_volume


def _candles_in_window(candles: list[Candle], start_ms: int | None, end_ms: int | None) -> list[Candle]:
    return [
        candle
        for candle in candles
        if (start_ms is None or candle.close_time >= start_ms)
        and (end_ms is None or candle.open_time <= end_ms)
    ]


def _market_allows_signal(
    signal: BreakoutSignal,
    timestamp: int,
    market_trend: MarketTrend | None,
    args: argparse.Namespace,
) -> bool:
    if market_trend is None:
        return True
    point = market_trend.at_or_before(timestamp)
    if point is None:
        return False
    slack = max(args.btc_ema_slack_pct, 0.0) / 100.0
    guard = max(args.btc_momentum_guard_pct, 0.0)
    if signal.side == "SHORT":
        return point.close <= point.ema * (1.0 + slack) and point.momentum_pct <= guard
    return point.close >= point.ema * (1.0 - slack) and point.momentum_pct >= -guard


def _instant_market_allows(
    regime: str,
    timestamp: int,
    market_trend: MarketTrend | None,
    args: argparse.Namespace,
) -> bool:
    if regime != "INSTANT" or not args.instant_market_guard:
        return True
    if market_trend is None:
        return False
    point = market_trend.at_or_before(timestamp)
    if point is None:
        return False
    ema_floor = point.ema * (1.0 - max(args.instant_guard_ema_slack_pct, 0.0) / 100.0)
    return point.momentum_pct >= args.instant_guard_momentum_pct and point.close >= ema_floor


def _regime_allowed_in_market(
    regime: str,
    timestamp: int,
    market_trend: MarketTrend | None,
    args: argparse.Namespace,
) -> bool:
    if not args.hostile_market_strict_only:
        return True
    if market_trend is None:
        return regime == "STRICT_RETEST"
    point = market_trend.at_or_before(timestamp)
    if point is None:
        return regime == "STRICT_RETEST"
    ema_floor = point.ema * (1.0 - max(args.hostile_ema_slack_pct, 0.0) / 100.0)
    hostile = point.momentum_pct < args.hostile_momentum_pct or point.close < ema_floor
    return not hostile or regime == "STRICT_RETEST"


def _side_allowed_in_market(
    side: str,
    timestamp: int,
    market_trend: MarketTrend | None,
    args: argparse.Namespace,
) -> bool:
    if side != "SHORT" or not args.short_market_guard:
        return True
    if market_trend is None:
        return False
    point = market_trend.at_or_before(timestamp)
    if point is None:
        return False
    ema_ceiling = point.ema * (1.0 + max(args.short_guard_ema_slack_pct, 0.0) / 100.0)
    return point.momentum_pct <= args.short_guard_momentum_pct and point.close <= ema_ceiling


def _ml_feature_map(
    signal: BreakoutSignal,
    regime: str,
    context_features: dict[str, float] | None = None,
) -> dict[str, float]:
    feature_map: dict[str, float] = {
        "feat_score": signal.score,
        "feat_breakout_pct": signal.breakout_pct,
        "feat_distance_to_trigger_pct": signal.distance_to_trigger_pct,
        "feat_risk_pct": signal.risk_pct,
        "feat_reward_pct": signal.reward_pct,
        "feat_reward_risk": signal.reward_risk,
        "feat_volume_ratio": signal.volume_ratio,
        "feat_avg_quote_volume": signal.avg_quote_volume,
        "feat_compression_pct": signal.compression_pct,
        "feat_atr_pct": signal.atr_pct,
        "feat_trend_score": signal.trend_score,
        "feat_close_position": signal.close_position,
        "feat_quote_volume_24h": signal.quote_volume_24h,
        "feat_range_pct_24h": signal.range_pct_24h,
        "feat_price_change_pct_24h": signal.price_change_pct_24h,
        "feat_momentum_score": _momentum_score(signal),
    }
    if context_features:
        feature_map.update(context_features)
    return feature_map


def _ml_market_regime(context_features: dict[str, float] | None) -> str:
    context_features = context_features or {}
    btc_mom = context_features.get("feat_btc_momentum_pct", 0.0)
    btc_ema = context_features.get("feat_btc_ema_distance_pct", 0.0)
    if btc_mom >= 2.0 and btc_ema >= 0.0:
        return "UP"
    if btc_mom <= -2.0 and btc_ema <= 0.0:
        return "DOWN"
    return "FLAT"


def _ml_feature_vector(
    signal: BreakoutSignal,
    regime: str,
    model: dict,
    context_features: dict[str, float] | None = None,
) -> list[float]:
    feature_map = _ml_feature_map(signal, regime, context_features)
    features: list[float] = [feature_map.get(name, 0.0) for name in model["feature_names_numeric"]]
    for r in model["feature_names_regime_dummies"]:
        features.append(1.0 if regime == r else 0.0)
    return features


def _ml_apply_head(features: list[float], model: dict, head: dict | None) -> float | None:
    if not head:
        return None
    z = float(head["intercept"])
    coef = head["coef"]
    scaler_mean = model["scaler_mean"]
    scaler_scale = model["scaler_scale"]
    for i, x in enumerate(features):
        scale = scaler_scale[i] if scaler_scale[i] > 0 else 1.0
        z += (x - scaler_mean[i]) / scale * coef[i]
    if head.get("kind") == "logistic":
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))
    return z


def _ml_scores_signal(
    signal: BreakoutSignal,
    regime: str,
    model: dict | None,
    context_features: dict[str, float] | None = None,
    regime_aware: bool = False,
) -> dict[str, float | str] | None:
    """Pure-Python ML inference for backtest-only artifacts from train_model.py."""
    if model is None:
        return None
    features = _ml_feature_vector(signal, regime, model, context_features)
    heads = model.get("heads")
    if not heads:
        # Backward compatibility with linear-classifier-v1 artifacts.
        heads = {
            "pwin": {
                "kind": "logistic",
                "intercept": model["lr_intercept"],
                "coef": model["lr_coef"],
            }
        }
    market_regime = _ml_market_regime(context_features)
    if regime_aware:
        regime_heads = model.get("regime_heads", {}).get(market_regime)
        if regime_heads:
            override_heads = {
                name: head
                for name, head in {
                    "pwin": regime_heads.get("pwin"),
                    "expected_r": regime_heads.get("expected_r"),
                    "tail": regime_heads.get("tail"),
                    "bad": regime_heads.get("bad"),
                }.items()
                if head
            }
            heads = {**heads, **override_heads}
    pwin = _ml_apply_head(features, model, heads.get("pwin"))
    expected_r = _ml_apply_head(features, model, heads.get("expected_r"))
    tail = _ml_apply_head(features, model, heads.get("tail"))
    bad = _ml_apply_head(features, model, heads.get("bad"))
    not_bad = 1.0 - bad if bad is not None else None
    composite = None
    if expected_r is not None:
        composite = expected_r
        if tail is not None:
            composite += 0.35 * tail
        if bad is not None:
            composite -= 0.35 * bad
    return {
        "pwin": float(pwin) if pwin is not None else 0.0,
        "expected-r": float(expected_r) if expected_r is not None else 0.0,
        "tail": float(tail) if tail is not None else 0.0,
        "not-bad": float(not_bad) if not_bad is not None else 0.0,
        "composite": float(composite) if composite is not None else 0.0,
        "market_regime": market_regime,
    }


def _rel_strength_allows(
    signal: BreakoutSignal,
    window: list[Candle],
    args: argparse.Namespace,
    market_trend: MarketTrend | None,
) -> bool:
    """Detection upgrade: require the coin to be outpacing BTC by --min-rel-strength-pct.

    A breakout while the coin leads BTC is genuine independent strength; one that
    only keeps up because the whole market is rising is a weaker signal.
    """
    if args.min_rel_strength_pct is None or market_trend is None:
        return True
    lookback = args.btc_momentum_candles
    if len(window) <= lookback:
        return True  # not enough history to judge - do not block
    point = market_trend.at_or_before(window[-1].close_time)
    if point is None:
        return True
    coin_momentum = (window[-1].close / max(window[-1 - lookback].close, 1e-9) - 1.0) * 100.0
    return coin_momentum - point.momentum_pct >= args.min_rel_strength_pct


def _resolve_symbols(client: BinanceClient, args: argparse.Namespace) -> list[str]:
    """Resolve liquid perpetual futures symbols for the requested quote asset."""
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    quote = args.quote.upper()
    exchange_info = client.exchange_info()
    perpetuals = {
        str(item.get("symbol", ""))
        for item in exchange_info.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and (quote == "ALL" or item.get("quoteAsset") == quote)
    }
    if args.start_ms is not None or args.end_ms is not None:
        if args.top > 0:
            print(f"Ignoring --top {args.top} for date-window run; scanning all eligible symbols.")
        # Avoid ranking or filtering the historical universe with today's 24h ticker.
        # Historical candle volume filters inside the detector handle liquidity per window.
        return sorted(symbol for symbol in perpetuals if symbol.isascii() and symbol.isalnum())

    tickers = client.ticker_24h()
    rows: list[tuple[str, float]] = []
    if isinstance(tickers, list):
        for ticker in tickers:
            symbol = str(ticker.get("symbol", ""))
            if symbol not in perpetuals or not symbol.isascii() or not symbol.isalnum():
                continue
            if _as_float(ticker.get("quoteVolume")) < args.min_quote_volume:
                continue  # too illiquid to trade
            last = _as_float(ticker.get("lastPrice"))
            low = _as_float(ticker.get("lowPrice"))
            high = _as_float(ticker.get("highPrice"))
            if last <= 0 or low <= 0:
                continue
            range_pct = (high - low) / last * 100.0
            rows.append((symbol, range_pct))
    if args.top <= 0:
        return sorted(symbol for symbol, _ in rows)
    rows.sort(key=lambda row: row[1], reverse=True)
    return [symbol for symbol, _ in rows[: args.top]]


def _backtest_symbol(
    symbol: str,
    candles: list[Candle],
    args: argparse.Namespace,
    settings: BreakoutSettings,
    start_ms: int | None,
    end_ms: int | None,
    market_trend: MarketTrend | None = None,
    mtf_align: MtfAlignment | None = None,
    ml_model: dict | None = None,
) -> list[BacktestTrade]:
    interval_ms = interval_to_ms(args.interval)
    candles_per_24h = max(int(86_400_000 / interval_ms), 1)
    min_needed = max(settings.resistance_lookback, settings.squeeze_lookback, settings.volume_lookback, 50) + 2
    start = max(min_needed, candles_per_24h)
    trades: list[BacktestTrade] = []
    symbol_r_history: list[float] = []
    index = start
    total = len(candles)
    while index < total - 1:
        window = candles[: index + 1]
        recent = window[-candles_per_24h:]
        quote_volume_24h = sum(candle.quote_volume for candle in recent)
        high_24h = max(candle.high for candle in recent)
        low_24h = min(candle.low for candle in recent)
        range_pct_24h = (high_24h - low_24h) / max(window[-1].close, 1e-9) * 100.0
        if args.detector == "simple":
            signal = _detect_simple_signal(
                symbol,
                window,
                quote_volume_24h,
                interval_ms,
                args,
                settings,
                range_pct_24h,
            )
        else:
            signal = evaluate_breakout(
                symbol,
                window,
                quote_volume_24h,
                interval_ms,
                interval=args.interval,
                range_pct_24h=range_pct_24h,
                settings=settings,
                include_confirmed=True,
                now_ms=window[-1].close_time + 1,
            )
        if (
            signal is None
            or signal.reward_risk < args.min_rr
            or signal.score < args.min_score
            or (args.longs_only and signal.side == "SHORT")
            or (args.detector == "squeeze" and _dead_coin_reason(signal))
        ):
            index += 1
            continue
        regime = _classify_entry_regime(signal)
        if (
            regime in args.skip_entry_regimes
            or not _market_allows_signal(signal, window[-1].close_time, market_trend, args)
            or not _instant_market_allows(regime, window[-1].close_time, market_trend, args)
            or not _regime_allowed_in_market(regime, window[-1].close_time, market_trend, args)
            or not _side_allowed_in_market(signal.side, window[-1].close_time, market_trend, args)
            or not _rel_strength_allows(signal, window, args, market_trend)
            or not _mtf_aligned(mtf_align, window[-1].close_time, signal.side)
        ):
            index += 1
            continue
        context_features = _signal_context_features(window, market_trend, args, symbol_r_history)
        ml_scores = _ml_scores_signal(
            signal,
            regime,
            ml_model,
            context_features,
            regime_aware=args.ml_regime_aware,
        )
        if ml_model is not None:
            selected_score = _ml_selected_score(ml_scores, args.ml_filter_score)
            ml_active = args.ml_filter_start_ms is None or window[-1].close_time >= args.ml_filter_start_ms
            if ml_active and not args.ml_score_only and selected_score is not None and selected_score < args.ml_filter_threshold:
                index += 1
                continue
        trade = _simulate_trade(signal, candles, index, args, regime)
        if trade is None:
            index += 1
            continue
        _apply_context_features(trade, context_features)
        _apply_ml_scores(trade, ml_scores, args.ml_filter_score)
        if (start_ms is not None and trade.entry_time < start_ms) or (end_ms is not None and trade.entry_time > end_ms):
            index += 1
            continue
        trades.append(trade)
        symbol_r_history.append(trade.r_multiple)
        index = max(trade.exit_time and _index_of_time(candles, trade.exit_time), index) + 1
    return trades


def _signal_context_features(
    window: list[Candle],
    market_trend: MarketTrend | None,
    args: argparse.Namespace,
    symbol_r_history: list[float],
) -> dict[str, float]:
    recent = symbol_r_history[-30:]
    context = {
        "feat_symbol_trades_30": float(len(recent)),
        "feat_symbol_win_rate_30": (
            sum(1 for r in recent if r > 0) / len(recent)
            if recent
            else 0.5
        ),
        "feat_symbol_avg_r_30": (sum(recent) / len(recent) if recent else 0.0),
        "feat_btc_momentum_pct": 0.0,
        "feat_btc_ema_distance_pct": 0.0,
        "feat_rel_momentum_pct": 0.0,
    }
    if market_trend is None:
        return context
    point = market_trend.at_or_before(window[-1].close_time)
    if point is None:
        return context
    context["feat_btc_momentum_pct"] = point.momentum_pct
    context["feat_btc_ema_distance_pct"] = (
        (point.close / max(point.ema, 1e-9) - 1.0) * 100.0
    )
    lookback = args.btc_momentum_candles
    if len(window) > lookback:
        coin_momentum = (window[-1].close / max(window[-1 - lookback].close, 1e-9) - 1.0) * 100.0
        context["feat_rel_momentum_pct"] = coin_momentum - point.momentum_pct
    return context


def _apply_context_features(trade: BacktestTrade, context: dict[str, float]) -> None:
    trade.feat_symbol_trades_30 = context["feat_symbol_trades_30"]
    trade.feat_symbol_win_rate_30 = context["feat_symbol_win_rate_30"]
    trade.feat_symbol_avg_r_30 = context["feat_symbol_avg_r_30"]
    trade.feat_btc_momentum_pct = context["feat_btc_momentum_pct"]
    trade.feat_btc_ema_distance_pct = context["feat_btc_ema_distance_pct"]
    trade.feat_rel_momentum_pct = context["feat_rel_momentum_pct"]


def _ml_selected_score(scores: dict[str, float | str] | None, score_name: str) -> float | None:
    if not scores:
        return None
    value = scores.get(score_name)
    return float(value) if isinstance(value, (int, float)) else None


def _apply_ml_scores(trade: BacktestTrade, scores: dict[str, float | str] | None, score_name: str) -> None:
    if not scores:
        return
    trade.ml_p_win = float(scores.get("pwin", 0.0))
    trade.ml_expected_r = float(scores.get("expected-r", 0.0))
    trade.ml_tail_prob = float(scores.get("tail", 0.0))
    trade.ml_not_bad_prob = float(scores.get("not-bad", 0.0))
    trade.ml_market_regime = str(scores.get("market_regime", ""))
    selected = _ml_selected_score(scores, score_name)
    trade.ml_score = selected if selected is not None else 0.0


def _detect_simple_signal(
    symbol: str,
    window: list[Candle],
    quote_volume_24h: float,
    interval_ms: int,
    args: argparse.Namespace,
    settings: BreakoutSettings,
    range_pct_24h: float,
) -> BreakoutSignal | None:
    now_ms = window[-1].close_time + 1
    long_signal = detect_long_breakout(
        symbol,
        window,
        quote_volume_24h,
        interval_ms,
        interval=args.interval,
        range_pct_24h=range_pct_24h,
        settings=settings,
        now_ms=now_ms,
    )
    if args.longs_only:
        return long_signal
    short_signal = detect_short_breakdown(
        symbol,
        window,
        quote_volume_24h,
        interval_ms,
        interval=args.interval,
        range_pct_24h=range_pct_24h,
        settings=settings,
        now_ms=now_ms,
    )
    if long_signal and short_signal:
        return max((long_signal, short_signal), key=lambda item: (item.reward_risk, item.score))
    return long_signal or short_signal


def _index_of_time(candles: list[Candle], close_time: int) -> int:
    for offset in range(len(candles) - 1, -1, -1):
        if candles[offset].close_time == close_time:
            return offset
    return len(candles) - 1


def _resolve_entry(
    signal: BreakoutSignal,
    candles: list[Candle],
    signal_index: int,
    args: argparse.Namespace,
) -> tuple[float, int] | None:
    """Resolve the entry price and entry-candle index for the chosen --entry-style.

    trigger          - enter when price crosses the breakout trigger (the v1 model).
    retest           - wait for a pullback to the broken level, limit-fill there.
    retest-confirmed - after the pullback, enter only once a candle closes back
                       through the level the right way; abort if one closes the
                       wrong side first (a failed breakout / liquidity sweep).
    """
    side = signal.side
    slip = args.slippage_pct / 100.0
    window = min(signal_index + 1 + args.trigger_wait_candles, len(candles))

    if args.entry_style in ("retest", "retest-confirmed"):
        level = signal.resistance if side == "LONG" else signal.support
        if level <= 0:
            return None
        touched: int | None = None
        for j in range(signal_index + 1, window):
            candle = candles[j]
            if (side == "LONG" and candle.low <= level) or (side == "SHORT" and candle.high >= level):
                touched = j
                break
        if touched is None:
            return None  # price never returned to retest the level
        if args.entry_style == "retest":
            raw, idx = level, touched
        else:
            raw, idx = 0.0, -1
            for k in range(touched, window):
                candle = candles[k]
                if side == "LONG":
                    if candle.close < level:
                        return None  # level lost on a close - failed breakout, abort
                    if candle.close > candle.open:
                        raw, idx = candle.close, k  # bullish reclaim - confirmed
                        break
                else:
                    if candle.close > level:
                        return None
                    if candle.close < candle.open:
                        raw, idx = candle.close, k
                        break
            if idx < 0:
                return None  # no confirmation within the window
        entry = raw * (1.0 + slip) if side == "LONG" else raw * (1.0 - slip)
        return entry, idx

    # trigger style
    trigger = signal.trigger_price
    if args.entry_on_signal_close and signal.status in ("BREAKOUT", "BREAKDOWN"):
        idx = signal_index
        raw = candles[idx].close
    else:
        idx = -1
        for j in range(signal_index + 1, window):
            candle = candles[j]
            if (side == "LONG" and candle.high >= trigger) or (side == "SHORT" and candle.low <= trigger):
                idx = j
                break
        if idx < 0:
            return None
        raw = max(trigger, candles[idx].open) if side == "LONG" else min(trigger, candles[idx].open)
    entry = raw * (1.0 + slip) if side == "LONG" else raw * (1.0 - slip)
    return entry, idx


def _simulate_trade(
    signal: BreakoutSignal,
    candles: list[Candle],
    signal_index: int,
    args: argparse.Namespace,
    regime: str | None = None,
) -> BacktestTrade | None:
    side = signal.side
    stop = signal.stop_price
    target = signal.target_price
    if signal.trigger_price <= 0 or stop <= 0 or target <= 0:
        return None

    resolved = _resolve_entry(signal, candles, signal_index, args)
    if resolved is None:
        return None
    entry, triggered_index = resolved
    if (side == "LONG" and stop >= entry) or (side == "SHORT" and stop <= entry):
        return None
    trigger_candle = candles[triggered_index]

    leverage = _dynamic_leverage(signal.atr_pct, signal.risk_pct, args.leverage) if args.dynamic_leverage else args.leverage
    stop = _leverage_capped_stop(side, entry, stop, int(leverage), args.max_sl_loss_pct)
    profile = (
        smart_take_profit_profile(
            signal,
            tp_count=args.tp_count,
            trailing_stop=args.trailing_stop,
            base_runner_pct=args.runner_pct,
            max_target_multiplier=args.smart_tp_max_target_multiplier,
            min_runner_pct=args.smart_tp_min_runner_pct,
            max_runner_pct=args.smart_tp_max_runner_pct,
        )
        if args.smart_tp
        else equal_take_profit_profile(signal, args.tp_count, args.trailing_stop, args.runner_pct)
    )
    signal = profile.signal
    target = signal.target_price
    avg_exit, exit_index = _simulate_exit(
        side,
        candles,
        triggered_index,
        entry,
        stop,
        target,
        args,
        tp_splits_pct=profile.tp_splits_pct,
        runner_pct=profile.runner_pct,
    )

    if side == "LONG":
        price_return = (avg_exit - entry) / entry
        risk = entry - stop
        r_multiple = (avg_exit - entry) / risk if risk > 0 else 0.0
    else:
        price_return = (entry - avg_exit) / entry
        risk = stop - entry
        r_multiple = (entry - avg_exit) / risk if risk > 0 else 0.0
    hold_candles = exit_index - triggered_index
    hold_hours = hold_candles * interval_to_ms(args.interval) / 3_600_000.0

    return BacktestTrade(
        symbol=signal.symbol,
        side=side,
        status=signal.status,
        regime=regime or _classify_entry_regime(signal),
        detected_time=candles[signal_index].close_time,
        entry_time=trigger_candle.close_time,
        exit_time=candles[exit_index].close_time,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        avg_exit_price=avg_exit,
        leverage=int(leverage),
        hold_candles=hold_candles,
        hold_hours=hold_hours,
        r_multiple=r_multiple,
        price_return=price_return,
        tp_runner_pct=profile.runner_pct,
        tp_target_multiplier=profile.target_multiplier,
        tp_conviction=profile.conviction,
        momentum_score=_momentum_score(signal),
        feat_score=signal.score,
        feat_breakout_pct=signal.breakout_pct,
        feat_distance_to_trigger_pct=signal.distance_to_trigger_pct,
        feat_risk_pct=signal.risk_pct,
        feat_reward_pct=signal.reward_pct,
        feat_reward_risk=signal.reward_risk,
        feat_volume_ratio=signal.volume_ratio,
        feat_avg_quote_volume=signal.avg_quote_volume,
        feat_compression_pct=signal.compression_pct,
        feat_atr_pct=signal.atr_pct,
        feat_trend_score=signal.trend_score,
        feat_close_position=signal.close_position,
        feat_quote_volume_24h=signal.quote_volume_24h,
        feat_range_pct_24h=signal.range_pct_24h,
        feat_price_change_pct_24h=signal.price_change_pct_24h,
    )


def _leverage_capped_stop(side: str, entry: float, stop: float, leverage: int, max_loss_pct: float) -> float:
    if max_loss_pct <= 0 or entry <= 0 or leverage <= 0:
        return stop
    max_price_risk = max_loss_pct / 100.0 / leverage
    if max_price_risk <= 0:
        return stop
    if side == "LONG":
        capped = entry * (1.0 - max_price_risk)
        return max(stop, capped)
    capped = entry * (1.0 + max_price_risk)
    return min(stop, capped)


def _simulate_exit(
    side: str,
    candles: list[Candle],
    start_index: int,
    entry: float,
    stop: float,
    target: float,
    args: argparse.Namespace,
    tp_splits_pct: list[float] | None = None,
    runner_pct: float | None = None,
) -> tuple[float, int]:
    """Walk forward from the entry candle; return (avg exit price, exit candle index)."""
    splits = tp_splits_pct if tp_splits_pct is not None else [100.0 / max(args.tp_count, 1) for _ in range(max(args.tp_count, 1))]
    n_tp = len(splits)
    runner = args.runner_pct if runner_pct is None else runner_pct
    runner_frac = runner / 100.0 if args.trailing_stop else 0.0
    tp_fracs = [max(split, 0.0) / 100.0 for split in splits]
    callback = args.trailing_callback_pct / 100.0
    if side == "LONG":
        tp_prices = [entry + (target - entry) * (k / n_tp) for k in range(1, n_tp + 1)] if n_tp else []
    else:
        tp_prices = [entry - (entry - target) * (k / n_tp) for k in range(1, n_tp + 1)] if n_tp else []

    tp_taken = [False] * n_tp
    runner_open = runner_frac > 0
    remaining = 1.0
    realized = 0.0
    peak = entry
    trough = entry
    initial_risk = abs(entry - stop) if entry > 0 else 0.0
    # Track the bar where peak/trough last updated for the exhaustion-exit stall check.
    peak_idx = start_index
    trough_idx = start_index
    exit_index = len(candles) - 1

    for j in range(start_index + 1, len(candles)):
        candle = candles[j]
        prior_peak, prior_trough = peak, trough
        peak = max(peak, candle.high)
        trough = min(trough, candle.low)
        if peak > prior_peak:
            peak_idx = j
        if trough < prior_trough:
            trough_idx = j
        # Profit-lock ladder: ratchet the stop up as profit grows in R multiples.
        # Each rung (trigger_R, lock_R) means "once the peak reaches +trigger_R x
        # initial_risk in our favour, raise the stop floor to entry + lock_R*risk".
        # The locks never EXIT the trade - they only floor the downside, so this
        # never cuts a winner short. A lock_R of 0 uses --breakeven-offset-pct
        # (covers fees). If no ladder is set, fall back to the single
        # --breakeven-trigger-r rung. Applied on the PRIOR peak/trough so the
        # original stop has priority on the bar where a rung first activates.
        if initial_risk > 0:
            pairs = getattr(args, "profit_lock_pairs", None) or []
            if not pairs and args.breakeven_trigger_r > 0:
                pairs = [(args.breakeven_trigger_r, 0.0)]
            if pairs:
                if side == "LONG":
                    best_lock = 0.0
                    for trigger, lock in pairs:
                        if prior_peak >= entry + trigger * initial_risk:
                            lock_price = (
                                entry * (1.0 + args.breakeven_offset_pct / 100.0)
                                if lock == 0
                                else entry + lock * initial_risk
                            )
                            if lock_price > best_lock:
                                best_lock = lock_price
                    if best_lock > stop:
                        stop = best_lock
                else:
                    best_lock = float("inf")
                    for trigger, lock in pairs:
                        if prior_trough <= entry - trigger * initial_risk:
                            lock_price = (
                                entry * (1.0 - args.breakeven_offset_pct / 100.0)
                                if lock == 0
                                else entry - lock * initial_risk
                            )
                            if lock_price < best_lock:
                                best_lock = lock_price
                    if best_lock < float("inf") and best_lock < stop:
                        stop = best_lock
        if args.dynamic_sl:
            window = candles[max(0, j - args.sl_lookback):j]
            if window:
                if side == "LONG":
                    candidate = min(c.low for c in window) * (1.0 - _DYN_SL_BUFFER)
                    if stop < candidate < candle.open:
                        stop = candidate  # ratchet up to recent support, never above price
                else:
                    candidate = max(c.high for c in window) * (1.0 + _DYN_SL_BUFFER)
                    if candle.open < candidate < stop:
                        stop = candidate  # ratchet down to recent resistance
        if side == "LONG":
            if candle.low <= stop:
                realized += remaining * stop
                return realized, j
            for k in range(n_tp):
                if not tp_taken[k] and candle.high >= tp_prices[k]:
                    tp_taken[k] = True
                    realized += tp_fracs[k] * tp_prices[k]
                    remaining -= tp_fracs[k]
            if runner_open:
                trail = peak * (1.0 - callback)
                if candle.low <= trail and trail > stop:
                    realized += runner_frac * trail
                    remaining -= runner_frac
                    runner_open = False
        else:
            if candle.high >= stop:
                realized += remaining * stop
                return realized, j
            for k in range(n_tp):
                if not tp_taken[k] and candle.low <= tp_prices[k]:
                    tp_taken[k] = True
                    realized += tp_fracs[k] * tp_prices[k]
                    remaining -= tp_fracs[k]
            if runner_open:
                trail = trough * (1.0 + callback)
                if candle.high >= trail and trail < stop:
                    realized += runner_frac * trail
                    remaining -= runner_frac
                    runner_open = False
        # Exhaustion exit: once the trade has reached +0.5R, close on a bearish
        # rejection candle near the peak (long upper wick + bearish close) or
        # after 4 candles with no new favourable extreme. Catches give-back
        # (INU-style) and stall (LYN-style) without waiting for the SL to lag in.
        if args.exhaustion_exit and initial_risk > 0 and remaining > 1e-9:
            cr = candle.high - candle.low
            body = abs(candle.close - candle.open)
            if side == "LONG" and peak >= entry + 0.5 * initial_risk:
                upper_wick = candle.high - max(candle.open, candle.close)
                rejection = (
                    cr > 0
                    and upper_wick >= max(body, cr * 1e-6) * 1.5
                    and upper_wick >= cr * 0.4
                    and candle.close < candle.open
                    and candle.high >= peak * 0.99
                )
                stall = (j - peak_idx) >= 4
                if rejection or stall:
                    realized += remaining * candle.close
                    return realized, j
            elif side == "SHORT" and trough <= entry - 0.5 * initial_risk:
                lower_wick = min(candle.open, candle.close) - candle.low
                rejection = (
                    cr > 0
                    and lower_wick >= max(body, cr * 1e-6) * 1.5
                    and lower_wick >= cr * 0.4
                    and candle.close > candle.open
                    and candle.low <= trough * 1.01
                )
                stall = (j - trough_idx) >= 4
                if rejection or stall:
                    realized += remaining * candle.close
                    return realized, j
        # Stagnation exit: only fires after the trade reached +stagnation_after_r,
        # then exits if no new favourable extreme for stagnation_candles bars.
        # Narrower than exhaustion-exit - unproven trades are never closed by stall.
        if (
            args.stagnation_after_r > 0
            and initial_risk > 0
            and remaining > 1e-9
        ):
            if side == "LONG" and peak >= entry + args.stagnation_after_r * initial_risk:
                if (j - peak_idx) >= args.stagnation_candles:
                    realized += remaining * candle.close
                    return realized, j
            elif side == "SHORT" and trough <= entry - args.stagnation_after_r * initial_risk:
                if (j - trough_idx) >= args.stagnation_candles:
                    realized += remaining * candle.close
                    return realized, j
        if remaining <= 1e-9:
            return realized, j

    realized += remaining * candles[-1].close
    return realized, exit_index


def _fmt_time(ms: int) -> str:
    if ms <= 0:
        return "?"
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _portfolio_ordered_trades(trades: list[BacktestTrade], args: argparse.Namespace) -> list[BacktestTrade]:
    if getattr(args, "ml_rank_signals", False):
        ml_start = getattr(args, "ml_filter_start_ms", None)
        return sorted(
            trades,
            key=lambda t: (
                t.entry_time,
                -t.ml_score if ml_start is None or t.entry_time >= ml_start else 0.0,
                -t.momentum_score,
                t.symbol,
            ),
        )
    return sorted(trades, key=lambda t: t.entry_time)


def _simulate_portfolio(
    trades: list[BacktestTrade], args: argparse.Namespace
) -> tuple[list[BacktestTrade], float, float]:
    """Walk signals in time order with a concurrency cap and optional compounding.

    Returns (trades actually taken, final equity, max drawdown percent).
    """
    ordered = _portfolio_ordered_trades(trades, args)
    equity = args.capital
    peak = equity
    max_drawdown = 0.0
    open_positions: list[BacktestTrade] = []
    taken: list[BacktestTrade] = []
    fee_rate = args.fee_pct / 100.0
    funding_rate = args.funding_pct_per_8h / 100.0
    interval_ms = interval_to_ms(args.interval)
    cooldown_until_by_symbol: dict[str, int] = {}
    consecutive_losses_by_symbol: dict[str, int] = {}

    def realize(cutoff: float) -> None:
        nonlocal equity, peak, max_drawdown
        for position in sorted(
            [p for p in open_positions if p.exit_time <= cutoff], key=lambda p: p.exit_time
        ):
            equity += position.net_pnl
            open_positions.remove(position)
            if position.net_pnl < 0:
                losses = consecutive_losses_by_symbol.get(position.symbol, 0) + 1
                consecutive_losses_by_symbol[position.symbol] = losses
                if args.loss_cooldown_candles > 0 and losses >= args.loss_cooldown_after:
                    cooldown_until_by_symbol[position.symbol] = max(
                        cooldown_until_by_symbol.get(position.symbol, 0),
                        position.exit_time + args.loss_cooldown_candles * interval_ms,
                    )
            else:
                consecutive_losses_by_symbol[position.symbol] = 0
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak * 100.0)

    for trade in ordered:
        realize(trade.entry_time)
        if equity <= 0:
            break  # account blown up
        if args.loss_cooldown_candles > 0 and trade.entry_time < cooldown_until_by_symbol.get(trade.symbol, 0):
            continue
        max_concurrent = _max_concurrent_for_equity(equity, args)
        if len(open_positions) >= max_concurrent:
            continue  # no free slot
        if (
            args.reserve_last_slot_s_tier
            and max_concurrent >= 2
            and len(open_positions) == max_concurrent - 1
            and trade.momentum_score < args.s_tier_momentum_threshold
        ):
            continue  # last slot reserved for S-tier setups
        position_pct = _position_pct_for_trade(equity, peak, trade, args)
        margin = equity * (position_pct / 100.0) if args.compound else args.order_margin
        notional = margin * max(trade.leverage, 1)
        if notional < 5.0:
            continue  # below the exchange minimum order size
        fees = notional * fee_rate * 2.0
        funding = notional * funding_rate * (trade.hold_hours / 8.0)
        if trade.side == "SHORT":
            funding = -funding
        trade.taken = True
        trade.margin = margin
        trade.position_pct = position_pct if args.compound else 0.0
        trade.fees_usdt = fees
        trade.funding_usdt = funding
        trade.net_pnl = notional * trade.price_return - fees - funding
        open_positions.append(trade)
        taken.append(trade)

    realize(float("inf"))
    return taken, equity, max_drawdown


def _mark_price_at(candles: list[Candle], target_time_ms: int) -> float:
    """Close of the candle whose close_time <= target. Used by the rotation sim
    to price an early close at an arbitrary moment."""
    if not candles:
        return 0.0
    if target_time_ms <= candles[0].open_time:
        return candles[0].open
    if target_time_ms >= candles[-1].close_time:
        return candles[-1].close
    lo, hi = 0, len(candles) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if candles[mid].close_time <= target_time_ms:
            lo = mid
        else:
            hi = mid - 1
    return candles[lo].close


def _simulate_portfolio_with_rotation(
    trades: list[BacktestTrade],
    candles_by_symbol: dict[str, list[Candle]],
    args: argparse.Namespace,
) -> tuple[list[BacktestTrade], float, float]:
    """Walk signals in time order with a concurrency cap, like _simulate_portfolio,
    plus rotation: when all slots are full and a higher-momentum INSTANT signal
    arrives, close a weaker profitable position to make room. Mirrors the live
    guards (ROTATION_MIN_EDGE, ROTATION_MIN_HOLD_SECONDS, ROTATION_COOLDOWN_SECONDS).

    A position closed by rotation has its trade fields (exit_time, exit_price,
    r_multiple, price_return, hold_*) overwritten so the trade log reflects what
    actually happened, not the would-have-been natural exit.
    """
    ordered = _portfolio_ordered_trades(trades, args)
    equity = args.capital
    peak = equity
    max_drawdown = 0.0
    fee_rate = args.fee_pct / 100.0
    funding_rate = args.funding_pct_per_8h / 100.0
    interval_ms = interval_to_ms(args.interval)
    cooldown_until_by_symbol: dict[str, int] = {}
    consecutive_losses_by_symbol: dict[str, int] = {}
    open_positions: list[dict] = []
    taken: list[BacktestTrade] = []
    last_rotation_time_ms = 0
    rotation_cooldown_ms = _ROTATION_COOLDOWN_SECONDS * 1000
    rotation_min_hold_ms = _ROTATION_MIN_HOLD_SECONDS * 1000

    def close_position(pos: dict, exit_time_ms: int, exit_price: float, by_rotation: bool) -> None:
        nonlocal equity, peak, max_drawdown
        trade = pos["trade"]
        side = pos["side"]
        entry_price = pos["entry_price"]
        notional = pos["notional"]
        fees = pos["fees"]
        if side == "LONG":
            price_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
        else:
            price_return = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0
        hold_hours = max((exit_time_ms - pos["entry_time"]) / 3_600_000.0, 0.0)
        funding = notional * funding_rate * (hold_hours / 8.0)
        if side == "SHORT":
            funding = -funding
        net_pnl = notional * price_return - fees - funding
        trade.taken = True
        trade.margin = pos["margin"]
        trade.position_pct = pos["position_pct"]
        trade.fees_usdt = fees
        trade.funding_usdt = funding
        trade.net_pnl = net_pnl
        if by_rotation:
            trade.exit_time = exit_time_ms
            trade.avg_exit_price = exit_price
            trade.price_return = price_return
            risk = abs(entry_price - trade.stop_price) if entry_price > 0 else 0.0
            if side == "LONG":
                trade.r_multiple = (exit_price - entry_price) / risk if risk > 0 else 0.0
            else:
                trade.r_multiple = (entry_price - exit_price) / risk if risk > 0 else 0.0
            trade.hold_candles = max(int((exit_time_ms - pos["entry_time"]) / max(interval_ms, 1)), 0)
            trade.hold_hours = hold_hours
        equity += net_pnl
        if net_pnl < 0:
            losses = consecutive_losses_by_symbol.get(trade.symbol, 0) + 1
            consecutive_losses_by_symbol[trade.symbol] = losses
            if args.loss_cooldown_candles > 0 and losses >= args.loss_cooldown_after:
                cooldown_until_by_symbol[trade.symbol] = max(
                    cooldown_until_by_symbol.get(trade.symbol, 0),
                    exit_time_ms + args.loss_cooldown_candles * interval_ms,
                )
        else:
            consecutive_losses_by_symbol[trade.symbol] = 0
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100.0)

    def realize_natural_exits(cutoff_ms: int) -> None:
        due = [p for p in open_positions if p["natural_exit_time"] <= cutoff_ms]
        due.sort(key=lambda p: p["natural_exit_time"])
        for pos in due:
            close_position(pos, pos["natural_exit_time"], pos["natural_exit_price"], by_rotation=False)
            open_positions.remove(pos)

    def open_new(trade: BacktestTrade) -> bool:
        position_pct = _position_pct_for_trade(equity, peak, trade, args)
        margin = equity * (position_pct / 100.0) if args.compound else args.order_margin
        notional = margin * max(trade.leverage, 1)
        if notional < 5.0:
            return False
        fees = notional * fee_rate * 2.0
        open_positions.append({
            "trade": trade,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "natural_exit_time": trade.exit_time,
            "natural_exit_price": trade.avg_exit_price,
            "margin": margin,
            "notional": notional,
            "position_pct": position_pct if args.compound else 0.0,
            "fees": fees,
            "momentum_score": trade.momentum_score,
        })
        taken.append(trade)
        return True

    for trade in ordered:
        realize_natural_exits(trade.entry_time)
        if equity <= 0:
            break
        if args.loss_cooldown_candles > 0 and trade.entry_time < cooldown_until_by_symbol.get(trade.symbol, 0):
            continue
        max_concurrent = _max_concurrent_for_equity(equity, args)
        if len(open_positions) < max_concurrent:
            # Reserve the final slot for S-tier signals when configured.
            if (
                args.reserve_last_slot_s_tier
                and max_concurrent >= 2
                and len(open_positions) == max_concurrent - 1
                and trade.momentum_score < args.s_tier_momentum_threshold
            ):
                continue  # last slot reserved for S-tier setups
            open_new(trade)
            continue
        # Slots full - try rotation. Mirrors live: only INSTANT exploders rotate.
        if trade.regime != "INSTANT":
            continue
        if trade.entry_time - last_rotation_time_ms < rotation_cooldown_ms:
            continue
        candidates: list[tuple[dict, float]] = []
        for pos in open_positions:
            if pos["momentum_score"] + _ROTATION_MIN_EDGE >= trade.momentum_score:
                continue  # exploder does not clearly out-rank this position
            if trade.entry_time - pos["entry_time"] < rotation_min_hold_ms:
                continue  # position has not had room to work yet
            sym_candles = candles_by_symbol.get(pos["symbol"])
            if not sym_candles:
                continue
            mark = _mark_price_at(sym_candles, trade.entry_time)
            if mark <= 0:
                continue
            side = pos["side"]
            entry_price = pos["entry_price"]
            if side == "LONG":
                unrealized_pct = (mark - entry_price) / entry_price if entry_price > 0 else 0.0
            else:
                unrealized_pct = (entry_price - mark) / entry_price if entry_price > 0 else 0.0
            net_pct = unrealized_pct - _ROTATION_FEE_RATE
            if net_pct <= 0:
                continue  # only rotate out positions that are profitable net of close fees
            candidates.append((pos, mark))
        if not candidates:
            continue
        chosen, chosen_mark = min(candidates, key=lambda c: c[0]["momentum_score"])
        close_position(chosen, trade.entry_time, chosen_mark, by_rotation=True)
        open_positions.remove(chosen)
        last_rotation_time_ms = trade.entry_time
        open_new(trade)

    realize_natural_exits(2 ** 63 - 1)
    return taken, equity, max_drawdown


def _max_concurrent_for_equity(equity: float, args: argparse.Namespace) -> int:
    if args.max_concurrent > 0:
        return args.max_concurrent
    if not args.compound:
        return 2
    if equity < 25:
        return 2
    if equity < 75:
        return 3
    if equity < 200:
        return 4
    if equity < 500:
        return 5
    if equity < 1_000:
        return 6
    return 8


def _position_pct_for_trade(equity: float, peak: float, trade: BacktestTrade, args: argparse.Namespace) -> float:
    if not args.compound:
        return 0.0
    if args.position_pct > 0:
        return args.position_pct
    if args.sizing_mode == "guarded":
        return _auto_position_pct(equity)
    if args.sizing_mode == "auto":
        return _auto_curve_position_pct(equity, peak)

    if args.sizing_mode == "aggressive":
        pct = _aggressive_base_position_pct(equity, args.capital)
        dd_caps = (45.0, 60.0, 75.0)
        ceiling = 85.0
    else:  # moonshot
        pct = _moonshot_base_position_pct(equity, args.capital)
        dd_caps = (12.0, 20.0, 30.0)
        ceiling = 50.0

    drawdown = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
    if args.capital > 0 and peak >= args.capital * 2.0:
        if drawdown >= 35.0:
            pct = min(pct, dd_caps[0])
        elif drawdown >= 20.0:
            pct = min(pct, dd_caps[1])
        elif drawdown >= 10.0:
            pct = min(pct, dd_caps[2])

    if trade.regime == "RETEST":
        pct *= args.retest_size_multiplier
    elif trade.regime == "INSTANT":
        pct *= args.instant_size_multiplier
    elif trade.regime == "TRAILING_RETEST":
        pct *= args.trailing_retest_size_multiplier

    return min(max(pct, 5.0), ceiling)


def _auto_curve_position_pct(equity: float, peak: float) -> float:
    """Equity-tier base % with a drawdown haircut.

    Designed for micro-account growth: aggressive (~55%) on very small balances,
    tapering as the account becomes meaningful. Drawdown-aware so a deep
    drawdown progressively de-risks (down to a 5% floor) rather than letting
    the equity-tier % keep compounding losses.

    Tiers are based on ABSOLUTE equity in USDT (not multiple-of-start), so
    behaviour is consistent regardless of starting capital.
    """
    if equity < 25.0:
        base = 55.0
    elif equity < 100.0:
        base = 45.0
    elif equity < 500.0:
        base = 32.0
    elif equity < 2_500.0:
        base = 22.0
    elif equity < 10_000.0:
        base = 15.0
    else:
        base = 10.0

    drawdown = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
    if drawdown < 15.0:
        dd_mult = 1.00
    elif drawdown < 25.0:
        dd_mult = 0.85
    elif drawdown < 35.0:
        dd_mult = 0.65
    elif drawdown < 50.0:
        dd_mult = 0.45
    else:
        dd_mult = 0.25

    return min(max(base * dd_mult, 5.0), 60.0)


def _moonshot_base_position_pct(equity: float, starting_capital: float) -> float:
    if starting_capital <= 0:
        return 10.0
    multiple = equity / starting_capital
    if multiple < 1.5:
        return 50.0
    if multiple < 2.5:
        return 40.0
    if multiple < 5.0:
        return 30.0
    if multiple < 10.0:
        return 22.0
    if multiple < 25.0:
        return 16.0
    if multiple < 50.0:
        return 12.0
    return 8.0


def _aggressive_base_position_pct(equity: float, starting_capital: float) -> float:
    """Aggressive dynamic tiers - stay large as equity compounds (high path risk)."""
    if starting_capital <= 0:
        return 25.0
    multiple = equity / starting_capital
    if multiple < 2.0:
        return 65.0
    if multiple < 5.0:
        return 50.0
    if multiple < 15.0:
        return 40.0
    if multiple < 50.0:
        return 32.0
    return 26.0


def _write_trade_log(
    trades: list[BacktestTrade],
    path: str | Path = "backtest_trades.csv",
    args: argparse.Namespace | None = None,
) -> Path:
    output = Path(path)
    fieldnames = [
        "detected_date",
        "entry_date",
        "exited_date",
        "symbol",
        "side",
        "status",
        "regime",
        "entry",
        "stop",
        "target",
        "tp_runner_pct",
        "tp_target_multiplier",
        "tp_conviction",
        "exit_price",
        "hold_candles",
        "r_multiple",
        "price_return_pct",
        "leverage",
        "taken",
        "position_pct",
        "margin",
        "net_pnl",
        "fees",
        "funding",
        "outcome",
        "ml_score",
        "ml_p_win",
        "ml_expected_r",
        "ml_tail_prob",
        "ml_not_bad_prob",
        "ml_market_regime",
        # ML feature snapshot (at signal-detection time)
        "feat_score",
        "feat_breakout_pct",
        "feat_distance_to_trigger_pct",
        "feat_risk_pct",
        "feat_reward_pct",
        "feat_reward_risk",
        "feat_volume_ratio",
        "feat_avg_quote_volume",
        "feat_compression_pct",
        "feat_atr_pct",
        "feat_trend_score",
        "feat_close_position",
        "feat_quote_volume_24h",
        "feat_range_pct_24h",
        "feat_price_change_pct_24h",
        "feat_momentum_score",
        "feat_btc_momentum_pct",
        "feat_btc_ema_distance_pct",
        "feat_rel_momentum_pct",
        "feat_symbol_trades_30",
        "feat_symbol_win_rate_30",
        "feat_symbol_avg_r_30",
        "feat_interval",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in sorted(trades, key=lambda item: item.detected_time):
            outcome = "skipped"
            if trade.taken:
                outcome = "WIN" if trade.is_win else "LOSS"
            writer.writerow(
                {
                    "detected_date": _fmt_time(trade.detected_time),
                    "entry_date": _fmt_time(trade.entry_time),
                    "exited_date": _fmt_time(trade.exit_time),
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "status": trade.status,
                    "regime": trade.regime,
                    "entry": f"{trade.entry_price:.12g}",
                    "stop": f"{trade.stop_price:.12g}",
                    "target": f"{trade.target_price:.12g}",
                    "tp_runner_pct": f"{trade.tp_runner_pct:.2f}",
                    "tp_target_multiplier": f"{trade.tp_target_multiplier:.3f}",
                    "tp_conviction": f"{trade.tp_conviction:.3f}",
                    "exit_price": f"{trade.avg_exit_price:.12g}",
                    "hold_candles": trade.hold_candles,
                    "r_multiple": f"{trade.r_multiple:.4f}",
                    "price_return_pct": f"{trade.price_return * 100.0:.4f}",
                    "leverage": trade.leverage,
                    "taken": "yes" if trade.taken else "no",
                    "position_pct": f"{trade.position_pct:.2f}" if trade.taken else "",
                    "margin": f"{trade.margin:.6f}",
                    "net_pnl": f"{trade.net_pnl:.6f}",
                    "fees": f"{trade.fees_usdt:.6f}",
                    "funding": f"{trade.funding_usdt:.6f}",
                    "outcome": outcome,
                    "ml_score": f"{trade.ml_score:.6f}",
                    "ml_p_win": f"{trade.ml_p_win:.6f}",
                    "ml_expected_r": f"{trade.ml_expected_r:.6f}",
                    "ml_tail_prob": f"{trade.ml_tail_prob:.6f}",
                    "ml_not_bad_prob": f"{trade.ml_not_bad_prob:.6f}",
                    "ml_market_regime": trade.ml_market_regime,
                    "feat_score": f"{trade.feat_score:.6f}",
                    "feat_breakout_pct": f"{trade.feat_breakout_pct:.6f}",
                    "feat_distance_to_trigger_pct": f"{trade.feat_distance_to_trigger_pct:.6f}",
                    "feat_risk_pct": f"{trade.feat_risk_pct:.6f}",
                    "feat_reward_pct": f"{trade.feat_reward_pct:.6f}",
                    "feat_reward_risk": f"{trade.feat_reward_risk:.6f}",
                    "feat_volume_ratio": f"{trade.feat_volume_ratio:.6f}",
                    "feat_avg_quote_volume": f"{trade.feat_avg_quote_volume:.6f}",
                    "feat_compression_pct": f"{trade.feat_compression_pct:.6f}",
                    "feat_atr_pct": f"{trade.feat_atr_pct:.6f}",
                    "feat_trend_score": f"{trade.feat_trend_score:.6f}",
                    "feat_close_position": f"{trade.feat_close_position:.6f}",
                    "feat_quote_volume_24h": f"{trade.feat_quote_volume_24h:.6f}",
                    "feat_range_pct_24h": f"{trade.feat_range_pct_24h:.6f}",
                    "feat_price_change_pct_24h": f"{trade.feat_price_change_pct_24h:.6f}",
                    "feat_momentum_score": f"{trade.momentum_score:.6f}",
                    "feat_btc_momentum_pct": f"{trade.feat_btc_momentum_pct:.6f}",
                    "feat_btc_ema_distance_pct": f"{trade.feat_btc_ema_distance_pct:.6f}",
                    "feat_rel_momentum_pct": f"{trade.feat_rel_momentum_pct:.6f}",
                    "feat_symbol_trades_30": f"{trade.feat_symbol_trades_30:.6f}",
                    "feat_symbol_win_rate_30": f"{trade.feat_symbol_win_rate_30:.6f}",
                    "feat_symbol_avg_r_30": f"{trade.feat_symbol_avg_r_30:.6f}",
                    "feat_interval": args.interval if args is not None else "",
                }
            )
    return output


def _print_report(
    trades: list[BacktestTrade],
    taken: list[BacktestTrade],
    final_equity: float,
    max_drawdown: float,
    args: argparse.Namespace,
    window_start: int,
    window_end: int,
    btc_return: float | None,
) -> None:
    print()
    print("=" * 64)
    print("BACKTEST REPORT")
    print("=" * 64)
    window_span = f"full date window x {args.interval}" if args.start_ms is not None else f"{args.history} x {args.interval}"
    print(f"Window      {_fmt_time(window_start)} -> {_fmt_time(window_end)} UTC  ({window_span})")
    if btc_return is not None:
        regime = "uptrend" if btc_return > 3 else "downtrend" if btc_return < -3 else "flat"
        print(f"Market      BTC {btc_return:+.1f}% over the window  ({regime})")
    sizing = (
        _sizing_label(args)
        if args.compound
        else f"fixed {args.order_margin:.0f} USDT margin"
    )
    concurrency = f"max {args.max_concurrent} concurrent" if args.max_concurrent > 0 else "dynamic concurrency"
    print(f"Account     {args.capital:.2f} USDT start, {sizing}, {concurrency}")
    print(f"Direction   {'longs only' if args.longs_only else 'longs + shorts'}")
    if args.skip_entry_regimes:
        print(f"Filter      skip regimes: {', '.join(sorted(args.skip_entry_regimes))}")
    if args.btc_trend_filter:
        print(
            f"Filter      BTC trend EMA {args.btc_ema_candles}, "
            f"momentum {args.btc_momentum_candles}"
        )
    if args.instant_market_guard:
        print(
            f"Filter      INSTANT BTC guard: momentum >= {args.instant_guard_momentum_pct:.1f}%, "
            f"EMA slack {args.instant_guard_ema_slack_pct:.1f}%"
        )
    if args.hostile_market_strict_only:
        print(
            f"Filter      hostile BTC -> STRICT_RETEST only "
            f"(mom < {args.hostile_momentum_pct:.1f}% or EMA slack {args.hostile_ema_slack_pct:.1f}%)"
        )
    if args.short_market_guard:
        print(
            f"Filter      SHORT BTC guard: momentum <= {args.short_guard_momentum_pct:.1f}%, "
            f"EMA slack {args.short_guard_ema_slack_pct:.1f}%"
        )
    if args.ml_filter_model:
        ml_mode = "score-only" if args.ml_score_only else f"threshold >= {args.ml_filter_threshold:.3f}"
        ml_rank = ", ranked" if args.ml_rank_signals else ""
        ml_regime = ", regime-aware" if args.ml_regime_aware else ""
        ml_start = f", active from {args.ml_filter_start_date}" if args.ml_filter_start_ms is not None else ""
        print(f"Filter      ML {args.ml_filter_score}: {ml_mode}{ml_rank}{ml_regime}{ml_start}")
    if args.loss_cooldown_candles > 0:
        print(
            f"Risk guard  pause each symbol {args.loss_cooldown_candles} candle(s) after "
            f"{args.loss_cooldown_after} realized loss(es) on that symbol"
        )
    if args.entry_on_signal_close:
        print("Entry      confirmed breakouts enter on signal close")
    if args.entry_style != "trigger":
        print(f"Entry      style: {args.entry_style}")
    if args.trailing_stop:
        if args.smart_tp:
            print(
                f"Exit       smart TP: target up to x{args.smart_tp_max_target_multiplier:.2f}, "
                f"runner {args.smart_tp_min_runner_pct:.0f}-{args.smart_tp_max_runner_pct:.0f}%"
            )
        else:
            print(f"Exit       trailing runner {args.runner_pct:.0f}% of position")
    if args.max_sl_loss_pct > 0:
        print(f"Stop cap   max leveraged stop loss {args.max_sl_loss_pct:.1f}% of margin")
    if args.compound and args.position_pct <= 0 and args.sizing_mode in ("moonshot", "aggressive"):
        print(
            f"Sizing mods INSTANT x{args.instant_size_multiplier:.2f}, "
            f"RETEST x{args.retest_size_multiplier:.2f}, "
            f"TRAILING_RETEST x{args.trailing_retest_size_multiplier:.2f}"
        )
    if not trades:
        print("No signals were generated. Try a longer --history or more --top symbols.")
        return

    if not taken:
        print(f"{len(trades)} signal(s) generated, but none could be sized/slotted.")
        return

    wins = [t for t in taken if t.is_win]
    losses = [t for t in taken if not t.is_win]
    total = len(taken)
    win_rate = len(wins) / total * 100.0
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)
    total_pnl = final_equity - args.capital
    total_fees = sum(t.fees_usdt for t in taken)
    total_funding = sum(t.funding_usdt for t in taken)
    expectancy_r = sum(t.r_multiple for t in taken) / total
    avg_win_r = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
    avg_loss_r = sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    ret_pct = (final_equity / args.capital - 1.0) * 100.0 if args.capital > 0 else 0.0

    print(f"Signals               {len(trades)} generated -> {total} taken, {len(trades) - total} not taken")
    print(f"Final equity          {final_equity:.2f} USDT   ({ret_pct:+.1f}%, net {total_pnl:+.2f})")
    print(f"Win rate              {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Expectancy            {expectancy_r:+.3f} R per trade")
    print(f"Avg win / avg loss    {avg_win_r:+.2f} R  /  {avg_loss_r:+.2f} R")
    print(f"Profit factor         {profit_factor:.2f}")
    print(f"Fees paid             {total_fees:.2f} USDT")
    print(f"Funding paid          {total_funding:+.2f} USDT  (estimated @ {args.funding_pct_per_8h:.3f}%/8h)")
    print(f"Max drawdown          {max_drawdown:.1f}%")
    print(f"Avg hold              {sum(t.hold_candles for t in taken) / total:.1f} candles")
    if args.smart_tp:
        print(
            f"Smart TP avg          conviction {sum(t.tp_conviction for t in taken) / total:.2f}, "
            f"target x{sum(t.tp_target_multiplier for t in taken) / total:.2f}, "
            f"runner {sum(t.tp_runner_pct for t in taken) / total:.1f}%"
        )
    _print_conservative_sizing_comparison(trades, args)

    print()
    print("By signal status:")
    _print_grouped(taken, key=lambda t: t.status)
    print()
    print("By side:")
    _print_grouped(taken, key=lambda t: t.side)
    print()
    print("By entry regime:")
    _print_grouped(taken, key=lambda t: t.regime)

    print()
    print("NOTE: v1 model - market entry on breakout, static stop, no dynamic-SL.")
    print("Drawdown is on realized equity only. Over ONE window/regime - not a guarantee.")


def _print_conservative_sizing_comparison(trades: list[BacktestTrade], args: argparse.Namespace) -> None:
    if not args.compound:
        return
    print()
    print("Sizing comparison (same signals, compounding):")
    print(f"  Primary sizing: {_sizing_label(args)}.")
    for pct in (5.0, 10.0, 50.0):
        variant_args = argparse.Namespace(**vars(args))
        variant_args.position_pct = pct
        variant_args.sizing_mode = "guarded"
        variant_trades = _fresh_trade_copies(trades)
        taken, final_equity, max_drawdown = _simulate_portfolio(variant_trades, variant_args)
        ret_pct = (final_equity / args.capital - 1.0) * 100.0 if args.capital > 0 else 0.0
        print(
            f"  {pct:>4.0f}% per trade: {len(taken):4d} taken, final {final_equity:.2f} USDT "
            f"({ret_pct:+.1f}%), max DD {max_drawdown:.1f}%"
        )


def _sizing_label(args: argparse.Namespace) -> str:
    if args.position_pct > 0:
        return f"compounding fixed {args.position_pct:.0f}% of equity"
    if args.sizing_mode == "auto":
        return "compounding auto curve (absolute-equity tiers + drawdown haircut)"
    if args.sizing_mode == "moonshot":
        return "compounding moonshot dynamic tiers"
    if args.sizing_mode == "aggressive":
        return "compounding aggressive dynamic tiers"
    return f"compounding guarded {_auto_position_pct(args.capital):.0f}% of equity"


def _fresh_trade_copies(trades: list[BacktestTrade]) -> list[BacktestTrade]:
    return [
        replace(
            trade,
            taken=False,
            margin=0.0,
            net_pnl=0.0,
            fees_usdt=0.0,
            funding_usdt=0.0,
            position_pct=0.0,
        )
        for trade in trades
    ]


def _print_grouped(trades: list[BacktestTrade], key) -> None:
    groups: dict[str, list[BacktestTrade]] = {}
    for trade in trades:
        groups.setdefault(key(trade), []).append(trade)
    for name in sorted(groups):
        group = groups[name]
        wins = sum(1 for t in group if t.is_win)
        expectancy = sum(t.r_multiple for t in group) / len(group)
        net = sum(t.net_pnl for t in group)
        print(
            f"  {name:16s} {len(group):4d} trades   win {wins / len(group) * 100:5.1f}%   "
            f"exp {expectancy:+.3f} R   net {net:+.2f} USDT"
        )


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
