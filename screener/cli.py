from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .binance_client import (
    BinanceClient,
    BinanceClientError,
    SymbolInfo,
    SymbolUniverse,
    discover_symbols,
    filter_by_order_book,
    filter_by_open_interest,
    load_explicit_symbols,
)
from .breakout import (
    BreakoutSettings,
    BreakoutSignal,
    candles_from_klines,
    detect_long_breakout,
    evaluate_breakout,
    interval_to_ms,
)
from .orders import (
    ConditionalOrderPlan,
    OrderPlanError,
    TradingRule,
    build_entry_order_plan,
    build_exit_order_plans,
    trading_rules_from_exchange_info,
    _decimal as _to_decimal,
    _format_decimal as _format_decimal,
    _round_to_step as _round_to_step,
)
from .take_profit import TakeProfitProfile, equal_take_profit_profile, smart_take_profit_profile


API_KEY_ENV_NAMES = ("BINANCE_API_KEY", "BINANCE_FUTURES_API_KEY", "BINANCE_KEY", "API_KEY")
API_SECRET_ENV_NAMES = (
    "BINANCE_API_SECRET",
    "BINANCE_API_SECRET_KEY",
    "BINANCE_FUTURES_API_SECRET",
    "BINANCE_FUTURES_SECRET",
    "BINANCE_FUTURES_SECRET_KEY",
    "BINANCE_SECRET",
    "BINANCE_SECRET_KEY",
    "API_SECRET",
    "SECRET_KEY",
)
TESTNET_API_KEY_ENV_NAMES = ("BINANCE_TESTNET_API_KEY", "BINANCE_DEMO_API_KEY")
TESTNET_API_SECRET_ENV_NAMES = ("BINANCE_TESTNET_API_SECRET", "BINANCE_DEMO_API_SECRET")


@dataclass(frozen=True)
class OrderExecution:
    mode: str
    role: str
    symbol: str
    interval: str
    side: str
    order_type: str
    trigger_price: str
    limit_price: str
    quantity: str
    requested_margin: str
    estimated_notional: str
    leverage: str
    margin_type: str
    client_order_id: str
    status: str
    order_id: str
    message: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ml_rank_model:
        try:
            args.ml_rank_model_data = _load_ml_rank_model(args.ml_rank_model)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"ML ranker disabled - could not load {args.ml_rank_model}: {exc}", file=sys.stderr)
            return 2
    else:
        args.ml_rank_model_data = None
    _load_env_file(args.env_file)
    settings = BreakoutSettings(
        resistance_lookback=args.resistance_lookback,
        squeeze_lookback=args.squeeze_lookback,
        volume_lookback=args.volume_lookback,
        min_breakout_pct=args.min_breakout_pct / 100,
        max_extension_pct=args.max_extension_pct / 100,
        prior_break_tolerance_pct=args.prior_break_tolerance_pct / 100,
        watch_distance_pct=args.max_trigger_distance_pct / 100,
        max_shakeout_distance_pct=args.max_shakeout_distance_pct / 100,
        max_pre_trigger_move_pct=args.max_pre_trigger_move_pct / 100,
        entry_buffer_pct=args.entry_buffer_pct / 100,
        stop_buffer_pct=args.stop_buffer_pct / 100,
        target_range_multiple=args.target_range_multiple,
        min_sweep_pct=args.min_sweep_pct / 100,
        max_fakeout_close_position=args.max_fakeout_close_position,
        min_volume_ratio=args.min_volume_ratio,
        min_watch_volume_ratio=args.min_watch_volume_ratio,
        min_avg_quote_volume=args.min_candle_quote_volume,
        min_close_position=args.min_close_position,
        max_compression_pct=args.max_compression_pct / 100,
        entry_atr_buffer_multiple=args.entry_atr_buffer_multiple,
        trigger_reject_lookback=args.trigger_reject_lookback,
    )

    api_key_names = TESTNET_API_KEY_ENV_NAMES + API_KEY_ENV_NAMES if args.testnet else API_KEY_ENV_NAMES
    api_secret_names = TESTNET_API_SECRET_ENV_NAMES + API_SECRET_ENV_NAMES if args.testnet else API_SECRET_ENV_NAMES
    api_key_name, api_key = _first_env_match(api_key_names)
    api_secret_name, api_secret = _first_env_match(api_secret_names)
    base_url = args.base_url or ("https://demo-fapi.binance.com" if args.testnet else None)
    client = BinanceClient(
        market="futures",
        base_url=base_url,
        timeout=args.timeout,
        retries=args.retries,
        api_key=api_key,
        api_secret=api_secret,
        rate_limit_rpm=args.rate_limit_rpm if args.rate_limit_rpm > 0 else None,
    )
    if args.auth_check:
        return print_auth_check(
            client=client,
            args=args,
            api_key_name=api_key_name,
            api_secret_name=api_secret_name,
        )
    if not client.api_key or not client.api_secret:
        print(
            "Auto-trader needs API keys: set BINANCE_API_KEY and BINANCE_API_SECRET in .env or environment.",
            file=sys.stderr,
        )
        return 2
    return run_auto_trader(client, args, settings)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Binance futures coins setting up before long or short conditional triggers.",
    )
    parser.add_argument("--quote", default="USDT", help="Futures quote asset to scan, e.g. USDT, USDC, or ALL.")
    parser.add_argument("--interval", help="Single kline interval to scan, e.g. 15m, 1h, 4h. Overrides --timeframes.")
    parser.add_argument("--timeframes", default="15m,1h,4h", help="Comma-separated intervals to scan. Default: 15m,1h,4h.")
    parser.add_argument("--history", type=int, default=120, help="Klines to fetch per symbol.")
    parser.add_argument("--top", type=int, default=0, help="Scan the top N symbols by 24h quote volume. Use 0 for all.")
    parser.add_argument("--limit", type=int, default=15, help="Maximum number of signals to print.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Binance kline requests.")
    parser.add_argument("--min-rr", type=float, default=1.2, help="Minimum measured reward/risk to show.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum setup score to show.")
    parser.add_argument("--min-quote-volume", type=float, default=1_000_000, help="Minimum 24h quote volume.")
    parser.add_argument("--min-trades", type=int, default=2_000, help="Minimum 24h trade count.")
    parser.add_argument("--min-24h-range-pct", type=float, default=2.0, help="Minimum 24h high-low range, percent.")
    parser.add_argument("--min-book-depth", type=float, default=25_000, help="Minimum bid and ask quote depth near mid price.")
    parser.add_argument("--book-depth-pct", type=float, default=1.0, help="Order-book depth window around mid price, percent.")
    parser.add_argument("--max-spread-bps", type=float, default=25.0, help="Maximum best bid/ask spread in basis points.")
    parser.add_argument("--book-limit", type=int, default=50, help="Order-book levels to request per symbol.")
    parser.add_argument("--skip-book-filter", action="store_true", help="Disable order-book depth and spread filters.")
    parser.add_argument("--min-open-interest-notional", type=float, default=5_000_000, help="Minimum estimated futures open-interest notional.")
    parser.add_argument("--skip-oi-filter", action="store_true", help="Disable the open-interest notional filter.")
    parser.add_argument("--include-dead", action="store_true", help="Disable liquidity/activity filters and scan every trading perpetual.")
    parser.add_argument("--symbols", help="Comma-separated symbols to scan instead of auto-discovery.")
    parser.add_argument("--symbols-file", type=Path, help="File with one symbol per line to scan instead of auto-discovery.")
    parser.add_argument("--include-watch", action="store_true", help="Compatibility alias; pre-breakout setups are included by default.")
    parser.add_argument("--include-confirmed", action="store_true", help="Also show coins that already confirmed above resistance.")
    parser.add_argument("--include-rejected", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--closed-candles-only", action="store_true", help="Ignore the currently forming candle.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")
    parser.add_argument("--verbose", action="store_true", help="Print per-symbol request failures.")
    parser.add_argument("--base-url", help="Override Binance USD-M futures base URL.")
    parser.add_argument("--testnet", action="store_true", help="Use Binance USD-M demo/testnet REST URL and prefer testnet env keys.")
    parser.add_argument("--auth-check", action="store_true", help="Check which API key is loaded and whether signed futures account access works.")
    parser.add_argument("--auth-check-trade", action="store_true", help="With --auth-check, also validate TRADE permission using Binance's test order endpoint.")
    parser.add_argument("--auth-check-symbol", default="BTCUSDT", help="Symbol used for --auth-check-trade validation.")
    parser.add_argument("--auth-check-quantity", default="0.001", help="Quantity used for --auth-check-trade validation.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient request errors.")
    parser.add_argument("--rate-limit-rpm", type=float, default=1100.0, help="Max Binance requests per minute. Stays under Binance's 2400/min IP cap so scans do not get 429'd. 0 disables.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to .env file with Binance API credentials.")

    parser.add_argument("--scan-interval-minutes", type=int, default=3, help="How often the auto-trader scans for new opportunities. Default 3 - a slow scan discovers fast breakouts after the move is over.")
    parser.add_argument("--live-orders", action="store_true", help="Deprecated/no-op: the auto-trader always trades live. Use --testnet for a safe environment.")
    parser.add_argument("--order-count", type=int, default=3, help="Number of best unique-symbol signals to order.")
    parser.add_argument("--max-concurrent-orders", type=int, default=0, help="Maximum number of entries that can be active (placed or in open position) at the same time. 0 = unlimited.")
    parser.add_argument("--queue-size", type=int, default=0, help="How many coins to arm and watch for a breakout at once. 0 (default) = watch all qualifying candidates. Independent of --max-concurrent-orders.")
    parser.add_argument("--order-notional", type=float, default=0.0, help="USDT position notional per entry order.")
    parser.add_argument("--order-margin", type=float, default=0.0, help="Approximate USDT margin per entry order. Position notional is margin multiplied by --leverage. Ignored when --sizing-mode auto sets margin from current equity.")
    parser.add_argument("--sizing-mode", choices=["fixed", "auto"], default="fixed", help="fixed = use --order-margin per trade; auto = size each trade as %% of current wallet equity (55%% on <$25, 45%% on $25-100, 32%% on $100-500, 22%% on $500-2.5k, 15%% on $2.5k-10k, 10%% above) with a drawdown haircut. Designed for micro-account growth; backtested 6700x vs moonshot's 320x but with 54%% peak drawdown.")
    parser.add_argument("--equity-peak-file", default=".equity_peak.json", help="Path to JSON state file storing the running equity peak for the --sizing-mode auto drawdown haircut. Reset by deleting this file.")
    parser.add_argument("--leverage", type=int, default=0, help="Set initial leverage for each ordered symbol before placing the entry order.")
    parser.add_argument("--dynamic-leverage", action="store_true", help="Scale leverage per coin from ATR volatility and stop distance. --leverage sets the base (default 10 when enabled).")
    parser.add_argument("--max-sl-loss-pct", type=float, default=35.0, help="Maximum leveraged loss percent on margin if the stop loss is hit. Use 0 to disable leverage-aware stop tightening.")
    parser.add_argument("--margin-type", choices=["ISOLATED", "CROSSED"], help="Set symbol margin type before placing the entry order.")
    parser.add_argument("--hedge-mode", action="store_true", help="Send positionSide LONG/SHORT for Binance Hedge Mode accounts.")
    parser.add_argument("--order-working-type", choices=["MARK_PRICE", "CONTRACT_PRICE"], default="MARK_PRICE", help="Trigger price type for conditional orders.")
    parser.add_argument("--order-price-protect", action="store_true", help="Enable Binance priceProtect on conditional orders.")
    parser.add_argument("--entry-mode", choices=["SMART_RETEST", "RETEST_LIMIT", "STOP_MARKET"], default="SMART_RETEST", help="Entry execution mode. SMART_RETEST watches trigger, waits for retest, then falls back to market.")
    parser.add_argument("--entry-pullback-pct", type=float, default=0.5, help="For RETEST_LIMIT, place the entry limit this percent better than the trigger.")
    parser.add_argument("--retest-timeout-seconds", type=float, default=300.0, help="For SMART_RETEST, seconds to wait for retest before entering at market.")
    parser.add_argument("--max-market-deviation-pct", type=float, default=1.5, help="For SMART_RETEST market fallback: skip entry if price has moved more than this percent beyond the trigger. Default 1.5.")
    parser.add_argument("--no-market-fallback", action="store_true", help="For SMART_RETEST, never place a market order after timeout. Keep watching for the retest limit entry indefinitely.")
    parser.add_argument("--entry-stale-minutes", type=float, default=30.0, help="Drop unplaced triggered entries, or cancel LIMIT entries, after this many minutes. Prevents old queued signals and never-filling orders from blocking later opportunities. 0 = off.")
    parser.add_argument("--rotation-auto-cut-loss", action="store_true", help="Auto-approve cutting a losing position to make room for an exploder, instead of waiting on the interactive Windows popup. Required for headless / VPS deployments. Profitable rotations are always automatic regardless of this flag.")
    parser.add_argument("--equity-peak-reset-pct", type=float, default=15.0, help="Auto-reset .equity_peak.json when wallet equity changes by more than this percent in one scan tick (interpreted as a manual withdrawal or deposit, not a trade loss). Prevents drawdown-haircut sizing from triggering after you move money in or out. 0 = off (peak only resets on manual file deletion).")
    parser.add_argument("--adaptive-entry", action="store_true", help="Pick the entry execution mode per coin from its market regime: abnormal volume -> instant market entry, strong trend -> trailing retest, choppy -> retest, illiquid/meme -> strict retest. Also ranks candidates by momentum.")
    parser.add_argument("--skip-entry-regimes", default="", help="Comma-separated adaptive entry regimes to skip live, e.g. TRAILING_RETEST. Default: include all.")
    parser.add_argument("--ml-rank-model", type=Path, help="Live-compatible JSON model artifact used to rank qualifying signals. Ranking only; it does not hard-filter trades.")
    parser.add_argument("--ml-rank-score", choices=["pwin", "expected-r", "tail", "not-bad", "composite"], default="tail", help="Model score used by --ml-rank-model. Default: tail.")
    parser.add_argument("--btc-market-guards", action="store_true", help="Apply the backtested BTC regime guards live: block INSTANT entries in weak BTC conditions and allow only STRICT_RETEST in hostile BTC conditions.")
    parser.add_argument("--btc-guard-interval", default="1h", help="BTC timeframe used for --btc-market-guards. Default: 1h.")
    parser.add_argument("--btc-ema-candles", type=int, default=72, help="BTC EMA length for --btc-market-guards. Default: 72.")
    parser.add_argument("--btc-momentum-candles", type=int, default=72, help="BTC momentum lookback for --btc-market-guards. Default: 72.")
    parser.add_argument("--instant-guard-momentum-pct", type=float, default=-2.0, help="Minimum BTC momentum for INSTANT entries when --btc-market-guards is enabled. Default: -2.")
    parser.add_argument("--instant-guard-ema-slack-pct", type=float, default=1.5, help="Allowed BTC close distance below EMA for INSTANT entries when --btc-market-guards is enabled. Default: 1.5.")
    parser.add_argument("--hostile-momentum-pct", type=float, default=0.0, help="BTC momentum below this is hostile when --btc-market-guards is enabled. Default: 0.")
    parser.add_argument("--hostile-ema-slack-pct", type=float, default=0.0, help="BTC close below EMA by this slack is hostile when --btc-market-guards is enabled. Default: 0.")
    parser.add_argument("--two-sided-entry", action="store_true", help="For coiling coins with no clear direction, arm a breakout bracket on BOTH sides. Whichever side breaks out first enters; the opposite side is cancelled. SMART_RETEST live mode only.")
    parser.add_argument("--detector", choices=["simple", "squeeze"], default="simple", help="simple = high-recall long-only breakout detector (default); squeeze = the original pre-breakout/short detector.")
    parser.add_argument("--dynamic-sl", action="store_true", help="Reposition the stop loss on open positions as new support/resistance levels form.")
    parser.add_argument("--sl-update-interval-seconds", type=float, default=300.0, help="How often (seconds) to re-evaluate and reposition the dynamic stop loss. Default 300.")
    parser.add_argument("--sl-lookback", type=int, default=20, help="Number of candles to look back when computing support/resistance for dynamic SL. Default 20.")
    parser.add_argument("--breakeven-trigger-r", type=float, default=1.5, help="Once unrealized profit reaches this R multiple, ratchet the stop to breakeven + a small profit lock. 0 = off. Default 1.5 was the worst-case-best in a 12-run sweep.")
    parser.add_argument("--breakeven-offset-pct", type=float, default=0.1, help="When the breakeven trigger fires, the new stop sits this percent past entry (locks tiny profit + covers fees).")
    parser.add_argument("--exhaustion-exit", action="store_true", help="Monitor open positions and market-close once a trade has reached +0.5R and then shows peak rejection or 4 closed candles with no new high. NOTE: a 4-window sweep showed the hardcoded 4-candle stall is too tight for breakout consolidations (cut avg-win in half, -96%% on equity). Use --stagnation-after-r/--stagnation-candles instead.")
    parser.add_argument("--exhaustion-lookback", type=int, default=80, help="Closed candles fetched per monitored position for --exhaustion-exit.")
    parser.add_argument("--stagnation-after-r", type=float, default=0.0, help="Stagnation exit: once unrealized profit reaches this R multiple, exit if no new favourable extreme appears for --stagnation-candles bars. 0 = off. Parameterized replacement for --exhaustion-exit (no rejection-candle path, configurable stall). Validated default: 0.5.")
    parser.add_argument("--stagnation-candles", type=int, default=12, help="Stagnation exit: closed candles of no new favourable extreme before market-closing. Only fires after --stagnation-after-r is reached. 12 was the worst-case-best in a 4-window sweep (W3 +19831%% vs +11127%% with no stagnation).")
    parser.add_argument("--stagnation-lookback", type=int, default=80, help="Closed candles fetched per monitored position for --stagnation-after-r.")
    parser.add_argument("--no-exits", action="store_true", help="Only place entry orders; do not place stop-loss/take-profit exit algo orders.")
    parser.add_argument("--manage-exits", action="store_true", help="Place deferred TP/SL/trailing exits after pending entries become open positions.")
    parser.add_argument("--watch-exits", action="store_true", help="Keep polling with --manage-exits until all pending exits are placed or timeout is reached.")
    parser.add_argument("--no-auto-manage-exits", action="store_true", help="After live entry placement, do not automatically wait and attach deferred exits.")
    parser.add_argument("--exit-state-file", type=Path, default=Path(".pending_exit_orders.json"), help="JSON file used to store deferred exit plans.")
    parser.add_argument("--entry-state-file", type=Path, default=Path(".pending_entry_orders.json"), help="JSON file used to store managed smart-retest entry plans.")
    parser.add_argument("--ml-context-file", type=Path, default=Path(".ml_symbol_context.json"), help="JSON file storing live rolling symbol outcome context for ML ranking.")
    parser.add_argument("--exit-poll-seconds", type=float, default=5.0, help="Polling interval for --watch-exits.")
    parser.add_argument("--exit-heartbeat-seconds", type=float, default=60.0, help="Minimum seconds between unchanged exit-manager status lines.")
    parser.add_argument("--exit-watch-timeout", type=float, default=0.0, help="Seconds to watch for entries. Use 0 for no timeout.")
    parser.add_argument("--tp-count", type=int, default=1, help="Number of partial take-profit orders per entry.")
    parser.add_argument("--tp-splits", help="Comma-separated take-profit quantity percentages, e.g. 40,30,20. With --trailing-stop the remainder becomes the trailing runner.")
    parser.add_argument("--trailing-stop", action="store_true", help="Add a trailing-stop-market algo order for the runner quantity.")
    parser.add_argument("--trailing-callback-pct", type=float, default=1.2, help="Trailing stop callback rate, percent. Binance allows 0.1 to 10. Default 1.2 matches the backtest's validated +11127%% baseline; 1.0 (the old live default) was 20%% tighter than backtest and silently haircut avg-win.")
    parser.add_argument("--trailing-quantity-pct", type=float, default=50.0, help="Runner quantity percent reserved for trailing stop when --tp-splits is not set. A 27-run sweep showed a bigger runner (50%%) compounds far better - a breakout's edge is in the few trades that run.")
    parser.add_argument("--smart-tp", action="store_true", help="Adapt the target, TP splits, and runner size from each signal's conviction.")
    parser.add_argument("--smart-tp-max-target-multiplier", type=float, default=2.5, help="Maximum multiplier applied to the signal target when --smart-tp is enabled.")
    parser.add_argument("--smart-tp-min-runner-pct", type=float, default=20.0, help="Minimum trailing runner percent used by --smart-tp.")
    parser.add_argument("--smart-tp-max-runner-pct", type=float, default=55.0, help="Maximum trailing runner percent used by --smart-tp.")
    parser.add_argument("--recv-window", type=int, default=5000, help="Signed order recvWindow in milliseconds.")
    parser.add_argument("--client-order-prefix", default="bd", help="Prefix for generated client order IDs.")
    parser.add_argument("--allow-duplicate-symbol-orders", action="store_true", help="Allow more than one order for the same symbol.")
    parser.add_argument("--skip-pre-order-recheck", action="store_true", help="Do not re-fetch and re-evaluate the signal immediately before live order placement.")

    parser.add_argument("--resistance-lookback", type=int, default=40, help="Candles used to define resistance.")
    parser.add_argument("--squeeze-lookback", type=int, default=20, help="Candles used to define consolidation range.")
    parser.add_argument("--volume-lookback", type=int, default=20, help="Candles used for average volume.")
    parser.add_argument("--min-breakout-pct", type=float, default=0.15, help="Minimum close above resistance, percent.")
    parser.add_argument("--max-extension-pct", type=float, default=3.5, help="Reject late moves extended above resistance, percent.")
    parser.add_argument("--prior-break-tolerance-pct", type=float, default=0.4, help="Previous close tolerance above resistance, percent.")
    parser.add_argument("--max-trigger-distance-pct", type=float, default=2.0, help="Maximum distance from current price to conditional trigger, percent.")
    parser.add_argument("--max-shakeout-distance-pct", type=float, default=4.0, help="Maximum distance from spring/upthrust reclaim to conditional trigger, percent.")
    parser.add_argument("--max-pre-trigger-move-pct", type=float, default=3.0, help="Reject setups already extended this far from the recent base, percent.")
    parser.add_argument("--entry-buffer-pct", type=float, default=0.10, help="Trigger buffer beyond resistance/support, percent.")
    parser.add_argument("--entry-atr-buffer-multiple", type=float, default=0.0, help="Use at least this ATR multiple as breakout trigger buffer.")
    parser.add_argument("--trigger-reject-lookback", type=int, default=4, help="Reject setups if the trigger was swept and rejected within this many candles.")
    parser.add_argument("--stop-buffer-pct", type=float, default=0.20, help="Invalidation stop buffer beyond support/resistance or sweep wick, percent.")
    parser.add_argument("--target-range-multiple", type=float, default=1.5, help="Target distance as a multiple of the consolidation range. Keep it high enough that reward/risk stays above --min-rr, or signals get filtered out.")
    parser.add_argument("--min-sweep-pct", type=float, default=0.15, help="Minimum wick sweep beyond support/resistance, percent.")
    parser.add_argument("--max-fakeout-close-position", type=float, default=0.55, help=argparse.SUPPRESS)
    parser.add_argument("--min-volume-ratio", type=float, default=1.45, help="Latest volume versus prior average for confirmed breakouts and shakeouts.")
    parser.add_argument("--min-watch-volume-ratio", type=float, default=0.8, help="Latest volume versus prior average for pre-breakout setups.")
    parser.add_argument("--min-candle-quote-volume", type=float, default=25_000, help="Minimum average quote volume per 15m candle, scaled by timeframe.")
    parser.add_argument("--min-close-position", type=float, default=0.62, help="Close location within latest candle range, 0-1.")
    parser.add_argument("--max-compression-pct", type=float, default=18.0, help="Max consolidation range as percent of price.")
    args = parser.parse_args(argv)
    # The bot runs as a continuous live auto-trader; there is no scan-only path.
    args.place_orders = True
    args.live_orders = True
    # Always detect confirmed breakouts so a move is caught on its first candle, not missed.
    args.include_confirmed = True
    if args.order_notional <= 0 and args.order_margin <= 0 and args.sizing_mode != "auto":
        parser.error("--order-notional or --order-margin is required (or use --sizing-mode auto)")
    if args.scan_interval_minutes <= 0:
        parser.error("--scan-interval-minutes must be greater than 0")
    if 0 < args.queue_size < args.max_concurrent_orders:
        args.queue_size = args.max_concurrent_orders
    if args.dynamic_leverage and args.leverage <= 0:
        args.leverage = 10
    if args.order_margin > 0 and args.leverage <= 0:
        if args.dynamic_leverage:
            args.leverage = 10
        else:
            parser.error("--order-margin requires --leverage")
    if args.sizing_mode == "auto" and args.leverage <= 0:
        args.leverage = 10  # auto needs a leverage to compute notional; 10 is the sensible default
    if args.leverage and not 1 <= args.leverage <= 125:
        parser.error("--leverage must be between 1 and 125")
    if args.max_sl_loss_pct < 0:
        parser.error("--max-sl-loss-pct must be zero or greater")
    if args.order_count <= 0:
        parser.error("--order-count must be greater than 0")
    if args.entry_pullback_pct < 0:
        parser.error("--entry-pullback-pct must be non-negative")
    if args.retest_timeout_seconds < 0:
        parser.error("--retest-timeout-seconds must be non-negative")
    if args.trigger_reject_lookback < 0:
        parser.error("--trigger-reject-lookback must be non-negative")
    if args.tp_count <= 0:
        parser.error("--tp-count must be greater than 0")
    if args.trailing_callback_pct < 0.1 or args.trailing_callback_pct > 10:
        parser.error("--trailing-callback-pct must be between 0.1 and 10")
    if args.trailing_quantity_pct <= 0 or args.trailing_quantity_pct >= 100:
        parser.error("--trailing-quantity-pct must be greater than 0 and less than 100")
    if args.smart_tp and args.tp_splits:
        parser.error("--smart-tp cannot be combined with manual --tp-splits")
    if args.smart_tp_max_target_multiplier < 1:
        parser.error("--smart-tp-max-target-multiplier must be at least 1")
    if not 0 <= args.smart_tp_min_runner_pct < 100 or not 0 <= args.smart_tp_max_runner_pct < 100:
        parser.error("--smart-tp runner bounds must be from 0 to less than 100")
    if args.smart_tp_min_runner_pct > args.smart_tp_max_runner_pct:
        parser.error("--smart-tp-min-runner-pct must be <= --smart-tp-max-runner-pct")
    if args.btc_ema_candles <= 1:
        parser.error("--btc-ema-candles must be greater than 1")
    if args.btc_momentum_candles <= 0:
        parser.error("--btc-momentum-candles must be greater than 0")
    if args.exhaustion_lookback < 10:
        parser.error("--exhaustion-lookback must be at least 10")
    args.skip_entry_regimes = _parse_entry_regime_set(args.skip_entry_regimes, parser)
    args.tp_splits_pct = []
    args.trailing_runner_pct = 0.0
    if not args.no_exits:
        args.tp_splits_pct, args.trailing_runner_pct = _resolve_exit_splits(args, parser)
    return args


def scan_symbols(
    client: BinanceClient,
    symbols: list[SymbolInfo],
    args: argparse.Namespace,
    settings: BreakoutSettings,
    intervals: list[str],
    now_ms: int,
) -> tuple[list[BreakoutSignal], list[str]]:
    signals: list[BreakoutSignal] = []
    failures: list[str] = []
    signal_contexts: dict[tuple[str, str], dict[str, float]] = {}
    context_lock = threading.Lock()
    history_limit = args.history + 1 if args.closed_candles_only else args.history
    interval_ms_by_interval = {interval: interval_to_ms(interval) for interval in intervals}

    def analyze(symbol_info: SymbolInfo, interval: str) -> BreakoutSignal | None:
        klines = client.klines(symbol_info.symbol, interval, history_limit)
        candles = candles_from_klines(klines)
        if args.closed_candles_only and candles and candles[-1].close_time > now_ms:
            candles = candles[:-1]
        if args.detector == "simple":
            signal = detect_long_breakout(
                symbol=symbol_info.symbol,
                candles=candles,
                quote_volume_24h=symbol_info.quote_volume_24h,
                interval_ms=interval_ms_by_interval[interval],
                interval=interval,
                range_pct_24h=symbol_info.range_pct_24h,
                settings=settings,
                now_ms=now_ms,
            )
        else:
            signal = evaluate_breakout(
                symbol=symbol_info.symbol,
                candles=candles,
                quote_volume_24h=symbol_info.quote_volume_24h,
                interval_ms=interval_ms_by_interval[interval],
                interval=interval,
                trade_count_24h=symbol_info.trade_count_24h,
                range_pct_24h=symbol_info.range_pct_24h,
                price_change_pct_24h=symbol_info.price_change_pct_24h,
                book_min_depth=symbol_info.book_min_depth,
                open_interest_notional=symbol_info.open_interest_notional,
                settings=settings,
                include_confirmed=args.include_confirmed,
                include_rejected=args.include_rejected,
                now_ms=now_ms,
            )
        if signal is not None and getattr(args, "ml_rank_model_data", None):
            context = _live_market_signal_context(signal, candles, args)
            with context_lock:
                signal_contexts[(signal.symbol, signal.interval)] = context
        return signal

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        future_by_label = {
            executor.submit(analyze, symbol_info, interval): f"{symbol_info.symbol}@{interval}"
            for interval in intervals
            for symbol_info in symbols
        }
        for future in as_completed(future_by_label):
            label = future_by_label[future]
            try:
                signal = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad symbol should not kill a scan.
                failures.append(f"{label}: {exc}")
                if args.verbose:
                    print(f"Failed {label}: {exc}", file=sys.stderr)
                continue
            if signal:
                signals.append(signal)

    args._live_ml_signal_contexts = signal_contexts
    return signals, failures


def print_report(
    signals: list[BreakoutSignal],
    universe: SymbolUniverse,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    stats = universe.stats
    dead_filtered = (
        stats.filtered_low_volume
        + stats.filtered_low_trades
        + stats.filtered_low_range
        + stats.filtered_thin_book
        + stats.filtered_wide_spread
        + stats.order_book_failures
        + stats.filtered_low_open_interest
    )
    filters = _filter_summary(args)
    print("Binance USD-M futures pre-breakout scan")
    print(
        f"quote={args.quote.upper()}  timeframes={','.join(_resolve_intervals(args))}  active={len(universe.symbols)}/{stats.perpetual_symbols}  "
        f"dead-filtered={dead_filtered}  signals={len(signals)}"
    )
    print(
        f"filters: 24h vol >= {_compact_money(float(filters['min_quote_volume']))}  "
        f"trades >= {int(filters['min_trades']):,}  "
        f"24h range >= {float(filters['min_24h_range_pct']):.2f}%  "
        f"book >= {_compact_money(float(filters['min_book_depth']))}/{args.book_depth_pct:.1f}%  "
        f"spread <= {args.max_spread_bps:.1f}bps  "
        f"OI >= {_compact_money(float(filters['min_open_interest_notional']))}  "
        f"flow >= {_compact_money(args.min_candle_quote_volume)}/15m  "
        f"max trigger distance <= {args.max_trigger_distance_pct:.2f}%  "
        f"min R:R >= {args.min_rr:.2f}"
    )
    print(
        f"filtered: volume={stats.filtered_low_volume}  trades={stats.filtered_low_trades}  "
        f"range={stats.filtered_low_range}  thin-book={stats.filtered_thin_book}  "
        f"wide-spread={stats.filtered_wide_spread}  low-oi={stats.filtered_low_open_interest}  "
        f"book-errors={stats.order_book_failures}  oi-errors={stats.open_interest_failures}"
    )
    if not signals:
        print("No pre-breakout candidates matched the current filters.")
        print("Tip: raise --max-trigger-distance-pct or lower --min-watch-volume-ratio for earlier, noisier setups.")
        if dead_filtered:
            print("Tip: lower --min-quote-volume, --min-trades, or --min-24h-range-pct to include quieter contracts.")
            print("Tip: lower --min-book-depth or add --skip-book-filter to include thinner books.")
            print("Tip: lower --min-open-interest-notional or add --skip-oi-filter to include smaller futures markets.")
        if failures:
            print(f"{len(failures)} symbols failed. Re-run with --verbose to inspect them.")
        return

    headers = ["#", "tf", "symbol", "side", "setup", "grade", "order", "score", "price", "dist%", "trigger", "stop", "target", "risk%", "rr", "volx", "flow", "OI", "book", "why"]
    rows = [
        [
            str(index),
            signal.interval,
            signal.symbol,
            signal.side,
            signal.status + ("*" if signal.open_candle else ""),
            _grade(signal),
            _display_order(signal, args),
            f"{signal.score:.1f}",
            _price(signal.close),
            f"{signal.distance_to_trigger_pct * 100:.2f}",
            _price(signal.trigger_price),
            _price(signal.stop_price),
            _price(signal.target_price),
            f"{signal.risk_pct * 100:.2f}",
            f"{signal.reward_risk:.2f}",
            f"{signal.volume_ratio:.2f}",
            _compact_money(signal.avg_quote_volume),
            _compact_money(signal.open_interest_notional),
            _compact_money(signal.book_min_depth),
            _why(signal),
        ]
        for index, signal in enumerate(signals, start=1)
    ]
    _print_table(headers, rows, right_align={"#", "score", "price", "dist%", "trigger", "stop", "target", "risk%", "rr", "volx", "flow", "OI", "book"})
    print("* means the latest candle is still forming; volume is projected for the full candle.")
    if args.entry_mode == "SMART_RETEST":
        print("SMART_RETEST watches the breakout trigger first, then waits for retest; timeout falls back to market.")
    else:
        print("Orders are conditional levels only. LONG triggers when mark >= trigger; SHORT triggers when mark <= trigger.")
    print("Setups: PRE_BREAKOUT/SPRING for longs, PRE_BREAKDOWN/UPTHRUST for shorts.")
    if args.include_confirmed:
        print("Optional rows: BREAKOUT/BREAKDOWN are already confirmed moves.")
    if failures:
        print(f"{len(failures)} symbols failed. Re-run with --verbose to inspect them.")


def _display_order(signal: BreakoutSignal, args: argparse.Namespace) -> str:
    if args.entry_mode == "SMART_RETEST":
        return "BUY SMART_RETEST" if signal.side == "LONG" else "SELL SMART_RETEST"
    if args.entry_mode == "RETEST_LIMIT":
        return "BUY STOP_LIMIT" if signal.side == "LONG" else "SELL STOP_LIMIT"
    return signal.order_type


def place_best_orders(
    client: BinanceClient,
    signals: list[BreakoutSignal],
    args: argparse.Namespace,
    settings: BreakoutSettings,
    account: dict[str, object] | None = None,
) -> tuple[list[OrderExecution], list[str]]:
    results: list[OrderExecution] = []
    failures: list[str] = []

    if args.adaptive_entry and args.detector == "squeeze":
        live_signals: list[BreakoutSignal] = []
        for candidate in signals:
            dead_reason = _dead_coin_reason(candidate)
            if dead_reason:
                failures.append(f"{candidate.symbol}@{candidate.interval}: skipped, dead coin ({dead_reason})")
            else:
                live_signals.append(candidate)
        signals = live_signals

    btc_guard_point = _current_btc_guard_point(client, args) if args.btc_market_guards else None
    if args.btc_market_guards and btc_guard_point is None:
        failures.append("BTC market guard: BTCUSDT trend unavailable; treating market as hostile")

    if args.skip_entry_regimes or args.btc_market_guards:
        tradable_signals: list[BreakoutSignal] = []
        for candidate in signals:
            entry_regime = _classify_entry_regime(candidate)
            if entry_regime in args.skip_entry_regimes:
                failures.append(f"{candidate.symbol}@{candidate.interval}: skipped, entry regime {entry_regime}")
            elif btc_reason := _btc_guard_reject_reason(entry_regime, btc_guard_point, args):
                failures.append(f"{candidate.symbol}@{candidate.interval}: skipped, {btc_reason}")
            else:
                tradable_signals.append(candidate)
        signals = tradable_signals

    selected = _select_order_signals(signals, args)
    if not selected:
        return results, failures + ["No eligible signals were available for order placement."]

    rules = trading_rules_from_exchange_info(client.exchange_info())
    mode = "LIVE" if args.live_orders else "TEST"
    order_notional = _order_notional(args)
    requested_margin = _requested_margin(args, order_notional)

    auto_margin = 0.0
    if args.sizing_mode == "auto":
        if account is None:
            try:
                account = client.account_info(recv_window=args.recv_window)
            except BinanceClientError as exc:
                return results, failures + [f"Auto-sizing: account info unavailable ({exc}), no orders placed."]
        auto_margin = _resolve_auto_order_margin(args, account or {})
        if auto_margin <= 0:
            return results, failures + ["Auto-sizing: wallet equity is zero or missing; no orders placed this scan."]
        print(f"Auto-sizing: equity={_current_equity(account or {}):.2f} USDT  margin/trade={auto_margin:.2f} USDT  ({(auto_margin / max(_current_equity(account or {}), 1e-9)) * 100:.1f}% of equity)")

    if args.live_orders and args.entry_mode == "SMART_RETEST":
        pending_symbols = {
            str(item.get("symbol", ""))
            for item in _load_pending_entry_plans(args.entry_state_file)
        }
        skipped = [s for s in selected if s.symbol in pending_symbols]
        selected = [s for s in selected if s.symbol not in pending_symbols]
        for s in skipped:
            failures.append(f"{s.symbol}@{s.interval}: skipped, already has a pending managed entry in {args.entry_state_file}")

    for index, signal in enumerate(selected, start=1):
        if args.live_orders and not args.skip_pre_order_recheck:
            fresh_signal = _fresh_order_signal(client, signal, args, settings)
            if not fresh_signal or fresh_signal.side != signal.side:
                failures.append(f"{signal.symbol}@{signal.interval}: skipped, setup failed immediate pre-order recheck")
                continue
            signal = fresh_signal
        rule = rules.get(signal.symbol)
        if not rule:
            failures.append(f"{signal.symbol}: exchange trading rules were not found")
            continue

        effective_leverage = _dynamic_leverage(signal.atr_pct, signal.risk_pct, args.leverage, conviction=_momentum_score(signal)) if args.dynamic_leverage else args.leverage
        if auto_margin > 0:
            effective_margin = auto_margin
            effective_notional = auto_margin * effective_leverage
        elif args.dynamic_leverage and args.order_margin > 0:
            effective_notional = args.order_margin * effective_leverage
            effective_margin = args.order_margin
        else:
            effective_notional = order_notional
            effective_margin = requested_margin
        signal = _with_leverage_capped_stop(signal, effective_leverage, args.max_sl_loss_pct)

        # Always classify regime (matching backtest behaviour); --adaptive-entry
        # is no longer a gate on classification — only on the optional skip
        # filter and BTC guards, which check themselves below. Live used to
        # force every signal to "RETEST" without --adaptive-entry, which is
        # the worst-performing regime in the validated dataset
        # (INSTANT carried +1145 USDT of +1182 in the W3 baseline).
        entry_regime = _classify_entry_regime(signal)
        if entry_regime in args.skip_entry_regimes:
            failures.append(f"{signal.symbol}@{signal.interval}: skipped after recheck, entry regime {entry_regime}")
            continue
        if btc_reason := _btc_guard_reject_reason(entry_regime, btc_guard_point, args):
            failures.append(f"{signal.symbol}@{signal.interval}: skipped after recheck, {btc_reason}")
            continue

        if (
            args.live_orders
            and args.entry_mode == "SMART_RETEST"
            and args.two_sided_entry
            and _is_coiling_no_bias(signal)
        ):
            bracket = _bracket_signals(signal, settings)
            if bracket:
                bracket_id = f"{signal.symbol}-{int(time.time())}-{index}"
                placed_sides = 0
                for side_signal in bracket:
                    if _place_bracket_side(
                        signal=side_signal,
                        rule=rule,
                        index=index,
                        args=args,
                        mode=mode,
                        leverage=effective_leverage,
                        notional=effective_notional,
                        margin=effective_margin,
                        entry_regime=entry_regime,
                        bracket_id=bracket_id,
                        results=results,
                        failures=failures,
                    ):
                        placed_sides += 1
                if placed_sides:
                    print(f"  {signal.symbol}@{signal.interval} armed two-sided bracket ({placed_sides} side(s))")
                continue

        try:
            if args.live_orders and args.entry_mode != "SMART_RETEST":
                _apply_order_account_settings(client, signal, args, leverage=effective_leverage)
            plan = build_entry_order_plan(
                signal=signal,
                rule=rule,
                requested_notional=effective_notional,
                client_order_id=_client_order_id(signal, index, args.client_order_prefix),
                working_type=args.order_working_type,
                price_protect=args.order_price_protect,
                hedge_mode=args.hedge_mode,
                entry_mode="RETEST_LIMIT" if args.entry_mode == "SMART_RETEST" else args.entry_mode,
                entry_pullback_pct=args.entry_pullback_pct,
            )
            if args.live_orders and args.entry_mode == "SMART_RETEST":
                response = {"algoStatus": "WAIT_BREAKOUT"}
            else:
                response = _submit_order_plan(client, plan.payload, args)
        except (BinanceClientError, OrderPlanError) as exc:
            failures.append(f"{signal.symbol}@{signal.interval}: {exc}")
            continue

        results.append(_order_execution_from_plan(plan, response, mode, effective_margin, args, leverage=effective_leverage))

        exit_plans: list[ConditionalOrderPlan] = []
        if args.live_orders and args.entry_mode == "SMART_RETEST":
            if not args.no_exits:
                try:
                    tp_profile = _take_profit_profile_for_signal(signal, args)
                    exit_signal = tp_profile.signal
                    exit_plans = build_exit_order_plans(
                        signal=exit_signal,
                        rule=rule,
                        entry_quantity=plan.quantity,
                        stop_client_order_id=_child_client_order_id(plan.client_order_id, "sl"),
                        target_client_order_ids=[
                            _child_client_order_id(plan.client_order_id, f"tp{target_index}")
                            for target_index in range(1, len(tp_profile.tp_splits_pct) + 1)
                        ],
                        target_splits_pct=tp_profile.tp_splits_pct,
                        trailing_client_order_id=_child_client_order_id(plan.client_order_id, "trl") if args.trailing_stop else None,
                        trailing_callback_pct=args.trailing_callback_pct if args.trailing_stop else None,
                        trailing_quantity_pct=tp_profile.runner_pct,
                        working_type=args.order_working_type,
                        price_protect=args.order_price_protect,
                        hedge_mode=args.hedge_mode,
                    )
                except OrderPlanError as exc:
                    failures.append(f"{signal.symbol}@{signal.interval} exits: {exc}")
                    continue
            _save_pending_entry_plan(
                path=args.entry_state_file,
                signal=exit_signal if exit_plans else signal,
                entry_plan=plan,
                exit_plans=exit_plans,
                args=args,
                leverage=effective_leverage,
                entry_regime=entry_regime,
            )
            if args.adaptive_entry:
                print(f"  {signal.symbol}@{signal.interval} adaptive entry regime: {entry_regime}")
            if args.smart_tp and exit_plans:
                print(
                    f"  {signal.symbol}@{signal.interval} smart TP: "
                    f"target x{tp_profile.target_multiplier:.2f}, runner {tp_profile.runner_pct:.1f}%"
                )
            for exit_plan in exit_plans:
                results.append(_order_execution_from_plan(exit_plan, {"algoStatus": "DEFERRED_UNTIL_ENTRY_FILLS"}, mode, effective_margin, args, leverage=effective_leverage))
            continue

        if args.no_exits:
            continue
        try:
            tp_profile = _take_profit_profile_for_signal(signal, args)
            exit_signal = tp_profile.signal
            exit_plans = build_exit_order_plans(
                signal=exit_signal,
                rule=rule,
                entry_quantity=plan.quantity,
                stop_client_order_id=_child_client_order_id(plan.client_order_id, "sl"),
                target_client_order_ids=[
                    _child_client_order_id(plan.client_order_id, f"tp{target_index}")
                    for target_index in range(1, len(tp_profile.tp_splits_pct) + 1)
                ],
                target_splits_pct=tp_profile.tp_splits_pct,
                trailing_client_order_id=_child_client_order_id(plan.client_order_id, "trl") if args.trailing_stop else None,
                trailing_callback_pct=args.trailing_callback_pct if args.trailing_stop else None,
                trailing_quantity_pct=tp_profile.runner_pct,
                working_type=args.order_working_type,
                price_protect=args.order_price_protect,
                hedge_mode=args.hedge_mode,
            )
        except OrderPlanError as exc:
            failures.append(f"{signal.symbol}@{signal.interval} exits: {exc}")
            continue
        if args.live_orders:
            _save_pending_exit_plans(
                path=args.exit_state_file,
                signal=exit_signal,
                entry_plan=plan,
                exit_plans=exit_plans,
                args=args,
            )
            for exit_plan in exit_plans:
                results.append(_order_execution_from_plan(exit_plan, {"algoStatus": "DEFERRED_UNTIL_ENTRY_FILLS"}, mode, effective_margin, args, leverage=effective_leverage))
            continue
        for exit_plan in exit_plans:
            exit_response = _submit_order_plan(client, exit_plan.payload, args)
            results.append(_order_execution_from_plan(exit_plan, exit_response, mode, effective_margin, args, leverage=effective_leverage))

    return results, failures


def print_order_report(results: list[OrderExecution], failures: list[str], args: argparse.Namespace) -> None:
    mode = "LIVE" if args.live_orders else "TEST"
    entry_count = sum(1 for result in results if result.role == "ENTRY")
    if args.entry_mode == "SMART_RETEST":
        entry_label = "managed entries" if args.live_orders else "planned entries"
    else:
        entry_label = "submitted entries"
    print()
    print(f"Order placement mode: {mode}  requested entries={args.order_count}  {entry_label}={entry_count}  order rows={len(results)}")
    if results:
        headers = ["#", "role", "symbol", "tf", "side", "type", "trigger", "limit", "qty", "margin", "notional", "lev", "marginType", "status", "algoId", "clientId"]
        rows = [
            [
                str(index),
                result.role,
                result.symbol,
                result.interval,
                result.side,
                result.order_type,
                result.trigger_price,
                result.limit_price or "-",
                result.quantity,
                result.requested_margin,
                result.estimated_notional,
                result.leverage,
                result.margin_type,
                result.status,
                result.order_id or "-",
                result.client_order_id,
            ]
            for index, result in enumerate(results, start=1)
        ]
        _print_table(headers, rows, right_align={"#", "trigger", "limit", "qty", "margin", "notional"})
    if not args.live_orders:
        print("TEST mode only validated planned payloads locally, so nothing was placed. Add --live-orders to submit real orders.")
    if args.no_exits:
        print("Entry-only mode: --no-exits skipped take-profit and stop-loss algo orders.")
        if args.live_orders and args.entry_mode == "SMART_RETEST":
            print(
                "SMART_RETEST live mode: no entry is placed yet; the manager watches breakout, "
                "tries the retest limit, then falls back to market after timeout."
            )
    else:
        if args.trailing_stop:
            if args.smart_tp:
                print(
                    f"Exit orders: smart TP with {args.tp_count} partial TP level(s), close-all stop loss, "
                    f"target up to x{args.smart_tp_max_target_multiplier:.2f}, "
                    f"runner {args.smart_tp_min_runner_pct:.0f}-{args.smart_tp_max_runner_pct:.0f}% "
                    f"at {args.trailing_callback_pct:.2f}% callback."
                )
            else:
                print(
                    f"Exit orders: {len(args.tp_splits_pct)} partial TP(s), close-all stop loss, "
                    f"and {args.trailing_runner_pct:.2f}% trailing runner at {args.trailing_callback_pct:.2f}% callback."
                )
        if args.live_orders and args.entry_mode == "SMART_RETEST":
            print(
                "SMART_RETEST live mode: no entry is placed yet; the manager watches breakout, "
                "tries the retest limit, then falls back to market after timeout."
            )
        elif args.live_orders:
            print("Live exits are deferred until entry fills. Run: python .\\breakout_detector.py --manage-exits --watch-exits")
        print("Exit orders are not OCO-linked; cancel leftover exits manually if needed.")
    if failures:
        print("Order failures:")
        for failure in failures:
            print(f"- {failure}")


def _submit_order_plan(client: BinanceClient, payload: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    if args.live_orders:
        return client.place_algo_order(payload, recv_window=args.recv_window)
    return {"algoStatus": "LOCAL_VALIDATED"}


def _should_auto_manage_exits(args: argparse.Namespace) -> bool:
    return bool(
        args.live_orders
        and args.place_orders
        and not args.no_auto_manage_exits
        and (args.entry_mode == "SMART_RETEST" or not args.no_exits)
    )


def _has_pending_trade_work(args: argparse.Namespace) -> bool:
    return bool(_load_pending_entry_plans(args.entry_state_file) or _load_pending_exit_plans(args.exit_state_file))


def _heartbeat_due(last_status_time: float, args: argparse.Namespace) -> bool:
    heartbeat = max(args.exit_heartbeat_seconds, 0.0)
    return heartbeat > 0 and time.time() - last_status_time >= heartbeat


def _position_unrealized_pnl(position: dict[str, object]) -> float:
    for key in ("unrealizedProfit", "unRealizedProfit", "unrealizedPnl"):
        if position.get(key) is not None:
            return _safe_float(position.get(key))
    return 0.0


def _account_position(
    account: dict[str, object], symbol: str, side: str, hedge_mode: bool
) -> dict[str, object] | None:
    for position in account.get("positions", []):
        if not isinstance(position, dict) or str(position.get("symbol", "")) != symbol:
            continue
        amount = _safe_float(position.get("positionAmt"))
        if abs(amount) <= 0:
            continue
        if hedge_mode:
            if str(position.get("positionSide", "")) == side:
                return position
        elif (side == "LONG" and amount > 0) or (side == "SHORT" and amount < 0):
            return position
    return None


def _exit_algo_ids(item: dict[str, object]) -> list[str]:
    placed = item.get("placed_exit_client_order_ids", [])
    if isinstance(placed, list) and placed:
        return [str(client_id) for client_id in placed]
    ids: list[str] = []
    for plan in item.get("exit_plans", []):
        if isinstance(plan, dict) and plan.get("client_order_id"):
            ids.append(str(plan["client_order_id"]))
    return ids


def _remove_pending_entry(args: argparse.Namespace, item: dict[str, object]) -> None:
    target_id = str(item.get("entry_client_order_id", ""))
    pending = _load_pending_entry_plans(args.entry_state_file)
    kept = [it for it in pending if str(it.get("entry_client_order_id", "")) != target_id]
    _write_pending_entry_plans(args.entry_state_file, kept)


def _close_position(
    client: BinanceClient,
    item: dict[str, object],
    account: dict[str, object],
    args: argparse.Namespace,
    reason: str,
) -> bool:
    """Market-close an open position and cancel its leftover exit algo orders."""
    symbol = str(item.get("symbol", ""))
    side = str(item.get("side", ""))
    hedge_mode = bool(item.get("hedge_mode", False))
    position = _account_position(account, symbol, side, hedge_mode)
    if not position:
        print(f"Position close: {symbol} position not found on the account; skipping close.")
        return False
    quantity = str(position.get("positionAmt", "0")).lstrip("-").strip()
    if _safe_float(quantity) <= 0:
        return False
    try:
        exit_price = client.mark_price(symbol)
    except BinanceClientError:
        exit_price = 0.0
    payload: dict[str, str] = {
        "symbol": symbol,
        "side": "SELL" if side == "LONG" else "BUY",
        "type": "MARKET",
        "quantity": quantity,
    }
    if hedge_mode:
        payload["positionSide"] = side
    else:
        payload["reduceOnly"] = "true"
    try:
        client.place_order(payload, test=False, recv_window=args.recv_window)
    except BinanceClientError as exc:
        print(f"Position close: failed to close {symbol}: {exc}")
        return False
    if exit_price > 0:
        _record_symbol_r_from_close(args, item, position, exit_price)
    cancelled = 0
    for algo_id in _exit_algo_ids(item):
        try:
            client.cancel_algo_order(symbol, algo_id, recv_window=args.recv_window)
            cancelled += 1
        except BinanceClientError:
            pass
    print(f"Position close: closed {symbol} at market ({reason}); cancelled {cancelled} exit order(s).")
    return True


def _prompt_cut_losing_trade(symbol: str, pnl: float, exploding_symbol: str, args: argparse.Namespace | None = None) -> bool:
    """Decide whether to cut a losing trade to chase an exploding coin.

    Auto-approve ONLY when --rotation-auto-cut-loss is set. The backtest's
    rotation never cuts losing positions (it skips when no profitable
    candidate is available) and a 4x3 sweep showed cutting losers is a
    net negative on every window (-53% W1, -31% W2, -11% W3 on equity;
    -25% worst-case R). So:

    - Interactive Windows session: show the popup (operator decides).
    - Headless (no TTY): default to KEEP the position - matches backtest.
    - --rotation-auto-cut-loss: opt-in, cuts the loser (unvalidated path,
      kept as an escape hatch for operators who want it).
    """
    message = (
        f"{symbol} is in a LOSS of {pnl:.2f} USDT. "
        f"{exploding_symbol} is exploding and there is no free slot. "
        f"Cut {symbol} at a loss to chase {exploding_symbol}?"
    )
    auto_cut = bool(args and getattr(args, "rotation_auto_cut_loss", False))
    headless = not sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else True
    if auto_cut:
        print(f"Rotation auto-cut (--rotation-auto-cut-loss set): cutting {symbol} at {pnl:.2f} USDT to chase {exploding_symbol}.")
        return True
    if headless:
        print(f"Rotation: keeping {symbol} (headless run, no operator to answer; --rotation-auto-cut-loss not set).")
        return False
    print(
        f"\n*** ROTATION DECISION NEEDED ***\n{message}\n"
        f"(Windows prompt shown - {ROTATION_PROMPT_TIMEOUT}s to answer; no answer = keep {symbol})\n"
    )
    if os.name != "nt":
        return False
    safe = message.replace("'", "''")
    ps_command = (
        "$wshell = New-Object -ComObject Wscript.Shell; "
        f"$answer = $wshell.Popup('{safe}', {ROTATION_PROMPT_TIMEOUT}, "
        "'Auto-trader: cut losing trade?', 4 + 48); "
        "Write-Output $answer"
    )
    try:
        import subprocess

        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=ROTATION_PROMPT_TIMEOUT + 20,
        )
        return completed.stdout.strip() == "6"  # 6 = Yes, 7 = No, -1 = timed out
    except Exception as exc:  # noqa: BLE001 - never let a prompt failure stop the loop
        print(f"Rotation prompt failed ({exc}); keeping {symbol}.")
        return False


def _exploder_entry_ready(
    client: BinanceClient,
    exploder: BreakoutSignal,
    args: argparse.Namespace,
    settings: BreakoutSettings,
    account: dict[str, object],
) -> tuple[BreakoutSignal | None, str]:
    """Confirm an exploding coin can actually be entered before any held
    position is closed to chase it.

    Rotation that closes a winner only to discover the replacement cannot be
    placed (breakout faded, or the order is below the exchange min notional)
    frees a slot for nothing. Returns ``(fresh_signal, "")`` when the coin
    still confirms AND builds into a placeable order, else ``(None, reason)``.
    """
    fresh = _fresh_order_signal(client, exploder, args, settings)
    if fresh is None:
        return None, "no longer confirms on recheck"
    try:
        rules = trading_rules_from_exchange_info(client.exchange_info())
    except BinanceClientError as exc:
        return None, f"could not load trading rules ({exc})"
    rule = rules.get(fresh.symbol)
    if rule is None:
        return None, "no exchange trading rule found"
    leverage = (
        _dynamic_leverage(fresh.atr_pct, fresh.risk_pct, args.leverage, conviction=_momentum_score(fresh))
        if args.dynamic_leverage
        else args.leverage
    )
    if args.sizing_mode == "auto":
        margin = _resolve_auto_order_margin(args, account)
        if margin <= 0:
            return None, "auto-sizing margin unavailable"
        notional = margin * leverage
    elif args.dynamic_leverage and args.order_margin > 0:
        notional = args.order_margin * leverage
    else:
        notional = _order_notional(args)
    capped = _with_leverage_capped_stop(fresh, leverage, args.max_sl_loss_pct)
    try:
        build_entry_order_plan(
            signal=capped,
            rule=rule,
            requested_notional=notional,
            client_order_id="bd_rotation_precheck",
            working_type=args.order_working_type,
            price_protect=args.order_price_protect,
            hedge_mode=args.hedge_mode,
            entry_mode="RETEST_LIMIT" if args.entry_mode == "SMART_RETEST" else args.entry_mode,
            entry_pullback_pct=args.entry_pullback_pct,
        )
    except OrderPlanError as exc:
        return None, f"entry order is not placeable ({exc})"
    return fresh, ""


def _consider_rotation(
    client: BinanceClient,
    args: argparse.Namespace,
    settings: BreakoutSettings,
    fresh: list[BreakoutSignal],
    pending: list[dict[str, object]],
    account: dict[str, object],
) -> int:
    """Free a slot for an exploding coin by closing a weaker open position. Returns slots freed."""
    exploders = [s for s in fresh if _classify_entry_regime(s) == "INSTANT"]
    if not exploders:
        return 0
    exploder = exploders[0]
    # Cooldown: one rotation per ROTATION_COOLDOWN_SECONDS regardless of how
    # often the scan runs - a 3-min scan must not thrash between exploders.
    global _last_rotation_ts
    cooldown_left = ROTATION_COOLDOWN_SECONDS - (time.time() - _last_rotation_ts)
    if cooldown_left > 0:
        print(
            f"Rotation: {exploder.symbol} is exploding but a rotation ran recently - "
            f"holding positions for {cooldown_left / 60:.0f} more min."
        )
        return 0
    # Never close a real position unless the exploder still confirms AND its
    # entry order can actually be placed - otherwise the slot is freed for nothing.
    ready, skip_reason = _exploder_entry_ready(client, exploder, args, settings, account)
    if ready is None:
        print(f"Rotation: {exploder.symbol} {skip_reason} - keeping current positions.")
        return 0
    exploder_score = _momentum_score(exploder)

    candidates: list[tuple[dict[str, object], float, float, float]] = []
    for item in pending:
        if str(item.get("state", "")) != "MONITORING":
            continue
        # Minimum hold: a freshly opened position has not had room to work yet -
        # rotating it out one scan after entry just churns fees.
        opened_at = _safe_float(item.get("entry_submitted_at")) or _safe_float(item.get("triggered_at"))
        if opened_at > 0 and time.time() - opened_at < ROTATION_MIN_HOLD_SECONDS:
            continue
        score = _safe_float(item.get("momentum_score"))
        if score + ROTATION_MIN_EDGE >= exploder_score:
            continue  # exploder does not clearly out-rank this position
        position = _account_position(
            account, str(item.get("symbol", "")), str(item.get("side", "")), bool(item.get("hedge_mode", False))
        )
        if not position:
            continue
        pnl = _position_unrealized_pnl(position)
        notional = abs(_safe_float(position.get("positionAmt"))) * _safe_float(position.get("entryPrice"))
        net_pnl = pnl - notional * ROTATION_FEE_RATE
        candidates.append((item, net_pnl, pnl, score))
    if not candidates:
        return 0

    profitable = [c for c in candidates if c[1] > 0]
    if profitable:
        item, _net, pnl, _score = min(profitable, key=lambda c: c[3])
        print(
            f"Rotation: {item.get('symbol')} is in profit ({pnl:.2f} USDT) and out-ranked by "
            f"exploding {exploder.symbol} - closing to chase it."
        )
        if _close_position(client, item, account, args, reason=f"rotate into {exploder.symbol}"):
            _remove_pending_entry(args, item)
            _last_rotation_ts = time.time()
            return 1
        return 0

    # Only losing positions can free a slot - ask before realizing a loss.
    item, _net, pnl, _score = min(candidates, key=lambda c: c[3])
    if _prompt_cut_losing_trade(str(item.get("symbol", "")), pnl, exploder.symbol, args):
        if _close_position(client, item, account, args, reason=f"user cut loss to chase {exploder.symbol}"):
            _remove_pending_entry(args, item)
            _last_rotation_ts = time.time()
            return 1
    else:
        print(f"Rotation: keeping {item.get('symbol')} - loss not cut.")
    return 0


def _opportunity_count(pending_entries: list[dict[str, object]]) -> int:
    """Distinct opportunities in the pending file (a two-sided bracket counts as one)."""
    brackets: set[str] = set()
    singles = 0
    for item in pending_entries:
        bracket_id = str(item.get("bracket_id", ""))
        if bracket_id:
            brackets.add(bracket_id)
        else:
            singles += 1
    return len(brackets) + singles


def _queue_room(pending_entries: list[dict[str, object]], args: argparse.Namespace) -> int:
    """How many more coins the bot may arm and watch. Independent of the position cap."""
    limit = args.queue_size if args.queue_size > 0 else 999
    return limit - _opportunity_count(pending_entries)


def _active_position_count(pending_entries: list[dict[str, object]], account: dict[str, object]) -> int:
    """Entries consuming a concurrency slot: live positions plus orders already placed."""
    count = sum(
        1
        for position in account.get("positions", [])
        if isinstance(position, dict) and abs(_safe_float(position.get("positionAmt"))) > 0
    )
    count += sum(1 for item in pending_entries if str(item.get("state", "")) == "ENTRY_ORDER_PLACED")
    return count


MANAGER_STATE_PRIORITY = {
    "MONITORING": 0,
    "ENTRY_ORDER_PLACED": 1,
    "WAIT_RETEST": 2,
    "WAIT_BREAKOUT": 3,
}
MANAGER_REGIME_PRIORITY = {
    "INSTANT": 0,
    "TRAILING_RETEST": 1,
    "RETEST": 2,
    "STRICT_RETEST": 3,
}


def _prioritize_pending_entry_plans(pending_entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Order pending entries before a manager pass spends any free slot."""
    indexed = list(enumerate(pending_entries))

    def key(pair: tuple[int, dict[str, object]]) -> tuple[int, int, float, float, int]:
        index, item = pair
        state = str(item.get("state", "WAIT_BREAKOUT"))
        state_rank = MANAGER_STATE_PRIORITY.get(state, 9)
        if state_rank < MANAGER_STATE_PRIORITY["WAIT_RETEST"]:
            return (state_rank, 0, 0.0, 0.0, index)
        regime_rank = MANAGER_REGIME_PRIORITY.get(str(item.get("entry_regime", "RETEST")), 9)
        return (
            state_rank,
            regime_rank,
            -_safe_float(item.get("momentum_score")),
            _safe_float(item.get("created_at")),
            index,
        )

    return [item for _, item in sorted(indexed, key=key)]


def _entry_stale_minutes_for_item(item: dict[str, object], args: argparse.Namespace) -> float:
    item_minutes = _safe_float(item.get("entry_stale_minutes"))
    if item_minutes > 0:
        return item_minutes
    return _safe_float(getattr(args, "entry_stale_minutes", 0.0))


def _stale_waiting_entry_reason(
    item: dict[str, object],
    args: argparse.Namespace,
    now: float | None = None,
) -> str:
    if str(item.get("state", "")) != "WAIT_RETEST":
        return ""
    threshold_min = _entry_stale_minutes_for_item(item, args)
    if threshold_min <= 0:
        return ""
    started_at = _safe_float(item.get("triggered_at")) or _safe_float(item.get("created_at"))
    if started_at <= 0:
        return ""
    age_seconds = (time.time() if now is None else now) - started_at
    threshold_seconds = threshold_min * 60.0
    if age_seconds < threshold_seconds:
        return ""
    return (
        f"{item.get('symbol')}@{item.get('interval', '')}: stale queued entry dropped "
        f"after {age_seconds / 60.0:.1f} min without a free slot; will rescan/rebuild if still valid"
    )


def _drop_stale_waiting_entries(
    pending_entries: list[dict[str, object]],
    args: argparse.Namespace,
    now: float | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    kept: list[dict[str, object]] = []
    dropped: list[str] = []
    for item in pending_entries:
        reason = _stale_waiting_entry_reason(item, args, now=now)
        if reason:
            dropped.append(reason)
        else:
            kept.append(item)
    return kept, dropped


def _scan_and_arm(client: BinanceClient, args: argparse.Namespace, settings: BreakoutSettings) -> None:
    """Scan the market, re-rank the queue, and arm the strongest fresh opportunities."""
    args._live_ml_symbol_context = _load_ml_symbol_context(args.ml_context_file)
    args._live_ml_btc_context = _current_btc_ml_context(client, args) if getattr(args, "ml_rank_model_data", None) else {}
    universe = _resolve_symbols(client, args)
    universe = _filter_order_books(client, universe, args)
    universe = _filter_open_interest(client, universe, args)
    if not universe.symbols:
        print("Auto-trader scan: no symbols matched the universe filters.")
        return

    now_ms = int(time.time() * 1000)
    signals, _ = scan_symbols(client, universe.symbols, args, settings, _resolve_intervals(args), now_ms)
    signals = [s for s in signals if s.reward_risk >= args.min_rr and s.score >= args.min_score]
    _rank_order_signals(signals, args)
    if not signals:
        _quiet_scan_heartbeat()
        return

    pending = _load_pending_entry_plans(args.entry_state_file)
    account = client.account_info(recv_window=args.recv_window)

    # Rebuild untriggered and stale triggered entries from scratch every scan:
    # re-arming the top-ranked coins means regime, trigger, retest limit and
    # exit plans reflect the latest market state. Placed/monitoring entries are
    # kept untouched because they already have exchange/account state attached.
    undropped = [item for item in pending if str(item.get("state", "")) != "WAIT_BREAKOUT"]
    dropped = len(pending) - len(undropped)
    committed, stale_waiting = _drop_stale_waiting_entries(undropped, args)
    dropped += len(stale_waiting)
    for reason in stale_waiting:
        print(reason)
    if dropped:
        _write_pending_entry_plans(args.entry_state_file, committed)
    pending = committed

    pending_symbols = {str(item.get("symbol", "")) for item in pending}
    fresh = [s for s in signals if s.symbol not in pending_symbols]

    # Clear stale ENTRY_ORDER_PLACED items whose LIMIT entry has been waiting
    # too long. A never-filling order blocks a concurrency slot indefinitely;
    # when a higher-scoring fresh signal exists, swap to it.
    cleared_stale: list[dict[str, object]] = []
    for item in pending:
        if str(item.get("state", "")) != "ENTRY_ORDER_PLACED":
            continue
        stale_failures: list[str] = []
        if _maybe_clear_stale_entry_order(client, item, args, fresh, account, stale_failures):
            cleared_stale.append(item)
        for failure in stale_failures:
            print(f"- {failure}")
    if cleared_stale:
        stale_ids = {str(it.get("entry_client_order_id", "")) for it in cleared_stale}
        pending = [it for it in pending if str(it.get("entry_client_order_id", "")) not in stale_ids]
        _write_pending_entry_plans(args.entry_state_file, pending)
        pending_symbols = {str(item.get("symbol", "")) for item in pending}
        fresh = [s for s in signals if s.symbol not in pending_symbols]

    # Position rotation: when every concurrency slot is taken and an exploding coin is
    # waiting, close a weaker open position to make room for it.
    cap = args.max_concurrent_orders
    if cap > 0 and fresh and _active_position_count(pending, account) >= cap:
        if _consider_rotation(client, args, settings, fresh, pending, account):
            pending = _load_pending_entry_plans(args.entry_state_file)

    room = _queue_room(pending, args)
    if room <= 0:
        print(f"Auto-trader scan: watch queue full ({args.queue_size}); {len(fresh)} candidate(s) waiting.")
        return

    # Reserve the last queue slot if no fresh candidate is a loaded coil ready to fire.
    if room == 1 and fresh and not _is_high_conviction(fresh[0]):
        print("Auto-trader scan: holding the last queue slot open for a high-conviction breakout.")
        return

    args.order_count = room
    results, order_failures = place_best_orders(client, signals, args, settings, account=account)
    armed = sum(1 for result in results if result.role == "ENTRY")
    print(
        f"Auto-trader scan: rebuilt queue (dropped {dropped} untriggered/stale), armed {armed} entr(ies); "
        f"watching {'all candidates' if args.queue_size <= 0 else f'up to {args.queue_size}'}, "
        f"max {cap or 'unlimited'} concurrent positions."
    )
    for failure in order_failures[:6]:
        print(f"- {failure}")


def _quiet_scan_heartbeat() -> None:
    """Print a low-rate alive-signal when scans keep finding nothing.

    Default cap: one line per hour, with a UTC timestamp so the user can see
    the bot is running without the every-3-minute spam.
    """
    global _LAST_QUIET_HEARTBEAT_TS
    now = time.time()
    if now - _LAST_QUIET_HEARTBEAT_TS < _QUIET_HEARTBEAT_INTERVAL_SECONDS:
        return
    import datetime as _dt
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M UTC")
    print(f"[{ts}] Auto-trader: quiet market - no qualifying setups; bot still scanning.")
    _LAST_QUIET_HEARTBEAT_TS = now


def _print_monitored_coins(pending_entries: list[dict[str, object]]) -> None:
    """Show the current roster of coins the auto-trader is watching or holding."""
    if not pending_entries:
        return  # silent when nothing is being watched; _quiet_scan_heartbeat covers liveness
    print(f"Monitoring {len(pending_entries)} entr(y/ies):")
    for item in pending_entries:
        print(
            f"  - {item.get('symbol', '?')} {item.get('side', '')} {item.get('interval', '')}"
            f"  state={item.get('state', '')}  regime={item.get('entry_regime', 'RETEST')}"
            f"  trigger={item.get('trigger_price', '')}"
        )


def run_auto_trader(client: BinanceClient, args: argparse.Namespace, settings: BreakoutSettings) -> int:
    """Continuous loop: scan every --scan-interval-minutes, manage entries/exits each fast tick."""
    args.watch_exits = False  # each manage_pending_exits call performs a single pass
    scan_interval = args.scan_interval_minutes * 60
    last_scan = 0.0
    print(
        f"Auto-trader started. Scanning every {args.scan_interval_minutes} min; "
        f"max {args.max_concurrent_orders or 'unlimited'} concurrent positions. Ctrl-C to stop."
    )
    if args.ml_rank_model_data:
        print(f"ML ranker: {args.ml_rank_model} score={args.ml_rank_score} mode=ranking-only.")
    if args.exhaustion_exit:
        print("Exhaustion exit: live monitoring enabled (+0.5R then rejection/stall close).")
    if args.stagnation_after_r > 0:
        print(
            f"Stagnation exit: live monitoring enabled "
            f"(+{args.stagnation_after_r:.2g}R then {args.stagnation_candles}c no new extreme)."
        )
    _print_monitored_coins(_load_pending_entry_plans(args.entry_state_file))
    while True:
        if time.time() - last_scan >= scan_interval:
            try:
                _scan_and_arm(client, args, settings)
            except BinanceClientError as exc:
                print(f"Auto-trader scan failed, will retry next cycle: {exc}")
            _print_monitored_coins(_load_pending_entry_plans(args.entry_state_file))
            last_scan = time.time()

        if _has_pending_trade_work(args):
            try:
                manage_pending_exits(client, args)
            except BinanceClientError as exc:
                print(f"Auto-trader manage pass failed, will retry: {exc}")
            time.sleep(max(args.exit_poll_seconds, 1.0))
        else:
            idle_for = max(scan_interval - (time.time() - last_scan), 1.0)
            time.sleep(min(idle_for, scan_interval))


# Status-line dedup state, persisted across the auto-trader's repeated manage passes.
_mgr_last_status: tuple[int, int, int, int, int, int, int, int, int, int] | None = None
_mgr_last_status_time = 0.0
_mgr_last_error = ""


def manage_pending_exits(client: BinanceClient, args: argparse.Namespace) -> int:
    if not client.api_key or not client.api_secret:
        print("Could not manage entries/exits: set BINANCE_API_KEY and BINANCE_API_SECRET in .env or environment.", file=sys.stderr)
        return 2
    global _mgr_last_status, _mgr_last_status_time, _mgr_last_error
    deadline = time.time() + args.exit_watch_timeout if args.watch_exits and args.exit_watch_timeout > 0 else None
    while True:
        pending_entries = _load_pending_entry_plans(args.entry_state_file)
        pending_exits = _load_pending_exit_plans(args.exit_state_file)
        if not pending_entries and not pending_exits:
            print(f"No pending smart entries in {args.entry_state_file}; no pending exits in {args.exit_state_file}.")
            return 0

        try:
            account = client.account_info(recv_window=args.recv_window)
        except BinanceClientError as exc:
            message = f"Trade manager: account check failed, retrying: {exc}"
            if message != _mgr_last_error or _heartbeat_due(_mgr_last_status_time, args):
                print(message)
                _mgr_last_status_time = time.time()
                _mgr_last_error = message
            if not args.watch_exits:
                return 2
            if deadline and time.time() >= deadline:
                print("Trade manager watch timeout reached.")
                return 0
            time.sleep(max(args.exit_poll_seconds, 1.0))
            continue

        # One bulk mark-price request per tick instead of one per watched coin -
        # keeps API weight flat regardless of how many coins are queued.
        try:
            mark_prices = client.mark_prices()
        except BinanceClientError:
            mark_prices = {}

        if args.max_concurrent_orders <= 0:
            saved_cap = next(
                (int(_safe_float(item.get("max_concurrent_orders"))) for item in pending_entries if item.get("max_concurrent_orders")),
                0,
            )
            effective_max_concurrent = saved_cap
        else:
            effective_max_concurrent = args.max_concurrent_orders

        if args.max_market_deviation_pct <= 0:
            saved_deviation = next(
                (_safe_float(item.get("max_market_deviation_pct")) for item in pending_entries if item.get("max_market_deviation_pct")),
                1.5,
            )
            args.max_market_deviation_pct = saved_deviation

        if not args.no_market_fallback:
            args.no_market_fallback = any(item.get("no_market_fallback") for item in pending_entries)

        if not args.dynamic_sl:
            args.dynamic_sl = any(item.get("dynamic_sl") for item in pending_entries)
        if args.dynamic_sl and args.sl_update_interval_seconds <= 0:
            args.sl_update_interval_seconds = next(
                (_safe_float(item.get("sl_update_interval_seconds")) for item in pending_entries if item.get("sl_update_interval_seconds")),
                300.0,
            )
        if args.sl_lookback <= 0:
            args.sl_lookback = int(next(
                (_safe_float(item.get("sl_lookback")) for item in pending_entries if item.get("sl_lookback")),
                20.0,
            ))

        trading_rules: dict[str, TradingRule] = {}
        if args.dynamic_sl and not trading_rules:
            try:
                trading_rules = trading_rules_from_exchange_info(client.exchange_info())
            except BinanceClientError as exc:
                print(f"Trade manager: could not load trading rules for dynamic SL: {exc}")

        remaining_entries: list[dict[str, object]] = []
        remaining_exits: list[dict[str, object]] = []
        entry_orders_placed = 0
        exits_placed = 0
        closed_positions = 0
        waiting_breakout = 0
        waiting_retest = 0
        waiting_fill = 0
        waiting_exit_entry = 0
        failures: list[str] = []
        pending_entries, stale_waiting = _drop_stale_waiting_entries(pending_entries, args)
        for reason in stale_waiting:
            print(reason)
        pending_entries = _prioritize_pending_entry_plans(pending_entries)

        # Bracket sides whose opposite side has already entered (OCO): the loser is cancelled.
        committed_brackets: set[str] = {
            str(item.get("bracket_id"))
            for item in pending_entries
            if item.get("bracket_id") and str(item.get("state", "")) in ("ENTRY_ORDER_PLACED", "MONITORING")
        }

        # Count all open positions on the account (covers positions that have already
        # been graduated out of the pending file but are still consuming margin).
        # Also add entries that are placed but not yet filled.
        active_entry_count = sum(
            1 for position in account.get("positions", [])
            if isinstance(position, dict) and abs(_safe_float(position.get("positionAmt"))) > 0
        )
        active_entry_count += sum(
            1 for item in pending_entries
            if str(item.get("state", "")) == "ENTRY_ORDER_PLACED"
        )

        for item in pending_entries:
            symbol = str(item.get("symbol", ""))
            side = str(item.get("side", ""))
            hedge_mode = bool(item.get("hedge_mode", False))
            position = _account_position(account, symbol=symbol, side=side, hedge_mode=hedge_mode)
            if position:
                state = str(item.get("state", "WAIT_BREAKOUT"))
                open_bracket_id = str(item.get("bracket_id", ""))
                if open_bracket_id:
                    committed_brackets.add(open_bracket_id)
                if state != "MONITORING":
                    placed, item_failed, item_failures = _place_exit_plans_from_item(client, item, args)
                    exits_placed += placed
                    failures.extend(item_failures)
                    if item_failed:
                        # If we emergency-closed the position because the SL would
                        # have fired immediately, drop the item entirely - keeping
                        # it pending would lead to endless retry attempts on a
                        # position that no longer exists.
                        if not item.get("_emergency_closed"):
                            remaining_entries.append(item)
                        continue
                    item["state"] = "MONITORING"
                    sl_entry = next(
                        (p for p in item.get("exit_plans", []) if isinstance(p, dict) and p.get("role") == "STOP_LOSS"),
                        None,
                    )
                    if sl_entry:
                        item["sl_client_order_id"] = str(sl_entry.get("client_order_id", ""))
                        item["sl_trigger_price"] = str(sl_entry.get("payload", {}).get("triggerPrice", "0"))
                    item["last_sl_update"] = 0.0
                    item["entry_filled_at"] = time.time()
                item["live_entry_price"] = _safe_float(position.get("entryPrice"))
                item["live_initial_stop"] = item.get("live_initial_stop") or _initial_stop_from_item(item)
                # Dust sweep: when TP1 + trail floor-round independently they can
                # leave a sub-lot remainder (e.g. 12.7 entry / 50-50 split / 0.1
                # step -> 6.3 + 6.3 = 12.6, 0.1 dust). Once the remainder falls
                # below the exchange's min-notional it can never close itself, so
                # market-close it now and cancel the leftover algos.
                rule = trading_rules.get(symbol)
                if rule and _is_dust_position(item, position, rule, mark_prices.get(symbol, 0.0)):
                    if _sweep_dust_position(client, item, account, args, failures):
                        closed_positions += 1
                        continue
                if _maybe_exhaustion_exit(client, item, args, account, position, failures):
                    closed_positions += 1
                    continue
                if _maybe_stagnation_exit(client, item, args, account, position, failures):
                    closed_positions += 1
                    continue
                if args.dynamic_sl:
                    rule = trading_rules.get(symbol)
                    if rule:
                        _maybe_reposition_sl(client, item, args, rule, account, failures)
                remaining_entries.append(item)
                continue

            state = str(item.get("state", "WAIT_BREAKOUT"))
            if state == "MONITORING":
                mark = mark_prices.get(symbol, 0.0)
                if mark > 0:
                    _record_symbol_r_from_close(args, item, None, mark)
                cancelled = _cancel_exit_algos_for_closed_entry(client, item, args, failures)
                closed_positions += 1
                if cancelled:
                    print(f"{symbol}@{item.get('interval', '')}: position closed; cancelled {cancelled} leftover exit order(s).")
                else:
                    print(f"{symbol}@{item.get('interval', '')}: position closed; removed from monitoring.")
                continue

            if state == "ENTRY_ORDER_PLACED":
                waiting_fill += 1
                remaining_entries.append(item)
                continue

            bracket_id = str(item.get("bracket_id", ""))
            if bracket_id and bracket_id in committed_brackets:
                print(f"{symbol} {side} bracket side cancelled: opposite side entered")
                continue

            mark_price = mark_prices.get(symbol, 0.0)
            if mark_price <= 0:
                failures.append(f"{symbol}@{item.get('interval', '')}: mark price unavailable")
                remaining_entries.append(item)
                continue

            if state == "WAIT_BREAKOUT":
                if not _trigger_reached(side, mark_price, _safe_float(item.get("trigger_price"))):
                    waiting_breakout += 1
                    remaining_entries.append(item)
                    continue
                item["state"] = "WAIT_RETEST"
                item["triggered_at"] = time.time()
                state = "WAIT_RETEST"

            if state == "WAIT_RETEST":
                regime = str(item.get("entry_regime", "RETEST"))
                orig_limit = _safe_float(item.get("limit_price"))
                effective_limit = orig_limit
                if regime == "TRAILING_RETEST":
                    effective_limit = _trailing_retest_limit(item, side, mark_price, orig_limit)
                retest_reached = _retest_reached(side, mark_price, effective_limit)
                # Enter at the trigger (market) the moment the breakout fires, for
                # every regime - never sit waiting for a retest pullback. A 27-run
                # sweep over 3 regime windows showed trigger entry beats the
                # retest-wait in every case (49x vs 37x compounded); the retest-wait
                # was the worst entry path. Regime still drives BTC guards + sizing.
                if effective_max_concurrent > 0 and active_entry_count >= effective_max_concurrent:
                    waiting_retest += 1
                    remaining_entries.append(item)
                    continue
                order_kind = "LIMIT" if retest_reached else "MARKET"
                # Structural-break guard: if mark has already moved past the planned
                # SL trigger, the entry would fill into a position that's immediately
                # stopped out — and Binance rejects the SL placement with "Order would
                # immediately trigger", leaving the position unprotected. Abandon the
                # trade entirely (don't re-queue) on both MARKET and LIMIT paths.
                # DYDX 2026-05: triggered at 0.1523, fell for 10h, LIMIT at 0.1515
                # filled into mark ~0.14, SL 0.1501 rejected — unprotected long.
                sl_trigger = _initial_stop_from_item(item)
                if sl_trigger > 0:
                    sl_breached = (
                        (side == "LONG" and mark_price <= sl_trigger)
                        or (side == "SHORT" and mark_price >= sl_trigger)
                    )
                    if sl_breached:
                        failures.append(
                            f"{symbol}@{item.get('interval', '')} {order_kind} entry abandoned: "
                            f"mark {mark_price} already past SL trigger {sl_trigger} ({side}); "
                            f"structure broken before fill ({regime})"
                        )
                        continue
                if order_kind == "MARKET":
                    deviation_cap = INSTANT_MAX_DEVIATION_PCT if regime == "INSTANT" else args.max_market_deviation_pct
                    if deviation_cap > 0:
                        trigger = _safe_float(item.get("trigger_price"))
                        if trigger > 0:
                            if side == "LONG":
                                deviation = (mark_price - trigger) / trigger
                            else:
                                deviation = (trigger - mark_price) / trigger
                            if deviation > deviation_cap / 100:
                                # Trade abandoned (no remaining_entries.append): the
                                # deviation cap means "this move ran away, skip" — not
                                # "wait for a retest". Leaving in pending would let it
                                # fill via the LIMIT/retest path on the way down.
                                failures.append(
                                    f"{symbol}@{item.get('interval', '')} MARKET entry abandoned: "
                                    f"price moved {deviation * 100:.2f}% from trigger, exceeds "
                                    f"{deviation_cap:.1f}% cap ({regime})"
                                )
                                continue
                limit_override = effective_limit if regime == "TRAILING_RETEST" and order_kind == "LIMIT" else 0.0
                try:
                    _apply_pending_account_settings(client, item, args)
                except BinanceClientError as exc:
                    failures.append(f"{symbol}@{item.get('interval', '')} {order_kind} entry: {exc}")
                    remaining_entries.append(item)
                    continue
                response, resize_note, entry_error = _place_entry_order(
                    client, item, order_kind, limit_override, mark_price, account, trading_rules, args
                )
                if entry_error or response is None:
                    failures.append(f"{symbol}@{item.get('interval', '')} {order_kind} entry: {entry_error}")
                    remaining_entries.append(item)
                    continue
                if resize_note:
                    print(f"{symbol}@{item.get('interval', '')}: {resize_note}")
                item["state"] = "ENTRY_ORDER_PLACED"
                item["entry_order_type"] = order_kind
                item["entry_order_status"] = str(response.get("status") or "SUBMITTED")
                item["entry_order_id"] = str(response.get("orderId") or "")
                item["entry_submitted_at"] = int(time.time())
                if bracket_id:
                    committed_brackets.add(bracket_id)
                entry_orders_placed += 1
                active_entry_count += 1
                waiting_fill += 1
                remaining_entries.append(item)
                continue

            failures.append(f"{symbol}@{item.get('interval', '')}: unknown pending entry state {state}")
            remaining_entries.append(item)

        for item in pending_exits:
            symbol = str(item.get("symbol", ""))
            side = str(item.get("side", ""))
            hedge_mode = bool(item.get("hedge_mode", False))
            if not _has_open_position(account, symbol=symbol, side=side, hedge_mode=hedge_mode):
                waiting_exit_entry += 1
                remaining_exits.append(item)
                continue
            placed, item_failed, item_failures = _place_exit_plans_from_item(client, item, args)
            exits_placed += placed
            failures.extend(item_failures)
            if item_failed:
                remaining_exits.append(item)

        # OCO: once one side of a bracket has entered, drop the un-triggered opposite side.
        if committed_brackets:
            surviving_entries: list[dict[str, object]] = []
            for item in remaining_entries:
                item_bracket = str(item.get("bracket_id", ""))
                if (
                    item_bracket
                    and item_bracket in committed_brackets
                    and str(item.get("state", "")) in ("WAIT_BREAKOUT", "WAIT_RETEST")
                ):
                    print(f"{item.get('symbol', '')} {item.get('side', '')} bracket side cancelled: opposite side entered")
                    if str(item.get("state", "")) == "WAIT_BREAKOUT":
                        waiting_breakout -= 1
                    else:
                        waiting_retest -= 1
                    continue
                surviving_entries.append(item)
            remaining_entries = surviving_entries

        _write_pending_entry_plans(args.entry_state_file, remaining_entries)
        _write_pending_exit_plans(args.exit_state_file, remaining_exits)
        total_remaining = len(remaining_entries) + len(remaining_exits)
        current_status = (
            entry_orders_placed,
            exits_placed,
            closed_positions,
            waiting_breakout,
            waiting_retest,
            waiting_fill,
            waiting_exit_entry,
            len(remaining_entries),
            len(remaining_exits),
            total_remaining,
        )
        if entry_orders_placed > 0 or exits_placed > 0 or failures or current_status != _mgr_last_status:
            print(
                "Trade manager: "
                f"entry_orders={entry_orders_placed}  exits={exits_placed}  "
                f"closed={closed_positions}  "
                f"wait_breakout={waiting_breakout}  wait_retest={waiting_retest}  "
                f"wait_fill={waiting_fill}  exit_wait={waiting_exit_entry}  remaining={total_remaining}"
            )
            for failure in failures:
                print(f"- {failure}")
            _mgr_last_status = current_status
            _mgr_last_status_time = time.time()
            _mgr_last_error = ""
        if not args.watch_exits or not total_remaining:
            return 2 if failures else 0
        if deadline and time.time() >= deadline:
            print("Trade manager watch timeout reached.")
            return 0
        time.sleep(max(args.exit_poll_seconds, 1.0))


def _emergency_close_position(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
    reason: str,
) -> bool:
    """Market-close a position whose SL placement was rejected as 'would immediately
    trigger'. Cancel any leftover exit orders. Returns True on success."""
    symbol = str(item.get("symbol", ""))
    side = str(item.get("side", ""))
    hedge_mode = bool(item.get("hedge_mode", False))
    try:
        account = client.account_info(recv_window=args.recv_window)
    except BinanceClientError as exc:
        print(f"{symbol} EMERGENCY CLOSE could not read account: {exc}")
        return False
    position = _account_position(account, symbol, side, hedge_mode)
    if not position:
        return True  # no position - nothing to close
    qty = abs(_safe_float(position.get("positionAmt")))
    if qty <= 0:
        return True
    payload: dict[str, str] = {
        "symbol": symbol,
        "side": "SELL" if side == "LONG" else "BUY",
        "type": "MARKET",
        "quantity": str(qty),
    }
    if hedge_mode:
        payload["positionSide"] = side
    else:
        payload["reduceOnly"] = "true"
    try:
        client.place_order(payload, test=False, recv_window=args.recv_window)
    except BinanceClientError as exc:
        print(f"{symbol} EMERGENCY CLOSE FAILED ({reason}): {exc}")
        return False
    print(f"{symbol} EMERGENCY CLOSE at market ({reason})")
    # Cancel any exit algos that already got placed so they cannot misfire later.
    for algo_id in _exit_algo_ids(item):
        try:
            client.cancel_algo_order(symbol, algo_id, recv_window=args.recv_window)
        except BinanceClientError:
            pass
    return True


def _place_exit_plans_from_item(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
) -> tuple[int, bool, list[str]]:
    symbol = str(item.get("symbol", ""))
    placed = 0
    failures: list[str] = []
    raw_plans = item.get("exit_plans", [])
    if not isinstance(raw_plans, list):
        return 0, False, failures
    placed_ids = item.get("placed_exit_client_order_ids", [])
    if not isinstance(placed_ids, list):
        placed_ids = []
    placed_id_set = {str(client_id) for client_id in placed_ids}
    for raw_plan in raw_plans:
        payload = dict(raw_plan.get("payload", {})) if isinstance(raw_plan, dict) else {}
        role = raw_plan.get("role", "EXIT") if isinstance(raw_plan, dict) else "EXIT"
        client_order_id = str(raw_plan.get("client_order_id", "")) if isinstance(raw_plan, dict) else ""
        if client_order_id and client_order_id in placed_id_set:
            continue
        if payload.get("type") == "TRAILING_STOP_MARKET":
            payload.pop("activatePrice", None)
        try:
            client.place_algo_order(payload, recv_window=args.recv_window)
            placed += 1
            if client_order_id:
                placed_id_set.add(client_order_id)
                placed_ids.append(client_order_id)
                item["placed_exit_client_order_ids"] = placed_ids
        except BinanceClientError as exc:
            # Critical safety net: if the STOP_LOSS would fire immediately, the
            # SL has effectively already triggered - market-close the position
            # instead of leaving it unprotected while we retry indefinitely.
            exc_msg = str(exc)
            if role == "STOP_LOSS" and "would immediately trigger" in exc_msg.lower():
                _emergency_close_position(
                    client, item, args,
                    reason=f"SL placement rejected (mark already past trigger): {exc_msg}",
                )
                item["_emergency_closed"] = True
                failures.append(
                    f"{symbol} {role}: {exc_msg} -> emergency market-close (position was unprotected)"
                )
                return placed, True, failures
            failures.append(f"{symbol} {role}: {exc}")
            return placed, True, failures
    return placed, False, failures


def _is_dust_position(
    item: dict[str, object],
    position: dict[str, object],
    rule: object,
    mark_price: float,
) -> bool:
    """A position is dust when it survives at all (positionAmt != 0) but its
    notional is below the exchange's min-notional. Such positions can't be
    closed by another algo order and would otherwise sit around forever.
    Also treats 'tiny relative to original' (<5%) as dust to catch cases where
    min_notional is itself very small."""
    amt = abs(_safe_float(position.get("positionAmt")))
    if amt <= 0 or mark_price <= 0:
        return False
    entry_qty_str = str(item.get("quantity", "0")).lstrip("-")
    try:
        entry_qty = float(entry_qty_str) if entry_qty_str else 0.0
    except ValueError:
        entry_qty = 0.0
    if entry_qty > 0 and amt < entry_qty * 0.05:
        return True
    min_notional = float(getattr(rule, "min_notional", 0) or 0)
    if min_notional > 0 and amt * mark_price < min_notional:
        return True
    return False


def _sweep_dust_position(
    client: BinanceClient,
    item: dict[str, object],
    account: dict[str, object],
    args: argparse.Namespace,
    failures: list[str],
) -> bool:
    """Market-close a sub-min-notional residue, then cancel any leftover algos.
    Returns True when the dust was actually flattened."""
    symbol = str(item.get("symbol", ""))
    if not _close_position(client, item, account, args, reason="dust sweep"):
        failures.append(f"{symbol}: dust sweep failed - position too small to market-close cleanly")
        return False
    cancelled = _cancel_exit_algos_for_closed_entry(client, item, args, failures)
    if cancelled:
        print(f"{symbol}: dust sweep closed residual position + cancelled {cancelled} leftover algo(s).")
    else:
        print(f"{symbol}: dust sweep closed residual position.")
    _remove_pending_entry(args, item)
    return True


def _cancel_entry_order(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
) -> bool:
    """Cancel the live entry order for an ENTRY_ORDER_PLACED item. Returns True on
    success (or if the order was already gone). The caller is responsible for
    dropping the item from the pending file."""
    symbol = str(item.get("symbol", ""))
    client_order_id = str(item.get("entry_client_order_id", ""))
    if not symbol or not client_order_id:
        return False
    try:
        client._signed_request(
            "DELETE",
            "/order",
            {"symbol": symbol, "origClientOrderId": client_order_id, "recvWindow": args.recv_window},
        )
        return True
    except BinanceClientError as exc:
        msg = str(exc)
        # "Unknown order" or "Order does not exist" means Binance already removed
        # it (filled, expired, or cancelled by the user). Treat as success.
        if "Unknown order" in msg or "Order does not exist" in msg or "-2011" in msg:
            return True
        return False


def _maybe_clear_stale_entry_order(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
    fresh: list[object],
    account: dict[str, object],
    failures: list[str],
) -> bool:
    """Drop an ENTRY_ORDER_PLACED item when its LIMIT entry has been waiting too
    long. Two trigger conditions, OR'd:
      1. age > entry_stale_minutes AND a higher-scoring fresh signal is waiting
         for a slot (priority swap to the best opportunity).
      2. age > 2 * entry_stale_minutes (hard timeout, even with no replacement).
    Returns True when the stale order was cancelled and the item should be
    dropped from pending."""
    threshold_min = _safe_float(args.entry_stale_minutes)
    if threshold_min <= 0:
        return False
    submitted_at = _safe_float(item.get("entry_submitted_at"))
    if submitted_at <= 0:
        return False
    age_seconds = time.time() - submitted_at
    threshold_seconds = threshold_min * 60.0
    if age_seconds < threshold_seconds:
        return False

    symbol = str(item.get("symbol", ""))
    side = str(item.get("side", ""))
    hedge_mode = bool(item.get("hedge_mode", False))

    # Defensive: if a position actually exists now, the manager should promote it
    # to MONITORING, not cancel it. Skip the stale check.
    if _has_open_position(account, symbol=symbol, side=side, hedge_mode=hedge_mode):
        return False

    own_score = _safe_float(item.get("momentum_score"))
    has_better_fresh = any(
        _momentum_score(s) > own_score
        for s in fresh
        if str(getattr(s, "symbol", "")) != symbol
    )
    hard_timeout = age_seconds >= 2 * threshold_seconds
    if not (has_better_fresh or hard_timeout):
        return False

    age_min = age_seconds / 60.0
    if _cancel_entry_order(client, item, args):
        reason = "hard timeout" if hard_timeout else f"swapped for higher-scoring fresh signal"
        print(
            f"{symbol}@{item.get('interval', '')}: stale entry order cancelled "
            f"after {age_min:.1f} min ({reason}); slot freed."
        )
        return True
    failures.append(
        f"{symbol}@{item.get('interval', '')}: stale entry cancel failed at {age_min:.1f} min, "
        f"will retry next scan."
    )
    return False


def _cancel_exit_algos_for_closed_entry(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
    failures: list[str],
) -> int:
    symbol = str(item.get("symbol", ""))
    cancelled = 0
    for algo_id in _exit_algo_ids(item):
        try:
            client.cancel_algo_order(symbol, algo_id, recv_window=args.recv_window)
            cancelled += 1
        except BinanceClientError as exc:
            if "Unknown order" not in str(exc) and "Order does not exist" not in str(exc):
                failures.append(f"{symbol} closed trade exit cancel {algo_id}: {exc}")
    return cancelled


def _maybe_exhaustion_exit(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
    account: dict[str, object],
    position: dict[str, object],
    failures: list[str],
) -> bool:
    # CLI is authoritative: if the operator removes --exhaustion-exit, stale
    # per-item flags carried in the pending file must not keep firing. (Bug:
    # the OR-check used to honour the persisted item flag, so disabling the
    # feature on the CLI did nothing for orders queued earlier.)
    if not args.exhaustion_exit:
        return False
    symbol = str(item.get("symbol", ""))
    side = str(item.get("side", ""))
    interval = str(item.get("interval", "1h"))
    entry = _safe_float(position.get("entryPrice")) or _safe_float(item.get("live_entry_price"))
    initial_stop = _safe_float(item.get("live_initial_stop")) or _initial_stop_from_item(item)
    initial_risk = abs(entry - initial_stop)
    if not symbol or entry <= 0 or initial_risk <= 0:
        return False

    now_ms = int(time.time() * 1000)
    try:
        lookback = int(_safe_float(item.get("exhaustion_lookback")) or args.exhaustion_lookback)
        klines = client.klines(symbol, interval, max(lookback, 10))
    except BinanceClientError as exc:
        failures.append(f"{symbol}@{interval} exhaustion check: {exc}")
        return False
    candles = [c for c in candles_from_klines(klines) if c.close_time <= now_ms]
    if not candles:
        return False

    last_checked = int(_safe_float(item.get("exhaustion_last_close_time")))
    entry_filled_at = _safe_float(item.get("entry_filled_at"))
    entry_ms = int(entry_filled_at * 1000) if entry_filled_at > 0 else 0
    fresh = [
        candle
        for candle in candles
        if candle.close_time > last_checked and (entry_ms <= 0 or candle.close_time >= entry_ms)
    ]
    if not fresh:
        return False

    peak = _safe_float(item.get("exhaustion_peak")) or entry
    trough = _safe_float(item.get("exhaustion_trough")) or entry
    bars_since_peak = int(_safe_float(item.get("exhaustion_bars_since_peak")))
    bars_since_trough = int(_safe_float(item.get("exhaustion_bars_since_trough")))

    for candle in fresh:
        if side == "LONG":
            if candle.high > peak:
                peak = candle.high
                bars_since_peak = 0
            else:
                bars_since_peak += 1
            if peak >= entry + 0.5 * initial_risk and _long_exhausted(candle, peak, bars_since_peak):
                if _close_position(client, item, account, args, reason="exhaustion exit"):
                    return True
        else:
            if candle.low < trough:
                trough = candle.low
                bars_since_trough = 0
            else:
                bars_since_trough += 1
            if trough <= entry - 0.5 * initial_risk and _short_exhausted(candle, trough, bars_since_trough):
                if _close_position(client, item, account, args, reason="exhaustion exit"):
                    return True
        item["exhaustion_last_close_time"] = candle.close_time

    item["exhaustion_peak"] = peak
    item["exhaustion_trough"] = trough
    item["exhaustion_bars_since_peak"] = bars_since_peak
    item["exhaustion_bars_since_trough"] = bars_since_trough
    return False


def _long_exhausted(candle: object, peak: float, bars_since_peak: int) -> bool:
    candle_range = candle.high - candle.low
    body = abs(candle.close - candle.open)
    upper_wick = candle.high - max(candle.open, candle.close)
    rejection = (
        candle_range > 0
        and upper_wick >= max(body, candle_range * 1e-6) * 1.5
        and upper_wick >= candle_range * 0.4
        and candle.close < candle.open
        and candle.high >= peak * 0.99
    )
    return rejection or bars_since_peak >= 4


def _short_exhausted(candle: object, trough: float, bars_since_trough: int) -> bool:
    candle_range = candle.high - candle.low
    body = abs(candle.close - candle.open)
    lower_wick = min(candle.open, candle.close) - candle.low
    rejection = (
        candle_range > 0
        and lower_wick >= max(body, candle_range * 1e-6) * 1.5
        and lower_wick >= candle_range * 0.4
        and candle.close > candle.open
        and candle.low <= trough * 1.01
    )
    return rejection or bars_since_trough >= 4


def _maybe_stagnation_exit(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
    account: dict[str, object],
    position: dict[str, object],
    failures: list[str],
) -> bool:
    """Parameterized stagnation exit: once the trade has reached +stagnation_after_r
    of unrealized profit, market-close it if no new favourable extreme has formed
    for stagnation_candles closed bars. Mirrors the backtest mechanism in
    backtest.py:1518-1533, without the buggy 4-candle rejection-candle path that
    makes --exhaustion-exit cut breakout consolidations short.
    """
    after_r = _safe_float(args.stagnation_after_r)
    stall_n = int(args.stagnation_candles)
    if after_r <= 0 or stall_n < 1:
        return False
    symbol = str(item.get("symbol", ""))
    side = str(item.get("side", ""))
    interval = str(item.get("interval", "1h"))
    entry = _safe_float(position.get("entryPrice")) or _safe_float(item.get("live_entry_price"))
    initial_stop = _safe_float(item.get("live_initial_stop")) or _initial_stop_from_item(item)
    initial_risk = abs(entry - initial_stop)
    if not symbol or entry <= 0 or initial_risk <= 0:
        return False

    now_ms = int(time.time() * 1000)
    try:
        lookback = max(int(args.stagnation_lookback), stall_n + 5, 10)
        klines = client.klines(symbol, interval, lookback)
    except BinanceClientError as exc:
        failures.append(f"{symbol}@{interval} stagnation check: {exc}")
        return False
    candles = [c for c in candles_from_klines(klines) if c.close_time <= now_ms]
    if not candles:
        return False

    last_checked = int(_safe_float(item.get("stagnation_last_close_time")))
    entry_filled_at = _safe_float(item.get("entry_filled_at"))
    entry_ms = int(entry_filled_at * 1000) if entry_filled_at > 0 else 0
    fresh = [
        candle
        for candle in candles
        if candle.close_time > last_checked and (entry_ms <= 0 or candle.close_time >= entry_ms)
    ]
    if not fresh:
        return False

    peak = _safe_float(item.get("stagnation_peak")) or entry
    trough = _safe_float(item.get("stagnation_trough")) or entry
    bars_since_peak = int(_safe_float(item.get("stagnation_bars_since_peak")))
    bars_since_trough = int(_safe_float(item.get("stagnation_bars_since_trough")))

    for candle in fresh:
        if side == "LONG":
            if candle.high > peak:
                peak = candle.high
                bars_since_peak = 0
            else:
                bars_since_peak += 1
            if peak >= entry + after_r * initial_risk and bars_since_peak >= stall_n:
                if _close_position(
                    client, item, account, args,
                    reason=f"stagnation exit ({stall_n}c no new high after +{after_r:.2g}R)",
                ):
                    return True
        else:
            if candle.low < trough:
                trough = candle.low
                bars_since_trough = 0
            else:
                bars_since_trough += 1
            if trough <= entry - after_r * initial_risk and bars_since_trough >= stall_n:
                if _close_position(
                    client, item, account, args,
                    reason=f"stagnation exit ({stall_n}c no new low after +{after_r:.2g}R)",
                ):
                    return True
        item["stagnation_last_close_time"] = candle.close_time

    item["stagnation_peak"] = peak
    item["stagnation_trough"] = trough
    item["stagnation_bars_since_peak"] = bars_since_peak
    item["stagnation_bars_since_trough"] = bars_since_trough
    return False


def _trigger_reached(side: str, mark_price: float, trigger_price: float) -> bool:
    if trigger_price <= 0:
        return False
    if side == "LONG":
        return mark_price >= trigger_price
    if side == "SHORT":
        return mark_price <= trigger_price
    return False


def _retest_reached(side: str, mark_price: float, limit_price: float) -> bool:
    if limit_price <= 0:
        return False
    if side == "LONG":
        return mark_price <= limit_price
    if side == "SHORT":
        return mark_price >= limit_price
    return False


def _retest_timeout_reached(item: dict[str, object]) -> bool:
    timeout_seconds = _safe_float(item.get("retest_timeout_seconds"))
    if timeout_seconds <= 0:
        return True
    triggered_at = _safe_float(item.get("triggered_at"))
    return triggered_at > 0 and time.time() - triggered_at >= timeout_seconds


def _apply_pending_account_settings(client: BinanceClient, item: dict[str, object], args: argparse.Namespace) -> None:
    symbol = str(item.get("symbol", ""))
    margin_type = str(item.get("margin_type") or "")
    leverage = int(_safe_float(item.get("leverage")))
    if margin_type:
        try:
            client.change_margin_type(symbol, margin_type, recv_window=args.recv_window)
        except BinanceClientError as exc:
            if "No need to change margin type" not in str(exc):
                raise
    if leverage:
        client.change_leverage(symbol, leverage, recv_window=args.recv_window)


def _entry_order_payload(item: dict[str, object], order_kind: str, limit_price: float = 0.0) -> dict[str, str]:
    entry_client_order_id = str(item.get("entry_client_order_id", "bd_entry"))
    payload = {
        "symbol": str(item.get("symbol", "")),
        "side": str(item.get("binance_side", "")),
        "type": order_kind,
        "quantity": str(item.get("quantity", "")),
        "newClientOrderId": _child_client_order_id(entry_client_order_id, "mkt" if order_kind == "MARKET" else "lim"),
    }
    if order_kind == "LIMIT":
        if limit_price > 0:
            payload["price"] = _match_price_precision(str(item.get("limit_price", "")), limit_price)
        else:
            payload["price"] = str(item.get("limit_price", ""))
        payload["timeInForce"] = "GTC"
    if bool(item.get("hedge_mode", False)):
        payload["positionSide"] = str(item.get("side", ""))
    return payload


# When an entry is rejected for insufficient margin we re-size it to fit the
# free balance. This factor leaves headroom for taker fees and a little
# MARKET-fill slippage so the resized order does not bounce off the same wall.
MARGIN_RESIZE_SAFETY = "0.95"


def _entry_exec_price(item: dict[str, object], order_kind: str, limit_override: float, mark_price: float) -> float:
    """Best estimate of the price an entry will fill at, for margin sizing."""
    if order_kind == "LIMIT":
        if limit_override > 0:
            return limit_override
        return _safe_float(item.get("limit_price"))
    return mark_price


def _rescale_exit_legs(item: dict[str, object], ratio: "Decimal", rule: TradingRule | None) -> None:
    """Scale reduceOnly exit-leg quantities by ``ratio`` after a down-sized
    entry fill so the exits match the smaller position. closePosition stop
    losses carry no quantity and are left untouched; legs that fall below the
    exchange minimum are dropped (the close-all stop still protects the trade).
    """
    from decimal import ROUND_DOWN

    raw_plans = item.get("exit_plans", [])
    if not isinstance(raw_plans, list) or ratio <= 0 or ratio >= 1:
        return
    step = rule.quantity_step_size if rule and rule.quantity_step_size > 0 else _to_decimal("0.00000001")
    min_qty = rule.min_qty if rule else _to_decimal("0")
    min_notional = rule.min_notional if rule else _to_decimal("0")
    entry_price = _to_decimal(item.get("limit_price")) or _to_decimal("0")
    kept: list[object] = []
    for plan in raw_plans:
        if not isinstance(plan, dict):
            kept.append(plan)
            continue
        payload = plan.get("payload", {})
        qty_raw = payload.get("quantity") if isinstance(payload, dict) else None
        if qty_raw in (None, ""):
            kept.append(plan)  # close-all stop loss: no quantity to scale
            continue
        old_qty = _to_decimal(qty_raw) or _to_decimal("0")
        new_qty = _round_to_step(old_qty * ratio, step, rounding=ROUND_DOWN)
        ref_price = _to_decimal(payload.get("triggerPrice")) or entry_price
        if new_qty <= 0 or new_qty < min_qty:
            continue
        if min_notional > 0 and ref_price > 0 and new_qty * ref_price < min_notional:
            continue
        payload["quantity"] = _format_decimal(new_qty)
        kept.append(plan)
    item["exit_plans"] = kept


def _resize_entry_for_available_margin(
    item: dict[str, object],
    order_kind: str,
    exec_price: float,
    account: dict[str, object],
    rule: TradingRule | None,
) -> str | None:
    """Largest entry quantity the account's free margin can currently afford.

    Returns the new quantity string (never larger than the planned size), or
    None when even the exchange-minimum order does not fit. When it shrinks
    the order it also rescales the saved exit legs to match.
    """
    from decimal import ROUND_DOWN

    available = _safe_float(account.get("availableBalance"))
    leverage = int(_safe_float(item.get("leverage"))) or 1
    if available <= 0 or exec_price <= 0 or rule is None:
        return None
    step = rule.quantity_step_size if rule.quantity_step_size > 0 else _to_decimal("0.00000001")
    affordable_notional = _to_decimal(available) * _to_decimal(leverage) * _to_decimal(MARGIN_RESIZE_SAFETY)
    price_dec = _to_decimal(exec_price)
    if price_dec is None or price_dec <= 0:
        return None
    new_qty = _round_to_step(affordable_notional / price_dec, step, rounding=ROUND_DOWN)
    old_qty = _to_decimal(item.get("quantity")) or _to_decimal("0")
    # Only ever shrink. If the affordable size already covers the plan the
    # rejection was not about size, so there is nothing useful to retry.
    if old_qty > 0 and new_qty >= old_qty:
        return None
    if new_qty <= 0 or new_qty < rule.min_qty:
        return None
    if rule.min_notional > 0 and new_qty * price_dec < rule.min_notional:
        return None
    if old_qty > 0:
        _rescale_exit_legs(item, ratio=new_qty / old_qty, rule=rule)
    return _format_decimal(new_qty)


def _place_entry_order(
    client: BinanceClient,
    item: dict[str, object],
    order_kind: str,
    limit_override: float,
    mark_price: float,
    account: dict[str, object],
    trading_rules: dict[str, TradingRule],
    args: argparse.Namespace,
) -> tuple[dict[str, object] | None, str, str]:
    """Place an entry order, down-sizing it to the maximum the free balance
    affords on a 'Margin is insufficient' rejection.

    Returns ``(response, note, error)``: exactly one of ``response``/``error``
    is set; ``note`` is a non-empty status line when the order was resized.
    """
    payload = _entry_order_payload(item, order_kind, limit_price=limit_override)
    try:
        return client.place_order(payload, test=False, recv_window=args.recv_window), "", ""
    except BinanceClientError as exc:
        if "Margin is insufficient" not in str(exc):
            return None, "", str(exc)

    symbol = str(item.get("symbol", ""))
    # Re-fetch the account: earlier entries placed in this same manager tick
    # may already have consumed margin the tick-start snapshot still counts.
    try:
        account = client.account_info(recv_window=args.recv_window)
    except BinanceClientError:
        pass
    rule = trading_rules.get(symbol)
    if rule is None:
        try:
            trading_rules.update(trading_rules_from_exchange_info(client.exchange_info()))
            rule = trading_rules.get(symbol)
        except BinanceClientError:
            rule = None
    exec_price = _entry_exec_price(item, order_kind, limit_override, mark_price)
    new_qty = _resize_entry_for_available_margin(item, order_kind, exec_price, account, rule)
    if not new_qty:
        return None, "", "margin insufficient and free balance is below the exchange minimum order"
    item["quantity"] = new_qty
    resized = _entry_order_payload(item, order_kind, limit_price=limit_override)
    try:
        response = client.place_order(resized, test=False, recv_window=args.recv_window)
    except BinanceClientError as retry_exc:
        return None, "", f"resized {order_kind} entry to qty {new_qty} but still rejected: {retry_exc}"
    return response, f"margin tight; resized {order_kind} entry to maximum affordable qty {new_qty}", ""


def _save_pending_entry_plan(
    path: Path,
    signal: BreakoutSignal,
    entry_plan: ConditionalOrderPlan,
    exit_plans: list[ConditionalOrderPlan],
    args: argparse.Namespace,
    leverage: int = 0,
    entry_regime: str = "RETEST",
    bracket_id: str = "",
) -> None:
    pending = _load_pending_entry_plans(path)
    pending.append(
        {
            "created_at": int(time.time()),
            "state": "WAIT_BREAKOUT",
            "symbol": signal.symbol,
            "side": signal.side,
            "interval": signal.interval,
            "hedge_mode": args.hedge_mode,
            "binance_side": entry_plan.binance_side,
            "quantity": entry_plan.quantity,
            "trigger_price": entry_plan.trigger_price,
            "limit_price": entry_plan.limit_price,
            "retest_timeout_seconds": args.retest_timeout_seconds,
            "entry_stale_minutes": args.entry_stale_minutes,
            "entry_client_order_id": entry_plan.client_order_id,
            "leverage": leverage or args.leverage,
            "margin_type": args.margin_type or "",
            "max_concurrent_orders": args.max_concurrent_orders,
            "max_market_deviation_pct": args.max_market_deviation_pct,
            "no_market_fallback": args.no_market_fallback,
            "entry_regime": entry_regime,
            "trailing_retest_band_pct": _trailing_retest_band_pct(signal) if entry_regime == "TRAILING_RETEST" else 0.0,
            "bracket_id": bracket_id,
            "momentum_score": round(_momentum_score(signal), 4),
            "ml_rank_score": round(_live_ml_rank_score(signal, args), 6) if getattr(args, "ml_rank_model_data", None) else 0.0,
            "ml_rank_score_name": args.ml_rank_score if getattr(args, "ml_rank_model_data", None) else "",
            "dynamic_sl": args.dynamic_sl,
            "sl_update_interval_seconds": args.sl_update_interval_seconds,
            "sl_lookback": args.sl_lookback,
            "exhaustion_exit": args.exhaustion_exit,
            "exhaustion_lookback": args.exhaustion_lookback,
            "stagnation_after_r": args.stagnation_after_r,
            "stagnation_candles": args.stagnation_candles,
            "stagnation_lookback": args.stagnation_lookback,
            "max_sl_loss_pct": args.max_sl_loss_pct,
            "smart_tp": args.smart_tp,
            "smart_tp_max_target_multiplier": args.smart_tp_max_target_multiplier,
            "smart_tp_min_runner_pct": args.smart_tp_min_runner_pct,
            "smart_tp_max_runner_pct": args.smart_tp_max_runner_pct,
            "exit_plans": _serialize_exit_plans(exit_plans),
        }
    )
    _write_pending_entry_plans(path, pending)


def _save_pending_exit_plans(
    path: Path,
    signal: BreakoutSignal,
    entry_plan: ConditionalOrderPlan,
    exit_plans: list[ConditionalOrderPlan],
    args: argparse.Namespace,
) -> None:
    pending = _load_pending_exit_plans(path)
    pending.append(
        {
            "created_at": int(time.time()),
            "symbol": signal.symbol,
            "side": signal.side,
            "interval": signal.interval,
            "hedge_mode": args.hedge_mode,
            "entry_client_order_id": entry_plan.client_order_id,
            "exit_plans": _serialize_exit_plans(exit_plans),
        }
    )
    _write_pending_exit_plans(path, pending)


def _serialize_exit_plans(exit_plans: list[ConditionalOrderPlan]) -> list[dict[str, object]]:
    return [
        {
            "role": plan.role,
            "client_order_id": plan.client_order_id,
            "payload": plan.payload,
        }
        for plan in exit_plans
    ]


def _load_pending_entry_plans(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _write_pending_entry_plans(path: Path, pending: list[dict[str, object]]) -> None:
    if pending:
        path.write_text(json.dumps(pending, indent=2), encoding="utf-8")
    elif path.exists():
        path.unlink()


def _load_pending_exit_plans(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _write_pending_exit_plans(path: Path, pending: list[dict[str, object]]) -> None:
    if pending:
        path.write_text(json.dumps(pending, indent=2), encoding="utf-8")
    elif path.exists():
        path.unlink()


def _load_ml_symbol_context(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    raw_symbols = payload.get("symbols", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw_symbols, dict):
        return {}
    context: dict[str, list[float]] = {}
    for symbol, values in raw_symbols.items():
        if isinstance(values, list):
            context[str(symbol)] = [_safe_float(value) for value in values[-30:]]
    return context


def _write_ml_symbol_context(path: Path, context: dict[str, list[float]]) -> None:
    if not context:
        if path.exists():
            path.unlink()
        return
    payload = {"symbols": {symbol: values[-30:] for symbol, values in sorted(context.items())}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _record_symbol_r(args: argparse.Namespace, symbol: str, r_multiple: float) -> None:
    context = _load_ml_symbol_context(args.ml_context_file)
    values = context.get(symbol, [])
    values.append(float(r_multiple))
    context[symbol] = values[-30:]
    _write_ml_symbol_context(args.ml_context_file, context)
    args._live_ml_symbol_context = context


def _record_symbol_r_from_close(
    args: argparse.Namespace,
    item: dict[str, object],
    position: dict[str, object] | None,
    exit_price: float,
) -> None:
    symbol = str(item.get("symbol", ""))
    side = str(item.get("side", ""))
    entry = _safe_float(position.get("entryPrice")) if position else _safe_float(item.get("live_entry_price"))
    initial_stop = _safe_float(item.get("live_initial_stop"))
    if initial_stop <= 0:
        initial_stop = _initial_stop_from_item(item)
    risk = abs(entry - initial_stop)
    if not symbol or entry <= 0 or exit_price <= 0 or risk <= 0:
        return
    if side == "SHORT":
        r_multiple = (entry - exit_price) / risk
    else:
        r_multiple = (exit_price - entry) / risk
    _record_symbol_r(args, symbol, r_multiple)


def _initial_stop_from_item(item: dict[str, object]) -> float:
    for plan in item.get("exit_plans", []) or []:
        if not isinstance(plan, dict) or plan.get("role") != "STOP_LOSS":
            continue
        payload = plan.get("payload", {})
        if isinstance(payload, dict):
            stop = _safe_float(payload.get("triggerPrice"))
            if stop > 0:
                return stop
    return _safe_float(item.get("sl_trigger_price"))


def _has_open_position(account: dict[str, object], symbol: str, side: str, hedge_mode: bool) -> bool:
    for position in account.get("positions", []):
        if not isinstance(position, dict) or position.get("symbol") != symbol:
            continue
        amount = _safe_float(position.get("positionAmt"))
        position_side = str(position.get("positionSide", "BOTH"))
        if hedge_mode and position_side in {"LONG", "SHORT"}:
            return position_side == side and abs(amount) > 0
        if side == "LONG" and amount > 0:
            return True
        if side == "SHORT" and amount < 0:
            return True
    return False


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _order_execution_from_plan(
    plan: ConditionalOrderPlan,
    response: dict[str, str],
    mode: str,
    requested_margin: float,
    args: argparse.Namespace,
    leverage: int = 0,
) -> OrderExecution:
    effective_leverage = leverage or args.leverage
    order_type = "SMART_RETEST" if args.entry_mode == "SMART_RETEST" and plan.role == "ENTRY" else plan.order_type
    return OrderExecution(
        mode=mode,
        role=plan.role,
        symbol=plan.symbol,
        interval=plan.interval,
        side=plan.signal_side,
        order_type=order_type,
        trigger_price=plan.trigger_price,
        limit_price=plan.limit_price,
        quantity=plan.quantity,
        requested_margin=_order_margin_label(plan, requested_margin),
        estimated_notional=plan.estimated_notional,
        leverage=f"{effective_leverage}x" if effective_leverage else "account",
        margin_type=args.margin_type or "account",
        client_order_id=plan.client_order_id,
        status=str(response.get("algoStatus") or response.get("status") or ("SUBMITTED" if args.live_orders else "LOCAL_VALIDATED")),
        order_id=str(response.get("algoId") or response.get("orderId") or ""),
        message="live algo order submitted" if args.live_orders else "locally validated only; no live order placed",
    )


def _order_margin_label(plan: ConditionalOrderPlan, requested_margin: float) -> str:
    if plan.role == "ENTRY":
        return _compact_money(requested_margin)
    if plan.quantity == "close-all":
        return "close-all"
    return "reduce"


def print_auth_check(
    client: BinanceClient,
    args: argparse.Namespace,
    api_key_name: str | None,
    api_secret_name: str | None,
) -> int:
    print("Binance USD-M futures auth check")
    print(f"env file: {args.env_file} ({'found' if args.env_file.exists() else 'missing'})")
    print(f"base URL: {client.base_url}")
    print(f"testnet: {args.testnet}")
    print(f"api key: {_masked_env_value(api_key_name, client.api_key)}")
    print(f"api secret: {_masked_env_value(api_secret_name, client.api_secret, secret=True)}")

    if not client.api_key or not client.api_secret:
        print("Result: missing API credentials. Expected BINANCE_API_KEY and BINANCE_API_SECRET, or BINANCE_TESTNET_* with --testnet.")
        return 2

    account_ok = False
    try:
        account = client.account_info(recv_window=args.recv_window)
        account_ok = True
        print(
            "signed account: OK  "
            f"wallet={account.get('totalWalletBalance')}  available={account.get('availableBalance')}"
        )
    except BinanceClientError as exc:
        print(f"signed account: FAILED  {exc}")

    if args.auth_check_trade:
        try:
            client.place_order(
                {
                    "symbol": args.auth_check_symbol.upper(),
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": args.auth_check_quantity,
                },
                test=True,
                recv_window=args.recv_window,
            )
            print(f"trade permission test: OK  {args.auth_check_symbol.upper()} MARKET test order accepted")
        except BinanceClientError as exc:
            print(f"trade permission test: FAILED  {exc}")
            if "Invalid API-key, IP, or permissions" in str(exc):
                print("Meaning: Binance is rejecting this key for TRADE access on this endpoint/environment.")
            elif account_ok:
                print("Meaning: the key signs correctly; this is likely order validation, not key loading.")
            return 2

    if not account_ok:
        print("Most likely causes: live/testnet key mismatch, wrong .env variable selected, IP whitelist mismatch, or Futures API permission not enabled.")
        return 2
    print("Auth check complete.")
    return 0


def _resolve_symbols(client: BinanceClient, args: argparse.Namespace) -> SymbolUniverse:
    explicit = _read_symbols(args)
    if explicit:
        return load_explicit_symbols(client, explicit, args.quote)
    min_quote_volume = 0 if args.include_dead else args.min_quote_volume
    min_trade_count = 0 if args.include_dead else args.min_trades
    min_range_pct = 0 if args.include_dead else args.min_24h_range_pct
    return discover_symbols(
        client=client,
        quote_asset=args.quote,
        min_quote_volume=min_quote_volume,
        min_trade_count=min_trade_count,
        min_range_pct=min_range_pct,
        top=args.top,
        include_leveraged=False,
    )


def _filter_order_books(client: BinanceClient, universe: SymbolUniverse, args: argparse.Namespace) -> SymbolUniverse:
    if args.include_dead or args.skip_book_filter:
        return universe
    return filter_by_order_book(
        client=client,
        universe=universe,
        min_depth=args.min_book_depth,
        depth_pct=args.book_depth_pct,
        max_spread_bps=args.max_spread_bps,
        limit=args.book_limit,
        workers=max(min(args.workers, 4), 1),
    )


def _filter_open_interest(client: BinanceClient, universe: SymbolUniverse, args: argparse.Namespace) -> SymbolUniverse:
    if args.include_dead or args.skip_oi_filter:
        return universe
    return filter_by_open_interest(
        client=client,
        universe=universe,
        min_notional=args.min_open_interest_notional,
        workers=max(min(args.workers, 4), 1),
    )


def _select_order_signals(signals: list[BreakoutSignal], args: argparse.Namespace) -> list[BreakoutSignal]:
    selected: list[BreakoutSignal] = []
    seen_symbols: set[str] = set()
    for signal in signals:
        if not args.allow_duplicate_symbol_orders and signal.symbol in seen_symbols:
            continue
        selected.append(signal)
        seen_symbols.add(signal.symbol)
        if len(selected) >= args.order_count:
            break
    return selected


def _fresh_order_signal(
    client: BinanceClient,
    signal: BreakoutSignal,
    args: argparse.Namespace,
    settings: BreakoutSettings,
) -> BreakoutSignal | None:
    now_ms = int(time.time() * 1000)
    history_limit = args.history + 1 if args.closed_candles_only else args.history
    klines = client.klines(signal.symbol, signal.interval, history_limit)
    candles = candles_from_klines(klines)
    if args.closed_candles_only and candles and candles[-1].close_time > now_ms:
        candles = candles[:-1]
    if args.detector == "simple":
        # Re-check with the SAME detector the scan used. A squeeze recheck of a
        # simple-detector breakout fails by construction, so the entry never arms.
        signal = detect_long_breakout(
            symbol=signal.symbol,
            candles=candles,
            quote_volume_24h=signal.quote_volume_24h,
            interval_ms=interval_to_ms(signal.interval),
            interval=signal.interval,
            range_pct_24h=signal.range_pct_24h,
            settings=settings,
            now_ms=now_ms,
        )
    else:
        signal = evaluate_breakout(
            symbol=signal.symbol,
            candles=candles,
            quote_volume_24h=signal.quote_volume_24h,
            interval_ms=interval_to_ms(signal.interval),
            interval=signal.interval,
            trade_count_24h=signal.trade_count_24h,
            range_pct_24h=signal.range_pct_24h,
            price_change_pct_24h=signal.price_change_pct_24h,
            book_min_depth=signal.book_min_depth,
            open_interest_notional=signal.open_interest_notional,
            settings=settings,
            include_confirmed=args.include_confirmed,
            include_rejected=args.include_rejected,
            now_ms=now_ms,
        )
    if signal and getattr(args, "ml_rank_model_data", None):
        contexts = getattr(args, "_live_ml_signal_contexts", None)
        if isinstance(contexts, dict):
            contexts[(signal.symbol, signal.interval)] = _live_market_signal_context(signal, candles, args)
    return signal


def _apply_order_account_settings(client: BinanceClient, signal: BreakoutSignal, args: argparse.Namespace, leverage: int = 0) -> None:
    if args.margin_type:
        try:
            client.change_margin_type(signal.symbol, args.margin_type, recv_window=args.recv_window)
        except BinanceClientError as exc:
            if "No need to change margin type" not in str(exc):
                raise
    effective = leverage or args.leverage
    if effective:
        client.change_leverage(signal.symbol, effective, recv_window=args.recv_window)


def _auto_margin_for_equity(equity: float, peak: float) -> float:
    """Live mirror of backtest._auto_curve_position_pct: returns the USDT margin
    for the next order based on the current wallet equity and running peak.

    Aggressive on small balances (55% on <$25, 45% on $25-100) and tapers as
    the account grows. A drawdown haircut multiplies the base % down to 0.25x
    at >50% drawdown to slow bleeding without halting trading.
    """
    if equity <= 0:
        return 0.0
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
    pct = min(max(base * dd_mult, 5.0), 60.0)
    return equity * pct / 100.0


def _load_equity_peak(path: Path) -> float:
    try:
        if not path.exists():
            return 0.0
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("peak", 0.0))
    except (OSError, ValueError, TypeError):
        return 0.0


def _save_equity_peak(path: Path, peak: float) -> None:
    try:
        path.write_text(
            json.dumps({"peak": peak, "updated_at": int(time.time())}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _current_equity(account: dict[str, object]) -> float:
    """Best-effort wallet equity in USDT from a Binance futures account snapshot."""
    raw = account.get("totalWalletBalance")
    if raw is None:
        raw = account.get("availableBalance")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _resolve_auto_order_margin(args: argparse.Namespace, account: dict[str, object]) -> float:
    """Compute an --sizing-mode auto margin from the live account snapshot, with
    persistent peak tracking. Returns 0 if not in auto mode or equity unavailable.

    Auto-reset: when the wallet jumps by more than --equity-peak-reset-pct in
    one tick the move is treated as a manual withdrawal/deposit (no single
    trade can move the equity that fast under the bot's own sizing) and the
    peak resets to the new equity. This lets the operator move money in or
    out without the drawdown haircut sticking to a stale high.
    """
    if args.sizing_mode != "auto":
        return 0.0
    equity = _current_equity(account)
    if equity <= 0:
        return 0.0
    peak_path = Path(args.equity_peak_file)
    peak = _load_equity_peak(peak_path)
    reset_pct = _safe_float(getattr(args, "equity_peak_reset_pct", 0.0))
    if reset_pct > 0 and peak > 0:
        change_pct = abs(equity - peak) / peak * 100.0
        if change_pct >= reset_pct:
            direction = "withdrawal" if equity < peak else "deposit"
            print(
                f"Auto-sizing: detected {direction} "
                f"(equity {peak:.2f} -> {equity:.2f}, {change_pct:.1f}% change >= "
                f"{reset_pct:.1f}% threshold); resetting peak to current equity."
            )
            peak = equity
            _save_equity_peak(peak_path, peak)
    if equity > peak:
        peak = equity
        _save_equity_peak(peak_path, peak)
    margin = _auto_margin_for_equity(equity, peak)
    return margin


def _order_notional(args: argparse.Namespace) -> float:
    if args.order_notional > 0:
        return args.order_notional
    return args.order_margin * args.leverage


def _requested_margin(args: argparse.Namespace, order_notional: float) -> float:
    if args.order_margin > 0:
        return args.order_margin
    if args.leverage > 0:
        return order_notional / args.leverage
    return order_notional


def _resolve_exit_splits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[list[float], float]:
    if args.tp_splits:
        splits = _parse_percent_list(args.tp_splits, parser)
        args.tp_count = len(splits)
        split_total = sum(splits)
        if args.trailing_stop:
            if split_total >= 100:
                parser.error("--tp-splits must total less than 100 when --trailing-stop is enabled")
            return splits, 100 - split_total
        if abs(split_total - 100) > 0.0001:
            parser.error("--tp-splits must total exactly 100 unless --trailing-stop is enabled")
        return splits, 0.0

    if args.trailing_stop:
        tp_total = 100 - args.trailing_quantity_pct
        if tp_total <= 0:
            parser.error("--trailing-quantity-pct leaves no quantity for take profits")
        return [tp_total / args.tp_count for _ in range(args.tp_count)], args.trailing_quantity_pct
    return [100 / args.tp_count for _ in range(args.tp_count)], 0.0


def _take_profit_profile_for_signal(signal: BreakoutSignal, args: argparse.Namespace) -> TakeProfitProfile:
    if args.smart_tp:
        return smart_take_profit_profile(
            signal,
            tp_count=args.tp_count,
            trailing_stop=args.trailing_stop,
            base_runner_pct=args.trailing_runner_pct,
            max_target_multiplier=args.smart_tp_max_target_multiplier,
            min_runner_pct=args.smart_tp_min_runner_pct,
            max_runner_pct=args.smart_tp_max_runner_pct,
        )
    return equal_take_profit_profile(signal, args.tp_count, args.trailing_stop, args.trailing_runner_pct)


def _parse_percent_list(raw: str, parser: argparse.ArgumentParser) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError:
        parser.error("--tp-splits must be comma-separated numbers")
    if not values:
        parser.error("--tp-splits cannot be empty")
    if any(value <= 0 for value in values):
        parser.error("--tp-splits values must be positive")
    if sum(values) > 100:
        parser.error("--tp-splits cannot total more than 100")
    return values


ENTRY_REGIMES = {"INSTANT", "RETEST", "STRICT_RETEST", "TRAILING_RETEST"}


def _parse_entry_regime_set(raw: str, parser: argparse.ArgumentParser) -> set[str]:
    if raw.strip().lower() in {"", "none", "off"}:
        return set()
    regimes = {item.strip().upper() for item in raw.split(",") if item.strip()}
    unknown = regimes - ENTRY_REGIMES
    if unknown:
        parser.error(f"--skip-entry-regimes contains unknown regime(s): {', '.join(sorted(unknown))}")
    return regimes


def _current_btc_guard_point(client: BinanceClient, args: argparse.Namespace) -> tuple[float, float, float] | None:
    """Return latest BTC close, EMA, and momentum percentage for live market guards."""
    context = _current_btc_ml_context(client, args)
    if not context:
        return None
    close = float(context.get("_btc_close", 0.0))
    ema = float(context.get("_btc_ema", 0.0))
    momentum_pct = float(context.get("feat_btc_momentum_pct", 0.0))
    if close <= 0 or ema <= 0:
        return None
    return close, ema, momentum_pct


def _current_btc_ml_context(client: BinanceClient, args: argparse.Namespace) -> dict[str, float]:
    """Return the BTC context features used by both guards and ML ranking."""
    limit = min(max(args.btc_ema_candles, args.btc_momentum_candles) + 80, 1500)
    try:
        candles = candles_from_klines(client.klines("BTCUSDT", args.btc_guard_interval, limit))
    except BinanceClientError:
        return {}
    if len(candles) <= args.btc_momentum_candles:
        return {}

    alpha = 2.0 / (args.btc_ema_candles + 1.0)
    ema = candles[0].close
    for candle in candles:
        ema = candle.close * alpha + ema * (1.0 - alpha)
    latest = candles[-1]
    previous = candles[-1 - args.btc_momentum_candles]
    momentum_pct = (latest.close / max(previous.close, 1e-9) - 1.0) * 100.0
    ema_distance_pct = (latest.close / max(ema, 1e-9) - 1.0) * 100.0
    return {
        "_btc_close": latest.close,
        "_btc_ema": ema,
        "feat_btc_momentum_pct": momentum_pct,
        "feat_btc_ema_distance_pct": ema_distance_pct,
    }


def _btc_guard_reject_reason(
    entry_regime: str,
    guard_point: tuple[float, float, float] | None,
    args: argparse.Namespace,
) -> str:
    if not args.btc_market_guards:
        return ""
    if guard_point is None:
        return "" if entry_regime == "STRICT_RETEST" else "BTC hostile/unavailable; only STRICT_RETEST allowed"

    close, ema, momentum_pct = guard_point
    if entry_regime == "INSTANT":
        ema_floor = ema * (1.0 - max(args.instant_guard_ema_slack_pct, 0.0) / 100.0)
        if momentum_pct < args.instant_guard_momentum_pct or close < ema_floor:
            return (
                f"BTC guard blocked INSTANT "
                f"(momentum {momentum_pct:.2f}%, close/EMA {(close / max(ema, 1e-9) - 1.0) * 100:.2f}%)"
            )

    ema_floor = ema * (1.0 - max(args.hostile_ema_slack_pct, 0.0) / 100.0)
    hostile = momentum_pct < args.hostile_momentum_pct or close < ema_floor
    if hostile and entry_regime != "STRICT_RETEST":
        return (
            f"BTC hostile; only STRICT_RETEST allowed "
            f"(momentum {momentum_pct:.2f}%, close/EMA {(close / max(ema, 1e-9) - 1.0) * 100:.2f}%)"
        )
    return ""


def _client_order_id(signal: BreakoutSignal, index: int, prefix: str) -> str:
    safe_prefix = "".join(char for char in prefix if char.isalnum() or char in "._-")[:8] or "bd"
    stamp = str(int(time.time() * 1000) % 10_000_000)
    text = f"{safe_prefix}_{signal.symbol[:14]}_{signal.side[0]}{signal.interval}_{index}_{stamp}"
    return text[:36]


def _child_client_order_id(parent_id: str, suffix: str) -> str:
    clean_suffix = "".join(char for char in suffix if char.isalnum() or char in "._-")[:4] or "x"
    return f"{parent_id[: 35 - len(clean_suffix)]}_{clean_suffix}"[:36]


def _load_env_file(path: Path | None) -> None:
    if not path or not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("'\"")
        if key:
            os.environ[key] = value


def _first_env(names: tuple[str, ...]) -> str | None:
    return _first_env_match(names)[1]


def _first_env_match(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return name, value
    return None, None


def _masked_env_value(name: str | None, value: str | None, secret: bool = False) -> str:
    if not name or not value:
        return "not found"
    if secret:
        return f"{name}=present len={len(value)}"
    if len(value) <= 8:
        masked = "*" * len(value)
    else:
        masked = f"{value[:4]}...{value[-4:]}"
    return f"{name}={masked} len={len(value)}"


def _read_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols.extend(args.symbols.split(","))
    if args.symbols_file:
        symbols.extend(args.symbols_file.read_text(encoding="utf-8").splitlines())
    return [symbol.strip() for symbol in symbols if symbol.strip() and not symbol.strip().startswith("#")]


def _resolve_intervals(args: argparse.Namespace) -> list[str]:
    raw = args.interval if args.interval else args.timeframes
    intervals: list[str] = []
    for interval in raw.split(","):
        normalized = interval.strip()
        if normalized and normalized not in intervals:
            interval_to_ms(normalized)
            intervals.append(normalized)
    if not intervals:
        raise ValueError("at least one interval is required")
    return intervals


def _interval_rank(interval: str) -> int:
    return {
        "4h": 0,
        "1h": 1,
        "15m": 2,
        "5m": 3,
    }.get(interval, 99)


ADAPTIVE_LOW_LIQUIDITY_24H_QV = 15_000_000.0
ADAPTIVE_ABNORMAL_VOLUME_RATIO = 3.0
ADAPTIVE_STRONG_TREND_SCORE = 0.85
INSTANT_MAX_DEVIATION_PCT = 5.0
# A coin is "dead" (never traded by --adaptive-entry) if it is barely traded
# or barely moves. Size alone is not disqualifying: a small but volatile,
# actively traded coin still passes.
ADAPTIVE_DEAD_MIN_24H_QV = 20_000_000.0   # below this 24h quote volume = thinly traded
ADAPTIVE_DEAD_MIN_ATR_PCT = 0.010         # below this ATR% = flat / no volatility
ADAPTIVE_DEAD_MIN_24H_RANGE_PCT = 8.0     # a coin with a wider 24h range is hot, not flat
ADAPTIVE_LIQUIDITY_RANK_CAP = 200_000_000.0  # 24h quote volume that maxes the rank term


def _dead_coin_reason(signal: BreakoutSignal) -> str:
    """Return why a coin counts as dead (untradeable), or '' if it is alive."""
    if 0.0 < signal.quote_volume_24h < ADAPTIVE_DEAD_MIN_24H_QV:
        return f"24h volume ${signal.quote_volume_24h / 1e6:.1f}M too thin"
    # Flat only if recent ATR is low AND the 24h range is small - a coin that pumped
    # and is now consolidating has a low ATR but a big 24h range, and is not dead.
    if signal.atr_pct < ADAPTIVE_DEAD_MIN_ATR_PCT and signal.range_pct_24h < ADAPTIVE_DEAD_MIN_24H_RANGE_PCT:
        return f"ATR {signal.atr_pct * 100:.2f}% + 24h range {signal.range_pct_24h:.1f}% too flat"
    return ""


def _classify_entry_regime(signal: BreakoutSignal) -> str:
    """Pick an entry execution mode from the coin's current market regime."""
    if 0.0 < signal.quote_volume_24h < ADAPTIVE_LOW_LIQUIDITY_24H_QV:
        return "STRICT_RETEST"
    if signal.volume_ratio >= ADAPTIVE_ABNORMAL_VOLUME_RATIO:
        return "INSTANT"
    if signal.trend_score >= ADAPTIVE_STRONG_TREND_SCORE:
        return "TRAILING_RETEST"
    return "RETEST"


def _trailing_retest_band_pct(signal: BreakoutSignal) -> float:
    """Shallow-pullback band for a trailing retest, derived from ATR."""
    band = signal.atr_pct * 100.0 * 0.5
    return round(min(max(band, 0.3), 1.5), 3)


def _trailing_retest_limit(item: dict[str, object], side: str, mark_price: float, orig_limit: float) -> float:
    """Ratchet the retest limit toward the breakout extreme so a shallow pullback fills."""
    band = _safe_float(item.get("trailing_retest_band_pct")) / 100.0
    if band <= 0:
        return orig_limit
    if side == "LONG":
        peak = max(_safe_float(item.get("retest_peak")), mark_price)
        item["retest_peak"] = peak
        return max(orig_limit, peak * (1.0 - band))
    if side == "SHORT":
        prev = _safe_float(item.get("retest_trough"))
        trough = mark_price if prev <= 0 else min(prev, mark_price)
        item["retest_trough"] = trough
        trailing = trough * (1.0 + band)
        return min(orig_limit, trailing) if orig_limit > 0 else trailing
    return orig_limit


def _match_price_precision(reference: str, value: float) -> str:
    ref = str(reference).strip()
    decimals = len(ref.split(".", 1)[1]) if "." in ref else 0
    return f"{value:.{decimals}f}"


COILING_TREND_MIN = 0.35
COILING_TREND_MAX = 0.70
COILING_CENTER_MIN = 0.25
COILING_CENTER_MAX = 0.75


def _is_coiling_no_bias(signal: BreakoutSignal) -> bool:
    """True when a coin is consolidating with no clear directional bias."""
    if signal.status not in ("PRE_BREAKOUT", "PRE_BREAKDOWN"):
        return False
    if not (COILING_TREND_MIN <= signal.trend_score <= COILING_TREND_MAX):
        return False
    span = signal.resistance - signal.support
    if span <= 0:
        return False
    position = (signal.close - signal.support) / span
    return COILING_CENTER_MIN <= position <= COILING_CENTER_MAX


def _bracket_signals(
    signal: BreakoutSignal, settings: BreakoutSettings
) -> tuple[BreakoutSignal, BreakoutSignal] | None:
    """Build mirrored LONG and SHORT signals for a two-sided breakout bracket."""
    resistance = signal.resistance
    support = signal.support
    if resistance <= 0 or support <= 0 or support >= resistance:
        return None
    span = resistance - support
    trigger_buffer = max(settings.entry_buffer_pct, signal.atr_pct * settings.entry_atr_buffer_multiple)
    stop_buffer = settings.stop_buffer_pct

    long_trigger = resistance * (1 + trigger_buffer)
    long_signal = replace(
        signal,
        side="LONG",
        status="PRE_BREAKOUT",
        order_type="BUY STOP_MARKET",
        trigger_price=long_trigger,
        stop_price=support * (1 - stop_buffer),
        target_price=long_trigger + span,
    )

    short_trigger = support * (1 - trigger_buffer)
    short_target = short_trigger - span
    if short_target <= 0:
        return None
    short_signal = replace(
        signal,
        side="SHORT",
        status="PRE_BREAKDOWN",
        order_type="SELL STOP_MARKET",
        trigger_price=short_trigger,
        stop_price=resistance * (1 + stop_buffer),
        target_price=short_target,
    )
    return long_signal, short_signal


def _place_bracket_side(
    signal: BreakoutSignal,
    rule: TradingRule,
    index: int,
    args: argparse.Namespace,
    mode: str,
    leverage: int,
    notional: float,
    margin: float,
    entry_regime: str,
    bracket_id: str,
    results: list[OrderExecution],
    failures: list[str],
) -> bool:
    """Build and persist one managed SMART_RETEST entry side of a two-sided bracket."""
    signal = _with_leverage_capped_stop(signal, leverage, args.max_sl_loss_pct)
    try:
        plan = build_entry_order_plan(
            signal=signal,
            rule=rule,
            requested_notional=notional,
            client_order_id=_client_order_id(signal, index, args.client_order_prefix),
            working_type=args.order_working_type,
            price_protect=args.order_price_protect,
            hedge_mode=args.hedge_mode,
            entry_mode="RETEST_LIMIT",
            entry_pullback_pct=args.entry_pullback_pct,
        )
        exit_plans: list[ConditionalOrderPlan] = []
        if not args.no_exits:
            tp_profile = _take_profit_profile_for_signal(signal, args)
            exit_signal = tp_profile.signal
            exit_plans = build_exit_order_plans(
                signal=exit_signal,
                rule=rule,
                entry_quantity=plan.quantity,
                stop_client_order_id=_child_client_order_id(plan.client_order_id, "sl"),
                target_client_order_ids=[
                    _child_client_order_id(plan.client_order_id, f"tp{target_index}")
                    for target_index in range(1, len(tp_profile.tp_splits_pct) + 1)
                ],
                target_splits_pct=tp_profile.tp_splits_pct,
                trailing_client_order_id=_child_client_order_id(plan.client_order_id, "trl") if args.trailing_stop else None,
                trailing_callback_pct=args.trailing_callback_pct if args.trailing_stop else None,
                trailing_quantity_pct=tp_profile.runner_pct,
                working_type=args.order_working_type,
                price_protect=args.order_price_protect,
                hedge_mode=args.hedge_mode,
            )
    except (BinanceClientError, OrderPlanError) as exc:
        failures.append(f"{signal.symbol}@{signal.interval} {signal.side} bracket side: {exc}")
        return False

    results.append(_order_execution_from_plan(plan, {"algoStatus": "WAIT_BREAKOUT"}, mode, margin, args, leverage=leverage))
    _save_pending_entry_plan(
        path=args.entry_state_file,
        signal=exit_signal if exit_plans else signal,
        entry_plan=plan,
        exit_plans=exit_plans,
        args=args,
        leverage=leverage,
        entry_regime=entry_regime,
        bracket_id=bracket_id,
    )
    for exit_plan in exit_plans:
        results.append(_order_execution_from_plan(exit_plan, {"algoStatus": "DEFERRED_UNTIL_ENTRY_FILLS"}, mode, margin, args, leverage=leverage))
    return True


# Slot/queue tuning for the continuous auto-trader.
HIGH_CONVICTION_MAX_COMPRESSION = 0.08   # compression fraction at/under which a coil is "loaded"
HIGH_CONVICTION_MIN_VOL_RATIO = 1.3      # volume building into the coil
ROTATION_MIN_EDGE = 0.20                 # an exploding coin must beat a held position by this margin
ROTATION_FEE_RATE = 0.0005               # estimated cost to market-close a position (fraction of notional)
ROTATION_PROMPT_TIMEOUT = 120            # seconds to wait on the Windows cut-loss prompt
ROTATION_MIN_HOLD_SECONDS = 90 * 60      # a position must run this long before it can be rotated out
ROTATION_COOLDOWN_SECONDS = 30 * 60      # minimum gap between rotations, so a fast scan cannot thrash

# Wall-clock time of the last completed rotation; gates ROTATION_COOLDOWN_SECONDS.
_last_rotation_ts = 0.0
# Throttle "no qualifying signals" prints to at most one per hour so a quiet
# market does not spam the log. Active scans print normally and do not reset
# this; the user still sees an hourly heartbeat that the bot is alive.
_LAST_QUIET_HEARTBEAT_TS = 0.0
_QUIET_HEARTBEAT_INTERVAL_SECONDS = 60 * 60


LIVE_ML_NUMERIC_FEATURES = {
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
}


def _load_ml_rank_model(path: Path) -> dict[str, object]:
    model = json.loads(path.read_text(encoding="utf-8"))
    feature_names = list(model.get("feature_names_numeric") or [])
    unsupported = sorted(name for name in feature_names if name not in LIVE_ML_NUMERIC_FEATURES)
    if unsupported:
        raise ValueError(
            "model uses offline-only features; retrain with train_model.py --feature-set live. "
            f"Unsupported: {', '.join(unsupported)}"
        )
    regime_names = list(model.get("feature_names_regime_dummies") or [])
    expected_len = len(feature_names) + len(regime_names)
    for key in ("scaler_mean", "scaler_scale"):
        values = model.get(key)
        if not isinstance(values, list) or len(values) != expected_len:
            raise ValueError(f"{key} length does not match model feature count")
    heads = model.get("heads")
    if not isinstance(heads, dict):
        raise ValueError("missing model heads")
    return model


def _live_market_signal_context(signal: BreakoutSignal, candles: list[object], args: argparse.Namespace) -> dict[str, float]:
    btc_context = dict(getattr(args, "_live_ml_btc_context", {}) or {})
    context = {
        "feat_btc_momentum_pct": float(btc_context.get("feat_btc_momentum_pct", 0.0)),
        "feat_btc_ema_distance_pct": float(btc_context.get("feat_btc_ema_distance_pct", 0.0)),
        "feat_rel_momentum_pct": 0.0,
    }
    lookback = max(int(getattr(args, "btc_momentum_candles", 0) or 0), 0)
    if lookback > 0 and len(candles) > lookback:
        latest = candles[-1]
        previous = candles[-1 - lookback]
        coin_momentum = (latest.close / max(previous.close, 1e-9) - 1.0) * 100.0
        context["feat_rel_momentum_pct"] = coin_momentum - context["feat_btc_momentum_pct"]
    return context


def _live_symbol_context(signal: BreakoutSignal, args: argparse.Namespace) -> dict[str, float]:
    raw_context = getattr(args, "_live_ml_symbol_context", {}) or {}
    recent = raw_context.get(signal.symbol, []) if isinstance(raw_context, dict) else []
    if not isinstance(recent, list):
        recent = []
    values = [_safe_float(value) for value in recent[-30:]]
    return {
        "feat_symbol_trades_30": float(len(values)),
        "feat_symbol_win_rate_30": (sum(1 for value in values if value > 0) / len(values)) if values else 0.5,
        "feat_symbol_avg_r_30": (sum(values) / len(values)) if values else 0.0,
    }


def _live_ml_feature_map(signal: BreakoutSignal, args: argparse.Namespace) -> dict[str, float]:
    feature_map = {
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
    signal_contexts = getattr(args, "_live_ml_signal_contexts", {}) or {}
    if isinstance(signal_contexts, dict):
        feature_map.update(signal_contexts.get((signal.symbol, signal.interval), {}))
    feature_map.update(_live_symbol_context(signal, args))
    return feature_map


def _live_ml_feature_vector(signal: BreakoutSignal, model: dict[str, object], args: argparse.Namespace) -> list[float]:
    feature_map = _live_ml_feature_map(signal, args)
    features = [feature_map.get(name, 0.0) for name in model.get("feature_names_numeric", [])]
    regime = _classify_entry_regime(signal)
    for name in model.get("feature_names_regime_dummies", []):
        features.append(1.0 if regime == name else 0.0)
    return [float(value) for value in features]


def _live_ml_apply_head(features: list[float], model: dict[str, object], head: dict[str, object] | None) -> float | None:
    if not head:
        return None
    z = float(head["intercept"])
    coef = list(head["coef"])
    scaler_mean = list(model["scaler_mean"])
    scaler_scale = list(model["scaler_scale"])
    for i, x in enumerate(features):
        scale = float(scaler_scale[i]) if float(scaler_scale[i]) > 0 else 1.0
        z += (x - float(scaler_mean[i])) / scale * float(coef[i])
    if head.get("kind") == "logistic":
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))
    return z


def _live_ml_scores_with_args(signal: BreakoutSignal, model: dict[str, object], args: argparse.Namespace) -> dict[str, float]:
    features = _live_ml_feature_vector(signal, model, args)
    heads = model.get("heads") or {}
    if not isinstance(heads, dict):
        return {}
    pwin = _live_ml_apply_head(features, model, heads.get("pwin"))
    expected_r = _live_ml_apply_head(features, model, heads.get("expected_r"))
    tail = _live_ml_apply_head(features, model, heads.get("tail"))
    bad = _live_ml_apply_head(features, model, heads.get("bad"))
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
    }


def _live_ml_rank_score(signal: BreakoutSignal, args: argparse.Namespace) -> float:
    model = getattr(args, "ml_rank_model_data", None)
    if not model:
        return 0.0
    return _live_ml_scores_with_args(signal, model, args).get(args.ml_rank_score, 0.0)


def _rank_order_signals(signals: list[BreakoutSignal], args: argparse.Namespace) -> None:
    # Sort by momentum so the priority key matches what the rotation system
    # and stale-entry watchdog use (momentum_score). Previously this was gated
    # on --adaptive-entry, so without that flag the queue was ranked by
    # quality_score while rotation compared by momentum_score - the "best"
    # signal in the queue could be a different one than rotation would
    # actually swap to. Quality sort remains available as a tiebreaker.
    fallback_key = _momentum_sort_key
    if getattr(args, "ml_rank_model_data", None):
        signals.sort(key=lambda signal: (-_live_ml_rank_score(signal, args), *fallback_key(signal)))
    else:
        signals.sort(key=fallback_key)


def _momentum_score(signal: BreakoutSignal) -> float:
    """Scalar momentum rank: relative surge + volatility + trend + absolute liquidity."""
    surge = min(signal.volume_ratio / ADAPTIVE_ABNORMAL_VOLUME_RATIO, 2.0)
    volatility = min(signal.atr_pct / 0.04, 1.5)
    liquidity = min(signal.quote_volume_24h / ADAPTIVE_LIQUIDITY_RANK_CAP, 1.0)
    return surge * 0.34 + volatility * 0.26 + signal.trend_score * 0.16 + liquidity * 0.24


def _is_high_conviction(signal: BreakoutSignal) -> bool:
    """A loaded coil: pre-breakout, tightly compressed, with volume building."""
    return (
        signal.status in ("PRE_BREAKOUT", "PRE_BREAKDOWN")
        and 0.0 < signal.compression_pct <= HIGH_CONVICTION_MAX_COMPRESSION
        and signal.volume_ratio >= HIGH_CONVICTION_MIN_VOL_RATIO
    )


def _momentum_sort_key(signal: BreakoutSignal) -> tuple[float, float, float, int, str]:
    return (
        -_momentum_score(signal),
        -signal.score,
        -signal.reward_risk,
        _interval_rank(signal.interval),
        signal.symbol,
    )


def _quality_sort_key(signal: BreakoutSignal) -> tuple[float, float, int, float, int, int, str]:
    return (
        -signal.score,
        -signal.reward_risk,
        _interval_rank(signal.interval),
        signal.distance_to_trigger_pct,
        _setup_quality_rank(signal.status),
        0 if signal.side == "LONG" else 1,
        signal.symbol,
    )


def _setup_quality_rank(status: str) -> int:
    return {
        "SPRING": 0,
        "UPTHRUST": 1,
        "PRE_BREAKOUT": 2,
        "PRE_BREAKDOWN": 3,
        "BREAKOUT": 4,
        "BREAKDOWN": 5,
    }.get(status, 99)


def _print_table(headers: list[str], rows: list[list[str]], right_align: set[str] | None = None) -> None:
    right_align = right_align or set()
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    print("  ".join(_align(header, width, header in right_align) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(_align(cell, width, header in right_align) for header, cell, width in zip(headers, row, widths)))


def _align(value: str, width: int, right: bool) -> str:
    return value.rjust(width) if right else value.ljust(width)


def _price(value: float) -> str:
    if value >= 100:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.4f}"
    if value >= 0.01:
        return f"{value:.6f}"
    return f"{value:.8f}"


def _compact_money(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def _swap_sl_placed_id(item: dict[str, object], old_id: str, new_id: str) -> None:
    placed_ids = item.get("placed_exit_client_order_ids")
    if isinstance(placed_ids, list):
        item["placed_exit_client_order_ids"] = [
            new_id if str(client_id) == old_id else client_id for client_id in placed_ids
        ]


def _maybe_reposition_sl(
    client: BinanceClient,
    item: dict[str, object],
    args: argparse.Namespace,
    rule: TradingRule,
    account: dict[str, object],
    failures: list[str],
) -> None:
    from decimal import Decimal, ROUND_DOWN, ROUND_UP

    symbol = str(item.get("symbol", ""))
    side = str(item.get("side", ""))
    interval = str(item.get("interval", ""))
    sl_client_order_id = str(item.get("sl_client_order_id", ""))
    current_sl = _safe_float(item.get("sl_trigger_price"))
    last_update = _safe_float(item.get("last_sl_update"))

    if not sl_client_order_id or current_sl <= 0:
        return
    if time.time() - last_update < args.sl_update_interval_seconds:
        return

    item["last_sl_update"] = time.time()

    try:
        klines = client.klines(symbol, interval, args.sl_lookback + 5)
        mark = client.mark_price(symbol)
    except BinanceClientError:
        return

    candles = candles_from_klines(klines)
    if len(candles) < args.sl_lookback:
        return

    window = candles[-args.sl_lookback:]
    stop_buffer = args.stop_buffer_pct / 100
    max_loss_pct = _safe_float(item.get("max_sl_loss_pct")) or args.max_sl_loss_pct
    leverage = int(_safe_float(item.get("leverage")) or args.leverage or 1)
    entry_ref = _safe_float(item.get("trigger_price"))

    # Breakeven ratchet: once unrealized profit reaches breakeven_trigger_r x
    # initial_risk, pull the stop to entry + a tiny profit lock. Validated by a
    # 12-run sweep across 3 regime windows (compounded 61x -> 117x at +1.5R).
    breakeven_sl = 0.0
    if _safe_float(args.breakeven_trigger_r) > 0:
        position = _account_position(
            account, symbol, side, bool(item.get("hedge_mode", False))
        )
        entry_price = _safe_float(position.get("entryPrice")) if position else 0.0
        initial_stop = 0.0
        for plan in item.get("exit_plans", []) or []:
            if isinstance(plan, dict) and plan.get("role") == "STOP_LOSS":
                pl = plan.get("payload", {})
                if isinstance(pl, dict):
                    initial_stop = _safe_float(pl.get("triggerPrice"))
                break
        if entry_price > 0 and initial_stop > 0:
            initial_risk = abs(entry_price - initial_stop)
            if initial_risk > 0:
                trigger_r = args.breakeven_trigger_r
                offset = args.breakeven_offset_pct / 100.0
                if side == "LONG" and mark >= entry_price + trigger_r * initial_risk:
                    breakeven_sl = entry_price * (1.0 + offset)
                elif side == "SHORT" and mark <= entry_price - trigger_r * initial_risk:
                    breakeven_sl = entry_price * (1.0 - offset)

    if side == "LONG":
        support = min(c.low for c in window)
        new_sl = support * (1 - stop_buffer)
        new_sl = _leverage_capped_stop(side, entry_ref, new_sl, leverage, max_loss_pct)
        if breakeven_sl > 0 and breakeven_sl > new_sl:
            new_sl = breakeven_sl  # breakeven floor outranks the swing-low SL
        if new_sl <= current_sl:
            return
        if new_sl >= mark * 0.995:
            return
        rounding = ROUND_DOWN
    else:
        resistance = max(c.high for c in window)
        new_sl = resistance * (1 + stop_buffer)
        new_sl = _leverage_capped_stop(side, entry_ref, new_sl, leverage, max_loss_pct)
        if breakeven_sl > 0 and breakeven_sl < new_sl:
            new_sl = breakeven_sl  # breakeven ceiling outranks the swing-high SL
        if new_sl >= current_sl:
            return
        if new_sl <= mark * 1.005:
            return
        rounding = ROUND_UP

    tick = rule.price_tick_size
    new_sl_dec = Decimal(str(new_sl))
    if tick > 0:
        rounded = (new_sl_dec / tick).to_integral_value(rounding=rounding) * tick
    else:
        rounded = new_sl_dec
    normalized = rounded.normalize()
    formatted_sl = format(normalized, "f")
    if "." in formatted_sl:
        formatted_sl = formatted_sl.rstrip("0").rstrip(".")

    # The raw comparison passed because the float was fractionally better, but
    # after tick-size rounding the new SL equals the existing one. Suppress the
    # no-op cancel+replace cycle (and the matching log spam) - it wastes API
    # weight and clutters the terminal every dynamic-sl tick.
    rounded_new = float(formatted_sl) if formatted_sl else 0.0
    if rounded_new > 0:
        if side == "LONG" and rounded_new <= current_sl:
            return
        if side == "SHORT" and rounded_new >= current_sl:
            return

    hedge_mode = bool(item.get("hedge_mode", False))
    order_side = "SELL" if side == "LONG" else "BUY"

    def _stop_payload(trigger_str: str) -> dict[str, str]:
        payload: dict[str, str] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": order_side,
            "type": "STOP_MARKET",
            "triggerPrice": trigger_str,
            "workingType": "MARK_PRICE",
            "priceProtect": "false",
            "closePosition": "true",
            "newOrderRespType": "ACK",
            "clientAlgoId": f"bd_dsl_{int(time.time() * 1000) % 10_000_000_000}",
        }
        if hedge_mode:
            payload["positionSide"] = side
        return payload

    # Binance rejects a second closePosition stop in the same direction, so the
    # old stop must be cancelled before the new one is placed. If the new placement
    # fails, the original stop is restored so the position is never left unprotected.
    try:
        client.cancel_algo_order(symbol, sl_client_order_id, recv_window=args.recv_window)
    except BinanceClientError as exc:
        if "Unknown order" not in str(exc) and "Order does not exist" not in str(exc):
            failures.append(f"{symbol} dynamic SL: could not cancel old stop, keeping it: {exc}")
            return

    new_payload = _stop_payload(formatted_sl)
    try:
        client.place_algo_order(new_payload, recv_window=args.recv_window)
    except BinanceClientError as exc:
        restore_payload = _stop_payload(str(item.get("sl_trigger_price") or formatted_sl))
        try:
            client.place_algo_order(restore_payload, recv_window=args.recv_window)
            item["sl_client_order_id"] = str(restore_payload["clientAlgoId"])
            _swap_sl_placed_id(item, sl_client_order_id, str(restore_payload["clientAlgoId"]))
            failures.append(f"{symbol} dynamic SL move failed, original stop restored: {exc}")
        except BinanceClientError as restore_exc:
            failures.append(
                f"{symbol} CRITICAL: dynamic SL move failed and the stop could not be "
                f"restored - position may be unprotected: {restore_exc}"
            )
        return

    item["sl_client_order_id"] = str(new_payload["clientAlgoId"])
    item["sl_trigger_price"] = formatted_sl
    _swap_sl_placed_id(item, sl_client_order_id, str(new_payload["clientAlgoId"]))
    print(f"{symbol} dynamic SL: {current_sl:.8g} -> {formatted_sl}  (mark={mark:.8g})")


def _with_leverage_capped_stop(signal: BreakoutSignal, leverage: int, max_loss_pct: float) -> BreakoutSignal:
    capped_stop = _leverage_capped_stop(
        side=signal.side,
        entry=signal.trigger_price,
        stop=signal.stop_price,
        leverage=leverage,
        max_loss_pct=max_loss_pct,
    )
    if capped_stop == signal.stop_price:
        return signal
    if signal.side == "LONG":
        risk_pct = max(signal.trigger_price / max(capped_stop, 1e-12) - 1.0, 0.0)
    else:
        risk_pct = max(capped_stop / max(signal.trigger_price, 1e-12) - 1.0, 0.0)
    reward_risk = signal.reward_pct / risk_pct if risk_pct > 0 else signal.reward_risk
    return replace(signal, stop_price=capped_stop, risk_pct=risk_pct, reward_risk=reward_risk)


def _leverage_capped_stop(side: str, entry: float, stop: float, leverage: int, max_loss_pct: float) -> float:
    if max_loss_pct <= 0 or entry <= 0 or leverage <= 0:
        return stop
    max_price_risk = max_loss_pct / 100.0 / leverage
    if max_price_risk <= 0:
        return stop
    if side == "LONG":
        return max(stop, entry * (1.0 - max_price_risk))
    return min(stop, entry * (1.0 + max_price_risk))


def _dynamic_leverage(atr_pct: float, risk_pct: float, base: int, conviction: float = 1.0) -> int:
    """Conviction-scaled dynamic leverage.

    The strategy's edge is asymmetric: a small minority of trades produce the
    bulk of returns. To beat flat-base leverage, dynamic must *concentrate*
    capital on high-conviction setups (where momentum_score is high) and
    de-emphasize weak setups - not just normalize risk across all trades.

    Conviction (momentum_score 0..~2) drives the multiplier:
      >= 1.5  -> base * 1.6  (S+ tier explosive breakouts)
      >= 1.0  -> base * 1.3  (strong)
      >= 0.5  -> base        (normal)
      <  0.5  -> base * 0.8  (weak)

    Safety: a liquidation-buffer cap limits per-trade max-loss to LIQUIDATION_BUFFER
    of margin so Binance never auto-liquidates before the configured SL fires.
    The function is purely conviction-driven now - stop distance no longer
    scales leverage, because a "risk-parity" approach equalizes exposure and
    destroys the strategy's fat-tail edge (lost 10x to flat in the prior
    iteration).

    The previous fixed atr -> vol-factor table downscaled high-ATR coins,
    which broke worst of all - those are exactly where the breakout runners
    come from. The history of this function is a cautionary tale: this
    strategy wants MORE exposure on the volatile, high-conviction signals,
    not less.
    """
    if base <= 0:
        return 1
    if risk_pct <= 0:
        return max(1, base)

    LIQUIDATION_BUFFER = 0.75
    HARD_CAP = 25

    if conviction >= 1.5:
        mult = 1.6
    elif conviction >= 1.0:
        mult = 1.3
    elif conviction >= 0.5:
        mult = 1.0
    else:
        mult = 0.8

    target = round(base * mult)

    safety_cap = int(LIQUIDATION_BUFFER / risk_pct)
    target = min(target, safety_cap)

    return max(1, min(target, HARD_CAP))


def _compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _level(signal: BreakoutSignal) -> str:
    if signal.status == "SHAKEOUT":
        return f"S:{_price(signal.support)}"
    return f"R:{_price(signal.resistance)}"


def _why(signal: BreakoutSignal) -> str:
    if signal.status == "PRE_BREAKOUT":
        return f"{signal.distance_to_trigger_pct * 100:.2f}% below long trigger"
    if signal.status == "SPRING":
        return f"swept support {signal.sweep_pct * 100:.2f}%, reclaimed"
    if signal.status == "PRE_BREAKDOWN":
        return f"{signal.distance_to_trigger_pct * 100:.2f}% above short trigger"
    if signal.status == "UPTHRUST":
        return f"swept resistance {signal.sweep_pct * 100:.2f}%, rejected"
    if signal.status == "BREAKOUT":
        return f"closed over R, volume {signal.volume_ratio:.1f}x"
    if signal.status == "BREAKDOWN":
        return f"closed under S, volume {signal.volume_ratio:.1f}x"
    return f"volume {signal.volume_ratio:.1f}x"


def _grade(signal: BreakoutSignal) -> str:
    if signal.reward_risk >= 1.5 and signal.score >= 75:
        return "A"
    if signal.reward_risk >= 1.2 and signal.score >= 65:
        return "B"
    if signal.reward_risk >= 1.0:
        return "C"
    return "skip"


def _filter_summary(args: argparse.Namespace) -> dict[str, float | int | bool]:
    return {
        "include_dead": args.include_dead,
        "include_confirmed": args.include_confirmed,
        "include_rejected": args.include_rejected,
        "min_quote_volume": 0 if args.include_dead else args.min_quote_volume,
        "min_trades": 0 if args.include_dead else args.min_trades,
        "min_24h_range_pct": 0 if args.include_dead else args.min_24h_range_pct,
        "skip_book_filter": args.skip_book_filter,
        "min_book_depth": 0 if args.include_dead or args.skip_book_filter else args.min_book_depth,
        "book_depth_pct": args.book_depth_pct,
        "max_spread_bps": 0 if args.include_dead or args.skip_book_filter else args.max_spread_bps,
        "book_limit": args.book_limit,
        "skip_oi_filter": args.skip_oi_filter,
        "min_open_interest_notional": 0 if args.include_dead or args.skip_oi_filter else args.min_open_interest_notional,
        "max_trigger_distance_pct": args.max_trigger_distance_pct,
        "max_shakeout_distance_pct": args.max_shakeout_distance_pct,
        "max_pre_trigger_move_pct": args.max_pre_trigger_move_pct,
        "entry_buffer_pct": args.entry_buffer_pct,
        "entry_mode": args.entry_mode,
        "entry_pullback_pct": args.entry_pullback_pct,
        "entry_atr_buffer_multiple": args.entry_atr_buffer_multiple,
        "trigger_reject_lookback": args.trigger_reject_lookback,
        "stop_buffer_pct": args.stop_buffer_pct,
        "target_range_multiple": args.target_range_multiple,
        "min_rr": args.min_rr,
        "min_score": args.min_score,
        "min_candle_quote_volume": args.min_candle_quote_volume,
        "top": args.top,
    }
