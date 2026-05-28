from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable


EPSILON = 1e-12
BASE_FLOW_INTERVAL_MS = 15 * 60_000


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float


@dataclass(frozen=True)
class BreakoutSettings:
    resistance_lookback: int = 40
    squeeze_lookback: int = 20
    volume_lookback: int = 20
    atr_lookback: int = 14
    min_breakout_pct: float = 0.0015
    target_breakout_pct: float = 0.012
    max_extension_pct: float = 0.035
    prior_break_tolerance_pct: float = 0.004
    watch_distance_pct: float = 0.02
    max_shakeout_distance_pct: float = 0.04
    max_pre_trigger_move_pct: float = 0.03
    entry_buffer_pct: float = 0.001
    stop_buffer_pct: float = 0.002
    target_range_multiple: float = 1.5
    min_sweep_pct: float = 0.0015
    max_fakeout_close_position: float = 0.55
    min_volume_ratio: float = 1.45
    min_watch_volume_ratio: float = 0.8
    min_avg_quote_volume: float = 25_000
    min_close_position: float = 0.62
    max_compression_pct: float = 0.18
    entry_atr_buffer_multiple: float = 0.0
    trigger_reject_lookback: int = 4
    breakout_lookback: int = 24
    breakout_stop_lookback: int = 10
    # Explosiveness floor for the simple detector. A 12-run sweep (whole universe,
    # 3 windows spanning a -21.9% downtrend, a -2.9% flat tape and a +13.7%
    # uptrend) found 2.5x volume with the ATR floor left at 1.0% the most robust:
    # positive expectancy in all three regimes (+0.098 / +0.110 / +0.626 R).
    # Raising the ATR floor to 1.5% lost money in the flat window - ATR is a
    # trailing average, so a high floor screens out quiet-coil breakouts.
    min_breakout_volume_ratio: float = 2.5
    min_breakout_atr_pct: float = 0.010
    overhead_ma_guard: bool = True
    # Detection upgrades validated by a 27-run sweep (whole universe, 3 regime
    # windows). A 1.5x-ATR breakout candle + an extension cap (reject breaks
    # already >5% past the level) together lifted expectancy +0.278 -> +0.377 R
    # and the worst-case window +0.098 -> +0.264 R. Relative-strength-vs-BTC and
    # the tight-base filter each lost money in a regime and were dropped.
    breakout_min_candle_range_mult: float = 1.5   # breakout candle range must be >= this x ATR
    breakout_max_extension_pct: float = 0.05      # reject breakouts already this far above the level
    breakout_max_base_range_pct: float = 0.0      # off - the tight-base filter killed the uptrend edge
    # MA-aware take-profit: a 6-run sweep (whole universe, 3 regime windows)
    # showed capping the target just below an overhead MA99 beat the uncapped
    # target on every metric in every window - expectancy, win rate, profit
    # factor and drawdown all improved; compounded 49x -> 61x.
    tp_cap_below_ma: bool = True
    # Case-aware target capping: the existing target = trigger + risk x 1.5 x intensity
    # produces 6R targets on high-ATR coins (intensity caps at 4). These two knobs
    # bring the target back to something hittable when the move is already volatile
    # or near a recent swing high.
    target_intensity_max: float = 4.0          # ceiling on the ATR intensity multiplier (tested: 2.0 hurt the right tail)
    # Cap the target just past the prior swing high/low of the last N candles
    # (excluding the breakout candle itself) when that resistance is meaningfully
    # above the trigger. Validated by a 12-run sweep: at 200 candles, expectancy
    # +0.432 -> +0.518 R, max drawdown 37.8% -> 32.3%, compounded 117x -> 203x.
    tp_cap_recent_swing_high_candles: int = 200
    # Volume-trend filter: require the base (pre-breakout window) to show net
    # accumulation - up-candle volume >= ratio x down-candle volume. 0 = off.
    breakout_min_up_down_volume_ratio: float = 0.0


@dataclass(frozen=True)
class BreakoutSignal:
    symbol: str
    interval: str
    side: str
    status: str
    score: float
    close: float
    resistance: float
    support: float
    breakout_pct: float
    move_pct: float
    sweep_pct: float
    distance_to_trigger_pct: float
    condition: str
    order_type: str
    trigger_price: float
    stop_price: float
    target_price: float
    risk_pct: float
    reward_pct: float
    reward_risk: float
    volume_ratio: float
    avg_quote_volume: float
    min_required_quote_volume: float
    compression_pct: float
    atr_pct: float
    trend_score: float
    close_position: float
    quote_volume_24h: float
    trade_count_24h: int
    range_pct_24h: float
    price_change_pct_24h: float
    book_min_depth: float
    open_interest_notional: float
    open_candle: bool


def candle_from_kline(kline: list[Any]) -> Candle:
    return Candle(
        open_time=int(kline[0]),
        open=float(kline[1]),
        high=float(kline[2]),
        low=float(kline[3]),
        close=float(kline[4]),
        volume=float(kline[5]),
        close_time=int(kline[6]),
        quote_volume=float(kline[7]),
    )


def candles_from_klines(klines: Iterable[list[Any]]) -> list[Candle]:
    return [candle_from_kline(kline) for kline in klines]


def evaluate_breakout(
    symbol: str,
    candles: list[Candle],
    quote_volume_24h: float,
    interval_ms: int,
    interval: str = "",
    trade_count_24h: int = 0,
    range_pct_24h: float = 0.0,
    price_change_pct_24h: float = 0.0,
    book_min_depth: float = 0.0,
    open_interest_notional: float = 0.0,
    settings: BreakoutSettings | None = None,
    include_confirmed: bool = False,
    include_rejected: bool = False,
    now_ms: int | None = None,
) -> BreakoutSignal | None:
    settings = settings or BreakoutSettings()
    min_needed = max(settings.resistance_lookback, settings.squeeze_lookback, settings.volume_lookback, 50) + 2
    if len(candles) < min_needed:
        return None

    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    latest = candles[-1]
    previous = candles[-2]
    resistance_window = candles[-settings.resistance_lookback - 1 : -1]
    squeeze_window = candles[-settings.squeeze_lookback - 1 : -1]
    volume_window = candles[-settings.volume_lookback - 1 : -1]

    resistance = max(candle.high for candle in resistance_window)
    support = min(candle.low for candle in squeeze_window)
    if resistance <= 0 or support <= 0 or support >= resistance:
        return None

    atr_pct = _atr_pct(candles[-settings.atr_lookback - 1 :], latest.close)
    trigger_buffer_pct = max(settings.entry_buffer_pct, atr_pct * settings.entry_atr_buffer_multiple)
    # Volatility-scaled gates: a coin with a wide 24h range gets a proportionally wider
    # trigger window. Fixed low-volatility gates structurally exclude genuinely hot coins
    # (post-pump consolidations sit too far from their high to ever register otherwise).
    vol_scale = _clamp(range_pct_24h / 7.0, 1.0, 7.0)
    effective_watch_distance = settings.watch_distance_pct * vol_scale
    effective_pre_trigger_move = settings.max_pre_trigger_move_pct * vol_scale
    effective_shakeout_distance = settings.max_shakeout_distance_pct * vol_scale
    effective_max_extension = settings.max_extension_pct * vol_scale
    long_trigger = resistance * (1 + trigger_buffer_pct)
    short_trigger = support * (1 - trigger_buffer_pct)
    breakout_pct = latest.close / resistance - 1.0
    breakdown_pct = support / max(latest.close, EPSILON) - 1.0
    prior_break_pct = previous.close / resistance - 1.0
    prior_breakdown_pct = support / max(previous.close, EPSILON) - 1.0
    reclaim_pct = latest.close / support - 1.0
    prior_support_pct = previous.close / support - 1.0
    support_sweep_pct = support / max(latest.low, EPSILON) - 1.0
    resistance_sweep_pct = latest.high / resistance - 1.0
    long_distance_to_trigger_pct = long_trigger / max(latest.close, EPSILON) - 1.0
    short_distance_to_trigger_pct = latest.close / max(short_trigger, EPSILON) - 1.0
    long_move_from_base_pct = latest.close / support - 1.0
    short_move_from_base_pct = resistance / max(latest.close, EPSILON) - 1.0
    compression_pct = _range_pct(squeeze_window, latest.close)
    close_position = _close_position(latest)
    bearish_close_position = 1 - close_position
    volume_ratio, open_candle = _volume_ratio(latest, volume_window, interval_ms, now_ms)
    avg_quote_volume = _average_quote_volume(volume_window)
    min_required_quote_volume = settings.min_avg_quote_volume * interval_ms / BASE_FLOW_INTERVAL_MS
    has_enough_flow = avg_quote_volume >= min_required_quote_volume
    long_trend_score = _trend_score(candles, side="LONG")
    short_trend_score = _trend_score(candles, side="SHORT")

    long_is_fresh = prior_break_pct <= settings.prior_break_tolerance_pct
    short_is_fresh = prior_breakdown_pct <= settings.prior_break_tolerance_pct
    long_has_not_triggered = latest.high < long_trigger and latest.close < resistance
    short_has_not_triggered = latest.low > short_trigger and latest.close > support
    long_trigger_was_rejected = _long_trigger_was_rejected(candles, settings, trigger_buffer_pct)
    short_trigger_was_rejected = _short_trigger_was_rejected(candles, settings, trigger_buffer_pct)

    candidates: list[BreakoutSignal] = []

    if (
        long_has_not_triggered
        and has_enough_flow
        and long_distance_to_trigger_pct >= 0
        and long_distance_to_trigger_pct <= effective_watch_distance
        and long_move_from_base_pct <= effective_pre_trigger_move
        and long_is_fresh
        and not long_trigger_was_rejected
        and volume_ratio >= settings.min_watch_volume_ratio
        and close_position >= settings.min_close_position
        and compression_pct <= settings.max_compression_pct * 1.25
    ):
        candidates.append(
            _build_signal(
                symbol=symbol,
                interval=interval,
                side="LONG",
                status="PRE_BREAKOUT",
                close=latest.close,
                resistance=resistance,
                support=support,
                breakout_pct=breakout_pct,
                move_pct=breakout_pct,
                sweep_pct=0.0,
                distance_to_trigger_pct=long_distance_to_trigger_pct,
                trigger_price=long_trigger,
                stop_price=support * (1 - settings.stop_buffer_pct),
                target_price=long_trigger + (resistance - support) * settings.target_range_multiple,
                volume_ratio=volume_ratio,
                avg_quote_volume=avg_quote_volume,
                min_required_quote_volume=min_required_quote_volume,
                compression_pct=compression_pct,
                atr_pct=atr_pct,
                close_position=close_position,
                trend_score=long_trend_score,
                quote_volume_24h=quote_volume_24h,
                trade_count_24h=trade_count_24h,
                range_pct_24h=range_pct_24h,
                price_change_pct_24h=price_change_pct_24h,
                open_candle=open_candle,
                settings=settings,
                book_min_depth=book_min_depth,
                open_interest_notional=open_interest_notional,
            )
        )

    if (
        long_has_not_triggered
        and has_enough_flow
        and support_sweep_pct >= settings.min_sweep_pct
        and reclaim_pct >= 0
        and prior_support_pct >= -settings.prior_break_tolerance_pct
        and long_distance_to_trigger_pct <= effective_shakeout_distance
        and volume_ratio >= settings.min_volume_ratio
        and close_position >= settings.min_close_position
        and compression_pct <= settings.max_compression_pct
    ):
        candidates.append(
            _build_signal(
                symbol=symbol,
                interval=interval,
                side="LONG",
                status="SPRING",
                close=latest.close,
                resistance=resistance,
                support=support,
                breakout_pct=breakout_pct,
                move_pct=reclaim_pct,
                sweep_pct=support_sweep_pct,
                distance_to_trigger_pct=long_distance_to_trigger_pct,
                trigger_price=long_trigger,
                stop_price=latest.low * (1 - settings.stop_buffer_pct),
                target_price=long_trigger + (resistance - support) * settings.target_range_multiple,
                volume_ratio=volume_ratio,
                avg_quote_volume=avg_quote_volume,
                min_required_quote_volume=min_required_quote_volume,
                compression_pct=compression_pct,
                atr_pct=atr_pct,
                close_position=close_position,
                trend_score=long_trend_score,
                quote_volume_24h=quote_volume_24h,
                trade_count_24h=trade_count_24h,
                range_pct_24h=range_pct_24h,
                price_change_pct_24h=price_change_pct_24h,
                open_candle=open_candle,
                settings=settings,
                book_min_depth=book_min_depth,
                open_interest_notional=open_interest_notional,
            )
        )

    if (
        include_confirmed
        and has_enough_flow
        and breakout_pct >= settings.min_breakout_pct
        and breakout_pct <= effective_max_extension
        and long_is_fresh
        and volume_ratio >= settings.min_volume_ratio
        and close_position >= settings.min_close_position
        and compression_pct <= settings.max_compression_pct
    ):
        candidates.append(
            _build_signal(
                symbol=symbol,
                interval=interval,
                side="LONG",
                status="BREAKOUT",
                close=latest.close,
                resistance=resistance,
                support=support,
                breakout_pct=breakout_pct,
                move_pct=breakout_pct,
                sweep_pct=max(resistance_sweep_pct, 0.0),
                distance_to_trigger_pct=max(long_distance_to_trigger_pct, 0.0),
                trigger_price=long_trigger,
                stop_price=support * (1 - settings.stop_buffer_pct),
                target_price=long_trigger + (resistance - support) * settings.target_range_multiple,
                volume_ratio=volume_ratio,
                avg_quote_volume=avg_quote_volume,
                min_required_quote_volume=min_required_quote_volume,
                compression_pct=compression_pct,
                atr_pct=atr_pct,
                close_position=close_position,
                trend_score=long_trend_score,
                quote_volume_24h=quote_volume_24h,
                trade_count_24h=trade_count_24h,
                range_pct_24h=range_pct_24h,
                price_change_pct_24h=price_change_pct_24h,
                open_candle=open_candle,
                settings=settings,
                book_min_depth=book_min_depth,
                open_interest_notional=open_interest_notional,
            )
        )

    if (
        short_has_not_triggered
        and has_enough_flow
        and short_distance_to_trigger_pct >= 0
        and short_distance_to_trigger_pct <= effective_watch_distance
        and short_move_from_base_pct <= effective_pre_trigger_move
        and short_is_fresh
        and not short_trigger_was_rejected
        and volume_ratio >= settings.min_watch_volume_ratio
        and bearish_close_position >= settings.min_close_position
        and compression_pct <= settings.max_compression_pct * 1.25
    ):
        candidates.append(
            _build_signal(
                symbol=symbol,
                interval=interval,
                side="SHORT",
                status="PRE_BREAKDOWN",
                close=latest.close,
                resistance=resistance,
                support=support,
                breakout_pct=breakdown_pct,
                move_pct=breakdown_pct,
                sweep_pct=0.0,
                distance_to_trigger_pct=short_distance_to_trigger_pct,
                trigger_price=short_trigger,
                stop_price=resistance * (1 + settings.stop_buffer_pct),
                target_price=max(short_trigger - (resistance - support) * settings.target_range_multiple, EPSILON),
                volume_ratio=volume_ratio,
                avg_quote_volume=avg_quote_volume,
                min_required_quote_volume=min_required_quote_volume,
                compression_pct=compression_pct,
                atr_pct=atr_pct,
                close_position=bearish_close_position,
                trend_score=short_trend_score,
                quote_volume_24h=quote_volume_24h,
                trade_count_24h=trade_count_24h,
                range_pct_24h=range_pct_24h,
                price_change_pct_24h=price_change_pct_24h,
                open_candle=open_candle,
                settings=settings,
                book_min_depth=book_min_depth,
                open_interest_notional=open_interest_notional,
            )
        )

    if (
        short_has_not_triggered
        and has_enough_flow
        and resistance_sweep_pct >= settings.min_sweep_pct
        and latest.close <= resistance
        and short_distance_to_trigger_pct <= effective_shakeout_distance
        and volume_ratio >= settings.min_volume_ratio
        and bearish_close_position >= settings.min_close_position
        and compression_pct <= settings.max_compression_pct
    ):
        candidates.append(
            _build_signal(
                symbol=symbol,
                interval=interval,
                side="SHORT",
                status="UPTHRUST",
                close=latest.close,
                resistance=resistance,
                support=support,
                breakout_pct=breakdown_pct,
                move_pct=resistance / max(latest.close, EPSILON) - 1.0,
                sweep_pct=resistance_sweep_pct,
                distance_to_trigger_pct=short_distance_to_trigger_pct,
                trigger_price=short_trigger,
                stop_price=latest.high * (1 + settings.stop_buffer_pct),
                target_price=max(short_trigger - (resistance - support) * settings.target_range_multiple, EPSILON),
                volume_ratio=volume_ratio,
                avg_quote_volume=avg_quote_volume,
                min_required_quote_volume=min_required_quote_volume,
                compression_pct=compression_pct,
                atr_pct=atr_pct,
                close_position=bearish_close_position,
                trend_score=short_trend_score,
                quote_volume_24h=quote_volume_24h,
                trade_count_24h=trade_count_24h,
                range_pct_24h=range_pct_24h,
                price_change_pct_24h=price_change_pct_24h,
                open_candle=open_candle,
                settings=settings,
                book_min_depth=book_min_depth,
                open_interest_notional=open_interest_notional,
            )
        )

    if (
        include_confirmed
        and has_enough_flow
        and breakdown_pct >= settings.min_breakout_pct
        and breakdown_pct <= effective_max_extension
        and short_is_fresh
        and volume_ratio >= settings.min_volume_ratio
        and bearish_close_position >= settings.min_close_position
        and compression_pct <= settings.max_compression_pct
    ):
        candidates.append(
            _build_signal(
                symbol=symbol,
                interval=interval,
                side="SHORT",
                status="BREAKDOWN",
                close=latest.close,
                resistance=resistance,
                support=support,
                breakout_pct=breakdown_pct,
                move_pct=breakdown_pct,
                sweep_pct=max(support_sweep_pct, 0.0),
                distance_to_trigger_pct=max(short_distance_to_trigger_pct, 0.0),
                trigger_price=short_trigger,
                stop_price=resistance * (1 + settings.stop_buffer_pct),
                target_price=max(short_trigger - (resistance - support) * settings.target_range_multiple, EPSILON),
                volume_ratio=volume_ratio,
                avg_quote_volume=avg_quote_volume,
                min_required_quote_volume=min_required_quote_volume,
                compression_pct=compression_pct,
                atr_pct=atr_pct,
                close_position=bearish_close_position,
                trend_score=short_trend_score,
                quote_volume_24h=quote_volume_24h,
                trade_count_24h=trade_count_24h,
                range_pct_24h=range_pct_24h,
                price_change_pct_24h=price_change_pct_24h,
                open_candle=open_candle,
                settings=settings,
                book_min_depth=book_min_depth,
                open_interest_notional=open_interest_notional,
            )
        )

    if include_rejected and not candidates and resistance_sweep_pct >= settings.min_sweep_pct and latest.close < resistance:
        return None

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: (_candidate_priority(candidate.status), candidate.score))


# Overhead-resistance guard: a breakout that fires straight into a long moving
# average (price still below a near MA99) tends to get rejected there. Skip it
# unless there is real headroom above, or the MA has already been cleared.
HEADROOM_MA_PERIOD = 99
HEADROOM_MIN_PCT = 0.025
HEADROOM_ATR_MULT = 0.6
# When tp_cap_below_ma is on, a target that projects past an overhead MA99 is
# pulled back to just inside that wall (bank the move where it is likely to stall).
TP_MA_CAP_BUFFER = 0.005
# Swing-high cap only fires when there is meaningfully higher resistance overhead
# (or lower support for SHORT); below this threshold there is no real ceiling to
# bound against and capping would just amputate the target.
TP_SWING_CAP_MIN_HEADROOM = 0.01


def _long_ma(candles: list[Candle]) -> float:
    """Simple moving average of the last HEADROOM_MA_PERIOD closes, or 0 if short."""
    if len(candles) < HEADROOM_MA_PERIOD:
        return 0.0
    return sum(candle.close for candle in candles[-HEADROOM_MA_PERIOD:]) / HEADROOM_MA_PERIOD


def _overhead_ma_blocks(candles: list[Candle], close: float, atr_pct: float, side: str) -> bool:
    """True when a long moving average sits as a near wall against the breakout."""
    if len(candles) < HEADROOM_MA_PERIOD or close <= 0:
        return False
    ma_long = sum(candle.close for candle in candles[-HEADROOM_MA_PERIOD:]) / HEADROOM_MA_PERIOD
    required = max(HEADROOM_MIN_PCT, atr_pct * HEADROOM_ATR_MULT)
    if side == "LONG":
        return ma_long > close and (ma_long - close) / close < required
    return ma_long < close and (close - ma_long) / close < required


def detect_long_breakout(
    symbol: str,
    candles: list[Candle],
    quote_volume_24h: float,
    interval_ms: int,
    interval: str = "",
    range_pct_24h: float = 0.0,
    settings: BreakoutSettings | None = None,
    now_ms: int | None = None,
) -> BreakoutSignal | None:
    """High-recall long-breakout detector.

    Fires whenever a coin closes above its recent range high on above-average
    volume - built to catch almost every real breakout, not just tidy coils.
    """
    settings = settings or BreakoutSettings()
    need = settings.breakout_lookback + max(settings.atr_lookback, settings.volume_lookback) + 2
    if len(candles) < need:
        return None
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    latest = candles[-1]
    prior_window = candles[-settings.breakout_lookback - 1 : -1]
    prior_high = max(candle.high for candle in prior_window)
    if prior_high <= 0 or latest.close <= prior_high:
        return None  # no new high -> no breakout

    if (
        settings.breakout_max_extension_pct > 0
        and latest.close / prior_high - 1.0 > settings.breakout_max_extension_pct
    ):
        return None  # already too far above the level - entering here is chasing
    if settings.breakout_max_base_range_pct > 0:
        prior_low = min(candle.low for candle in prior_window)
        if prior_low <= 0 or (prior_high - prior_low) / prior_high > settings.breakout_max_base_range_pct:
            return None  # base too wide/sloppy - not a tight coil

    atr_pct = _atr_pct(candles[-settings.atr_lookback - 1 :], latest.close)
    if atr_pct < settings.min_breakout_atr_pct:
        return None  # too sluggish to be a "big" breakout
    if settings.breakout_min_candle_range_mult > 0:
        atr_abs = atr_pct * latest.close
        if atr_abs <= 0 or (latest.high - latest.low) < settings.breakout_min_candle_range_mult * atr_abs:
            return None  # breakout candle is not a wide-range ignition bar
    if settings.breakout_min_up_down_volume_ratio > 0:
        up_vol = sum(_volume(c) for c in prior_window if c.close > c.open)
        down_vol = sum(_volume(c) for c in prior_window if c.close < c.open)
        if up_vol < settings.breakout_min_up_down_volume_ratio * down_vol:
            return None  # base lacks net accumulation - up-day volume < required ratio
    volume_window = candles[-settings.volume_lookback - 1 : -1]
    volume_ratio, open_candle = _volume_ratio(latest, volume_window, interval_ms, now_ms)
    if volume_ratio < settings.min_breakout_volume_ratio:
        return None  # breakout not backed by volume
    avg_quote_volume = _average_quote_volume(volume_window)
    min_required_quote_volume = settings.min_avg_quote_volume * interval_ms / BASE_FLOW_INTERVAL_MS
    if avg_quote_volume < min_required_quote_volume:
        return None  # too illiquid to trade
    close_position = _close_position(latest)
    if close_position < 0.5:
        return None  # breakout candle closed weak / rejected
    if settings.overhead_ma_guard and _overhead_ma_blocks(candles, latest.close, atr_pct, "LONG"):
        return None  # breakout fired straight into an overhead moving average

    stop_window = candles[-settings.breakout_stop_lookback :]
    swing_low = min(candle.low for candle in stop_window)
    stop_price = swing_low * (1 - settings.stop_buffer_pct)
    trigger_price = prior_high * (1 + settings.entry_buffer_pct)
    if stop_price <= 0 or stop_price >= trigger_price:
        return None
    intensity = min(max(atr_pct / 0.02, 1.0), max(settings.target_intensity_max, 1.0))
    target_price = trigger_price + (trigger_price - stop_price) * settings.target_range_multiple * intensity
    if settings.tp_cap_below_ma:
        ma_long = _long_ma(candles)
        if trigger_price < ma_long < target_price:
            capped = ma_long * (1 - TP_MA_CAP_BUFFER)
            if capped > trigger_price:
                target_price = capped  # bank the move just under the overhead MA99
    if settings.tp_cap_recent_swing_high_candles > 0:
        # Exclude the breakout candle itself - it just MADE a new high, so the
        # meaningful overhead resistance is whatever is left in the lookback.
        # Only cap when that prior swing is materially above the trigger; else
        # there is no real ceiling and capping just amputates the target.
        n = min(settings.tp_cap_recent_swing_high_candles, len(candles) - 1)
        if n > 0:
            prior_n = candles[-n - 1 : -1]
            if prior_n:
                swing_high = max(c.high for c in prior_n)
                if swing_high > trigger_price * (1 + TP_SWING_CAP_MIN_HEADROOM):
                    capped = swing_high * (1 + TP_MA_CAP_BUFFER)
                    if trigger_price < capped < target_price:
                        target_price = capped  # bank just past the higher resistance

    return _build_signal(
        symbol=symbol,
        interval=interval,
        side="LONG",
        status="BREAKOUT",
        close=latest.close,
        resistance=prior_high,
        support=swing_low,
        breakout_pct=latest.close / prior_high - 1.0,
        move_pct=latest.close / prior_high - 1.0,
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        trigger_price=trigger_price,
        stop_price=stop_price,
        target_price=target_price,
        volume_ratio=volume_ratio,
        avg_quote_volume=avg_quote_volume,
        min_required_quote_volume=min_required_quote_volume,
        compression_pct=0.0,
        atr_pct=atr_pct,
        close_position=close_position,
        trend_score=_trend_score(candles, side="LONG"),
        quote_volume_24h=quote_volume_24h,
        trade_count_24h=0,
        range_pct_24h=range_pct_24h,
        price_change_pct_24h=0.0,
        open_candle=open_candle,
        settings=settings,
    )


def detect_short_breakdown(
    symbol: str,
    candles: list[Candle],
    quote_volume_24h: float,
    interval_ms: int,
    interval: str = "",
    range_pct_24h: float = 0.0,
    settings: BreakoutSettings | None = None,
    now_ms: int | None = None,
) -> BreakoutSignal | None:
    """High-recall confirmed breakdown detector, mirroring ``detect_long_breakout``."""
    settings = settings or BreakoutSettings()
    need = settings.breakout_lookback + max(settings.atr_lookback, settings.volume_lookback) + 2
    if len(candles) < need:
        return None
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    latest = candles[-1]
    prior_window = candles[-settings.breakout_lookback - 1 : -1]
    prior_low = min(candle.low for candle in prior_window)
    if prior_low <= 0 or latest.close >= prior_low:
        return None

    if (
        settings.breakout_max_extension_pct > 0
        and prior_low / max(latest.close, EPSILON) - 1.0 > settings.breakout_max_extension_pct
    ):
        return None  # already too far below the level - entering here is chasing
    if settings.breakout_max_base_range_pct > 0:
        prior_high = max(candle.high for candle in prior_window)
        if prior_high <= 0 or (prior_high - prior_low) / prior_low > settings.breakout_max_base_range_pct:
            return None  # base too wide/sloppy - not a tight coil

    atr_pct = _atr_pct(candles[-settings.atr_lookback - 1 :], latest.close)
    if atr_pct < settings.min_breakout_atr_pct:
        return None
    if settings.breakout_min_candle_range_mult > 0:
        atr_abs = atr_pct * latest.close
        if atr_abs <= 0 or (latest.high - latest.low) < settings.breakout_min_candle_range_mult * atr_abs:
            return None  # breakdown candle is not a wide-range ignition bar
    volume_window = candles[-settings.volume_lookback - 1 : -1]
    volume_ratio, open_candle = _volume_ratio(latest, volume_window, interval_ms, now_ms)
    if volume_ratio < settings.min_breakout_volume_ratio:
        return None
    avg_quote_volume = _average_quote_volume(volume_window)
    min_required_quote_volume = settings.min_avg_quote_volume * interval_ms / BASE_FLOW_INTERVAL_MS
    if avg_quote_volume < min_required_quote_volume:
        return None
    close_position = _close_position(latest)
    if close_position > 0.5:
        return None
    if settings.overhead_ma_guard and _overhead_ma_blocks(candles, latest.close, atr_pct, "SHORT"):
        return None  # breakdown fired straight into a moving-average support
    if settings.breakout_min_up_down_volume_ratio > 0:
        up_vol = sum(_volume(c) for c in prior_window if c.close > c.open)
        down_vol = sum(_volume(c) for c in prior_window if c.close < c.open)
        if down_vol < settings.breakout_min_up_down_volume_ratio * up_vol:
            return None  # base lacks net distribution - down-day volume below required ratio

    stop_window = candles[-settings.breakout_stop_lookback :]
    swing_high = max(candle.high for candle in stop_window)
    stop_price = swing_high * (1 + settings.stop_buffer_pct)
    trigger_price = prior_low * (1 - settings.entry_buffer_pct)
    if trigger_price <= 0 or stop_price <= trigger_price:
        return None
    intensity = min(max(atr_pct / 0.02, 1.0), max(settings.target_intensity_max, 1.0))
    target_price = max(
        trigger_price - (stop_price - trigger_price) * settings.target_range_multiple * intensity,
        EPSILON,
    )
    if settings.tp_cap_below_ma:
        ma_long = _long_ma(candles)
        if target_price < ma_long < trigger_price:
            capped = ma_long * (1 + TP_MA_CAP_BUFFER)
            if capped < trigger_price:
                target_price = capped  # bank the move just above the overhead MA99
    if settings.tp_cap_recent_swing_high_candles > 0:
        n = min(settings.tp_cap_recent_swing_high_candles, len(candles) - 1)
        if n > 0:
            prior_n = candles[-n - 1 : -1]
            if prior_n:
                swing_low = min(c.low for c in prior_n)
                if swing_low < trigger_price * (1 - TP_SWING_CAP_MIN_HEADROOM):
                    capped = swing_low * (1 - TP_MA_CAP_BUFFER)
                    if target_price < capped < trigger_price:
                        target_price = capped  # bank just past the lower support

    return _build_signal(
        symbol=symbol,
        interval=interval,
        side="SHORT",
        status="BREAKDOWN",
        close=latest.close,
        resistance=swing_high,
        support=prior_low,
        breakout_pct=prior_low / max(latest.close, EPSILON) - 1.0,
        move_pct=prior_low / max(latest.close, EPSILON) - 1.0,
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        trigger_price=trigger_price,
        stop_price=stop_price,
        target_price=target_price,
        volume_ratio=volume_ratio,
        avg_quote_volume=avg_quote_volume,
        min_required_quote_volume=min_required_quote_volume,
        compression_pct=0.0,
        atr_pct=atr_pct,
        close_position=1.0 - close_position,
        trend_score=_trend_score(candles, side="SHORT"),
        quote_volume_24h=quote_volume_24h,
        trade_count_24h=0,
        range_pct_24h=range_pct_24h,
        price_change_pct_24h=0.0,
        open_candle=open_candle,
        settings=settings,
    )


def interval_to_ms(interval: str) -> int:
    units = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    if interval == "1M":
        return 30 * 86_400_000
    unit = interval[-1]
    if unit not in units:
        raise ValueError(f"unsupported interval: {interval}")
    amount = int(interval[:-1])
    return amount * units[unit]


def _build_signal(
    symbol: str,
    interval: str,
    side: str,
    status: str,
    close: float,
    resistance: float,
    support: float,
    breakout_pct: float,
    move_pct: float,
    sweep_pct: float,
    distance_to_trigger_pct: float,
    trigger_price: float,
    stop_price: float,
    target_price: float,
    volume_ratio: float,
    avg_quote_volume: float,
    min_required_quote_volume: float,
    compression_pct: float,
    atr_pct: float,
    close_position: float,
    trend_score: float,
    quote_volume_24h: float,
    trade_count_24h: int,
    range_pct_24h: float,
    price_change_pct_24h: float,
    open_candle: bool,
    settings: BreakoutSettings,
    book_min_depth: float = 0.0,
    open_interest_notional: float = 0.0,
) -> BreakoutSignal:
    if side == "SHORT":
        risk_pct = max(stop_price / max(trigger_price, EPSILON) - 1.0, 0.0)
        reward_pct = max(trigger_price / max(target_price, EPSILON) - 1.0, 0.0)
        condition = f"mark <= {trigger_price:.12g}"
        order_type = "SELL STOP_MARKET"
    else:
        risk_pct = max(trigger_price / max(stop_price, EPSILON) - 1.0, 0.0)
        reward_pct = max(target_price / max(trigger_price, EPSILON) - 1.0, 0.0)
        condition = f"mark >= {trigger_price:.12g}"
        order_type = "BUY STOP_MARKET"

    reward_risk = reward_pct / risk_pct if risk_pct > 0 else 0.0
    score = _score(
        distance_to_trigger_pct=distance_to_trigger_pct,
        reward_risk=reward_risk,
        volume_ratio=volume_ratio,
        compression_pct=compression_pct,
        close_position=close_position,
        trend_score=trend_score,
        settings=settings,
    )

    return BreakoutSignal(
        symbol=symbol,
        interval=interval,
        side=side,
        status=status,
        score=score,
        close=close,
        resistance=resistance,
        support=support,
        breakout_pct=breakout_pct,
        move_pct=move_pct,
        sweep_pct=sweep_pct,
        distance_to_trigger_pct=distance_to_trigger_pct,
        condition=condition,
        order_type=order_type,
        trigger_price=trigger_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_pct=risk_pct,
        reward_pct=reward_pct,
        reward_risk=reward_risk,
        volume_ratio=volume_ratio,
        avg_quote_volume=avg_quote_volume,
        min_required_quote_volume=min_required_quote_volume,
        compression_pct=compression_pct,
        atr_pct=atr_pct,
        trend_score=trend_score,
        close_position=close_position,
        quote_volume_24h=quote_volume_24h,
        trade_count_24h=trade_count_24h,
        range_pct_24h=range_pct_24h,
        price_change_pct_24h=price_change_pct_24h,
        book_min_depth=book_min_depth,
        open_interest_notional=open_interest_notional,
        open_candle=open_candle,
    )


def _volume_ratio(
    latest: Candle,
    volume_window: list[Candle],
    interval_ms: int,
    now_ms: int,
) -> tuple[float, bool]:
    average_volume = sum(_volume(candle) for candle in volume_window) / max(len(volume_window), 1)
    if average_volume <= 0:
        return 0.0, latest.close_time > now_ms

    latest_volume = _volume(latest)
    open_candle = latest.close_time > now_ms
    if open_candle:
        elapsed = max(now_ms - latest.open_time, 1)
        elapsed_fraction = min(max(elapsed / max(interval_ms, 1), 0.08), 1.0)
        latest_volume = latest_volume / elapsed_fraction

    return latest_volume / average_volume, open_candle


def _long_trigger_was_rejected(candles: list[Candle], settings: BreakoutSettings, trigger_buffer_pct: float) -> bool:
    if settings.trigger_reject_lookback <= 0:
        return False
    end = len(candles) - 1
    start = max(settings.resistance_lookback, end - settings.trigger_reject_lookback)
    for index in range(start, end):
        prior_window = candles[index - settings.resistance_lookback : index]
        if not prior_window:
            continue
        resistance = max(candle.high for candle in prior_window)
        trigger = resistance * (1 + trigger_buffer_pct)
        candle = candles[index]
        if candle.high >= trigger and candle.close < resistance:
            return True
    return False


def _short_trigger_was_rejected(candles: list[Candle], settings: BreakoutSettings, trigger_buffer_pct: float) -> bool:
    if settings.trigger_reject_lookback <= 0:
        return False
    end = len(candles) - 1
    start = max(settings.squeeze_lookback, end - settings.trigger_reject_lookback)
    for index in range(start, end):
        prior_window = candles[index - settings.squeeze_lookback : index]
        if not prior_window:
            continue
        support = min(candle.low for candle in prior_window)
        trigger = support * (1 - trigger_buffer_pct)
        candle = candles[index]
        if candle.low <= trigger and candle.close > support:
            return True
    return False


def _volume(candle: Candle) -> float:
    return candle.quote_volume if candle.quote_volume > 0 else candle.volume


def _average_quote_volume(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    return sum(_volume(candle) for candle in candles) / len(candles)


def _range_pct(candles: list[Candle], price: float) -> float:
    high = max(candle.high for candle in candles)
    low = min(candle.low for candle in candles)
    return (high - low) / max(price, EPSILON)


def _close_position(candle: Candle) -> float:
    width = candle.high - candle.low
    if width <= 0:
        return 0.5
    return (candle.close - candle.low) / width


def _atr_pct(candles: list[Candle], price: float) -> float:
    if len(candles) < 2:
        return 0.0
    true_ranges: list[float] = []
    for previous, current in zip(candles, candles[1:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return (sum(true_ranges) / len(true_ranges)) / max(price, EPSILON)


def _trend_score(candles: list[Candle], side: str = "LONG") -> float:
    closes = [candle.close for candle in candles]
    ema20 = _ema(closes[-60:], 20)
    ema50 = _ema(closes[-80:], 50)
    if math.isnan(ema20) or math.isnan(ema50):
        return 0.5

    latest_close = closes[-1]
    score = 0.2
    if side == "SHORT":
        if latest_close < ema20:
            score += 0.35
        if ema20 <= ema50:
            score += 0.3
        if latest_close <= min(closes[-8:]):
            score += 0.15
    else:
        if latest_close > ema20:
            score += 0.35
        if ema20 >= ema50:
            score += 0.3
        if latest_close >= max(closes[-8:]):
            score += 0.15
    return min(score, 1.0)


def _ema(values: list[float], period: int) -> float:
    if len(values) < period:
        return float("nan")
    alpha = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for value in values[period:]:
        ema = value * alpha + ema * (1 - alpha)
    return ema


def _score(
    distance_to_trigger_pct: float,
    reward_risk: float,
    volume_ratio: float,
    compression_pct: float,
    close_position: float,
    trend_score: float,
    settings: BreakoutSettings,
) -> float:
    distance_score = 1.0 - distance_to_trigger_pct / max(settings.watch_distance_pct, EPSILON)
    distance_score = _clamp(distance_score, 0.0, 1.0)
    volume_score = _clamp(volume_ratio / 2.5, 0.0, 1.0)
    compression_score = 1.0 - compression_pct / max(settings.max_compression_pct, EPSILON)
    compression_score = _clamp(compression_score, 0.0, 1.0)
    reward_score = _clamp(reward_risk / 2.0, 0.0, 1.0)

    score = (
        distance_score * 0.25
        + volume_score * 0.2
        + compression_score * 0.18
        + close_position * 0.15
        + trend_score * 0.12
        + reward_score * 0.1
    )
    return round(score * 100, 1)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


# ---------------------------------------------------------------------------
# Bear-market oversold-bounce detector
# ---------------------------------------------------------------------------
# In sustained downtrends, breakout longs get faded.  The edge shifts to
# mean-reversion: coins that are deeply discounted, showing capitulation
# volume, and printing their first green candle after a red streak.
#
# Design choices (backed by the W1 crash/recovery pattern analysis):
#   * Require the coin to be near the bottom of its 72-candle range (close
#     position <= 0.25) -- it must genuinely be "on sale".
#   * Require at least 3 of the last 5 candles to be red (bearish streak)
#     so we are catching a reversal, not joining a dip that keeps dipping.
#   * The entry candle must close green (close > open) as confirmation the
#     bounce has begun.
#   * Volume on the entry candle must be >= 1.3x the 20-period average
#     (capitulation or accumulation volume -- either is fine for a bounce).
#   * Stop goes just below the recent swing low (tight, because if the
#     bounce fails we want out fast).
#   * Target is a quick 2R: entry + 2 * risk.  Bear-market rallies are
#     sharp but short-lived; trying to capture more than 2R usually gives
#     it all back.
#   * The detector is only activated when BTC itself is bearish (checked
#     by the caller in backtest.py), so it never fires in bull markets.

OVERSOLD_LOOKBACK = 72
OVERSOLD_MAX_CLOSE_POSITION = 0.25  # must be in bottom quartile of range
OVERSOLD_RED_STREAK_MIN = 3
OVERSOLD_RED_STREAK_WINDOW = 5
OVERSOLD_MIN_VOLUME_RATIO = 1.3
OVERSOLD_VOLUME_LOOKBACK = 20
OVERSOLD_TARGET_R_MULTIPLE = 2.0
OVERSOLD_STOP_LOOKBACK = 10
OVERSOLD_MIN_ATR_PCT = 0.008  # 0.8% -- must have some volatility to bounce


def detect_oversold_bounce(
    symbol: str,
    candles: list[Candle],
    quote_volume_24h: float,
    interval_ms: int,
    interval: str = "",
    range_pct_24h: float = 0.0,
    settings: BreakoutSettings | None = None,
    now_ms: int | None = None,
) -> BreakoutSignal | None:
    """Detect mean-reversion long setups in bear markets.

    Looks for coins that are deeply discounted (bottom quartile of their
    recent range), in a red streak, and printing their first green candle
    with elevated volume -- a capitulation-to-reversal signature.
    """
    settings = settings or BreakoutSettings()
    need = OVERSOLD_LOOKBACK + 2
    if len(candles) < need:
        return None
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    latest = candles[-1]

    # ── 1. Deep discount check ──────────────────────────────────────────
    window_72 = candles[-OVERSOLD_LOOKBACK - 1 :]
    high_72 = max(c.high for c in window_72)
    low_72 = min(c.low for c in window_72)
    if high_72 <= low_72:
        return None
    close_position_72 = (latest.close - low_72) / (high_72 - low_72)
    if close_position_72 > OVERSOLD_MAX_CLOSE_POSITION:
        return None  # not discounted enough

    # ── 2. Red streak check ─────────────────────────────────────────────
    recent_5 = candles[-OVERSOLD_RED_STREAK_WINDOW - 1 : -1]  # exclude latest
    red_count = sum(1 for c in recent_5 if c.close < c.open)
    if red_count < OVERSOLD_RED_STREAK_MIN:
        return None  # no bearish momentum to reverse

    # ── 3. Green entry candle ───────────────────────────────────────────
    if latest.close <= latest.open:
        return None  # entry candle must be green (reversal confirmation)

    # ── 4. Volume check ─────────────────────────────────────────────────
    vol_lookback = min(OVERSOLD_VOLUME_LOOKBACK, len(candles) - 2)
    vol_window = candles[-vol_lookback - 1 : -1]
    volume_ratio, open_candle = _volume_ratio(latest, vol_window, interval_ms, now_ms)
    if volume_ratio < OVERSOLD_MIN_VOLUME_RATIO:
        return None  # no capitulation/accumulation volume

    # ── 5. ATR / liquidity check ────────────────────────────────────────
    atr_pct = _atr_pct(candles[-settings.atr_lookback - 1 :], latest.close)
    if atr_pct < OVERSOLD_MIN_ATR_PCT:
        return None  # too dead to bounce

    avg_quote_volume = _average_quote_volume(vol_window)
    min_required_quote_volume = settings.min_avg_quote_volume * interval_ms / BASE_FLOW_INTERVAL_MS
    if avg_quote_volume < min_required_quote_volume:
        return None

    # ── 6. Stop: below the recent swing low ─────────────────────────────
    stop_lookback = min(OVERSOLD_STOP_LOOKBACK, len(candles) - 1)
    stop_window = candles[-stop_lookback - 1 : -1]
    swing_low = min(c.low for c in stop_window)
    stop_price = swing_low * (1.0 - settings.stop_buffer_pct)

    # ── 7. Entry & target ───────────────────────────────────────────────
    trigger_price = latest.close  # market entry at signal close
    if stop_price <= 0 or stop_price >= trigger_price:
        return None
    risk = trigger_price - stop_price
    target_price = trigger_price + risk * OVERSOLD_TARGET_R_MULTIPLE

    # ── 8. Build signal ─────────────────────────────────────────────────
    close_pos = _close_position(latest)
    return _build_signal(
        symbol=symbol,
        interval=interval,
        side="LONG",
        status="OVERSOLD_BOUNCE",
        close=latest.close,
        resistance=high_72,
        support=swing_low,
        breakout_pct=close_position_72,  # repurposed: how deep the discount
        move_pct=(latest.close - latest.open) / max(latest.open, EPSILON),
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        trigger_price=trigger_price,
        stop_price=stop_price,
        target_price=target_price,
        volume_ratio=volume_ratio,
        avg_quote_volume=avg_quote_volume,
        min_required_quote_volume=min_required_quote_volume,
        compression_pct=0.0,
        atr_pct=atr_pct,
        close_position=close_pos,
        trend_score=_trend_score(candles, side="LONG"),
        quote_volume_24h=quote_volume_24h,
        trade_count_24h=0,
        range_pct_24h=range_pct_24h,
        price_change_pct_24h=0.0,
        open_candle=open_candle,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Bear-market short filter helpers
# ---------------------------------------------------------------------------
# The standard short detector mirrors the long-breakout logic, which works
# poorly in bear markets because breakdowns get bought up (bear-trap rallies).
# Two extra filters improve short quality in bear regimes:
#
#   1. TREND CONFIRMATION – the coin must already be below its 72-period
#      EMA (downtrend in place).  Shorting a coin that is still above its
#      long-term average is fighting the primary trend even if the local
#      breakdown looks good.
#
#   2. FAILED-BOUNCE PATTERN – the best shorts come after a dead-cat bounce:
#      at least one of the last 3 candles closed green (attempted rally)
#      and the current candle is now breaking down through it.

SHORT_EMA_PERIOD = 72
SHORT_FAILED_BOUNCE_LOOKBACK = 3


def short_has_bear_trend(candles: list[Candle]) -> bool:
    """True when the coin is in a sustained downtrend (close below 72-EMA)."""
    if len(candles) < SHORT_EMA_PERIOD:
        return True  # not enough data; let it through
    closes = [c.close for c in candles[-SHORT_EMA_PERIOD:]]
    ema = _ema(closes, SHORT_EMA_PERIOD)
    if math.isnan(ema):
        return True
    return candles[-1].close < ema


def short_has_failed_bounce(candles: list[Candle]) -> bool:
    """True when a recent green candle (failed rally) precedes the breakdown."""
    lookback = min(SHORT_FAILED_BOUNCE_LOOKBACK, len(candles) - 2)
    prior = candles[-lookback - 1 : -1]
    return any(c.close > c.open for c in prior)


def _candidate_priority(status: str) -> int:
    return {
        "SPRING": 3,
        "UPTHRUST": 3,
        "PRE_BREAKOUT": 2,
        "PRE_BREAKDOWN": 2,
        "BREAKOUT": 1,
        "BREAKDOWN": 1,
    }.get(status, 0)


def _status(is_pre_breakout: bool, is_shakeout: bool, is_confirmed_breakout: bool, is_fakeout: bool) -> str:
    if is_shakeout:
        return "SHAKEOUT"
    if is_pre_breakout:
        return "PRE_BREAKOUT"
    if is_confirmed_breakout:
        return "BREAKOUT"
    if is_fakeout:
        return "FAKEOUT"
    return "NONE"


# ---------------------------------------------------------------------------
# Bear-bounce detector — high-volume mean-reversion signal generator
# ---------------------------------------------------------------------------
# Previous iterations were too strict (2-18 trades), making evaluation
# impossible.  This version generates MANY signals with loose entry
# criteria, relying on the exit strategy (breakeven, stagnation, trail)
# to separate winners from losers.
#
# The hypothesis: in bear markets, buying ANY green candle after a red
# streak, with a tight stop and quick target, has a small but real edge.
# Volume of trades × small edge = profit.

# Data-driven thresholds from 92-trade W1 analysis:
#   Winners: ATR 2.74%, vol 2.04x, rel mom -6.95%, BTC mom 2.66%
#   Losers:  ATR 2.20%, vol 2.07x, rel mom -3.36%, BTC mom 1.99%
#   BTC < EMA: 20% WR / -0.56R  →  BLOCK
#   rel mom < -5%: 44% WR / +0.39R  →  REQUIRE  (contrarian bounce)
#   vol > 2.0x: 24% WR  →  BLOCK (exhaustion)
#   hold=1c: 0% WR (24 trades!)  →  need close_position filter

BB_RED_STREAK_MIN = 2
BB_MIN_VOLUME_RATIO = 1.2
BB_MAX_VOLUME_RATIO = 2.5    # cap: extreme vol = exhaustion, not ignition
BB_TARGET_R = 2.0
BB_MIN_ATR_PCT = 0.015        # need 1.5%+ ATR (data: winners avg 2.74%)
BB_MIN_CLOSE_POS = 0.55       # close in top 55% (filter wicky fakeouts)
BB_MIN_BODY_ATR = 0.2         # body >= 0.2x ATR (real buying, not a wick)


def detect_bear_bounce(
    symbol: str,
    candles: list[Candle],
    quote_volume_24h: float,
    interval_ms: int,
    interval: str = "",
    range_pct_24h: float = 0.0,
    settings: BreakoutSettings | None = None,
    now_ms: int | None = None,
) -> BreakoutSignal | None:
    """High-volume bear-market bounce signal generator.

    Simple hypothesis: after a short red streak, a green candle with
    slightly above-average volume is a bounce candidate.  Enter at close
    with a tight stop below the candle low.  Let the exit strategy
    (breakeven, stagnation, trail) do the filtering.
    """
    settings = settings or BreakoutSettings()
    need = 20
    if len(candles) < need:
        return None
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    latest = candles[-1]

    # ── 1. Green candle after red streak ─────────────────────────────────
    if latest.close <= latest.open:
        return None
    reds = 0
    for c in candles[-BB_RED_STREAK_MIN - 1 : -1]:
        if c.close < c.open:
            reds += 1
    if reds < BB_RED_STREAK_MIN:
        return None

    # ── 2. Basic quality ─────────────────────────────────────────────────
    atr_pct = _atr_pct(candles[-settings.atr_lookback - 1 :], latest.close)
    if atr_pct < BB_MIN_ATR_PCT:
        return None

    # ── 2b. Close position: must close in top 55% (not a wicky fakeout)
    # 24/92 trades stopped out in 1c — those were wicky fakeouts.
    candle_range = latest.high - latest.low
    if candle_range <= 0:
        return None
    close_pos = (latest.close - latest.low) / candle_range
    if close_pos < BB_MIN_CLOSE_POS:
        return None
    # Body must be meaningful (>= 0.2x ATR)
    atr_abs = atr_pct * latest.close
    body = latest.close - latest.open
    if body < BB_MIN_BODY_ATR * atr_abs:
        return None

    # ── 3. Volume ────────────────────────────────────────────────────────
    vol_lookback = min(20, len(candles) - 2)
    vol_window = candles[-vol_lookback - 1 : -1]
    volume_ratio, open_candle = _volume_ratio(latest, vol_window, interval_ms, now_ms)
    if volume_ratio < BB_MIN_VOLUME_RATIO:
        return None
    if volume_ratio > BB_MAX_VOLUME_RATIO:
        return None  # extreme vol = exhaustion, not ignition (24% WR)

    avg_quote_volume = _average_quote_volume(vol_window)
    min_required_quote_volume = settings.min_avg_quote_volume * interval_ms / BASE_FLOW_INTERVAL_MS
    if avg_quote_volume < min_required_quote_volume:
        return None

    # ── 4. Stop & target ─────────────────────────────────────────────────
    stop_price = latest.low * (1.0 - settings.stop_buffer_pct)
    trigger_price = latest.close
    if stop_price <= 0 or stop_price >= trigger_price:
        return None
    risk = trigger_price - stop_price
    target_price = trigger_price + risk * BB_TARGET_R

    return _build_signal(
        symbol=symbol,
        interval=interval,
        side="LONG",
        status="BEAR_BOUNCE",
        close=latest.close,
        resistance=latest.high,
        support=latest.low,
        breakout_pct=(latest.close - latest.open) / max(latest.open, EPSILON),
        move_pct=(latest.close - latest.open) / max(latest.open, EPSILON),
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        trigger_price=trigger_price,
        stop_price=stop_price,
        target_price=target_price,
        volume_ratio=volume_ratio,
        avg_quote_volume=avg_quote_volume,
        min_required_quote_volume=min_required_quote_volume,
        compression_pct=0.0,
        atr_pct=atr_pct,
        close_position=_close_position(latest),
        trend_score=_trend_score(candles, side="LONG"),
        quote_volume_24h=quote_volume_24h,
        trade_count_24h=0,
        range_pct_24h=range_pct_24h,
        price_change_pct_24h=0.0,
        open_candle=open_candle,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Failed-rally detector — short bear-market bounces that get rejected
# ---------------------------------------------------------------------------
# Pattern (visible on the PIEVERSEUSDT chart):
#   1. Downtrend: price below MA(25) and MA(99) — bears in control
#   2. Rally attempt: 1-3 green candles pushed price up, breaking a minor
#      resistance (made a higher high vs the prior 5 candles)
#   3. Rejection: current candle is RED, closing near its low, with volume
#      confirming sellers took back control
#   4. The rejection close should be BELOW the rally's starting point
#      (the rally completely failed — no higher low was established)
#
# Entry: short at the rejection candle's close
# Stop:   above the rally high (point of maximum optimism)
# Target: 1.5R (failed rallies often retrace the entire bounce quickly)

FR_MA_PERIOD = 25              # must be below this MA (downtrend)
FR_RALLY_LOOKBACK = 3           # candles before rally to establish "prior range"
FR_RALLY_WINDOW = 3             # candles to scan for the rally attempt
FR_REJECTION_MIN_BODY_ATR = float(os.environ.get('HP_FR_BODY', '0.15'))
FR_REJECTION_MAX_CLOSE_POS = float(os.environ.get('HP_FR_CP', '0.45'))
FR_MIN_VOLUME_RATIO = float(os.environ.get('HP_FR_VOL', '1.1'))
FR_TARGET_R = 1.5
FR_MIN_ATR_PCT = 0.010


def detect_failed_rally(
    symbol: str,
    candles: list[Candle],
    quote_volume_24h: float,
    interval_ms: int,
    interval: str = "",
    range_pct_24h: float = 0.0,
    settings: BreakoutSettings | None = None,
    now_ms: int | None = None,
) -> BreakoutSignal | None:
    """Detect failed rallies in bear markets.

    Three-part pattern:
      1. DOWNTEND: price is below its MA(25) — bears in control
      2. RALLY: a recent green candle pushed above the prior range
         (made a higher high, attempting a recovery)
      3. REJECTION: current candle is RED, closing low, with volume
         The close is BELOW the rally's starting point — rally fully erased

    This is the classic bear-market pattern visible on the chart:
    price grinds down, pops up (trapping longs), then reverses hard.
    """
    settings = settings or BreakoutSettings()
    need = FR_MA_PERIOD + FR_RALLY_LOOKBACK + FR_RALLY_WINDOW + 3
    if len(candles) < need:
        return None
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    latest = candles[-1]          # rejection candle

    # ── 1. DOWNTEND: price below MA(25) ──────────────────────────────────
    ma_closes = [c.close for c in candles[-FR_MA_PERIOD - 1:]]
    ma25 = sum(ma_closes[-FR_MA_PERIOD:]) / FR_MA_PERIOD
    if latest.close >= ma25:
        return None  # not in a downtrend — don't short

    # ── 2. RALLY ATTEMPT: any candle in rally_window pushed above prior range ─
    rally_window = candles[-FR_RALLY_WINDOW - 1 : -1]  # exclude latest
    prior_range = candles[-FR_RALLY_LOOKBACK - FR_RALLY_WINDOW - 1 : -FR_RALLY_WINDOW - 1]
    if len(prior_range) < 3 or len(rally_window) < 1:
        return None
    prior_high = max(c.high for c in prior_range)
    prior_low = min(c.low for c in prior_range)

    # Find the rally peak: highest high in the rally window that exceeds prior_high
    rally_high = max(c.high for c in rally_window)
    if rally_high <= prior_high:
        return None  # no rally — price never broke above the prior range

    # ── 3. REJECTION: current candle is RED, making a lower low ──────────
    if latest.close >= latest.open:
        return None  # must be a red candle (rejection)
    # Must break below the rally window's low (rally fully erased)
    rally_window_low = min(c.low for c in rally_window)
    if latest.low >= rally_window_low:
        return None  # holding above rally zone — rejection incomplete
    candle_range = latest.high - latest.low
    if candle_range <= 0:
        return None
    close_pos = (latest.close - latest.low) / candle_range
    if close_pos > FR_REJECTION_MAX_CLOSE_POS:
        return None  # didn't close weak enough

    # Body must be meaningful (real selling, not a doji)
    atr_pct = _atr_pct(candles[-settings.atr_lookback - 1 :], latest.close)
    if atr_pct < FR_MIN_ATR_PCT:
        return None
    atr_abs = atr_pct * latest.close
    body = latest.open - latest.close  # red body
    if body < FR_REJECTION_MIN_BODY_ATR * atr_abs:
        return None

    # ── 4. Volume confirmation ───────────────────────────────────────────
    vol_lookback = min(20, len(candles) - 2)
    vol_window = candles[-vol_lookback - 1 : -1]
    volume_ratio, open_candle = _volume_ratio(latest, vol_window, interval_ms, now_ms)
    if volume_ratio < FR_MIN_VOLUME_RATIO:
        return None
    avg_quote_volume = _average_quote_volume(vol_window)
    min_required_quote_volume = settings.min_avg_quote_volume * interval_ms / BASE_FLOW_INTERVAL_MS
    if avg_quote_volume < min_required_quote_volume:
        return None

    # ── 5. Stop & target ─────────────────────────────────────────────────
    # The logical invalidation for a failed-rally short is the rally peak.
    # If price gets back above the rally high, the "failed rally" thesis
    # is wrong — buyers are still in control.
    stop_price = rally_high * (1.0 + settings.stop_buffer_pct)
    trigger_price = latest.close
    if stop_price <= trigger_price:
        return None
    risk = stop_price - trigger_price
    target_price = max(trigger_price - risk * FR_TARGET_R, EPSILON)

    return _build_signal(
        symbol=symbol,
        interval=interval,
        side="SHORT",
        status="FADE",
        close=latest.close,
        resistance=latest.high,
        support=latest.low,
        breakout_pct=(latest.high - latest.close) / max(latest.high, EPSILON),
        move_pct=(latest.open - latest.close) / max(latest.open, EPSILON),
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        trigger_price=trigger_price,
        stop_price=stop_price,
        target_price=target_price,
        volume_ratio=volume_ratio,
        avg_quote_volume=avg_quote_volume,
        min_required_quote_volume=min_required_quote_volume,
        compression_pct=0.0,
        atr_pct=atr_pct,
        close_position=1.0 - close_pos,
        trend_score=_trend_score(candles, side="SHORT"),
        quote_volume_24h=quote_volume_24h,
        trade_count_24h=0,
        range_pct_24h=range_pct_24h,
        price_change_pct_24h=0.0,
        open_candle=open_candle,
        settings=settings,
    )


def _move_pct(status: str, breakout_pct: float, reclaim_pct: float, resistance_sweep_pct: float) -> float:
    if status == "SHAKEOUT":
        return reclaim_pct
    if status == "FAKEOUT":
        return resistance_sweep_pct
    return breakout_pct


def _stop_price(status: str, support: float, latest_low: float, settings: BreakoutSettings) -> float:
    base = latest_low if status == "SHAKEOUT" else support
    return base * (1 - settings.stop_buffer_pct)
