import unittest

from screener.breakout import BreakoutSettings, Candle, evaluate_breakout, interval_to_ms


class BreakoutEvaluationTests(unittest.TestCase):
    def test_detects_pre_breakout_before_trigger(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=100.8, high=101.9, low=100.7, close=101.8, volume=1_400.0))

        signal = evaluate_breakout(
            "READYUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.status, "PRE_BREAKOUT")
        self.assertGreater(signal.distance_to_trigger_pct, 0)
        self.assertGreater(signal.trigger_price, signal.close)

    def test_excludes_confirmed_breakout_by_default(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=101.5, high=103.4, low=101.2, close=103.0, volume=2_200.0))

        signal = evaluate_breakout(
            "TESTUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNone(signal)

    def test_detects_confirmed_breakout_only_when_requested(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=101.5, high=103.4, low=101.2, close=103.0, volume=2_200.0))

        signal = evaluate_breakout(
            "TESTUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            include_confirmed=True,
            now_ms=10**12,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.status, "BREAKOUT")
        self.assertGreater(signal.breakout_pct, 0)
        self.assertGreaterEqual(signal.volume_ratio, 1.45)

    def test_rejects_late_extended_move(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=102.0, high=111.5, low=101.8, close=110.0, volume=3_000.0))

        signal = evaluate_breakout(
            "LATEUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNone(signal)

    def test_rejects_setup_with_too_little_candle_flow(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=100.8, high=101.9, low=100.7, close=101.8, volume=1_400.0))

        signal = evaluate_breakout(
            "FLOWUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=BreakoutSettings(min_avg_quote_volume=25_000),
            now_ms=10**12,
        )

        self.assertIsNone(signal)

    def test_rejects_pre_breakout_that_already_impulsed_from_base(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=98.0, volume=1_000.0)
        candles.append(_candle(70, open_=100.9, high=101.8, low=100.8, close=101.7, volume=1_400.0))

        signal = evaluate_breakout(
            "CHASEUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNone(signal)

    def test_rejects_pre_breakout_after_recent_trigger_wick_rejection(self):
        candles = _flat_candles(count=66, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(66, open_=101.0, high=102.3, low=100.8, close=101.2, volume=1_600.0))
        candles.extend(_flat_candles(count=3, price=101.0, high=101.8, low=100.4, volume=1_000.0))
        candles.append(_candle(70, open_=100.9, high=101.9, low=100.7, close=101.8, volume=1_400.0))

        signal = evaluate_breakout(
            "WICKUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNone(signal)

    def test_detects_bullish_shakeout_support_sweep(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=99.4, high=101.0, low=98.4, close=100.6, volume=2_300.0))

        signal = evaluate_breakout(
            "SHAKEUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "LONG")
        self.assertEqual(signal.status, "SPRING")
        self.assertGreater(signal.sweep_pct, 0)
        self.assertGreater(signal.trigger_price, signal.close)

    def test_detects_short_pre_breakdown_before_trigger(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=100.1, high=100.2, low=99.1, close=99.2, volume=1_400.0))

        signal = evaluate_breakout(
            "SHORTUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "SHORT")
        self.assertEqual(signal.status, "PRE_BREAKDOWN")
        self.assertLess(signal.trigger_price, signal.close)

    def test_detects_resistance_sweep_as_short_upthrust(self):
        candles = _flat_candles(count=70, price=100.0, high=102.0, low=99.0, volume=1_000.0)
        candles.append(_candle(70, open_=101.4, high=103.2, low=101.0, close=101.3, volume=1_600.0))

        signal = evaluate_breakout(
            "FAKEUSDT",
            candles,
            quote_volume_24h=10_000_000,
            interval_ms=interval_to_ms("15m"),
            settings=_settings(),
            now_ms=10**12,
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "SHORT")
        self.assertEqual(signal.status, "UPTHRUST")
        self.assertLess(signal.trigger_price, signal.close)


def _flat_candles(count: int, price: float, high: float, low: float, volume: float) -> list[Candle]:
    return [_candle(index, open_=price, high=high, low=low, close=price, volume=volume) for index in range(count)]


def _settings() -> BreakoutSettings:
    return BreakoutSettings(min_avg_quote_volume=0)


def _candle(index: int, open_: float, high: float, low: float, close: float, volume: float) -> Candle:
    open_time = index * 900_000
    return Candle(
        open_time=open_time,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time=open_time + 899_999,
        quote_volume=volume,
    )
