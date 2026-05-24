import unittest

from screener.breakout import BreakoutSignal
from screener.take_profit import equal_take_profit_profile, smart_take_profit_profile


class TakeProfitProfileTests(unittest.TestCase):
    def test_smart_take_profit_expands_strong_long_and_backloads_splits(self):
        signal = _signal(
            side="LONG",
            trigger=100.0,
            stop=96.0,
            target=110.0,
            volume_ratio=6.0,
            atr_pct=0.05,
            trend_score=1.0,
            close_position=0.9,
            range_pct_24h=55.0,
            reward_risk=2.5,
        )

        profile = smart_take_profit_profile(signal, tp_count=2, trailing_stop=True, base_runner_pct=25.0)

        self.assertGreater(profile.signal.target_price, signal.target_price)
        self.assertGreater(profile.runner_pct, 25.0)
        self.assertAlmostEqual(sum(profile.tp_splits_pct) + profile.runner_pct, 100.0)
        self.assertGreater(profile.tp_splits_pct[1], profile.tp_splits_pct[0])

    def test_equal_take_profit_keeps_original_target_and_even_splits(self):
        signal = _signal(side="LONG", trigger=100.0, stop=96.0, target=110.0)

        profile = equal_take_profit_profile(signal, tp_count=2, trailing_stop=True, runner_pct=25.0)

        self.assertEqual(profile.signal.target_price, 110.0)
        self.assertEqual(profile.tp_splits_pct, [37.5, 37.5])
        self.assertEqual(profile.runner_pct, 25.0)

    def test_smart_take_profit_expands_short_target_lower(self):
        signal = _signal(
            side="SHORT",
            trigger=100.0,
            stop=104.0,
            target=90.0,
            volume_ratio=6.0,
            atr_pct=0.05,
            trend_score=1.0,
            close_position=0.9,
            range_pct_24h=55.0,
            reward_risk=2.5,
        )

        profile = smart_take_profit_profile(signal, tp_count=2, trailing_stop=True, base_runner_pct=25.0)

        self.assertLess(profile.signal.target_price, signal.target_price)
        self.assertGreater(profile.runner_pct, 25.0)


def _signal(
    *,
    side: str,
    trigger: float,
    stop: float,
    target: float,
    volume_ratio: float = 1.5,
    atr_pct: float = 0.02,
    trend_score: float = 0.6,
    close_position: float = 0.7,
    range_pct_24h: float = 10.0,
    reward_risk: float = 1.5,
) -> BreakoutSignal:
    return BreakoutSignal(
        symbol="TESTUSDT",
        interval="1h",
        side=side,
        status="BREAKOUT" if side == "LONG" else "BREAKDOWN",
        score=80.0,
        close=trigger,
        resistance=trigger,
        support=stop,
        breakout_pct=0.0,
        move_pct=0.0,
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        condition="",
        order_type="BUY STOP_MARKET" if side == "LONG" else "SELL STOP_MARKET",
        trigger_price=trigger,
        stop_price=stop,
        target_price=target,
        risk_pct=abs(trigger - stop) / trigger,
        reward_pct=abs(target - trigger) / trigger,
        reward_risk=reward_risk,
        volume_ratio=volume_ratio,
        avg_quote_volume=100_000,
        min_required_quote_volume=25_000,
        compression_pct=0.03,
        atr_pct=atr_pct,
        trend_score=trend_score,
        close_position=close_position,
        quote_volume_24h=50_000_000,
        trade_count_24h=50_000,
        range_pct_24h=range_pct_24h,
        price_change_pct_24h=1.0,
        book_min_depth=100_000,
        open_interest_notional=10_000_000,
        open_candle=False,
    )


if __name__ == "__main__":
    unittest.main()
