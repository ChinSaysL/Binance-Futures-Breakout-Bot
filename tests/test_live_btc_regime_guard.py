import unittest
from argparse import Namespace

from screener.breakout import BreakoutSignal
from screener import cli


def _signal(**overrides):
    values = dict(
        symbol="TESTUSDT",
        interval="1h",
        side="LONG",
        status="BREAKOUT",
        score=95.0,
        close=10.0,
        resistance=9.8,
        support=9.0,
        breakout_pct=2.0,
        move_pct=0.0,
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        condition="",
        order_type="BUY",
        trigger_price=10.0,
        stop_price=9.0,
        target_price=12.0,
        risk_pct=10.0,
        reward_pct=20.0,
        reward_risk=2.0,
        volume_ratio=4.0,
        avg_quote_volume=100_000.0,
        min_required_quote_volume=25_000.0,
        compression_pct=0.0,
        atr_pct=2.0,
        trend_score=0.0,
        close_position=0.8,
        quote_volume_24h=50_000_000.0,
        trade_count_24h=50_000,
        range_pct_24h=12.0,
        price_change_pct_24h=12.0,
        book_min_depth=50_000.0,
        open_interest_notional=10_000_000.0,
        open_candle=False,
    )
    values.update(overrides)
    return BreakoutSignal(**values)


class LiveBtcRegimeGuardTests(unittest.TestCase):
    def test_btc_regime_router_classifies_bull_strong_from_range_and_return(self):
        context = {
            "_btc_range_pos_60d": 0.82,
            "_btc_return_7d_pct": 2.1,
        }

        self.assertEqual(cli._btc_regime_from_context(context), "BULL_STRONG")
        context["_btc_range_pos_60d"] = 0.79
        self.assertEqual(cli._btc_regime_from_context(context), "BULL_RECOVERY")

    def test_btc_regime_quality_filter_matches_backtest_gates(self):
        # The live quality gate must mirror the backtest INSTANT entry gates:
        # only relative-momentum (HP_INST_REL) and volume-ratio (HP_INST_VOL) are
        # enforced. The score/RR/BTC-momentum "quality" filters were removed because
        # cross-window validation showed they filter out winning trades.
        args = Namespace(
            btc_market_guards=True,
            btc_regime_guards=True,
            _live_ml_btc_context={
                "_btc_range_pos_60d": 0.82,
                "_btc_return_7d_pct": 2.1,
                "feat_btc_momentum_pct": 3.0,
            },
            # rel momentum 7.0 clears the default 3.0 gate; no regime override map.
            _live_ml_signal_contexts={("TESTUSDT", "1h"): {"feat_rel_momentum_pct": 7.0, "feat_btc_momentum_pct": 3.0}},
            regime_override_map={},
        )

        self.assertEqual(cli._btc_regime_guard_reject_reason("INSTANT", args), "")
        # Score 95 / RR 2.0 / BTC momentum 3.0 would all have been blocked by the
        # old BULL_STRONG quality filters; now only rel/vol gate, which passes.
        self.assertEqual(cli._btc_regime_quality_reject_reason(_signal(), "INSTANT", args), "")

        # A regime override raising HP_INST_REL to 8.0 blocks rel momentum 7.0.
        args.regime_override_map = {"BULL_STRONG": {"HP_INST_REL": 8.0}}
        self.assertIn("rel strength", cli._btc_regime_quality_reject_reason(_signal(), "INSTANT", args))

        # Low volume signal blocked by the HP_INST_VOL gate (default 3.5).
        args.regime_override_map = {}
        self.assertIn("volume", cli._btc_regime_quality_reject_reason(_signal(volume_ratio=2.0), "INSTANT", args))

    def test_btc_range_position_from_candles_uses_close(self):
        from screener.breakout import Candle
        # We construct candles where candle.low/high are more extreme,
        # but the range position should only be computed from candle.close values.
        # Window size is 60 days. Let's create candles separated by 1 day (86400000 ms).
        # We need at least enough lookback.
        now_ms = 100_000_000_000
        day_ms = 86_400_000
        candles = []
        for i in range(70):
            t = now_ms - (70 - i) * day_ms
            # If we are at index 35, let close be the minimum (10.0), but set low to 5.0
            # If we are at index 55, let close be the maximum (20.0), but set high to 25.0
            # Latest candle has close 15.0
            if i == 35:
                c, l, h = 10.0, 5.0, 12.0
            elif i == 55:
                c, l, h = 20.0, 18.0, 25.0
            elif i == 69:
                c, l, h = 15.0, 14.0, 16.0
            else:
                c, l, h = 15.0, 14.0, 16.0
            
            candles.append(Candle(
                open_time=t,
                open=c,
                high=h,
                low=l,
                close=c,
                volume=100.0,
                close_time=t + day_ms - 1000,
                quote_volume=1000.0
            ))
        
        # Position should be based on close values: low close is 10.0, high close is 20.0, latest close is 15.0.
        # Position should be (15.0 - 10.0) / (20.0 - 10.0) = 0.5.
        # If it used high/low, the range would be [5.0, 25.0] and position would be (15.0 - 5.0) / (25.0 - 5.0) = 0.5 too.
        # Let's adjust values so the outputs differ.
        # If we use close: range is [10.0, 20.0], latest close is 15.0. Position = 0.5.
        # If we use low/high: low low is 5.0, high high is 25.0. Position = (15.0 - 5.0) / (25.0 - 5.0) = 0.5.
        # Let's make latest close 12.0:
        # If we use close: (12.0 - 10.0) / (20.0 - 10.0) = 0.2.
        # If we use low/high: (12.0 - 5.0) / (25.0 - 5.0) = 7 / 20 = 0.35.
        candles[-1] = Candle(
            open_time=candles[-1].open_time,
            open=12.0,
            high=13.0,
            low=11.0,
            close=12.0,
            volume=100.0,
            close_time=candles[-1].close_time,
            quote_volume=1000.0
        )
        
        pos = cli._btc_range_position_from_candles(candles, 60 * 24 * 3600 * 1000)
        self.assertAlmostEqual(pos, 0.2)  # verifies it uses close: (12 - 10) / (20 - 10) = 0.2


if __name__ == "__main__":
    unittest.main()
