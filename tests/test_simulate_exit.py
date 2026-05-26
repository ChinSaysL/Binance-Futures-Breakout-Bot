"""Tests for the path-conservative exit simulator.

Locks in the rule that the trailing stop in _simulate_exit uses the PRIOR
peak (not the same-bar peak) when checking against the bar's low. OHLC data
cannot say whether the high or the low came first within a bar; the live
tick path is unknown. The conservative model matches what the live trailing-
stop actually does on a worst-case tick sequence (down before up).
"""

from __future__ import annotations

import unittest
from argparse import Namespace

from screener.backtest import _simulate_exit
from screener.breakout import Candle


def _candle(idx: int, o: float, h: float, l: float, c: float) -> Candle:
    return Candle(
        open_time=idx * 3_600_000,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=1.0,
        close_time=(idx + 1) * 3_600_000 - 1,
        quote_volume=1.0,
    )


def _args(**overrides) -> Namespace:
    base = dict(
        tp_count=1,
        trailing_stop=True,
        runner_pct=50.0,
        trailing_callback_pct=1.2,
        adaptive_trailing_callback=False,
        adaptive_trailing_callback_multiplier=0.3,
        trail_activation_r=0.0,
        breakeven_trigger_r=0.0,
        breakeven_offset_pct=0.0,
        profit_lock_pairs=None,
        dynamic_sl=False,
        sl_lookback=20,
        exhaustion_exit=False,
        stagnation_after_r=0.0,
        stagnation_candles=0,
        max_sl_loss_pct=0.0,
    )
    base.update(overrides)
    return Namespace(**base)


class SimulateExitPathDependenceTests(unittest.TestCase):
    def test_long_zigzag_books_trail_loss_not_phantom_win(self):
        """The live-bot scenario: post-entry candle zig-zags down then up.

        Entry 100, stop 92 (deep enough not to fire on the dip). Callback 1.2%.
        Candle 1: open=100, high=105, low=94, close=100.
          - Live: peak starts at 100, trail at 98.8. Worst-case path: down
            first (low=94 hits 98.8), trail fires at 98.8. Bounce to 105 is
            too late.
          - Buggy old sim: peak updated to 105 first, trail=103.74,
            low<=trail, books +3.7% phantom win that doesn't happen in live.
        Fixed sim must exit the runner at 98.8.
        Remaining 50% rides into candle 2 and stops out at 92.
        Expected avg = 0.5 * 98.8 + 0.5 * 92.0 = 95.40.
        """
        candles = [
            _candle(0, 99.0, 100.5, 98.5, 100.0),   # entry bar (skipped)
            _candle(1, 100.0, 105.0, 94.0, 100.0),  # zigzag bar
            _candle(2, 100.0, 100.0, 91.0, 91.0),   # forces SL on remaining 50%
        ]
        avg_exit, _ = _simulate_exit(
            side="LONG",
            candles=candles,
            start_index=0,
            entry=100.0,
            stop=92.0,
            target=110.0,
            args=_args(),
            tp_splits_pct=[50.0],
            runner_pct=50.0,
        )
        self.assertAlmostEqual(avg_exit, 0.5 * 98.8 + 0.5 * 92.0, places=4)
        # Strictly worse than the buggy version that used same-bar peak.
        buggy_avg = 0.5 * (105.0 * 0.988) + 0.5 * 92.0
        self.assertLess(avg_exit, buggy_avg)

    def test_long_normal_uptrend_trail_still_fires_on_later_pullback(self):
        """Sanity: when the price actually trends up and pulls back on a LATER
        bar, the trail still fires at the realised peak * (1-callback). The
        fix only changes same-bar zigzag handling."""
        candles = [
            _candle(0, 99.0, 100.5, 98.5, 100.0),
            _candle(1, 100.0, 110.0, 99.6, 109.0),  # peak set to 110, low above 100*0.988
            _candle(2, 109.0, 109.5, 108.0, 108.5), # 108 <= 110*0.988=108.68 -> trail fires at 108.68
        ]
        avg_exit, _ = _simulate_exit(
            side="LONG",
            candles=candles,
            start_index=0,
            entry=100.0,
            stop=92.0,
            target=130.0,  # TP far away so it never fires in this window
            args=_args(),
            tp_splits_pct=[50.0],
            runner_pct=50.0,
        )
        # 50% runner exits at 108.68 on candle 2; remaining 50% exits at
        # last close (108.5) when the loop ends.
        self.assertAlmostEqual(avg_exit, 0.5 * 108.68 + 0.5 * 108.5, places=2)

    def test_short_zigzag_books_trail_loss_not_phantom_win(self):
        """Short mirror: post-entry candle spikes up first then dives down.

        Entry 100 short, stop 108. Callback 1.2%.
        Candle 1: open=100, high=105, low=95, close=100.
          - Live: trough at 100, trail at 101.2. Spike up to 105 hits 101.2,
            trail fires at 101.2 (1.2% loss). The dive to 95 is too late.
          - Buggy: trough updated to 95, trail=95*1.012=96.14, high>=trail,
            books +3.86% phantom win.
        Remaining 50% stopped out at 108 on candle 2.
        Expected: 0.5 * 101.2 + 0.5 * 108.0 = 104.60.
        """
        candles = [
            _candle(0, 101.0, 101.5, 99.5, 100.0),
            _candle(1, 100.0, 105.0, 95.0, 100.0),
            _candle(2, 100.0, 110.0, 100.0, 110.0),  # forces SL on remaining 50%
        ]
        avg_exit, _ = _simulate_exit(
            side="SHORT",
            candles=candles,
            start_index=0,
            entry=100.0,
            stop=108.0,
            target=90.0,
            args=_args(),
            tp_splits_pct=[50.0],
            runner_pct=50.0,
        )
        self.assertAlmostEqual(avg_exit, 0.5 * 101.2 + 0.5 * 108.0, places=4)
        buggy_avg = 0.5 * (95.0 * 1.012) + 0.5 * 108.0
        self.assertGreater(avg_exit, buggy_avg)  # for shorts, higher exit = worse

    def test_activation_uses_activation_threshold_not_full_bar_high(self):
        """With --trail-activation-r=0.5 (trail armed only after +0.5R), a bar
        that both activates AND zigzags must use the activation threshold as
        the effective peak -- never the full bar high. This prevents the
        previous bug where the trail simultaneously activated at the bar high
        AND fired at peak * (1-callback) using that same high.

        Entry 100, stop 92 (risk 8 -> activation_buffer = 0.5*8 = 4 ->
        activation at 104). Callback 1.2%.
        Candle 1: open=100, high=104.5, low=94, close=100.
          - Worst-case path: low first (95.5; trail not armed yet), then up
            to 104.5 (arms trail at peak=104 minimum, NOT 104.5).
          - effective_peak = max(prior_peak=100, 104) = 104.
          - trail = 104 * 0.988 = 102.752.
          - candle.low (94) <= 102.752 AND 102.752 > stop (92) -> trail fires.
        Fixed sim books 102.752 (using activation threshold), NOT
        104.5 * 0.988 = 103.246 (full bar high).
        """
        candles = [
            _candle(0, 99.0, 100.5, 98.5, 100.0),
            _candle(1, 100.0, 104.5, 94.0, 100.0),
            _candle(2, 100.0, 100.0, 91.0, 91.0),
        ]
        avg_exit, _ = _simulate_exit(
            side="LONG",
            candles=candles,
            start_index=0,
            entry=100.0,
            stop=92.0,
            target=130.0,
            args=_args(trail_activation_r=0.5),
            tp_splits_pct=[50.0],
            runner_pct=50.0,
        )
        # 50% runner trails out at 102.752; remaining 50% stops out at 92.
        self.assertAlmostEqual(avg_exit, 0.5 * 102.752 + 0.5 * 92.0, places=4)
        # Strictly worse than the bug that used full bar high as peak.
        buggy_avg = 0.5 * (104.5 * 0.988) + 0.5 * 92.0
        self.assertLess(avg_exit, buggy_avg)


if __name__ == "__main__":
    unittest.main()
