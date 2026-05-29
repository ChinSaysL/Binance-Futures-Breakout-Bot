import unittest
from argparse import Namespace

from screener import cli


class LiveBtcRegimeGuardTests(unittest.TestCase):
    def test_btc_regime_router_classifies_bull_strong_from_range_and_return(self):
        context = {
            "_btc_range_pos_60d": 0.82,
            "_btc_return_7d_pct": 2.1,
        }

        self.assertEqual(cli._btc_regime_from_context(context), "BULL_STRONG")

    def test_btc_regime_guard_blocks_non_strict_in_bull_strong(self):
        args = Namespace(
            btc_market_guards=True,
            btc_regime_guards=True,
            _live_ml_btc_context={
                "_btc_range_pos_60d": 0.82,
                "_btc_return_7d_pct": 2.1,
            },
        )

        self.assertIn("BULL_STRONG", cli._btc_regime_guard_reject_reason("INSTANT", args))
        self.assertEqual(cli._btc_regime_guard_reject_reason("STRICT_RETEST", args), "")


if __name__ == "__main__":
    unittest.main()
