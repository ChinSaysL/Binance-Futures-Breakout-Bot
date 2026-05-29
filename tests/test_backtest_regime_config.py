import json
import unittest
from argparse import Namespace

from screener.backtest import (
    BacktestTrade,
    _normalize_regime_overrides,
    _parse_args,
    _portfolio_ordered_trades,
)


def _trade(symbol: str, *, rel: float, score: float, momentum: float, interval: str = "1h") -> BacktestTrade:
    trade = BacktestTrade(
        symbol=symbol,
        side="LONG",
        status="BREAKOUT",
        regime="INSTANT",
        detected_time=900,
        entry_time=1_000,
        exit_time=2_000,
        entry_price=10.0,
        stop_price=9.0,
        target_price=12.0,
        avg_exit_price=11.0,
        leverage=10,
        hold_candles=1,
        hold_hours=1.0,
        r_multiple=1.0,
        price_return=0.1,
    )
    trade.feat_rel_momentum_pct = rel
    trade.feat_score = score
    trade.feat_reward_risk = 2.0
    trade.momentum_score = momentum
    trade.interval = interval
    return trade


class BacktestRegimeConfigTests(unittest.TestCase):
    def test_window_config_aliases_map_to_simulator_keys(self):
        overrides = _normalize_regime_overrides(
            {
                "breakeven_r": 1.0,
                "stag_after_r": 0.5,
                "stag_candles": 12,
                "HP_INST_REL": 5.0,
            }
        )

        self.assertEqual(overrides["breakeven_trigger_r"], 1.0)
        self.assertEqual(overrides["stagnation_after_r"], 0.5)
        self.assertEqual(overrides["stagnation_candles"], 12)
        self.assertEqual(overrides["HP_INST_REL"], 5.0)

    def test_window_config_can_override_allowed_entry_regimes(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "window_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "W3_bull_strong": {
                            "regime": "BULL_STRONG",
                            "allowed_entry_regimes": ["STRICT_RETEST"],
                            "overrides": {"stag_after_r": 0.5},
                        }
                    }
                ),
                encoding="utf-8",
            )

            args = _parse_args(["--window-config", str(config_path)])

        self.assertEqual(args.regime_allowed_entry_map["BULL_STRONG"], frozenset({"STRICT_RETEST"}))
        self.assertEqual(args.regime_override_map["BULL_STRONG"]["stagnation_after_r"], 0.5)
        self.assertEqual(args.workers, 16)

    def test_portfolio_ordering_is_stable_for_same_candle_candidates(self):
        args = Namespace(ml_rank_signals=False, ml_filter_start_ms=None)
        weak = _trade("WEAKUSDT", rel=1.0, score=91.0, momentum=0.4)
        strong = _trade("STRONGUSDT", rel=10.0, score=98.0, momentum=0.9)

        self.assertEqual(_portfolio_ordered_trades([weak, strong], args)[0].symbol, "STRONGUSDT")
        self.assertEqual(_portfolio_ordered_trades([strong, weak], args)[0].symbol, "STRONGUSDT")


if __name__ == "__main__":
    unittest.main()
