import unittest
from decimal import Decimal

from screener.orders import OrderPlanError, TradingRule, build_entry_order_plan, build_exit_order_plans
from screener.breakout import BreakoutSignal


class OrderPlanTests(unittest.TestCase):
    def test_builds_long_stop_market_order_with_exchange_rounding(self):
        signal = _signal(side="LONG", trigger=100.004)

        plan = build_entry_order_plan(
            signal=signal,
            rule=_rule(),
            requested_notional=25,
            client_order_id="test_long",
            working_type="MARK_PRICE",
            price_protect=False,
            hedge_mode=False,
        )

        self.assertEqual(plan.binance_side, "BUY")
        self.assertEqual(plan.trigger_price, "100.01")
        self.assertEqual(plan.limit_price, "")
        self.assertEqual(plan.quantity, "0.2")
        self.assertEqual(plan.payload["algoType"], "CONDITIONAL")
        self.assertEqual(plan.payload["type"], "STOP_MARKET")
        self.assertEqual(plan.payload["triggerPrice"], "100.01")
        self.assertEqual(plan.payload["clientAlgoId"], "test_long")
        self.assertNotIn("stopPrice", plan.payload)
        self.assertNotIn("newClientOrderId", plan.payload)
        self.assertNotIn("positionSide", plan.payload)

    def test_builds_retest_limit_entry_below_long_trigger(self):
        signal = _signal(side="LONG", trigger=100.0)

        plan = build_entry_order_plan(
            signal=signal,
            rule=_rule(),
            requested_notional=25,
            client_order_id="test_retest",
            working_type="MARK_PRICE",
            price_protect=False,
            hedge_mode=False,
            entry_mode="RETEST_LIMIT",
            entry_pullback_pct=0.5,
        )

        self.assertEqual(plan.order_type, "STOP_LIMIT")
        self.assertEqual(plan.payload["type"], "STOP")
        self.assertEqual(plan.payload["triggerPrice"], "100")
        self.assertEqual(plan.payload["price"], "99.5")
        self.assertEqual(plan.payload["timeInForce"], "GTC")
        self.assertEqual(plan.limit_price, "99.5")
        self.assertNotIn("priceProtect", plan.payload)

    def test_builds_short_stop_market_order_with_hedge_position_side(self):
        signal = _signal(side="SHORT", trigger=99.996)

        plan = build_entry_order_plan(
            signal=signal,
            rule=_rule(),
            requested_notional=25,
            client_order_id="test_short",
            working_type="MARK_PRICE",
            price_protect=True,
            hedge_mode=True,
        )

        self.assertEqual(plan.binance_side, "SELL")
        self.assertEqual(plan.trigger_price, "99.99")
        self.assertEqual(plan.payload["positionSide"], "SHORT")
        self.assertEqual(plan.payload["priceProtect"], "true")

    def test_builds_close_position_stop_loss_and_take_profit_exits(self):
        signal = _signal(side="LONG", trigger=100.0)

        stop_loss, take_profit = build_exit_order_plans(
            signal=signal,
            rule=_rule(),
            entry_quantity="1.0",
            stop_client_order_id="test_sl",
            target_client_order_ids=["test_tp"],
            target_splits_pct=[100],
            trailing_client_order_id=None,
            trailing_callback_pct=None,
            trailing_quantity_pct=0,
            working_type="MARK_PRICE",
            price_protect=False,
            hedge_mode=True,
        )

        self.assertEqual(stop_loss.role, "STOP_LOSS")
        self.assertEqual(stop_loss.binance_side, "SELL")
        self.assertEqual(stop_loss.order_type, "STOP_MARKET")
        self.assertEqual(stop_loss.trigger_price, "95")
        self.assertEqual(stop_loss.payload["closePosition"], "true")
        self.assertEqual(stop_loss.payload["positionSide"], "LONG")
        self.assertNotIn("quantity", stop_loss.payload)
        self.assertEqual(take_profit.role, "TAKE_PROFIT_1")
        self.assertEqual(take_profit.binance_side, "SELL")
        self.assertEqual(take_profit.order_type, "TAKE_PROFIT_MARKET")
        self.assertEqual(take_profit.trigger_price, "110")
        self.assertEqual(take_profit.quantity, "1")

    def test_builds_multiple_take_profits_and_trailing_stop(self):
        signal = _signal(side="LONG", trigger=100.0)

        stop_loss, tp1, tp2, trailing = build_exit_order_plans(
            signal=signal,
            rule=_rule(),
            entry_quantity="1.0",
            stop_client_order_id="test_sl",
            target_client_order_ids=["test_tp1", "test_tp2"],
            target_splits_pct=[40, 35],
            trailing_client_order_id="test_trl",
            trailing_callback_pct=1.5,
            trailing_quantity_pct=25,
            working_type="MARK_PRICE",
            price_protect=False,
            hedge_mode=False,
        )

        self.assertEqual(stop_loss.role, "STOP_LOSS")
        self.assertEqual(tp1.role, "TAKE_PROFIT_1")
        self.assertEqual(tp1.quantity, "0.4")
        self.assertEqual(tp1.trigger_price, "105")
        self.assertEqual(tp1.payload["reduceOnly"], "true")
        self.assertEqual(tp2.role, "TAKE_PROFIT_2")
        self.assertEqual(tp2.quantity, "0.3")
        self.assertEqual(tp2.trigger_price, "110")
        self.assertEqual(trailing.role, "TRAILING_STOP")
        self.assertEqual(trailing.order_type, "TRAILING_STOP_MARKET")
        # Trail consumes the remainder (entry - sum(TPs)) instead of an
        # independently-floored 25% slice, so TP1+TP2+trail = entry exactly
        # with no lot-step dust. Prior expectation was 0.2 (= 25% floored),
        # leaving 0.1 dust on a 1.0-unit position.
        self.assertEqual(trailing.quantity, "0.3")
        self.assertNotIn("activatePrice", trailing.payload)
        self.assertEqual(trailing.payload["callbackRate"], "1.5")
        self.assertEqual(trailing.payload["reduceOnly"], "true")

    def test_rejects_notional_below_exchange_minimum(self):
        with self.assertRaises(OrderPlanError):
            build_entry_order_plan(
                signal=_signal(side="LONG", trigger=100.0),
                rule=_rule(min_notional=Decimal("50")),
                requested_notional=10,
                client_order_id="too_small",
                working_type="MARK_PRICE",
                price_protect=False,
                hedge_mode=False,
            )


def _rule(min_notional=Decimal("5")):
    return TradingRule(
        symbol="TESTUSDT",
        price_tick_size=Decimal("0.01"),
        quantity_step_size=Decimal("0.1"),
        min_qty=Decimal("0.1"),
        max_qty=Decimal("1000"),
        min_notional=min_notional,
    )


def _signal(side: str, trigger: float):
    return BreakoutSignal(
        symbol="TESTUSDT",
        interval="1h",
        side=side,
        status="PRE_BREAKOUT" if side == "LONG" else "PRE_BREAKDOWN",
        score=80.0,
        close=99.0,
        resistance=100.0,
        support=95.0,
        breakout_pct=0.0,
        move_pct=0.0,
        sweep_pct=0.0,
        distance_to_trigger_pct=0.01,
        condition="",
        order_type="BUY STOP_MARKET" if side == "LONG" else "SELL STOP_MARKET",
        trigger_price=trigger,
        stop_price=95.0,
        target_price=110.0,
        risk_pct=0.05,
        reward_pct=0.1,
        reward_risk=2.0,
        volume_ratio=1.2,
        avg_quote_volume=100_000,
        min_required_quote_volume=25_000,
        compression_pct=0.03,
        atr_pct=0.01,
        trend_score=0.7,
        close_position=0.7,
        quote_volume_24h=10_000_000,
        trade_count_24h=50_000,
        range_pct_24h=5.0,
        price_change_pct_24h=1.0,
        book_min_depth=100_000,
        open_interest_notional=10_000_000,
        open_candle=True,
    )
