import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import screener.cli as cli
from screener.binance_client import BinanceClientError
from screener.breakout import BreakoutSettings, BreakoutSignal
from screener.cli import (
    _dynamic_leverage,
    _live_blocked_entry_symbols,
    _nudge_crossed_entry_trigger,
    _place_exit_plans_from_item,
    _submit_entry_order_plan,
    manage_pending_exits,
)
from screener.orders import TradingRule, build_entry_order_plan


class SmartRetestManagerTests(unittest.TestCase):
    def test_places_limit_entry_when_retest_is_reached(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            entry_file.write_text(json.dumps([_pending_entry(state="WAIT_RETEST", mark_side="retest")]), encoding="utf-8")
            client = _FakeClient(mark_price=99.5)

            result = _quiet_manage(client, _args(entry_file, exit_file))

            self.assertEqual(result, 0)
            self.assertEqual(client.orders[0]["type"], "LIMIT")
            self.assertEqual(client.orders[0]["price"], "99.5")
            remaining = json.loads(entry_file.read_text(encoding="utf-8"))
            self.assertEqual(remaining[0]["state"], "ENTRY_ORDER_PLACED")

    def test_places_market_entry_after_retest_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            entry_file.write_text(json.dumps([_pending_entry(state="WAIT_RETEST", mark_side="timeout")]), encoding="utf-8")
            client = _FakeClient(mark_price=100.5)

            result = _quiet_manage(client, _args(entry_file, exit_file))

            self.assertEqual(result, 0)
            self.assertEqual(client.orders[0]["type"], "MARKET")
            self.assertNotIn("price", client.orders[0])
            remaining = json.loads(entry_file.read_text(encoding="utf-8"))
            self.assertEqual(remaining[0]["entry_order_type"], "MARKET")

    def test_waits_when_breakout_has_not_triggered(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            entry_file.write_text(json.dumps([_pending_entry(state="WAIT_BREAKOUT", mark_side="waiting")]), encoding="utf-8")
            client = _FakeClient(mark_price=99.0)

            result = _quiet_manage(client, _args(entry_file, exit_file))

            self.assertEqual(result, 0)
            self.assertEqual(client.orders, [])
            remaining = json.loads(entry_file.read_text(encoding="utf-8"))
            self.assertEqual(remaining[0]["state"], "WAIT_BREAKOUT")

    def test_free_slot_uses_pending_priority_before_file_order(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            near = _pending_entry(
                state="WAIT_RETEST",
                mark_side="waiting",
                symbol="NEARUSDT",
                regime="TRAILING_RETEST",
                momentum=0.8541,
                trigger="2.476",
                limit="2.463",
            )
            nil = _pending_entry(
                state="WAIT_RETEST",
                mark_side="waiting",
                symbol="NILUSDT",
                regime="INSTANT",
                momentum=0.9689,
                trigger="0.07208",
                limit="0.07171",
            )
            entry_file.write_text(json.dumps([near, nil]), encoding="utf-8")
            client = _FakeClient({"NEARUSDT": 2.50, "NILUSDT": 0.073})
            args = _args(entry_file, exit_file)
            args.max_concurrent_orders = 1

            result = _quiet_manage(client, args)

            self.assertEqual(result, 0)
            self.assertEqual(client.orders[0]["symbol"], "NILUSDT")
            remaining = {item["symbol"]: item for item in json.loads(entry_file.read_text(encoding="utf-8"))}
            self.assertEqual(remaining["NILUSDT"]["state"], "ENTRY_ORDER_PLACED")
            self.assertEqual(remaining["NEARUSDT"]["state"], "WAIT_RETEST")

    def test_drops_stale_waiting_entry_before_spending_free_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            entry = _pending_entry(
                state="WAIT_RETEST",
                mark_side="waiting",
                symbol="NILUSDT",
                regime="INSTANT",
                momentum=0.9689,
                trigger="0.07208",
                limit="0.07171",
            )
            entry["triggered_at"] = time.time() - 31 * 60
            entry["entry_stale_minutes"] = 30.0
            entry_file.write_text(json.dumps([entry]), encoding="utf-8")
            client = _FakeClient({"NILUSDT": 0.073})
            args = _args(entry_file, exit_file)
            args.max_concurrent_orders = 1

            result = _quiet_manage(client, args)

            self.assertEqual(result, 0)
            self.assertEqual(client.orders, [])
            self.assertFalse(entry_file.exists())

    def test_removes_monitoring_entry_after_position_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            entry = _pending_entry(state="MONITORING", mark_side="waiting")
            entry["placed_exit_client_order_ids"] = ["test_sl", "test_tp"]
            entry_file.write_text(json.dumps([entry]), encoding="utf-8")
            client = _FakeClient(mark_price=99.0)

            result = _quiet_manage(client, _args(entry_file, exit_file))

            self.assertEqual(result, 0)
            self.assertFalse(entry_file.exists())
            self.assertEqual(client.cancelled_algos, [("TESTUSDT", "test_sl"), ("TESTUSDT", "test_tp")])

    def test_normalizes_saved_trailing_callback_before_placing_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            entry = _pending_entry(state="ENTRY_ORDER_PLACED", mark_side="waiting")
            entry["exit_plans"] = [{
                "role": "TRAILING_STOP",
                "client_order_id": "test_trl",
                "payload": {
                    "algoType": "CONDITIONAL",
                    "symbol": "TESTUSDT",
                    "side": "SELL",
                    "type": "TRAILING_STOP_MARKET",
                    "quantity": "1",
                    "callbackRate": "9.876543",
                    "activatePrice": "101",
                    "workingType": "MARK_PRICE",
                    "clientAlgoId": "test_trl",
                },
            }]
            entry_file.write_text(json.dumps([entry]), encoding="utf-8")
            client = _FakeClient(
                mark_price=99.0,
                account={"positions": [{"symbol": "TESTUSDT", "positionAmt": "1", "entryPrice": "100"}]},
            )

            result = _quiet_manage(client, _args(entry_file, exit_file))

            self.assertEqual(result, 0)
            self.assertEqual(client.algo_orders[0]["callbackRate"], "9.8")
            self.assertNotIn("activatePrice", client.algo_orders[0])

    def test_existing_close_position_stop_is_treated_as_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory) / "entries.json", Path(directory) / "exits.json")
            item = {
                "symbol": "TESTUSDT",
                "exit_plans": [
                    {
                        "role": "STOP_LOSS",
                        "client_order_id": "test_sl",
                        "payload": {
                            "algoType": "CONDITIONAL",
                            "symbol": "TESTUSDT",
                            "side": "SELL",
                            "type": "STOP_MARKET",
                            "triggerPrice": "95",
                            "closePosition": "true",
                            "clientAlgoId": "test_sl",
                        },
                    },
                    {
                        "role": "TAKE_PROFIT_1",
                        "client_order_id": "test_tp",
                        "payload": {
                            "algoType": "CONDITIONAL",
                            "symbol": "TESTUSDT",
                            "side": "SELL",
                            "type": "TAKE_PROFIT_MARKET",
                            "quantity": "1",
                            "triggerPrice": "105",
                            "clientAlgoId": "test_tp",
                        },
                    },
                ],
            }
            client = _DuplicateClosePositionStopClient()

            placed, failed, failures = _place_exit_plans_from_item(client, item, args)

        self.assertEqual(placed, 1)
        self.assertFalse(failed)
        self.assertEqual(failures, [])
        self.assertTrue(item["sl_existing_close_position"])
        self.assertEqual(item["placed_exit_client_order_ids"], ["test_sl", "test_tp"])
        self.assertEqual(client.algo_orders[-1]["type"], "TAKE_PROFIT_MARKET")

    def test_live_blocked_symbols_include_pending_exits_and_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            entry_file = Path(directory) / "entries.json"
            exit_file = Path(directory) / "exits.json"
            entry_file.write_text(json.dumps([{"symbol": "NEARUSDT"}]), encoding="utf-8")
            exit_file.write_text(json.dumps([{"symbol": "SAGAUSDT"}]), encoding="utf-8")
            args = _args(entry_file, exit_file)
            account = {"positions": [{"symbol": "DASHUSDT", "positionAmt": "1.2"}]}

            blocked = _live_blocked_entry_symbols(args, account)

        self.assertEqual(blocked, {"NEARUSDT", "SAGAUSDT", "DASHUSDT"})


class DynamicLeverageTests(unittest.TestCase):
    def test_safety_cap_reduces_leverage_on_dangerously_wide_stop(self):
        # 20% stop with base=10 would be a 200% margin loss = forced liquidation.
        # Safety floor (75% of margin) caps leverage at floor(0.75 / 0.20) = 3,
        # regardless of conviction.
        self.assertEqual(
            _dynamic_leverage(atr_pct=0.01, risk_pct=0.20, base=10, conviction=1.5),
            3,
        )

    def test_normal_conviction_keeps_base_leverage(self):
        # momentum_score around 0.5-1.0 = normal signal -> base leverage.
        self.assertEqual(
            _dynamic_leverage(atr_pct=0.03, risk_pct=0.04, base=10, conviction=0.7),
            10,
        )

    def test_weak_conviction_downscales_to_80pct(self):
        self.assertEqual(
            _dynamic_leverage(atr_pct=0.03, risk_pct=0.04, base=10, conviction=0.3),
            8,
        )

    def test_strong_conviction_scales_to_130pct(self):
        self.assertEqual(
            _dynamic_leverage(atr_pct=0.03, risk_pct=0.04, base=10, conviction=1.1),
            13,
        )

    def test_s_tier_conviction_scales_to_160pct(self):
        self.assertEqual(
            _dynamic_leverage(atr_pct=0.03, risk_pct=0.04, base=10, conviction=1.6),
            16,
        )

    def test_hard_cap_at_25x(self):
        # Conviction 1.7 boosts to base*1.6=32; needs a tight stop so the
        # safety_cap doesn't fire first. risk=2% -> safety_cap=37.
        self.assertEqual(
            _dynamic_leverage(atr_pct=0.03, risk_pct=0.02, base=20, conviction=1.7),
            25,
        )


class ConditionalEntryNudgeTests(unittest.TestCase):
    def test_nudges_crossed_long_stop_trigger_one_tick_above_mark(self):
        signal = _breakout_signal()
        rule = _rule_for(signal)
        plan = build_entry_order_plan(
            signal=signal,
            rule=rule,
            requested_notional=25,
            client_order_id="test_entry",
            working_type="MARK_PRICE",
            price_protect=False,
            hedge_mode=False,
        )
        args = Namespace(live_orders=True, order_working_type="MARK_PRICE")

        nudged = _nudge_crossed_entry_trigger(_FakeClient(mark_price=0.073), signal, plan, rule, args)

        self.assertEqual(nudged.trigger_price, "0.07301")
        self.assertEqual(nudged.payload["triggerPrice"], "0.07301")
        self.assertEqual(nudged.payload["type"], "STOP_MARKET")

    def test_keeps_uncrossed_stop_trigger_unchanged(self):
        signal = _breakout_signal()
        rule = _rule_for(signal)
        plan = build_entry_order_plan(
            signal=signal,
            rule=rule,
            requested_notional=25,
            client_order_id="test_entry",
            working_type="MARK_PRICE",
            price_protect=False,
            hedge_mode=False,
        )
        args = Namespace(live_orders=True, order_working_type="MARK_PRICE")

        nudged = _nudge_crossed_entry_trigger(_FakeClient(mark_price=0.071), signal, plan, rule, args)

        self.assertIs(nudged, plan)

    def test_retries_nudge_when_binance_mark_moves_before_submit(self):
        signal = _breakout_signal()
        rule = _rule_for(signal)
        plan = build_entry_order_plan(
            signal=signal,
            rule=rule,
            requested_notional=25,
            client_order_id="test_entry",
            working_type="MARK_PRICE",
            price_protect=False,
            hedge_mode=False,
        )
        args = Namespace(live_orders=True, order_working_type="MARK_PRICE", recv_window=5000)
        client = _RetryImmediateTriggerClient([0.073, 0.07302])

        retry_plan, response = _submit_entry_order_plan(client, signal, plan, rule, args)

        self.assertEqual(response["algoStatus"], "NEW")
        self.assertEqual(retry_plan.trigger_price, "0.07303")
        self.assertEqual(client.algo_orders[-1]["triggerPrice"], "0.07303")


class RotationPrecheckTests(unittest.TestCase):
    def test_auto_sizing_rotation_precheck_uses_account_margin(self):
        signal = _breakout_signal()
        rule = TradingRule(
            symbol=signal.symbol,
            price_tick_size=Decimal("0.00001"),
            quantity_step_size=Decimal("0.1"),
            min_qty=Decimal("0.1"),
            max_qty=Decimal("0"),
            min_notional=Decimal("5"),
        )
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                sizing_mode="auto",
                equity_peak_file=str(Path(directory) / "equity_peak.json"),
                leverage=10,
                dynamic_leverage=False,
                order_margin=0.0,
                order_notional=0.0,
                max_sl_loss_pct=35.0,
                order_working_type="MARK_PRICE",
                order_price_protect=False,
                hedge_mode=False,
                entry_mode="SMART_RETEST",
                entry_pullback_pct=0.5,
            )
            account = {"totalWalletBalance": "22.24", "positions": []}

            with (
                patch("screener.cli._fresh_order_signal", return_value=signal),
                patch("screener.cli.trading_rules_from_exchange_info", return_value={signal.symbol: rule}),
                patch("screener.cli.build_entry_order_plan") as build_entry_order_plan,
            ):
                ready, reason = cli._exploder_entry_ready(
                    _RotationPrecheckClient(),
                    signal,
                    args,
                    BreakoutSettings(),
                    account,
                )

        self.assertIs(ready, signal)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(build_entry_order_plan.call_args.kwargs["requested_notional"], 122.32, places=2)


class _FakeClient:
    api_key = "key"
    api_secret = "secret"

    def __init__(self, mark_price: float | dict[str, float], account: dict | None = None) -> None:
        self._mark_price = mark_price
        self._account = account or {"positions": []}
        self.orders: list[dict[str, str]] = []
        self.algo_orders: list[dict[str, str]] = []
        self.cancelled_algos: list[tuple[str, str]] = []

    def account_info(self, recv_window: int = 5000):
        return self._account

    def mark_price(self, symbol: str) -> float:
        if isinstance(self._mark_price, dict):
            return self._mark_price.get(symbol, 0.0)
        return self._mark_price

    def mark_prices(self) -> dict[str, float]:
        if isinstance(self._mark_price, dict):
            return dict(self._mark_price)
        return {"TESTUSDT": self._mark_price}

    def change_leverage(self, symbol: str, leverage: int, recv_window: int = 5000):
        return {"leverage": leverage}

    def change_margin_type(self, symbol: str, margin_type: str, recv_window: int = 5000):
        return {"marginType": margin_type}

    def place_order(self, payload: dict[str, str], test: bool, recv_window: int = 5000):
        self.orders.append(payload)
        return {"status": "NEW", "orderId": "123"}

    def place_algo_order(self, payload: dict[str, str], recv_window: int = 5000):
        self.algo_orders.append(payload)
        return {"algoStatus": "NEW", "algoId": "456"}

    def cancel_algo_order(self, symbol: str, client_algo_id: str, recv_window: int = 5000):
        self.cancelled_algos.append((symbol, client_algo_id))
        return {"status": "CANCELED"}


class _RotationPrecheckClient:
    def exchange_info(self):
        return {}


class _RetryImmediateTriggerClient:
    def __init__(self, marks: list[float]) -> None:
        self._marks = marks
        self.algo_orders: list[dict[str, str]] = []

    def mark_price(self, symbol: str) -> float:
        return self._marks.pop(0)

    def place_algo_order(self, payload: dict[str, str], recv_window: int = 5000):
        self.algo_orders.append(dict(payload))
        if len(self.algo_orders) == 1:
            raise BinanceClientError("Binance HTTP 400 for /algoOrder: Order would immediately trigger.")
        return {"algoStatus": "NEW", "algoId": "456"}


class _DuplicateClosePositionStopClient:
    def __init__(self) -> None:
        self.algo_orders: list[dict[str, str]] = []

    def place_algo_order(self, payload: dict[str, str], recv_window: int = 5000):
        self.algo_orders.append(dict(payload))
        if payload.get("type") == "STOP_MARKET" and payload.get("closePosition") == "true":
            raise BinanceClientError(
                "Binance HTTP 400 for /algoOrder: "
                "An open stop or take profit order with GTE and closePosition in the direction is existing."
            )
        return {"algoStatus": "NEW", "algoId": "456"}


def _rule_for(signal: BreakoutSignal) -> TradingRule:
    return TradingRule(
        symbol=signal.symbol,
        price_tick_size=Decimal("0.00001"),
        quantity_step_size=Decimal("0.1"),
        min_qty=Decimal("0.1"),
        max_qty=Decimal("0"),
        min_notional=Decimal("5"),
    )


def _breakout_signal() -> BreakoutSignal:
    return BreakoutSignal(
        symbol="NILUSDT",
        interval="15m",
        side="LONG",
        status="BREAKOUT",
        score=1.0,
        close=0.073,
        resistance=0.072,
        support=0.069,
        breakout_pct=0.01,
        move_pct=0.0,
        sweep_pct=0.0,
        distance_to_trigger_pct=0.0,
        condition="test",
        order_type="STOP_MARKET",
        trigger_price=0.07208,
        stop_price=0.06954,
        target_price=0.0866,
        risk_pct=0.035,
        reward_pct=0.20,
        reward_risk=4.0,
        volume_ratio=3.2,
        avg_quote_volume=1_000_000.0,
        min_required_quote_volume=25_000.0,
        compression_pct=0.05,
        atr_pct=0.03,
        trend_score=0.8,
        close_position=0.9,
        quote_volume_24h=100_000_000.0,
        trade_count_24h=100_000,
        range_pct_24h=18.0,
        price_change_pct_24h=7.0,
        book_min_depth=100_000.0,
        open_interest_notional=1_000_000.0,
        open_candle=False,
    )


def _pending_entry(
    state: str,
    mark_side: str,
    symbol: str = "TESTUSDT",
    regime: str = "RETEST",
    momentum: float = 0.0,
    trigger: str = "100",
    limit: str = "99.5",
) -> dict[str, object]:
    triggered_at = time.time()
    if mark_side == "timeout":
        triggered_at -= 301
    return {
        "created_at": int(time.time()),
        "state": state,
        "symbol": symbol,
        "side": "LONG",
        "interval": "15m",
        "hedge_mode": False,
        "binance_side": "BUY",
        "quantity": "1",
        "trigger_price": trigger,
        "limit_price": limit,
        "retest_timeout_seconds": 300,
        "triggered_at": triggered_at,
        "entry_client_order_id": f"bd_{symbol}_L15m_1",
        "leverage": 5,
        "margin_type": "",
        "entry_regime": regime,
        "momentum_score": momentum,
        "exit_plans": [],
    }


def _args(entry_file: Path, exit_file: Path) -> Namespace:
    return Namespace(
        entry_state_file=entry_file,
        exit_state_file=exit_file,
        recv_window=5000,
        watch_exits=False,
        exit_watch_timeout=0,
        exit_poll_seconds=1,
        exit_heartbeat_seconds=0,
        max_concurrent_orders=0,
        max_market_deviation_pct=1.5,
        no_market_fallback=False,
        entry_stale_minutes=30.0,
        dynamic_sl=False,
        sl_update_interval_seconds=300.0,
        sl_lookback=20,
        exhaustion_exit=False,
        stagnation_after_r=0.0,
        stagnation_candles=0,
    )


def _quiet_manage(client: _FakeClient, args: Namespace) -> int:
    with redirect_stdout(StringIO()):
        return manage_pending_exits(client, args)
