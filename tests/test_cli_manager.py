import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from argparse import Namespace
from pathlib import Path

from screener.cli import _dynamic_leverage, manage_pending_exits


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


class DynamicLeverageTests(unittest.TestCase):
    def test_safety_cap_reduces_leverage_on_dangerously_wide_stop(self):
        # 20% stop with base=10 would be a 200% margin loss = forced liquidation.
        # Safety floor (75% of margin) caps leverage at floor(0.75 / 0.20) = 3.
        self.assertEqual(_dynamic_leverage(atr_pct=0.01, risk_pct=0.20, base=10), 3)

    def test_tight_stop_scales_leverage_up_from_base(self):
        # 2% stop is half the reference 4%; risk-parity doubles leverage.
        self.assertEqual(_dynamic_leverage(atr_pct=0.01, risk_pct=0.02, base=10), 20)

    def test_normal_stop_keeps_base_leverage(self):
        # 4% stop = reference; no adjustment.
        self.assertEqual(_dynamic_leverage(atr_pct=0.03, risk_pct=0.04, base=10), 10)

    def test_wide_but_safe_stop_does_not_downscale(self):
        # Previous behaviour would have downscaled to 7x on 6% ATR. New
        # behaviour keeps base because the stop is still within safety bounds.
        self.assertEqual(_dynamic_leverage(atr_pct=0.06, risk_pct=0.05, base=10), 10)

    def test_hard_cap_at_25x(self):
        # Very tight 0.5% stop would imply 80x by pure risk parity.
        self.assertEqual(_dynamic_leverage(atr_pct=0.005, risk_pct=0.005, base=10), 25)


class _FakeClient:
    api_key = "key"
    api_secret = "secret"

    def __init__(self, mark_price: float) -> None:
        self._mark_price = mark_price
        self.orders: list[dict[str, str]] = []
        self.cancelled_algos: list[tuple[str, str]] = []

    def account_info(self, recv_window: int = 5000):
        return {"positions": []}

    def mark_price(self, symbol: str) -> float:
        return self._mark_price

    def mark_prices(self) -> dict[str, float]:
        return {"TESTUSDT": self._mark_price}

    def change_leverage(self, symbol: str, leverage: int, recv_window: int = 5000):
        return {"leverage": leverage}

    def change_margin_type(self, symbol: str, margin_type: str, recv_window: int = 5000):
        return {"marginType": margin_type}

    def place_order(self, payload: dict[str, str], test: bool, recv_window: int = 5000):
        self.orders.append(payload)
        return {"status": "NEW", "orderId": "123"}

    def place_algo_order(self, payload: dict[str, str], recv_window: int = 5000):
        return {"algoStatus": "NEW", "algoId": "456"}

    def cancel_algo_order(self, symbol: str, client_algo_id: str, recv_window: int = 5000):
        self.cancelled_algos.append((symbol, client_algo_id))
        return {"status": "CANCELED"}


def _pending_entry(state: str, mark_side: str) -> dict[str, object]:
    triggered_at = time.time()
    if mark_side == "timeout":
        triggered_at -= 301
    return {
        "created_at": int(time.time()),
        "state": state,
        "symbol": "TESTUSDT",
        "side": "LONG",
        "interval": "15m",
        "hedge_mode": False,
        "binance_side": "BUY",
        "quantity": "1",
        "trigger_price": "100",
        "limit_price": "99.5",
        "retest_timeout_seconds": 300,
        "triggered_at": triggered_at,
        "entry_client_order_id": "bd_TESTUSDT_L15m_1",
        "leverage": 5,
        "margin_type": "",
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
        dynamic_sl=False,
        sl_update_interval_seconds=300.0,
        sl_lookback=20,
    )


def _quiet_manage(client: _FakeClient, args: Namespace) -> int:
    with redirect_stdout(StringIO()):
        return manage_pending_exits(client, args)
