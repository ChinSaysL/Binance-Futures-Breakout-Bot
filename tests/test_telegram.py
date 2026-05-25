import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import screener.cli as cli
from screener.telegram import CommandContext, TelegramBot, format_entry_filled, format_position_closed


class TelegramCommandTests(unittest.TestCase):
    def test_init_deletes_webhook_before_long_polling(self):
        args = Namespace(
            telegram_bot_token="token",
            telegram_chat_id="123",
            max_concurrent_orders=2,
            sizing_mode="auto",
            scan_interval_minutes=3,
        )

        with patch("screener.telegram.TelegramAPI._post", return_value={"ok": True}) as post:
            bot = cli._init_telegram_bot(args, _FakeClient())

        self.assertTrue(bot.enabled)
        self.assertEqual(post.call_args_list[0].args[0], "deleteWebhook")
        self.assertEqual(post.call_args_list[0].args[1], {"drop_pending_updates": "true"})
        methods = [call.args[0] for call in post.call_args_list]
        self.assertIn("setMyCommands", methods)
        self.assertIn("sendMessage", methods)
        send_payload = next(call.args[1] for call in post.call_args_list if call.args[0] == "sendMessage")
        keyboard = json.loads(send_payload["reply_markup"])
        self.assertEqual(keyboard["keyboard"][0][0]["text"], "/status")
        self.assertNotIn("/cancel_all", str(keyboard))

    def test_help_command_uses_valid_html(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _command_args(directory)
            bot = TelegramBot("token", "123")
            bot._min_send_interval = 0
            bot.api = _FakeTelegramAPI([_message_update("/help", chat_id="123")], chat_id="123")
            cli._register_telegram_commands(bot, args, _FakeClient())

            handled = bot.poll_commands(CommandContext(args=args, client=_FakeClient()))

        self.assertEqual(handled, 1)
        self.assertEqual(len(bot.api.sent), 1)
        text, parse_mode = bot.api.sent[0]
        self.assertEqual(parse_mode, "HTML")
        self.assertIn("<code>/cancel SYMBOL</code>", text)
        self.assertNotIn("<SYMBOL>", text)

    def test_positions_command_uses_live_mark_when_account_prices_are_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _command_args(directory)
            bot = TelegramBot("token", "123")
            bot._min_send_interval = 0
            bot.api = _FakeTelegramAPI([_message_update("/positions", chat_id="123")], chat_id="123")
            client = _FakeClient(
                account={
                    "positions": [{
                        "symbol": "CHIPUSDT",
                        "positionAmt": "100",
                        "entryPrice": "0",
                        "markPrice": "0",
                        "unrealizedProfit": "-2.5",
                    }]
                },
                marks={"CHIPUSDT": 0.125},
            )
            cli._register_telegram_commands(bot, args, client)

            handled = bot.poll_commands(CommandContext(args=args, client=client))

        self.assertEqual(handled, 1)
        text, _parse_mode = bot.api.sent[0]
        self.assertIn("<code>CHIPUSDT</code> <b>LONG</b>", text)
        self.assertIn("• <b>Entry:</b> <code>0.15</code>", text)
        self.assertIn("• <b>Mark:</b> <code>0.125</code>", text)
        self.assertNotIn("• <b>Entry:</b> <code>0</code>", text)

    def test_send_falls_back_to_plain_text_when_html_parse_fails(self):
        bot = TelegramBot("token", "123")
        bot._min_send_interval = 0
        bot.api = _FakeTelegramAPI([], chat_id="123", fail_html_once=True)

        sent = bot.send("/cancel <SYMBOL>")

        self.assertTrue(sent)
        self.assertEqual([call[1] for call in bot.api.sent], ["HTML", ""])

    def test_stop_update_can_be_acknowledged_before_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _command_args(directory)
            bot = TelegramBot("token", "123")
            bot._min_send_interval = 0
            bot.api = _FakeTelegramAPI([_message_update("/stop", chat_id="123", update_id=41)], chat_id="123")
            cli._register_telegram_commands(bot, args, _FakeClient())

            handled = bot.poll_commands(CommandContext(args=args, client=_FakeClient()))
            bot.acknowledge_processed_updates()

        self.assertEqual(handled, 1)
        self.assertTrue(bot.state["_stop_requested"])
        self.assertIn(42, bot.api.requested_offsets)

    def test_cancel_command_drops_pending_exit_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _command_args(directory)
            args.exit_state_file.write_text(json.dumps([{
                "symbol": "PHAUSDT",
                "entry_client_order_id": "pha_entry",
            }]), encoding="utf-8")
            bot = TelegramBot("token", "123")
            bot._min_send_interval = 0
            bot.api = _FakeTelegramAPI([_message_update("/cancel PHAUSDT", chat_id="123")], chat_id="123")
            client = _FakeClient()
            cli._register_telegram_commands(bot, args, client)

            handled = bot.poll_commands(CommandContext(args=args, client=client))

        self.assertEqual(handled, 1)
        self.assertFalse(args.exit_state_file.exists())
        self.assertEqual(client.cancelled_algos, [("PHAUSDT", "pha_entry")])
        self.assertIn("Pending Entry Dropped", bot.api.sent[0][0])

    def test_cancel_all_clears_pending_exit_file(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _command_args(directory)
            args.entry_state_file.write_text(json.dumps([{"symbol": "TONUSDT"}]), encoding="utf-8")
            args.exit_state_file.write_text(json.dumps([{
                "symbol": "PHAUSDT",
                "entry_client_order_id": "pha_entry",
            }]), encoding="utf-8")
            bot = TelegramBot("token", "123")
            bot._min_send_interval = 0
            bot.api = _FakeTelegramAPI([_message_update("/cancel_all", chat_id="123")], chat_id="123")
            client = _FakeClient()
            cli._register_telegram_commands(bot, args, client)

            handled = bot.poll_commands(CommandContext(args=args, client=client))

        self.assertEqual(handled, 1)
        self.assertFalse(args.entry_state_file.exists())
        self.assertFalse(args.exit_state_file.exists())
        self.assertEqual(client.cancelled_algos, [("PHAUSDT", "pha_entry")])

    def test_trade_notifications_use_rich_layout(self):
        entry = format_entry_filled("BTCUSDT", "LONG", 100.0, 0.25, 95.0, 110.0)
        closed = format_position_closed("BTCUSDT", "LONG", "Take Profit", 110.0, 12.5, 5.2, 45)

        self.assertIn("<b>Entry Filled</b>", entry)
        self.assertIn("━━━━━━━━━━━━━━━━━━", entry)
        self.assertIn("• <b>Stop loss:</b>", entry)
        self.assertIn("<b>Position Closed</b>", closed)
        self.assertIn("• <b>PnL:</b> <b>+12.50 USDT</b>", closed)


class _FakeTelegramAPI:
    enabled = True

    def __init__(self, updates: list[dict], chat_id: str, fail_html_once: bool = False) -> None:
        self._updates = updates
        self.chat_id = chat_id
        self.fail_html_once = fail_html_once
        self.last_error = ""
        self.sent: list[tuple[str, str]] = []
        self.deleted_webhooks: list[bool] = []
        self.requested_offsets: list[int] = []

    def get_updates(self, offset: int, long_poll_timeout: int = 0) -> list[dict]:
        self.requested_offsets.append(offset)
        updates = [update for update in self._updates if int(update["update_id"]) >= offset]
        self._updates = []
        return updates

    def send_message(
        self,
        text: str,
        silent: bool = False,
        parse_mode: str = "HTML",
        reply_markup: dict | None = None,
    ) -> bool:
        if self.fail_html_once and parse_mode == "HTML":
            self.fail_html_once = False
            self.last_error = "Bad Request: can't parse entities"
            self.sent.append((text, parse_mode))
            return False
        self.last_error = ""
        self.sent.append((text, parse_mode))
        return True

    def delete_webhook(self, drop_pending_updates: bool = False) -> bool:
        self.deleted_webhooks.append(drop_pending_updates)
        return True


class _FakeClient:
    def __init__(self, account: dict | None = None, marks: dict[str, float] | None = None) -> None:
        self._account = account or {"totalWalletBalance": "100", "positions": []}
        self._marks = marks or {}
        self.orders: list[dict[str, str]] = []
        self.cancelled_algos: list[tuple[str, str]] = []
        self.signed_requests: list[tuple[str, str, dict[str, object]]] = []

    def account_info(self, recv_window: int = 5000):
        return self._account

    def mark_prices(self):
        return dict(self._marks)

    def mark_price(self, symbol: str):
        return self._marks.get(symbol, 0.0)

    def place_order(self, payload: dict[str, str], test: bool, recv_window: int = 5000):
        self.orders.append(payload)
        return {"status": "NEW"}

    def cancel_algo_order(self, symbol: str, client_algo_id: str, recv_window: int = 5000):
        self.cancelled_algos.append((symbol, client_algo_id))
        return {"status": "CANCELED"}

    def cancel_all_algo_orders(self, symbol: str, recv_window: int = 5000):
        self.cancelled_algos.append((symbol, "*"))
        return {"status": "CANCELED"}

    def _signed_request(self, method: str, path: str, params: dict[str, object]):
        self.signed_requests.append((method, path, dict(params)))
        return {"status": "OK"}


def _message_update(text: str, chat_id: str = "123", update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": chat_id},
            "text": text,
        },
    }


def _command_args(directory: str) -> Namespace:
    root = Path(directory)
    return Namespace(
        recv_window=5000,
        entry_state_file=root / "entries.json",
        exit_state_file=root / "exits.json",
        equity_peak_file=root / "equity_peak.json",
        max_concurrent_orders=2,
    )
