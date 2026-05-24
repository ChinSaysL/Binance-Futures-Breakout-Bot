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
        self.assertEqual(post.call_args_list[0].args[1], {"drop_pending_updates": "false"})
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

    def test_send_falls_back_to_plain_text_when_html_parse_fails(self):
        bot = TelegramBot("token", "123")
        bot._min_send_interval = 0
        bot.api = _FakeTelegramAPI([], chat_id="123", fail_html_once=True)

        sent = bot.send("/cancel <SYMBOL>")

        self.assertTrue(sent)
        self.assertEqual([call[1] for call in bot.api.sent], ["HTML", ""])

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

    def get_updates(self, offset: int, long_poll_timeout: int = 0) -> list[dict]:
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
    def account_info(self, recv_window: int = 5000):
        return {"totalWalletBalance": "100", "positions": []}


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
        equity_peak_file=root / "equity_peak.json",
        max_concurrent_orders=2,
    )
