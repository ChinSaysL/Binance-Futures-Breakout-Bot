"""Telegram integration: push notifications + long-poll command handling.

Stdlib only - mirrors the rest of the project. The notifier is non-blocking
on the bot's main loop (network failures degrade gracefully) and the poller
reads /getUpdates at scan boundaries so commands take effect within seconds.

Configure via .env:
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=123456789

Get a token from @BotFather. Get your chat_id by messaging @userinfobot
(or any chat-id bot) on Telegram.
"""

from __future__ import annotations

import html
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable


class TelegramAPI:
    """Thin wrapper around the Telegram Bot HTTP API."""

    BASE = "https://api.telegram.org"

    def __init__(self, token: str, chat_id: str, timeout: float = 10.0) -> None:
        self.token = (token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.timeout = timeout
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def _post(self, method: str, params: dict[str, object]) -> dict | None:
        if not self.enabled:
            return None
        url = f"{self.BASE}/bot{self.token}/{method}"
        try:
            data = urllib.parse.urlencode(params).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("ok"):
                    self.last_error = ""
                    return payload
                self.last_error = _telegram_error_description(payload)
                return None
        except urllib.error.HTTPError as exc:
            self.last_error = _telegram_http_error(exc)
            return None
        except (urllib.error.URLError, json.JSONDecodeError, ValueError, OSError) as exc:
            self.last_error = str(exc)
            return None

    def _get(self, method: str, params: dict[str, object], timeout: float | None = None) -> dict | None:
        if not self.enabled:
            return None
        query = urllib.parse.urlencode(params)
        url = f"{self.BASE}/bot{self.token}/{method}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=timeout or self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("ok"):
                    self.last_error = ""
                    return payload
                self.last_error = _telegram_error_description(payload)
                return None
        except urllib.error.HTTPError as exc:
            self.last_error = _telegram_http_error(exc)
            return None
        except (urllib.error.URLError, json.JSONDecodeError, ValueError, OSError) as exc:
            self.last_error = str(exc)
            return None

    def send_message(
        self,
        text: str,
        silent: bool = False,
        parse_mode: str = "HTML",
        reply_markup: dict | None = None,
    ) -> bool:
        params: dict[str, object] = {
            "chat_id": self.chat_id,
            "text": text[:4090],  # Telegram limit is 4096; small buffer for safety
            "disable_notification": "true" if silent else "false",
            "disable_web_page_preview": "true",
        }
        if parse_mode:
            params["parse_mode"] = parse_mode
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)
        return self._post("sendMessage", params) is not None

    def delete_webhook(self, drop_pending_updates: bool = False) -> bool:
        return self._post("deleteWebhook", {
            "drop_pending_updates": "true" if drop_pending_updates else "false",
        }) is not None

    def set_my_commands(self, commands: list[dict[str, str]]) -> bool:
        return self._post("setMyCommands", {"commands": json.dumps(commands)}) is not None

    def get_updates(self, offset: int, long_poll_timeout: int = 0) -> list[dict]:
        params = {
            "offset": offset,
            "timeout": long_poll_timeout,
            "allowed_updates": json.dumps(["message"]),
        }
        # urlopen timeout must exceed the long-poll timeout server-side.
        resp = self._get("getUpdates", params, timeout=max(self.timeout, long_poll_timeout + 5))
        if not resp:
            return []
        return resp.get("result") or []


@dataclass
class CommandContext:
    """Mutable state passed to every command handler."""

    args: object
    client: object  # BinanceClient, kept untyped to avoid a runtime import cycle
    state: dict = field(default_factory=dict)


CommandHandler = Callable[[CommandContext, list[str]], str]


class TelegramBot:
    """Notifier + command poller + state holder."""

    def __init__(self, token: str, chat_id: str, timeout: float = 10.0) -> None:
        self.api = TelegramAPI(token, chat_id, timeout=timeout)
        self._next_offset = 0
        self._last_poll = 0.0
        self._commands: dict[str, CommandHandler] = {}
        self.reply_markup: dict | None = None
        # Shared mutable state (e.g. paused flag, session stats) - exposed to handlers.
        self.state: dict = {
            "paused": False,
            "session_start": time.time(),
            "trades": [],  # list of {"symbol","reason","pnl","time"}
            "errors": 0,
        }
        # 1 message per second is well under the per-chat rate limit (~30/min).
        self._last_send_ts = 0.0
        self._min_send_interval = 0.5

    @property
    def enabled(self) -> bool:
        return self.api.enabled

    def register(self, command: str, handler: CommandHandler) -> None:
        self._commands[command.lower()] = handler

    def prepare_for_polling(self) -> bool:
        """Make long-polling usable even if a webhook was configured earlier."""
        if not self.enabled:
            return False
        return self.api.delete_webhook(drop_pending_updates=False)

    def set_commands(self, commands: list[dict[str, str]]) -> bool:
        if not self.enabled:
            return False
        return self.api.set_my_commands(commands)

    # ---- outbound ----

    def send(self, text: str, silent: bool = False, reply_markup: dict | None = None) -> bool:
        if not self.enabled:
            return False
        if reply_markup is None:
            reply_markup = self.reply_markup
        # Trivial throttle so a burst of events doesn't trip Telegram's rate limit.
        gap = time.time() - self._last_send_ts
        if gap < self._min_send_interval:
            time.sleep(self._min_send_interval - gap)
        sent = self.api.send_message(text, silent=silent, reply_markup=reply_markup)
        if not sent and "parse" in self.api.last_error.lower():
            sent = self.api.send_message(text, silent=silent, parse_mode="", reply_markup=reply_markup)
        self._last_send_ts = time.time()
        return sent

    # ---- inbound (long-poll) ----

    def poll_commands(self, ctx: CommandContext, long_poll_timeout: int = 0) -> int:
        """Fetch and dispatch any pending /commands from the user. Returns the
        count of commands handled this tick.

        Pass long_poll_timeout > 0 to block on the server side; 0 returns
        immediately. The bot uses 0 because we poll between scan iterations
        and don't want to stall the manager loop.
        """
        if not self.enabled:
            return 0
        updates = self.api.get_updates(self._next_offset, long_poll_timeout=long_poll_timeout)
        if not updates and "webhook" in self.api.last_error.lower():
            self.api.delete_webhook(drop_pending_updates=False)
        if not updates:
            return 0
        handled = 0
        for update in updates:
            self._next_offset = int(update.get("update_id", 0)) + 1
            msg = update.get("message") or {}
            chat_id = str(((msg.get("chat") or {}).get("id")) or "")
            if chat_id != self.api.chat_id:
                continue  # only respond to the configured operator
            text = str(msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue
            parts = text.split()
            command = parts[0].split("@", 1)[0].lower()  # strip @BotName suffix
            handler = self._commands.get(command)
            if not handler:
                self.send(format_warning(
                    "Unknown command",
                    f"{fmt_code(command)} is not registered. Send {fmt_code('/help')} for the command list.",
                ))
                continue
            try:
                reply = handler(ctx, parts[1:])
            except Exception as exc:  # noqa: BLE001 - never let a bad command crash the trader
                reply = f"⚠️ command failed: <code>{html.escape(str(exc))}</code>"
            if reply:
                self.send(reply)
            handled += 1
        return handled

    # ---- shared state helpers ----

    def record_trade(self, symbol: str, reason: str, pnl: float) -> None:
        """Append a closed-trade record to the rolling session log."""
        self.state["trades"].append({
            "symbol": symbol,
            "reason": reason,
            "pnl": float(pnl),
            "time": time.time(),
        })
        # Cap memory: keep only the most recent 500 trades.
        if len(self.state["trades"]) > 500:
            self.state["trades"] = self.state["trades"][-500:]

    def session_stats(self) -> dict:
        trades = self.state.get("trades", [])
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        return {
            "n": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
            "total_pnl": total_pnl,
            "avg_win": (sum(t["pnl"] for t in wins) / len(wins)) if wins else 0.0,
            "avg_loss": (sum(t["pnl"] for t in losses) / len(losses)) if losses else 0.0,
            "session_hours": (time.time() - self.state["session_start"]) / 3600.0,
        }


def _telegram_error_description(payload: dict) -> str:
    description = payload.get("description")
    if description:
        return str(description)
    return json.dumps(payload, sort_keys=True)


def _telegram_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        return _telegram_error_description(payload)
    except (json.JSONDecodeError, ValueError, OSError):
        return f"HTTP {exc.code}: {exc.reason}"


# -------- formatting helpers --------


def fmt_money(value: float, decimals: int = 2) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


TELEGRAM_RULE = "\u2501" * 18
TELEGRAM_BULLET = "\u2022"


def fmt_code(value: object) -> str:
    return f"<code>{html.escape(str(value))}</code>"


def format_header(icon: str, title: str, subtitle: str = "") -> str:
    lines = [f"{icon} <b>{html.escape(title)}</b>", TELEGRAM_RULE]
    if subtitle:
        lines.append(subtitle)
    return "\n".join(lines)


def format_kv(label: str, value: str) -> str:
    return f"{TELEGRAM_BULLET} <b>{html.escape(label)}:</b> {value}"


def format_success(title: str, body: str) -> str:
    return "\n".join([
        format_header("\U00002705", title),
        body,
    ])


def format_warning(title: str, body: str) -> str:
    return "\n".join([
        format_header("\U000026A0\ufe0f", title),
        body,
    ])


def format_empty(title: str, body: str) -> str:
    return "\n".join([
        format_header("\U0001f4ed", title),
        body,
    ])


def command_keyboard() -> dict:
    """Persistent safe-action keyboard shown under Telegram messages."""
    return {
        "keyboard": [
            [{"text": "/status"}, {"text": "/positions"}],
            [{"text": "/queue"}, {"text": "/stats"}, {"text": "/equity"}],
            [{"text": "/pause"}, {"text": "/resume"}, {"text": "/help"}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
        "one_time_keyboard": False,
    }


def format_entry_filled(symbol: str, side: str, entry_price: float, quantity: float, sl: float, tp1: float | None = None) -> str:
    lines = [
        format_header("\U0001f7e2", "Entry Filled"),
        format_kv("Symbol", fmt_code(symbol)),
        format_kv("Side", f"<b>{html.escape(side)}</b>"),
        format_kv("Entry", fmt_code(f"{entry_price:g}")),
        format_kv("Size", fmt_code(f"{quantity:g}")),
        format_kv("Stop loss", fmt_code(f"{sl:g}")),
    ]
    if tp1 is not None and tp1 > 0:
        lines.append(format_kv("TP1", fmt_code(f"{tp1:g}")))
    return "\n".join(lines)


def format_position_closed(symbol: str, side: str, reason: str, exit_price: float, pnl_usdt: float, pnl_pct: float, hold_minutes: float) -> str:
    icon = "\U0001f7e2" if pnl_usdt > 0 else "\U0001f534"
    return "\n".join([
        format_header(icon, "Position Closed"),
        format_kv("Symbol", fmt_code(symbol)),
        format_kv("Side", f"<b>{html.escape(side)}</b>"),
        format_kv("Reason", html.escape(reason)),
        format_kv("Exit", fmt_code(f"{exit_price:g}")),
        format_kv("PnL", f"<b>{fmt_money(pnl_usdt)} USDT</b> ({fmt_pct(pnl_pct)})"),
        format_kv("Held", f"{hold_minutes:.0f} min"),
    ])
    return "\n".join([
        f"{icon} <b>CLOSED</b> {html.escape(symbol)} {side} — {html.escape(reason)}",
        f"  exit:  <code>{exit_price:g}</code>",
        f"  PnL:   <b>{fmt_money(pnl_usdt)} USDT</b> ({fmt_pct(pnl_pct)})",
        f"  held:  {hold_minutes:.0f} min",
    ])


def format_error(message: str) -> str:
    return format_warning("Error", fmt_code(message[:1500]))
    return f"⚠️ <b>ERROR</b>\n<code>{html.escape(message[:1500])}</code>"
