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
                return payload if payload.get("ok") else None
        except (urllib.error.URLError, json.JSONDecodeError, ValueError, OSError):
            return None

    def _get(self, method: str, params: dict[str, object], timeout: float | None = None) -> dict | None:
        if not self.enabled:
            return None
        query = urllib.parse.urlencode(params)
        url = f"{self.BASE}/bot{self.token}/{method}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=timeout or self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload if payload.get("ok") else None
        except (urllib.error.URLError, json.JSONDecodeError, ValueError, OSError):
            return None

    def send_message(self, text: str, silent: bool = False, parse_mode: str = "HTML") -> bool:
        params: dict[str, object] = {
            "chat_id": self.chat_id,
            "text": text[:4090],  # Telegram limit is 4096; small buffer for safety
            "disable_notification": "true" if silent else "false",
            "disable_web_page_preview": "true",
        }
        if parse_mode:
            params["parse_mode"] = parse_mode
        return self._post("sendMessage", params) is not None

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

    # ---- outbound ----

    def send(self, text: str, silent: bool = False) -> bool:
        if not self.enabled:
            return False
        # Trivial throttle so a burst of events doesn't trip Telegram's rate limit.
        gap = time.time() - self._last_send_ts
        if gap < self._min_send_interval:
            time.sleep(self._min_send_interval - gap)
        sent = self.api.send_message(text, silent=silent)
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
                self.send(f"Unknown command <code>{html.escape(command)}</code>. Try /help.")
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


# -------- formatting helpers --------


def fmt_money(value: float, decimals: int = 2) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def format_entry_filled(symbol: str, side: str, entry_price: float, quantity: float, sl: float, tp1: float | None = None) -> str:
    lines = [
        f"\U0001f7e2 <b>ENTRY FILLED</b> {html.escape(symbol)} {side}",
        f"  entry: <code>{entry_price:g}</code>",
        f"  size:  <code>{quantity:g}</code>",
        f"  SL:    <code>{sl:g}</code>",
    ]
    if tp1 is not None and tp1 > 0:
        lines.append(f"  TP1:   <code>{tp1:g}</code>")
    return "\n".join(lines)


def format_position_closed(symbol: str, side: str, reason: str, exit_price: float, pnl_usdt: float, pnl_pct: float, hold_minutes: float) -> str:
    icon = "\U0001f7e2" if pnl_usdt > 0 else "\U0001f534"
    return "\n".join([
        f"{icon} <b>CLOSED</b> {html.escape(symbol)} {side} — {html.escape(reason)}",
        f"  exit:  <code>{exit_price:g}</code>",
        f"  PnL:   <b>{fmt_money(pnl_usdt)} USDT</b> ({fmt_pct(pnl_pct)})",
        f"  held:  {hold_minutes:.0f} min",
    ])


def format_error(message: str) -> str:
    return f"⚠️ <b>ERROR</b>\n<code>{html.escape(message[:1500])}</code>"
