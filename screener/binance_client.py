from __future__ import annotations

import hashlib
import hmac
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"


class BinanceClientError(RuntimeError):
    """Raised when Binance cannot return usable market data."""


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    quote_volume_24h: float
    trade_count_24h: int = 0
    range_pct_24h: float = 0.0
    price_change_pct_24h: float = 0.0
    last_price: float = 0.0
    book_bid_depth: float = 0.0
    book_ask_depth: float = 0.0
    book_min_depth: float = 0.0
    book_spread_bps: float = 0.0
    open_interest: float = 0.0
    open_interest_notional: float = 0.0


@dataclass(frozen=True)
class UniverseStats:
    total_symbols: int = 0
    quote_symbols: int = 0
    perpetual_symbols: int = 0
    active_symbols: int = 0
    missing_ticker: int = 0
    filtered_low_volume: int = 0
    filtered_low_trades: int = 0
    filtered_low_range: int = 0
    filtered_thin_book: int = 0
    filtered_wide_spread: int = 0
    order_book_failures: int = 0
    filtered_low_open_interest: int = 0
    open_interest_failures: int = 0
    top_limited: int = 0


@dataclass(frozen=True)
class SymbolUniverse:
    symbols: list[SymbolInfo]
    stats: UniverseStats


@dataclass(frozen=True)
class OrderBookMetrics:
    bid_depth: float
    ask_depth: float
    min_depth: float
    spread_bps: float


class BinanceClient:
    """Tiny Binance public market-data client using the standard library."""

    def __init__(
        self,
        market: str = "futures",
        base_url: str | None = None,
        timeout: float = 10.0,
        retries: int = 2,
        retry_sleep: float = 0.35,
        api_key: str | None = None,
        api_secret: str | None = None,
        rate_limit_rpm: float | None = None,
    ) -> None:
        if market not in {"spot", "futures"}:
            raise ValueError("market must be 'spot' or 'futures'")
        self.market = market
        self.base_url = (base_url or (SPOT_BASE_URL if market == "spot" else FUTURES_BASE_URL)).rstrip("/")
        self.path_prefix = "/api/v3" if market == "spot" else "/fapi/v1"
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.api_key = api_key
        self.api_secret = api_secret
        if rate_limit_rpm is not None and rate_limit_rpm <= 0:
            rate_limit_rpm = None
        self.rate_limit_rpm = rate_limit_rpm
        self._rate_limit_interval = 60.0 / rate_limit_rpm if rate_limit_rpm else 0.0
        self._rate_limit_lock = threading.Lock()
        self._next_request_time = 0.0
        # Offset (ms) between this machine's clock and Binance server time.
        # Signed requests add it so local clock drift does not trip the
        # "Timestamp for this request is outside of the recvWindow" 400.
        self._time_offset_ms = 0
        self._time_synced = False

    def exchange_info(self) -> dict[str, Any]:
        return self._get("/exchangeInfo")

    def ticker_24h(self, symbol: str | None = None) -> Any:
        params = {"symbol": symbol} if symbol else None
        return self._get("/ticker/24hr", params=params)

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self._get("/klines", params=params)

    def order_book(self, symbol: str, limit: int = 50) -> dict[str, Any]:
        return self._get("/depth", params={"symbol": symbol, "limit": limit})

    def open_interest(self, symbol: str) -> dict[str, Any]:
        return self._get("/openInterest", params={"symbol": symbol})

    def mark_price(self, symbol: str) -> float:
        payload = self._get("/premiumIndex", params={"symbol": symbol})
        return _as_float(payload.get("markPrice"))

    def mark_prices(self) -> dict[str, float]:
        """Fetch mark prices for every symbol in a single request."""
        payload = self._get("/premiumIndex")
        prices: dict[str, float] = {}
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict) and entry.get("symbol"):
                    prices[str(entry["symbol"])] = _as_float(entry.get("markPrice"))
        return prices

    def account_info(self, recv_window: int = 5000) -> dict[str, Any]:
        payload = self._signed_get("/fapi/v3/account", {"recvWindow": recv_window})
        self._enrich_account_positions(payload, recv_window=recv_window)
        return payload

    def position_risk(self, symbol: str | None = None, recv_window: int = 5000) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"recvWindow": recv_window}
        if symbol:
            params["symbol"] = symbol
        try:
            payload = self._signed_get("/fapi/v3/positionRisk", params)
        except BinanceClientError:
            payload = self._signed_get("/fapi/v2/positionRisk", params)
        return payload if isinstance(payload, list) else []

    def place_order(self, params: dict[str, Any], test: bool, recv_window: int = 5000) -> dict[str, Any]:
        path = "/order/test" if test else "/order"
        payload = dict(params)
        payload["recvWindow"] = recv_window
        return self._signed_post(path, payload)

    def _enrich_account_positions(self, account: dict[str, Any], recv_window: int = 5000) -> None:
        positions = account.get("positions")
        if not isinstance(positions, list):
            return
        needs_risk = False
        for position in positions:
            if not isinstance(position, dict):
                continue
            amount = _as_float(position.get("positionAmt"))
            if abs(amount) <= 0:
                continue
            if (
                _as_float(position.get("entryPrice")) <= 0
                or _as_float(position.get("breakEvenPrice")) <= 0
                or _as_float(position.get("leverage")) <= 0
            ):
                needs_risk = True
                break
        if not needs_risk:
            return
        try:
            risk_positions = self.position_risk(recv_window=recv_window)
        except BinanceClientError:
            return
        risk_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        risk_by_symbol: dict[str, dict[str, Any]] = {}
        for risk in risk_positions:
            if not isinstance(risk, dict):
                continue
            symbol = str(risk.get("symbol", ""))
            if not symbol:
                continue
            side = str(risk.get("positionSide", "BOTH"))
            risk_by_key[(symbol, side)] = risk
            if abs(_as_float(risk.get("positionAmt"))) > 0:
                risk_by_symbol[symbol] = risk
        for position in positions:
            if not isinstance(position, dict):
                continue
            symbol = str(position.get("symbol", ""))
            if not symbol or abs(_as_float(position.get("positionAmt"))) <= 0:
                continue
            side = str(position.get("positionSide", "BOTH"))
            risk = risk_by_key.get((symbol, side)) or risk_by_symbol.get(symbol)
            if not risk:
                continue
            for key in (
                "entryPrice",
                "breakEvenPrice",
                "leverage",
                "markPrice",
                "unRealizedProfit",
                "unrealizedProfit",
                "liquidationPrice",
                "marginType",
                "isolatedMargin",
            ):
                value = risk.get(key)
                if value is not None and value != "":
                    position[key] = value

    def place_algo_order(self, params: dict[str, Any], recv_window: int = 5000) -> dict[str, Any]:
        payload = dict(params)
        payload["recvWindow"] = recv_window
        return self._signed_post("/algoOrder", payload)

    def open_algo_orders(self, symbol: str | None = None, recv_window: int = 5000) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"recvWindow": recv_window}
        if symbol:
            params["symbol"] = symbol
        payload = self._signed_get("/algoOpenOrders", params)
        return payload if isinstance(payload, list) else []

    def cancel_algo_order(self, symbol: str, client_algo_id: str, recv_window: int = 5000) -> dict[str, Any]:
        return self._signed_request(
            "DELETE",
            "/algoOrder",
            {
                "symbol": symbol,
                "clientAlgoId": client_algo_id,
                "recvWindow": recv_window,
            },
        )

    def cancel_all_algo_orders(self, symbol: str, recv_window: int = 5000) -> dict[str, Any]:
        return self._signed_request(
            "DELETE",
            "/algoOpenOrders",
            {
                "symbol": symbol,
                "recvWindow": recv_window,
            },
        )

    def change_leverage(self, symbol: str, leverage: int, recv_window: int = 5000) -> dict[str, Any]:
        return self._signed_post(
            "/leverage",
            {
                "symbol": symbol,
                "leverage": leverage,
                "recvWindow": recv_window,
            },
        )

    def change_margin_type(self, symbol: str, margin_type: str, recv_window: int = 5000) -> dict[str, Any]:
        return self._signed_post(
            "/marginType",
            {
                "symbol": symbol,
                "marginType": margin_type,
                "recvWindow": recv_window,
            },
        )

    def _throttle(self) -> None:
        if self._rate_limit_interval <= 0:
            return
        with self._rate_limit_lock:
            now = time.time()
            if now < self._next_request_time:
                time.sleep(self._next_request_time - now)
                now = time.time()
            self._next_request_time = now + self._rate_limit_interval

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.base_url}{self.path_prefix}{path}{query}"
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                request = Request(url, headers={"User-Agent": "binance-breakout-screener/0.1"})
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                message = _binance_error_message(body) or body[:300] or exc.reason
                raise BinanceClientError(f"Binance HTTP {exc.code} for {path}: {message}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))

        raise BinanceClientError(f"Binance request failed for {path}: {last_error}") from last_error

    def _sync_time(self) -> None:
        """Align signed-request timestamps to Binance server time.

        Fetches the exchange clock and stores the offset from this machine's
        clock, bracketed by the request round-trip so network latency does not
        skew it. Failures are swallowed - the request falls back to the local
        clock, which is what it used before this sync existed.
        """
        try:
            local_before = int(time.time() * 1000)
            payload = self._get("/time")
            local_after = int(time.time() * 1000)
            server_ms = int(payload.get("serverTime", 0))
            if server_ms > 0:
                self._time_offset_ms = server_ms - (local_before + local_after) // 2
                self._time_synced = True
        except (BinanceClientError, ValueError, TypeError, AttributeError):
            pass

    def _signed_get(self, path: str, params: dict[str, Any]) -> Any:
        return self._signed_request("GET", path, params)

    def _signed_post(self, path: str, params: dict[str, Any]) -> Any:
        return self._signed_request("POST", path, params)

    def _signed_request(self, method: str, path: str, params: dict[str, Any]) -> Any:
        if not self.api_key or not self.api_secret:
            raise BinanceClientError("Binance API key/secret are required for signed trade endpoints.")

        if not self._time_synced:
            self._sync_time()

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            payload = {key: value for key, value in params.items() if value is not None}
            payload["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            query = urlencode(payload)
            signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
            signed_query = f"{query}&signature={signature}"
            url = self._url_for_path(path)
            data = signed_query.encode("utf-8") if method == "POST" else None
            if method in {"GET", "DELETE"}:
                url = f"{url}?{signed_query}"

            try:
                request = Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "User-Agent": "binance-breakout-screener/0.1",
                        "X-MBX-APIKEY": self.api_key,
                    },
                    method=method,
                )
                with urlopen(request, timeout=self.timeout) as response:
                    raw_body = response.read().decode("utf-8")
                    return json.loads(raw_body) if raw_body else {}
            except HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                message = _binance_error_message(body_text) or body_text[:300] or exc.reason
                # A timestamp/recvWindow 400 means our clock drifted - re-sync
                # to the server clock and retry instead of failing the request.
                if (
                    exc.code == 400
                    and ("recvWindow" in message or "Timestamp for this request" in message)
                    and attempt < self.retries
                ):
                    self._sync_time()
                    last_error = BinanceClientError(f"Binance HTTP 400 for {path}: {message}")
                    time.sleep(self.retry_sleep)
                    continue
                raise BinanceClientError(f"Binance HTTP {exc.code} for {path}: {message}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))

        raise BinanceClientError(f"Binance signed request failed for {path}: {last_error}") from last_error

    def _url_for_path(self, path: str) -> str:
        if path.startswith(("/api/", "/fapi/", "/dapi/", "/sapi/")):
            return f"{self.base_url}{path}"
        return f"{self.base_url}{self.path_prefix}{path}"


def discover_symbols(
    client: BinanceClient,
    quote_asset: str,
    min_quote_volume: float,
    min_trade_count: int,
    min_range_pct: float,
    top: int,
    include_leveraged: bool = False,
) -> SymbolUniverse:
    quote_asset = quote_asset.upper()
    exchange_info = client.exchange_info()
    tickers = client.ticker_24h()
    if isinstance(tickers, dict):
        tickers = [tickers]
    ticker_by_symbol = {ticker["symbol"]: ticker for ticker in tickers}

    symbols: list[SymbolInfo] = []
    total_symbols = 0
    quote_symbols = 0
    perpetual_symbols = 0
    missing_ticker = 0
    filtered_low_volume = 0
    filtered_low_trades = 0
    filtered_low_range = 0

    for raw_symbol in exchange_info.get("symbols", []):
        total_symbols += 1
        symbol = raw_symbol.get("symbol", "")
        base = raw_symbol.get("baseAsset", "")
        quote = raw_symbol.get("quoteAsset", "")

        if raw_symbol.get("status") != "TRADING":
            continue
        if quote_asset != "ALL" and quote != quote_asset:
            continue
        quote_symbols += 1
        if client.market == "futures" and raw_symbol.get("contractType") != "PERPETUAL":
            continue
        perpetual_symbols += 1
        if not include_leveraged and _looks_like_leveraged_token(base):
            continue

        ticker = ticker_by_symbol.get(symbol)
        if not ticker:
            missing_ticker += 1
            continue

        quote_volume = _as_float(ticker.get("quoteVolume"))
        trade_count = int(_as_float(ticker.get("count")))
        last_price = _as_float(ticker.get("lastPrice"))
        high_price = _as_float(ticker.get("highPrice"))
        low_price = _as_float(ticker.get("lowPrice"))
        range_pct = _range_pct(high_price=high_price, low_price=low_price, last_price=last_price)
        price_change_pct = _as_float(ticker.get("priceChangePercent"))

        if quote_volume < min_quote_volume:
            filtered_low_volume += 1
            continue
        if trade_count < min_trade_count:
            filtered_low_trades += 1
            continue
        if range_pct < min_range_pct:
            filtered_low_range += 1
            continue

        symbols.append(
            SymbolInfo(
                symbol=symbol,
                base_asset=base,
                quote_asset=quote,
                quote_volume_24h=quote_volume,
                trade_count_24h=trade_count,
                range_pct_24h=range_pct,
                price_change_pct_24h=price_change_pct,
                last_price=last_price,
            )
        )

    symbols.sort(key=lambda item: item.quote_volume_24h, reverse=True)
    top_limited = 0
    if top > 0 and len(symbols) > top:
        top_limited = len(symbols) - top
        symbols = symbols[:top]

    return SymbolUniverse(
        symbols=symbols,
        stats=UniverseStats(
            total_symbols=total_symbols,
            quote_symbols=quote_symbols,
            perpetual_symbols=perpetual_symbols,
            active_symbols=len(symbols),
            missing_ticker=missing_ticker,
            filtered_low_volume=filtered_low_volume,
            filtered_low_trades=filtered_low_trades,
            filtered_low_range=filtered_low_range,
            top_limited=top_limited,
        ),
    )


def load_explicit_symbols(client: BinanceClient, symbols: list[str], quote_asset: str) -> SymbolUniverse:
    quote_asset = quote_asset.upper()
    loaded: list[SymbolInfo] = []
    for symbol in symbols:
        normalized = symbol.strip().upper().replace("/", "")
        if not normalized:
            continue
        ticker = client.ticker_24h(normalized)
        last_price = _as_float(ticker.get("lastPrice"))
        high_price = _as_float(ticker.get("highPrice"))
        low_price = _as_float(ticker.get("lowPrice"))
        loaded.append(
            SymbolInfo(
                symbol=normalized,
                base_asset=normalized.removesuffix(quote_asset),
                quote_asset=quote_asset,
                quote_volume_24h=_as_float(ticker.get("quoteVolume")),
                trade_count_24h=int(_as_float(ticker.get("count"))),
                range_pct_24h=_range_pct(high_price=high_price, low_price=low_price, last_price=last_price),
                price_change_pct_24h=_as_float(ticker.get("priceChangePercent")),
                last_price=last_price,
            )
        )
    return SymbolUniverse(
        symbols=loaded,
        stats=UniverseStats(
            total_symbols=len(loaded),
            quote_symbols=len(loaded),
            perpetual_symbols=len(loaded),
            active_symbols=len(loaded),
        ),
    )


def filter_by_order_book(
    client: BinanceClient,
    universe: SymbolUniverse,
    min_depth: float,
    depth_pct: float,
    max_spread_bps: float,
    limit: int,
    workers: int,
) -> SymbolUniverse:
    if min_depth <= 0 and max_spread_bps <= 0:
        return universe

    kept: list[SymbolInfo] = []
    filtered_thin_book = 0
    filtered_wide_spread = 0
    order_book_failures = 0

    def enrich(symbol_info: SymbolInfo) -> tuple[SymbolInfo, OrderBookMetrics | None, Exception | None]:
        try:
            book = client.order_book(symbol_info.symbol, limit=limit)
            metrics = order_book_metrics(book, depth_pct=depth_pct)
            return symbol_info, metrics, None
        except Exception as exc:  # noqa: BLE001 - one bad book should not stop the universe.
            return symbol_info, None, exc

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = [executor.submit(enrich, symbol_info) for symbol_info in universe.symbols]
        for future in as_completed(futures):
            symbol_info, metrics, error = future.result()
            if error or metrics is None:
                order_book_failures += 1
                continue
            if max_spread_bps > 0 and metrics.spread_bps > max_spread_bps:
                filtered_wide_spread += 1
                continue
            if min_depth > 0 and metrics.min_depth < min_depth:
                filtered_thin_book += 1
                continue
            kept.append(
                replace(
                    symbol_info,
                    book_bid_depth=metrics.bid_depth,
                    book_ask_depth=metrics.ask_depth,
                    book_min_depth=metrics.min_depth,
                    book_spread_bps=metrics.spread_bps,
                )
            )

    kept.sort(key=lambda item: item.quote_volume_24h, reverse=True)
    stats = universe.stats
    return SymbolUniverse(
        symbols=kept,
        stats=UniverseStats(
            total_symbols=stats.total_symbols,
            quote_symbols=stats.quote_symbols,
            perpetual_symbols=stats.perpetual_symbols,
            active_symbols=len(kept),
            missing_ticker=stats.missing_ticker,
            filtered_low_volume=stats.filtered_low_volume,
            filtered_low_trades=stats.filtered_low_trades,
            filtered_low_range=stats.filtered_low_range,
            filtered_thin_book=filtered_thin_book,
            filtered_wide_spread=filtered_wide_spread,
            order_book_failures=order_book_failures,
            filtered_low_open_interest=stats.filtered_low_open_interest,
            open_interest_failures=stats.open_interest_failures,
            top_limited=stats.top_limited,
        ),
    )


def filter_by_open_interest(
    client: BinanceClient,
    universe: SymbolUniverse,
    min_notional: float,
    workers: int,
) -> SymbolUniverse:
    if min_notional <= 0:
        return universe

    kept: list[SymbolInfo] = []
    filtered_low_open_interest = 0
    open_interest_failures = 0

    def enrich(symbol_info: SymbolInfo) -> tuple[SymbolInfo, float, float, Exception | None]:
        try:
            payload = client.open_interest(symbol_info.symbol)
            open_interest = _as_float(payload.get("openInterest"))
            open_interest_notional = open_interest * symbol_info.last_price
            return symbol_info, open_interest, open_interest_notional, None
        except Exception as exc:  # noqa: BLE001 - an OI miss should not kill the whole scan.
            return symbol_info, 0.0, 0.0, exc

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = [executor.submit(enrich, symbol_info) for symbol_info in universe.symbols]
        for future in as_completed(futures):
            symbol_info, open_interest, open_interest_notional, error = future.result()
            if error:
                open_interest_failures += 1
                kept.append(symbol_info)
                continue
            if open_interest_notional < min_notional:
                filtered_low_open_interest += 1
                continue
            kept.append(
                replace(
                    symbol_info,
                    open_interest=open_interest,
                    open_interest_notional=open_interest_notional,
                )
            )

    kept.sort(key=lambda item: item.quote_volume_24h, reverse=True)
    stats = universe.stats
    return SymbolUniverse(
        symbols=kept,
        stats=UniverseStats(
            total_symbols=stats.total_symbols,
            quote_symbols=stats.quote_symbols,
            perpetual_symbols=stats.perpetual_symbols,
            active_symbols=len(kept),
            missing_ticker=stats.missing_ticker,
            filtered_low_volume=stats.filtered_low_volume,
            filtered_low_trades=stats.filtered_low_trades,
            filtered_low_range=stats.filtered_low_range,
            filtered_thin_book=stats.filtered_thin_book,
            filtered_wide_spread=stats.filtered_wide_spread,
            order_book_failures=stats.order_book_failures,
            filtered_low_open_interest=filtered_low_open_interest,
            open_interest_failures=open_interest_failures,
            top_limited=stats.top_limited,
        ),
    )


def order_book_metrics(book: dict[str, Any], depth_pct: float) -> OrderBookMetrics:
    bids = _levels(book.get("bids", []))
    asks = _levels(book.get("asks", []))
    if not bids or not asks:
        return OrderBookMetrics(bid_depth=0.0, ask_depth=0.0, min_depth=0.0, spread_bps=float("inf"))

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2
    if mid <= 0:
        return OrderBookMetrics(bid_depth=0.0, ask_depth=0.0, min_depth=0.0, spread_bps=float("inf"))

    depth_fraction = max(depth_pct, 0.0) / 100.0
    min_bid = mid * (1 - depth_fraction)
    max_ask = mid * (1 + depth_fraction)
    bid_depth = sum(price * qty for price, qty in bids if price >= min_bid)
    ask_depth = sum(price * qty for price, qty in asks if price <= max_ask)
    spread_bps = (best_ask - best_bid) / mid * 10_000
    return OrderBookMetrics(
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        min_depth=min(bid_depth, ask_depth),
        spread_bps=spread_bps,
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _looks_like_leveraged_token(base_asset: str) -> bool:
    return base_asset.endswith(("UP", "DOWN", "BULL", "BEAR"))


def _range_pct(high_price: float, low_price: float, last_price: float) -> float:
    if high_price <= 0 or low_price <= 0 or last_price <= 0:
        return 0.0
    return (high_price - low_price) / last_price * 100.0


def _levels(raw_levels: list[Any]) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    for raw_level in raw_levels:
        if len(raw_level) < 2:
            continue
        price = _as_float(raw_level[0])
        qty = _as_float(raw_level[1])
        if price > 0 and qty > 0:
            levels.append((price, qty))
    return levels


def _binance_error_message(body: str) -> str | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    message = payload.get("msg")
    return str(message) if message else None
