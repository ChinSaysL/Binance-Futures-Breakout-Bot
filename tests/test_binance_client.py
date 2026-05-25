import unittest

from screener.binance_client import BinanceClient, discover_symbols, filter_by_open_interest, filter_by_order_book, order_book_metrics


class DiscoverSymbolsTests(unittest.TestCase):
    def test_filters_dead_futures_contracts(self):
        client = FakeClient()

        universe = discover_symbols(
            client=client,
            quote_asset="USDT",
            min_quote_volume=5_000_000,
            min_trade_count=10_000,
            min_range_pct=0.8,
            top=0,
        )

        self.assertEqual([symbol.symbol for symbol in universe.symbols], ["LIVEUSDT"])
        self.assertEqual(universe.stats.perpetual_symbols, 4)
        self.assertEqual(universe.stats.filtered_low_volume, 1)
        self.assertEqual(universe.stats.filtered_low_trades, 1)
        self.assertEqual(universe.stats.filtered_low_range, 1)

    def test_order_book_metrics_measure_both_sides_near_mid(self):
        book = {
            "bids": [["99.90", "100"], ["99.20", "50"], ["97.00", "1000"]],
            "asks": [["100.10", "80"], ["100.80", "50"], ["103.00", "1000"]],
        }

        metrics = order_book_metrics(book, depth_pct=1.0)

        self.assertAlmostEqual(metrics.bid_depth, 99.90 * 100 + 99.20 * 50)
        self.assertAlmostEqual(metrics.ask_depth, 100.10 * 80 + 100.80 * 50)
        self.assertAlmostEqual(metrics.min_depth, min(metrics.bid_depth, metrics.ask_depth))
        self.assertAlmostEqual(metrics.spread_bps, 20.0)

    def test_filters_thin_order_books(self):
        client = FakeClient()
        universe = discover_symbols(
            client=client,
            quote_asset="USDT",
            min_quote_volume=0,
            min_trade_count=0,
            min_range_pct=0,
            top=0,
        )

        filtered = filter_by_order_book(
            client=client,
            universe=universe,
            min_depth=10_000,
            depth_pct=1.0,
            max_spread_bps=25.0,
            limit=50,
            workers=2,
        )

        self.assertEqual([symbol.symbol for symbol in filtered.symbols], ["LIVEUSDT"])
        self.assertEqual(filtered.stats.filtered_thin_book, 2)
        self.assertEqual(filtered.stats.filtered_wide_spread, 1)

    def test_filters_low_open_interest_notional(self):
        client = FakeClient()
        universe = discover_symbols(
            client=client,
            quote_asset="USDT",
            min_quote_volume=0,
            min_trade_count=0,
            min_range_pct=0,
            top=0,
        )

        filtered = filter_by_open_interest(
            client=client,
            universe=universe,
            min_notional=5_000_000,
            workers=2,
        )

        self.assertEqual([symbol.symbol for symbol in filtered.symbols], ["LIVEUSDT"])
        self.assertEqual(filtered.symbols[0].open_interest_notional, 10_200_000)
        self.assertEqual(filtered.stats.filtered_low_open_interest, 3)

    def test_account_info_enriches_null_position_fields_from_position_risk(self):
        client = EnrichingClient()

        account = client.account_info()
        position = account["positions"][0]

        self.assertEqual(position["entryPrice"], "0.321")
        self.assertEqual(position["breakEvenPrice"], "0.3215")
        self.assertEqual(position["leverage"], "10")
        self.assertEqual(position["markPrice"], "0.323")
        self.assertIn("/fapi/v3/positionRisk", client.paths)

    def test_position_risk_falls_back_to_v2_for_missing_leverage(self):
        client = V2FallbackClient()

        account = client.account_info()
        position = account["positions"][0]

        self.assertEqual(position["leverage"], "20")
        self.assertIn("/fapi/v2/positionRisk", client.paths)


class FakeClient:
    market = "futures"

    def exchange_info(self):
        return {
            "symbols": [
                _symbol("LIVEUSDT"),
                _symbol("QUIETUSDT"),
                _symbol("THINUSDT"),
                _symbol("FLATUSDT"),
                {"symbol": "OLDUSDT", "status": "BREAK", "baseAsset": "OLD", "quoteAsset": "USDT", "contractType": "PERPETUAL"},
                {"symbol": "DATEDUSDT", "status": "TRADING", "baseAsset": "DATED", "quoteAsset": "USDT", "contractType": "CURRENT_QUARTER"},
            ]
        }

    def ticker_24h(self):
        return [
            _ticker("LIVEUSDT", quote_volume=12_000_000, count=25_000, high=105, low=99, last=102),
            _ticker("QUIETUSDT", quote_volume=1_000_000, count=25_000, high=105, low=99, last=102),
            _ticker("THINUSDT", quote_volume=12_000_000, count=2_000, high=105, low=99, last=102),
            _ticker("FLATUSDT", quote_volume=12_000_000, count=25_000, high=100.2, low=100.0, last=100.1),
        ]

    def order_book(self, symbol, limit=50):
        books = {
            "LIVEUSDT": {
                "bids": [["99.90", "200"], ["99.40", "100"]],
                "asks": [["100.10", "200"], ["100.60", "100"]],
            },
            "QUIETUSDT": {
                "bids": [["99.90", "10"]],
                "asks": [["100.10", "10"]],
            },
            "THINUSDT": {
                "bids": [["99.90", "200"]],
                "asks": [["101.00", "200"]],
            },
            "FLATUSDT": {
                "bids": [["99.90", "10"]],
                "asks": [["100.10", "10"]],
            },
        }
        return books[symbol]

    def open_interest(self, symbol):
        values = {
            "LIVEUSDT": 100_000,
            "QUIETUSDT": 10_000,
            "THINUSDT": 10,
            "FLATUSDT": 1_000,
        }
        return {"openInterest": str(values[symbol])}


class EnrichingClient(BinanceClient):
    def __init__(self):
        super().__init__(api_key="key", api_secret="secret")
        self.paths: list[str] = []

    def _signed_get(self, path, params):
        self.paths.append(path)
        if path == "/fapi/v3/account":
            return {
                "positions": [{
                    "symbol": "WLDUSDT",
                    "positionAmt": "377",
                    "positionSide": "BOTH",
                    "entryPrice": None,
                    "breakEvenPrice": None,
                    "leverage": None,
                    "unrealizedProfit": "0.8294",
                }]
            }
        if path == "/fapi/v3/positionRisk":
            return [{
                "symbol": "WLDUSDT",
                "positionAmt": "377",
                "positionSide": "BOTH",
                "entryPrice": "0.321",
                "breakEvenPrice": "0.3215",
                "leverage": "10",
                "markPrice": "0.323",
                "unRealizedProfit": "0.8294",
            }]
        raise AssertionError(path)


class V2FallbackClient(BinanceClient):
    def __init__(self):
        super().__init__(api_key="key", api_secret="secret")
        self.paths: list[str] = []

    def _signed_get(self, path, params):
        self.paths.append(path)
        if path == "/fapi/v3/account":
            return {
                "positions": [{
                    "symbol": "PHAUSDT",
                    "positionAmt": "2410",
                    "positionSide": "BOTH",
                    "entryPrice": None,
                    "breakEvenPrice": None,
                    "leverage": None,
                    "unrealizedProfit": "0.68465690",
                }]
            }
        if path == "/fapi/v3/positionRisk":
            return [{
                "symbol": "PHAUSDT",
                "positionAmt": "2410",
                "positionSide": "BOTH",
                "entryPrice": "0.05116",
                "breakEvenPrice": "0.05118558",
                "leverage": None,
                "markPrice": "0.05144409",
                "unRealizedProfit": "0.68465690",
            }]
        if path == "/fapi/v2/positionRisk":
            return [{
                "symbol": "PHAUSDT",
                "positionAmt": "2410",
                "positionSide": "BOTH",
                "entryPrice": "0.05116",
                "breakEvenPrice": "0.05118558",
                "leverage": "20",
                "markPrice": "0.05144409",
                "unRealizedProfit": "0.68465690",
            }]
        raise AssertionError(path)


def _symbol(symbol):
    return {
        "symbol": symbol,
        "status": "TRADING",
        "baseAsset": symbol.removesuffix("USDT"),
        "quoteAsset": "USDT",
        "contractType": "PERPETUAL",
    }


def _ticker(symbol, quote_volume, count, high, low, last):
    return {
        "symbol": symbol,
        "quoteVolume": str(quote_volume),
        "count": count,
        "highPrice": str(high),
        "lowPrice": str(low),
        "lastPrice": str(last),
        "priceChangePercent": "1.25",
    }
