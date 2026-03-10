"""
orderbook_reader 测试

验证 bisect 正确性、无前瞻偏差、接口兼容性
"""

import json
import os
import tempfile

import numpy as np
import pytest

from backtest.orderbook_reader import (
    OrderbookPriceLookup,
    OrderbookQuote,
    load_markets_from_events_json,
)
from backtest.polymarket_discovery import PolymarketMarketInfo


@pytest.fixture
def cache_dir_with_data():
    """创建包含合成 npz 数据的缓存目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # conditionId = "0xaabbccdd11223344..."
        cid = "0xaabbccdd11223344"
        timestamps = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
        bids = np.array([0.40, 0.45, 0.50, 0.48, 0.52], dtype=np.float64)
        asks = np.array([0.42, 0.47, 0.52, 0.50, 0.54], dtype=np.float64)

        np.savez_compressed(
            os.path.join(tmpdir, f"{cid[:16]}.npz"),
            timestamps_ms=timestamps,
            best_bids=bids,
            best_asks=asks,
        )
        yield tmpdir, cid


@pytest.fixture
def events_json_path():
    """创建合成 events.json"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "events.json")
        events = {
            "2026-02-21": {
                "title": "Bitcoin above ___ on February 21?",
                "markets": [
                    {
                        "question": "Will the price of Bitcoin be above $85,000 on February 21?",
                        "conditionId": "0xaabbccdd11223344",
                        "clobTokenIds": ["token_yes_1", "token_no_1"],
                    },
                    {
                        "question": "Will the price of Bitcoin be above $90,000 on February 21?",
                        "conditionId": "0x1122334455667788",
                        "clobTokenIds": ["token_yes_2", "token_no_2"],
                    },
                ],
            }
        }
        with open(path, "w") as f:
            json.dump(events, f)
        yield path


class TestOrderbookPriceLookup:
    """OrderbookPriceLookup 查询测试"""

    def test_get_price_at_exact(self, cache_dir_with_data):
        """精确时间戳命中"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        price = lookup.get_price_at(cid, 3000)
        # mid = (0.50 + 0.52) / 2 = 0.51
        assert price is not None
        assert abs(price - 0.51) < 1e-10

    def test_get_price_at_between(self, cache_dir_with_data):
        """时间戳在两个点之间: 返回较早的（forward-fill）"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        price = lookup.get_price_at(cid, 2500)
        # 应返回 ts=2000 的值: mid = (0.45 + 0.47) / 2 = 0.46
        assert price is not None
        assert abs(price - 0.46) < 1e-10

    def test_no_lookahead_bias(self, cache_dir_with_data):
        """不使用未来数据: 查询时间早于所有数据 → None"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        price = lookup.get_price_at(cid, 500)
        assert price is None

    def test_get_price_at_unknown_cid(self, cache_dir_with_data):
        """未知 conditionId → None"""
        cache_dir, _ = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)

        price = lookup.get_price_at("0xunknown", 3000)
        assert price is None

    def test_get_quote_at(self, cache_dir_with_data):
        """get_quote_at 返回完整报价"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        quote = lookup.get_quote_at(cid, 3000)
        assert quote is not None
        assert isinstance(quote, OrderbookQuote)
        assert quote.timestamp_ms == 3000
        assert abs(quote.best_bid - 0.50) < 1e-10
        assert abs(quote.best_ask - 0.52) < 1e-10
        assert abs(quote.mid_price - 0.51) < 1e-10

    def test_get_first_timestamp(self, cache_dir_with_data):
        """get_first_timestamp 返回最早时间"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        first = lookup.get_first_timestamp(cid)
        assert first == 1000

    def test_get_market_prices_at(self, cache_dir_with_data):
        """get_market_prices_at 接口兼容性"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        markets = {
            ("2026-02-21", 85000.0): PolymarketMarketInfo(
                event_date="2026-02-21",
                strike=85000.0,
                condition_id=cid,
                yes_token_id="tok1",
                no_token_id="tok2",
                question="test",
            ),
        }

        prices = lookup.get_market_prices_at(
            markets=markets,
            event_date="2026-02-21",
            timestamp_ms=3000,
            k_grid=[85000.0],
            max_snap_diff=250.0,
        )
        assert 85000.0 in prices
        assert abs(prices[85000.0] - 0.51) < 1e-10

    def test_get_market_prices_at_snap(self, cache_dir_with_data):
        """strike snap: k_grid 中的 K 与 Polymarket K 不完全匹配"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        markets = {
            ("2026-02-21", 85000.0): PolymarketMarketInfo(
                event_date="2026-02-21",
                strike=85000.0,
                condition_id=cid,
                yes_token_id="tok1",
                no_token_id="tok2",
                question="test",
            ),
        }

        # k=85100 与 85000 差 100 < max_snap_diff=250
        prices = lookup.get_market_prices_at(
            markets=markets,
            event_date="2026-02-21",
            timestamp_ms=3000,
            k_grid=[85100.0],
        )
        assert 85100.0 in prices

    def test_get_market_prices_at_wrong_date(self, cache_dir_with_data):
        """错误日期 → 空结果"""
        cache_dir, cid = cache_dir_with_data
        lookup = OrderbookPriceLookup(cache_dir=cache_dir)
        lookup.preload([cid])

        markets = {
            ("2026-02-21", 85000.0): PolymarketMarketInfo(
                event_date="2026-02-21",
                strike=85000.0,
                condition_id=cid,
                yes_token_id="tok1",
                no_token_id="tok2",
                question="test",
            ),
        }

        prices = lookup.get_market_prices_at(
            markets=markets,
            event_date="2026-02-22",  # 没有这个日期的市场
            timestamp_ms=3000,
            k_grid=[85000.0],
        )
        assert len(prices) == 0

    def test_bid_only_fallback(self):
        """ask=0 时 mid_price 回退到 bid"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cid = "0xbidonly00000000"
            np.savez_compressed(
                os.path.join(tmpdir, f"{cid[:16]}.npz"),
                timestamps_ms=np.array([1000], dtype=np.int64),
                best_bids=np.array([0.50], dtype=np.float64),
                best_asks=np.array([0.0], dtype=np.float64),
            )
            lookup = OrderbookPriceLookup(cache_dir=tmpdir)
            lookup.preload([cid])

            price = lookup.get_price_at(cid, 1000)
            assert price == 0.50


class TestLoadMarketsFromEventsJson:
    """load_markets_from_events_json 测试"""

    def test_basic(self, events_json_path):
        """基础加载"""
        markets = load_markets_from_events_json(events_json_path)
        assert len(markets) == 2
        assert ("2026-02-21", 85000.0) in markets
        assert ("2026-02-21", 90000.0) in markets

        info = markets[("2026-02-21", 85000.0)]
        assert info.condition_id == "0xaabbccdd11223344"
        assert info.yes_token_id == "token_yes_1"
        assert info.event_date == "2026-02-21"
