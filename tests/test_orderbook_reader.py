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
    PriceFilterStats,
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


# ── 数据质量过滤测试 ──────────────────────────────────────


def _make_lookup(tmpdir, cid, timestamps, bids, asks, **kwargs):
    """辅助: 创建带合成数据的 OrderbookPriceLookup"""
    np.savez_compressed(
        os.path.join(tmpdir, f"{cid[:16]}.npz"),
        timestamps_ms=np.array(timestamps, dtype=np.int64),
        best_bids=np.array(bids, dtype=np.float64),
        best_asks=np.array(asks, dtype=np.float64),
    )
    lookup = OrderbookPriceLookup(cache_dir=tmpdir, **kwargs)
    lookup.preload([cid])
    return lookup


class TestStaleFilter:
    """过期数据过滤"""

    CID = "0xstaletest000000"

    def test_stale_returns_none(self):
        """快照过期 → None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.50], asks=[0.52],
                max_stale_ms=2000,
            )
            # 查询 ts=5000, 快照 ts=1000, age=4000 > 2000
            assert lookup.get_price_at(self.CID, 5000) is None
            assert lookup.filter_stats.filtered_stale == 1

    def test_fresh_returns_mid(self):
        """快照未过期 → mid-price"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.50], asks=[0.52],
                max_stale_ms=2000,
            )
            price = lookup.get_price_at(self.CID, 2000)
            assert price is not None
            assert abs(price - 0.51) < 1e-10

    def test_stale_exact_boundary(self):
        """age == max_stale_ms → 还算有效（用 > 而非 >=）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.50], asks=[0.52],
                max_stale_ms=2000,
            )
            price = lookup.get_price_at(self.CID, 3000)
            assert price is not None

    def test_stale_disabled(self):
        """max_stale_ms=0 → 不检查过期"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.50], asks=[0.52],
                max_stale_ms=0,
            )
            price = lookup.get_price_at(self.CID, 999_999_999)
            assert price is not None
            assert lookup.filter_stats.filtered_stale == 0


class TestSpreadFilter:
    """宽幅价差过滤"""

    CID = "0xspreadtest00000"

    def test_wide_spread_returns_none(self):
        """spread > max_spread → None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.20], asks=[0.51],
                max_spread=0.15, max_stale_ms=0,
            )
            assert lookup.get_price_at(self.CID, 1000) is None
            assert lookup.filter_stats.filtered_wide_spread == 1

    def test_narrow_spread_returns_mid(self):
        """spread ≤ max_spread → mid-price"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.40], asks=[0.50],
                max_spread=0.15, max_stale_ms=0,
            )
            price = lookup.get_price_at(self.CID, 1000)
            assert price is not None
            assert abs(price - 0.45) < 1e-10

    def test_spread_just_under(self):
        """spread 刚好低于阈值 → 有效"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.40], asks=[0.54],
                max_spread=0.15, max_stale_ms=0,
            )
            price = lookup.get_price_at(self.CID, 1000)
            assert price is not None

    def test_spread_disabled(self):
        """max_spread=0 → 不检查价差"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.10], asks=[0.90],
                max_spread=0, max_stale_ms=0,
            )
            price = lookup.get_price_at(self.CID, 1000)
            assert price is not None


class TestZeroSideFilter:
    """零边报价过滤"""

    CID = "0xzerosidetest000"

    def test_reject_bid_zero(self):
        """reject 策略: bid=0 → None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.0], asks=[0.02],
                zero_side_policy="reject", max_stale_ms=0,
            )
            assert lookup.get_price_at(self.CID, 1000) is None
            assert lookup.filter_stats.filtered_zero_side == 1

    def test_reject_ask_zero(self):
        """reject 策略: ask=0 → None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.50], asks=[0.0],
                zero_side_policy="reject", max_stale_ms=0,
            )
            assert lookup.get_price_at(self.CID, 1000) is None

    def test_accept_settled_high_bid(self):
        """accept_settled 策略: bid=0.999, ask=0 → 返回 0.999"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.999], asks=[0.0],
                zero_side_policy="accept_settled", max_stale_ms=0,
            )
            price = lookup.get_price_at(self.CID, 1000)
            assert price is not None
            assert abs(price - 0.999) < 1e-10

    def test_accept_settled_low_ask(self):
        """accept_settled 策略: bid=0, ask=0.02 → None（未结算）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.0], asks=[0.02],
                zero_side_policy="accept_settled", max_stale_ms=0,
            )
            assert lookup.get_price_at(self.CID, 1000) is None

    def test_legacy_bid_zero_returns_ask(self):
        """legacy 策略: bid=0 → 返回 ask"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.0], asks=[0.02],
                zero_side_policy="legacy", max_stale_ms=0,
            )
            price = lookup.get_price_at(self.CID, 1000)
            assert price is not None
            assert abs(price - 0.02) < 1e-10

    def test_legacy_ask_zero_returns_bid(self):
        """legacy 策略: ask=0 → 返回 bid"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.50], asks=[0.0],
                zero_side_policy="legacy", max_stale_ms=0,
            )
            price = lookup.get_price_at(self.CID, 1000)
            assert price is not None
            assert abs(price - 0.50) < 1e-10


class TestFilterStats:
    """过滤统计计数"""

    CID = "0xfilterstats0000"

    def test_stats_counting(self):
        """各过滤路径计数正确"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000, 2000, 3000, 4000],
                bids=[0.50, 0.0, 0.10, 0.40],
                asks=[0.52, 0.02, 0.90, 0.42],
                max_stale_ms=500, max_spread=0.15,
                zero_side_policy="reject",
            )

            # ts=1000: 正常快照
            lookup.get_price_at(self.CID, 1000)
            # ts=2500: 最近是 ts=2000 (age=500, 刚好), bid=0 → reject
            lookup.get_price_at(self.CID, 2500)
            # ts=3000: 正常时间, spread=0.80 → wide
            lookup.get_price_at(self.CID, 3000)
            # ts=9000: 最近是 ts=4000, age=5000 > 500 → stale
            lookup.get_price_at(self.CID, 9000)
            # 未知 cid
            lookup.get_price_at("0xunknown", 1000)

            s = lookup.filter_stats
            assert s.total_queries == 5
            assert s.returned == 1
            assert s.filtered_zero_side == 1
            assert s.filtered_wide_spread == 1
            assert s.filtered_stale == 1
            assert s.no_data == 1

    def test_reset_stats(self):
        """reset_stats 清空计数"""
        with tempfile.TemporaryDirectory() as tmpdir:
            lookup = _make_lookup(
                tmpdir, self.CID,
                timestamps=[1000],
                bids=[0.50], asks=[0.52],
                max_stale_ms=0,
            )
            lookup.get_price_at(self.CID, 1000)
            assert lookup.filter_stats.total_queries == 1

            lookup.reset_stats()
            assert lookup.filter_stats.total_queries == 0
            assert lookup.filter_stats.returned == 0
