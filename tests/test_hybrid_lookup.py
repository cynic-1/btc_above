"""
HybridPriceLookup 测试

覆盖:
- 新鲜 orderbook 优先
- 过期 orderbook → CLOB 回退
- 仅 CLOB / 仅 orderbook / 都无数据
- staleness 边界
"""

import csv
import gzip
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from backtest.hybrid_lookup import HybridPriceLookup
from backtest.orderbook_reader import OrderbookPriceLookup, OrderbookQuote
from backtest.polymarket_discovery import PolymarketMarketInfo
from backtest.polymarket_trades import PolymarketPriceLookup, PolymarketTradeCache


def _make_orderbook(tmpdir: str, cid: str, data: list) -> OrderbookPriceLookup:
    """创建含测试数据的 orderbook lookup

    data: [(ts_ms, bid, ask), ...]
    """
    npz_path = os.path.join(tmpdir, f"{cid[:16]}.npz")
    ts = np.array([d[0] for d in data], dtype=np.int64)
    bids = np.array([d[1] for d in data], dtype=np.float64)
    asks = np.array([d[2] for d in data], dtype=np.float64)
    np.savez(npz_path, timestamps_ms=ts, best_bids=bids, best_asks=asks)
    lookup = OrderbookPriceLookup(cache_dir=tmpdir)
    lookup.preload([cid])
    return lookup


def _make_clob(tmpdir: str, cid: str, data: list) -> PolymarketPriceLookup:
    """创建含测试数据的 CLOB lookup

    data: [(ts_ms, price), ...]
    """
    cache = PolymarketTradeCache(cache_dir=tmpdir)
    path = cache._file_path(cid)
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "price", "size"])
        for ts, price in data:
            writer.writerow([ts, price, 0])
    lookup = PolymarketPriceLookup(cache)
    lookup.preload([cid])
    return lookup


class TestHybridFreshOrderbook:
    """orderbook 新鲜时应优先使用"""

    def test_fresh_orderbook_used(self):
        """orderbook 数据时间戳在阈值内 → 使用 orderbook mid_price"""
        cid = "cid_fresh_test_01"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [
            (1000, 0.40, 0.50),
            (2000, 0.45, 0.55),
        ])
        clob = _make_clob(clob_dir, cid, [
            (1000, 0.60),
            (2000, 0.70),
        ])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=5000)

        # query_ts=2500, data_ts=2000, diff=500 < 5000 → 新鲜
        price = hybrid.get_price_at(cid, 2500)
        # orderbook mid = (0.45+0.55)/2 = 0.50
        assert price == pytest.approx(0.50)

    def test_fresh_orderbook_quote(self):
        """新鲜时 get_quote_at 返回 orderbook quote"""
        cid = "cid_fresh_quote_1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.60)])
        clob = _make_clob(clob_dir, cid, [(1000, 0.55)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=5000)
        quote = hybrid.get_quote_at(cid, 1500)

        assert quote is not None
        assert quote.best_bid == 0.40
        assert quote.best_ask == 0.60
        assert quote.mid_price == pytest.approx(0.50)


class TestHybridStaleOrderbook:
    """orderbook 过期时应回退 CLOB"""

    def test_stale_falls_back_to_clob(self):
        """orderbook 数据过期 → 使用 CLOB 价格"""
        cid = "cid_stale_test_01"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [
            (1000, 0.40, 0.50),
        ])
        clob = _make_clob(clob_dir, cid, [
            (1000, 0.60),
            (5000, 0.75),
        ])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=2000)

        # query_ts=5000, data_ts=1000, diff=4000 > 2000 → 过期
        price = hybrid.get_price_at(cid, 5000)
        assert price == 0.75  # CLOB 的价格

    def test_stale_quote_returns_none(self):
        """过期时 get_quote_at 返回 None（不构造虚假报价）"""
        cid = "cid_stale_quote_1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.60)])
        clob = _make_clob(clob_dir, cid, [(5000, 0.80)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=2000)
        quote = hybrid.get_quote_at(cid, 5000)

        assert quote is None


class TestHybridStaleReturnsNone:
    """orderbook 过期时 get_quote_at 返回 None（不构造虚假报价）"""

    def test_stale_quote_returns_none(self):
        """过期时 get_quote_at 应返回 None 而非合成 quote"""
        cid = "cid_stale_none_01"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.60)])
        clob = _make_clob(clob_dir, cid, [(5000, 0.80)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=2000)

        # query_ts=5000, data_ts=1000, diff=4000 > 2000 → 过期
        quote = hybrid.get_quote_at(cid, 5000)
        assert quote is None

    def test_stale_bid_ask_excluded(self):
        """过期时 get_bid_ask_at 不包含该 strike"""
        cid = "cid_stale_ba_no1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.60)])
        clob = _make_clob(clob_dir, cid, [(5000, 0.80)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=2000)
        markets = {
            ("2026-03-05", 85000.0): PolymarketMarketInfo(
                event_date="2026-03-05", strike=85000.0,
                condition_id=cid, yes_token_id="y", no_token_id="n",
                question="test",
            ),
        }

        ba = hybrid.get_bid_ask_at(markets, "2026-03-05", 5000, [85000.0])
        assert 85000.0 not in ba


class TestHybridSingleSource:
    """仅有一个数据源的场景"""

    def test_only_clob(self):
        """orderbook 无数据 → CLOB"""
        cid = "cid_only_clob_01"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = OrderbookPriceLookup(cache_dir=ob_dir)
        ob.preload([cid])
        clob = _make_clob(clob_dir, cid, [(1000, 0.65)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=5000)
        price = hybrid.get_price_at(cid, 1500)
        assert price == 0.65

    def test_only_orderbook(self):
        """CLOB 无数据 → orderbook（如果新鲜）"""
        cid = "cid_only_ob_0001"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.50)])
        clob = PolymarketPriceLookup(PolymarketTradeCache(cache_dir=clob_dir))
        clob.preload([cid])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=5000)
        price = hybrid.get_price_at(cid, 1500)
        assert price == pytest.approx(0.45)  # mid = (0.40+0.50)/2

    def test_both_missing(self):
        """两个源都无数据 → None"""
        cid = "cid_both_miss_01"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = OrderbookPriceLookup(cache_dir=ob_dir)
        ob.preload([cid])
        clob = PolymarketPriceLookup(PolymarketTradeCache(cache_dir=clob_dir))
        clob.preload([cid])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=5000)
        assert hybrid.get_price_at(cid, 1000) is None


class TestHybridStalenessEdge:
    """staleness 边界测试"""

    def test_exact_threshold_is_not_stale(self):
        """diff == threshold → 不过期（用 >，不用 >=）"""
        cid = "cid_edge_exact_1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.50)])
        clob = _make_clob(clob_dir, cid, [(1000, 0.99)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=2000)

        # query_ts=3000, data_ts=1000, diff=2000 == threshold → 不过期
        price = hybrid.get_price_at(cid, 3000)
        assert price == pytest.approx(0.45)  # orderbook mid

    def test_one_over_threshold_is_stale(self):
        """diff == threshold + 1 → 过期"""
        cid = "cid_edge_over__1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.50)])
        clob = _make_clob(clob_dir, cid, [(1000, 0.99)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=2000)

        # query_ts=3001, data_ts=1000, diff=2001 > 2000 → 过期
        price = hybrid.get_price_at(cid, 3001)
        assert price == 0.99  # CLOB


class TestHybridMarketPrices:
    """get_market_prices_at / get_bid_ask_at 集成测试"""

    def _make_markets(self, cid: str, event_date: str, strike: float):
        return {
            (event_date, strike): PolymarketMarketInfo(
                event_date=event_date, strike=strike,
                condition_id=cid, yes_token_id="y", no_token_id="n",
                question="test",
            ),
        }

    def test_market_prices_fresh(self):
        cid = "cid_mkt_fresh_01"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.50)])
        clob = _make_clob(clob_dir, cid, [(1000, 0.80)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=5000)
        markets = self._make_markets(cid, "2026-03-05", 85000.0)

        prices = hybrid.get_market_prices_at(
            markets, "2026-03-05", 1500, [85000.0],
        )
        assert 85000.0 in prices
        assert prices[85000.0] == pytest.approx(0.45)

    def test_bid_ask_fresh(self):
        cid = "cid_ba_fresh_001"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.60)])
        clob = _make_clob(clob_dir, cid, [(1000, 0.55)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=5000)
        markets = self._make_markets(cid, "2026-03-05", 85000.0)

        ba = hybrid.get_bid_ask_at(
            markets, "2026-03-05", 1500, [85000.0],
        )
        assert 85000.0 in ba
        assert ba[85000.0] == (0.40, 0.60)

    def test_bid_ask_stale_excluded(self):
        """过期时 get_bid_ask_at 不包含该 strike（quote 为 None）"""
        cid = "cid_ba_stale_001"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.60)])
        clob = _make_clob(clob_dir, cid, [(5000, 0.75)])

        hybrid = HybridPriceLookup(ob, clob, staleness_threshold_ms=2000)
        markets = self._make_markets(cid, "2026-03-05", 85000.0)

        ba = hybrid.get_bid_ask_at(
            markets, "2026-03-05", 5000, [85000.0],
        )
        assert 85000.0 not in ba


class TestHybridFirstTimestamp:
    """get_first_timestamp 测试"""

    def test_both_sources(self):
        """取两源中更早的时间戳"""
        cid = "cid_first_both_1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(2000, 0.40, 0.50)])
        clob = _make_clob(clob_dir, cid, [(1000, 0.60)])

        hybrid = HybridPriceLookup(ob, clob)
        assert hybrid.get_first_timestamp(cid) == 1000

    def test_only_orderbook_ts(self):
        cid = "cid_first_ob___1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(2000, 0.40, 0.50)])
        clob = PolymarketPriceLookup(PolymarketTradeCache(cache_dir=clob_dir))
        clob.preload([cid])

        hybrid = HybridPriceLookup(ob, clob)
        assert hybrid.get_first_timestamp(cid) == 2000

    def test_neither(self):
        cid = "cid_first_none_1"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = OrderbookPriceLookup(cache_dir=ob_dir)
        clob = PolymarketPriceLookup(PolymarketTradeCache(cache_dir=clob_dir))

        hybrid = HybridPriceLookup(ob, clob)
        assert hybrid.get_first_timestamp(cid) is None


class TestHybridPreload:
    """preload 测试"""

    def test_preload_both(self):
        """preload 应调用两个源的 preload"""
        cid = "cid_preload_0001"
        ob_dir = tempfile.mkdtemp()
        clob_dir = tempfile.mkdtemp()

        ob = _make_orderbook(ob_dir, cid, [(1000, 0.40, 0.50)])
        clob = _make_clob(clob_dir, cid, [(1000, 0.60)])

        hybrid = HybridPriceLookup(ob, clob)
        # preload 不应报错
        hybrid.preload([cid])

        # 验证数据可用
        assert hybrid.get_price_at(cid, 1500) is not None
