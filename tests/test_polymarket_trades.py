"""
Polymarket 交易数据缓存 + 价格查询测试
"""

import csv
import gzip
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from backtest.polymarket_discovery import PolymarketMarketInfo
from backtest.polymarket_trades import (
    PolymarketPriceLookup,
    PolymarketTrade,
    PolymarketTradeCache,
)


class TestPolymarketTradeCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = PolymarketTradeCache(cache_dir=self.tmpdir)

    def test_file_path(self):
        path = self.cache._file_path("abcdef1234567890xyz")
        assert "trades_abcdef1234567890" in path
        assert path.endswith(".csv.gz")

    def test_has_trades_false(self):
        assert not self.cache.has_trades("nonexistent")

    def test_has_trades_true(self):
        # 创建空缓存文件
        cid = "test_condition_id"
        path = self.cache._file_path(cid)
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "price", "size"])
        assert self.cache.has_trades(cid)

    @patch("backtest.polymarket_trades.requests.get")
    def test_download_prices(self, mock_get):
        """下载并缓存价格历史"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # CLOB /prices-history 返回 unix 秒级时间戳
        mock_resp.json.return_value = {
            "history": [
                {"t": 1, "p": 0.65},
                {"t": 2, "p": 0.70},
                {"t": 3, "p": 0.68},
            ]
        }
        mock_get.return_value = mock_resp

        cid = "download_test_cid"
        self.cache.download_prices(cid, "yes_token_123")

        assert self.cache.has_trades(cid)
        trades = self.cache.load_trades(cid)
        assert len(trades) == 3
        # 按时间排序, 秒→毫秒
        assert trades[0].timestamp_ms == 1000
        assert trades[1].timestamp_ms == 2000
        assert trades[2].timestamp_ms == 3000
        assert trades[0].price == 0.65

        # 验证 API 调用参数
        mock_get.assert_called()
        call_kwargs = mock_get.call_args
        assert "prices-history" in call_kwargs[1].get("url", call_kwargs[0][0] if call_kwargs[0] else "")

    @patch("backtest.polymarket_trades.requests.get")
    def test_download_empty_prices(self, mock_get):
        """无价格历史时写空文件"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"history": []}
        mock_get.return_value = mock_resp

        cid = "empty_cid"
        self.cache.download_prices(cid, "yes_token_empty")
        assert self.cache.has_trades(cid)
        trades = self.cache.load_trades(cid)
        assert len(trades) == 0

    @patch("backtest.polymarket_trades.requests.get")
    def test_download_failure(self, mock_get):
        """API 失败时不崩溃"""
        mock_get.side_effect = Exception("timeout")
        cid = "fail_cid"
        self.cache.download_prices(cid, "yes_token_fail")
        assert not self.cache.has_trades(cid)

    def test_load_trades_missing(self):
        """加载不存在的文件返回空列表"""
        trades = self.cache.load_trades("missing_cid")
        assert trades == []

    @patch("backtest.polymarket_trades.time.sleep")
    @patch("backtest.polymarket_trades.requests.get")
    def test_ensure_trades(self, mock_get, mock_sleep):
        """批量下载缺失数据"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "history": [{"t": 1, "p": 0.5}]
        }
        mock_get.return_value = mock_resp

        markets = {
            ("2026-02-01", 85000.0): PolymarketMarketInfo(
                event_date="2026-02-01", strike=85000.0,
                condition_id="cid_a", yes_token_id="tok_a", no_token_id="n",
                question="test",
            ),
            ("2026-02-01", 86000.0): PolymarketMarketInfo(
                event_date="2026-02-01", strike=86000.0,
                condition_id="cid_b", yes_token_id="tok_b", no_token_id="n",
                question="test",
            ),
        }
        self.cache.ensure_trades(markets)
        assert self.cache.has_trades("cid_a")
        assert self.cache.has_trades("cid_b")


class TestPolymarketPriceLookup:
    def _make_cache_with_data(self, cid: str, trades_data: list):
        """创建含测试数据的缓存"""
        tmpdir = tempfile.mkdtemp()
        cache = PolymarketTradeCache(cache_dir=tmpdir)
        path = cache._file_path(cid)
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "price", "size"])
            for ts, price, size in trades_data:
                writer.writerow([ts, price, size])
        return cache

    def test_get_price_at_exact(self):
        """精确时间戳查询"""
        cache = self._make_cache_with_data("cid1", [
            (1000, 0.50, 10),
            (2000, 0.60, 5),
            (3000, 0.55, 8),
        ])
        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid1"])

        assert lookup.get_price_at("cid1", 2000) == 0.60

    def test_get_price_at_between(self):
        """介于两笔交易之间 → 返回前一笔（forward-fill）"""
        cache = self._make_cache_with_data("cid1", [
            (1000, 0.50, 10),
            (3000, 0.55, 8),
        ])
        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid1"])

        assert lookup.get_price_at("cid1", 2000) == 0.50

    def test_get_price_at_before_first(self):
        """早于所有交易 → None"""
        cache = self._make_cache_with_data("cid1", [
            (1000, 0.50, 10),
        ])
        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid1"])

        assert lookup.get_price_at("cid1", 500) is None

    def test_get_price_at_unknown_cid(self):
        """未知 condition_id → None"""
        cache = self._make_cache_with_data("cid1", [(1000, 0.50, 10)])
        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid1"])

        assert lookup.get_price_at("unknown", 1000) is None

    def test_get_market_prices_at(self):
        """综合查询：snap + bisect"""
        cache = self._make_cache_with_data("cid_85k", [
            (1000, 0.65, 10),
            (2000, 0.70, 5),
        ])
        # 手动加第二个合约的数据
        path2 = cache._file_path("cid_86k")
        with gzip.open(path2, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "price", "size"])
            writer.writerow([1000, 0.55, 10])
            writer.writerow([2000, 0.60, 5])

        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid_85k", "cid_86k"])

        markets = {
            ("2026-03-05", 85000.0): PolymarketMarketInfo(
                event_date="2026-03-05", strike=85000.0,
                condition_id="cid_85k", yes_token_id="y", no_token_id="n",
                question="test",
            ),
            ("2026-03-05", 86000.0): PolymarketMarketInfo(
                event_date="2026-03-05", strike=86000.0,
                condition_id="cid_86k", yes_token_id="y", no_token_id="n",
                question="test",
            ),
        }

        # 回测 K=85000 和 K=86000
        prices = lookup.get_market_prices_at(
            markets=markets,
            event_date="2026-03-05",
            timestamp_ms=1500,
            k_grid=[85000.0, 86000.0],
        )
        assert prices[85000.0] == 0.65  # ts=1000 的价格
        assert prices[86000.0] == 0.55

    def test_get_market_prices_at_snap(self):
        """K 网格 snap 到最近的 Polymarket K"""
        cache = self._make_cache_with_data("cid_85k", [
            (1000, 0.65, 10),
        ])
        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid_85k"])

        markets = {
            ("2026-03-05", 85000.0): PolymarketMarketInfo(
                event_date="2026-03-05", strike=85000.0,
                condition_id="cid_85k", yes_token_id="y", no_token_id="n",
                question="test",
            ),
        }

        # K=85100 应 snap 到 85000
        prices = lookup.get_market_prices_at(
            markets=markets,
            event_date="2026-03-05",
            timestamp_ms=1500,
            k_grid=[85100.0],
            max_snap_diff=250.0,
        )
        assert 85100.0 in prices
        assert prices[85100.0] == 0.65

    def test_get_market_prices_at_no_snap(self):
        """K 差距太大 → 不返回"""
        cache = self._make_cache_with_data("cid_85k", [
            (1000, 0.65, 10),
        ])
        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid_85k"])

        markets = {
            ("2026-03-05", 85000.0): PolymarketMarketInfo(
                event_date="2026-03-05", strike=85000.0,
                condition_id="cid_85k", yes_token_id="y", no_token_id="n",
                question="test",
            ),
        }

        prices = lookup.get_market_prices_at(
            markets=markets,
            event_date="2026-03-05",
            timestamp_ms=1500,
            k_grid=[90000.0],  # 远离 85000
            max_snap_diff=250.0,
        )
        assert 90000.0 not in prices

    def test_get_market_prices_wrong_date(self):
        """日期不匹配 → 空结果"""
        cache = self._make_cache_with_data("cid_85k", [
            (1000, 0.65, 10),
        ])
        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid_85k"])

        markets = {
            ("2026-03-05", 85000.0): PolymarketMarketInfo(
                event_date="2026-03-05", strike=85000.0,
                condition_id="cid_85k", yes_token_id="y", no_token_id="n",
                question="test",
            ),
        }

        prices = lookup.get_market_prices_at(
            markets=markets,
            event_date="2026-03-06",  # 不同日期
            timestamp_ms=1500,
            k_grid=[85000.0],
        )
        assert prices == {}


class TestNoLookaheadBias:
    """验证价格查询的防前瞻偏差"""

    def test_future_trades_not_visible(self):
        """未来交易不应返回"""
        tmpdir = tempfile.mkdtemp()
        cache = PolymarketTradeCache(cache_dir=tmpdir)
        path = cache._file_path("cid_bias")
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "price", "size"])
            # 过去的交易
            writer.writerow([1000, 0.50, 10])
            writer.writerow([2000, 0.55, 5])
            # 未来的交易
            writer.writerow([5000, 0.90, 20])

        lookup = PolymarketPriceLookup(cache)
        lookup.preload(["cid_bias"])

        # 查询 t=3000，应该只看到 t<=3000 的交易
        price = lookup.get_price_at("cid_bias", 3000)
        assert price == 0.55  # 不是 0.90
