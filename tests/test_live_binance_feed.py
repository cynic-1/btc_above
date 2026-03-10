"""Binance K线流单元测试"""

import json
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from pricing_core.binance_data import Kline
from live.config import LiveTradingConfig
from live.binance_feed import BinanceKlineFeed


@pytest.fixture
def config():
    return LiveTradingConfig(
        binance_ws_url="wss://test.binance.com/ws/btcusdt@kline_1m",
        binance_rest_url="https://test.binance.com",
    )


@pytest.fixture
def feed(config):
    return BinanceKlineFeed(config)


def _make_kline(open_time: int, close: float) -> Kline:
    return Kline(
        open_time=open_time,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=100.0,
        close_time=open_time + 59999,
    )


class TestUpdateKline:
    """K线更新测试"""

    def test_append_new_kline(self, feed):
        """新 K线应追加到缓冲区"""
        feed._klines = [_make_kline(1000 * 60000, 85000.0)]

        k_data = {
            "t": 1001 * 60000,
            "T": 1001 * 60000 + 59999,
            "o": "85010.0",
            "h": "85020.0",
            "l": "84990.0",
            "c": "85015.0",
            "v": "50.0",
            "x": True,
        }
        feed._update_kline(k_data)

        assert len(feed._klines) == 2
        assert feed._current_price == 85015.0

    def test_update_existing_kline(self, feed):
        """同一 openTime 应更新而非追加"""
        feed._klines = [_make_kline(1000 * 60000, 85000.0)]

        k_data = {
            "t": 1000 * 60000,
            "T": 1000 * 60000 + 59999,
            "o": "85000.0",
            "h": "85050.0",
            "l": "84990.0",
            "c": "85040.0",
            "v": "75.0",
            "x": False,
        }
        feed._update_kline(k_data)

        assert len(feed._klines) == 1
        assert feed._klines[0].close == 85040.0
        assert feed._current_price == 85040.0

    def test_24h_window_trim(self, feed):
        """应保持 24h 窗口，移除过旧数据"""
        # 填充 1445 条 K线（超过 24h = 1440 条）
        base_time = 0
        for i in range(1445):
            feed._klines.append(_make_kline(base_time + i * 60000, 85000.0 + i))

        # 添加新 K线
        new_time = base_time + 1445 * 60000
        k_data = {
            "t": new_time,
            "T": new_time + 59999,
            "o": "86000.0",
            "h": "86010.0",
            "l": "85990.0",
            "c": "86005.0",
            "v": "100.0",
            "x": True,
        }
        feed._update_kline(k_data)

        # 应该裁剪到 ~1440 条
        assert len(feed._klines) <= 1441


class TestGetData:
    """数据获取测试"""

    def test_get_current_price(self, feed):
        """获取当前价格"""
        feed._current_price = 85123.45
        assert feed.get_current_price() == 85123.45

    def test_get_klines_returns_copy(self, feed):
        """get_klines 应返回副本"""
        feed._klines = [_make_kline(1000 * 60000, 85000.0)]
        klines = feed.get_klines()
        klines.clear()
        assert len(feed._klines) == 1

    def test_get_close_prices(self, feed):
        """获取 close 价格数组"""
        feed._klines = [
            _make_kline(1000 * 60000, 85000.0),
            _make_kline(1001 * 60000, 85100.0),
            _make_kline(1002 * 60000, 85050.0),
        ]
        prices = feed.get_close_prices()
        assert isinstance(prices, np.ndarray)
        assert len(prices) == 3
        assert prices[0] == 85000.0
        assert prices[2] == 85050.0


class TestWsMessage:
    """WebSocket 消息处理测试"""

    def test_parse_kline_message(self, feed):
        """应正确解析 WS kline 消息"""
        msg = json.dumps({
            "e": "kline",
            "k": {
                "t": 1000 * 60000,
                "T": 1000 * 60000 + 59999,
                "o": "85000.0",
                "h": "85050.0",
                "l": "84990.0",
                "c": "85030.0",
                "v": "123.456",
                "x": True,
            },
        })

        feed._on_kline_message(None, msg)

        assert len(feed._klines) == 1
        assert feed._current_price == 85030.0

    def test_ignore_non_kline(self, feed):
        """非 kline 消息应忽略"""
        msg = json.dumps({"e": "trade", "p": "85000.0"})
        feed._on_kline_message(None, msg)
        assert len(feed._klines) == 0

    def test_ignore_invalid_json(self, feed):
        """无效 JSON 应忽略"""
        feed._on_kline_message(None, "not json")
        assert len(feed._klines) == 0
