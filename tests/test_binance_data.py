"""binance_data 模块测试（HTTP mock）"""

import json
import pytest
from unittest.mock import patch, MagicMock

from pricing_core.binance_data import BinanceClient, Kline


def _make_kline_row(open_time, close_price, open_price=90000.0):
    """构造 Binance kline API 响应行"""
    return [
        open_time,           # openTime
        str(open_price),     # open
        str(close_price + 10),  # high
        str(close_price - 10),  # low
        str(close_price),    # close
        "100.0",             # volume
        open_time + 59999,   # closeTime
        "9000000",           # quoteAssetVolume
        100,                 # numberOfTrades
        "50.0",              # takerBuyBaseAssetVolume
        "4500000",           # takerBuyQuoteAssetVolume
        "0",                 # ignore
    ]


class TestBinanceGetKlines:

    @patch("pricing_core.binance_data.requests.Session.get")
    def test_get_klines_parses_response(self, mock_get):
        """正确解析 kline 数据"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            _make_kline_row(1000000, 91000.0),
            _make_kline_row(1060000, 91500.0),
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = BinanceClient()
        klines = client.get_klines(limit=2)

        assert len(klines) == 2
        assert klines[0].open_time == 1000000
        assert klines[0].close == 91000.0
        assert klines[1].close == 91500.0

    @patch("pricing_core.binance_data.requests.Session.get")
    def test_get_klines_empty(self, mock_get):
        """空响应返回空列表"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = BinanceClient()
        klines = client.get_klines()
        assert klines == []


class TestBinanceGetCloseAtEvent:

    @patch("pricing_core.binance_data.requests.Session.get")
    def test_returns_close_price(self, mock_get):
        """返回正确的 close 价格"""
        event_ms = 1700000000000
        mock_resp = MagicMock()
        mock_resp.json.return_value = [_make_kline_row(event_ms, 95000.0)]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = BinanceClient()
        close = client.get_close_at_event(event_ms)
        assert close == 95000.0

    @patch("pricing_core.binance_data.requests.Session.get")
    def test_raises_on_empty(self, mock_get):
        """无数据时抛出异常"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = BinanceClient()
        with pytest.raises(ValueError, match="未找到"):
            client.get_close_at_event(1700000000000)

    @patch("pricing_core.binance_data.requests.Session.get")
    def test_raises_on_mismatch(self, mock_get):
        """openTime 不匹配时抛出异常"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [_make_kline_row(9999999, 95000.0)]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = BinanceClient()
        with pytest.raises(ValueError, match="不匹配"):
            client.get_close_at_event(1700000000000)


class TestBinanceGetCurrentPrice:

    @patch("pricing_core.binance_data.requests.Session.get")
    def test_returns_price(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"symbol": "BTCUSDT", "price": "92345.67"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = BinanceClient()
        price = client.get_current_price()
        assert price == pytest.approx(92345.67)
