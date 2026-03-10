"""Polymarket WebSocket 客户端单元测试"""

import json
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from live.config import LiveTradingConfig
from live.models import OrderBookLevel, OrderBookState
from live.polymarket_ws import PolymarketOrderbookWS


@pytest.fixture
def config():
    return LiveTradingConfig(
        polymarket_ws_url="wss://test.example.com/ws/market",
    )


@pytest.fixture
def ws_client(config):
    return PolymarketOrderbookWS(config)


class TestOrderbookParsing:
    """Orderbook 解析测试"""

    def test_parse_book(self, ws_client):
        """应正确解析 book snapshot"""
        data = {
            "asset_id": "token_123",
            "bids": [
                {"price": "0.55", "size": "100"},
                {"price": "0.50", "size": "200"},
            ],
            "asks": [
                {"price": "0.60", "size": "150"},
                {"price": "0.65", "size": "50"},
            ],
        }

        ob = ws_client._parse_book(data)

        assert ob.asset_id == "token_123"
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2
        # bids 降序
        assert ob.bids[0].price == 0.55
        assert ob.bids[1].price == 0.50
        # asks 升序
        assert ob.asks[0].price == 0.60
        assert ob.asks[1].price == 0.65
        assert ob.best_bid == 0.55
        assert ob.best_ask == 0.60

    def test_parse_empty_book(self, ws_client):
        """空 orderbook"""
        data = {"asset_id": "token_empty", "bids": [], "asks": []}
        ob = ws_client._parse_book(data)
        assert ob.best_bid == 0.0
        assert ob.best_ask == 0.0

    def test_filter_zero_size(self, ws_client):
        """应过滤 size=0 的层"""
        data = {
            "asset_id": "token_filter",
            "bids": [
                {"price": "0.50", "size": "100"},
                {"price": "0.45", "size": "0"},
            ],
            "asks": [{"price": "0.60", "size": "0"}],
        }
        ob = ws_client._parse_book(data)
        assert len(ob.bids) == 1
        assert len(ob.asks) == 0


class TestPriceChange:
    """增量更新测试（官方文档 price_changes 格式）"""

    def test_apply_new_bid(self, ws_client):
        """添加新 bid（通过 _apply_single_change）"""
        ob = OrderBookState(
            asset_id="token_1",
            bids=[OrderBookLevel(price=0.50, size=100)],
            asks=[OrderBookLevel(price=0.60, size=100)],
            best_bid=0.50,
            best_ask=0.60,
        )
        change = {"asset_id": "token_1", "side": "BUY", "price": "0.55", "size": "200"}
        ws_client._apply_single_change(ob, change)

        assert len(ob.bids) == 2
        assert ob.best_bid == 0.55
        assert ob.bids[0].price == 0.55

    def test_remove_bid(self, ws_client):
        """size=0 应移除该层"""
        ob = OrderBookState(
            asset_id="token_1",
            bids=[
                OrderBookLevel(price=0.55, size=200),
                OrderBookLevel(price=0.50, size=100),
            ],
            asks=[],
            best_bid=0.55,
            best_ask=0.0,
        )
        change = {"asset_id": "token_1", "side": "BUY", "price": "0.55", "size": "0"}
        ws_client._apply_single_change(ob, change)

        assert len(ob.bids) == 1
        assert ob.best_bid == 0.50

    def test_update_ask(self, ws_client):
        """更新已有 ask"""
        ob = OrderBookState(
            asset_id="token_1",
            bids=[],
            asks=[OrderBookLevel(price=0.60, size=100)],
            best_bid=0.0,
            best_ask=0.60,
        )
        change = {"asset_id": "token_1", "side": "SELL", "price": "0.60", "size": "300"}
        ws_client._apply_single_change(ob, change)

        assert len(ob.asks) == 1
        assert ob.asks[0].size == 300

    def test_handle_price_change_message(self, ws_client):
        """完整 price_change 消息处理（官方文档格式）"""
        # 先初始化 orderbook
        ws_client.orderbooks["token_1"] = OrderBookState(
            asset_id="token_1",
            bids=[OrderBookLevel(price=0.50, size=100)],
            asks=[OrderBookLevel(price=0.60, size=100)],
            best_bid=0.50,
            best_ask=0.60,
        )
        # 模拟 price_change 消息
        data = {
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "token_1", "side": "BUY", "price": "0.55", "size": "200"},
            ],
        }
        ws_client._handle_price_change(data)

        ob = ws_client.get_orderbook("token_1")
        assert ob.best_bid == 0.55
        assert len(ob.bids) == 2

    def test_handle_price_change_multi_asset(self, ws_client):
        """price_change 消息中包含多个 asset_id 的更新"""
        ws_client.orderbooks["token_a"] = OrderBookState(
            asset_id="token_a",
            bids=[OrderBookLevel(price=0.40, size=50)],
            asks=[OrderBookLevel(price=0.60, size=50)],
            best_bid=0.40,
            best_ask=0.60,
        )
        ws_client.orderbooks["token_b"] = OrderBookState(
            asset_id="token_b",
            bids=[OrderBookLevel(price=0.30, size=80)],
            asks=[OrderBookLevel(price=0.70, size=80)],
            best_bid=0.30,
            best_ask=0.70,
        )
        data = {
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "token_a", "side": "BUY", "price": "0.45", "size": "100"},
                {"asset_id": "token_b", "side": "SELL", "price": "0.65", "size": "120"},
            ],
        }
        ws_client._handle_price_change(data)

        assert ws_client.get_orderbook("token_a").best_bid == 0.45
        assert ws_client.get_orderbook("token_b").best_ask == 0.65


class TestCallback:
    """回调测试"""

    def test_callback_on_book(self, ws_client):
        """book snapshot 应触发回调"""
        received = []
        ws_client.add_callback(lambda aid, ob: received.append((aid, ob)))

        data = {
            "asset_id": "token_cb",
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.60", "size": "100"}],
        }
        ws_client._handle_book(data)

        assert len(received) == 1
        assert received[0][0] == "token_cb"


class TestArrayMessage:
    """数组消息测试"""

    def test_array_of_books(self, ws_client):
        """WS 返回 JSON 数组时应逐条处理"""
        import json
        msg = json.dumps([
            {
                "type": "book",
                "asset_id": "token_a",
                "bids": [{"price": "0.40", "size": "50"}],
                "asks": [{"price": "0.60", "size": "50"}],
            },
            {
                "type": "book",
                "asset_id": "token_b",
                "bids": [{"price": "0.30", "size": "80"}],
                "asks": [{"price": "0.70", "size": "80"}],
            },
        ])
        ws_client._on_message(None, msg)

        assert ws_client.get_orderbook("token_a") is not None
        assert ws_client.get_orderbook("token_b") is not None
        assert ws_client.get_orderbook("token_a").best_bid == 0.40
        assert ws_client.get_orderbook("token_b").best_ask == 0.70

    def test_non_dict_in_array_ignored(self, ws_client):
        """数组中非 dict 元素应跳过"""
        import json
        msg = json.dumps(["hello", 123, None])
        ws_client._on_message(None, msg)  # 不应抛异常


class TestPongHandling:
    """PONG 消息测试"""

    def test_pong_ignored(self, ws_client):
        """纯文本 PONG 应被静默忽略"""
        ws_client._on_message(None, "PONG")  # 不应抛异常


class TestGetOrderbook:
    """线程安全获取测试"""

    def test_get_nonexistent(self, ws_client):
        """不存在的 asset_id 返回 None"""
        assert ws_client.get_orderbook("unknown") is None

    def test_get_returns_copy(self, ws_client):
        """应返回副本而非引用"""
        ws_client.orderbooks["token_1"] = OrderBookState(
            asset_id="token_1",
            bids=[OrderBookLevel(price=0.50, size=100)],
            asks=[OrderBookLevel(price=0.60, size=100)],
            best_bid=0.50,
            best_ask=0.60,
        )

        ob = ws_client.get_orderbook("token_1")
        assert ob is not None
        ob.bids.clear()

        # 原始数据不应被修改
        orig = ws_client.orderbooks["token_1"]
        assert len(orig.bids) == 1
