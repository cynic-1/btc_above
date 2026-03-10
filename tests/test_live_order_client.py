"""Polymarket 订单客户端单元测试"""

import sys
import pytest
from unittest.mock import MagicMock, patch, call

from live.config import LiveTradingConfig


# Mock py-clob-client 模块，避免依赖实际安装
@pytest.fixture(autouse=True)
def mock_clob_client(monkeypatch):
    """Mock ClobClient 及其依赖"""
    mock_module = MagicMock()

    # 设置需要的类和常量
    mock_module.ClobClient = MagicMock()
    mock_module.ApiCreds = MagicMock()
    mock_module.OrderArgs = MagicMock()
    mock_module.OrderType = MagicMock()
    mock_module.OrderType.GTC = "GTC"
    mock_module.OrderType.FOK = "FOK"
    mock_module.PartialCreateOrderOptions = MagicMock()

    monkeypatch.setitem(sys.modules, "py_clob_client", mock_module)

    return mock_module


@pytest.fixture
def config():
    return LiveTradingConfig(
        polymarket_host="https://test-clob.polymarket.com",
        polymarket_private_key="0xtest_private_key",
        polymarket_chain_id=137,
    )


def _make_mock_instance():
    """创建标准 mock ClobClient 实例"""
    mock_instance = MagicMock()
    mock_instance.get_address.return_value = "0xtest"
    mock_instance.create_or_derive_api_creds.return_value = MagicMock(
        api_key="derived-key",
        api_secret="derived-secret",
        api_passphrase="derived-pass",
    )
    return mock_instance


class TestOrderClient:
    """订单客户端测试"""

    @patch("live.order_client.ClobClient")
    def test_init_derives_creds(self, mock_clob_cls, config):
        """初始化应自动派生 API 凭证"""
        from live.order_client import PolymarketOrderClient

        mock_instance = _make_mock_instance()
        mock_clob_cls.return_value = mock_instance

        client = PolymarketOrderClient(config)

        # 应先 L1 创建，再派生凭证，再 set_api_creds
        mock_clob_cls.assert_called_once()
        mock_instance.create_or_derive_api_creds.assert_called_once()
        mock_instance.set_api_creds.assert_called_once()

    @patch("live.order_client.ClobClient")
    def test_init_fails_on_no_creds(self, mock_clob_cls, config):
        """派生凭证失败应抛异常"""
        from live.order_client import PolymarketOrderClient

        mock_instance = _make_mock_instance()
        mock_instance.create_or_derive_api_creds.return_value = None
        mock_clob_cls.return_value = mock_instance

        with pytest.raises(RuntimeError, match="无法派生"):
            PolymarketOrderClient(config)

    @patch("live.order_client.ClobClient")
    def test_place_order_success(self, mock_clob_cls, config):
        """成功下单"""
        from live.order_client import PolymarketOrderClient

        mock_instance = _make_mock_instance()
        mock_instance.create_order.return_value = {"signed": True}
        mock_instance.post_order.return_value = {
            "orderID": "order-123",
            "status": "live",
        }
        mock_clob_cls.return_value = mock_instance

        client = PolymarketOrderClient(config)
        result = client.place_order(
            token_id="token_abc",
            side="BUY",
            price=0.55,
            size=200,
            order_type="GTC",
        )

        assert result["orderID"] == "order-123"
        mock_instance.create_order.assert_called_once()
        mock_instance.post_order.assert_called_once()

    @patch("live.order_client.ClobClient")
    def test_place_order_retry(self, mock_clob_cls, config):
        """下单失败应重试"""
        from live.order_client import PolymarketOrderClient

        mock_instance = _make_mock_instance()
        # 前两次失败，第三次成功
        mock_instance.create_order.side_effect = [
            Exception("网络错误"),
            Exception("超时"),
            {"signed": True},
        ]
        mock_instance.post_order.return_value = {"orderID": "order-retry"}
        mock_clob_cls.return_value = mock_instance

        client = PolymarketOrderClient(config)
        result = client.place_order(
            token_id="token_abc",
            side="BUY",
            price=0.55,
            size=200,
        )

        assert result["orderID"] == "order-retry"
        assert mock_instance.create_order.call_count == 3

    @patch("live.order_client.ClobClient")
    def test_place_order_all_retries_fail(self, mock_clob_cls, config):
        """所有重试都失败应返回错误"""
        from live.order_client import PolymarketOrderClient

        mock_instance = _make_mock_instance()
        mock_instance.create_order.side_effect = Exception("持续错误")
        mock_clob_cls.return_value = mock_instance

        client = PolymarketOrderClient(config)
        result = client.place_order(
            token_id="token_abc",
            side="BUY",
            price=0.55,
            size=200,
        )

        assert "error" in result
        assert result["success"] is False

    @patch("live.order_client.ClobClient")
    def test_cancel_order(self, mock_clob_cls, config):
        """撤单测试"""
        from live.order_client import PolymarketOrderClient

        mock_instance = _make_mock_instance()
        mock_instance.cancel.return_value = {"canceled": True}
        mock_clob_cls.return_value = mock_instance

        client = PolymarketOrderClient(config)
        result = client.cancel_order("order-123")

        assert result["canceled"] is True
        mock_instance.cancel.assert_called_once_with("order-123")

    @patch("live.order_client.ClobClient")
    def test_cancel_all(self, mock_clob_cls, config):
        """撤销所有订单测试"""
        from live.order_client import PolymarketOrderClient

        mock_instance = _make_mock_instance()
        mock_instance.cancel_all.return_value = {"canceled": True}
        mock_clob_cls.return_value = mock_instance

        client = PolymarketOrderClient(config)
        result = client.cancel_all()

        assert result["canceled"] is True
