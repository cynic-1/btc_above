"""deribit_data 模块测试（HTTP mock）"""

import pytest
from unittest.mock import patch, MagicMock

from pricing_core.deribit_data import DeribitClient


class TestDeribitGetIndexPrice:

    @patch("pricing_core.deribit_data.requests.Session.get")
    def test_returns_index_price(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": {"index_price": 91234.56}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = DeribitClient()
        price = client.get_index_price()
        assert price == pytest.approx(91234.56)


class TestDeribitGetPerpMark:

    @patch("pricing_core.deribit_data.requests.Session.get")
    def test_returns_perp_info(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": {
                "mark_price": 91300.0,
                "index_price": 91200.0,
                "current_funding": 0.0001,
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = DeribitClient()
        info = client.get_perp_mark()
        assert info.mark_price == 91300.0
        assert info.index_price == 91200.0
        assert info.basis == pytest.approx(100.0)
        assert info.funding_rate == pytest.approx(0.0001)


class TestDeribitApiError:

    @patch("pricing_core.deribit_data.requests.Session.get")
    def test_raises_on_missing_result(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": {"code": -1, "message": "bad"}}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        client = DeribitClient()
        with pytest.raises(ValueError, match="异常响应"):
            client.get_index_price()
