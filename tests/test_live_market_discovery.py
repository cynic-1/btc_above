"""市场发现模块单元测试"""

import json
import pytest
from unittest.mock import patch, MagicMock

from live.market_discovery import (
    discover_today_markets,
    parse_date_from_question,
    parse_strike_from_question,
)
from live.models import MarketInfo


class TestParseStrike:
    """行权价解析测试"""

    def test_normal_strike(self):
        q = "Will Bitcoin be above $85,000 on March 5?"
        assert parse_strike_from_question(q) == 85000.0

    def test_no_comma(self):
        q = "Will Bitcoin be above $85000 on March 5?"
        assert parse_strike_from_question(q) == 85000.0

    def test_large_strike(self):
        q = "Will Bitcoin be above $100,500 on March 10?"
        assert parse_strike_from_question(q) == 100500.0

    def test_no_match(self):
        q = "Will Bitcoin go up?"
        assert parse_strike_from_question(q) is None


class TestParseDate:
    """日期解析测试"""

    def test_normal_date(self):
        q = "Will Bitcoin be above $85,000 on March 5?"
        assert parse_date_from_question(q, year=2026) == "2026-03-05"

    def test_december(self):
        q = "Will Bitcoin be above $90,000 on December 25?"
        assert parse_date_from_question(q, year=2026) == "2026-12-25"

    def test_case_insensitive(self):
        q = "Will Bitcoin be above $85,000 on MARCH 5?"
        assert parse_date_from_question(q, year=2026) == "2026-03-05"

    def test_no_match(self):
        q = "Will Bitcoin be above $85,000?"
        assert parse_date_from_question(q) is None


class TestDiscoverTodayMarkets:
    """市场发现测试"""

    @patch("live.market_discovery._fetch_tick_size_and_neg_risk")
    @patch("live.market_discovery.requests.get")
    def test_discover_markets(self, mock_get, mock_tick):
        """应正确解析 Gamma API 返回的市场"""
        mock_tick.return_value = ("0.01", False)

        api_response = {
            "events": [
                {
                    "markets": [
                        {
                            "question": "Will Bitcoin be above $85,000 on March 9?",
                            "conditionId": "cond_abc123",
                            "clobTokenIds": json.dumps(["yes_token", "no_token"]),
                        },
                        {
                            "question": "Will Bitcoin be above $90,000 on March 9?",
                            "conditionId": "cond_xyz456",
                            "clobTokenIds": ["yes_token2", "no_token2"],
                        },
                    ]
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        markets = discover_today_markets(
            event_date="2026-03-09",
            gamma_api="https://gamma-api.test.com",
            clob_host="https://clob.test.com",
            year=2026,
        )

        assert len(markets) == 2
        assert markets[0].strike == 85000.0
        assert markets[0].condition_id == "cond_abc123"
        assert markets[0].yes_token_id == "yes_token"
        assert markets[0].no_token_id == "no_token"
        assert markets[0].event_date == "2026-03-09"
        assert markets[1].strike == 90000.0

    @patch("live.market_discovery._fetch_tick_size_and_neg_risk")
    @patch("live.market_discovery.requests.get")
    def test_filters_wrong_date(self, mock_get, mock_tick):
        """应过滤掉非目标日期的市场"""
        mock_tick.return_value = ("0.01", False)

        api_response = {
            "events": [
                {
                    "markets": [
                        {
                            "question": "Will Bitcoin be above $85,000 on March 10?",
                            "conditionId": "cond_wrong",
                            "clobTokenIds": ["yes", "no"],
                        },
                    ]
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        markets = discover_today_markets(
            event_date="2026-03-09",
            gamma_api="https://gamma-api.test.com",
        )

        assert len(markets) == 0

    @patch("live.market_discovery.requests.get")
    def test_handles_api_error(self, mock_get):
        """API 错误时应返回空列表"""
        mock_get.side_effect = Exception("连接超时")

        markets = discover_today_markets(
            event_date="2026-03-09",
            gamma_api="https://gamma-api.test.com",
        )

        assert markets == []
