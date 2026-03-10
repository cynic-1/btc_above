"""
Polymarket 市场发现模块测试
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from backtest.polymarket_discovery import (
    PolymarketMarketInfo,
    _cache_key,
    _load_discovery_cache,
    _save_discovery_cache,
    _search_gamma_api,
    discover_markets_for_range,
    parse_date_from_question,
    parse_strike_from_question,
    snap_strike,
)


class TestParseStrikeFromQuestion:
    def test_standard_format(self):
        q = "Will Bitcoin be above $85,000 on March 5?"
        assert parse_strike_from_question(q) == 85000.0

    def test_no_comma(self):
        q = "Will Bitcoin be above $85000 on March 5?"
        assert parse_strike_from_question(q) == 85000.0

    def test_large_number(self):
        q = "Will Bitcoin be above $100,000 on February 14?"
        assert parse_strike_from_question(q) == 100000.0

    def test_no_match(self):
        assert parse_strike_from_question("No price here") is None

    def test_smaller_number(self):
        q = "Will Bitcoin be above $500 on January 1?"
        assert parse_strike_from_question(q) == 500.0


class TestParseDateFromQuestion:
    def test_standard_format(self):
        q = "Will Bitcoin be above $85,000 on March 5?"
        assert parse_date_from_question(q) == "2026-03-05"

    def test_different_month(self):
        q = "Will Bitcoin be above $90,000 on February 14?"
        assert parse_date_from_question(q) == "2026-02-14"

    def test_single_digit_day(self):
        q = "Will Bitcoin be above $80,000 on January 1?"
        assert parse_date_from_question(q) == "2026-01-01"

    def test_custom_year(self):
        q = "Will Bitcoin be above $80,000 on January 1?"
        assert parse_date_from_question(q, year=2025) == "2025-01-01"

    def test_no_match(self):
        assert parse_date_from_question("No date here") is None

    def test_case_insensitive(self):
        q = "Will Bitcoin be above $85,000 on MARCH 5?"
        assert parse_date_from_question(q) == "2026-03-05"


class TestSnapStrike:
    def test_exact_match(self):
        assert snap_strike(85000.0, [84000.0, 85000.0, 86000.0]) == 85000.0

    def test_snap_to_nearest(self):
        assert snap_strike(85100.0, [84000.0, 85000.0, 86000.0]) == 85000.0

    def test_too_far(self):
        assert snap_strike(85000.0, [80000.0, 90000.0], max_diff=250.0) is None

    def test_within_max_diff(self):
        assert snap_strike(85100.0, [85000.0], max_diff=250.0) == 85000.0

    def test_empty_list(self):
        assert snap_strike(85000.0, []) is None


class TestDiscoveryCache:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            info = PolymarketMarketInfo(
                event_date="2026-03-05",
                strike=85000.0,
                condition_id="abc123",
                yes_token_id="tok_yes",
                no_token_id="tok_no",
                question="Will Bitcoin be above $85,000 on March 5?",
            )
            cache = {"2026-03-05|85000": info}
            _save_discovery_cache(cache, cache_path)

            loaded = _load_discovery_cache(cache_path)
            assert "2026-03-05|85000" in loaded
            loaded_info = loaded["2026-03-05|85000"]
            assert loaded_info.event_date == "2026-03-05"
            assert loaded_info.strike == 85000.0
            assert loaded_info.condition_id == "abc123"

    def test_load_missing_file(self):
        result = _load_discovery_cache("/nonexistent/path.json")
        assert result == {}


class TestCacheKey:
    def test_format(self):
        assert _cache_key("2026-03-05", 85000.0) == "2026-03-05|85000"
        assert _cache_key("2026-01-01", 100000.0) == "2026-01-01|100000"


class TestSearchGammaApi:
    @patch("backtest.polymarket_discovery.requests.get")
    def test_parse_markets(self, mock_get):
        """从 Gamma API 响应解析市场"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "events": [{
                "markets": [{
                    "question": "Will Bitcoin be above $85,000 on March 5?",
                    "conditionId": "cid_123",
                    "clobTokenIds": json.dumps(["yes_tok", "no_tok"]),
                }]
            }]
        }
        mock_get.return_value = mock_resp

        results = _search_gamma_api("Bitcoin above March 5", "https://gamma-api.polymarket.com")
        assert len(results) >= 1
        info = results[0]
        assert info.strike == 85000.0
        assert info.event_date == "2026-03-05"
        assert info.condition_id == "cid_123"
        assert info.yes_token_id == "yes_tok"

    @patch("backtest.polymarket_discovery.requests.get")
    def test_skip_non_bitcoin(self, mock_get):
        """忽略非 Bitcoin 市场"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [{
                "markets": [{
                    "question": "Will Ethereum be above $5,000 on March 5?",
                    "conditionId": "cid_eth",
                    "clobTokenIds": json.dumps(["a", "b"]),
                }]
            }]
        }
        mock_get.return_value = mock_resp

        results = _search_gamma_api("test", "https://example.com")
        assert len(results) == 0

    @patch("backtest.polymarket_discovery.requests.get")
    def test_api_failure(self, mock_get):
        """API 调用失败时返回空列表"""
        mock_get.side_effect = Exception("network error")
        results = _search_gamma_api("test", "https://example.com")
        assert results == []


class TestDiscoverMarketsForRange:
    @patch("backtest.polymarket_discovery._search_gamma_api")
    def test_uses_cache(self, mock_search):
        """已缓存的日期不再搜索"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            # 预填充缓存
            info = PolymarketMarketInfo(
                event_date="2026-02-01",
                strike=85000.0,
                condition_id="cid_1",
                yes_token_id="y1",
                no_token_id="n1",
                question="test",
            )
            _save_discovery_cache({"2026-02-01|85000": info}, cache_path)

            result = discover_markets_for_range(
                "2026-02-01", "2026-02-02",
                cache_path=cache_path,
            )

            # 缓存命中，不应调用 API
            mock_search.assert_not_called()
            assert ("2026-02-01", 85000.0) in result

    @patch("backtest.polymarket_discovery.time.sleep")
    @patch("backtest.polymarket_discovery._search_gamma_api")
    def test_searches_missing_dates(self, mock_search, mock_sleep):
        """缺失日期调用 API 搜索"""
        info = PolymarketMarketInfo(
            event_date="2026-02-01",
            strike=90000.0,
            condition_id="cid_new",
            yes_token_id="y",
            no_token_id="n",
            question="test",
        )
        mock_search.return_value = [info]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            result = discover_markets_for_range(
                "2026-02-01", "2026-02-02",
                cache_path=cache_path,
            )

            assert mock_search.called
            assert ("2026-02-01", 90000.0) in result
