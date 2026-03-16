"""
触碰合约市场发现解析测试
"""

import json
import os
import tempfile

import pytest

from touch.market_discovery import (
    parse_barrier_from_question,
    parse_direction_from_question,
    parse_month_from_slug,
    discover_touch_markets,
)


class TestParseBarrier:
    """障碍价格解析测试"""

    def test_standard_format(self):
        """标准格式 "$90,000 or above" """
        assert parse_barrier_from_question("$90,000 or above") == 90000.0

    def test_no_comma(self):
        """无逗号 "$90000 or above" """
        assert parse_barrier_from_question("$90000 or above") == 90000.0

    def test_large_number(self):
        """大数字 "$100,000 or above" """
        assert parse_barrier_from_question("$100,000 or above") == 100000.0

    def test_small_number(self):
        """较小数字 "$75,000 or below" """
        assert parse_barrier_from_question("$75,000 or below") == 75000.0

    def test_no_match(self):
        """无价格 → None"""
        assert parse_barrier_from_question("Some random text") is None

    def test_embedded_in_sentence(self):
        """嵌入句子中"""
        q = "Will Bitcoin reach $95,000 or above in March?"
        assert parse_barrier_from_question(q) == 95000.0


class TestParseDirection:
    """方向解析测试"""

    def test_above(self):
        assert parse_direction_from_question("$90,000 or above") == "up"

    def test_higher(self):
        assert parse_direction_from_question("$90,000 or higher") == "up"

    def test_below(self):
        assert parse_direction_from_question("$75,000 or below") == "down"

    def test_lower(self):
        assert parse_direction_from_question("$75,000 or lower") == "down"

    def test_case_insensitive(self):
        assert parse_direction_from_question("$90,000 or ABOVE") == "up"
        assert parse_direction_from_question("$75,000 or Below") == "down"

    def test_no_match(self):
        assert parse_direction_from_question("$90,000") is None


class TestParseMonthFromSlug:
    """slug 月份解析测试"""

    def test_standard(self):
        slug = "what-price-will-bitcoin-hit-in-march-2026"
        assert parse_month_from_slug(slug) == "2026-03"

    def test_different_months(self):
        assert parse_month_from_slug("hit-in-january-2026") == "2026-01"
        assert parse_month_from_slug("hit-in-december-2025") == "2025-12"

    def test_case_insensitive(self):
        slug = "what-price-will-bitcoin-hit-in-March-2026"
        assert parse_month_from_slug(slug) == "2026-03"

    def test_for_prefix(self):
        slug = "bitcoin-touch-for-april-2026"
        assert parse_month_from_slug(slug) == "2026-04"

    def test_no_match(self):
        assert parse_month_from_slug("random-slug") is None


class TestDiscoverTouchMarkets:
    """市场发现测试（使用缓存）"""

    def test_load_from_cache(self):
        """从缓存文件加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "touch_cache.json")

            # 创建缓存
            cache_data = {
                "90000.0": {
                    "month": "2026-03",
                    "barrier": 90000.0,
                    "direction": "up",
                    "condition_id": "cid_90k",
                    "yes_token_id": "yes_90k",
                    "no_token_id": "no_90k",
                    "question": "$90,000 or above",
                },
                "75000.0": {
                    "month": "2026-03",
                    "barrier": 75000.0,
                    "direction": "down",
                    "condition_id": "cid_75k",
                    "yes_token_id": "yes_75k",
                    "no_token_id": "no_75k",
                    "question": "$75,000 or below",
                },
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)

            # 加载
            result = discover_touch_markets(
                slug="what-price-will-bitcoin-hit-in-march-2026",
                cache_path=cache_path,
            )

            assert len(result) == 2
            assert 90000.0 in result
            assert 75000.0 in result
            assert result[90000.0].direction == "up"
            assert result[75000.0].direction == "down"
            assert result[90000.0].condition_id == "cid_90k"
