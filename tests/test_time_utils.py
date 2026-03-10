"""time_utils 模块测试"""

import pytest
from pricing_core.time_utils import (
    et_noon_to_utc_ms,
    utc_ms_to_binance_kline_open,
    minutes_until_event,
    utc_ms_to_datetime,
)


class TestEtNoonToUtcMs:
    """ET 12:00 -> UTC 毫秒转换测试"""

    def test_winter_time_est(self):
        """冬令时 (EST): ET 12:00 = UTC 17:00"""
        # 2026-01-15 是冬令时
        ms = et_noon_to_utc_ms("2026-01-15")
        dt = utc_ms_to_datetime(ms)
        assert dt.hour == 17
        assert dt.minute == 0

    def test_summer_time_edt(self):
        """夏令时 (EDT): ET 12:00 = UTC 16:00"""
        # 2026-07-15 是夏令时
        ms = et_noon_to_utc_ms("2026-07-15")
        dt = utc_ms_to_datetime(ms)
        assert dt.hour == 16
        assert dt.minute == 0

    def test_march_dst_transition(self):
        """2026-03-05 应该是冬令时（DST 转换在 3 月第二个周日）"""
        # 2026 年 DST 从 3/8 开始，3/5 仍是 EST
        ms = et_noon_to_utc_ms("2026-03-05")
        dt = utc_ms_to_datetime(ms)
        assert dt.hour == 17  # EST: UTC-5

    def test_returns_milliseconds(self):
        """返回值应该是毫秒"""
        ms = et_noon_to_utc_ms("2026-03-05")
        assert ms > 1_000_000_000_000  # 大于 2001 年

    def test_consistent_same_date(self):
        """同一日期多次调用结果一致"""
        ms1 = et_noon_to_utc_ms("2026-03-05")
        ms2 = et_noon_to_utc_ms("2026-03-05")
        assert ms1 == ms2


class TestUtcMsToBinanceKlineOpen:
    """Binance kline openTime 对齐测试"""

    def test_already_aligned(self):
        """已对齐的时间戳不变"""
        # 整分钟: 2026-01-01 00:00:00 UTC
        aligned = 1767225600000
        assert utc_ms_to_binance_kline_open(aligned) == aligned

    def test_rounds_down(self):
        """非整分钟向下对齐"""
        base = 1767225600000  # 整分钟
        assert utc_ms_to_binance_kline_open(base + 30_000) == base
        assert utc_ms_to_binance_kline_open(base + 59_999) == base

    def test_next_minute(self):
        """下一分钟正确对齐"""
        base = 1767225600000
        assert utc_ms_to_binance_kline_open(base + 60_000) == base + 60_000


class TestMinutesUntilEvent:
    """距事件分钟数测试"""

    def test_positive_future(self):
        """事件在未来"""
        now = 1000000
        event = 1000000 + 60_000 * 30  # 30 分钟后
        assert minutes_until_event(now, event) == pytest.approx(30.0)

    def test_zero_at_event(self):
        """事件时刻"""
        ms = 1000000
        assert minutes_until_event(ms, ms) == 0.0

    def test_negative_past(self):
        """事件已过"""
        now = 2000000
        event = 1000000
        assert minutes_until_event(now, event) < 0
