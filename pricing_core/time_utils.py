"""
时间工具模块
处理 ET→UTC 转换（含 DST）和 Binance kline 时间对齐
"""

import logging
from datetime import datetime

import pytz

logger = logging.getLogger(__name__)

# 美国东部时区（自动处理 DST）
ET = pytz.timezone("US/Eastern")
UTC = pytz.utc


def et_noon_to_utc_ms(date_str: str) -> int:
    """
    将日期字符串转换为该日 ET 12:00:00 对应的 UTC 毫秒时间戳

    自动处理夏令时：
    - EST (冬令时): ET 12:00 = UTC 17:00
    - EDT (夏令时): ET 12:00 = UTC 16:00

    Args:
        date_str: 日期字符串，格式 "YYYY-MM-DD"

    Returns:
        UTC 毫秒时间戳
    """
    # 解析日期并定位到 ET 12:00:00
    naive_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=12, minute=0, second=0, microsecond=0
    )
    et_dt = ET.localize(naive_dt)
    utc_dt = et_dt.astimezone(UTC)

    utc_ms = int(utc_dt.timestamp() * 1000)
    logger.debug(f"et_noon_to_utc_ms({date_str}): ET={et_dt}, UTC={utc_dt}, ms={utc_ms}")
    return utc_ms


def utc_ms_to_binance_kline_open(utc_ms: int, interval_ms: int = 60_000) -> int:
    """
    将 UTC 毫秒时间戳对齐到 Binance kline 的 openTime

    Binance 1m kline 的 openTime 是整分钟的毫秒时间戳。
    例如 ET 12:00:00 对应的 kline openTime 就是该时刻本身（如果已经是整分钟）。

    Args:
        utc_ms: UTC 毫秒时间戳
        interval_ms: K线间隔毫秒数（默认 60000 = 1分钟）

    Returns:
        对齐后的 openTime（UTC 毫秒）
    """
    return (utc_ms // interval_ms) * interval_ms


def utc_ms_to_datetime(utc_ms: int) -> datetime:
    """UTC 毫秒转 datetime 对象"""
    return datetime.fromtimestamp(utc_ms / 1000, tz=UTC)


def minutes_until_event(now_utc_ms: int, event_utc_ms: int) -> float:
    """计算距离事件的分钟数"""
    return (event_utc_ms - now_utc_ms) / 60_000
