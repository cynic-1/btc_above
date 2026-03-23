"""
K线数据缓存
按天下载并缓存 Binance 1m K线到 gzip CSV 文件
"""

import gzip
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from pricing_core.binance_data import BinanceClient, Kline
from pricing_core.time_utils import UTC

logger = logging.getLogger(__name__)

# 一天的毫秒数
_DAY_MS = 86_400_000
# 一分钟的毫秒数
_MINUTE_MS = 60_000


def _date_to_utc_ms(date_str: str) -> int:
    """日期字符串 → UTC 00:00:00 毫秒时间戳"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _date_range(start_date: str, end_date: str) -> List[str]:
    """生成 [start, end) 的日期列表"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    cur = start
    while cur < end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


class KlineCache:
    """
    K线缓存管理器

    一天一个文件: {symbol}USDT_1m_2026-01-15.csv.gz (1440 rows/day)
    """

    def __init__(
        self,
        cache_dir: str = "data/klines",
        symbol: str = "BTC",
        binance_client: Optional[BinanceClient] = None,
    ):
        self.cache_dir = cache_dir
        self.symbol = symbol
        self.client = binance_client or BinanceClient()
        os.makedirs(cache_dir, exist_ok=True)

    def _file_path(self, date_str: str) -> str:
        return os.path.join(self.cache_dir, f"{self.symbol}USDT_1m_{date_str}.csv.gz")

    def _day_exists(self, date_str: str) -> bool:
        path = self._file_path(date_str)
        if not os.path.exists(path):
            return False
        # 检查文件非空且有足够行数
        try:
            df = pd.read_csv(path, compression="gzip", nrows=2)
            return len(df) > 0
        except Exception:
            return False

    def download_day(self, date_str: str) -> None:
        """
        下载一天的 1m K线数据

        Binance 每次最多返回 1000 条，一天 1440 条需要 2 次调用
        """
        day_start_ms = _date_to_utc_ms(date_str)
        day_end_ms = day_start_ms + _DAY_MS - 1  # 包含最后一分钟

        logger.info(f"下载 {date_str} K线数据...")

        all_klines = self.client.get_klines_extended(
            symbol=f"{self.symbol}USDT",
            start_ms=day_start_ms,
            end_ms=day_end_ms,
        )

        if not all_klines:
            logger.warning(f"{date_str}: 无数据")
            return

        # 保存为 gzip CSV
        rows = []
        for k in all_klines:
            rows.append({
                "open_time": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
                "close_time": k.close_time,
            })

        df = pd.DataFrame(rows)
        path = self._file_path(date_str)
        df.to_csv(path, index=False, compression="gzip")
        logger.info(f"{date_str}: 保存 {len(df)} 条 → {path}")

    def ensure_range(self, start_date: str, end_date: str) -> List[str]:
        """
        确保日期范围内的数据已缓存，下载缺失的天数

        Returns:
            下载的日期列表
        """
        dates = _date_range(start_date, end_date)
        downloaded = []

        for d in dates:
            if not self._day_exists(d):
                self.download_day(d)
                downloaded.append(d)
            else:
                logger.debug(f"{d}: 已缓存")

        logger.info(f"缓存检查完成: {len(dates)} 天, 新下载 {len(downloaded)} 天")
        return downloaded

    def load_day(self, date_str: str) -> List[Kline]:
        """从缓存加载一天的 K线"""
        path = self._file_path(date_str)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缓存文件不存在: {path}")

        df = pd.read_csv(path, compression="gzip")
        klines = []
        for _, row in df.iterrows():
            klines.append(Kline(
                open_time=int(row["open_time"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                close_time=int(row["close_time"]),
            ))
        return klines

    def load_range_ms(self, start_ms: int, end_ms: int) -> List[Kline]:
        """
        按毫秒范围从缓存加载 K线

        自动确定涉及的天数并加载
        """
        # 确定起止日期
        start_dt = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=UTC)

        start_date = start_dt.strftime("%Y-%m-%d")
        # end_ms 可能跨到下一天，需要包含
        end_date = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        dates = _date_range(start_date, end_date)
        all_klines: List[Kline] = []

        for d in dates:
            try:
                day_klines = self.load_day(d)
                all_klines.extend(day_klines)
            except FileNotFoundError:
                logger.warning(f"缓存文件不存在: {d}，跳过")

        # 过滤到请求的范围
        filtered = [k for k in all_klines if start_ms <= k.open_time <= end_ms]
        return filtered
