"""
Deribit IV 数据管道

实时: DeribitIVSource → 从期权链获取 ATM IV
回测: DeribitIVCache → DVOL 历史缓存 + bisect 查询
"""

import csv
import gzip
import logging
import os
from bisect import bisect_right
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class DeribitIVSource:
    """
    实时 IV 获取

    从 Deribit 期权链提取 ATM IV，或直接获取 DVOL
    """

    def __init__(self, deribit_client=None):
        self._client = deribit_client

    def get_atm_iv(self, target_expiry_ms: int, currency: str = "BTC") -> Optional[float]:
        """
        获取最接近指定到期日的 ATM 隐含波动率

        从期权链中找 ATM 期权（strike 最接近当前 index 价格），
        取 call 和 put 的 mark_iv 均值

        Args:
            target_expiry_ms: 目标到期时间 (UTC ms)
            currency: 币种

        Returns:
            ATM IV（年化小数形式，如 0.65 = 65%），或 None
        """
        if self._client is None:
            return None

        try:
            quotes = self._client.get_option_chain_by_expiry(
                currency=currency,
                expiry_timestamp=target_expiry_ms,
            )
            if not quotes:
                return None

            # 获取 underlying price
            underlying = quotes[0].underlying_price
            if underlying <= 0:
                return None

            # 找 ATM: strike 最接近 underlying
            best_call_iv = None
            best_put_iv = None
            best_diff = float("inf")

            for q in quotes:
                diff = abs(q.strike - underlying)
                if diff < best_diff:
                    best_diff = diff
                    if q.option_type == "call":
                        best_call_iv = q.mark_iv
                    else:
                        best_put_iv = q.mark_iv
                elif diff == best_diff:
                    if q.option_type == "call":
                        best_call_iv = q.mark_iv
                    else:
                        best_put_iv = q.mark_iv

            # 取 call/put 均值
            ivs = [v for v in [best_call_iv, best_put_iv] if v and v > 0]
            if not ivs:
                return None

            # mark_iv 是百分比形式 (如 65.0)，转为小数 (0.65)
            atm_iv = sum(ivs) / len(ivs) / 100.0
            logger.info(f"ATM IV = {atm_iv:.4f} (expiry={target_expiry_ms})")
            return atm_iv

        except Exception as e:
            logger.warning(f"获取 ATM IV 失败: {e}")
            return None

    def get_dvol(self, currency: str = "BTC") -> Optional[float]:
        """
        获取 Deribit DVOL 指数（当前值）

        通过 DeribitClient.get_dvol() 获取 btc_dvol 指数价格

        Returns:
            DVOL（年化小数形式，如 0.55 = 55%），或 None
        """
        if self._client is None:
            return None

        try:
            dvol_pct = self._client.get_dvol(currency)
            if dvol_pct is not None and dvol_pct > 0:
                # API 返回百分比 (55.3)，转为小数 (0.553)
                return dvol_pct / 100.0
            return None
        except Exception as e:
            logger.debug(f"获取 DVOL 失败: {e}")
            return None


class DeribitIVCache:
    """
    回测用 IV 历史缓存

    支持两种数据源:
    - DVOL: 从 Deribit get_volatility_index_data 下载的 30天恒定到期 IV 指数
    - ATM IV: 由 iv_collector.py 定期采集的月底交割期权 ATM IV

    缓存为 gzip CSV，回测时通过 bisect 查询指定时刻的 IV
    """

    def __init__(self, cache_dir: str = "data/deribit_iv"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # 内存缓存: [(ts_ms, vol), ...] 已排序
        self._data: List[Tuple[int, float]] = []

    def _file_path(self, currency: str) -> str:
        return os.path.join(self.cache_dir, f"dvol_{currency.lower()}.csv.gz")

    def _atm_iv_path(self, month: str) -> str:
        """iv_collector 采集的月底 ATM IV 缓存路径"""
        return os.path.join(self.cache_dir, f"atm_iv_{month}.csv.gz")

    def download_dvol_history(
        self,
        currency: str = "BTC",
        start_ms: int = 0,
        end_ms: int = 0,
        deribit_client=None,
    ) -> None:
        """
        从 Deribit API 下载 DVOL 隐含波动率指数历史并缓存

        使用 public/get_volatility_index_data 获取 DVOL OHLC 蜡烛，
        提取 close 值。返回百分比形式 (55.3 = 55.3%)。

        Args:
            currency: 币种
            start_ms: 起始时间 (UTC ms)，0 = 不限
            end_ms: 结束时间 (UTC ms)，0 = 不限
            deribit_client: DeribitClient 实例
        """
        if deribit_client is None:
            logger.warning("未提供 DeribitClient，跳过下载")
            return

        try:
            raw_data = deribit_client.get_volatility_index_data(
                currency=currency,
                start_timestamp=start_ms,
                end_timestamp=end_ms,
                resolution=3600,  # 1小时蜡烛
            )
        except Exception as e:
            logger.warning(f"下载 DVOL 历史失败: {e}")
            return

        if not raw_data:
            logger.info(f"无 DVOL 历史数据 ({currency})")
            return

        # 写入 gzip CSV（存储百分比形式原始值）
        path = self._file_path(currency)
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "volatility"])
            for ts, vol in raw_data:
                writer.writerow([ts, vol])

        logger.info(f"缓存 DVOL 历史: {len(raw_data)} 条 → {path}")

    def _load_csv(self, path: str) -> List[Tuple[int, float]]:
        """从 gzip CSV 加载 (timestamp_ms, volatility) 数据"""
        if not os.path.exists(path):
            return []
        data = []
        with gzip.open(path, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(row["timestamp_ms"])
                    vol = float(row["volatility"])
                    data.append((ts, vol))
                except (ValueError, KeyError):
                    continue
        data.sort(key=lambda x: x[0])
        return data

    def load(self, currency: str = "BTC") -> bool:
        """
        从缓存加载 DVOL 数据到内存

        Returns:
            是否成功加载
        """
        path = self._file_path(currency)
        if not os.path.exists(path):
            logger.warning(f"DVOL 缓存不存在: {path}")
            return False

        self._data = self._load_csv(path)
        logger.info(f"加载 DVOL 缓存: {len(self._data)} 条")
        return len(self._data) > 0

    def load_atm_iv(self, month: str) -> bool:
        """
        加载 iv_collector 采集的月底 ATM IV 缓存

        如果有 ATM IV 数据，优先使用（比 DVOL 更精确）。
        ATM IV 数据会合并到 _data 中（覆盖同时间戳的 DVOL 数据）。

        Args:
            month: 月份 "YYYY-MM"

        Returns:
            是否成功加载
        """
        path = self._atm_iv_path(month)
        if not os.path.exists(path):
            return False

        atm_data = self._load_csv(path)
        if not atm_data:
            return False

        if self._data:
            # 合并: ATM IV 数据追加到 DVOL 数据之后
            # bisect 查询时 ATM IV（时间戳更新）会自然覆盖 DVOL
            combined = self._data + atm_data
            combined.sort(key=lambda x: x[0])
            self._data = combined
            logger.info(f"合并 ATM IV 缓存: {len(atm_data)} 条 (总计 {len(self._data)})")
        else:
            self._data = atm_data
            logger.info(f"加载 ATM IV 缓存: {len(atm_data)} 条")

        return True

    def set_data(self, data: List[Tuple[int, float]]) -> None:
        """直接设置数据（测试用）"""
        self._data = sorted(data, key=lambda x: x[0])

    def get_iv_at(self, timestamp_ms: int) -> Optional[float]:
        """
        查询 <= timestamp_ms 的最近 DVOL 值

        使用 bisect_right 保证不使用未来数据（防前瞻偏差）

        Args:
            timestamp_ms: 查询时刻 (UTC ms)

        Returns:
            年化波动率（小数形式，如 0.65），或 None
        """
        if not self._data:
            return None

        # bisect_right 找到第一个 > timestamp_ms 的位置
        idx = bisect_right(self._data, (timestamp_ms, float("inf"))) - 1
        if idx < 0:
            return None

        _, vol = self._data[idx]
        # Deribit 历史波动率可能以百分比形式存储 (65.0 = 65%)
        # 也可能以小数形式 (0.65) — 根据数值范围判断
        if vol > 5.0:
            return vol / 100.0
        return vol
