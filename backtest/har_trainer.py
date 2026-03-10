"""
Walk-Forward HAR 系数训练
用事件日之前的数据训练 HAR 模型，按间隔缓存系数避免重复训练
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from pricing_core.binance_data import Kline
from pricing_core.models import HARCoefficients
from pricing_core.vol_forecast import compute_log_returns, compute_rv, har_fit

from .data_cache import KlineCache

logger = logging.getLogger(__name__)

# HAR 特征窗口（分钟）
_HAR_WINDOWS = [30, 120, 360, 1440]
# 前向预测 horizon（分钟）：预测未来 6h 的 RV
_FORWARD_HORIZON = 360


class HARTrainer:
    """
    Walk-forward HAR 系数训练器

    按 retrain_interval 天缓存系数，避免每天重复训练
    """

    def __init__(
        self,
        cache: KlineCache,
        train_days: int = 30,
        retrain_interval: int = 7,
        ridge_alpha: float = 0.01,
    ):
        self.cache = cache
        self.train_days = train_days
        self.retrain_interval = retrain_interval
        self.ridge_alpha = ridge_alpha
        self._coeffs_cache: Dict[str, HARCoefficients] = {}

    def _cache_key(self, as_of_date: str) -> str:
        """按 retrain_interval 对齐日期作为缓存 key"""
        dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        # 对齐到最近的 retrain_interval 边界
        epoch = datetime(2026, 1, 1)
        days_since = (dt - epoch).days
        aligned_days = (days_since // self.retrain_interval) * self.retrain_interval
        aligned_dt = epoch + timedelta(days=aligned_days)
        return aligned_dt.strftime("%Y-%m-%d")

    def get_coeffs(self, as_of_date: str) -> HARCoefficients:
        """
        获取截至某日的 HAR 系数

        如果缓存命中则直接返回，否则训练新系数

        Args:
            as_of_date: 事件日期（只用事件日之前的数据）

        Returns:
            训练好的 HARCoefficients
        """
        key = self._cache_key(as_of_date)

        if key in self._coeffs_cache:
            logger.debug(f"HAR 系数缓存命中: {as_of_date} → key={key}")
            return self._coeffs_cache[key]

        logger.info(f"训练 HAR 系数: as_of={as_of_date}, key={key}, "
                     f"lookback={self.train_days}d")

        coeffs = self._train(as_of_date)
        self._coeffs_cache[key] = coeffs
        return coeffs

    def _train(self, as_of_date: str) -> HARCoefficients:
        """
        用 as_of_date 前 train_days 天的数据训练 HAR 模型

        构建训练集:
          X: [rv_30m, rv_2h, rv_6h, rv_24h] 在每个时间步
          y: 未来 _FORWARD_HORIZON 分钟的 RV
        """
        dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        end_date = as_of_date  # 不包含事件日当天
        start_dt = dt - timedelta(days=self.train_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        # 从缓存加载 K线
        from .data_cache import _date_to_utc_ms
        start_ms = _date_to_utc_ms(start_date)
        end_ms = _date_to_utc_ms(end_date) - 1  # 不包含事件日

        klines = self.cache.load_range_ms(start_ms, end_ms)
        if len(klines) < _HAR_WINDOWS[-1] + _FORWARD_HORIZON + 100:
            logger.warning(f"训练数据不足: {len(klines)} 条 K线, "
                           f"回退使用默认系数")
            return HARCoefficients()

        prices = np.array([k.close for k in klines])
        returns = compute_log_returns(prices)

        # 构建训练样本
        X_list: List[List[float]] = []
        y_list: List[float] = []

        min_start = _HAR_WINDOWS[-1]  # 需要足够历史计算最大窗口特征
        max_end = len(returns) - _FORWARD_HORIZON  # 需要未来数据作为目标

        # 每 60 分钟取一个样本（避免过多重叠）
        step = 60
        for i in range(min_start, max_end, step):
            # 特征: 各窗口的 RV
            features = []
            for w in _HAR_WINDOWS:
                rv = float(np.sum(returns[i - w:i] ** 2))
                features.append(rv)
            X_list.append(features)

            # 目标: 未来 horizon 的 RV
            y_rv = float(np.sum(returns[i:i + _FORWARD_HORIZON] ** 2))
            y_list.append(y_rv)

        if len(X_list) < 10:
            logger.warning(f"训练样本不足: {len(X_list)} 个, 回退使用默认系数")
            return HARCoefficients()

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"HAR 训练: {len(X)} 样本, 特征范围 "
                     f"rv_30m=[{X[:,0].min():.8f}, {X[:,0].max():.8f}]")

        coeffs = har_fit(X, y, ridge_alpha=self.ridge_alpha)
        return coeffs
