"""
实时定价引擎

改编自 FastPricingEngine (backtest/chart_engine.py:75-170)，
去掉 HistoricalBinanceClient 依赖，直接使用 BinanceKlineFeed 的数据
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytz

from pricing_core.binance_data import Kline
from pricing_core.distribution import fit_student_t
from pricing_core.models import BasisParams, DistParams, HARCoefficients
from pricing_core.pricing import prob_above_K_analytical_batch
from pricing_core.time_utils import et_noon_to_utc_ms, minutes_until_event
from pricing_core.vol_forecast import (
    compute_hourly_rv_profile,
    compute_log_returns,
    compute_rv,
    get_path_hours,
    har_features,
    har_fit,
    har_predict,
    intraday_seasonality_factor,
)

from .config import LiveTradingConfig

logger = logging.getLogger(__name__)

# HAR 前向预测 horizon（分钟）
_HAR_FORWARD_HORIZON = 360
# HAR 特征窗口
_HAR_WINDOWS = [30, 120, 360, 1440]


class LivePricingEngine:
    """
    实时定价引擎

    流程:
    1. 从 BinanceKlineFeed 获取 24h klines + 当前价
    2. compute_log_returns → har_features → har_predict
    3. intraday_seasonality_factor 校正
    4. VRP 缩放 + 时间衰减
    5. fit_student_t（缓存 30min）
    6. simulate_ST → prob_above_K
    """

    def __init__(self, config: LiveTradingConfig):
        self._config = config
        self._cached_dist_params: Optional[DistParams] = None
        self._cached_dist_time_ms: int = 0
        self._har_coeffs: Optional[HARCoefficients] = None

    def compute_prices(
        self,
        event_date: str,
        now_utc_ms: int,
        s0: float,
        klines: List[Kline],
        k_list: List[float],
    ) -> Dict[float, float]:
        """
        计算各 strike 的 p_physical

        Args:
            event_date: 事件日期 "YYYY-MM-DD"
            now_utc_ms: 当前 UTC 毫秒
            s0: 当前 BTC 价格
            klines: 24h K线数据
            k_list: 行权价列表

        Returns:
            {strike: probability}
        """
        if not k_list:
            return {}

        # 过滤 K线到最近 24h+1min（多取 1 条，确保差分后 returns >= 1440）
        lookback_ms = 24 * 60 * 60 * 1000 + 60_000
        cutoff_ms = now_utc_ms - lookback_ms
        klines = [k for k in klines if k.open_time >= cutoff_ms]

        if len(klines) < 60:
            logger.warning(f"K线数据不足: {len(klines)} < 60，跳过定价")
            return {}

        # 1. 计算 log returns
        prices = np.array([k.close for k in klines])
        returns = compute_log_returns(prices)
        timestamps = np.array([k.open_time for k in klines[1:]])

        # 2. HAR-RV 预测
        coeffs = self._har_coeffs or HARCoefficients()
        features = har_features(returns)
        rv_hat = har_predict(features, coeffs)

        # 3. 日内季节性校正
        event_utc_ms = et_noon_to_utc_ms(event_date)
        hourly_profile = compute_hourly_rv_profile(returns, timestamps)
        now_hour = int((now_utc_ms / 1000) % 86400) // 3600
        event_hour = int((event_utc_ms / 1000) % 86400) // 3600
        path_hours = get_path_hours(now_hour, event_hour)
        seasonality = intraday_seasonality_factor(hourly_profile, path_hours)
        rv_hat_adj = rv_hat * seasonality

        # 4. VRP 缩放
        rv_hat_vrp = rv_hat_adj * (self._config.vrp_k ** 2)

        # 5. 按剩余时间缩放方差
        mins_to_expiry = minutes_until_event(now_utc_ms, event_utc_ms)
        tau_scale = max(min(mins_to_expiry, _HAR_FORWARD_HORIZON), 0.0) / _HAR_FORWARD_HORIZON
        rv_hat_final = rv_hat_vrp * tau_scale

        # 6. 基差参数（默认值）
        basis_params = BasisParams(mu_b=0.0, sigma_b=0.0)

        # 7. 分布拟合（带缓存）
        need_refit = (
            self._cached_dist_params is None
            or (now_utc_ms - self._cached_dist_time_ms)
            > self._config.dist_refit_minutes * 60_000
        )
        if need_refit:
            rv_rolling = np.sqrt(
                np.convolve(returns ** 2, np.ones(30) / 30, mode="valid")
            )
            if len(rv_rolling) > 30:
                z_samples = returns[29:29 + len(rv_rolling)] / np.maximum(
                    rv_rolling, 1e-12
                )
                self._cached_dist_params = fit_student_t(z_samples)
            else:
                self._cached_dist_params = DistParams(df=5.0, loc=0.0, scale=1.0)
            self._cached_dist_time_ms = now_utc_ms
            logger.info(
                f"分布重拟合: df={self._cached_dist_params.df:.2f}, "
                f"loc={self._cached_dist_params.loc:.6f}, "
                f"scale={self._cached_dist_params.scale:.6f}"
            )

        # 8. Student-t 解析定价
        probs = prob_above_K_analytical_batch(
            s0, k_list, rv_hat_final, self._cached_dist_params,
        )

        result = dict(zip(k_list, probs))

        logger.debug(
            f"定价完成: s0={s0:.2f}, rv_hat={rv_hat:.8f}, "
            f"seasonality={seasonality:.4f}, tau_scale={tau_scale:.4f}, "
            f"mins_to_expiry={mins_to_expiry:.1f}"
        )

        return result

    def train_har(self, klines: List[Kline], event_date: str) -> None:
        """
        用历史 K线训练 HAR 系数

        与回测 har_trainer 一致：排除 event_date 当天数据，
        仅用 event_date 前的数据训练

        Args:
            klines: 历史 K线数据（至少需要 train_days 天）
            event_date: 事件日期 "YYYY-MM-DD"，当天数据不参与训练
        """
        # 排除事件日当天数据（与回测 har_trainer 一致）
        midnight_utc_ms = int(
            datetime.strptime(event_date, "%Y-%m-%d")
            .replace(tzinfo=pytz.utc)
            .timestamp() * 1000
        )
        klines = [k for k in klines if k.open_time < midnight_utc_ms]

        if len(klines) < _HAR_WINDOWS[-1] + _HAR_FORWARD_HORIZON + 100:
            logger.warning(
                f"训练数据不足: {len(klines)} 条 K线，使用默认系数"
            )
            self._har_coeffs = HARCoefficients()
            return

        prices = np.array([k.close for k in klines])
        returns = compute_log_returns(prices)

        # 构建训练样本
        X_list: List[List[float]] = []
        y_list: List[float] = []

        min_start = _HAR_WINDOWS[-1]
        max_end = len(returns) - _HAR_FORWARD_HORIZON

        step = 60  # 每 60 分钟取一个样本
        for i in range(min_start, max_end, step):
            features = []
            for w in _HAR_WINDOWS:
                rv = float(np.sum(returns[i - w:i] ** 2))
                features.append(rv)
            X_list.append(features)
            y_rv = float(np.sum(returns[i:i + _HAR_FORWARD_HORIZON] ** 2))
            y_list.append(y_rv)

        if len(X_list) < 10:
            logger.warning(f"训练样本不足: {len(X_list)} 个，使用默认系数")
            self._har_coeffs = HARCoefficients()
            return

        X = np.array(X_list)
        y = np.array(y_list)

        self._har_coeffs = har_fit(X, y, ridge_alpha=self._config.har_ridge_alpha)
        logger.info(f"HAR 训练完成: {len(X_list)} 个样本")

    def reset_cache(self) -> None:
        """重置分布缓存（新 event_date 时调用）"""
        self._cached_dist_params = None
        self._cached_dist_time_ms = 0
