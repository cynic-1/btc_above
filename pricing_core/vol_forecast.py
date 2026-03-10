"""
波动率预测模块
HAR-RV 模型 + 日内季节性校正
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .models import HARFeatures, HARCoefficients

logger = logging.getLogger(__name__)


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    计算对数收益率

    Args:
        prices: 价格序列

    Returns:
        对数收益率序列（长度 = len(prices) - 1）
    """
    return np.diff(np.log(prices))


def compute_rv(returns: np.ndarray, window: int) -> float:
    """
    计算已实现方差 (Realized Variance)

    RV = sum(r_t^2) 在指定窗口内

    Args:
        returns: 对数收益率序列
        window: 窗口大小（分钟数，对应 1m returns 的条数）

    Returns:
        已实现方差
    """
    if len(returns) < window:
        logger.warning(f"returns 长度 {len(returns)} < window {window}，使用全部数据")
        return float(np.sum(returns ** 2))
    return float(np.sum(returns[-window:] ** 2))


def har_features(returns: np.ndarray, windows: List[int] = None) -> HARFeatures:
    """
    从收益率序列提取 HAR 特征

    Args:
        returns: 1m 对数收益率序列
        windows: HAR 窗口大小列表 [30, 120, 360, 1440]（分钟）

    Returns:
        HARFeatures 数据
    """
    if windows is None:
        windows = [30, 120, 360, 1440]

    rvs = []
    for w in windows:
        rvs.append(compute_rv(returns, w))

    return HARFeatures(
        rv_30m=rvs[0],
        rv_2h=rvs[1],
        rv_6h=rvs[2],
        rv_24h=rvs[3],
    )


def har_predict(features: HARFeatures, coeffs: HARCoefficients = None) -> float:
    """
    HAR-RV 模型预测

    RV_hat = b0 + b1*RV_30m + b2*RV_2h + b3*RV_6h + b4*RV_24h

    Args:
        features: HAR 特征
        coeffs: 模型系数（默认等权重）

    Returns:
        预测的已实现方差
    """
    if coeffs is None:
        coeffs = HARCoefficients()

    rv_hat = (
        coeffs.b0
        + coeffs.b1 * features.rv_30m
        + coeffs.b2 * features.rv_2h
        + coeffs.b3 * features.rv_6h
        + coeffs.b4 * features.rv_24h
    )

    # 方差不能为负
    rv_hat = max(rv_hat, 1e-12)
    logger.debug(f"HAR 预测: RV_hat={rv_hat:.8f}")
    return rv_hat


def har_fit(
    X: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float = 0.0,
) -> HARCoefficients:
    """
    拟合 HAR-RV 模型系数（OLS 或岭回归）

    Args:
        X: 特征矩阵 shape (n, 4)，列顺序 [rv_30m, rv_2h, rv_6h, rv_24h]
        y: 目标向量 shape (n,)，未来 horizon 的 RV
        ridge_alpha: 岭回归正则化系数（0 = OLS）

    Returns:
        拟合后的 HARCoefficients
    """
    n = X.shape[0]
    # 添加截距列
    X_aug = np.column_stack([np.ones(n), X])

    if ridge_alpha > 0:
        # 岭回归: (X'X + alpha*I)^{-1} X'y
        I = np.eye(X_aug.shape[1])
        I[0, 0] = 0  # 不正则化截距
        beta = np.linalg.solve(X_aug.T @ X_aug + ridge_alpha * I, X_aug.T @ y)
    else:
        # OLS
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

    coeffs = HARCoefficients(
        b0=float(beta[0]),
        b1=float(beta[1]),
        b2=float(beta[2]),
        b3=float(beta[3]),
        b4=float(beta[4]),
    )
    logger.info(f"HAR 拟合: b0={coeffs.b0:.6f}, b1={coeffs.b1:.4f}, b2={coeffs.b2:.4f}, "
                f"b3={coeffs.b3:.4f}, b4={coeffs.b4:.4f}")
    return coeffs


def intraday_seasonality_factor(
    hourly_rv: np.ndarray,
    path_hours: List[int],
) -> float:
    """
    计算日内季节性倍率

    根据从 now 到 event 覆盖的 UTC 小时段，计算波动率季节性校正因子。
    因子 = (路径小时的平均 RV 密度) / (全天平均 RV 密度)

    Args:
        hourly_rv: 长度 24 的数组，每个 UTC 小时的平均 RV 密度
        path_hours: 从 now 到 event 覆盖的 UTC 小时列表

    Returns:
        季节性倍率 g（>1 表示路径经过高波动时段）
    """
    if len(hourly_rv) != 24:
        raise ValueError(f"hourly_rv 长度必须为 24，得到 {len(hourly_rv)}")

    if not path_hours:
        return 1.0

    # 全天平均
    daily_mean = float(np.mean(hourly_rv))
    if daily_mean <= 0:
        return 1.0

    # 路径覆盖小时的平均
    path_rv = float(np.mean([hourly_rv[h % 24] for h in path_hours]))

    factor = path_rv / daily_mean
    logger.debug(f"季节性因子: path_hours={path_hours}, factor={factor:.4f}")
    return factor


def compute_hourly_rv_profile(
    returns: np.ndarray,
    timestamps_ms: np.ndarray,
) -> np.ndarray:
    """
    统计每个 UTC 小时的平均 RV 密度

    Args:
        returns: 1m 对数收益率序列
        timestamps_ms: 对应的 UTC 毫秒时间戳（与 returns 等长）

    Returns:
        长度 24 的数组，每个 UTC 小时的平均 RV
    """
    hourly_rv = np.zeros(24)
    hourly_count = np.zeros(24)

    for i, ts_ms in enumerate(timestamps_ms):
        hour = int((ts_ms / 1000) % 86400) // 3600
        hourly_rv[hour] += returns[i] ** 2
        hourly_count[hour] += 1

    # 避免除零
    mask = hourly_count > 0
    hourly_rv[mask] /= hourly_count[mask]

    return hourly_rv


def get_path_hours(now_utc_hour: int, event_utc_hour: int) -> List[int]:
    """
    获取从当前小时到事件小时覆盖的 UTC 小时列表

    Args:
        now_utc_hour: 当前 UTC 小时 (0-23)
        event_utc_hour: 事件 UTC 小时 (0-23)

    Returns:
        覆盖的小时列表
    """
    hours = []
    h = now_utc_hour
    while h != event_utc_hour:
        hours.append(h % 24)
        h = (h + 1) % 24
    hours.append(event_utc_hour % 24)
    return hours
