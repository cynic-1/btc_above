"""
Above 合约解析定价

GBM 下的终端概率公式:
P(S_T > K) = Phi(d2)
其中 d2 = (ln(S0/K) + (mu - sigma^2/2)*T) / (sigma*sqrt(T))
"""

import logging
import math
from typing import List

from scipy.stats import norm

from .models import AboveStrikeResult

logger = logging.getLogger(__name__)


def prob_above_k_gbm(s0: float, K: float, sigma: float, T: float, mu: float = 0.0) -> float:
    """
    GBM 下 P(S_T > K) = Phi(d2)

    Args:
        s0: 当前价格
        K: 行权价
        sigma: 年化波动率
        T: 剩余时间（年）
        mu: 漂移率

    Returns:
        P(S_T > K) 概率 [0, 1]
    """
    # 边界: K <= 0 → 必定 above
    if K <= 0:
        return 1.0

    # 边界: T <= 0 → 已到期，确定性判断
    if T <= 0:
        return 1.0 if s0 > K else 0.0

    # 边界: sigma <= 0 → 确定性路径 S_T = s0 * exp(mu * T)
    if sigma <= 0:
        s_T = s0 * math.exp(mu * T)
        return 1.0 if s_T > K else 0.0

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    # 防止 sigma*sqrt(T) 极小导致数值溢出
    if sigma_sqrt_T < 1e-10:
        s_T = s0 * math.exp(mu * T)
        return 1.0 if s_T > K else 0.0

    d2 = (math.log(s0 / K) + (mu - sigma ** 2 / 2) * T) / sigma_sqrt_T
    return float(norm.cdf(d2))


def price_above_strikes(
    s0: float,
    k_grid: List[float],
    sigma: float,
    T: float,
    mu: float = 0.0,
) -> List[AboveStrikeResult]:
    """
    批量定价 above 合约

    Args:
        s0: 当前价格
        k_grid: 行权价列表
        sigma: 年化波动率
        T: 剩余时间（年）
        mu: 漂移率

    Returns:
        List[AboveStrikeResult]
    """
    results = []
    for K in k_grid:
        p = prob_above_k_gbm(s0, K, sigma, T, mu)
        results.append(AboveStrikeResult(strike=K, p_above=p))
    return results
