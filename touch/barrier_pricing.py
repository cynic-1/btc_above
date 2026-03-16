"""
一触碰障碍期权解析定价

GBM 下的一触碰概率公式：
设 S_t = S0 * exp((mu - sigma^2/2)*t + sigma*W_t)
令 nu = mu - sigma^2/2, m = ln(K/S0)

上触碰 (K > S0): P(max S_t >= K) = Phi((nu*T - m)/(sigma*sqrt(T)))
                                    + exp(2*nu*m/sigma^2) * Phi((-nu*T - m)/(sigma*sqrt(T)))

下触碰 (K < S0): P(min S_t <= K) = Phi((m - nu*T)/(sigma*sqrt(T)))
                                    + exp(2*nu*m/sigma^2) * Phi((m + nu*T)/(sigma*sqrt(T)))
"""

import logging
import math
from typing import List

from scipy.stats import norm

from .models import TouchStrikeResult

logger = logging.getLogger(__name__)


def one_touch_up(s0: float, K: float, sigma: float, T: float, mu: float = 0.0) -> float:
    """
    上触碰概率: P(max_{0<=t<=T} S_t >= K)，K >= S0

    Args:
        s0: 当前价格
        K: 障碍价格 (>= s0)
        sigma: 年化波动率
        T: 剩余时间（年）
        mu: 漂移率

    Returns:
        触碰概率 [0, 1]
    """
    # 边界处理
    if K <= s0:
        return 1.0
    if T <= 0:
        return 0.0
    if sigma <= 0:
        # 确定性路径: max S_t = s0 * exp(mu * T) (mu > 0 时)
        if mu > 0:
            max_price = s0 * math.exp(mu * T)
            return 1.0 if max_price >= K else 0.0
        return 0.0

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    # 防止 sigma*sqrt(T) 极小导致数值溢出
    if sigma_sqrt_T < 1e-10:
        if mu > 0:
            max_price = s0 * math.exp(mu * T)
            return 1.0 if max_price >= K else 0.0
        return 0.0

    nu = mu - sigma ** 2 / 2
    m = math.log(K / s0)

    d1 = (nu * T - m) / sigma_sqrt_T
    d2 = (-nu * T - m) / sigma_sqrt_T

    # exp(2*nu*m/sigma^2) 可能溢出，需 clamp
    exp_arg = 2 * nu * m / (sigma ** 2)
    exp_arg = max(min(exp_arg, 500), -500)  # 防止溢出
    exp_term = math.exp(exp_arg)

    p = norm.cdf(d1) + exp_term * norm.cdf(d2)
    return max(0.0, min(1.0, p))


def one_touch_down(s0: float, K: float, sigma: float, T: float, mu: float = 0.0) -> float:
    """
    下触碰概率: P(min_{0<=t<=T} S_t <= K)，K <= S0

    Args:
        s0: 当前价格
        K: 障碍价格 (<= s0)
        sigma: 年化波动率
        T: 剩余时间（年）
        mu: 漂移率

    Returns:
        触碰概率 [0, 1]
    """
    # 边界处理
    if K >= s0:
        return 1.0
    if T <= 0:
        return 0.0
    if K <= 0:
        return 0.0
    if sigma <= 0:
        # 确定性路径: min S_t = s0 * exp(mu * T) (mu < 0 时)
        if mu < 0:
            min_price = s0 * math.exp(mu * T)
            return 1.0 if min_price <= K else 0.0
        return 0.0

    sqrt_T = math.sqrt(T)
    sigma_sqrt_T = sigma * sqrt_T

    if sigma_sqrt_T < 1e-10:
        if mu < 0:
            min_price = s0 * math.exp(mu * T)
            return 1.0 if min_price <= K else 0.0
        return 0.0

    nu = mu - sigma ** 2 / 2
    m = math.log(K / s0)  # m < 0 when K < s0

    d1 = (m - nu * T) / sigma_sqrt_T
    d2 = (m + nu * T) / sigma_sqrt_T

    # exp(2*nu*m/sigma^2) clamp
    exp_arg = 2 * nu * m / (sigma ** 2)
    exp_arg = max(min(exp_arg, 500), -500)
    exp_term = math.exp(exp_arg)

    p = norm.cdf(d1) + exp_term * norm.cdf(d2)
    return max(0.0, min(1.0, p))


def one_touch(s0: float, K: float, sigma: float, T: float, mu: float = 0.0) -> float:
    """
    自动判断方向的一触碰概率

    K > s0 → 上触碰
    K < s0 → 下触碰
    K = s0 → 1.0

    Args:
        s0: 当前价格
        K: 障碍价格
        sigma: 年化波动率
        T: 剩余时间（年）
        mu: 漂移率

    Returns:
        触碰概率 [0, 1]
    """
    if abs(K - s0) < 1e-10:
        return 1.0
    if K > s0:
        return one_touch_up(s0, K, sigma, T, mu)
    return one_touch_down(s0, K, sigma, T, mu)


def price_touch_barriers(
    s0: float,
    barriers: List[float],
    sigma: float,
    T: float,
    mu: float,
    running_high: float,
    running_low: float,
) -> List[TouchStrikeResult]:
    """
    批量定价触碰障碍

    已触碰判断：
    - 上触碰: running_high >= barrier → P = 1.0
    - 下触碰: running_low <= barrier → P = 1.0

    Args:
        s0: 当前价格
        barriers: 障碍价格列表
        sigma: 年化波动率
        T: 剩余时间（年）
        mu: 漂移率
        running_high: 月内目前为止的最高价
        running_low: 月内目前为止的最低价

    Returns:
        List[TouchStrikeResult]
    """
    results = []
    for K in barriers:
        # 判断方向
        direction = "up" if K > s0 else "down"

        # 已触碰检查
        already_touched = False
        if direction == "up" and running_high >= K:
            already_touched = True
        elif direction == "down" and running_low <= K:
            already_touched = True

        if already_touched:
            p_touch = 1.0
        else:
            p_touch = one_touch(s0, K, sigma, T, mu)

        results.append(TouchStrikeResult(
            barrier=K,
            direction=direction,
            p_touch=p_touch,
            already_touched=already_touched,
        ))

    return results
