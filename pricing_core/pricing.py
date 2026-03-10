"""
Monte Carlo 定价引擎
模拟 S_T 并计算 P(S_T > K)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .models import BasisParams, DistParams, StrikeResult
from .distribution import sample_return

logger = logging.getLogger(__name__)


def simulate_ST(
    s0: float,
    rv_hat: float,
    dist_params: DistParams,
    basis_params: BasisParams,
    n: int = 10000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    模拟事件时刻的 Binance BTC/USDT 价格

    S_T = S0 * exp(R) + b_T
    其中 R ~ 合成 return, b_T ~ N(mu_b, sigma_b^2)

    Args:
        s0: 当前 Binance 价格
        rv_hat: 预测已实现方差（季节性校正后）
        dist_params: 分布参数
        basis_params: 基差参数
        n: MC 采样数
        rng: 随机数生成器

    Returns:
        S_T 采样数组 shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 采样 return
    R = sample_return(rv_hat, dist_params, n, rng)

    # S_T (乘性 return)
    ST = s0 * np.exp(R)

    # 加性基差
    if basis_params.sigma_b > 0:
        b_T = rng.normal(basis_params.mu_b, basis_params.sigma_b, size=n)
        ST = ST + b_T
    elif basis_params.mu_b != 0:
        ST = ST + basis_params.mu_b

    return ST


def prob_above_K(
    ST_samples: np.ndarray,
    K_list: List[float],
) -> List[float]:
    """
    计算 P(S_T > K) 对多个 K

    Args:
        ST_samples: S_T 模拟样本
        K_list: 行权价列表

    Returns:
        概率列表，与 K_list 顺序对应
    """
    n = len(ST_samples)
    probs = []
    for K in K_list:
        p = float(np.sum(ST_samples > K)) / n
        probs.append(p)
    return probs


def confidence_interval(
    ST_samples: np.ndarray,
    K: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator = None,
) -> Tuple[float, float]:
    """
    用 bootstrap 估计 P(S_T > K) 的置信区间

    Args:
        ST_samples: S_T 模拟样本
        K: 行权价
        n_bootstrap: bootstrap 重采样次数
        ci_level: 置信水平
        rng: 随机数生成器

    Returns:
        (ci_lower, ci_upper) 元组
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(ST_samples)
    boot_probs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_probs[i] = np.mean(ST_samples[idx] > K)

    alpha = (1 - ci_level) / 2
    ci_lower = float(np.quantile(boot_probs, alpha))
    ci_upper = float(np.quantile(boot_probs, 1 - alpha))

    return ci_lower, ci_upper


def price_strikes(
    s0: float,
    rv_hat: float,
    dist_params: DistParams,
    basis_params: BasisParams,
    k_grid: List[float],
    n_mc: int = 10000,
    n_bootstrap: int = 500,
    rng: np.random.Generator = None,
) -> List[StrikeResult]:
    """
    对一组行权价进行完整定价

    Args:
        s0: 当前价格
        rv_hat: 预测已实现方差
        dist_params: 分布参数
        basis_params: 基差参数
        k_grid: 行权价网格
        n_mc: MC 采样数
        n_bootstrap: bootstrap 次数
        rng: 随机数生成器

    Returns:
        StrikeResult 列表
    """
    if rng is None:
        rng = np.random.default_rng()

    # 一次性模拟 S_T
    ST = simulate_ST(s0, rv_hat, dist_params, basis_params, n_mc, rng)

    # 向量化计算概率
    probs = prob_above_K(ST, k_grid)

    results = []
    for i, K in enumerate(k_grid):
        ci_lo, ci_hi = confidence_interval(ST, K, n_bootstrap, rng=rng)
        result = StrikeResult(
            strike=K,
            p_physical=probs[i],
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        )
        results.append(result)
        logger.debug(f"K={K}: p={probs[i]:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    return results
