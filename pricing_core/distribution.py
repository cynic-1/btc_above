"""
分布拟合与采样模块
Student-t 拟合、经验分布、标准化残差采样
"""

import logging
from typing import Tuple

import numpy as np
from scipy import stats

from .models import DistParams

logger = logging.getLogger(__name__)


def fit_student_t(z_samples: np.ndarray) -> DistParams:
    """
    拟合 Student-t 分布到标准化残差

    Args:
        z_samples: 标准化残差样本

    Returns:
        DistParams（df, loc, scale）
    """
    df, loc, scale = stats.t.fit(z_samples)
    params = DistParams(df=float(df), loc=float(loc), scale=float(scale))
    logger.info(f"Student-t 拟合: df={params.df:.2f}, loc={params.loc:.6f}, scale={params.scale:.6f}")
    return params


def sample_return(
    rv_hat: float,
    dist_params: DistParams,
    n: int = 10000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    从拟合分布采样合成 return

    R = z * sqrt(RV_hat)
    其中 z ~ Student-t(df, loc, scale)

    Args:
        rv_hat: 预测的已实现方差
        dist_params: 分布参数
        n: 采样数量
        rng: 随机数生成器

    Returns:
        合成 return 数组 shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 从 Student-t 采样标准化残差
    z = stats.t.rvs(
        df=dist_params.df,
        loc=dist_params.loc,
        scale=dist_params.scale,
        size=n,
        random_state=rng,
    )

    # 缩放到目标方差
    R = z * np.sqrt(rv_hat)
    return R


def build_empirical_cdf(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建经验 CDF

    Args:
        samples: 样本数据

    Returns:
        (sorted_values, cdf_values) 元组
    """
    sorted_vals = np.sort(samples)
    cdf_vals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, cdf_vals


def compute_standardized_residuals(
    log_returns: np.ndarray,
    realized_vols: np.ndarray,
) -> np.ndarray:
    """
    计算标准化残差

    z = log_return / realized_vol

    Args:
        log_returns: 对数收益率
        realized_vols: 对应的已实现波动率（标准差）

    Returns:
        标准化残差数组
    """
    # 过滤掉波动率为零或极小的样本
    mask = realized_vols > 1e-12
    z = log_returns[mask] / realized_vols[mask]
    logger.debug(f"标准化残差: {len(z)} 个有效样本, mean={np.mean(z):.4f}, std={np.std(z):.4f}")
    return z
