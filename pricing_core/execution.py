"""
交易执行模块
计算费用、收缩概率、edge、仓位大小
"""

import logging
from typing import Optional

from .models import TradeSignal

logger = logging.getLogger(__name__)


def compute_opinion_fee(price: float, min_fee: float = 0.50) -> float:
    """
    计算 Opinion 平台手续费率

    fee_rate = 0.06 * price * (1 - price) + 0.0025
    实际费用 = max(fee_rate, min_fee / 合约面值)

    Args:
        price: 合约价格 (0, 1)
        min_fee: 最低费用（美元）

    Returns:
        费率（占价格的比例）
    """
    fee_rate = 0.06 * price * (1 - price) + 0.0025
    return fee_rate


def shrink_probability(
    p_physical: float,
    market_price: float,
    lam: float = 0.6,
) -> float:
    """
    收缩概率（防过拟合）

    p_trade = lambda * p_P + (1 - lambda) * market_price

    Args:
        p_physical: 模型估计的物理概率
        market_price: 市场价格
        lam: 收缩系数（默认 0.6）

    Returns:
        交易概率
    """
    p_trade = lam * p_physical + (1 - lam) * market_price
    return max(0.001, min(0.999, p_trade))


def compute_edge(
    p_trade: float,
    market_price: float,
) -> float:
    """
    计算优势

    对 YES 方向: edge = p_trade - market_price
    正 edge 表示应买 YES，负 edge 表示应买 NO

    Args:
        p_trade: 交易概率
        market_price: 市场价格

    Returns:
        edge 值
    """
    return p_trade - market_price


def should_trade(
    edge: float,
    threshold: float = 0.03,
    uncertainty_buffer: float = 0.0,
) -> bool:
    """
    判断是否应该交易

    条件: |edge| > threshold + uncertainty_buffer

    Args:
        edge: 优势
        threshold: 基础门槛
        uncertainty_buffer: 不确定性缓冲

    Returns:
        是否交易
    """
    return abs(edge) > threshold + uncertainty_buffer


def kelly_position(
    edge: float,
    p_trade: float,
    eta: float = 0.2,
    max_position: float = 1000.0,
) -> float:
    """
    Kelly 仓位计算

    f = eta * edge / (p_trade * (1 - p_trade))

    Args:
        edge: 优势
        p_trade: 交易概率
        eta: Kelly 系数（保守缩放）
        max_position: 最大仓位限制

    Returns:
        建议仓位大小
    """
    variance = p_trade * (1 - p_trade)
    if variance < 1e-6:
        return 0.0

    f = eta * abs(edge) / variance
    # 限制最大仓位
    f = min(f, max_position)
    return f


def generate_signal(
    strike: float,
    p_trade: float,
    market_price: float,
    threshold: float = 0.03,
    uncertainty_buffer: float = 0.0,
    eta: float = 0.2,
    min_fee: float = 0.50,
    max_position: float = 1000.0,
) -> Optional[TradeSignal]:
    """
    生成交易信号

    Args:
        strike: 行权价
        p_trade: 交易概率
        market_price: 市场价格
        threshold: 入场门槛
        uncertainty_buffer: 不确定性缓冲
        eta: Kelly 系数
        min_fee: 最低费用
        max_position: 最大仓位

    Returns:
        TradeSignal 或 None（不交易）
    """
    edge = compute_edge(p_trade, market_price)

    if not should_trade(edge, threshold, uncertainty_buffer):
        return None

    # 确定方向
    if edge > 0:
        direction = "BUY_YES"
        fee = compute_opinion_fee(market_price, min_fee)
    else:
        direction = "BUY_NO"
        fee = compute_opinion_fee(1 - market_price, min_fee)

    net_edge = abs(edge) - fee
    if net_edge <= 0:
        logger.debug(f"K={strike}: edge={edge:.4f} 不足以覆盖费用 {fee:.4f}")
        return None

    position = kelly_position(edge, p_trade, eta, max_position)

    signal = TradeSignal(
        strike=strike,
        direction=direction,
        market_price=market_price,
        p_trade=p_trade,
        edge=edge,
        position_size=position,
        fee=fee,
        net_edge=net_edge,
    )

    logger.info(
        f"交易信号: K={strike}, {direction}, edge={edge:.4f}, "
        f"fee={fee:.4f}, net_edge={net_edge:.4f}, size={position:.2f}"
    )
    return signal
