"""
数据模型定义模块
包含定价系统使用的所有数据类
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HARFeatures:
    """HAR-RV 模型特征"""
    rv_30m: float
    rv_2h: float
    rv_6h: float
    rv_24h: float


@dataclass
class HARCoefficients:
    """HAR-RV 模型系数"""
    b0: float = 0.0
    b1: float = 0.25
    b2: float = 0.25
    b3: float = 0.25
    b4: float = 0.25


@dataclass
class BasisParams:
    """Deribit-Binance 基差参数"""
    mu_b: float = 0.0       # 基差均值
    sigma_b: float = 0.0    # 基差标准差


@dataclass
class DistParams:
    """分布参数（Student-t）"""
    df: float = 5.0          # 自由度
    loc: float = 0.0         # 位置参数
    scale: float = 1.0       # 尺度参数


@dataclass
class PricingInput:
    """定价引擎输入快照"""
    now_utc_ms: int                        # 当前 UTC 毫秒
    event_utc_ms: int                      # 事件时刻 UTC 毫秒
    minutes_to_expiry: float               # 距到期分钟数
    s0: float                              # 当前 Binance BTC/USDT 价格
    rv_hat: float                          # HAR-RV 预测值
    seasonality_factor: float              # 日内季节性倍率
    vrp_k: float                           # VRP 缩放系数
    basis_params: BasisParams              # 基差参数
    dist_params: DistParams                # 分布参数
    k_grid: List[float] = field(default_factory=list)  # 行权价网格


@dataclass
class StrikeResult:
    """单个行权价的定价结果"""
    strike: float
    p_physical: float          # 物理概率 P(S_T > K)
    ci_lower: float            # 置信区间下界
    ci_upper: float            # 置信区间上界
    p_trade: float = 0.0      # 收缩后交易概率
    edge: float = 0.0         # 优势 = p_trade - market_price
    position_size: float = 0.0  # 建议仓位


@dataclass
class PricingResult:
    """完整定价输出"""
    pricing_input: PricingInput
    strike_results: List[StrikeResult] = field(default_factory=list)
    mc_samples: int = 0


@dataclass
class TradeSignal:
    """交易信号"""
    strike: float
    direction: str             # "BUY_YES" 或 "BUY_NO"
    market_price: float
    p_trade: float
    edge: float
    position_size: float
    fee: float
    net_edge: float            # edge - fee_rate
