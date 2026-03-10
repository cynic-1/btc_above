"""
回测数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EventOutcome:
    """单个事件日的结算结果"""
    event_date: str              # "YYYY-MM-DD"
    event_utc_ms: int            # 事件时刻 UTC 毫秒
    settlement_price: float      # 结算价格
    labels: Dict[float, int] = field(default_factory=dict)  # {K: 0/1}


@dataclass
class ObservationResult:
    """单次观测的定价结果"""
    event_date: str              # 事件日期
    obs_minutes: int             # 距事件分钟数
    now_utc_ms: int              # 观测时刻 UTC 毫秒
    s0: float                    # 观测时的 BTC 价格
    settlement_price: float      # 结算价格
    k_grid: List[float] = field(default_factory=list)
    # 各 K 的预测概率 {K: p_physical}
    predictions: Dict[float, float] = field(default_factory=dict)
    # 各 K 的标签 {K: 0/1}
    labels: Dict[float, int] = field(default_factory=dict)
    # 各 K 的置信区间 {K: (ci_lower, ci_upper)}
    confidence_intervals: Dict[float, tuple] = field(default_factory=dict)
    # 各 K 的市场价格 {K: price}（来自 Polymarket）
    market_prices: Dict[float, float] = field(default_factory=dict)
    # 各 K 的 bid/ask 报价 {K: (bid, ask)}（来自订单簿）
    market_bid_ask: Dict[float, tuple] = field(default_factory=dict)
    # 运行耗时（秒）
    elapsed_seconds: float = 0.0


@dataclass
class BacktestResult:
    """完整回测结果"""
    start_date: str
    end_date: str
    observations: List[ObservationResult] = field(default_factory=list)
    event_outcomes: List[EventOutcome] = field(default_factory=list)
    # 缓存的汇总指标
    summary: Optional[Dict] = None
