"""
实盘交易领域模型

MarketInfo, OrderBookState, Position, Signal, TradeRecord
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MarketInfo:
    """从 Gamma API 发现的市场"""
    event_date: str
    strike: float
    condition_id: str
    yes_token_id: str
    no_token_id: str
    question: str
    tick_size: str = "0.01"
    neg_risk: bool = False


@dataclass
class OrderBookLevel:
    """订单簿单层"""
    price: float
    size: float


@dataclass
class OrderBookState:
    """单个市场的 orderbook 快照"""
    asset_id: str
    bids: List[OrderBookLevel] = field(default_factory=list)  # 降序
    asks: List[OrderBookLevel] = field(default_factory=list)  # 升序
    timestamp_ms: int = 0
    best_bid: float = 0.0
    best_ask: float = 0.0


@dataclass
class Position:
    """单个市场的仓位"""
    condition_id: str
    strike: float
    yes_shares: int = 0
    yes_cost: float = 0.0
    no_shares: int = 0
    no_cost: float = 0.0

    @property
    def net_shares(self) -> int:
        """净持仓 = YES - NO"""
        return self.yes_shares - self.no_shares


@dataclass
class Signal:
    """交易信号"""
    strike: float
    condition_id: str
    token_id: str
    direction: str          # "YES" or "NO"
    side: str               # "BUY" or "SELL"
    model_price: float
    market_price: float
    edge: float
    shares: int
    price: float            # 下单价
    timestamp_ms: int = 0


@dataclass
class TradeRecord:
    """已成交记录"""
    order_id: str
    signal: Signal
    status: str             # "live" / "matched" / "delayed" / "failed"
    response: dict = field(default_factory=dict)
    timestamp_ms: int = 0
