"""
Above 合约数据模型

BTC "above $K at ET noon" 二元合约的定价结果、观测记录、回测配置
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AboveStrikeResult:
    """单个行权价的定价结果"""
    strike: float
    p_above: float          # P(S_T > K) 模型估计
    p_trade: float = 0.0    # 收缩后交易概率
    edge: float = 0.0


@dataclass
class AboveObservation:
    """单次观测记录"""
    event_date: str              # "YYYY-MM-DD"
    obs_utc_ms: int
    event_utc_ms: int
    s0: float
    sigma: float                 # 使用的年化波动率
    T_years: float
    settlement_price: float      # ET noon 结算价（未知时为 0）
    k_grid: List[float] = field(default_factory=list)
    predictions: Dict[float, float] = field(default_factory=dict)     # {K: p_above}
    labels: Dict[float, int] = field(default_factory=dict)            # {K: 1 if settlement > K else 0}
    market_prices: Dict[float, float] = field(default_factory=dict)   # {K: polymarket_mid}


@dataclass
class AboveBacktestConfig:
    """Above 合约回测配置"""
    start_date: str = "2026-03-01"
    end_date: str = "2026-03-24"
    symbol: str = "BTC"

    # 数据目录
    cache_dir: str = "data/klines"
    iv_cache_dir: str = "data/deribit_iv"

    # 观测频率
    step_minutes: int = 15
    lookback_hours: int = 24

    # IV 来源: "dvol" | "default"
    iv_source: str = "dvol"

    # 模型参数
    vrp_k: float = 1.0
    default_sigma: float = 0.65    # 年化 IV 默认值
    mu: float = 0.0                # 漂移率

    # K grid
    use_fixed_strikes: bool = True
    k_offsets: List[int] = field(
        default_factory=lambda: [-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000]
    )

    # 交易参数
    entry_threshold: float = 0.03
    shrinkage_lambda: float = 0.6
    shares_per_trade: int = 200

    # Polymarket 配置
    use_market_prices: bool = True
    polymarket_cache_dir: str = "data/polymarket"
    polymarket_gamma_api: str = "https://gamma-api.polymarket.com"
    orderbook_cache_dir: str = "data/orderbook_cache"
    orderbook_events_json: str = "data/btc_above_events.json"
    orderbook_max_stale_minutes: float = 60.0
    orderbook_max_spread: float = 0.15
    orderbook_zero_side_policy: str = "reject"
    polygon_rpc_url: str = "https://1rpc.io/matic"
    polygon_chunk_size: int = 500
    polygon_sleep: float = 0.15
    polygon_concurrent_workers: int = 3

    # 输出目录
    output_dir: str = "above_backtest_results"
