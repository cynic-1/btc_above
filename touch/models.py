"""
一触碰障碍期权数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TouchMarketInfo:
    """Polymarket 触碰合约信息"""
    month: str              # "2026-03"
    barrier: float          # 触碰价格
    direction: str          # "up" or "down"
    condition_id: str
    yes_token_id: str
    no_token_id: str
    question: str


@dataclass
class TouchStrikeResult:
    """单个障碍价格的定价结果"""
    barrier: float
    direction: str          # "up" or "down"
    p_touch: float          # 触碰概率
    already_touched: bool
    p_trade: float = 0.0
    edge: float = 0.0


@dataclass
class TouchObservationResult:
    """单次观测的定价结果"""
    obs_utc_ms: int
    s0: float
    running_high: float
    running_low: float
    T_remaining_years: float
    sigma: float
    barriers: List[float] = field(default_factory=list)
    # 各 barrier 的预测概率 {barrier: p_touch}
    predictions: Dict[float, float] = field(default_factory=dict)
    # 各 barrier 的标签 {barrier: 0/1}（月末是否触碰）
    labels: Dict[float, int] = field(default_factory=dict)
    # 各 barrier 的市场价格 {barrier: polymarket_price}
    market_prices: Dict[float, float] = field(default_factory=dict)
    # 各 barrier 是否已触碰
    already_touched: Dict[float, bool] = field(default_factory=dict)


@dataclass
class TouchBacktestConfig:
    """触碰期权回测配置"""
    month: str = "2026-03"
    symbol: str = "BTC"  # 币种: "BTC" | "ETH"

    # 数据目录
    cache_dir: str = "data/klines"
    iv_cache_dir: str = "data/deribit_iv"

    # 观测频率（月度跨度大，小时级即可）
    step_minutes: int = 60

    # IV 来源: "dvol" | "option_chain" | "har_rv"
    iv_source: str = "dvol"

    # 模型参数
    vrp_k: float = 1.0
    default_sigma: float = 0.65    # 年化 IV 默认值
    mu: float = 0.0                # 漂移率（零漂移，保守）

    # DVOL 期限结构校正
    # sigma_adj = dvol * (30 / T_days_remaining)^alpha
    # alpha=0 表示不校正; alpha=0.05 为 BTC 温和反向期限结构
    term_structure_alpha: float = 0.05

    # 交易参数
    initial_capital: float = 100_000.0
    shares_per_trade: int = 200
    entry_threshold: float = 0.03
    shrinkage_lambda: float = 0.6

    # Polymarket 配置
    use_market_prices: bool = True
    polymarket_cache_dir: str = "data/polymarket"
    polymarket_gamma_api: str = "https://gamma-api.polymarket.com"
    polymarket_clob_api: str = "https://clob.polymarket.com"
    orderbook_cache_dir: str = "data/orderbook_cache"

    # 输出目录
    output_dir: str = "touch_backtest_results"
