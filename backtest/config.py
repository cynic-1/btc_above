"""
回测配置
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestConfig:
    """回测参数配置"""

    # 回测日期范围
    start_date: str = "2026-02-07"
    end_date: str = "2026-03-01"

    # K线缓存目录
    cache_dir: str = "data/klines"

    # 观测频率
    step_minutes: int = 1          # 每分钟观测一次
    lookback_hours: int = 24       # 事件前多少小时开始

    # HAR 训练参数
    har_train_days: int = 30
    har_retrain_interval: int = 7
    har_ridge_alpha: float = 0.01

    # K 网格偏移量（相对于当前价格对齐到 500 的整数倍）
    k_offsets: List[float] = field(
        default_factory=lambda: [-2000, -1000, -500, 0, 500, 1000, 2000]
    )

    # 输出目录
    output_dir: str = "backtest_results"

    # MC 采样数
    mc_samples: int = 2000

    # Student-t 重拟合间隔（分钟），缓存避免重复拟合
    dist_refit_minutes: int = 30

    # 交易参数
    initial_capital: float = 100_000.0
    shares_per_trade: int = 200
    max_net_shares: int = 10_000
    entry_threshold: float = 0.03

    # 校准分析
    apply_calibration: bool = False
    calibration_train_frac: float = 0.6

    # Walk-forward 验证
    run_walk_forward: bool = False
    wf_train_days: int = 7
    wf_test_days: int = 5
    wf_step_days: int = 3

    # Polymarket 市场价格配置
    use_market_prices: bool = True
    polymarket_cache_dir: str = "data/polymarket"
    polymarket_gamma_api: str = "https://gamma-api.polymarket.com"
    polymarket_clob_api: str = "https://clob.polymarket.com"
    polymarket_max_snap_diff: float = 250.0

    # 订单簿数据配置
    use_orderbook: bool = True
    orderbook_cache_dir: str = "data/orderbook_cache"
    orderbook_events_json: str = "data/btc_above_events.json"

    # 订单簿过期阈值（小时），超过后回退 CLOB
    orderbook_staleness_hours: float = 2.0

    # 方向过滤: "both" (默认), "yes_only", "no_only"
    direction_filter: str = "both"

    # 交易冷却期（分钟）: 同一 (date, strike) 两次交易间最少等待分钟数
    # 0 = 禁用（每分钟都可交易），30 = 每 30 分钟最多交易一次
    cooldown_minutes: int = 0
