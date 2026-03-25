"""
实盘交易配置

从环境变量加载敏感信息，dataclass 定义所有可调参数
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LiveTradingConfig:
    """实盘交易系统配置"""

    # ==================== Polymarket ====================
    polymarket_host: str = field(
        default_factory=lambda: os.getenv("PM_HOST", "https://clob.polymarket.com")
    )
    polymarket_ws_url: str = field(
        default_factory=lambda: os.getenv(
            "PM_WS_URL",
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        )
    )
    polymarket_private_key: str = field(
        default_factory=lambda: os.getenv("PM_PRIVATE_KEY", "")
    )
    polymarket_chain_id: int = 137
    polymarket_funder: str = field(
        default_factory=lambda: os.getenv("PM_FUNDER", "")
    )
    polymarket_signature_type: int = 2  # 2=proxy
    gamma_api: str = "https://gamma-api.polymarket.com"

    # ==================== Binance ====================
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
    binance_rest_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    )
    binance_max_rps: float = 10.0

    # ==================== 交易参数 ====================
    shares_per_trade: int = 200
    max_net_shares: int = 10_000
    max_total_cost: float = 50_000.0
    entry_threshold: float = 0.03
    order_type: str = "GTC"
    order_cooldown_seconds: float = 60.0  # 同市场每分钟最多下一单
    order_timeout_seconds: float = 300.0  # GTC 挂单超时取消秒数 (默认 5 分钟)

    # 交易过滤器（与回测对齐）
    direction_filter: str = "both"                  # both / yes_only / no_only
    yes_threshold: Optional[float] = None           # YES 方向独立阈值 (None=沿用 entry_threshold)
    no_threshold: Optional[float] = None            # NO 方向独立阈值 (None=沿用 entry_threshold)
    max_spread: Optional[float] = None              # 最大 bid-ask 价差 (None=不限)
    min_minutes_to_event: int = 0                   # 距事件最少分钟数 (0=不限)
    max_minutes_to_event: int = 99999               # 距事件最多分钟数 (99999=不限)

    # ==================== 定价 ====================
    mc_samples: int = 2000
    dist_refit_minutes: int = 30
    pricing_interval_seconds: float = 10.0
    vrp_k: float = 1.0

    # ==================== 系统 ====================
    log_dir: str = field(
        default_factory=lambda: os.getenv("LOG_DIR", "logs")
    )
    event_date: str = ""
    dry_run: bool = False

    # ==================== HAR 训练 ====================
    har_train_days: int = 30
    har_ridge_alpha: float = 0.01
