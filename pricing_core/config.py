"""
配置管理模块
集中管理定价系统所有配置参数，支持环境变量覆盖
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PricingConfig:
    """定价系统配置"""

    # ==================== 数据源 ====================
    binance_base_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    )
    binance_max_rps: float = field(
        default_factory=lambda: float(os.getenv("BINANCE_MAX_RPS", "10"))
    )
    deribit_base_url: str = field(
        default_factory=lambda: os.getenv("DERIBIT_BASE_URL", "https://www.deribit.com/api/v2")
    )
    deribit_max_rps: float = field(
        default_factory=lambda: float(os.getenv("DERIBIT_MAX_RPS", "10"))
    )

    # ==================== HAR-RV ====================
    rv_frequency: str = "1m"
    har_windows_minutes: List[int] = field(default_factory=lambda: [30, 120, 360, 1440])
    seasonality_lookback_days: int = field(
        default_factory=lambda: int(os.getenv("SEASONALITY_LOOKBACK_DAYS", "60"))
    )

    # ==================== VRP ====================
    vrp_calibration_days: int = field(
        default_factory=lambda: int(os.getenv("VRP_CALIBRATION_DAYS", "45"))
    )
    vrp_default_k: float = field(
        default_factory=lambda: float(os.getenv("VRP_DEFAULT_K", "1.0"))
    )

    # ==================== 基差 ====================
    basis_lookback_days: int = field(
        default_factory=lambda: int(os.getenv("BASIS_LOOKBACK_DAYS", "3"))
    )

    # ==================== Monte Carlo ====================
    mc_samples: int = field(
        default_factory=lambda: int(os.getenv("MC_SAMPLES", "10000"))
    )

    # ==================== 交易参数 ====================
    shrinkage_lambda: float = field(
        default_factory=lambda: float(os.getenv("SHRINKAGE_LAMBDA", "0.6"))
    )
    kelly_eta: float = field(
        default_factory=lambda: float(os.getenv("KELLY_ETA", "0.2"))
    )
    entry_threshold: float = field(
        default_factory=lambda: float(os.getenv("ENTRY_THRESHOLD", "0.03"))
    )
    opinion_min_fee: float = field(
        default_factory=lambda: float(os.getenv("OPINION_MIN_FEE", "0.50"))
    )

    # ==================== 日志 ====================
    log_dir: str = field(
        default_factory=lambda: os.getenv("LOG_DIR", "logs")
    )

    # ==================== 默认 K 网格 ====================
    default_k_grid: Optional[List[float]] = None
