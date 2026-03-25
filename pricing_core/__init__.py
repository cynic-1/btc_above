"""
BTC 二元预测市场定价系统核心模块
"""

__version__ = "0.1.0"

from .models import (
    HARFeatures,
    HARCoefficients,
    BasisParams,
    DistParams,
    PricingInput,
    PricingResult,
    StrikeResult,
    TradeSignal,
)
from .config import PricingConfig
from .time_utils import et_noon_to_utc_ms, utc_ms_to_binance_kline_open
from .binance_data import BinanceClient, Kline
from .deribit_data import DeribitClient, OptionQuote, PerpInfo
from .vol_forecast import (
    compute_log_returns,
    compute_rv,
    har_features,
    har_predict,
    har_fit,
    intraday_seasonality_factor,
)
from .distribution import fit_student_t, sample_return, build_empirical_cdf
from .pricing import (
    simulate_ST, prob_above_K, confidence_interval, price_strikes,
    prob_above_K_analytical, prob_above_K_analytical_batch,
)
from .execution import (
    compute_opinion_fee,
    shrink_probability,
    compute_edge,
    should_trade,
    kelly_position,
    generate_signal,
)
from .pipeline import PricingPipeline, generate_trade_signals

__all__ = [
    # 数据模型
    "HARFeatures", "HARCoefficients", "BasisParams", "DistParams",
    "PricingInput", "PricingResult", "StrikeResult", "TradeSignal",
    # 配置
    "PricingConfig",
    # 时间工具
    "et_noon_to_utc_ms", "utc_ms_to_binance_kline_open",
    # 数据客户端
    "BinanceClient", "Kline", "DeribitClient", "OptionQuote", "PerpInfo",
    # 波动率
    "compute_log_returns", "compute_rv", "har_features", "har_predict",
    "har_fit", "intraday_seasonality_factor",
    # 分布
    "fit_student_t", "sample_return", "build_empirical_cdf",
    # 定价
    "simulate_ST", "prob_above_K", "confidence_interval", "price_strikes",
    # 执行
    "compute_opinion_fee", "shrink_probability", "compute_edge",
    "should_trade", "kelly_position", "generate_signal",
    # 管线
    "PricingPipeline", "generate_trade_signals",
]
