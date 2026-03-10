"""
定价管线（Orchestrator）
将所有模块串联，执行完整定价流程
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from .config import PricingConfig
from .models import (
    BasisParams,
    DistParams,
    HARCoefficients,
    PricingInput,
    PricingResult,
    StrikeResult,
    TradeSignal,
)
from .time_utils import et_noon_to_utc_ms, utc_ms_to_binance_kline_open, minutes_until_event
from .binance_data import BinanceClient
from .deribit_data import DeribitClient
from .vol_forecast import (
    compute_log_returns,
    har_features,
    har_predict,
    intraday_seasonality_factor,
    compute_hourly_rv_profile,
    get_path_hours,
)
from .distribution import fit_student_t, compute_standardized_residuals
from .pricing import price_strikes
from .execution import shrink_probability, compute_edge, should_trade, kelly_position, generate_signal

logger = logging.getLogger(__name__)


class PricingPipeline:
    """
    完整定价管线

    将 Binance 数据、波动率预测、分布拟合、MC 定价、交易信号生成串联
    """

    def __init__(
        self,
        config: PricingConfig = None,
        binance_client: BinanceClient = None,
        deribit_client: DeribitClient = None,
    ):
        self.config = config or PricingConfig()
        self.binance = binance_client or BinanceClient(
            base_url=self.config.binance_base_url,
            max_rps=self.config.binance_max_rps,
        )
        self.deribit = deribit_client or DeribitClient(
            base_url=self.config.deribit_base_url,
            max_rps=self.config.deribit_max_rps,
        )

    def run(
        self,
        event_date: str,
        k_grid: Optional[List[float]] = None,
        market_prices: Optional[Dict[float, float]] = None,
        har_coeffs: Optional[HARCoefficients] = None,
        now_utc_ms: Optional[int] = None,
    ) -> PricingResult:
        """
        执行完整定价

        Args:
            event_date: 事件日期 "YYYY-MM-DD"
            k_grid: 行权价网格（可选，默认围绕当前价）
            market_prices: 各 K 对应的市场价格 {K: price}（用于计算 edge）
            har_coeffs: HAR 模型系数（可选，默认等权重）
            now_utc_ms: 当前 UTC 毫秒（可选，默认当前时刻；回测时传入模拟时刻）

        Returns:
            PricingResult
        """
        if now_utc_ms is None:
            now_utc_ms = int(time.time() * 1000)

        # 1. 计算事件时刻
        event_utc_ms = et_noon_to_utc_ms(event_date)
        event_open_ms = utc_ms_to_binance_kline_open(event_utc_ms)
        mins_to_expiry = minutes_until_event(now_utc_ms, event_utc_ms)

        logger.info(f"事件日期: {event_date}, 距到期: {mins_to_expiry:.1f} 分钟")

        if mins_to_expiry <= 0:
            logger.warning("事件已过期")

        # 2. 获取当前价格
        s0 = self.binance.get_current_price()
        logger.info(f"当前 BTC 价格: {s0:.2f}")

        # 3. 构建 K 网格
        if k_grid is None:
            k_grid = self.config.default_k_grid
        if k_grid is None:
            # 默认围绕当前价生成网格，间距 500
            base = round(s0 / 500) * 500
            k_grid = [base + i * 500 for i in range(-5, 6)]

        # 4. 获取历史 K线（过去 24h）
        lookback_ms = 24 * 60 * 60 * 1000
        klines = self.binance.get_klines_extended(
            start_ms=now_utc_ms - lookback_ms,
            end_ms=now_utc_ms,
        )
        if len(klines) < 60:
            raise ValueError(f"K线数据不足: 仅获取 {len(klines)} 条")

        prices = self.binance.get_close_prices(klines)
        returns = compute_log_returns(prices)
        timestamps = np.array([k.open_time for k in klines[1:]])  # returns 比 prices 少一条

        # 5. HAR-RV 预测
        features = har_features(returns)
        rv_hat = har_predict(features, har_coeffs)
        logger.info(f"HAR-RV 预测: RV_hat={rv_hat:.8f}, sigma_hat={np.sqrt(rv_hat):.6f}")

        # 6. 日内季节性校正
        # 获取更多历史数据用于季节性计算（使用已有数据简化）
        hourly_profile = compute_hourly_rv_profile(returns, timestamps)
        now_hour = int((now_utc_ms / 1000) % 86400) // 3600
        event_hour = int((event_utc_ms / 1000) % 86400) // 3600
        path_hours = get_path_hours(now_hour, event_hour)
        seasonality = intraday_seasonality_factor(hourly_profile, path_hours)
        rv_hat_adj = rv_hat * seasonality
        logger.info(f"季节性校正: factor={seasonality:.4f}, RV_adj={rv_hat_adj:.8f}")

        # 7. VRP 缩放
        vrp_k = self.config.vrp_default_k
        rv_hat_vrp = rv_hat_adj * (vrp_k ** 2)

        # 8. 按剩余时间缩放方差
        # HAR 模型预测的是未来 _HAR_FORWARD_HORIZON 分钟的 RV
        # 剩余 tau 分钟的期望方差 ≈ rv_hat * min(tau, horizon) / horizon
        _HAR_FORWARD_HORIZON = 360  # 与 har_trainer._FORWARD_HORIZON 一致
        tau_scale = max(min(mins_to_expiry, _HAR_FORWARD_HORIZON), 0.0) / _HAR_FORWARD_HORIZON
        rv_hat_final = rv_hat_vrp * tau_scale
        logger.info(f"时间缩放: tau={mins_to_expiry:.1f}min, scale={tau_scale:.6f}, "
                     f"RV_final={rv_hat_final:.8f}")

        # 9. 基差参数（默认值，完整实现需要 Deribit 数据）
        basis_params = BasisParams(mu_b=0.0, sigma_b=0.0)

        # 10. 分布拟合
        # 用当前 returns 的标准化残差拟合 Student-t
        rv_rolling = np.sqrt(np.convolve(returns ** 2, np.ones(30) / 30, mode='valid'))
        if len(rv_rolling) > 30:
            z_samples = returns[29:29 + len(rv_rolling)] / np.maximum(rv_rolling, 1e-12)
            dist_params = fit_student_t(z_samples)
        else:
            dist_params = DistParams(df=5.0, loc=0.0, scale=1.0)

        # 11. 构建输入快照
        pricing_input = PricingInput(
            now_utc_ms=now_utc_ms,
            event_utc_ms=event_utc_ms,
            minutes_to_expiry=mins_to_expiry,
            s0=s0,
            rv_hat=rv_hat_final,
            seasonality_factor=seasonality,
            vrp_k=vrp_k,
            basis_params=basis_params,
            dist_params=dist_params,
            k_grid=k_grid,
        )

        # 12. MC 定价
        n_mc = self.config.mc_samples
        strike_results = price_strikes(
            s0=s0,
            rv_hat=rv_hat_final,
            dist_params=dist_params,
            basis_params=basis_params,
            k_grid=k_grid,
            n_mc=n_mc,
        )

        # 13. 计算交易信号
        if market_prices:
            for sr in strike_results:
                mkt = market_prices.get(sr.strike)
                if mkt is not None:
                    sr.p_trade = shrink_probability(
                        sr.p_physical, mkt, self.config.shrinkage_lambda
                    )
                    sr.edge = compute_edge(sr.p_trade, mkt)
                    if should_trade(sr.edge, self.config.entry_threshold):
                        sr.position_size = kelly_position(
                            sr.edge, sr.p_trade, self.config.kelly_eta
                        )

        result = PricingResult(
            pricing_input=pricing_input,
            strike_results=strike_results,
            mc_samples=n_mc,
        )

        self._log_result(result)
        return result

    def _log_result(self, result: PricingResult) -> None:
        """记录定价结果（PRD §9.2）"""
        inp = result.pricing_input
        logger.info("=" * 60)
        logger.info("定价结果摘要")
        logger.info(f"  时间: now={inp.now_utc_ms}, event={inp.event_utc_ms}")
        logger.info(f"  距到期: {inp.minutes_to_expiry:.1f} 分钟")
        logger.info(f"  S0={inp.s0:.2f}, RV_hat={inp.rv_hat:.8f}")
        logger.info(f"  季节性={inp.seasonality_factor:.4f}, VRP_k={inp.vrp_k:.4f}")
        logger.info(f"  基差: mu={inp.basis_params.mu_b:.2f}, sigma={inp.basis_params.sigma_b:.2f}")
        logger.info(f"  分布: df={inp.dist_params.df:.2f}, loc={inp.dist_params.loc:.6f}, "
                     f"scale={inp.dist_params.scale:.6f}")
        logger.info(f"  MC 样本数: {result.mc_samples}")
        logger.info("-" * 60)
        for sr in result.strike_results:
            logger.info(
                f"  K={sr.strike:>10.1f}: p_P={sr.p_physical:.4f} "
                f"CI=[{sr.ci_lower:.4f},{sr.ci_upper:.4f}] "
                f"p_trade={sr.p_trade:.4f} edge={sr.edge:.4f} "
                f"size={sr.position_size:.2f}"
            )
        logger.info("=" * 60)


def generate_trade_signals(
    result: PricingResult,
    market_prices: Dict[float, float],
    config: PricingConfig = None,
) -> List[TradeSignal]:
    """
    从定价结果生成交易信号

    Args:
        result: 定价结果
        market_prices: 各 K 对应的市场价格
        config: 配置

    Returns:
        TradeSignal 列表
    """
    if config is None:
        config = PricingConfig()

    signals = []
    for sr in result.strike_results:
        mkt = market_prices.get(sr.strike)
        if mkt is None:
            continue

        signal = generate_signal(
            strike=sr.strike,
            p_trade=sr.p_trade if sr.p_trade > 0 else sr.p_physical,
            market_price=mkt,
            threshold=config.entry_threshold,
            eta=config.kelly_eta,
            min_fee=config.opinion_min_fee,
        )
        if signal is not None:
            signals.append(signal)

    return signals
