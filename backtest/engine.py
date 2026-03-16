"""
回测引擎
主编排器：遍历事件日、逐分钟运行定价，收集结果
"""

import logging
import os
import time as time_mod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from pricing_core.config import PricingConfig
from pricing_core.models import HARCoefficients
from pricing_core.time_utils import et_noon_to_utc_ms, utc_ms_to_binance_kline_open

from .config import BacktestConfig
from .chart_engine import ChartConfig, FastPricingEngine
from .data_cache import KlineCache, _date_range, _date_to_utc_ms
from .har_trainer import HARTrainer
from .historical_client import HistoricalBinanceClient
from .models import BacktestResult, EventOutcome, ObservationResult
from .hybrid_lookup import HybridPriceLookup
from .orderbook_reader import OrderbookPriceLookup, load_markets_from_events_json
from .polymarket_discovery import discover_markets_for_range, PolymarketMarketInfo
from .polymarket_trades import PolymarketTradeCache, PolymarketPriceLookup

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    回测引擎

    遍历日期范围内的每个事件日，逐分钟运行 FastPricingEngine 定价，
    每分钟检查买入/卖出信号，收集结果
    """

    def __init__(
        self,
        config: BacktestConfig = None,
        cache: KlineCache = None,
    ):
        self.config = config or BacktestConfig()
        self.cache = cache or KlineCache(cache_dir=self.config.cache_dir)
        self.trainer = HARTrainer(
            cache=self.cache,
            train_days=self.config.har_train_days,
            retrain_interval=self.config.har_retrain_interval,
            ridge_alpha=self.config.har_ridge_alpha,
        )
        # Polymarket 市场价格（PolymarketPriceLookup 或 OrderbookPriceLookup）
        self._poly_markets: Optional[dict] = None
        self._price_lookup = None  # PolymarketPriceLookup | OrderbookPriceLookup

    def download_data(self) -> None:
        """下载回测所需的全部 K线数据（含 HAR 训练用的额外历史）"""
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        extra_start = start_dt - timedelta(days=self.config.har_train_days + 1)
        extra_start_str = extra_start.strftime("%Y-%m-%d")

        logger.info(f"下载数据: {extra_start_str} ~ {self.config.end_date}")
        self.cache.ensure_range(extra_start_str, self.config.end_date)

    def download_polymarket_data(self) -> None:
        """下载 Polymarket 市场发现 + 交易数据（混合: orderbook + CLOB 回退）"""
        if not self.config.use_market_prices:
            logger.info("Polymarket 市场价格已禁用")
            return

        # 1. 始终从 events.json 加载市场映射
        events_json = self.config.orderbook_events_json
        if os.path.exists(events_json):
            try:
                self._poly_markets = load_markets_from_events_json(events_json)
            except Exception as e:
                logger.warning(f"加载 events.json 失败: {e}")

        # 回退到 API discovery
        if not self._poly_markets:
            cache_path = os.path.join(
                self.config.polymarket_cache_dir, "discovery_cache.json"
            )
            self._poly_markets = discover_markets_for_range(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                gamma_api=self.config.polymarket_gamma_api,
                cache_path=cache_path,
            )

        if not self._poly_markets:
            logger.warning("未发现任何 Polymarket 市场")
            return

        cids = list(set(info.condition_id for info in self._poly_markets.values()))

        # 2. 尝试初始化 OrderbookPriceLookup
        ob_lookup = None
        if self.config.use_orderbook:
            cache_dir = self.config.orderbook_cache_dir
            if os.path.isdir(cache_dir):
                npz_files = [f for f in os.listdir(cache_dir) if f.endswith(".npz")]
                if npz_files:
                    ob_lookup = OrderbookPriceLookup(cache_dir=cache_dir)
                    ob_lookup.preload(cids)
                    logger.info(f"订单簿查询初始化: {len(npz_files)} 个 npz 文件")

        # 3. 初始化 PolymarketPriceLookup (CLOB)
        trade_cache = PolymarketTradeCache(
            cache_dir=self.config.polymarket_cache_dir,
            clob_api=self.config.polymarket_clob_api,
        )
        trade_cache.ensure_trades(self._poly_markets)
        clob_lookup = PolymarketPriceLookup(trade_cache)
        clob_lookup.preload(cids)

        # 4. 构建最终 lookup
        if ob_lookup is not None:
            staleness_ms = int(self.config.orderbook_staleness_hours * 3_600_000)
            self._price_lookup = HybridPriceLookup(
                orderbook=ob_lookup,
                clob=clob_lookup,
                staleness_threshold_ms=staleness_ms,
            )
            logger.info("使用混合价格源 (orderbook + CLOB 回退)")
        else:
            self._price_lookup = clob_lookup
            logger.info("使用 CLOB 价格历史")

    def run(self) -> BacktestResult:
        """
        执行完整回测

        逐分钟运行 FastPricingEngine，每分钟检查交易信号

        Returns:
            BacktestResult 包含所有观测结果和事件标签
        """
        result = BacktestResult(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        # 预加载所有 K线到历史客户端
        hist_client = self._build_historical_client()

        # 下载 Polymarket 数据（如果尚未下载）
        if self.config.use_market_prices and self._poly_markets is None:
            self.download_polymarket_data()

        # 构建 FastPricingEngine
        chart_config = ChartConfig(
            mc_samples=self.config.mc_samples,
            dist_refit_minutes=self.config.dist_refit_minutes,
        )
        pricing_config = PricingConfig()
        engine = FastPricingEngine(
            hist_client=hist_client,
            config=chart_config,
            vrp_k=pricing_config.vrp_default_k,
        )

        # 预计算固定 strikes（从 Polymarket 市场发现）
        fixed_strikes_by_date: Dict[str, List[float]] = {}
        if (self.config.use_fixed_strikes
                and self.config.use_market_prices
                and self._poly_markets):
            for (d, strike) in self._poly_markets:
                fixed_strikes_by_date.setdefault(d, []).append(strike)
            for d in fixed_strikes_by_date:
                fixed_strikes_by_date[d] = sorted(fixed_strikes_by_date[d])
            logger.info(f"固定 strikes: {len(fixed_strikes_by_date)} 个事件日")

        # 遍历事件日
        event_dates = _date_range(self.config.start_date, self.config.end_date)
        total_dates = len(event_dates)
        step_ms = self.config.step_minutes * 60_000
        lookback_ms = self.config.lookback_hours * 60 * 60 * 1000

        for date_idx, event_date in enumerate(event_dates):
            t0 = time_mod.monotonic()
            logger.info(f"[{date_idx + 1}/{total_dates}] 处理事件日: {event_date}")

            # 计算事件时刻和结算价
            event_utc_ms = et_noon_to_utc_ms(event_date)
            event_open_ms = utc_ms_to_binance_kline_open(event_utc_ms)

            try:
                settlement = hist_client.get_close_at_event(event_open_ms)
            except ValueError:
                logger.warning(f"{event_date}: 无结算数据，跳过")
                continue

            # 获取 HAR 系数（walk-forward）
            har_coeffs = self.trainer.get_coeffs(event_date)

            # 重置分布缓存
            engine.reset_dist_cache()

            # 逐分钟步进
            start_ms = event_utc_ms - lookback_ms
            current_ms = start_ms
            obs_count = 0

            while current_ms < event_utc_ms:
                obs_minutes = int((event_utc_ms - current_ms) / 60_000)

                try:
                    obs = self._run_single_observation(
                        engine=engine,
                        hist_client=hist_client,
                        event_date=event_date,
                        event_utc_ms=event_utc_ms,
                        obs_minutes=obs_minutes,
                        now_ms=current_ms,
                        settlement=settlement,
                        har_coeffs=har_coeffs,
                        fixed_strikes=fixed_strikes_by_date.get(event_date),
                    )
                    result.observations.append(obs)
                    obs_count += 1
                except Exception as e:
                    logger.debug(f"{event_date} T-{obs_minutes}m: {e}")

                current_ms += step_ms

            # 记录事件结果
            labels = {}
            if result.observations:
                last_obs = result.observations[-1]
                if last_obs.event_date == event_date:
                    labels = last_obs.labels.copy()

            result.event_outcomes.append(EventOutcome(
                event_date=event_date,
                event_utc_ms=event_utc_ms,
                settlement_price=settlement,
                labels=labels,
            ))

            elapsed = time_mod.monotonic() - t0
            logger.info(f"  {event_date}: {obs_count} 观测, 耗时 {elapsed:.1f}s")

        logger.info(f"回测完成: {len(result.event_outcomes)} 事件, "
                     f"{len(result.observations)} 观测")
        return result

    def _build_historical_client(self) -> HistoricalBinanceClient:
        """构建并预加载历史客户端"""
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        extra_start = start_dt - timedelta(days=self.config.har_train_days + 1)
        start_ms = _date_to_utc_ms(extra_start.strftime("%Y-%m-%d"))
        end_ms = _date_to_utc_ms(self.config.end_date)

        klines = self.cache.load_range_ms(start_ms, end_ms)
        logger.info(f"历史客户端预加载: {len(klines)} 条 K线")

        client = HistoricalBinanceClient()
        client.preload(klines)
        return client

    def _run_single_observation(
        self,
        engine: FastPricingEngine,
        hist_client: HistoricalBinanceClient,
        event_date: str,
        event_utc_ms: int,
        obs_minutes: int,
        now_ms: int,
        settlement: float,
        har_coeffs: HARCoefficients,
        fixed_strikes: Optional[List[float]] = None,
    ) -> ObservationResult:
        """运行单次观测（使用 FastPricingEngine）"""
        # 获取当前价格构建 K 网格
        hist_client.set_now(now_ms)
        s0 = hist_client.get_current_price()

        if fixed_strikes is not None:
            k_grid = fixed_strikes
        else:
            base = round(s0 / 500) * 500
            k_grid = [base + offset for offset in self.config.k_offsets]

        # 运行快速定价
        s0_result, probs = engine.compute_for_timestep(
            event_date=event_date,
            now_utc_ms=now_ms,
            k_list=k_grid,
            har_coeffs=har_coeffs,
        )

        # 提取预测概率
        predictions = probs

        # 查询 Polymarket 市场价格
        market_prices: Dict[float, float] = {}
        market_bid_ask: Dict[float, tuple] = {}
        if self._price_lookup is not None and self._poly_markets is not None:
            market_prices = self._price_lookup.get_market_prices_at(
                markets=self._poly_markets,
                event_date=event_date,
                timestamp_ms=now_ms,
                k_grid=k_grid,
                max_snap_diff=self.config.polymarket_max_snap_diff,
            )
            # 如果是 OrderbookPriceLookup，额外获取 bid/ask
            if hasattr(self._price_lookup, 'get_bid_ask_at'):
                market_bid_ask = self._price_lookup.get_bid_ask_at(
                    markets=self._poly_markets,
                    event_date=event_date,
                    timestamp_ms=now_ms,
                    k_grid=k_grid,
                    max_snap_diff=self.config.polymarket_max_snap_diff,
                )

        # 生成标签
        labels = {k: int(settlement > k) for k in k_grid}

        return ObservationResult(
            event_date=event_date,
            obs_minutes=obs_minutes,
            now_utc_ms=now_ms,
            s0=s0,
            settlement_price=settlement,
            k_grid=k_grid,
            predictions=predictions,
            labels=labels,
            market_prices=market_prices,
            market_bid_ask=market_bid_ask,
        )
