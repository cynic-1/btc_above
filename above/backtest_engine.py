"""
Above 合约回测引擎

日历日循环: 遍历 [start_date, end_date) 每天的 ET noon 事件，
从 event_utc - lookback_hours 到 event_utc，按 step_minutes 步进观测。
"""

import logging
import os
import pickle
import time as time_mod
from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from pricing_core.execution import compute_edge, shrink_probability
from pricing_core.time_utils import (
    et_noon_to_utc_ms,
    utc_ms_to_binance_kline_open,
)

from backtest.data_cache import KlineCache, _date_range
from backtest.hybrid_lookup import HybridPriceLookup
from backtest.orderbook_reader import OrderbookPriceLookup, load_markets_from_events_json
from backtest.polymarket_discovery import (
    PolymarketMarketInfo,
    discover_markets_for_range,
)
from backtest.polymarket_trades import PolymarketPriceLookup, PolymarketTradeCache
from touch.iv_source import DeribitIVCache

from .dvol_pricing import price_above_strikes
from .models import AboveBacktestConfig, AboveObservation

logger = logging.getLogger(__name__)

# 一年的毫秒数（与 touch/backtest_engine.py 一致）
_YEAR_MS = 365.25 * 86400 * 1000


class AboveBacktestEngine:
    """
    Above 合约回测引擎

    日历日循环:
    1. 遍历日期范围内每天
    2. 计算 ET noon → UTC 事件时间
    3. 从 event_utc - lookback_hours 到 event_utc 按 step_minutes 步进
    4. 每步: 获取 S0、sigma、定价、查询市场价、计算 edge
    5. 结算: kline close at utc_ms_to_binance_kline_open(et_noon_to_utc_ms(date))
    6. 标签: 1 if settlement > K else 0
    """

    def __init__(self, config: AboveBacktestConfig):
        self.config = config
        self.cache = KlineCache(cache_dir=config.cache_dir)

        self.iv_cache = DeribitIVCache(cache_dir=config.iv_cache_dir)

        # Polymarket 市场: {(event_date, strike): PolymarketMarketInfo}
        self._markets: Dict[Tuple[str, float], PolymarketMarketInfo] = {}
        self._price_lookup: Optional[HybridPriceLookup] = None

        # K线内存缓存（按需加载）
        self._kline_times: List[int] = []
        self._kline_closes: List[float] = []

    # ------------------------------------------------------------------
    # 数据下载
    # ------------------------------------------------------------------

    def download_data(self) -> None:
        """下载回测所需的 K线数据"""
        # 需要 lookback 之前的数据
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        adjusted_start = start - timedelta(hours=self.config.lookback_hours + 24)
        adjusted_start_str = adjusted_start.strftime("%Y-%m-%d")

        # 包含 end_date 当天
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        adjusted_end_str = (end + timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"下载 K线数据: {adjusted_start_str} ~ {adjusted_end_str}")
        self.cache.ensure_range(adjusted_start_str, adjusted_end_str)

    def download_iv_data(self) -> None:
        """下载/加载 DVOL 历史数据"""
        if self.config.iv_source == "default":
            logger.info(f"IV 来源: 默认 sigma={self.config.default_sigma}")
            return

        loaded = self.iv_cache.load(currency=self.config.symbol)
        if loaded:
            logger.info("IV 来源: DVOL 缓存")
            return

        # 尝试下载
        try:
            from pricing_core.deribit_data import DeribitClient
            client = DeribitClient()

            start_ms = et_noon_to_utc_ms(self.config.start_date)
            end_ms = et_noon_to_utc_ms(self.config.end_date)
            self.iv_cache.ensure_range(
                currency=self.config.symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                deribit_client=client,
            )
            self.iv_cache.load(currency=self.config.symbol)
        except Exception as e:
            logger.warning(
                f"下载 DVOL 历史失败: {e}，将使用 default_sigma={self.config.default_sigma}"
            )

    def download_polymarket_data(self) -> None:
        """下载 Polymarket 市场发现 + 价格数据

        优先从 events.json 加载（完整历史数据），回退到 Gamma API 搜索。
        与 backtest/engine.py 的 BacktestEngine.download_polymarket_data 一致。
        """
        if not self.config.use_market_prices:
            logger.info("Polymarket 市场价格已禁用")
            return

        # 1. 优先从 events.json 加载（完整历史，包含已结算合约）
        events_json = self.config.orderbook_events_json
        if os.path.exists(events_json):
            try:
                self._markets = load_markets_from_events_json(events_json)
                logger.info(f"从 events.json 加载: {len(self._markets)} 个合约")
            except Exception as e:
                logger.warning(f"加载 events.json 失败: {e}")

        # 2. 回退到 Gamma API 搜索
        if not self._markets:
            cache_path = os.path.join(
                self.config.polymarket_cache_dir, "discovery_cache.json",
            )
            self._markets = discover_markets_for_range(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                gamma_api=self.config.polymarket_gamma_api,
                cache_path=cache_path,
            )

        if not self._markets:
            logger.warning("未发现 above 合约市场")
            return

        cids = list({info.condition_id for info in self._markets.values()})

        # 加载 Dome 订单簿
        ob_lookup = OrderbookPriceLookup(
            cache_dir=self.config.orderbook_cache_dir,
            max_stale_ms=int(self.config.orderbook_max_stale_minutes * 60_000),
            max_spread=self.config.orderbook_max_spread,
            zero_side_policy=self.config.orderbook_zero_side_policy,
        )
        ob_lookup.preload(cids)
        ob_loaded = sum(1 for c in cids if c in ob_lookup._data)

        # 加载 CLOB 交易数据
        trade_cache = PolymarketTradeCache(
            cache_dir=self.config.polymarket_cache_dir,
        )
        trade_cache.ensure_trades(self._markets)

        trade_lookup = PolymarketPriceLookup(trade_cache)
        trade_lookup.preload(cids)
        trade_loaded = sum(1 for c in cids if trade_lookup._data.get(c))

        # 混合价格源
        self._price_lookup = HybridPriceLookup(
            orderbook=ob_lookup,
            clob=trade_lookup,
            staleness_threshold_ms=int(self.config.orderbook_max_stale_minutes * 60_000),
        )
        logger.info(
            f"价格源: Hybrid (Dome {ob_loaded}/{len(cids)}, "
            f"Trades {trade_loaded}/{len(cids)})"
        )

    # ------------------------------------------------------------------
    # K线数据访问
    # ------------------------------------------------------------------

    def _load_klines_for_range(self, start_ms: int, end_ms: int) -> None:
        """加载指定毫秒范围的 K线数据到内存"""
        klines = self.cache.load_range_ms(start_ms, end_ms)
        self._kline_times = [k.open_time for k in klines]
        self._kline_closes = [k.close for k in klines]
        logger.debug(f"加载 K线: {len(klines)} 条")

    def _get_s0_at(self, timestamp_ms: int) -> Optional[float]:
        """获取 <= timestamp_ms 的最近 close 价格"""
        if not self._kline_times:
            return None
        idx = bisect_right(self._kline_times, timestamp_ms) - 1
        if idx < 0:
            return None
        return self._kline_closes[idx]

    def _get_settlement_close(self, event_utc_ms: int) -> Optional[float]:
        """获取结算时刻的 kline close 价格"""
        kline_open = utc_ms_to_binance_kline_open(event_utc_ms)
        # 精确匹配: 在已加载数据中查找
        if not self._kline_times:
            return None
        idx = bisect_right(self._kline_times, kline_open) - 1
        if idx < 0:
            return None
        # 检查是否精确匹配
        if self._kline_times[idx] == kline_open:
            return self._kline_closes[idx]
        # 容差 1 分钟内
        if abs(self._kline_times[idx] - kline_open) <= 60_000:
            return self._kline_closes[idx]
        return None

    # ------------------------------------------------------------------
    # IV 数据访问
    # ------------------------------------------------------------------

    def _get_sigma_at(self, timestamp_ms: int) -> float:
        """
        获取指定时刻的波动率

        回退链: dvol 缓存 → default_sigma
        所有源输出乘以 vrp_k
        """
        if self.config.iv_source == "dvol":
            iv = self.iv_cache.get_iv_at(timestamp_ms)
            if iv is not None:
                return iv * self.config.vrp_k

        return self.config.default_sigma

    # ------------------------------------------------------------------
    # 市场价格访问
    # ------------------------------------------------------------------

    def _get_market_price(
        self, event_date: str, strike: float, timestamp_ms: int
    ) -> Optional[float]:
        """获取指定 strike 在指定时刻的 Polymarket 价格"""
        if self._price_lookup is None:
            return None

        info = self._markets.get((event_date, strike))
        if info is None:
            return None

        return self._price_lookup.get_price_at(info.condition_id, timestamp_ms)

    # ------------------------------------------------------------------
    # K grid
    # ------------------------------------------------------------------

    def _build_k_grid(self, event_date: str, s0: float) -> List[float]:
        """
        构建行权价网格

        优先从 Polymarket 发现中获取，回退到固定偏移网格
        """
        # 从 Polymarket 市场获取该日期的行权价
        if self.config.use_market_prices and self._markets:
            pm_strikes = sorted(
                strike
                for (d, strike) in self._markets
                if d == event_date
            )
            if pm_strikes:
                return pm_strikes

        # 固定偏移网格
        if self.config.use_fixed_strikes:
            base = round(s0 / 500) * 500
            return [base + offset for offset in self.config.k_offsets]

        return []

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self) -> List[AboveObservation]:
        """
        执行完整回测

        Returns:
            List[AboveObservation] 所有观测结果
        """
        t_start = time_mod.monotonic()

        dates = _date_range(self.config.start_date, self.config.end_date)
        if not dates:
            logger.error("日期范围为空")
            return []

        # 加载 IV 数据
        self.iv_cache.load(currency=self.config.symbol)

        # 计算全局 K线加载范围
        first_event_ms = et_noon_to_utc_ms(dates[0])
        last_event_ms = et_noon_to_utc_ms(dates[-1])
        load_start_ms = first_event_ms - self.config.lookback_hours * 3_600_000
        load_end_ms = last_event_ms + 60_000  # 多加 1 分钟确保包含结算
        self._load_klines_for_range(load_start_ms, load_end_ms)

        if not self._kline_times:
            logger.error("无 K线数据，退出")
            return []

        step_ms = self.config.step_minutes * 60_000
        lookback_ms = self.config.lookback_hours * 3_600_000
        all_observations: List[AboveObservation] = []

        for date_str in dates:
            event_utc_ms = et_noon_to_utc_ms(date_str)
            obs_start_ms = event_utc_ms - lookback_ms

            # 获取 S0 确定 K grid
            s0_initial = self._get_s0_at(obs_start_ms)
            if s0_initial is None:
                logger.warning(f"[{date_str}] 无法获取初始价格，跳过")
                continue

            k_grid = self._build_k_grid(date_str, s0_initial)
            if not k_grid:
                logger.warning(f"[{date_str}] K grid 为空，跳过")
                continue

            # 获取结算价
            settlement_price = self._get_settlement_close(event_utc_ms)
            if settlement_price is None:
                logger.warning(f"[{date_str}] 结算价不可用，跳过标签")

            # 生成标签
            labels: Dict[float, int] = {}
            if settlement_price is not None:
                for K in k_grid:
                    labels[K] = 1 if settlement_price > K else 0

            # 内层循环: 步进观测
            current_ms = obs_start_ms
            while current_ms <= event_utc_ms:
                s0 = self._get_s0_at(current_ms)
                if s0 is None:
                    current_ms += step_ms
                    continue

                sigma = self._get_sigma_at(current_ms)
                T_remaining_ms = event_utc_ms - current_ms
                T_years = max(T_remaining_ms / _YEAR_MS, 0.0)

                # 定价
                strike_results = price_above_strikes(
                    s0=s0,
                    k_grid=k_grid,
                    sigma=sigma,
                    T=T_years,
                    mu=self.config.mu,
                )

                # 收集结果
                predictions: Dict[float, float] = {}
                market_prices: Dict[float, float] = {}

                for sr in strike_results:
                    predictions[sr.strike] = sr.p_above

                    # 查询市场价格 + 计算 edge
                    if self.config.use_market_prices:
                        mp = self._get_market_price(
                            date_str, sr.strike, current_ms
                        )
                        if mp is not None:
                            market_prices[sr.strike] = mp
                            p_trade = shrink_probability(
                                sr.p_above, mp, self.config.shrinkage_lambda
                            )
                            sr.p_trade = p_trade
                            sr.edge = compute_edge(p_trade, mp)

                obs = AboveObservation(
                    event_date=date_str,
                    obs_utc_ms=current_ms,
                    event_utc_ms=event_utc_ms,
                    s0=s0,
                    sigma=sigma,
                    T_years=T_years,
                    settlement_price=settlement_price or 0.0,
                    k_grid=k_grid,
                    predictions=predictions,
                    labels=labels,
                    market_prices=market_prices,
                )
                all_observations.append(obs)
                current_ms += step_ms

        elapsed = time_mod.monotonic() - t_start
        n_days = len(set(obs.event_date for obs in all_observations))
        logger.info(
            f"回测完成: {len(all_observations)} 观测, "
            f"{n_days} 天, 耗时 {elapsed:.1f}s"
        )
        return all_observations

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_observations(
        observations: List[AboveObservation],
        output_dir: str,
        tag: str = "above",
    ) -> str:
        """序列化观测结果到 pkl 文件"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"observations_{tag}.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {"version": 1, "tag": tag, "observations": observations}, f
            )
        logger.info(f"观测序列化: {len(observations)} 条 -> {path}")
        return path

    @staticmethod
    def load_observations(path: str) -> List[AboveObservation]:
        """从 pkl 文件加载观测结果"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["observations"]
