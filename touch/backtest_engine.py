"""
触碰障碍期权回测引擎

月度循环：遍历月内小时级时间步，使用解析公式定价一触碰概率
"""

import logging
import os
import time as time_mod
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

import numpy as np

from pricing_core.execution import compute_edge, shrink_probability
from pricing_core.time_utils import month_boundaries_utc_ms

from backtest.data_cache import KlineCache
from backtest.orderbook_reader import OrderbookPriceLookup
from backtest.polymarket_trades import PolymarketPriceLookup, PolymarketTradeCache

from .barrier_pricing import price_touch_barriers
from .iv_source import DeribitIVCache, DeribitIVSource
from .market_discovery import discover_touch_markets
from .models import TouchBacktestConfig, TouchMarketInfo, TouchObservationResult

logger = logging.getLogger(__name__)

# 一年的毫秒数（365.25 天）
_YEAR_MS = 365.25 * 86400 * 1000


class TouchBacktestEngine:
    """
    触碰障碍期权回测引擎

    IV 来源回退链（按优先级）:
    1. option_chain — 月底交割期权 ATM IV（最精确，但仅实时可用）
    2. dvol — Deribit DVOL 指数历史（30天滚动 ATM IV，回测可用）
    3. default_sigma — 固定默认值

    月度循环：
    1. 计算月份边界
    2. 下载/加载整月 K线
    3. 预计算 running_high / running_low（O(N) 单次遍历）
    4. 下载 IV 数据
    5. 发现 Polymarket 市场
    6. 主循环: 每 step_minutes 分钟观测一次
    7. 月末生成标签和汇总
    """

    def __init__(self, config: TouchBacktestConfig):
        self.config = config
        self.cache = KlineCache(cache_dir=config.cache_dir)
        self.iv_cache = DeribitIVCache(cache_dir=config.iv_cache_dir)

        # 实时 IV 源（option_chain 模式使用）
        self._iv_source: Optional[DeribitIVSource] = None
        # 缓存: 从期权链获取的 ATM IV（避免每个时间步都调 API）
        self._cached_atm_iv: Optional[float] = None
        self._cached_atm_iv_ms: int = 0
        # ATM IV 刷新间隔（毫秒），默认 1 小时
        self._atm_iv_refresh_ms: int = 3_600_000

        # Polymarket 市场
        self._touch_markets: Dict[float, TouchMarketInfo] = {}
        # price_lookup: OrderbookPriceLookup（优先）或 PolymarketPriceLookup
        self._price_lookup = None

        # K线数据（内存缓存）
        self._kline_times: List[int] = []     # open_time 列表
        self._kline_highs: List[float] = []   # high 列表
        self._kline_lows: List[float] = []    # low 列表
        self._kline_closes: List[float] = []  # close 列表

        # 预计算 running extremes（累积最大/最小值）
        self._running_highs: List[float] = []  # running_high[i] = max(highs[0..i])
        self._running_lows: List[float] = []   # running_low[i] = min(lows[0..i])

    def download_data(self) -> None:
        """下载回测所需的全部数据"""
        month_start_ms, month_end_ms = month_boundaries_utc_ms(self.config.month)

        # 将月份边界转换为日期范围
        from datetime import datetime
        from pricing_core.time_utils import UTC
        start_date = datetime.fromtimestamp(month_start_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
        # +1 天确保包含月末
        end_date = datetime.fromtimestamp(month_end_ms / 1000 + 86400, tz=UTC).strftime("%Y-%m-%d")

        logger.info(f"下载 K线数据: {start_date} ~ {end_date}")
        self.cache.ensure_range(start_date, end_date)

    def download_iv_data(self) -> None:
        """
        下载/初始化 IV 数据

        根据 iv_source 配置:
        - "option_chain": 初始化 DeribitIVSource（实时调用期权链取月底 ATM IV）
        - "dvol": 下载 DVOL 指数历史到缓存
        - 其他: 仅使用 default_sigma
        """
        month_start_ms, month_end_ms = month_boundaries_utc_ms(self.config.month)

        if self.config.iv_source == "option_chain":
            # 月底交割期权 ATM IV（最精确，但需要实时 API 访问）
            try:
                from pricing_core.deribit_data import DeribitClient
                client = DeribitClient()
                self._iv_source = DeribitIVSource(deribit_client=client)
                logger.info("IV 来源: 月底交割期权 ATM IV (实时)")
            except Exception as e:
                logger.warning(f"初始化期权链 IV 源失败: {e}，回退到 DVOL")
                self.config.iv_source = "dvol"

        if self.config.iv_source == "dvol":
            # DVOL 指数历史（30天滚动 ATM IV，回测用代理）
            # 注意: DVOL 是 30 天恒定到期 ATM IV 指数，与月底交割 IV 有 term structure 差异
            if self.iv_cache.load():
                logger.info("IV 来源: DVOL 缓存 (30天滚动 ATM IV)")
                return

            try:
                from pricing_core.deribit_data import DeribitClient
                client = DeribitClient()
                self.iv_cache.download_dvol_history(
                    currency="BTC",
                    start_ms=month_start_ms,
                    end_ms=month_end_ms,
                    deribit_client=client,
                )
                self.iv_cache.load()
            except Exception as e:
                logger.warning(f"下载 DVOL 历史失败: {e}，将使用默认 sigma={self.config.default_sigma}")
        else:
            logger.info(f"IV 来源: 默认 sigma={self.config.default_sigma}")

    def download_polymarket_data(self) -> None:
        """下载 Polymarket 市场发现 + 价格数据"""
        if not self.config.use_market_prices:
            logger.info("Polymarket 市场价格已禁用")
            return

        # 构造 slug
        from datetime import datetime
        year, month_num = map(int, self.config.month.split("-"))
        month_names = {
            1: "january", 2: "february", 3: "march", 4: "april",
            5: "may", 6: "june", 7: "july", 8: "august",
            9: "september", 10: "october", 11: "november", 12: "december",
        }
        month_name = month_names[month_num]
        slug = f"what-price-will-bitcoin-hit-in-{month_name}-{year}"

        cache_path = os.path.join(
            self.config.polymarket_cache_dir, f"touch_discovery_{self.config.month}.json"
        )

        self._touch_markets = discover_touch_markets(
            slug=slug,
            gamma_api=self.config.polymarket_gamma_api,
            cache_path=cache_path,
        )

        if not self._touch_markets:
            logger.warning("未发现触碰合约市场")
            return

        cids = [info.condition_id for info in self._touch_markets.values()]

        # 优先使用 Dome 订单簿缓存（已结算市场 CLOB 无数据）
        ob_lookup = OrderbookPriceLookup(
            cache_dir=self.config.orderbook_cache_dir
        )
        ob_lookup.preload(cids)
        loaded = sum(1 for c in cids if c in ob_lookup._data)

        if loaded > 0:
            logger.info(f"使用 Dome 订单簿: {loaded}/{len(cids)} 个合约有数据")
            self._price_lookup = ob_lookup
            return

        # 回退: CLOB 交易数据
        logger.info("订单簿无数据，回退到 CLOB 交易记录")
        trade_cache = PolymarketTradeCache(
            cache_dir=self.config.polymarket_cache_dir,
            clob_api=self.config.polymarket_clob_api,
        )

        from backtest.polymarket_discovery import PolymarketMarketInfo
        compat_markets: Dict[Tuple[str, float], PolymarketMarketInfo] = {}
        for barrier, info in self._touch_markets.items():
            compat_info = PolymarketMarketInfo(
                event_date=self.config.month,
                strike=barrier,
                condition_id=info.condition_id,
                yes_token_id=info.yes_token_id,
                no_token_id=info.no_token_id,
                question=info.question,
            )
            compat_markets[(self.config.month, barrier)] = compat_info

        trade_cache.ensure_trades(compat_markets)
        self._price_lookup = PolymarketPriceLookup(trade_cache)
        self._price_lookup.preload(cids)

    def _load_klines(self) -> None:
        """加载月份内的全部 K线数据到内存"""
        month_start_ms, month_end_ms = month_boundaries_utc_ms(self.config.month)
        klines = self.cache.load_range_ms(month_start_ms, month_end_ms)

        self._kline_times = [k.open_time for k in klines]
        self._kline_highs = [k.high for k in klines]
        self._kline_lows = [k.low for k in klines]
        self._kline_closes = [k.close for k in klines]

        logger.info(f"加载 K线: {len(klines)} 条 ({self.config.month})")

    def _precompute_running_extremes(self) -> None:
        """
        预计算 running_high 和 running_low（O(N) 单次遍历）

        使用 kline.high / kline.low（非 close）以捕获分钟内极值
        """
        n = len(self._kline_highs)
        if n == 0:
            self._running_highs = []
            self._running_lows = []
            return

        self._running_highs = [0.0] * n
        self._running_lows = [0.0] * n

        self._running_highs[0] = self._kline_highs[0]
        self._running_lows[0] = self._kline_lows[0]

        for i in range(1, n):
            self._running_highs[i] = max(self._running_highs[i - 1], self._kline_highs[i])
            self._running_lows[i] = min(self._running_lows[i - 1], self._kline_lows[i])

        logger.debug(
            f"Running extremes: high={self._running_highs[-1]:.2f}, "
            f"low={self._running_lows[-1]:.2f}"
        )

    def _get_kline_index_at(self, timestamp_ms: int) -> int:
        """
        获取 <= timestamp_ms 的最后一个 kline 索引

        使用 bisect_right 防止前瞻偏差
        """
        idx = bisect_right(self._kline_times, timestamp_ms) - 1
        return max(0, idx)

    def _get_s0_at(self, timestamp_ms: int) -> float:
        """获取指定时刻的 BTC 价格（close）"""
        idx = self._get_kline_index_at(timestamp_ms)
        return self._kline_closes[idx]

    def _get_running_extremes_at(self, timestamp_ms: int) -> Tuple[float, float]:
        """获取截至指定时刻的 running_high 和 running_low"""
        idx = self._get_kline_index_at(timestamp_ms)
        return self._running_highs[idx], self._running_lows[idx]

    def _apply_term_structure_correction(
        self, dvol: float, timestamp_ms: int
    ) -> float:
        """
        对 DVOL (30天恒定到期) 做期限结构校正

        DVOL 反映 30 天 ATM IV，但触碰合约的剩余期限随月内推进不断缩短。
        BTC 通常呈温和反向期限结构: 短期 IV > 长期 IV。

        校正公式: sigma_adj = dvol * (30 / T_days_remaining)^alpha
        - alpha=0: 不校正
        - alpha=0.05: BTC 温和反向期限结构 (7天时 +7.5%, 3天时 +12%)
        - alpha=0.10: 较陡的反向期限结构 (7天时 +15.5%, 3天时 +26%)

        Args:
            dvol: DVOL 值 (小数形式，如 0.50)
            timestamp_ms: 当前时刻 (UTC ms)

        Returns:
            校正后的 sigma (小数形式)
        """
        alpha = self.config.term_structure_alpha
        if alpha == 0.0:
            return dvol

        _, month_end_ms = month_boundaries_utc_ms(self.config.month)
        T_remaining_ms = max(month_end_ms - timestamp_ms, 60_000)  # 至少 1 分钟
        T_days = T_remaining_ms / (86_400 * 1000)

        # clamp: 剩余不足 0.5 天时不再放大（避免极端值）
        T_days = max(T_days, 0.5)

        factor = (30.0 / T_days) ** alpha
        sigma_adj = dvol * factor
        return sigma_adj

    def _get_sigma_at(self, timestamp_ms: int) -> float:
        """
        获取指定时刻的波动率

        回退链（按优先级）:
        1. option_chain → 月底交割期权 ATM IV（最精确，需实时 API，无需校正）
        2. dvol → DVOL 缓存 + 期限结构校正（30天 ATM IV 代理）
        3. default_sigma → 固定默认值

        所有 IV 源输出乘以 vrp_k 进行 Q→P 缩放
        """
        # 1. 月底交割期权 ATM IV（实时，已是正确期限，不需要校正）
        if self.config.iv_source == "option_chain" and self._iv_source is not None:
            # 带缓存：每 _atm_iv_refresh_ms 刷新一次，避免频繁调 API
            if (self._cached_atm_iv is None
                    or timestamp_ms - self._cached_atm_iv_ms > self._atm_iv_refresh_ms):
                _, month_end_ms = month_boundaries_utc_ms(self.config.month)
                iv = self._iv_source.get_atm_iv(target_expiry_ms=month_end_ms)
                if iv is not None:
                    self._cached_atm_iv = iv
                    self._cached_atm_iv_ms = timestamp_ms
                    logger.debug(f"期权链 ATM IV 刷新: {iv:.4f}")

            if self._cached_atm_iv is not None:
                return self._cached_atm_iv * self.config.vrp_k

            # option_chain 失败 → 回退到 DVOL
            logger.debug("期权链 ATM IV 不可用，回退 DVOL")

        # 2. DVOL 缓存 + 期限结构校正
        iv = self.iv_cache.get_iv_at(timestamp_ms)
        if iv is not None:
            iv = self._apply_term_structure_correction(iv, timestamp_ms)
            return iv * self.config.vrp_k

        # 3. 默认值
        return self.config.default_sigma

    def _get_market_price(self, barrier: float, timestamp_ms: int) -> Optional[float]:
        """获取指定 barrier 在指定时刻的 Polymarket 价格"""
        if self._price_lookup is None:
            return None

        info = self._touch_markets.get(barrier)
        if info is None:
            return None

        return self._price_lookup.get_price_at(info.condition_id, timestamp_ms)

    def run(self) -> List[TouchObservationResult]:
        """
        执行完整回测

        Returns:
            List[TouchObservationResult] 所有观测结果
        """
        t_start = time_mod.monotonic()

        month_start_ms, month_end_ms = month_boundaries_utc_ms(self.config.month)

        # 1. 加载 K线数据（如果尚未注入）
        if not self._kline_times:
            self._load_klines()
        if not self._kline_times:
            logger.error(f"无 K线数据 ({self.config.month})，退出")
            return []

        # 2. 预计算 running extremes（如果尚未预计算）
        if not self._running_highs:
            self._precompute_running_extremes()

        # 3. 加载 IV 数据
        # 优先加载 ATM IV 缓存（iv_collector 采集），再加载 DVOL 缓存
        # ATM IV 数据会合并覆盖同期 DVOL（更精确）
        self.iv_cache.load()
        self.iv_cache.load_atm_iv(self.config.month)

        # 4. 确定 barriers
        if self._touch_markets:
            barriers = sorted(self._touch_markets.keys())
        else:
            logger.warning("无触碰市场数据，使用默认 barrier 网格")
            # 默认 barrier: 从当前价格按 5000 间隔
            s0 = self._kline_closes[0]
            base = round(s0 / 5000) * 5000
            barriers = [base + offset for offset in [-10000, -5000, 0, 5000, 10000]]

        logger.info(f"回测 barriers: {barriers}")

        # 5. 主循环
        step_ms = self.config.step_minutes * 60_000
        current_ms = month_start_ms
        observations: List[TouchObservationResult] = []
        obs_count = 0

        while current_ms <= month_end_ms:
            # 检查是否有 K线数据覆盖
            if current_ms > self._kline_times[-1]:
                break

            # 获取当前状态
            s0 = self._get_s0_at(current_ms)
            running_high, running_low = self._get_running_extremes_at(current_ms)
            sigma = self._get_sigma_at(current_ms)

            # 计算剩余时间（年）
            T_remaining_ms = month_end_ms - current_ms
            T_remaining_years = max(T_remaining_ms / _YEAR_MS, 0.0)

            # 定价
            strike_results = price_touch_barriers(
                s0=s0,
                barriers=barriers,
                sigma=sigma,
                T=T_remaining_years,
                mu=self.config.mu,
                running_high=running_high,
                running_low=running_low,
            )

            # 收集结果
            predictions: Dict[float, float] = {}
            already_touched: Dict[float, bool] = {}
            market_prices: Dict[float, float] = {}

            for sr in strike_results:
                predictions[sr.barrier] = sr.p_touch
                already_touched[sr.barrier] = sr.already_touched

                # 查询市场价格 + 计算 edge
                if self.config.use_market_prices:
                    mp = self._get_market_price(sr.barrier, current_ms)
                    if mp is not None:
                        market_prices[sr.barrier] = mp
                        p_trade = shrink_probability(
                            sr.p_touch, mp, self.config.shrinkage_lambda
                        )
                        sr.p_trade = p_trade
                        sr.edge = compute_edge(p_trade, mp)

            # 生成最终标签（月末 running extremes 判断）
            final_idx = len(self._kline_times) - 1
            final_high = self._running_highs[final_idx]
            final_low = self._running_lows[final_idx]
            labels: Dict[float, int] = {}
            for barrier in barriers:
                if barrier > s0:
                    # 上触碰
                    labels[barrier] = 1 if final_high >= barrier else 0
                else:
                    # 下触碰
                    labels[barrier] = 1 if final_low <= barrier else 0

            obs = TouchObservationResult(
                obs_utc_ms=current_ms,
                s0=s0,
                running_high=running_high,
                running_low=running_low,
                T_remaining_years=T_remaining_years,
                sigma=sigma,
                barriers=barriers,
                predictions=predictions,
                labels=labels,
                market_prices=market_prices,
                already_touched=already_touched,
            )
            observations.append(obs)
            obs_count += 1

            current_ms += step_ms

        elapsed = time_mod.monotonic() - t_start
        logger.info(
            f"回测完成: {obs_count} 观测, "
            f"{len(barriers)} barriers, "
            f"耗时 {elapsed:.1f}s"
        )

        return observations

    def compute_metrics(
        self, observations: List[TouchObservationResult]
    ) -> Dict:
        """
        计算回测指标

        Returns:
            包含 brier_score、各 barrier 的 pnl 等
        """
        from backtest.metrics import brier_score as compute_brier

        if not observations:
            return {}

        # 收集所有预测和标签
        all_preds = []
        all_labels = []
        for obs in observations:
            for barrier in obs.barriers:
                if barrier in obs.predictions and barrier in obs.labels:
                    all_preds.append(obs.predictions[barrier])
                    all_labels.append(obs.labels[barrier])

        preds = np.array(all_preds)
        labels = np.array(all_labels)

        result = {
            "n_observations": len(observations),
            "n_predictions": len(preds),
        }

        if len(preds) > 0:
            result["brier_score"] = float(compute_brier(preds, labels))

        # 按 barrier 分组
        barrier_metrics: Dict[float, Dict] = {}
        for obs in observations:
            for barrier in obs.barriers:
                if barrier not in barrier_metrics:
                    barrier_metrics[barrier] = {
                        "preds": [],
                        "labels": [],
                        "edges": [],
                    }
                if barrier in obs.predictions and barrier in obs.labels:
                    barrier_metrics[barrier]["preds"].append(obs.predictions[barrier])
                    barrier_metrics[barrier]["labels"].append(obs.labels[barrier])
                if barrier in obs.market_prices and barrier in obs.predictions:
                    edge = obs.predictions[barrier] - obs.market_prices[barrier]
                    barrier_metrics[barrier]["edges"].append(edge)

        per_barrier = {}
        for barrier, data in barrier_metrics.items():
            bp = np.array(data["preds"])
            bl = np.array(data["labels"])
            info = {
                "n_obs": len(bp),
                "label": int(bl[0]) if len(bl) > 0 else -1,
                "mean_pred": float(np.mean(bp)) if len(bp) > 0 else 0.0,
            }
            if len(bp) > 0:
                info["brier_score"] = float(compute_brier(bp, bl))
            if data["edges"]:
                info["mean_edge"] = float(np.mean(data["edges"]))
            per_barrier[barrier] = info

        result["per_barrier"] = per_barrier
        return result
