"""
模型定价 vs Polymarket 真实价格对比图生成引擎

优化策略：
- 跳过 bootstrap CI（图表不需要）
- 缓存 fit_student_t 每 dist_refit_minutes 重拟合一次
- 降 MC 采样到 2000（标准误 ~1.1%，图表级精度足够）
- 同一 event_date 的所有 strike 共享 ST 样本
- 直接调用底层函数，跳过 PricingPipeline 封装
"""

import csv
import gzip
import logging
import os
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from pricing_core.binance_data import Kline
from pricing_core.config import PricingConfig
from pricing_core.distribution import fit_student_t
from pricing_core.models import (
    BasisParams,
    DistParams,
    HARCoefficients,
)
from pricing_core.pricing import prob_above_K_analytical_batch
from pricing_core.time_utils import et_noon_to_utc_ms, minutes_until_event, utc_ms_to_binance_kline_open
from pricing_core.vol_forecast import (
    compute_hourly_rv_profile,
    compute_log_returns,
    get_path_hours,
    har_features,
    har_predict,
    intraday_seasonality_factor,
)

from .data_cache import KlineCache, _date_range, _date_to_utc_ms
from .har_trainer import HARTrainer
from .historical_client import HistoricalBinanceClient
from .hybrid_lookup import HybridPriceLookup
from .orderbook_reader import OrderbookPriceLookup, load_markets_from_events_json
from .polymarket_discovery import PolymarketMarketInfo, discover_markets_for_range
from .polymarket_trades import PolymarketPriceLookup, PolymarketTradeCache

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """图表生成配置"""
    start_date: str = "2026-01-01"
    end_date: str = "2026-03-01"
    cache_dir: str = "data/klines"
    polymarket_cache_dir: str = "data/polymarket"
    output_dir: str = "charts"
    step_minutes: int = 1          # 时间分辨率
    lookback_hours: int = 24       # 事件前多少小时开始
    mc_samples: int = 2000         # 图表用低精度 MC
    dist_refit_minutes: int = 30   # Student-t 重拟合间隔
    har_train_days: int = 30
    har_retrain_interval: int = 7
    har_ridge_alpha: float = 0.01
    polymarket_gamma_api: str = "https://gamma-api.polymarket.com"
    polymarket_clob_api: str = "https://clob.polymarket.com"
    # 订单簿配置
    use_orderbook: bool = True
    orderbook_cache_dir: str = "data/orderbook_cache"
    orderbook_events_json: str = "data/btc_above_events.json"

    # 订单簿过期阈值（小时），超过后回退 CLOB
    orderbook_staleness_hours: float = 2.0


class FastPricingEngine:
    """
    轻量定价引擎，为图表生成优化

    直接调用底层函数，跳过 PricingPipeline 的封装：
    - 复用 pipeline.py 的计算逻辑
    - 跳过: bootstrap CI, trade signal, 详细日志
    - 缓存 dist_params 每 dist_refit_minutes 重拟合
    """

    def __init__(self, hist_client: HistoricalBinanceClient, config: ChartConfig, vrp_k: float = 1.0):
        self._client = hist_client
        self._config = config
        self._vrp_k = vrp_k
        # 分布参数缓存
        self._cached_dist_params: Optional[DistParams] = None
        self._cached_dist_time_ms: int = 0

    def reset_dist_cache(self) -> None:
        """新 event_date 时重置"""
        self._cached_dist_params = None
        self._cached_dist_time_ms = 0

    def compute_for_timestep(
        self,
        event_date: str,
        now_utc_ms: int,
        k_list: List[float],
        har_coeffs: HARCoefficients,
    ) -> Tuple[float, Dict[float, float]]:
        """
        单时间步定价，返回 (s0, {strike: p_physical})

        简化版 pipeline.py 逻辑，无 CI / trade signal
        """
        # 1. 设置时间、获取当前价
        self._client.set_now(now_utc_ms)
        s0 = self._client.get_current_price()

        # 2. 获取 24h+1min K线（多取 1 条，确保差分后 returns >= 1440）
        lookback_ms = 24 * 60 * 60 * 1000 + 60_000
        klines = self._client.get_klines_extended(
            start_ms=now_utc_ms - lookback_ms,
            end_ms=now_utc_ms,
        )
        if len(klines) < 60:
            return s0, {}

        prices = self._client.get_close_prices(klines)
        returns = compute_log_returns(prices)
        timestamps = np.array([k.open_time for k in klines[1:]])

        # 3. HAR-RV 预测
        features = har_features(returns)
        rv_hat = har_predict(features, har_coeffs)

        # 4. 日内季节性校正
        event_utc_ms = et_noon_to_utc_ms(event_date)
        hourly_profile = compute_hourly_rv_profile(returns, timestamps)
        now_hour = int((now_utc_ms / 1000) % 86400) // 3600
        event_hour = int((event_utc_ms / 1000) % 86400) // 3600
        path_hours = get_path_hours(now_hour, event_hour)
        seasonality = intraday_seasonality_factor(hourly_profile, path_hours)
        rv_hat_adj = rv_hat * seasonality

        # 5. VRP 缩放
        rv_hat_vrp = rv_hat_adj * (self._vrp_k ** 2)

        # 6. 按剩余时间缩放方差（HAR 预测 horizon=360min）
        mins_to_expiry = minutes_until_event(now_utc_ms, event_utc_ms)
        _HAR_FORWARD_HORIZON = 360
        tau_scale = max(min(mins_to_expiry, _HAR_FORWARD_HORIZON), 0.0) / _HAR_FORWARD_HORIZON
        rv_hat_final = rv_hat_vrp * tau_scale

        # 7. 基差参数（默认值）
        basis_params = BasisParams(mu_b=0.0, sigma_b=0.0)

        # 8. 分布拟合（带缓存）
        need_refit = (
            self._cached_dist_params is None
            or (now_utc_ms - self._cached_dist_time_ms) > self._config.dist_refit_minutes * 60_000
        )
        if need_refit:
            rv_rolling = np.sqrt(np.convolve(returns ** 2, np.ones(30) / 30, mode='valid'))
            if len(rv_rolling) > 30:
                z_samples = returns[29:29 + len(rv_rolling)] / np.maximum(rv_rolling, 1e-12)
                self._cached_dist_params = fit_student_t(z_samples)
            else:
                self._cached_dist_params = DistParams(df=5.0, loc=0.0, scale=1.0)
            self._cached_dist_time_ms = now_utc_ms

        # 9. Student-t 解析定价
        probs = prob_above_K_analytical_batch(s0, k_list, rv_hat_final, self._cached_dist_params)

        return s0, dict(zip(k_list, probs))


class ChartGenerator:
    """
    按日期批处理生成模型定价 vs Polymarket 对比图表

    流程：
    1. 构建 HistoricalBinanceClient
    2. 加载 Polymarket discovery + price lookup
    3. 按 event_date 遍历，逐时间步定价 + 查询市场价格
    4. 输出 CSV + PNG
    """

    def __init__(self, config: ChartConfig):
        self.config = config
        self.cache = KlineCache(cache_dir=config.cache_dir)
        self.trainer = HARTrainer(
            cache=self.cache,
            train_days=config.har_train_days,
            retrain_interval=config.har_retrain_interval,
            ridge_alpha=config.har_ridge_alpha,
        )
        self._poly_markets: Optional[Dict[Tuple[str, float], PolymarketMarketInfo]] = None
        self._price_lookup = None  # PolymarketPriceLookup | OrderbookPriceLookup

    def _load_polymarket_data(self) -> None:
        """加载 Polymarket 市场发现 + 价格数据（混合: orderbook + CLOB 回退）"""
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
                    logger.info(f"图表引擎: 订单簿初始化 ({len(npz_files)} 个 npz)")

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
            logger.info("图表引擎: 使用混合价格源 (orderbook + CLOB 回退)")
        else:
            self._price_lookup = clob_lookup
            logger.info("图表引擎: 使用 CLOB 价格历史")

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

    def _get_date_strikes(self, event_date: str) -> List[float]:
        """获取某日期所有有 Polymarket 数据的 strike"""
        if self._poly_markets is None:
            return []
        strikes = []
        for (d, strike), info in self._poly_markets.items():
            if d == event_date:
                strikes.append(strike)
        return sorted(strikes)

    def _get_market_price_at(
        self, event_date: str, strike: float, timestamp_ms: int
    ) -> Optional[float]:
        """查询某 strike 在某时刻的 Polymarket mid 价格"""
        if self._price_lookup is None or self._poly_markets is None:
            return None
        info = self._poly_markets.get((event_date, strike))
        if info is None:
            return None
        return self._price_lookup.get_price_at(info.condition_id, timestamp_ms)

    def _get_bid_ask_at(
        self, event_date: str, strike: float, timestamp_ms: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """查询某 strike 在某时刻的 best_bid / best_ask"""
        if self._price_lookup is None or self._poly_markets is None:
            return None, None
        info = self._poly_markets.get((event_date, strike))
        if info is None:
            return None, None
        # OrderbookPriceLookup 有 get_quote_at
        if hasattr(self._price_lookup, 'get_quote_at'):
            quote = self._price_lookup.get_quote_at(info.condition_id, timestamp_ms)
            if quote is None:
                return None, None
            return quote.best_bid, quote.best_ask
        # 回退: 仅有 mid 时用 mid 同时当 bid/ask
        mid = self._price_lookup.get_price_at(info.condition_id, timestamp_ms)
        return mid, mid

    def _get_first_trade_time(self, event_date: str, strike: float) -> Optional[int]:
        """获取某合约第一笔交易/报价的时间戳"""
        if self._price_lookup is None or self._poly_markets is None:
            return None
        info = self._poly_markets.get((event_date, strike))
        if info is None:
            return None
        # OrderbookPriceLookup 提供 get_first_timestamp 方法
        if hasattr(self._price_lookup, 'get_first_timestamp'):
            return self._price_lookup.get_first_timestamp(info.condition_id)
        # 回退: PolymarketPriceLookup 内部数据格式
        data = self._price_lookup._data.get(info.condition_id, [])
        if not data:
            return None
        return data[0][0]

    def run(
        self,
        filter_event_date: Optional[str] = None,
        filter_strike: Optional[float] = None,
    ) -> None:
        """
        生成全部对比图表

        Args:
            filter_event_date: 只生成特定日期
            filter_strike: 只生成特定 strike
        """
        t_start = time_mod.monotonic()

        # 加载数据
        logger.info("加载 Polymarket 数据...")
        self._load_polymarket_data()

        if self._poly_markets is None or not self._poly_markets:
            logger.error("无 Polymarket 市场数据，退出")
            return

        logger.info("构建历史客户端...")
        hist_client = self._build_historical_client()

        pricing_config = PricingConfig()
        engine = FastPricingEngine(
            hist_client=hist_client,
            config=self.config,
            vrp_k=pricing_config.vrp_default_k,
        )

        # 确定日期范围
        if filter_event_date:
            event_dates = [filter_event_date]
        else:
            event_dates = _date_range(self.config.start_date, self.config.end_date)

        total_charts = 0
        for date_idx, event_date in enumerate(event_dates):
            strikes = self._get_date_strikes(event_date)
            if filter_strike is not None:
                strikes = [s for s in strikes if s == filter_strike]
            if not strikes:
                continue

            logger.info(f"[{date_idx + 1}/{len(event_dates)}] {event_date}: {len(strikes)} 个 strike")

            event_utc_ms = et_noon_to_utc_ms(event_date)
            event_open_ms = utc_ms_to_binance_kline_open(event_utc_ms)

            # 获取结算价
            try:
                settlement = hist_client.get_close_at_event(event_open_ms)
            except ValueError:
                logger.warning(f"{event_date}: 无结算数据，跳过")
                continue

            # HAR 系数
            har_coeffs = self.trainer.get_coeffs(event_date)

            # 重置分布缓存
            engine.reset_dist_cache()

            # 计算时间范围
            lookback_ms = self.config.lookback_hours * 60 * 60 * 1000
            start_ms = event_utc_ms - lookback_ms
            step_ms = self.config.step_minutes * 60_000

            # 为每个 strike 确定实际起始时间（不早于首笔交易）
            strike_start_ms: Dict[float, int] = {}
            for strike in strikes:
                first_trade = self._get_first_trade_time(event_date, strike)
                if first_trade is not None:
                    strike_start_ms[strike] = max(start_ms, first_trade)
                else:
                    strike_start_ms[strike] = start_ms

            # 收集数据: {strike: [(ts_ms, model_p, bid, ask, s0)]}
            chart_data: Dict[float, List[Tuple[int, float, Optional[float], Optional[float], float]]] = {
                s: [] for s in strikes
            }

            # 逐时间步
            current_ms = start_ms
            step_count = 0
            while current_ms < event_utc_ms:
                # 哪些 strike 在这个时间步需要数据
                active_strikes = [
                    s for s in strikes if current_ms >= strike_start_ms[s]
                ]

                if active_strikes:
                    try:
                        s0, probs = engine.compute_for_timestep(
                            event_date=event_date,
                            now_utc_ms=current_ms,
                            k_list=active_strikes,
                            har_coeffs=har_coeffs,
                        )
                        for strike in active_strikes:
                            model_p = probs.get(strike)
                            if model_p is None:
                                continue
                            bid, ask = self._get_bid_ask_at(
                                event_date, strike, current_ms
                            )
                            chart_data[strike].append((current_ms, model_p, bid, ask, s0))
                    except Exception as e:
                        logger.debug(f"{event_date} ts={current_ms}: {e}")

                current_ms += step_ms
                step_count += 1

            # 生成图表
            for strike in strikes:
                data = chart_data[strike]
                if len(data) < 2:
                    logger.debug(f"{event_date} K={strike:.0f}: 数据点不足，跳过")
                    continue

                times_ms = [d[0] for d in data]
                model_prices = [d[1] for d in data]
                bid_prices = [d[2] for d in data]
                ask_prices = [d[3] for d in data]
                btc_prices = [d[4] for d in data]

                settled_yes = settlement > strike

                csv_path = self._write_chart_csv(
                    event_date, strike, event_utc_ms, data
                )
                png_path = self._generate_chart(
                    event_date, strike, event_utc_ms,
                    times_ms, model_prices, bid_prices, ask_prices,
                    btc_prices, settled_yes,
                )
                total_charts += 1
                logger.info(f"  K={strike:.0f}: {len(data)} 点 → {png_path}")

        elapsed = time_mod.monotonic() - t_start
        logger.info(f"图表生成完成: {total_charts} 张, 耗时 {elapsed:.1f}s")

    def _generate_chart(
        self,
        event_date: str,
        strike: float,
        event_utc_ms: int,
        times_ms: List[int],
        model_prices: List[float],
        bid_prices: List[Optional[float]],
        ask_prices: List[Optional[float]],
        btc_prices: List[float],
        settled_yes: bool,
    ) -> str:
        """
        matplotlib 生成单个市场对比图

        左 Y 轴: 概率 [0, 1]
        - 蓝线: 模型 p_P
        - 橙线: best_ask（实线）
        - 棕线: best_bid（实线）
        - bid/ask 之间浅橙色填充表示 spread
        - 绿色/红色填充: model vs ask 的 edge 区域

        右 Y 轴: BTC 价格
        - 灰线: BTC/USDT 价格
        - 红色虚线: Strike 水平线

        X 轴: 距事件分钟数 (T-1440 → T-0)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 转换为距事件分钟数
        minutes_to_event = [(event_utc_ms - t) / 60_000 for t in times_ms]

        fig, ax_prob = plt.subplots(figsize=(14, 6))

        # === 左 Y 轴: 概率 ===
        # 模型线
        ax_prob.plot(minutes_to_event, model_prices, color="royalblue", linewidth=1.2,
                     label="Model p_P", alpha=0.9)

        # Best ask 线（过滤 None）
        ask_x, ask_y = [], []
        for i, ap in enumerate(ask_prices):
            if ap is not None and ap > 0:
                ask_x.append(minutes_to_event[i])
                ask_y.append(ap)

        # Best bid 线（过滤 None）
        bid_x, bid_y = [], []
        for i, bp in enumerate(bid_prices):
            if bp is not None and bp > 0:
                bid_x.append(minutes_to_event[i])
                bid_y.append(bp)

        if ask_x:
            ax_prob.plot(ask_x, ask_y, color="darkorange", linewidth=1.0,
                         label="Best Ask", alpha=0.85)
        if bid_x:
            ax_prob.plot(bid_x, bid_y, color="saddlebrown", linewidth=1.0,
                         label="Best Bid", alpha=0.85)

        # Bid-Ask spread 填充
        if bid_x and ask_x:
            bid_arr = np.full(len(model_prices), np.nan)
            ask_arr = np.full(len(model_prices), np.nan)
            for i, bp in enumerate(bid_prices):
                if bp is not None and bp > 0:
                    bid_arr[i] = bp
            for i, ap in enumerate(ask_prices):
                if ap is not None and ap > 0:
                    ask_arr[i] = ap

            both_valid = ~np.isnan(bid_arr) & ~np.isnan(ask_arr)
            if np.sum(both_valid) > 1:
                bid_interp = np.interp(
                    range(len(bid_arr)), np.where(both_valid)[0], bid_arr[both_valid])
                ask_interp = np.interp(
                    range(len(ask_arr)), np.where(both_valid)[0], ask_arr[both_valid])
                ax_prob.fill_between(
                    minutes_to_event, bid_interp, ask_interp,
                    color="orange", alpha=0.10, label="Spread"
                )

                # Edge 填充: model vs ask
                model_arr = np.array(model_prices)
                ax_prob.fill_between(
                    minutes_to_event, model_arr, ask_interp,
                    where=model_arr > ask_interp,
                    color="green", alpha=0.15, label="Model > Ask"
                )
                ax_prob.fill_between(
                    minutes_to_event, model_arr, bid_interp,
                    where=model_arr < bid_interp,
                    color="red", alpha=0.15, label="Model < Bid"
                )

        ax_prob.set_ylabel("Probability", fontsize=11)
        ax_prob.set_ylim(-0.02, 1.02)

        # === 右 Y 轴: BTC 价格 ===
        ax_btc = ax_prob.twinx()
        ax_btc.plot(minutes_to_event, btc_prices, color="gray", linewidth=0.8,
                    alpha=0.5, label="BTC Price")
        ax_btc.axhline(y=strike, color="red", linestyle="--", linewidth=1.0,
                       alpha=0.6, label=f"Strike ${strike:,.0f}")
        ax_btc.set_ylabel("BTC Price (USDT)", fontsize=11)

        # BTC Y 轴范围: 围绕 strike 对称，至少包含所有价格
        btc_arr = np.array(btc_prices)
        price_range = max(abs(btc_arr.max() - strike), abs(btc_arr.min() - strike), 500)
        margin = price_range * 0.3
        ax_btc.set_ylim(strike - price_range - margin, strike + price_range + margin)

        # === 格式 ===
        strike_str = f"{strike:,.0f}"
        settled_str = "YES" if settled_yes else "NO"
        ax_prob.set_title(f"BTC > ${strike_str} on {event_date} (settled: {settled_str})",
                          fontsize=14)
        ax_prob.set_xlabel("Minutes to Event", fontsize=11)
        ax_prob.invert_xaxis()
        ax_prob.grid(True, alpha=0.3)

        # 合并两个轴的图例
        lines_prob, labels_prob = ax_prob.get_legend_handles_labels()
        lines_btc, labels_btc = ax_btc.get_legend_handles_labels()
        ax_prob.legend(lines_prob + lines_btc, labels_prob + labels_btc,
                       loc="upper left", fontsize=9)

        # 保存
        out_dir = os.path.join(self.config.output_dir, event_date)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"BTC_above_{int(strike)}.png"
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return path

    def _write_chart_csv(
        self,
        event_date: str,
        strike: float,
        event_utc_ms: int,
        data: List[Tuple[int, float, Optional[float], Optional[float], float]],
    ) -> str:
        """输出原始数据 CSV: timestamp, minutes_to_event, model_p, best_bid, best_ask, edge_yes, edge_no, btc_price"""
        out_dir = os.path.join(self.config.output_dir, event_date)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"BTC_above_{int(strike)}.csv"
        path = os.path.join(out_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_ms", "minutes_to_event", "model_p",
                "best_bid", "best_ask", "edge_yes", "edge_no", "btc_price",
            ])
            for ts_ms, model_p, bid, ask, s0 in data:
                mins = (event_utc_ms - ts_ms) / 60_000
                bid_str = f"{bid:.6f}" if bid is not None else ""
                ask_str = f"{ask:.6f}" if ask is not None else ""
                edge_yes = f"{model_p - ask:.6f}" if ask is not None else ""
                edge_no = f"{bid - model_p:.6f}" if bid is not None else ""
                writer.writerow([
                    ts_ms, f"{mins:.1f}", f"{model_p:.6f}",
                    bid_str, ask_str, edge_yes, edge_no, f"{s0:.2f}",
                ])

        return path
