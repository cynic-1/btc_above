"""
主编排引擎

组合所有组件，运行实盘交易主循环:
market_discovery → polymarket_ws → binance_feed → pricing_engine
→ signal generation → order execution
"""

import logging
import signal as signal_mod
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from pricing_core.time_utils import et_noon_to_utc_ms, minutes_until_event

from .binance_feed import BinanceKlineFeed
from .config import LiveTradingConfig
from .market_discovery import discover_today_markets
from .models import MarketInfo, Signal, TradeRecord
from .order_client import PolymarketOrderClient
from .polymarket_ws import PolymarketOrderbookWS
from .position_manager import PositionManager
from .pricing_engine import LivePricingEngine

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """
    主编排引擎

    主循环 (pricing_interval_seconds 间隔):
    1. 从 binance_feed 获取最新数据
    2. 调用 pricing_engine 计算各 strike 的 p_physical
    3. 对每个市场:
       a. 从 polymarket_ws 获取 best bid/ask
       b. 计算 edge = |model_price - market_price|
       c. 如果 edge > threshold 且仓位允许 → 生成 Signal
    4. 对每个 Signal → position_manager.can_trade() → order_client.place_order()
    5. 记录交易日志
    """

    def __init__(self, config: LiveTradingConfig):
        self._config = config
        self._markets: List[MarketInfo] = []
        self._market_by_token: Dict[str, MarketInfo] = {}  # yes_token_id → MarketInfo

        # 组件
        self._binance_feed = BinanceKlineFeed(config)
        self._polymarket_ws = PolymarketOrderbookWS(config)
        self._pricing_engine = LivePricingEngine(config)
        self._position_manager = PositionManager(config)
        self._order_client: Optional[PolymarketOrderClient] = None

        # 状态
        self._running = False
        self._stop_event = threading.Event()
        self._trade_records: List[TradeRecord] = []
        self._last_order_time: Dict[str, float] = {}  # condition_id → 上次下单时间
        self._pending_orders: Dict[str, dict] = {}  # order_id → {"signal", "placed_at"}

    def start(self) -> None:
        """初始化所有组件并启动主循环"""
        logger.info("=" * 60)
        logger.info("实盘交易引擎启动")
        logger.info(f"  事件日期: {self._config.event_date}")
        logger.info(f"  dry_run: {self._config.dry_run}")
        logger.info(f"  shares_per_trade: {self._config.shares_per_trade}")
        logger.info(f"  entry_threshold: {self._config.entry_threshold}")
        logger.info(f"  order_type: {self._config.order_type}")
        logger.info(f"  pricing_interval: {self._config.pricing_interval_seconds}s")
        logger.info("=" * 60)

        # 1. 市场发现
        self._markets = discover_today_markets(
            event_date=self._config.event_date,
            gamma_api=self._config.gamma_api,
            clob_host=self._config.polymarket_host,
        )
        if not self._markets:
            logger.error("未发现任何市场，退出")
            return

        # 构建 token → market 映射
        for m in self._markets:
            self._market_by_token[m.yes_token_id] = m

        strikes = sorted(set(m.strike for m in self._markets))
        logger.info(f"发现 {len(self._markets)} 个市场, strikes: {strikes}")

        # 2. 启动 Binance K线流
        if not self._binance_feed.start():
            logger.error("Binance K线流启动失败，退出")
            return

        # 3. 训练 HAR 系数（排除事件日当天数据，与回测一致）
        klines = self._binance_feed.get_klines()
        self._pricing_engine.train_har(klines, self._config.event_date)

        # 4. 初始化订单客户端
        if not self._config.dry_run:
            try:
                self._order_client = PolymarketOrderClient(self._config)
            except Exception as e:
                logger.error(f"订单客户端初始化失败: {e}")
                logger.info("切换到 dry-run 模式")
                self._config.dry_run = True

        # 5. 启动 Polymarket WebSocket
        asset_ids = [m.yes_token_id for m in self._markets]
        self._polymarket_ws.connect(asset_ids)

        # 6. 启动主循环
        self._running = True
        self._stop_event.clear()

        # 注册信号处理
        signal_mod.signal(signal_mod.SIGINT, self._signal_handler)
        signal_mod.signal(signal_mod.SIGTERM, self._signal_handler)

        logger.info("所有组件已启动，进入主循环")
        self._main_loop()

    def stop(self) -> None:
        """优雅关闭"""
        logger.info("正在关闭交易引擎...")
        self._running = False
        self._stop_event.set()
        self._polymarket_ws.close()
        self._binance_feed.stop()

        # 仓位汇总
        logger.info(f"仓位汇总:\n{self._position_manager.summary()}")
        logger.info(f"交易记录: {len(self._trade_records)} 笔")
        logger.info("交易引擎已关闭")

    def _signal_handler(self, signum, frame) -> None:
        """SIGINT/SIGTERM 处理"""
        logger.info(f"收到信号 {signum}，准备退出")
        self.stop()

    def _main_loop(self) -> None:
        """主循环"""
        event_utc_ms = et_noon_to_utc_ms(self._config.event_date)

        while self._running:
            loop_start = time.monotonic()
            now_utc_ms = int(time.time() * 1000)

            # 检查是否已过事件时间
            mins_left = minutes_until_event(now_utc_ms, event_utc_ms)
            if mins_left <= 0:
                logger.info("事件已到期，停止交易")
                break

            try:
                # 计算信号
                signals = self._compute_signals(now_utc_ms, event_utc_ms)

                # 执行信号
                for sig in signals:
                    self._execute_signal(sig)

                # 检查超时挂单
                self._check_pending_orders()

                # 健康检查日志（每 60s 输出一次）
                if int(time.time()) % 60 < self._config.pricing_interval_seconds:
                    health = self._health_check(mins_left)
                    logger.info(
                        f"[健康检查] 距到期 {mins_left:.1f}min | "
                        f"BTC={health['btc_price']:.2f} | "
                        f"WS连接: binance={health['binance_ok']}, "
                        f"polymarket={health['polymarket_ok']} | "
                        f"交易: {health['total_trades']}笔 | "
                        f"成本: ${health['total_cost']:.2f}"
                    )

            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)

            # 等待下一个周期
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0, self._config.pricing_interval_seconds - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

        self.stop()

    def _compute_signals(
        self, now_utc_ms: int, event_utc_ms: int,
    ) -> List[Signal]:
        """
        计算所有市场的交易信号

        Returns:
            Signal 列表
        """
        signals = []

        # 1. 获取 Binance 数据
        s0 = self._binance_feed.get_current_price()
        klines = self._binance_feed.get_klines()

        if s0 <= 0 or len(klines) < 60:
            logger.warning(f"数据不足，跳过定价: s0={s0:.2f}, klines={len(klines)}")
            return signals

        # 2. 定价
        k_list = sorted(set(m.strike for m in self._markets))
        mins_left = minutes_until_event(now_utc_ms, event_utc_ms)
        probs = self._pricing_engine.compute_prices(
            event_date=self._config.event_date,
            now_utc_ms=now_utc_ms,
            s0=s0,
            klines=klines,
            k_list=k_list,
        )

        if not probs:
            logger.warning("定价引擎返回空结果")
            return signals

        # 时间窗口过滤（与回测 min/max_obs_minutes 对齐）
        if mins_left < self._config.min_minutes_to_event or \
           mins_left > self._config.max_minutes_to_event:
            logger.debug(f"时间窗口外 T-{mins_left:.0f}min, 跳过")
            return signals

        # 方向过滤（与回测 direction_filter 对齐）
        allow_yes = self._config.direction_filter != "no_only"
        allow_no = self._config.direction_filter != "yes_only"

        # 非对称阈值（与回测 yes_threshold/no_threshold 对齐）
        eff_yes_threshold = self._config.yes_threshold if self._config.yes_threshold is not None \
            else self._config.entry_threshold
        eff_no_threshold = self._config.no_threshold if self._config.no_threshold is not None \
            else self._config.entry_threshold

        # 3. 汇总定价 + orderbook + edge，构建扫描表
        scan_lines = []
        no_ob_count = 0

        for market in self._markets:
            p_physical = probs.get(market.strike)
            if p_physical is None:
                continue

            ob = self._polymarket_ws.get_orderbook(market.yes_token_id)
            if ob is None or (ob.best_bid == 0 and ob.best_ask == 0):
                no_ob_count += 1
                scan_lines.append(
                    f"  K={market.strike:>7.0f}  model={p_physical:.4f}  "
                    f"bid/ask=--/--  edge=--  (无orderbook)"
                )
                continue

            best_bid = ob.best_bid
            best_ask = ob.best_ask

            if best_ask <= 0 or best_ask >= 1:
                scan_lines.append(
                    f"  K={market.strike:>7.0f}  model={p_physical:.4f}  "
                    f"bid/ask={best_bid:.3f}/{best_ask:.3f}  edge=--  (ask无效)"
                )
                continue

            # Spread 过滤（与回测 max_spread 对齐）
            spread = best_ask - best_bid
            if self._config.max_spread is not None and spread > self._config.max_spread:
                scan_lines.append(
                    f"  K={market.strike:>7.0f}  model={p_physical:.4f}  "
                    f"bid/ask={best_bid:.3f}/{best_ask:.3f}  edge=--  (spread={spread:.3f}>max)"
                )
                continue

            # 直接用模型概率（与回测一致，不收缩）
            edge_yes = p_physical - best_ask
            edge_no = best_bid - p_physical

            # 选最大方向的 edge 用于显示
            if edge_yes >= edge_no:
                display_edge = edge_yes
                display_dir = "YES"
            else:
                display_edge = edge_no
                display_dir = "NO"

            threshold = max(eff_yes_threshold, eff_no_threshold)
            marker = " <--" if display_edge > threshold else ""

            scan_lines.append(
                f"  K={market.strike:>7.0f}  model={p_physical:.4f}  "
                f"bid/ask={best_bid:.3f}/{best_ask:.3f}  "
                f"edge={display_edge:+.4f}({display_dir}){marker}"
            )

            # 检查冷却期
            now_time = time.time()
            last_order = self._last_order_time.get(market.condition_id, 0)
            if now_time - last_order < self._config.order_cooldown_seconds:
                continue

            # YES 方向: model > ask → BUY YES（按 best_bid 挂单）
            if allow_yes and edge_yes > eff_yes_threshold and best_ask < 0.99:
                sig = Signal(
                    strike=market.strike,
                    condition_id=market.condition_id,
                    token_id=market.yes_token_id,
                    direction="YES",
                    side="BUY",
                    model_price=p_physical,
                    market_price=best_ask,
                    edge=edge_yes,
                    shares=self._config.shares_per_trade,
                    price=best_bid,
                    timestamp_ms=now_utc_ms,
                )
                signals.append(sig)

            # NO 方向: bid > model → BUY NO（elif 互斥）
            elif allow_no and edge_no > eff_no_threshold and best_bid > 0.01:
                sig = Signal(
                    strike=market.strike,
                    condition_id=market.condition_id,
                    token_id=market.no_token_id,
                    direction="NO",
                    side="BUY",
                    model_price=1 - p_physical,
                    market_price=1 - best_bid,
                    edge=edge_no,
                    shares=self._config.shares_per_trade,
                    price=1 - best_ask,
                    timestamp_ms=now_utc_ms,
                )
                signals.append(sig)

        # 输出扫描表
        header = (
            f"[扫描] BTC={s0:.2f} T-{mins_left:.0f}min "
            f"klines={len(klines)} ob={len(self._markets)-no_ob_count}/{len(self._markets)}"
        )
        logger.info(header + "\n" + "\n".join(scan_lines))

        if signals:
            logger.info(f">>> 检测到 {len(signals)} 个交易信号")

        return signals

    def _execute_signal(self, signal: Signal) -> None:
        """执行交易信号"""
        # 1. 仓位检查
        can, reason = self._position_manager.can_trade(signal)
        if not can:
            logger.info(
                f"信号被拒: K={signal.strike:.0f} {signal.direction} "
                f"edge={signal.edge:.4f} — {reason}"
            )
            return

        logger.info(
            f"{'[DRY-RUN] ' if self._config.dry_run else ''}"
            f"执行信号: K={signal.strike:.0f} {signal.direction} "
            f"price={signal.price:.4f} shares={signal.shares} "
            f"edge={signal.edge:.4f} model={signal.model_price:.4f}"
        )

        if self._config.dry_run:
            # dry-run 模式仅记录
            record = TradeRecord(
                order_id="dry-run",
                signal=signal,
                status="dry-run",
                response={},
                timestamp_ms=int(time.time() * 1000),
            )
            self._trade_records.append(record)
            self._position_manager.record_trade(signal, "matched")
            self._last_order_time[signal.condition_id] = time.time()
            return

        # 2. 真实下单
        market = self._market_by_token.get(signal.token_id)
        tick_size = market.tick_size if market else "0.01"
        neg_risk = market.neg_risk if market else False

        response = self._order_client.place_order(
            token_id=signal.token_id,
            side=signal.side,
            price=signal.price,
            size=float(signal.shares),
            order_type=self._config.order_type,
            tick_size=tick_size,
            neg_risk=neg_risk,
        )

        # 解析结果
        status = "failed"
        order_id = ""
        if isinstance(response, dict):
            # 余额不足 → 自动切换 dry-run
            if response.get("reason") == "insufficient_balance":
                logger.warning(
                    "余额/授权不足，自动切换到 dry-run 模式。"
                    "后续信号将仅记录不下单。"
                )
                self._config.dry_run = True
                return

            if response.get("success") is not False:
                order_id = response.get("orderID", response.get("id", ""))
                status = response.get("status", "live")
        else:
            order_id = str(response)
            status = "live"

        record = TradeRecord(
            order_id=order_id,
            signal=signal,
            status=status,
            response=response if isinstance(response, dict) else {"raw": str(response)},
            timestamp_ms=int(time.time() * 1000),
        )
        self._trade_records.append(record)

        if self._config.order_type == "FOK":
            # FOK: 只有 matched 才计入仓位
            if status == "matched":
                self._position_manager.record_trade(signal, status)
            else:
                logger.warning(
                    f"FOK 未成交: K={signal.strike:.0f} {signal.direction} "
                    f"shares={signal.shares} status={status}, 不计入仓位"
                )
        else:
            # GTC: 立即计入仓位（保守过高计数，防止超限）
            # 超时后取消成功才减去仓位
            if status != "failed":
                self._position_manager.record_trade(signal, status)
                if status != "matched":
                    # 挂单中，追踪以便超时取消
                    self._pending_orders[order_id] = {
                        "signal": signal,
                        "placed_at": time.time(),
                    }

        self._last_order_time[signal.condition_id] = time.time()

        logger.info(f"下单结果: order_id={order_id}, status={status}")

    def _check_pending_orders(self) -> None:
        """检查超时的 GTC 挂单，取消未成交的并回退仓位

        策略：下单时已 record_trade（保守计入仓位）。
        超时后尝试取消：
        - 取消成功 → 订单确实未成交 → reverse_trade 回退仓位
        - 取消失败 → 订单已成交 → 仓位已正确，无需操作
        """
        if not self._pending_orders:
            return

        timeout = self._config.order_timeout_seconds
        now = time.time()
        expired = [
            (oid, info) for oid, info in self._pending_orders.items()
            if now - info["placed_at"] >= timeout
        ]

        for order_id, info in expired:
            signal = info["signal"]
            age = now - info["placed_at"]

            try:
                result = self._order_client.cancel_order(order_id)
                cancelled = isinstance(result, dict) and "error" not in result
            except Exception as e:
                logger.warning(f"取消订单异常: order_id={order_id}, {e}")
                cancelled = False

            if cancelled:
                # 取消成功 → 订单未成交，回退已计入的仓位
                self._position_manager.reverse_trade(signal)
                logger.info(
                    f"挂单超时取消+回退仓位: order_id={order_id} K={signal.strike:.0f} "
                    f"{signal.direction} {signal.shares}份, 挂了{age:.0f}s"
                )
            else:
                # 取消失败 → 已成交，仓位已正确记录
                logger.info(
                    f"挂单已成交（取消失败）: order_id={order_id} K={signal.strike:.0f} "
                    f"{signal.direction} {signal.shares}份"
                )

            del self._pending_orders[order_id]

    def _health_check(self, mins_left: float) -> dict:
        """健康检查"""
        return {
            "btc_price": self._binance_feed.get_current_price(),
            "binance_ok": self._binance_feed.is_connected(),
            "polymarket_ok": self._polymarket_ws.is_connected(),
            "total_trades": len(self._trade_records),
            "total_cost": self._position_manager.get_total_cost(),
            "mins_left": mins_left,
            "markets": len(self._markets),
        }
