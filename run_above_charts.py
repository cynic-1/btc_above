"""
DVOL 模型定价 vs Polymarket 对比图表 CLI 入口

用法:
    python run_above_charts.py --start 2026-03-01 --end 2026-03-24
    python run_above_charts.py --event-date 2026-03-10 --strike 85000
"""

import argparse
import csv
import logging
import os
import time as time_mod
from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from above.dvol_pricing import prob_above_k_gbm
from backtest.data_cache import KlineCache, _date_range, _date_to_utc_ms
from backtest.hybrid_lookup import HybridPriceLookup
from backtest.orderbook_reader import OrderbookPriceLookup
from backtest.polymarket_discovery import (
    PolymarketMarketInfo,
    discover_markets_for_range,
)
from backtest.polymarket_trades import PolymarketPriceLookup, PolymarketTradeCache
from pricing_core.time_utils import (
    UTC,
    et_noon_to_utc_ms,
    utc_ms_to_binance_kline_open,
)
from pricing_core.utils.logger import setup_logger
from touch.iv_source import DeribitIVCache

logger = logging.getLogger(__name__)

# 一年的毫秒数
_YEAR_MS = 365.25 * 86400 * 1000


# ------------------------------------------------------------------
# 数据加载辅助
# ------------------------------------------------------------------


def _load_klines(cache: KlineCache, start_ms: int, end_ms: int):
    """加载 K线并返回 (times, closes) 列表"""
    klines = cache.load_range_ms(start_ms, end_ms)
    times = [k.open_time for k in klines]
    closes = [k.close for k in klines]
    logger.info(f"加载 K线: {len(klines)} 条")
    return times, closes


def _get_s0_at(
    times: List[int], closes: List[float], timestamp_ms: int
) -> Optional[float]:
    """bisect 查询 <= timestamp_ms 的最近 close"""
    idx = bisect_right(times, timestamp_ms) - 1
    if idx < 0:
        return None
    return closes[idx]


def _load_polymarket(
    start_date: str,
    end_date: str,
    polymarket_cache_dir: str,
    orderbook_cache_dir: str,
    orderbook_staleness_hours: float,
    gamma_api: str,
):
    """加载 Polymarket 市场发现 + 混合价格源"""
    cache_path = os.path.join(polymarket_cache_dir, "discovery_cache.json")
    markets = discover_markets_for_range(
        start_date=start_date,
        end_date=end_date,
        gamma_api=gamma_api,
        cache_path=cache_path,
    )

    if not markets:
        logger.warning("未发现 Polymarket 市场")
        return markets, None

    cids = list({info.condition_id for info in markets.values()})

    # Orderbook
    ob_lookup = None
    if os.path.isdir(orderbook_cache_dir):
        npz_files = [f for f in os.listdir(orderbook_cache_dir) if f.endswith(".npz")]
        if npz_files:
            ob_lookup = OrderbookPriceLookup(cache_dir=orderbook_cache_dir)
            ob_lookup.preload(cids)
            logger.info(f"订单簿初始化: {len(npz_files)} 个 npz")

    # CLOB trades
    trade_cache = PolymarketTradeCache(cache_dir=polymarket_cache_dir)
    trade_cache.ensure_trades(markets)
    clob_lookup = PolymarketPriceLookup(trade_cache)
    clob_lookup.preload(cids)

    # 混合
    if ob_lookup is not None:
        staleness_ms = int(orderbook_staleness_hours * 3_600_000)
        price_lookup = HybridPriceLookup(
            orderbook=ob_lookup,
            clob=clob_lookup,
            staleness_threshold_ms=staleness_ms,
        )
        logger.info("价格源: Hybrid (orderbook + CLOB)")
    else:
        price_lookup = clob_lookup
        logger.info("价格源: CLOB")

    return markets, price_lookup


# ------------------------------------------------------------------
# 图表生成
# ------------------------------------------------------------------


def _generate_chart(
    event_date: str,
    strike: float,
    event_utc_ms: int,
    times_ms: List[int],
    model_prices: List[float],
    bid_prices: List[Optional[float]],
    ask_prices: List[Optional[float]],
    btc_prices: List[float],
    settled_yes: bool,
    output_dir: str,
) -> str:
    """
    生成双 Y 轴对比图 (仿照 backtest/chart_engine.py)

    左 Y: 概率 — Model (蓝), Ask (橙), Bid (棕), edge 填充
    右 Y: BTC 价格 (灰) + Strike 红线
    X: Minutes to Event (倒序)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minutes_to_event = [(event_utc_ms - t) / 60_000 for t in times_ms]

    fig, ax_prob = plt.subplots(figsize=(14, 6))

    # --- 左 Y: 概率 ---
    ax_prob.plot(
        minutes_to_event, model_prices, color="royalblue",
        linewidth=1.2, label="DVOL Model", alpha=0.9,
    )

    # Best ask
    ask_x = [minutes_to_event[i] for i, a in enumerate(ask_prices) if a is not None and a > 0]
    ask_y = [a for a in ask_prices if a is not None and a > 0]
    # Best bid
    bid_x = [minutes_to_event[i] for i, b in enumerate(bid_prices) if b is not None and b > 0]
    bid_y = [b for b in bid_prices if b is not None and b > 0]

    if ask_x:
        ax_prob.plot(ask_x, ask_y, color="darkorange", linewidth=1.0,
                     label="Best Ask", alpha=0.85)
    if bid_x:
        ax_prob.plot(bid_x, bid_y, color="saddlebrown", linewidth=1.0,
                     label="Best Bid", alpha=0.85)

    # Spread + edge 填充
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
                color="orange", alpha=0.10, label="Spread",
            )

            model_arr = np.array(model_prices)
            ax_prob.fill_between(
                minutes_to_event, model_arr, ask_interp,
                where=model_arr > ask_interp,
                color="green", alpha=0.15, label="Model > Ask",
            )
            ax_prob.fill_between(
                minutes_to_event, model_arr, bid_interp,
                where=model_arr < bid_interp,
                color="red", alpha=0.15, label="Model < Bid",
            )

    ax_prob.set_ylabel("Probability", fontsize=11)
    ax_prob.set_ylim(-0.02, 1.02)

    # --- 右 Y: BTC 价格 ---
    ax_btc = ax_prob.twinx()
    ax_btc.plot(
        minutes_to_event, btc_prices, color="gray",
        linewidth=0.8, alpha=0.5, label="BTC Price",
    )
    ax_btc.axhline(
        y=strike, color="red", linestyle="--",
        linewidth=1.0, alpha=0.6, label=f"Strike ${strike:,.0f}",
    )
    ax_btc.set_ylabel("BTC Price (USDT)", fontsize=11)

    btc_arr = np.array(btc_prices)
    price_range = max(abs(btc_arr.max() - strike), abs(btc_arr.min() - strike), 500)
    margin = price_range * 0.3
    ax_btc.set_ylim(strike - price_range - margin, strike + price_range + margin)

    # --- 格式 ---
    settled_str = "YES" if settled_yes else "NO"
    ax_prob.set_title(
        f"[DVOL] BTC > ${strike:,.0f} on {event_date} (settled: {settled_str})",
        fontsize=14,
    )
    ax_prob.set_xlabel("Minutes to Event", fontsize=11)
    ax_prob.invert_xaxis()
    ax_prob.grid(True, alpha=0.3)

    lines_prob, labels_prob = ax_prob.get_legend_handles_labels()
    lines_btc, labels_btc = ax_btc.get_legend_handles_labels()
    ax_prob.legend(
        lines_prob + lines_btc, labels_prob + labels_btc,
        loc="upper left", fontsize=9,
    )

    out_dir = os.path.join(output_dir, event_date)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"DVOL_above_{int(strike)}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return path


def _write_csv(
    event_date: str,
    strike: float,
    event_utc_ms: int,
    data: List[Tuple[int, float, Optional[float], Optional[float], float, float]],
    output_dir: str,
) -> str:
    """输出原始数据 CSV"""
    out_dir = os.path.join(output_dir, event_date)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"DVOL_above_{int(strike)}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ms", "minutes_to_event", "model_p",
            "best_bid", "best_ask", "edge_yes", "edge_no",
            "btc_price", "dvol",
        ])
        for ts_ms, model_p, bid, ask, s0, sigma in data:
            mins = (event_utc_ms - ts_ms) / 60_000
            bid_str = f"{bid:.6f}" if bid is not None else ""
            ask_str = f"{ask:.6f}" if ask is not None else ""
            edge_yes = f"{model_p - ask:.6f}" if ask is not None else ""
            edge_no = f"{bid - model_p:.6f}" if bid is not None else ""
            writer.writerow([
                ts_ms, f"{mins:.1f}", f"{model_p:.6f}",
                bid_str, ask_str, edge_yes, edge_no,
                f"{s0:.2f}", f"{sigma:.6f}",
            ])

    return path


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="DVOL 模型定价 vs Polymarket 对比图表"
    )
    parser.add_argument("--start", default="2026-03-01", help="起始日期")
    parser.add_argument("--end", default="2026-03-24", help="结束日期")
    parser.add_argument("--event-date", help="只生成特定日期")
    parser.add_argument("--strike", type=float, help="只生成特定 strike")
    parser.add_argument("--output-dir", default="above_charts", help="输出目录")
    parser.add_argument("--step-minutes", type=int, default=1, help="时间分辨率（分钟）")
    parser.add_argument("--lookback-hours", type=int, default=24, help="事件前多少小时开始")
    parser.add_argument("--vrp-k", type=float, default=1.0, help="VRP 缩放系数")
    parser.add_argument("--mu", type=float, default=0.0, help="漂移率")
    parser.add_argument("--default-sigma", type=float, default=0.65, help="默认波动率")
    parser.add_argument("--cache-dir", default="data/klines", help="K线缓存目录")
    parser.add_argument("--iv-cache-dir", default="data/deribit_iv", help="DVOL 缓存目录")
    parser.add_argument("--polymarket-cache-dir", default="data/polymarket")
    parser.add_argument("--orderbook-cache-dir", default="data/orderbook_cache")
    parser.add_argument("--orderbook-staleness-hours", type=float, default=2.0)
    args = parser.parse_args()

    setup_logger()

    # 日期范围
    start = args.start
    end = args.end
    if args.event_date:
        start = args.event_date
        dt = datetime.strptime(args.event_date, "%Y-%m-%d")
        end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

    event_dates = _date_range(start, end)
    if not event_dates:
        logger.error("日期范围为空")
        return

    t_start = time_mod.monotonic()

    # 1. K线
    cache = KlineCache(cache_dir=args.cache_dir)
    extra_start = datetime.strptime(start, "%Y-%m-%d") - timedelta(hours=args.lookback_hours + 24)
    kl_start_ms = _date_to_utc_ms(extra_start.strftime("%Y-%m-%d"))
    kl_end_ms = _date_to_utc_ms(end)
    kline_times, kline_closes = _load_klines(cache, kl_start_ms, kl_end_ms)

    if not kline_times:
        logger.error("无 K线数据")
        return

    # 2. DVOL
    iv_cache = DeribitIVCache(cache_dir=args.iv_cache_dir)
    iv_cache.load(currency="BTC")

    # 3. Polymarket
    markets, price_lookup = _load_polymarket(
        start_date=start,
        end_date=end,
        polymarket_cache_dir=args.polymarket_cache_dir,
        orderbook_cache_dir=args.orderbook_cache_dir,
        orderbook_staleness_hours=args.orderbook_staleness_hours,
        gamma_api="https://gamma-api.polymarket.com",
    )

    if not markets:
        logger.error("无 Polymarket 市场数据")
        return

    total_charts = 0
    for date_idx, event_date in enumerate(event_dates):
        # 该日的 strike
        strikes = sorted(
            strike for (d, strike) in markets if d == event_date
        )
        if args.strike is not None:
            strikes = [s for s in strikes if s == args.strike]
        if not strikes:
            continue

        logger.info(f"[{date_idx + 1}/{len(event_dates)}] {event_date}: {len(strikes)} 个 strike")

        event_utc_ms = et_noon_to_utc_ms(event_date)
        event_open_ms = utc_ms_to_binance_kline_open(event_utc_ms)

        # 结算价
        settlement = _get_s0_at(kline_times, kline_closes, event_open_ms)
        if settlement is None:
            logger.warning(f"{event_date}: 无结算数据，跳过")
            continue

        lookback_ms = args.lookback_hours * 3_600_000
        start_ms = event_utc_ms - lookback_ms
        step_ms = args.step_minutes * 60_000

        # 每个 strike 的首笔交易时间
        strike_start_ms: Dict[float, int] = {}
        for strike in strikes:
            info = markets.get((event_date, strike))
            if info and price_lookup and hasattr(price_lookup, 'get_first_timestamp'):
                first_ts = price_lookup.get_first_timestamp(info.condition_id)
                if first_ts is not None:
                    strike_start_ms[strike] = max(start_ms, first_ts)
                    continue
            strike_start_ms[strike] = start_ms

        # 收集: {strike: [(ts, model_p, bid, ask, s0, sigma)]}
        chart_data: Dict[float, List[Tuple[int, float, Optional[float], Optional[float], float, float]]] = {
            s: [] for s in strikes
        }

        current_ms = start_ms
        while current_ms < event_utc_ms:
            s0 = _get_s0_at(kline_times, kline_closes, current_ms)
            if s0 is None:
                current_ms += step_ms
                continue

            # DVOL
            iv = iv_cache.get_iv_at(current_ms)
            if iv is not None:
                sigma = iv * args.vrp_k
            else:
                sigma = args.default_sigma

            T_ms = event_utc_ms - current_ms
            T_years = max(T_ms / _YEAR_MS, 0.0)

            for strike in strikes:
                if current_ms < strike_start_ms[strike]:
                    continue

                model_p = prob_above_k_gbm(s0, strike, sigma, T_years, args.mu)

                bid, ask = None, None
                info = markets.get((event_date, strike))
                if info and price_lookup:
                    if hasattr(price_lookup, 'get_quote_at'):
                        quote = price_lookup.get_quote_at(info.condition_id, current_ms)
                        if quote is not None:
                            bid, ask = quote.best_bid, quote.best_ask
                    else:
                        mid = price_lookup.get_price_at(info.condition_id, current_ms)
                        bid, ask = mid, mid

                chart_data[strike].append((current_ms, model_p, bid, ask, s0, sigma))

            current_ms += step_ms

        # 生成图表
        for strike in strikes:
            data = chart_data[strike]
            if len(data) < 2:
                logger.debug(f"{event_date} K={strike:.0f}: 数据点不足")
                continue

            settled_yes = settlement > strike

            csv_path = _write_csv(event_date, strike, event_utc_ms, data, args.output_dir)
            png_path = _generate_chart(
                event_date, strike, event_utc_ms,
                [d[0] for d in data],
                [d[1] for d in data],
                [d[2] for d in data],
                [d[3] for d in data],
                [d[4] for d in data],
                settled_yes,
                args.output_dir,
            )
            total_charts += 1
            logger.info(f"  K={strike:.0f}: {len(data)} 点 → {png_path}")

    elapsed = time_mod.monotonic() - t_start
    print(f"\n生成 {total_charts} 张图表, 耗时 {elapsed:.1f}s → {args.output_dir}/")


if __name__ == "__main__":
    main()
