"""
Above 合约回测图表生成

每日图表:
- 左 Y: 概率 [0,1]（模型 p_above + Polymarket 价格 + edge 区域）
- 右 Y: BTC 价格（灰线 + strike 红色虚线）
- X: 观测时间
- 标注结算结果
"""

import csv
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from pricing_core.time_utils import UTC

from .models import AboveObservation

logger = logging.getLogger(__name__)


class AboveChartGenerator:
    """Above 合约回测图表生成器"""

    def __init__(self, output_dir: str = "above_backtest_results", symbol: str = "BTC"):
        self.output_dir = output_dir
        self.symbol = symbol

    def generate(
        self,
        observations: List[AboveObservation],
        event_dates: Optional[List[str]] = None,
    ) -> List[str]:
        """
        为每个事件日的每个 strike 生成对比图表

        Args:
            observations: 回测观测结果
            event_dates: 要生成的日期列表（None = 全部）

        Returns:
            生成的文件路径列表
        """
        if not observations:
            logger.warning("无观测数据，跳过图表生成")
            return []

        # 按日期分组
        day_groups: Dict[str, List[AboveObservation]] = defaultdict(list)
        for obs in observations:
            day_groups[obs.event_date].append(obs)

        if event_dates is None:
            event_dates = sorted(day_groups.keys())

        generated = []
        for date_str in event_dates:
            day_obs = day_groups.get(date_str, [])
            if not day_obs:
                continue
            # 为该日每个 strike 生成图表
            all_strikes = set()
            for obs in day_obs:
                all_strikes.update(obs.k_grid)

            for strike in sorted(all_strikes):
                paths = self._generate_strike_chart(day_obs, date_str, strike)
                generated.extend(paths)

        logger.info(f"图表生成完成: {len(generated)} 个文件")
        return generated

    def _generate_strike_chart(
        self,
        observations: List[AboveObservation],
        event_date: str,
        strike: float,
    ) -> List[str]:
        """为单个日期 + strike 生成图表 + CSV"""
        # 收集数据
        times_ms = []
        model_probs = []
        market_probs = []
        btc_prices = []

        for obs in observations:
            if strike not in obs.predictions:
                continue
            times_ms.append(obs.obs_utc_ms)
            model_probs.append(obs.predictions[strike])
            market_probs.append(obs.market_prices.get(strike))
            btc_prices.append(obs.s0)

        if len(times_ms) < 2:
            logger.debug(f"[{event_date}] strike={strike:.0f}: 数据点不足，跳过")
            return []

        # 标签
        label = observations[-1].labels.get(strike, 0)
        settlement = observations[-1].settlement_price

        # CSV
        csv_path = self._write_csv(
            event_date, strike, times_ms, model_probs, market_probs, btc_prices
        )

        # PNG
        png_path = self._generate_png(
            event_date, strike, times_ms, model_probs, market_probs,
            btc_prices, label, settlement,
        )

        return [p for p in [csv_path, png_path] if p]

    def _generate_png(
        self,
        event_date: str,
        strike: float,
        times_ms: List[int],
        model_probs: List[float],
        market_probs: List[Optional[float]],
        btc_prices: List[float],
        label: int,
        settlement: float,
    ) -> Optional[str]:
        """生成 matplotlib 图表"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib 未安装，跳过图表生成")
            return None

        # 转换时间
        dates = [datetime.fromtimestamp(t / 1000, tz=UTC) for t in times_ms]

        fig, ax_prob = plt.subplots(figsize=(14, 6))

        # === 左 Y 轴: 概率 ===
        ax_prob.plot(
            dates, model_probs, color="royalblue", linewidth=1.2,
            label="Model p_above", alpha=0.9,
        )

        # Polymarket 价格
        mkt_x, mkt_y = [], []
        for i, mp in enumerate(market_probs):
            if mp is not None and 0 < mp < 1:
                mkt_x.append(dates[i])
                mkt_y.append(mp)
        if mkt_x:
            ax_prob.plot(
                mkt_x, mkt_y, color="darkorange", linewidth=1.0,
                label="Polymarket", alpha=0.85,
            )

            # Edge 填充
            model_arr = np.array(model_probs)
            mkt_arr = np.full(len(model_probs), np.nan)
            for i, mp in enumerate(market_probs):
                if mp is not None and 0 < mp < 1:
                    mkt_arr[i] = mp

            valid = ~np.isnan(mkt_arr)
            if np.sum(valid) > 1:
                mkt_interp = np.interp(
                    range(len(mkt_arr)),
                    np.where(valid)[0],
                    mkt_arr[valid],
                )
                ax_prob.fill_between(
                    dates, model_arr, mkt_interp,
                    where=model_arr > mkt_interp,
                    color="green", alpha=0.12, label="Edge (Model > Mkt)",
                )
                ax_prob.fill_between(
                    dates, model_arr, mkt_interp,
                    where=model_arr < mkt_interp,
                    color="red", alpha=0.12, label="Edge (Model < Mkt)",
                )

        ax_prob.set_ylabel("Probability", fontsize=11)
        ax_prob.set_ylim(-0.02, 1.02)

        # === 右 Y 轴: BTC 价格 ===
        ax_btc = ax_prob.twinx()
        ax_btc.plot(
            dates, btc_prices, color="gray", linewidth=0.8,
            alpha=0.5, label=f"{self.symbol} Price",
        )
        ax_btc.axhline(
            y=strike, color="red", linestyle="--", linewidth=1.0,
            alpha=0.6, label=f"Strike ${strike:,.0f}",
        )
        ax_btc.set_ylabel(f"{self.symbol} Price (USDT)", fontsize=11)

        # BTC Y 轴范围
        btc_arr = np.array(btc_prices)
        price_range = max(
            abs(btc_arr.max() - strike), abs(btc_arr.min() - strike), 1000
        )
        margin = price_range * 0.3
        ax_btc.set_ylim(strike - price_range - margin, strike + price_range + margin)

        # === 格式 ===
        strike_str = f"{strike:,.0f}"
        settled_str = f"YES (above)" if label == 1 else "NO"
        ax_prob.set_title(
            f"Above ${strike_str} on {event_date} "
            f"(settlement: ${settlement:,.0f}) — {settled_str}",
            fontsize=13,
        )
        ax_prob.set_xlabel("Time (UTC)", fontsize=11)
        ax_prob.grid(True, alpha=0.3)

        ax_prob.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate(rotation=30)

        # 合并图例
        lines_prob, labels_prob = ax_prob.get_legend_handles_labels()
        lines_btc, labels_btc = ax_btc.get_legend_handles_labels()
        ax_prob.legend(
            lines_prob + lines_btc, labels_prob + labels_btc,
            loc="upper left", fontsize=9,
        )

        # 保存
        out_dir = os.path.join(self.output_dir, event_date)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"above_{int(strike)}.png"
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)  # 释放内存

        return path

    def _write_csv(
        self,
        event_date: str,
        strike: float,
        times_ms: List[int],
        model_probs: List[float],
        market_probs: List[Optional[float]],
        btc_prices: List[float],
    ) -> str:
        """输出原始数据 CSV"""
        out_dir = os.path.join(self.output_dir, event_date)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"above_{int(strike)}.csv"
        path = os.path.join(out_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_ms", "datetime_utc", "model_p_above",
                "market_price", "edge", "btc_price",
            ])
            for i in range(len(times_ms)):
                dt_str = datetime.fromtimestamp(
                    times_ms[i] / 1000, tz=UTC
                ).strftime("%Y-%m-%d %H:%M")
                mp = market_probs[i]
                mp_str = f"{mp:.6f}" if mp is not None else ""
                edge_str = f"{model_probs[i] - mp:.6f}" if mp is not None else ""
                writer.writerow([
                    times_ms[i], dt_str, f"{model_probs[i]:.6f}",
                    mp_str, edge_str, f"{btc_prices[i]:.2f}",
                ])

        return path
