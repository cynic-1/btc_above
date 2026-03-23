"""
触碰障碍期权回测图表生成

双轴图表:
- 左 Y: 概率 [0,1]（模型 p_touch + Polymarket 价格 + edge 区域）
- 右 Y: BTC 价格（灰线 + barrier 红色虚线）
- X: 月内日期
- 标注 "TOUCHED" 时刻
"""

import csv
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from pricing_core.time_utils import UTC

from .models import TouchObservationResult

logger = logging.getLogger(__name__)


class TouchChartGenerator:
    """触碰障碍期权回测图表生成器"""

    def __init__(self, output_dir: str = "touch_backtest_results", symbol: str = "BTC"):
        self.output_dir = output_dir
        self.symbol = symbol

    def generate(
        self,
        observations: List[TouchObservationResult],
        month: str,
        barriers: Optional[List[float]] = None,
    ) -> List[str]:
        """
        为每个 barrier 生成对比图表

        Args:
            observations: 回测观测结果
            month: 月份 "YYYY-MM"
            barriers: 要生成的 barrier 列表（None = 全部）

        Returns:
            生成的文件路径列表
        """
        if not observations:
            logger.warning("无观测数据，跳过图表生成")
            return []

        # 确定 barriers
        if barriers is None:
            all_barriers = set()
            for obs in observations:
                all_barriers.update(obs.barriers)
            barriers = sorted(all_barriers)

        generated = []
        for barrier in barriers:
            paths = self._generate_barrier_chart(observations, month, barrier)
            generated.extend(paths)

        logger.info(f"图表生成完成: {len(generated)} 个文件")
        return generated

    def _generate_barrier_chart(
        self,
        observations: List[TouchObservationResult],
        month: str,
        barrier: float,
    ) -> List[str]:
        """为单个 barrier 生成图表 + CSV"""
        # 收集数据
        times_ms = []
        model_probs = []
        market_probs = []
        btc_prices = []
        touched_flags = []

        for obs in observations:
            if barrier not in obs.predictions:
                continue
            times_ms.append(obs.obs_utc_ms)
            model_probs.append(obs.predictions[barrier])
            market_probs.append(obs.market_prices.get(barrier))
            btc_prices.append(obs.s0)
            touched_flags.append(obs.already_touched.get(barrier, False))

        if len(times_ms) < 2:
            logger.debug(f"barrier={barrier:.0f}: 数据点不足，跳过")
            return []

        # 找 "TOUCHED" 时刻
        touched_time_ms = None
        for i, touched in enumerate(touched_flags):
            if touched:
                touched_time_ms = times_ms[i]
                break

        # 判断月末标签
        final_label = observations[-1].labels.get(barrier, 0) if observations else 0

        # 生成 CSV
        csv_path = self._write_csv(
            month, barrier, times_ms, model_probs, market_probs, btc_prices
        )

        # 生成 PNG
        png_path = self._generate_png(
            month, barrier, times_ms, model_probs, market_probs,
            btc_prices, touched_time_ms, final_label,
        )

        return [p for p in [csv_path, png_path] if p]

    def _generate_png(
        self,
        month: str,
        barrier: float,
        times_ms: List[int],
        model_probs: List[float],
        market_probs: List[Optional[float]],
        btc_prices: List[float],
        touched_time_ms: Optional[int],
        final_label: int,
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

        fig, ax_prob = plt.subplots(figsize=(16, 7))

        # === 左 Y 轴: 概率 ===
        ax_prob.plot(dates, model_probs, color="royalblue", linewidth=1.2,
                     label="Model p_touch", alpha=0.9)

        # Polymarket 价格
        mkt_x, mkt_y = [], []
        for i, mp in enumerate(market_probs):
            if mp is not None and 0 < mp < 1:
                mkt_x.append(dates[i])
                mkt_y.append(mp)
        if mkt_x:
            ax_prob.plot(mkt_x, mkt_y, color="darkorange", linewidth=1.0,
                         label="Polymarket", alpha=0.85)

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

        # 标注 "TOUCHED" 时刻
        if touched_time_ms is not None:
            touched_dt = datetime.fromtimestamp(touched_time_ms / 1000, tz=UTC)
            ax_prob.axvline(x=touched_dt, color="green", linestyle="--",
                           linewidth=1.5, alpha=0.7, label="TOUCHED")

        ax_prob.set_ylabel("Probability", fontsize=11)
        ax_prob.set_ylim(-0.02, 1.02)

        # === 右 Y 轴: BTC 价格 ===
        ax_btc = ax_prob.twinx()
        ax_btc.plot(dates, btc_prices, color="gray", linewidth=0.8,
                    alpha=0.5, label="BTC Price")
        ax_btc.axhline(y=barrier, color="red", linestyle="--", linewidth=1.0,
                       alpha=0.6, label=f"Barrier ${barrier:,.0f}")
        ax_btc.set_ylabel("BTC Price (USDT)", fontsize=11)

        # BTC Y 轴范围
        btc_arr = np.array(btc_prices)
        price_range = max(abs(btc_arr.max() - barrier), abs(btc_arr.min() - barrier), 2000)
        margin = price_range * 0.3
        ax_btc.set_ylim(barrier - price_range - margin, barrier + price_range + margin)

        # === 格式 ===
        barrier_str = f"{barrier:,.0f}"
        settled_str = "YES (touched)" if final_label == 1 else "NO"
        direction = "above" if barrier > btc_prices[0] else "below"
        ax_prob.set_title(
            f"Touch ${barrier_str} ({direction}) in {month} — settled: {settled_str}",
            fontsize=14,
        )
        ax_prob.set_xlabel("Date (UTC)", fontsize=11)
        ax_prob.grid(True, alpha=0.3)

        # X 轴日期格式
        ax_prob.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_prob.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        fig.autofmt_xdate(rotation=30)

        # 合并图例
        lines_prob, labels_prob = ax_prob.get_legend_handles_labels()
        lines_btc, labels_btc = ax_btc.get_legend_handles_labels()
        ax_prob.legend(lines_prob + lines_btc, labels_prob + labels_btc,
                       loc="upper left", fontsize=9)

        # 保存
        out_dir = os.path.join(self.output_dir, month)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"touch_{int(barrier)}.png"
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return path

    def _write_csv(
        self,
        month: str,
        barrier: float,
        times_ms: List[int],
        model_probs: List[float],
        market_probs: List[Optional[float]],
        btc_prices: List[float],
    ) -> str:
        """输出原始数据 CSV"""
        out_dir = os.path.join(self.output_dir, month)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"touch_{int(barrier)}.csv"
        path = os.path.join(out_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_ms", "datetime_utc", "model_p_touch",
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
