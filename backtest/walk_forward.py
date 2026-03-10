"""
Walk-forward 分窗验证
将已有观测结果按时间窗分组，评估模型稳定性
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """单个 walk-forward 窗口结果"""
    window_id: int
    train_start: str  # YYYY-MM-DD
    train_end: str
    test_start: str
    test_end: str
    n_test_obs: int = 0
    brier: Optional[float] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    max_dd_pct: Optional[float] = None
    profit_factor: Optional[float] = None


@dataclass
class WalkForwardResult:
    """Walk-forward 验证汇总"""
    windows: List[WalkForwardWindow] = field(default_factory=list)
    aggregate_brier: Optional[float] = None
    aggregate_pnl: Optional[float] = None
    actual_params: Optional[Dict] = None  # 实际使用的窗口参数


class WalkForwardValidator:
    """
    Walk-forward 分窗验证器

    将观测按 event_date 分组，滚动产生 (train, test) 窗口，
    在每个 test 窗口上计算 Brier/PnL/Return/MaxDD/PF。
    """

    def __init__(
        self,
        train_days: int = 14,
        test_days: int = 7,
        step_days: int = 7,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def _adapt_window_params(self, total_days: int) -> tuple:
        """
        自适应窗口参数：数据不足时缩小窗口

        Returns:
            (train_days, test_days, step_days, adapted: bool)
        """
        train_days = self.train_days
        test_days = self.test_days
        step_days = self.step_days

        # 至少需要 train + 2*test 天才能产生 2 个窗口
        if total_days >= train_days + 2 * test_days:
            return train_days, test_days, step_days, False

        # 自适应缩小
        test_days = max(3, total_days // 5)
        train_days = max(5, total_days // 3)
        step_days = max(2, test_days // 2)

        logger.info(
            f"自适应 walk-forward 窗口: "
            f"train={train_days}, test={test_days}, step={step_days} "
            f"(总天数={total_days})"
        )
        return train_days, test_days, step_days, True

    def generate_windows(
        self, start_date: str, end_date: str
    ) -> List[tuple]:
        """
        产生滚动窗口 (train_start, train_end, test_start, test_end)
        日期格式 YYYY-MM-DD

        如果数据天数不足以按配置参数产生 ≥2 个窗口，自动缩小窗口参数。
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1

        train_days, test_days, step_days, adapted = self._adapt_window_params(total_days)
        self._actual_params = {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "adapted": adapted,
        }

        windows = []
        cursor = start

        while True:
            train_start = cursor
            train_end = cursor + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            if test_end > end + timedelta(days=1):
                break

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))
            cursor += timedelta(days=step_days)

        return windows

    def run(
        self,
        observations: list,
        initial_capital: float = 100_000.0,
        shares_per_trade: int = 200,
        max_net_shares: int = 10_000,
        entry_threshold: float = 0.03,
    ) -> WalkForwardResult:
        """
        在每个窗口的 test 期观测上计算指标

        Args:
            observations: 全部 ObservationResult 列表
            其他参数同 simulate_portfolio
        """
        from .metrics import brier_score, simulate_portfolio

        if not observations:
            return WalkForwardResult()

        # 获取日期范围
        dates = sorted(set(o.event_date for o in observations))
        if len(dates) < 2:
            logger.warning("事件日期不足，跳过 walk-forward")
            return WalkForwardResult()

        windows_spec = self.generate_windows(dates[0], dates[-1])
        if not windows_spec:
            logger.warning("无法生成 walk-forward 窗口")
            return WalkForwardResult()

        windows: List[WalkForwardWindow] = []
        all_briers = []
        total_pnl = 0.0

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows_spec):
            # 过滤 test 期观测
            test_obs = [
                o for o in observations
                if te_s <= o.event_date < te_e
            ]

            wf = WalkForwardWindow(
                window_id=i + 1,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                n_test_obs=len(test_obs),
            )

            if not test_obs:
                windows.append(wf)
                continue

            # Brier score
            preds, labels = [], []
            for obs in test_obs:
                for k in obs.k_grid:
                    p = obs.predictions.get(k)
                    y = obs.labels.get(k)
                    if p is not None and y is not None:
                        preds.append(p)
                        labels.append(y)

            if preds:
                wf.brier = float(brier_score(np.array(preds), np.array(labels)))
                all_briers.append(wf.brier)

            # Portfolio simulation
            portfolio = simulate_portfolio(
                test_obs,
                initial_capital=initial_capital,
                shares_per_trade=shares_per_trade,
                max_net_shares=max_net_shares,
                entry_threshold=entry_threshold,
            )
            wf.pnl = portfolio["total_pnl"]
            wf.return_pct = portfolio["total_return_pct"]
            wf.max_dd_pct = portfolio["max_drawdown_pct"]
            wf.profit_factor = portfolio["profit_factor"]
            total_pnl += portfolio["total_pnl"]

            windows.append(wf)

        result = WalkForwardResult(
            windows=windows,
            aggregate_brier=float(np.mean(all_briers)) if all_briers else None,
            aggregate_pnl=total_pnl,
            actual_params=getattr(self, '_actual_params', None),
        )

        logger.info(
            f"Walk-forward: {len(windows)} 窗口, "
            f"平均 Brier={result.aggregate_brier}, 总 PnL={total_pnl:.2f}"
        )
        return result
