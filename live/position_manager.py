"""
仓位追踪 + 风险限制

按 condition_id 跟踪每个市场的 YES/NO 仓位，
检查净仓位限制、总成本限制
"""

import logging
import threading
from typing import Dict, Tuple

from .config import LiveTradingConfig
from .models import Position, Signal

logger = logging.getLogger(__name__)


class PositionManager:
    """
    仓位追踪 + 风险限制

    - 按 condition_id 跟踪每个市场的 YES/NO 仓位
    - 检查净仓位限制 (max_net_shares)
    - 检查总成本限制 (max_total_cost)
    - 下单前验证是否允许（filled + pending 合计）
    - pending 订单超时后可取消释放额度
    """

    def __init__(self, config: LiveTradingConfig):
        self._config = config
        self._positions: Dict[str, Position] = {}  # filled 仓位
        self._pending: Dict[str, Signal] = {}  # order_id → Signal（待成交）
        self._total_cost: float = 0.0
        self._lock = threading.Lock()

    def _pending_net_for(self, condition_id: str) -> int:
        """计算某市场 pending 订单的净仓位"""
        net = 0
        for sig in self._pending.values():
            if sig.condition_id == condition_id:
                if sig.direction == "YES" and sig.side == "BUY":
                    net += sig.shares
                elif sig.direction == "NO" and sig.side == "BUY":
                    net -= sig.shares
        return net

    def can_trade(self, signal: Signal) -> Tuple[bool, str]:
        """
        检查是否允许下单（考虑 filled + pending 仓位）

        Args:
            signal: 交易信号

        Returns:
            (是否允许, 原因)
        """
        with self._lock:
            # 1. 检查总成本限制
            trade_cost = signal.shares * signal.price
            if self._total_cost + trade_cost > self._config.max_total_cost:
                return False, (
                    f"总成本超限: 当前={self._total_cost:.2f}, "
                    f"新增={trade_cost:.2f}, "
                    f"限制={self._config.max_total_cost:.2f}"
                )

            # 2. 检查净仓位限制（filled + pending）
            pos = self._positions.get(signal.condition_id)
            filled_net = pos.net_shares if pos is not None else 0
            pending_net = self._pending_net_for(signal.condition_id)
            current_net = filled_net + pending_net

            if signal.direction == "YES" and signal.side == "BUY":
                new_net = current_net + signal.shares
            elif signal.direction == "NO" and signal.side == "BUY":
                new_net = current_net - signal.shares
            else:
                new_net = current_net

            if abs(new_net) > self._config.max_net_shares:
                n_pending = sum(1 for s in self._pending.values()
                                if s.condition_id == signal.condition_id)
                return False, (
                    f"净仓位超限: filled={filled_net}, pending={pending_net}, "
                    f"新净仓={new_net}, pending单数={n_pending}, "
                    f"限制=±{self._config.max_net_shares}"
                )

            return True, "OK"

    def add_pending(self, signal: Signal, order_id: str) -> None:
        """GTC 订单已下单但未成交，加入 pending 追踪（占用仓位额度）"""
        with self._lock:
            self._pending[order_id] = signal
            logger.info(
                f"Pending 挂单: order_id={order_id}, K={signal.strike:.0f} "
                f"{signal.direction} {signal.shares}份"
            )

    def confirm_fill(self, order_id: str) -> None:
        """Pending 订单已成交，转为 filled 仓位"""
        with self._lock:
            signal = self._pending.pop(order_id, None)
            if signal is None:
                logger.warning(f"confirm_fill: order_id={order_id} 不在 pending 中")
                return
        # 调用 record_trade 更新 filled 仓位（需要在 lock 外调用以避免死锁）
        self.record_trade(signal, "matched")

    def cancel_pending(self, order_id: str) -> None:
        """Pending 订单已取消，释放占用的仓位额度"""
        with self._lock:
            signal = self._pending.pop(order_id, None)
            if signal is None:
                logger.warning(f"cancel_pending: order_id={order_id} 不在 pending 中")
                return
            logger.info(
                f"Pending 释放: order_id={order_id}, K={signal.strike:.0f} "
                f"{signal.direction} {signal.shares}份"
            )

    def record_trade(self, signal: Signal, status: str) -> None:
        """
        成交后更新仓位

        Args:
            signal: 交易信号
            status: 成交状态 ("matched" / "live" / "failed")
        """
        if status == "failed":
            return

        with self._lock:
            pos = self._positions.get(signal.condition_id)
            if pos is None:
                pos = Position(
                    condition_id=signal.condition_id,
                    strike=signal.strike,
                )
                self._positions[signal.condition_id] = pos

            trade_cost = signal.shares * signal.price

            if signal.direction == "YES" and signal.side == "BUY":
                pos.yes_shares += signal.shares
                pos.yes_cost += trade_cost
            elif signal.direction == "NO" and signal.side == "BUY":
                pos.no_shares += signal.shares
                pos.no_cost += trade_cost

            self._total_cost += trade_cost

            logger.info(
                f"仓位更新: K={signal.strike:.0f}, "
                f"YES={pos.yes_shares}, NO={pos.no_shares}, "
                f"净仓={pos.net_shares}, "
                f"总成本={self._total_cost:.2f}"
            )

    def get_position(self, condition_id: str) -> Position:
        """获取某市场仓位"""
        with self._lock:
            pos = self._positions.get(condition_id)
            if pos is None:
                return Position(condition_id=condition_id, strike=0.0)
            return Position(
                condition_id=pos.condition_id,
                strike=pos.strike,
                yes_shares=pos.yes_shares,
                yes_cost=pos.yes_cost,
                no_shares=pos.no_shares,
                no_cost=pos.no_cost,
            )

    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有仓位（副本）"""
        with self._lock:
            return {
                cid: Position(
                    condition_id=p.condition_id,
                    strike=p.strike,
                    yes_shares=p.yes_shares,
                    yes_cost=p.yes_cost,
                    no_shares=p.no_shares,
                    no_cost=p.no_cost,
                )
                for cid, p in self._positions.items()
            }

    def get_total_cost(self) -> float:
        """获取总成本"""
        with self._lock:
            return self._total_cost

    def summary(self) -> str:
        """仓位汇总字符串"""
        with self._lock:
            if not self._positions:
                return "无持仓"

            lines = [f"总成本: ${self._total_cost:.2f}"]
            for cid, pos in self._positions.items():
                lines.append(
                    f"  K={pos.strike:.0f}: "
                    f"YES={pos.yes_shares}(${pos.yes_cost:.2f}), "
                    f"NO={pos.no_shares}(${pos.no_cost:.2f}), "
                    f"净仓={pos.net_shares}"
                )
            return "\n".join(lines)
