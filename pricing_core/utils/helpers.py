"""
通用辅助函数模块
"""

import time
import threading
from typing import Any, Optional


def to_float(value: Any) -> Optional[float]:
    """安全转换为 float"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    """安全转换为 int"""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class TokenBucket:
    """
    令牌桶限速器

    Args:
        rate: 每秒允许的请求数
        capacity: 桶容量（突发上限）
    """

    def __init__(self, rate: float, capacity: float = 0.0):
        self.rate = rate
        self.capacity = capacity if capacity > 0 else rate
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """阻塞直到获取足够的令牌"""
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                # 计算需要等待的时间
                wait = (tokens - self.tokens) / self.rate
            time.sleep(wait)

    def _refill(self) -> None:
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
