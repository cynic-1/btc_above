"""
Binance 实时 K线流

双通道：
1. WebSocket: wss://stream.binance.com:9443/ws/btcusdt@kline_1m
2. REST 回退: GET /api/v3/klines

提供当前 BTC 价格 + 24h K线缓冲区
"""

import json
import logging
import threading
import time
from typing import List, Optional

import numpy as np
import requests

from pricing_core.binance_data import Kline
from pricing_core.utils.helpers import TokenBucket

from .config import LiveTradingConfig

logger = logging.getLogger(__name__)

# 24h 的分钟数
_24H_MINUTES = 24 * 60

# WebSocket 重连参数
_RECONNECT_MIN = 1.0
_RECONNECT_MAX = 30.0


class BinanceKlineFeed:
    """
    Binance 实时 K线流

    提供:
    - 当前 BTC 价格（最新 close）
    - 24h K线缓冲区（用于 HAR-RV 计算）
    - close 价格数组（用于 compute_log_returns）
    """

    def __init__(self, config: LiveTradingConfig):
        self._config = config
        self._ws_url = config.binance_ws_url
        self._rest_url = config.binance_rest_url.rstrip("/")
        self._limiter = TokenBucket(rate=config.binance_max_rps)

        self._buffer_minutes = config.har_train_days * 24 * 60

        self._klines: List[Kline] = []  # 滚动缓冲区（har_train_days 天）
        self._current_price: float = 0.0
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_delay = _RECONNECT_MIN

    def start(self) -> bool:
        """
        启动: REST 初始加载 24h + WebSocket 实时推送

        Returns:
            是否成功启动
        """
        # 1. REST 获取 24h 历史 K线
        try:
            self._load_initial_klines()
        except Exception as e:
            logger.error(f"Binance REST 初始加载失败: {e}")
            return False

        # 2. 启动 WebSocket
        self._running = True
        self._thread = threading.Thread(
            target=self._run_ws, daemon=True, name="binance-ws",
        )
        self._thread.start()

        # 等待 WS 连接（最多 10s）
        ok = self._connected.wait(timeout=10.0)
        if ok:
            logger.info(
                f"Binance K线流已启动: {len(self._klines)} 条历史, "
                f"当前价 {self._current_price:.2f}"
            )
        else:
            # WS 未连接也可以用 REST 数据，不阻塞
            logger.warning("Binance WS 连接超时，使用 REST 数据")

        return True

    def stop(self) -> None:
        """停止"""
        self._running = False
        self._connected.clear()
        # websocket-client 的 WebSocketApp 没有直接引用保存，
        # 通过 _running=False 让线程自然退出
        logger.info("Binance K线流已停止")

    def get_current_price(self) -> float:
        """获取最新 BTC 价格"""
        with self._lock:
            return self._current_price

    def get_klines(self) -> List[Kline]:
        """获取 24h K线缓冲区副本"""
        with self._lock:
            return list(self._klines)

    def get_close_prices(self) -> np.ndarray:
        """获取 close 价格数组"""
        with self._lock:
            return np.array([k.close for k in self._klines])

    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ==================== 内部方法 ====================

    def _load_initial_klines(self) -> None:
        """REST 获取历史 K线（har_train_days 天）"""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - self._buffer_minutes * 60 * 1000

        all_klines: List[Kline] = []
        cursor = start_ms

        while cursor < now_ms:
            self._limiter.acquire()
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": cursor,
                "endTime": now_ms,
                "limit": 1000,
            }
            resp = requests.get(
                f"{self._rest_url}/api/v3/klines",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                break

            for row in raw:
                kline = Kline(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time=int(row[6]),
                )
                all_klines.append(kline)

            cursor = int(raw[-1][0]) + 60_000
            if len(raw) < 1000:
                break

        with self._lock:
            self._klines = all_klines
            if all_klines:
                self._current_price = all_klines[-1].close

        logger.info(f"Binance REST 初始加载: {len(all_klines)} 条 K线")

    def _run_ws(self) -> None:
        """WebSocket 运行循环（含自动重连）"""
        import websocket as ws_lib

        while self._running:
            try:
                ws = ws_lib.WebSocketApp(
                    self._ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_kline_message,
                    on_close=self._on_ws_close,
                    on_error=self._on_ws_error,
                )
                ws.run_forever()
            except Exception as e:
                logger.error(f"Binance WS 异常: {e}")

            if not self._running:
                break

            self._connected.clear()
            logger.info(f"Binance WS 重连中 ({self._reconnect_delay:.1f}s)...")
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, _RECONNECT_MAX,
            )

    def _on_ws_open(self, ws) -> None:
        logger.info("Binance WS 连接已建立")
        self._connected.set()
        self._reconnect_delay = _RECONNECT_MIN

    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        logger.info(f"Binance WS 连接关闭: code={close_status_code}")
        self._connected.clear()

    def _on_ws_error(self, ws, error) -> None:
        logger.error(f"Binance WS 错误: {error}")

    def _on_kline_message(self, ws, message: str) -> None:
        """处理 WS kline 消息"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("e") != "kline":
            return

        k = data.get("k", {})
        self._update_kline(k)

    def _update_kline(self, k: dict) -> None:
        """更新/追加 K线到缓冲区"""
        open_time = int(k.get("t", 0))
        close_time = int(k.get("T", 0))
        is_closed = k.get("x", False)

        kline = Kline(
            open_time=open_time,
            open=float(k.get("o", 0)),
            high=float(k.get("h", 0)),
            low=float(k.get("l", 0)),
            close=float(k.get("c", 0)),
            volume=float(k.get("v", 0)),
            close_time=close_time,
        )

        with self._lock:
            # 更新当前价格
            self._current_price = kline.close

            if not self._klines:
                self._klines.append(kline)
                return

            last = self._klines[-1]

            if open_time == last.open_time:
                # 更新最后一根 K线
                self._klines[-1] = kline
            elif open_time > last.open_time:
                # 新 K线
                self._klines.append(kline)
                # 保持 har_train_days 窗口
                cutoff = open_time - self._buffer_minutes * 60 * 1000
                while self._klines and self._klines[0].open_time < cutoff:
                    self._klines.pop(0)
