"""
Polymarket WebSocket orderbook 客户端

连接 Polymarket WS，订阅 YES token asset_ids，
接收 book snapshot + price_change 增量更新，
维护本地 orderbook 缓存
"""

import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import websocket

from .config import LiveTradingConfig
from .models import OrderBookLevel, OrderBookState

logger = logging.getLogger(__name__)

# 心跳间隔
_PING_INTERVAL = 10
# 重连退避参数
_RECONNECT_MIN = 1.0
_RECONNECT_MAX = 60.0


class PolymarketOrderbookWS:
    """
    Polymarket WebSocket orderbook 客户端

    功能:
    - 连接 wss://ws-subscriptions-clob.polymarket.com/ws/market
    - 订阅 YES token asset_ids
    - 接收 book snapshot → 初始化本地 orderbook
    - 接收 price_change delta → 增量更新
    - PING/PONG 心跳每 10s
    - 断线自动重连（指数退避 1s → 60s）
    - 线程安全的 orderbook 缓存
    """

    def __init__(self, config: LiveTradingConfig):
        self._config = config
        self._ws_url = config.polymarket_ws_url
        self._ws: Optional[websocket.WebSocketApp] = None
        self._asset_ids: List[str] = []
        self.orderbooks: Dict[str, OrderBookState] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
        self._connected = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._reconnect_delay = _RECONNECT_MIN

    def connect(self, asset_ids: List[str]) -> bool:
        """
        启动 WebSocket 连接并订阅

        Args:
            asset_ids: 要订阅的 YES token IDs

        Returns:
            是否成功启动
        """
        self._asset_ids = asset_ids
        self._running = True
        self._connected.clear()

        self._thread = threading.Thread(
            target=self._run_ws, daemon=True, name="polymarket-ws",
        )
        self._thread.start()

        # 等待连接建立（最多 15s）
        ok = self._connected.wait(timeout=15.0)
        if ok:
            logger.info(f"Polymarket WS 已连接，订阅 {len(asset_ids)} 个市场")
            self._reconnect_delay = _RECONNECT_MIN
        else:
            logger.warning("Polymarket WS 连接超时")
        return ok

    def close(self) -> None:
        """关闭连接"""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._connected.clear()
        logger.info("Polymarket WS 已关闭")

    def get_orderbook(self, asset_id: str) -> Optional[OrderBookState]:
        """获取某 asset_id 的 orderbook 快照（线程安全）"""
        with self._lock:
            ob = self.orderbooks.get(asset_id)
            if ob is None:
                return None
            # 返回副本
            return OrderBookState(
                asset_id=ob.asset_id,
                bids=list(ob.bids),
                asks=list(ob.asks),
                timestamp_ms=ob.timestamp_ms,
                best_bid=ob.best_bid,
                best_ask=ob.best_ask,
            )

    def add_callback(self, callback: Callable[[str, OrderBookState], None]) -> None:
        """注册 orderbook 变更回调"""
        self._callbacks.append(callback)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ==================== 内部方法 ====================

    def _run_ws(self) -> None:
        """WebSocket 运行循环（含自动重连）"""
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self._ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_close=self._on_close,
                    on_error=self._on_error,
                )
                self._ws.run_forever(ping_interval=0)  # 手动 ping
            except Exception as e:
                logger.error(f"Polymarket WS 异常: {e}")

            if not self._running:
                break

            # 重连退避
            self._connected.clear()
            logger.info(f"Polymarket WS 重连中 ({self._reconnect_delay:.1f}s)...")
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, _RECONNECT_MAX,
            )

    def _on_open(self, ws) -> None:
        """连接建立 → 订阅 + 启动心跳"""
        logger.info("Polymarket WS 连接已建立")
        self._connected.set()

        # 订阅所有 asset（一条消息，格式遵循官方文档）
        sub_msg = {
            "assets_ids": self._asset_ids,
            "type": "market",
        }
        try:
            ws.send(json.dumps(sub_msg))
            logger.info(f"已发送订阅消息，包含 {len(self._asset_ids)} 个 asset")
        except Exception as e:
            logger.error(f"订阅失败: {e}")

        # 启动心跳线程
        self._ping_thread = threading.Thread(
            target=self._ping_loop, daemon=True, name="polymarket-ping",
        )
        self._ping_thread.start()

    def _on_message(self, ws, message: str) -> None:
        """处理 book / price_change 消息"""
        # 服务端对 PING 回复纯文本 PONG
        if message == "PONG":
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.debug(f"Polymarket WS 非 JSON 消息: {message[:100]}")
            return

        # WS 可能返回单个对象或数组
        if isinstance(data, list):
            for item in data:
                self._dispatch(item)
        else:
            self._dispatch(data)

    def _dispatch(self, data: dict) -> None:
        """分发单条消息"""
        if not isinstance(data, dict):
            return

        # Polymarket WS 用 event_type 或 type
        msg_type = data.get("event_type") or data.get("type", "")

        if msg_type == "book":
            self._handle_book(data)
        elif msg_type == "price_change":
            self._handle_price_change(data)
        elif msg_type in ("pong", ""):
            pass
        else:
            logger.debug(f"Polymarket WS 未知消息类型: {msg_type}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        logger.info(f"Polymarket WS 连接关闭: code={close_status_code}, msg={close_msg}")
        self._connected.clear()

    def _on_error(self, ws, error) -> None:
        logger.error(f"Polymarket WS 错误: {error}")

    def _ping_loop(self) -> None:
        """每 10s 发送 PING 保持连接（官方文档要求纯文本 PING）"""
        while self._running and self._connected.is_set():
            try:
                if self._ws:
                    self._ws.send("PING")
            except Exception:
                break
            time.sleep(_PING_INTERVAL)

    def _handle_book(self, data: dict) -> None:
        """处理 book snapshot → 初始化本地 orderbook"""
        asset_id = data.get("asset_id", "")
        if not asset_id:
            return

        ob = self._parse_book(data)
        with self._lock:
            self.orderbooks[asset_id] = ob

        self._notify_callbacks(asset_id, ob)
        logger.debug(
            f"Book snapshot: asset={asset_id[:12]}..., "
            f"bids={len(ob.bids)}, asks={len(ob.asks)}, "
            f"best_bid={ob.best_bid:.4f}, best_ask={ob.best_ask:.4f}"
        )

    def _handle_price_change(self, data: dict) -> None:
        """处理 price_change delta → 增量更新

        官方文档格式:
        {
            "event_type": "price_change",
            "price_changes": [
                {"asset_id": "...", "price": "0.5", "size": "100", "side": "BUY"},
                ...
            ]
        }
        asset_id 在每个 change 项内，而非外层。
        """
        changes = data.get("price_changes", [])
        if not changes:
            return

        # 按 asset_id 分组，批量更新
        updated_assets = set()
        with self._lock:
            for change in changes:
                asset_id = change.get("asset_id", "")
                if not asset_id:
                    continue
                ob = self.orderbooks.get(asset_id)
                if ob is None:
                    continue
                self._apply_single_change(ob, change)
                updated_assets.add(asset_id)

        # 通知回调
        for asset_id in updated_assets:
            ob = self.orderbooks.get(asset_id)
            if ob:
                self._notify_callbacks(asset_id, ob)

    def _parse_book(self, data: dict) -> OrderBookState:
        """解析 book snapshot 为 OrderBookState"""
        asset_id = data.get("asset_id", "")
        now_ms = int(time.time() * 1000)

        bids = []
        for level in data.get("bids", []):
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
            if size > 0:
                bids.append(OrderBookLevel(price=price, size=size))
        bids.sort(key=lambda x: x.price, reverse=True)

        asks = []
        for level in data.get("asks", []):
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
            if size > 0:
                asks.append(OrderBookLevel(price=price, size=size))
        asks.sort(key=lambda x: x.price)

        best_bid = bids[0].price if bids else 0.0
        best_ask = asks[0].price if asks else 0.0

        return OrderBookState(
            asset_id=asset_id,
            bids=bids,
            asks=asks,
            timestamp_ms=now_ms,
            best_bid=best_bid,
            best_ask=best_ask,
        )

    def _apply_single_change(self, ob: OrderBookState, change: dict) -> None:
        """应用单条 price_change 到 orderbook"""
        side = change.get("side", "")
        price = float(change.get("price", 0))
        size = float(change.get("size", 0))

        if side == "BUY":
            ob.bids = [b for b in ob.bids if b.price != price]
            if size > 0:
                ob.bids.append(OrderBookLevel(price=price, size=size))
            ob.bids.sort(key=lambda x: x.price, reverse=True)
            ob.best_bid = ob.bids[0].price if ob.bids else 0.0
        elif side == "SELL":
            ob.asks = [a for a in ob.asks if a.price != price]
            if size > 0:
                ob.asks.append(OrderBookLevel(price=price, size=size))
            ob.asks.sort(key=lambda x: x.price)
            ob.best_ask = ob.asks[0].price if ob.asks else 0.0

        ob.timestamp_ms = int(time.time() * 1000)

    def _notify_callbacks(self, asset_id: str, ob: OrderBookState) -> None:
        """通知所有回调"""
        for cb in self._callbacks:
            try:
                cb(asset_id, ob)
            except Exception as e:
                logger.error(f"Orderbook 回调异常: {e}")
