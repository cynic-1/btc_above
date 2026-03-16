"""
Dome API 订单簿历史下载与缓存

从 Dome API 获取 Polymarket 历史订单簿快照，
提取 best_bid/best_ask 存为 npz 缓存，供 OrderbookPriceLookup 使用

API: GET /polymarket/orderbooks?token_id=...&start_time=...&end_time=...
认证: Bearer token (从 .env 的 BEARER_TOKEN 读取)
数据起始: 2025-10-14
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

DOME_API = "https://api.domeapi.io/v1"


class DomeOrderbookFetcher:
    """
    Dome API 订单簿历史获取器

    下载指定 token_id 的历史订单簿快照（分页），
    提取 best_bid/best_ask 保存为 npz 缓存
    """

    def __init__(
        self,
        bearer_token: str,
        cache_dir: str = "data/orderbook_cache",
        request_sleep: float = 0.3,
    ):
        self.bearer_token = bearer_token
        self.cache_dir = cache_dir
        self.request_sleep = request_sleep
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {bearer_token}"
        os.makedirs(cache_dir, exist_ok=True)

    def _npz_path(self, condition_id: str) -> str:
        return os.path.join(self.cache_dir, f"{condition_id[:16]}.npz")

    def has_cache(self, condition_id: str) -> bool:
        return os.path.exists(self._npz_path(condition_id))

    def is_cache_complete(
        self,
        condition_id: str,
        settlement_ms: int,
        tolerance_ms: int = 3_600_000,  # 1h
    ) -> bool:
        """
        检查缓存数据是否覆盖到结算时间附近

        完整性判定: 缓存中最后一条快照的时间戳 >= settlement_ms - tolerance_ms
        即数据至少覆盖到结算前 1h（默认），否则视为不完整

        Args:
            condition_id: 合约 ID
            settlement_ms: 结算时间 (UTC ms)
            tolerance_ms: 允许的最大缺口 (默认 1h)

        Returns:
            True = 数据完整，False = 缺失或不覆盖结算时段
        """
        path = self._npz_path(condition_id)
        if not os.path.exists(path):
            return False

        try:
            data = np.load(path)
            ts = data["timestamps_ms"]
        except Exception:
            return False

        if len(ts) == 0:
            return False

        last_ts = int(ts[-1])
        return last_ts >= settlement_ms - tolerance_ms

    def fetch_snapshots(
        self,
        token_id: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> List[dict]:
        """
        分页获取指定时间范围内的所有 orderbook snapshots

        自动处理 pagination_key，直到 has_more=False
        """
        all_snapshots: List[dict] = []
        pagination_key: Optional[str] = None
        page = 0

        while True:
            params: dict = {
                "token_id": token_id,
                "start_time": start_time_ms,
                "end_time": end_time_ms,
                "limit": 200,
            }
            if pagination_key:
                params["pagination_key"] = pagination_key

            try:
                resp = self.session.get(
                    f"{DOME_API}/polymarket/orderbooks",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Dome API 请求失败 (page={page}): {e}")
                break

            snapshots = data.get("snapshots", [])
            all_snapshots.extend(snapshots)
            page += 1

            pagination = data.get("pagination", {})
            if pagination.get("has_more") and pagination.get("pagination_key"):
                pagination_key = pagination["pagination_key"]
                time.sleep(self.request_sleep)
            else:
                break

        return all_snapshots

    def download_and_cache(
        self,
        condition_id: str,
        token_id: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> int:
        """
        下载订单簿快照，提取 best_bid/best_ask，保存为 npz

        Returns: 快照数量
        """
        snapshots = self.fetch_snapshots(token_id, start_time_ms, end_time_ms)

        path = self._npz_path(condition_id)

        if not snapshots:
            logger.info(f"无订单簿数据 (cid={condition_id[:16]})")
            np.savez_compressed(
                path,
                timestamps_ms=np.array([], dtype=np.int64),
                best_bids=np.array([], dtype=np.float64),
                best_asks=np.array([], dtype=np.float64),
            )
            return 0

        # 提取 best_bid (最高买价) / best_ask (最低卖价)
        timestamps = []
        best_bids = []
        best_asks = []

        for snap in snapshots:
            ts = snap.get("timestamp", 0)
            if ts <= 0:
                continue

            bids = snap.get("bids", [])
            asks = snap.get("asks", [])

            # bids 按 price 降序排列（API 返回最低在前，取 max）
            best_bid = 0.0
            for b in bids:
                p = float(b["price"])
                if p > best_bid:
                    best_bid = p

            # asks 按 price 升序排列，取 min（排除 0）
            best_ask = 0.0
            for a in asks:
                p = float(a["price"])
                if best_ask == 0.0 or p < best_ask:
                    best_ask = p

            timestamps.append(ts)
            best_bids.append(best_bid)
            best_asks.append(best_ask)

        if not timestamps:
            return 0

        # 按时间排序
        order = np.argsort(timestamps)
        ts_arr = np.array(timestamps, dtype=np.int64)[order]
        bid_arr = np.array(best_bids, dtype=np.float64)[order]
        ask_arr = np.array(best_asks, dtype=np.float64)[order]

        np.savez_compressed(
            path,
            timestamps_ms=ts_arr,
            best_bids=bid_arr,
            best_asks=ask_arr,
        )

        logger.info(
            f"缓存订单簿: {len(ts_arr)} 快照 (cid={condition_id[:16]})"
        )
        return len(ts_arr)
