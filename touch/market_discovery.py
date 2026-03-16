"""
触碰合约市场发现

解析 Polymarket "What price will Bitcoin hit in March?" 类型事件，
提取各子合约的 barrier 价格和方向
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import requests

from .models import TouchMarketInfo

logger = logging.getLogger(__name__)

# 月份名称 → 数字
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# 匹配 "$90,000" / "$90000" / "$90K" / "$90k" 格式的价格
_PRICE_RE = re.compile(r"\$\s*([\d,]+)\s*([kK])?")

# 匹配方向:
#   "or above" / "or higher" / "reach" → up
#   "or below" / "or lower" / "dip to" / "dip below" / "drop to" / "fall to" → down
_DIRECTION_RE = re.compile(r"or\s+(above|higher|below|lower)", re.IGNORECASE)
_REACH_RE = re.compile(r"\breach\b", re.IGNORECASE)
_DIP_RE = re.compile(r"\b(dip|drop|fall|crash)\b", re.IGNORECASE)

# 从 slug 提取月份: "what-price-will-bitcoin-hit-in-march-2026"
_SLUG_MONTH_RE = re.compile(
    r"(?:in|for)-("
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december"
    r")-(\d{4})",
    re.IGNORECASE,
)


def parse_barrier_from_question(question: str) -> Optional[float]:
    """
    从问题文本解析障碍价格

    "$90,000 or above" → 90000.0
    "$75,000 or below" → 75000.0
    "$90K" / "$90k" → 90000.0
    """
    m = _PRICE_RE.search(question)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    # "$90K" → 90000
    if m.group(2):
        value *= 1000
    return value


def parse_direction_from_question(question: str) -> Optional[str]:
    """
    从问题文本解析方向

    "or above" / "or higher" / "reach" → "up"
    "or below" / "or lower" / "dip to" / "drop to" → "down"
    """
    m = _DIRECTION_RE.search(question)
    if m:
        word = m.group(1).lower()
        if word in ("above", "higher"):
            return "up"
        return "down"

    # "dip to $X" / "drop to $X" → down
    if _DIP_RE.search(question):
        return "down"

    # "reach $X" → up
    if _REACH_RE.search(question):
        return "up"

    return None


def parse_month_from_slug(slug: str) -> Optional[str]:
    """
    从 slug 提取月份

    "what-price-will-bitcoin-hit-in-march-2026" → "2026-03"
    """
    m = _SLUG_MONTH_RE.search(slug)
    if not m:
        return None
    month_name = m.group(1).lower()
    year = int(m.group(2))
    month_num = _MONTH_MAP.get(month_name)
    if month_num is None:
        return None
    return f"{year:04d}-{month_num:02d}"


def discover_touch_markets(
    slug: str,
    gamma_api: str = "https://gamma-api.polymarket.com",
    cache_path: Optional[str] = None,
) -> Dict[float, TouchMarketInfo]:
    """
    发现触碰合约市场

    通过 Gamma API 按 slug 查询事件，解析子合约的 barrier 和方向

    Args:
        slug: 事件 slug，如 "what-price-will-bitcoin-hit-in-march-2026"
        gamma_api: Gamma API base URL
        cache_path: 缓存文件路径（可选）

    Returns:
        {barrier: TouchMarketInfo}
    """
    # 尝试加载缓存
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
            result = {}
            for key, val in raw.items():
                info = TouchMarketInfo(**val)
                result[info.barrier] = info
            logger.info(f"加载触碰市场缓存: {len(result)} 个合约")
            return result
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")

    # 从 slug 提取月份
    month = parse_month_from_slug(slug)

    # 查询 Gamma API
    result: Dict[float, TouchMarketInfo] = {}

    for endpoint in ["events", "public-search"]:
        try:
            if endpoint == "events":
                resp = requests.get(
                    f"{gamma_api}/{endpoint}",
                    params={"slug": slug},
                    timeout=15,
                )
            else:
                resp = requests.get(
                    f"{gamma_api}/{endpoint}",
                    params={"q": slug},
                    timeout=15,
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Gamma API 查询失败 ({endpoint}): {e}")
            continue

        # 解析事件列表
        events = []
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict):
            events = data.get("events", [])
            # 单个事件也可能直接返回
            if not events and "markets" in data:
                events = [data]

        for event in events:
            event_slug = event.get("slug", "")
            # 从事件 slug 提取月份（如果还没有）
            if month is None:
                month = parse_month_from_slug(event_slug)

            markets = event.get("markets", [])
            for market in markets:
                question = market.get("question", "")

                barrier = parse_barrier_from_question(question)
                if barrier is None:
                    continue

                direction = parse_direction_from_question(question)
                if direction is None:
                    # 默认: 大于等于当前价格的为上触碰
                    direction = "up"

                # 解析 token IDs
                token_ids_raw = market.get("clobTokenIds", "[]")
                token_ids = (
                    json.loads(token_ids_raw)
                    if isinstance(token_ids_raw, str)
                    else token_ids_raw
                )
                if len(token_ids) < 2:
                    continue

                condition_id = market.get("conditionId", "")
                if not condition_id:
                    continue

                info = TouchMarketInfo(
                    month=month or "",
                    barrier=barrier,
                    direction=direction,
                    condition_id=condition_id,
                    yes_token_id=token_ids[0],
                    no_token_id=token_ids[1],
                    question=question,
                )
                result[barrier] = info

        if result:
            break  # 第一个有结果的 endpoint 即可

    logger.info(f"触碰市场发现: {len(result)} 个合约")

    # 保存缓存
    if cache_path and result:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw = {str(k): asdict(v) for k, v in result.items()}
        with open(cache_path, "w") as f:
            json.dump(raw, f, indent=2)
        logger.info(f"保存触碰市场缓存: {cache_path}")

    return result
