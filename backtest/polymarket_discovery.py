"""
Polymarket 市场发现

通过 Gamma API 搜索 BTC "Will Bitcoin be above $K" 类型的二元合约，
解析 strike/日期，缓存发现结果
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# 月份名称 → 数字
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# 匹配 "$85,000" 或 "$85000" 格式的价格
_STRIKE_RE = re.compile(r"\$\s*([\d,]+)")
# 匹配 "on March 5" 格式的日期
_DATE_RE = re.compile(
    r"on\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{1,2})",
    re.IGNORECASE,
)


@dataclass
class PolymarketMarketInfo:
    """Polymarket 市场信息"""
    event_date: str       # "YYYY-MM-DD"
    strike: float         # 行权价 85000.0
    condition_id: str
    yes_token_id: str
    no_token_id: str
    question: str


def parse_strike_from_question(question: str) -> Optional[float]:
    """
    从问题中解析行权价
    'Will Bitcoin be above $85,000 on March 5?' → 85000.0
    """
    m = _STRIKE_RE.search(question)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_date_from_question(question: str, year: int = 2026) -> Optional[str]:
    """
    从问题中解析事件日期
    '... on March 5?' → '2026-03-05'
    """
    m = _DATE_RE.search(question)
    if not m:
        return None
    month_name = m.group(1).lower()
    day = int(m.group(2))
    month = _MONTH_MAP.get(month_name)
    if month is None:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def snap_strike(
    k: float,
    available: List[float],
    max_diff: float = 250.0,
) -> Optional[float]:
    """将回测 K snap 到最近的 Polymarket K，差距过大返回 None"""
    if not available:
        return None
    best = min(available, key=lambda x: abs(x - k))
    if abs(best - k) > max_diff:
        return None
    return best


def _load_discovery_cache(cache_path: str) -> Dict[str, PolymarketMarketInfo]:
    """加载发现缓存，key = "date|strike" """
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r") as f:
            raw = json.load(f)
        result = {}
        for key, val in raw.items():
            result[key] = PolymarketMarketInfo(**val)
        return result
    except Exception as e:
        logger.warning(f"加载发现缓存失败: {e}")
        return {}


def _save_discovery_cache(
    cache: Dict[str, PolymarketMarketInfo],
    cache_path: str,
) -> None:
    """保存发现缓存"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw = {key: asdict(info) for key, info in cache.items()}
    with open(cache_path, "w") as f:
        json.dump(raw, f, indent=2)


def _cache_key(event_date: str, strike: float) -> str:
    return f"{event_date}|{strike:.0f}"


def _search_gamma_api(
    query: str,
    gamma_api: str,
    year: int = 2026,
) -> List[PolymarketMarketInfo]:
    """搜索 Gamma API 并解析返回的市场"""
    results = []

    for status_param in [None, "active"]:
        params = {"q": query}
        if status_param:
            params["events_status"] = status_param

        try:
            resp = requests.get(
                f"{gamma_api}/public-search",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Gamma API 搜索失败 (q={query}, status={status_param}): {e}")
            continue

        events = data.get("events", [])
        for event in events:
            markets = event.get("markets", [])
            for market in markets:
                question = market.get("question", "")
                # 只处理 Bitcoin above 类型
                if "bitcoin" not in question.lower() or "above" not in question.lower():
                    continue

                strike = parse_strike_from_question(question)
                event_date = parse_date_from_question(question, year=year)
                if strike is None or event_date is None:
                    continue

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

                info = PolymarketMarketInfo(
                    event_date=event_date,
                    strike=strike,
                    condition_id=condition_id,
                    yes_token_id=token_ids[0],
                    no_token_id=token_ids[1],
                    question=question,
                )
                results.append(info)

    return results


def discover_markets_for_range(
    start_date: str,
    end_date: str,
    gamma_api: str = "https://gamma-api.polymarket.com",
    cache_path: str = "data/polymarket/discovery_cache.json",
    year: int = 2026,
) -> Dict[Tuple[str, float], PolymarketMarketInfo]:
    """
    发现日期范围内所有 BTC "above $K" 合约

    Returns: {(event_date, strike): PolymarketMarketInfo}
    """
    # 加载缓存
    flat_cache = _load_discovery_cache(cache_path)

    # 计算需要搜索的日期
    from backtest.data_cache import _date_range
    all_dates = _date_range(start_date, end_date)

    # 检查哪些日期缓存中有数据
    cached_dates = set()
    for key in flat_cache:
        date_part = key.split("|")[0]
        cached_dates.add(date_part)

    missing_dates = [d for d in all_dates if d not in cached_dates]

    if missing_dates:
        logger.info(f"Polymarket 市场发现: 需搜索 {len(missing_dates)} 天")

        for date_str in missing_dates:
            # 构造搜索词: "Bitcoin above March 5" 格式
            from datetime import datetime
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            month_name = dt.strftime("%B")
            day = dt.day
            query = f"Bitcoin above {month_name} {day}"

            found = _search_gamma_api(query, gamma_api, year=year)
            for info in found:
                key = _cache_key(info.event_date, info.strike)
                if key not in flat_cache:
                    flat_cache[key] = info

            # 即使没找到也标记该日期已搜索（避免重复搜索）
            # 用一个哨兵 key
            if not any(k.startswith(f"{date_str}|") for k in flat_cache):
                # 不存入哨兵，下次还会搜索（符合预期：可能后续有新合约）
                pass

            # 限速
            time.sleep(0.3)

        # 保存缓存
        _save_discovery_cache(flat_cache, cache_path)
        logger.info(f"Polymarket 发现缓存更新: {len(flat_cache)} 条记录")

    # 转换为 (date, strike) → info 的字典
    result: Dict[Tuple[str, float], PolymarketMarketInfo] = {}
    for key, info in flat_cache.items():
        if info.event_date >= start_date and info.event_date < end_date:
            result[(info.event_date, info.strike)] = info

    logger.info(f"Polymarket 市场发现完成: {len(result)} 个合约 "
                f"(日期范围 {start_date} ~ {end_date})")
    return result
