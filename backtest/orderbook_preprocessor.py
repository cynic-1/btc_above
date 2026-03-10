"""
Parquet 订单簿数据 → npz 缓存预处理

将 data/polymarket_btc_above/*.parquet 中的 price_change 事件
提取为按 condition_id 分组的 (timestamps_ms, best_bids, best_asks) npz 文件
"""

import json
import logging
import os
import re
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 预编译正则: 从 data JSON 提取字段
_SIDE_RE = re.compile(r'"side"\s*:\s*"(YES|NO)"')
_TS_RE = re.compile(r'"timestamp"\s*:\s*([\d.]+)')
_BID_RE = re.compile(r'"best_bid"\s*:\s*"?([\d.]+|null)"?')
_ASK_RE = re.compile(r'"best_ask"\s*:\s*"?([\d.]+|null)"?')


def load_events_mapping(events_json: str) -> Dict[str, Dict]:
    """
    从 btc_above_events.json 构建 conditionId → {date, strike, question} 映射

    Returns: {conditionId: {"date": "YYYY-MM-DD", "strike": 85000.0, "question": "..."}}
    """
    with open(events_json, "r") as f:
        events = json.load(f)

    from backtest.polymarket_discovery import parse_strike_from_question

    mapping: Dict[str, Dict] = {}
    for date_str, event in events.items():
        for mkt in event.get("markets", []):
            cid = mkt.get("conditionId", "")
            question = mkt.get("question", "")
            if not cid:
                continue
            strike = parse_strike_from_question(question)
            if strike is None:
                continue
            mapping[cid] = {
                "date": date_str,
                "strike": strike,
                "question": question,
            }

    return mapping


def _extract_yes_rows_from_df(df, known_cids: set) -> Dict[str, np.ndarray]:
    """
    从 DataFrame 中向量化提取 YES price_change 行

    Returns: {condition_id: np.ndarray shape (N, 3) 列为 [ts_ms, bid, ask]}
    """
    # 向量化过滤: update_type + market_id
    mask = (df["update_type"] == "price_change") & (df["market_id"].isin(known_cids))
    pc_df = df.loc[mask, ["market_id", "data"]].copy()

    if len(pc_df) == 0:
        return {}

    # 向量化正则提取
    data_series = pc_df["data"]
    sides = data_series.str.extract(_SIDE_RE, expand=False)
    timestamps = data_series.str.extract(_TS_RE, expand=False)
    bids = data_series.str.extract(_BID_RE, expand=False)
    asks = data_series.str.extract(_ASK_RE, expand=False)

    # 过滤 YES + 有效 timestamp
    yes_mask = (sides == "YES") & timestamps.notna()
    pc_df = pc_df.loc[yes_mask].copy()
    timestamps = timestamps.loc[yes_mask]
    bids = bids.loc[yes_mask]
    asks = asks.loc[yes_mask]

    if len(pc_df) == 0:
        return {}

    # 转数值
    ts_arr = (timestamps.astype(float) * 1000).astype(np.int64).values
    bid_arr = bids.replace("null", "0").fillna("0").astype(float).values
    ask_arr = asks.replace("null", "0").fillna("0").astype(float).values
    mid_arr = pc_df["market_id"].values

    # 按 market_id 分组
    result: Dict[str, np.ndarray] = {}
    unique_mids = np.unique(mid_arr)
    for mid in unique_mids:
        idx = mid_arr == mid
        chunk = np.column_stack([ts_arr[idx], bid_arr[idx], ask_arr[idx]])
        result[mid] = chunk

    return result


def preprocess_parquet_files(
    parquet_dir: str = "data/polymarket_btc_above",
    cache_dir: str = "data/orderbook_cache",
    events_json: str = "data/btc_above_events.json",
) -> Dict[str, int]:
    """
    批量预处理 Parquet → npz 缓存

    内存优化: 逐文件处理，每个 cid 只保留 numpy 数组列表，
    最终 concatenate + 排序 + 去重

    Returns: {condition_id: row_count}
    """
    import pyarrow.parquet as pq

    # 加载已知市场
    events_mapping = load_events_mapping(events_json)
    known_cids = set(events_mapping.keys())
    logger.info(f"已知市场: {len(known_cids)} 个 conditionId")

    os.makedirs(cache_dir, exist_ok=True)

    # 收集: {condition_id: [np.ndarray(N,3), ...]}
    collected: Dict[str, List[np.ndarray]] = {}

    parquet_files = sorted(
        f for f in os.listdir(parquet_dir) if f.endswith(".parquet")
    )
    logger.info(f"发现 {len(parquet_files)} 个 Parquet 文件")

    total_parsed = 0
    for file_idx, filename in enumerate(parquet_files):
        filepath = os.path.join(parquet_dir, filename)
        try:
            table = pq.read_table(
                filepath,
                columns=["update_type", "market_id", "data"],
            )
            df = table.to_pandas()
        except Exception as e:
            logger.warning(f"读取 {filename} 失败: {e}")
            continue

        chunks = _extract_yes_rows_from_df(df, known_cids)
        del df, table  # 尽早释放

        file_parsed = 0
        for mid, arr in chunks.items():
            if mid not in collected:
                collected[mid] = []
            collected[mid].append(arr)
            file_parsed += len(arr)
        total_parsed += file_parsed

        if (file_idx + 1) % 50 == 0 or file_idx == len(parquet_files) - 1:
            logger.info(
                f"  [{file_idx + 1}/{len(parquet_files)}] {filename}: "
                f"{file_parsed} YES 行, 累计 {total_parsed}"
            )

    # 合并、排序、去重、写入 npz
    result: Dict[str, int] = {}
    for cid, chunks in collected.items():
        if not chunks:
            continue

        data = np.concatenate(chunks)
        # 按 timestamp_ms 排序
        order = np.argsort(data[:, 0])
        data = data[order]

        # 去重（相邻 timestamp 相同的保留最后一条）
        if len(data) > 1:
            # 找到 timestamp 变化的位置 + 最后一行
            ts = data[:, 0]
            keep = np.concatenate([ts[:-1] != ts[1:], [True]])
            data = data[keep]

        timestamps = data[:, 0].astype(np.int64)
        bids = data[:, 1]
        asks = data[:, 2]

        npz_path = os.path.join(cache_dir, f"{cid[:16]}.npz")
        np.savez_compressed(
            npz_path,
            timestamps_ms=timestamps,
            best_bids=bids,
            best_asks=asks,
        )
        result[cid] = len(timestamps)

    logger.info(
        f"预处理完成: {len(result)} 个市场, "
        f"共 {sum(result.values())} 条报价记录"
    )
    return result
