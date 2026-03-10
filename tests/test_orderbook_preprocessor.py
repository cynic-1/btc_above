"""
orderbook_preprocessor 测试

使用合成 Parquet 数据验证预处理流程
"""

import json
import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def temp_dirs():
    """创建临时目录结构"""
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_dir = os.path.join(tmpdir, "parquets")
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(parquet_dir)
        yield tmpdir, parquet_dir, cache_dir


@pytest.fixture
def events_json(temp_dirs):
    """创建合成 events.json"""
    tmpdir = temp_dirs[0]
    path = os.path.join(tmpdir, "events.json")
    events = {
        "2026-02-21": {
            "title": "Bitcoin above ___ on February 21?",
            "markets": [
                {
                    "question": "Will the price of Bitcoin be above $85,000 on February 21?",
                    "conditionId": "0xaabbccdd11223344",
                    "clobTokenIds": ["token_yes_1", "token_no_1"],
                },
                {
                    "question": "Will the price of Bitcoin be above $90,000 on February 21?",
                    "conditionId": "0x1122334455667788",
                    "clobTokenIds": ["token_yes_2", "token_no_2"],
                },
            ],
        }
    }
    with open(path, "w") as f:
        json.dump(events, f)
    return path


def _make_parquet(parquet_dir: str, filename: str, rows: list):
    """构造合成 Parquet 文件"""
    import pandas as pd
    from datetime import datetime, timezone

    records = []
    for update_type, market_id, side, ts, bid, ask in rows:
        data = json.dumps({
            "update_type": update_type,
            "market_id": market_id,
            "side": side,
            "best_bid": str(bid) if bid is not None else None,
            "best_ask": str(ask) if ask is not None else None,
            "timestamp": ts,
        })
        records.append({
            "timestamp_received": pd.Timestamp(datetime(2026, 2, 21, 17, 0, 0, tzinfo=timezone.utc)),
            "timestamp_created_at": pd.Timestamp(datetime(2026, 2, 21, 17, 0, 0, tzinfo=timezone.utc)),
            "market_id": market_id,
            "update_type": update_type,
            "data": data,
        })

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, os.path.join(parquet_dir, filename))


def test_preprocess_basic(temp_dirs, events_json):
    """基础预处理: YES 行被提取，NO 行被过滤"""
    _, parquet_dir, cache_dir = temp_dirs
    cid1 = "0xaabbccdd11223344"

    _make_parquet(parquet_dir, "test1.parquet", [
        # YES price_change — 应保留
        ("price_change", cid1, "YES", 1000.0, 0.45, 0.55),
        ("price_change", cid1, "YES", 1001.0, 0.46, 0.56),
        # NO price_change — 应过滤
        ("price_change", cid1, "NO", 1002.0, 0.44, 0.54),
        # book_snapshot — 应过滤
        ("book_snapshot", cid1, "YES", 1003.0, 0.47, 0.57),
    ])

    from backtest.orderbook_preprocessor import preprocess_parquet_files
    result = preprocess_parquet_files(
        parquet_dir=parquet_dir,
        cache_dir=cache_dir,
        events_json=events_json,
    )

    assert cid1 in result
    assert result[cid1] == 2  # 只有 2 条 YES price_change

    # 验证 npz 内容
    npz = np.load(os.path.join(cache_dir, f"{cid1[:16]}.npz"))
    assert len(npz["timestamps_ms"]) == 2
    assert npz["timestamps_ms"][0] == 1000000  # 1000.0 * 1000
    assert npz["timestamps_ms"][1] == 1001000
    np.testing.assert_allclose(npz["best_bids"], [0.45, 0.46])
    np.testing.assert_allclose(npz["best_asks"], [0.55, 0.56])


def test_preprocess_dedup(temp_dirs, events_json):
    """相同时间戳去重，保留最后一条"""
    _, parquet_dir, cache_dir = temp_dirs
    cid1 = "0xaabbccdd11223344"

    _make_parquet(parquet_dir, "test1.parquet", [
        ("price_change", cid1, "YES", 1000.0, 0.40, 0.50),
        ("price_change", cid1, "YES", 1000.0, 0.45, 0.55),  # 相同时间戳
        ("price_change", cid1, "YES", 1001.0, 0.46, 0.56),
    ])

    from backtest.orderbook_preprocessor import preprocess_parquet_files
    result = preprocess_parquet_files(
        parquet_dir=parquet_dir,
        cache_dir=cache_dir,
        events_json=events_json,
    )

    assert result[cid1] == 2  # 去重后只剩 2 条
    npz = np.load(os.path.join(cache_dir, f"{cid1[:16]}.npz"))
    # 保留最后一条
    np.testing.assert_allclose(npz["best_bids"][0], 0.45)


def test_preprocess_unknown_market_filtered(temp_dirs, events_json):
    """未知 conditionId 被过滤"""
    _, parquet_dir, cache_dir = temp_dirs

    _make_parquet(parquet_dir, "test1.parquet", [
        ("price_change", "0xunknown_market_id", "YES", 1000.0, 0.45, 0.55),
    ])

    from backtest.orderbook_preprocessor import preprocess_parquet_files
    result = preprocess_parquet_files(
        parquet_dir=parquet_dir,
        cache_dir=cache_dir,
        events_json=events_json,
    )

    assert len(result) == 0


def test_preprocess_multi_file_sorted(temp_dirs, events_json):
    """跨文件数据正确排序"""
    _, parquet_dir, cache_dir = temp_dirs
    cid1 = "0xaabbccdd11223344"

    # 第二个文件的时间戳比第一个早
    _make_parquet(parquet_dir, "b_later.parquet", [
        ("price_change", cid1, "YES", 2000.0, 0.50, 0.60),
    ])
    _make_parquet(parquet_dir, "a_earlier.parquet", [
        ("price_change", cid1, "YES", 1000.0, 0.45, 0.55),
    ])

    from backtest.orderbook_preprocessor import preprocess_parquet_files
    result = preprocess_parquet_files(
        parquet_dir=parquet_dir,
        cache_dir=cache_dir,
        events_json=events_json,
    )

    npz = np.load(os.path.join(cache_dir, f"{cid1[:16]}.npz"))
    # 应按时间排序
    assert npz["timestamps_ms"][0] < npz["timestamps_ms"][1]
