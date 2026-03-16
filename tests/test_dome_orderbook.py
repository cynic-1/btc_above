"""
DomeOrderbookFetcher 缓存完整性检查测试
"""

import os
import tempfile

import numpy as np
import pytest

from backtest.dome_orderbook import DomeOrderbookFetcher


class TestIsCacheComplete:
    """is_cache_complete 完整性检查"""

    def _make_fetcher(self, tmpdir: str) -> DomeOrderbookFetcher:
        return DomeOrderbookFetcher(
            bearer_token="test",
            cache_dir=tmpdir,
            request_sleep=0,
        )

    def _write_npz(self, tmpdir: str, cid: str, timestamps: list):
        """写入测试 npz 文件"""
        path = os.path.join(tmpdir, f"{cid[:16]}.npz")
        ts = np.array(timestamps, dtype=np.int64)
        bids = np.full(len(timestamps), 0.5, dtype=np.float64)
        asks = np.full(len(timestamps), 0.6, dtype=np.float64)
        np.savez_compressed(path, timestamps_ms=ts, best_bids=bids, best_asks=asks)

    def test_complete_cache(self):
        """最后快照在结算前 1h 内 → 完整"""
        tmpdir = tempfile.mkdtemp()
        cid = "cid_complete_001"
        settlement_ms = 100_000_000
        # 最后快照在结算前 30min
        self._write_npz(tmpdir, cid, [
            settlement_ms - 3_600_000,  # T-60m
            settlement_ms - 1_800_000,  # T-30m → 在 tolerance 内
        ])

        fetcher = self._make_fetcher(tmpdir)
        assert fetcher.is_cache_complete(cid, settlement_ms) is True

    def test_incomplete_cache(self):
        """最后快照距结算 > 1h → 不完整"""
        tmpdir = tempfile.mkdtemp()
        cid = "cid_incomplete01"
        settlement_ms = 100_000_000
        # 最后快照在结算前 3.7h
        self._write_npz(tmpdir, cid, [
            settlement_ms - 20_000_000,
            settlement_ms - 13_320_000,  # T-3.7h = 13,320,000ms
        ])

        fetcher = self._make_fetcher(tmpdir)
        assert fetcher.is_cache_complete(cid, settlement_ms) is False

    def test_no_cache_file(self):
        """文件不存在 → 不完整"""
        tmpdir = tempfile.mkdtemp()
        fetcher = self._make_fetcher(tmpdir)
        assert fetcher.is_cache_complete("nonexistent_cid", 100_000_000) is False

    def test_empty_cache(self):
        """空 npz (0 条快照) → 不完整"""
        tmpdir = tempfile.mkdtemp()
        cid = "cid_empty_cache1"
        self._write_npz(tmpdir, cid, [])

        fetcher = self._make_fetcher(tmpdir)
        assert fetcher.is_cache_complete(cid, 100_000_000) is False

    def test_exact_tolerance_boundary(self):
        """最后快照恰好在 tolerance 边界 → 完整"""
        tmpdir = tempfile.mkdtemp()
        cid = "cid_boundary_001"
        settlement_ms = 100_000_000
        tolerance_ms = 3_600_000  # 1h
        # 最后快照恰好 = settlement - tolerance
        self._write_npz(tmpdir, cid, [settlement_ms - tolerance_ms])

        fetcher = self._make_fetcher(tmpdir)
        assert fetcher.is_cache_complete(cid, settlement_ms, tolerance_ms) is True

    def test_one_ms_over_tolerance(self):
        """最后快照比 tolerance 多 1ms → 不完整"""
        tmpdir = tempfile.mkdtemp()
        cid = "cid_over_tol_001"
        settlement_ms = 100_000_000
        tolerance_ms = 3_600_000
        self._write_npz(tmpdir, cid, [settlement_ms - tolerance_ms - 1])

        fetcher = self._make_fetcher(tmpdir)
        assert fetcher.is_cache_complete(cid, settlement_ms, tolerance_ms) is False

    def test_custom_tolerance(self):
        """自定义 tolerance = 2h"""
        tmpdir = tempfile.mkdtemp()
        cid = "cid_custom_tol_1"
        settlement_ms = 100_000_000
        # 最后快照在结算前 90min，默认 1h tolerance 会判不完整
        self._write_npz(tmpdir, cid, [settlement_ms - 5_400_000])

        fetcher = self._make_fetcher(tmpdir)
        # 默认 1h tolerance → 不完整
        assert fetcher.is_cache_complete(cid, settlement_ms) is False
        # 2h tolerance → 完整
        assert fetcher.is_cache_complete(cid, settlement_ms, tolerance_ms=7_200_000) is True

    def test_data_after_settlement(self):
        """有结算后数据 → 完整"""
        tmpdir = tempfile.mkdtemp()
        cid = "cid_post_settle1"
        settlement_ms = 100_000_000
        # 最后快照在结算后
        self._write_npz(tmpdir, cid, [
            settlement_ms - 60_000,
            settlement_ms + 1_800_000,  # 结算后 30min
        ])

        fetcher = self._make_fetcher(tmpdir)
        assert fetcher.is_cache_complete(cid, settlement_ms) is True
