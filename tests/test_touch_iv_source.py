"""
IV 数据源测试
"""

import gzip
import os
import tempfile

import pytest

from touch.iv_source import DeribitIVCache, DeribitIVSource


class TestDeribitIVCache:
    """DVOL 历史缓存测试"""

    def test_set_data_and_query(self):
        """直接设置数据后查询"""
        cache = DeribitIVCache()
        cache.set_data([
            (1000, 0.65),
            (2000, 0.70),
            (3000, 0.60),
        ])

        # 精确时刻
        assert cache.get_iv_at(1000) == 0.65
        assert cache.get_iv_at(2000) == 0.70
        assert cache.get_iv_at(3000) == 0.60

        # 中间时刻（forward-fill: 返回 <= timestamp 的最后值）
        assert cache.get_iv_at(1500) == 0.65
        assert cache.get_iv_at(2500) == 0.70

        # 之前 → None
        assert cache.get_iv_at(500) is None

        # 之后 → 最后一个
        assert cache.get_iv_at(5000) == 0.60

    def test_empty_data(self):
        """空数据查询返回 None"""
        cache = DeribitIVCache()
        assert cache.get_iv_at(1000) is None

    def test_percentage_conversion(self):
        """百分比形式自动转换为小数"""
        cache = DeribitIVCache()
        cache.set_data([
            (1000, 65.0),  # 百分比形式
            (2000, 0.70),  # 小数形式
        ])

        # 65.0 > 5.0 → 转换为 0.65
        assert cache.get_iv_at(1000) == 0.65
        # 0.70 < 5.0 → 保持
        assert cache.get_iv_at(2000) == 0.70

    def test_save_and_load(self):
        """保存和加载缓存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建并保存
            cache = DeribitIVCache(cache_dir=tmpdir)
            cache.set_data([
                (1000, 0.65),
                (2000, 0.70),
            ])

            # 手动写 CSV 模拟 download
            path = cache._file_path("BTC")
            with gzip.open(path, "wt", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["timestamp_ms", "volatility"])
                writer.writerow([1000, 0.65])
                writer.writerow([2000, 0.70])

            # 重新加载
            cache2 = DeribitIVCache(cache_dir=tmpdir)
            assert cache2.load("BTC") is True
            assert cache2.get_iv_at(1000) == 0.65
            assert cache2.get_iv_at(2000) == 0.70

    def test_load_nonexistent(self):
        """加载不存在的缓存返回 False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DeribitIVCache(cache_dir=tmpdir)
            assert cache.load("BTC") is False

    def test_no_lookahead(self):
        """确保不使用未来数据"""
        cache = DeribitIVCache()
        cache.set_data([
            (1000, 0.50),
            (2000, 0.60),
            (3000, 0.70),
        ])

        # 在 1500 时只能看到 1000 时刻的数据
        assert cache.get_iv_at(1500) == 0.50
        # 在 2000 时可以看到 2000 时刻的数据
        assert cache.get_iv_at(2000) == 0.60


class TestDeribitIVCacheATMIV:
    """ATM IV 缓存合并测试"""

    def test_load_atm_iv(self):
        """加载 ATM IV 缓存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DeribitIVCache(cache_dir=tmpdir)

            # 创建 ATM IV 缓存文件
            path = os.path.join(tmpdir, "atm_iv_2026-03.csv.gz")
            with gzip.open(path, "wt", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["timestamp_ms", "volatility"])
                writer.writerow([1000, 48.5])
                writer.writerow([2000, 49.0])

            assert cache.load_atm_iv("2026-03") is True
            assert cache.get_iv_at(1000) == pytest.approx(0.485)
            assert cache.get_iv_at(2000) == pytest.approx(0.49)

    def test_atm_iv_overrides_dvol(self):
        """ATM IV 数据合并后覆盖同期 DVOL"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DeribitIVCache(cache_dir=tmpdir)

            # 先设置 DVOL 数据
            cache.set_data([(1000, 0.50), (3000, 0.52)])

            # 创建 ATM IV 缓存（时间戳 2000 在 DVOL 的 1000~3000 之间）
            path = os.path.join(tmpdir, "atm_iv_2026-03.csv.gz")
            with gzip.open(path, "wt", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["timestamp_ms", "volatility"])
                writer.writerow([2000, 47.0])  # 百分比形式

            cache.load_atm_iv("2026-03")

            # 在 2500 时, 应使用 ATM IV (2000 时刻的 0.47) 而非 DVOL (1000 时刻的 0.50)
            assert cache.get_iv_at(2500) == pytest.approx(0.47)
            # 在 3500 时, 应使用 DVOL (3000 时刻的 0.52)
            assert cache.get_iv_at(3500) == pytest.approx(0.52)

    def test_atm_iv_not_found(self):
        """ATM IV 缓存不存在"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DeribitIVCache(cache_dir=tmpdir)
            assert cache.load_atm_iv("2026-03") is False


class TestDeribitIVSource:
    """实时 IV 获取测试"""

    def test_no_client(self):
        """无 DeribitClient 时返回 None"""
        source = DeribitIVSource(deribit_client=None)
        assert source.get_atm_iv(1000) is None
        assert source.get_dvol() is None
