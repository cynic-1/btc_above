"""
尽调清单报告生成器测试
"""

import os
import tempfile

import pytest

from backtest.config import BacktestConfig
from backtest.dd_report import generate_dd_report, _compute_concentration

from .conftest import make_test_backtest_data


class TestGenerateDdReport:
    def test_returns_markdown_string(self):
        result, metrics = make_test_backtest_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_dd_report(result, metrics, config)

        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_template_sections(self):
        result, metrics = make_test_backtest_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_dd_report(result, metrics, config)

        assert "# 预测市场量化策略尽调清单" in report
        assert "## 0. 策略摘要" in report
        assert "## 1. 数据与时间戳审查" in report
        assert "## 2. 标签与样本构造审查" in report
        assert "## 4. 模型训练与预测审查" in report
        assert "## 6. 风险与组合审查" in report
        assert "## 10. 上线建议与最终结论" in report

    def test_fills_brier_score(self):
        result, metrics = make_test_backtest_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_dd_report(result, metrics, config)

        brier = metrics["overall"]["brier_score"]
        assert f"{brier:.6f}" in report

    def test_fills_date_range(self):
        result, metrics = make_test_backtest_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_dd_report(result, metrics, config)

        assert "2026-02-15" in report
        assert "2026-02-16" in report

    def test_writes_file(self):
        result, metrics = make_test_backtest_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            generate_dd_report(result, metrics, config)

            files = os.listdir(tmpdir)
            dd_files = [f for f in files if f.startswith("dd_") and f.endswith(".md")]
            assert len(dd_files) == 1

    def test_risk_metrics_present(self):
        """风险指标行应存在（可能有值或为空）"""
        result, metrics = make_test_backtest_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_dd_report(result, metrics, config)

        assert "Sharpe" in report
        assert "Calmar" in report


class TestComputeConcentration:
    def test_empty_markets(self):
        result = _compute_concentration([], 0)
        assert result["top5_pct"] == ""
        assert result["max_daily_loss"] == ""

    def test_with_markets(self):
        markets = [
            {"event_date": "2026-02-15", "pnl": 100.0},
            {"event_date": "2026-02-15", "pnl": -50.0},
            {"event_date": "2026-02-16", "pnl": 200.0},
        ]
        result = _compute_concentration(markets, 250.0)
        assert "%" in result["top5_pct"]
        assert "$" in result["pnl_ex_top5"]
        assert "$" in result["max_daily_loss"]
        assert "$" in result["max_market_loss"]
