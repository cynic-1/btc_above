"""
策略评审报告生成器测试
"""

import os
import tempfile

import pytest

from backtest.config import BacktestConfig
from backtest.strategy_report import (
    generate_strategy_report,
    _compute_pnl_by_bucket,
)

from .conftest import make_test_backtest_data


class TestGenerateStrategyReport:
    def test_returns_markdown_string(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        assert isinstance(report, str)
        assert len(report) > 100

    def test_contains_template_sections(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        assert "# 策略评审报告" in report
        assert "# 1. 执行摘要" in report
        assert "# 3. 数据范围与实验设置" in report
        assert "# 4. 预测表现" in report
        assert "# 5. 回测结果" in report
        assert "# 10. 上线建议" in report

    def test_fills_brier_and_logloss(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        brier = metrics["overall"]["brier_score"]
        log_loss = metrics["overall"]["log_loss"]
        assert f"{brier:.6f}" in report
        assert f"{log_loss:.6f}" in report

    def test_fills_portfolio_metrics(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        portfolio = metrics["overall"]["portfolio"]
        assert f"${portfolio['total_pnl']:,.2f}" in report

    def test_fills_sample_statistics(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        # 事件天数
        assert "| 事件天数 | 2 |" in report
        # 观测总数
        assert "| 观测总数 | 2 |" in report

    def test_fills_calibration_table(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        # 校准表应有 Bin/Predicted/Actual 列
        assert "| Bin | Predicted | Actual | Count |" in report

    def test_writes_file(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            generate_strategy_report(result, metrics, config)

            files = os.listdir(tmpdir)
            report_files = [f for f in files if f.startswith("report_") and f.endswith(".md")]
            assert len(report_files) == 1

    def test_new_metrics_present(self):
        """新增指标字段应出现在报告中"""
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        # Sharpe/Sortino/Calmar 行存在（可能有值或为空，可能带 per-event/短期 标注）
        assert "Sharpe" in report
        assert "Sortino" in report
        assert "Calmar" in report
        # AUC/ECE 行存在
        assert "| AUC |" in report
        assert "| ECE |" in report

    def test_time_bucket_pnl(self):
        result, metrics = make_test_backtest_data(obs2_minutes=400)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BacktestConfig(output_dir=tmpdir)
            report = generate_strategy_report(result, metrics, config)

        # 应包含时间段 PnL 表
        assert "| T-12h~6h" in report or "| T-1h~10m" in report


class TestComputePnlByBucket:
    def test_empty(self):
        result = _compute_pnl_by_bucket([], [])
        assert result == {}

    def test_with_trades(self):
        markets = [
            {
                "event_date": "2026-02-15",
                "settlement": "YES",
                "pnl": 100.0,
                "trades": [
                    {"obs_minutes": 30, "direction": "YES", "shares": 200, "market_price": 0.4},
                    {"obs_minutes": 400, "direction": "YES", "shares": 200, "market_price": 0.4},
                ],
            }
        ]
        result = _compute_pnl_by_bucket([], markets)
        assert "T-1h~10m" in result
        assert "T-12h~6h" in result
        assert result["T-1h~10m"]["n_trades"] == 1
        assert result["T-12h~6h"]["n_trades"] == 1
