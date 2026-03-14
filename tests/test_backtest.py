# tests/test_backtest.py
import pytest
import numpy as np
import pandas as pd
from src.backtest.engine import run_backtest


@pytest.fixture
def backtest_config():
    return {
        "mode": "backtest",
        "etf_universe": ["SPY", "TLT"],
        "hmm": {"n_states": 3, "training_window_years": 1, "retrain_frequency": "monthly"},
        "allocation": {
            "bull": {"SPY": 0.80, "TLT": 0.20},
            "sideways": {"SPY": 0.50, "TLT": 0.50},
            "bear": {"SPY": 0.20, "TLT": 0.80},
        },
        "risk": {
            "cash_floor": 0.10, "max_position_pct": 0.80,
            "rebalance_threshold_pp": 0.05, "max_daily_turnover": 0.50,
            "drawdown_circuit_breaker": 0.15,
        },
        "backtest": {
            "start_date": "2020-01-01", "end_date": "2020-12-31",
            "initial_capital": 10000, "transaction_cost": 1.0,
        },
        "data_dir": None,
    }


@pytest.fixture
def backtest_prices():
    """3 years of synthetic SPY and TLT prices."""
    dates = pd.bdate_range("2018-01-02", "2020-12-31")
    np.random.seed(42)
    spy_prices = 270.0 + np.cumsum(np.random.randn(len(dates)) * 1.5)
    tlt_prices = 120.0 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    records = []
    for i, date in enumerate(dates):
        for ticker, prices in [("SPY", spy_prices), ("TLT", tlt_prices)]:
            records.append({
                "date": date, "ticker": ticker, "open": prices[i],
                "high": prices[i] + 1.0, "low": prices[i] - 1.0,
                "close": prices[i], "volume": 1000000,
            })
    return pd.DataFrame(records)


def test_backtest_returns_results(backtest_config, backtest_prices, tmp_data_dir):
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert "total_return" in results
    assert "annualized_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert "win_rate" in results
    assert "equity_curve" in results
    assert "trade_log" in results
    assert "regime_history" in results


def test_backtest_equity_curve_starts_at_initial_capital(backtest_config, backtest_prices, tmp_data_dir):
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert abs(results["equity_curve"].iloc[0] - 10000.0) < 1.0


def test_backtest_max_drawdown_is_negative(backtest_config, backtest_prices, tmp_data_dir):
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert results["max_drawdown"] <= 0.0


def test_backtest_trade_log_not_empty(backtest_config, backtest_prices, tmp_data_dir):
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert len(results["trade_log"]) > 0
