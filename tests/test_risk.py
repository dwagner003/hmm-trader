import pytest
from datetime import date
from src.trading.risk import (
    check_position_limits, check_rebalance_needed,
    check_drawdown, check_turnover, is_trading_day,
)


def test_check_position_limits_caps_excess():
    weights = {"SPY": 0.50, "QQQ": 0.20, "TLT": 0.10, "GLD": 0.10, "_cash": 0.10}
    capped = check_position_limits(weights, max_position_pct=0.40)
    assert capped["SPY"] <= 0.40
    assert abs(sum(capped.values()) - 1.0) < 1e-9


def test_check_position_limits_no_change_if_within():
    weights = {"SPY": 0.30, "QQQ": 0.25, "TLT": 0.25, "_cash": 0.20}
    capped = check_position_limits(weights, max_position_pct=0.40)
    assert capped == weights


def test_rebalance_needed_when_deviation_exceeds_threshold():
    current = {"SPY": 0.35, "QQQ": 0.25, "TLT": 0.20, "_cash": 0.20}
    target = {"SPY": 0.28, "QQQ": 0.25, "TLT": 0.27, "_cash": 0.20}
    assert check_rebalance_needed(current, target, threshold_pp=0.05) is True


def test_rebalance_not_needed_when_within_threshold():
    current = {"SPY": 0.30, "QQQ": 0.25, "TLT": 0.25, "_cash": 0.20}
    target = {"SPY": 0.32, "QQQ": 0.24, "TLT": 0.24, "_cash": 0.20}
    assert check_rebalance_needed(current, target, threshold_pp=0.05) is False


def test_drawdown_triggers_circuit_breaker():
    assert check_drawdown(8400.0, 10000.0, threshold=0.15) is True


def test_drawdown_does_not_trigger_within_threshold():
    assert check_drawdown(9100.0, 10000.0, threshold=0.15) is False


def test_turnover_within_limit():
    current = {"SPY": 0.30, "QQQ": 0.30, "TLT": 0.20, "_cash": 0.20}
    target = {"SPY": 0.25, "QQQ": 0.35, "TLT": 0.20, "_cash": 0.20}
    capped = check_turnover(current, target, max_turnover=0.50)
    assert capped == target


def test_turnover_exceeding_limit_is_scaled():
    current = {"SPY": 0.40, "QQQ": 0.00, "TLT": 0.40, "_cash": 0.20}
    target = {"SPY": 0.00, "QQQ": 0.40, "TLT": 0.00, "_cash": 0.60}
    capped = check_turnover(current, target, max_turnover=0.50)
    turnover = sum(abs(capped[k] - current[k]) for k in current)
    assert turnover <= 0.50 + 1e-9


def test_is_trading_day_weekday():
    assert is_trading_day(date(2024, 1, 3)) is True  # Wednesday


def test_is_trading_day_weekend():
    assert is_trading_day(date(2024, 1, 6)) is False  # Saturday


def test_is_trading_day_holiday():
    assert is_trading_day(date(2024, 1, 15)) is False  # MLK Day
