# tests/test_allocator.py
import pytest
from src.model.allocator import compute_target_weights


@pytest.fixture
def config():
    return {
        "etf_universe": ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"],
        "allocation": {
            "bull":     {"SPY": 0.35, "QQQ": 0.30, "IWM": 0.15, "TLT": 0.05, "GLD": 0.05, "SHY": 0.10},
            "sideways": {"SPY": 0.15, "QQQ": 0.10, "IWM": 0.05, "TLT": 0.25, "GLD": 0.20, "SHY": 0.25},
            "bear":     {"SPY": 0.00, "QQQ": 0.00, "IWM": 0.00, "TLT": 0.35, "GLD": 0.30, "SHY": 0.35},
        },
        "risk": {"cash_floor": 0.10},
    }


def test_pure_bull_allocation(config):
    probs = {"bull": 1.0, "sideways": 0.0, "bear": 0.0}
    weights = compute_target_weights(probs, config)
    assert abs(weights["SPY"] - 0.35 * 0.90) < 1e-9
    assert abs(weights["_cash"] - 0.10) < 1e-9


def test_pure_bear_allocation(config):
    probs = {"bull": 0.0, "sideways": 0.0, "bear": 1.0}
    weights = compute_target_weights(probs, config)
    assert weights["SPY"] == 0.0
    assert weights["TLT"] > 0.0


def test_blended_allocation(config):
    probs = {"bull": 0.6, "sideways": 0.3, "bear": 0.1}
    weights = compute_target_weights(probs, config)
    expected_spy = 0.90 * (0.6 * 0.35 + 0.3 * 0.15 + 0.1 * 0.0)
    assert abs(weights["SPY"] - expected_spy) < 1e-9


def test_weights_sum_to_one(config):
    probs = {"bull": 0.5, "sideways": 0.3, "bear": 0.2}
    weights = compute_target_weights(probs, config)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_cash_floor_always_present(config):
    for probs in [
        {"bull": 1.0, "sideways": 0.0, "bear": 0.0},
        {"bull": 0.0, "sideways": 0.0, "bear": 1.0},
        {"bull": 0.33, "sideways": 0.34, "bear": 0.33},
    ]:
        weights = compute_target_weights(probs, config)
        assert weights["_cash"] >= config["risk"]["cash_floor"] - 1e-9
