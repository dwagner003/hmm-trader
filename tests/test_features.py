import pytest
import pandas as pd
import numpy as np
from src.data.features import compute_features


@pytest.fixture
def price_data_for_features():
    dates = pd.bdate_range("2024-01-02", periods=50)
    np.random.seed(42)
    spy_prices = 450.0 + np.cumsum(np.random.randn(50) * 2)
    tlt_prices = 100.0 + np.cumsum(np.random.randn(50) * 0.5)
    records = []
    for i, date in enumerate(dates):
        for ticker, prices in [("SPY", spy_prices), ("TLT", tlt_prices)]:
            records.append({
                "date": date, "ticker": ticker,
                "open": prices[i] - 0.5, "high": prices[i] + 1.0,
                "low": prices[i] - 1.0, "close": prices[i], "volume": 1000000,
            })
    return pd.DataFrame(records)


def test_compute_features_returns_expected_columns(price_data_for_features):
    features = compute_features(price_data_for_features)
    assert "date" in features.columns
    assert "spy_log_return" in features.columns
    assert "realized_vol" in features.columns
    assert "spy_tlt_spread" in features.columns


def test_compute_features_drops_nan_rows(price_data_for_features):
    features = compute_features(price_data_for_features)
    assert not features["realized_vol"].isna().any()
    assert not features["spy_log_return"].isna().any()
    assert len(features) == 30  # 50 - 20 rolling window


def test_compute_features_log_return_range(price_data_for_features):
    features = compute_features(price_data_for_features)
    assert features["spy_log_return"].abs().max() < 0.2


def test_compute_features_realized_vol_positive(price_data_for_features):
    features = compute_features(price_data_for_features)
    assert (features["realized_vol"] > 0).all()
