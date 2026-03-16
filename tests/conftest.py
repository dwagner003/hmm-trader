# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    return str(tmp_path)


@pytest.fixture
def sample_prices():
    """Sample OHLCV DataFrame for 30 trading days, 2 tickers."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    np.random.seed(42)
    records = []
    for ticker in ["SPY", "TLT"]:
        base = 450.0 if ticker == "SPY" else 100.0
        prices = base + np.cumsum(np.random.randn(30) * 2)
        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "ticker": ticker,
                "open": prices[i] - 0.5,
                "high": prices[i] + 1.0,
                "low": prices[i] - 1.0,
                "close": prices[i],
                "volume": int(np.random.uniform(1e6, 5e6)),
            })
    return pd.DataFrame(records)
