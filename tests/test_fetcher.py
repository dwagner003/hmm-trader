import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.fetcher import fetch_prices


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance download return with MultiIndex columns."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    data = {
        ("Close", "SPY"): [450.0, 451.0, 449.0, 452.0, 453.0],
        ("Open", "SPY"): [449.0, 450.0, 451.0, 449.0, 452.0],
        ("High", "SPY"): [452.0, 453.0, 452.0, 454.0, 455.0],
        ("Low", "SPY"): [448.0, 449.0, 447.0, 448.0, 451.0],
        ("Volume", "SPY"): [1000000, 1100000, 900000, 1200000, 1050000],
    }
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@patch("src.data.fetcher.yf.download")
def test_fetch_prices_returns_normalized_df(mock_download, mock_yfinance_data):
    mock_download.return_value = mock_yfinance_data
    result = fetch_prices(["SPY"], period="5d")
    assert "date" in result.columns
    assert "ticker" in result.columns
    assert "close" in result.columns
    assert len(result) == 5
    assert result.iloc[0]["ticker"] == "SPY"


@patch("src.data.fetcher.yf.download")
def test_fetch_prices_multiple_tickers(mock_download):
    dates = pd.bdate_range("2024-01-02", periods=3)
    data = {}
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        for ticker in ["SPY", "TLT"]:
            val = 450.0 if ticker == "SPY" else 100.0
            data[(col, ticker)] = [val] * 3
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    mock_download.return_value = df
    result = fetch_prices(["SPY", "TLT"], period="5d")
    tickers = result["ticker"].unique()
    assert "SPY" in tickers
    assert "TLT" in tickers
