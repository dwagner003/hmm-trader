import yfinance as yf
import pandas as pd
from typing import List, Optional


def fetch_prices(
    tickers: List[str],
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance. Returns DataFrame with columns: date, ticker, open, high, low, close, volume."""
    kwargs = {"tickers": tickers, "group_by": "ticker", "auto_adjust": True}
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    elif period:
        kwargs["period"] = period
    else:
        kwargs["period"] = "1d"

    raw = yf.download(**kwargs)
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    # Normalize columns: yfinance may return MultiIndex or flat columns.
    # Newer yfinance (>=0.2.50) returns MultiIndex with names ['Ticker', 'Price']
    # (ticker at level 0). Older versions used ('Price', 'Ticker') ordering with
    # ticker at level 1. Flat columns can appear for single-ticker in very old versions.
    if isinstance(raw.columns, pd.MultiIndex):
        # Detect which level holds the ticker names
        col_names = raw.columns.names  # e.g. ['Ticker', 'Price'] or [None, None]
        if col_names[0] == "Ticker":
            ticker_level = 0
        else:
            # Old format: Price at level 0, Ticker at level 1
            ticker_level = 1
    else:
        # Flat columns for single ticker: convert to MultiIndex (old-style, ticker at level 1)
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, tickers[0]) for col in raw.columns]
        )
        ticker_level = 1

    records = []
    for ticker in tickers:
        try:
            ticker_data = raw.xs(ticker, level=ticker_level, axis=1)
        except KeyError:
            continue
        for date_val, row in ticker_data.iterrows():
            if pd.isna(row.get("Close")):
                continue
            records.append({
                "date": pd.Timestamp(date_val),
                "ticker": ticker,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })

    return pd.DataFrame(records)


def backfill_history(tickers: List[str], years: int = 10) -> pd.DataFrame:
    """Download full historical data."""
    return fetch_prices(tickers, period=f"{years}y")
