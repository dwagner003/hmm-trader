import numpy as np
import pandas as pd


def compute_features(prices: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Compute 3-feature observation vector: spy_log_return, realized_vol, spy_tlt_spread.

    Pivots prices by ticker, computes log returns, rolling vol, and spread.
    Drops NaN rows from rolling window warmup.
    """
    pivot = prices.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index()

    spy_close = pivot["SPY"]
    tlt_close = pivot["TLT"]

    spy_log_ret = np.log(spy_close / spy_close.shift(1))
    tlt_log_ret = np.log(tlt_close / tlt_close.shift(1))

    features = pd.DataFrame({
        "date": pivot.index,
        "spy_log_return": spy_log_ret.values,
        "realized_vol": spy_log_ret.rolling(window=vol_window).std().values,
        "spy_tlt_spread": (spy_log_ret - tlt_log_ret).values,
    })

    features = features.dropna().reset_index(drop=True)
    return features
