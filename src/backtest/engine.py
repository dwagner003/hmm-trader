# src/backtest/engine.py
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from src.data.features import compute_features
from src.model.hmm import RegimeHMM
from src.model.allocator import compute_target_weights
from src.trading.risk import check_rebalance_needed, check_position_limits, check_turnover


def run_backtest(config: dict, prices: pd.DataFrame) -> dict:
    """Run walk-forward backtest. Returns dict with metrics, equity curve, trade log, regime history."""
    etf_universe = config["etf_universe"]
    initial_capital = config["backtest"]["initial_capital"]
    transaction_cost = config["backtest"]["transaction_cost"]
    training_years = config["hmm"]["training_window_years"]
    risk_cfg = config["risk"]
    backtest_start = pd.Timestamp(config["backtest"]["start_date"])
    backtest_end = config["backtest"].get("end_date")
    if backtest_end:
        backtest_end = pd.Timestamp(backtest_end)

    features = compute_features(prices)
    features["date"] = pd.to_datetime(features["date"])

    # Build price lookup: date -> ticker -> close
    price_lookup = {}  # type: Dict[pd.Timestamp, Dict[str, float]]
    for _, row in prices.iterrows():
        d = pd.Timestamp(row["date"])
        if d not in price_lookup:
            price_lookup[d] = {}
        price_lookup[d][row["ticker"]] = row["close"]

    feature_dates = sorted(pd.Timestamp(d) for d in features["date"].unique())
    test_dates = [d for d in feature_dates if d >= backtest_start]
    if backtest_end:
        test_dates = [d for d in test_dates if d <= backtest_end]

    training_days = int(training_years * 252)

    cash = initial_capital
    holdings = {t: 0.0 for t in etf_universe}

    equity_curve = []  # type: List[Dict[str, Any]]
    trade_log = []  # type: List[Dict[str, Any]]
    regime_history = []  # type: List[Dict[str, Any]]
    last_train_month = None
    model = RegimeHMM(n_states=config["hmm"]["n_states"])

    for test_date in test_dates:
        date_mask = features["date"] <= test_date
        available = features[date_mask]

        if len(available) < training_days + 1:
            continue

        # Retrain monthly
        current_month = pd.Timestamp(test_date).month
        if last_train_month != current_month:
            train_data = available.iloc[-training_days:]
            trained = False
            base_seed = config["hmm"].get("random_state", 10)
            # Try multiple seeds to handle degenerate covariance failures
            for seed in [base_seed] + list(range(20)):
                try:
                    candidate = RegimeHMM(
                        n_states=config["hmm"]["n_states"],
                        random_state=seed,
                    )
                    candidate.train(train_data)
                    model = candidate
                    trained = True
                    break
                except (ValueError, Exception):
                    continue
            if trained:
                last_train_month = current_month

        if not model.is_trained:
            continue

        window = available.iloc[-training_days:]
        try:
            probs = model.predict(window)
        except (ValueError, Exception):
            continue
        dominant_regime = max(probs, key=probs.get)
        regime_history.append({"date": test_date, "regime": dominant_regime, **probs})

        target_weights = compute_target_weights(probs, config)
        target_weights = check_position_limits(target_weights, risk_cfg["max_position_pct"])

        day_prices = price_lookup.get(test_date, {})
        if not day_prices:
            continue

        portfolio_value = cash
        for ticker in etf_universe:
            if ticker in day_prices:
                portfolio_value += holdings[ticker] * day_prices[ticker]

        current_weights = {"_cash": cash / portfolio_value if portfolio_value > 0 else 1.0}
        for ticker in etf_universe:
            if ticker in day_prices and portfolio_value > 0:
                current_weights[ticker] = (holdings[ticker] * day_prices[ticker]) / portfolio_value
            else:
                current_weights[ticker] = 0.0

        # Record equity using pre-trade portfolio value (mark-to-market at close)
        equity_curve.append({"date": test_date, "value": portfolio_value})

        if not check_rebalance_needed(current_weights, target_weights, risk_cfg["rebalance_threshold_pp"]):
            continue

        target_weights = check_turnover(current_weights, target_weights, risk_cfg["max_daily_turnover"])

        day_trades = []  # type: List[Dict[str, Any]]
        for ticker in etf_universe:
            if ticker not in day_prices:
                continue
            price = day_prices[ticker]
            target_value = target_weights.get(ticker, 0.0) * portfolio_value
            current_value = holdings[ticker] * price
            diff_value = target_value - current_value

            if abs(diff_value) < 1.0:
                continue

            shares_delta = diff_value / price
            holdings[ticker] += shares_delta
            cost = transaction_cost if abs(shares_delta) > 0 else 0
            cash -= diff_value + cost

            action = "BUY" if shares_delta > 0 else "SELL"
            day_trades.append({
                "date": test_date, "ticker": ticker, "action": action,
                "shares": abs(shares_delta), "price": price, "cost": cost,
            })

        trade_log.extend(day_trades)

    # Compute metrics
    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return {
            "total_return": 0.0, "annualized_return": 0.0, "sharpe_ratio": 0.0,
            "max_drawdown": 0.0, "win_rate": 0.0, "equity_curve": pd.Series(dtype=float),
            "trade_log": [], "regime_history": pd.DataFrame(),
        }

    equity_series = eq_df.set_index("date")["value"]
    total_return = (equity_series.iloc[-1] / initial_capital) - 1.0
    n_days = len(equity_series)
    annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1.0

    daily_returns = equity_series.pct_change().dropna()
    sharpe_ratio = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    rolling_max = equity_series.cummax()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    rebalance_dates = {t["date"] for t in trade_log}
    rebalance_returns = daily_returns[daily_returns.index.isin(rebalance_dates)]
    win_rate = 0.0
    if len(rebalance_returns) > 0:
        win_rate = float((rebalance_returns > 0).mean())

    return {
        "total_return": total_return, "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown, "win_rate": win_rate,
        "equity_curve": equity_series, "trade_log": trade_log,
        "regime_history": pd.DataFrame(regime_history),
    }
