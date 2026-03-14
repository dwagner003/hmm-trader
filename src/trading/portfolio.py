from typing import Dict, List


def compute_current_weights(holdings, prices, cash):
    # type: (Dict[str, float], Dict[str, float], float) -> Dict[str, float]
    """Compute current portfolio weights from holdings, prices, and cash."""
    total_value = cash
    for ticker, shares in holdings.items():
        if ticker in prices:
            total_value += shares * prices[ticker]

    if total_value <= 0:
        weights = {t: 0.0 for t in holdings}
        weights["_cash"] = 1.0
        return weights

    weights = {}
    for ticker, shares in holdings.items():
        if ticker in prices:
            weights[ticker] = (shares * prices[ticker]) / total_value
        else:
            weights[ticker] = 0.0

    weights["_cash"] = cash / total_value
    return weights


def compute_orders(current_holdings, target_weights, prices, total_value):
    # type: (Dict[str, float], Dict[str, float], Dict[str, float], float) -> List[dict]
    """Compute orders needed to move from current holdings to target weights."""
    orders = []
    for ticker, target_weight in target_weights.items():
        if ticker == "_cash":
            continue
        if ticker not in prices or prices[ticker] <= 0:
            continue
        target_value = target_weight * total_value
        current_shares = current_holdings.get(ticker, 0.0)
        current_value = current_shares * prices[ticker]
        diff_value = target_value - current_value
        shares = abs(diff_value) / prices[ticker]
        if shares < 0.01:
            continue
        orders.append({
            "ticker": ticker,
            "action": "BUY" if diff_value > 0 else "SELL",
            "shares": round(shares, 4),
            "price": prices[ticker],
        })
    return orders
