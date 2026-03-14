# src/model/allocator.py
from typing import Dict


def compute_target_weights(
    regime_probs: Dict[str, float],
    config: dict,
) -> Dict[str, float]:
    """Compute blended target weights from regime probabilities.
    Weights are fractions of total portfolio (not just investable).
    Cash floor is reserved, allocation weights applied to remaining investable portion.
    """
    allocation = config["allocation"]
    cash_floor = config["risk"]["cash_floor"]
    investable = 1.0 - cash_floor
    tickers = config["etf_universe"]

    weights = {}
    for ticker in tickers:
        blended = 0.0
        for regime, prob in regime_probs.items():
            blended += prob * allocation[regime].get(ticker, 0.0)
        weights[ticker] = blended * investable

    weights["_cash"] = cash_floor
    return weights
