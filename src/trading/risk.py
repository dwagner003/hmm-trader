from datetime import date
from typing import Dict

import pandas_market_calendars as mcal

_nyse = mcal.get_calendar("NYSE")


def check_position_limits(weights: Dict[str, float], max_position_pct: float) -> Dict[str, float]:
    """Cap any position exceeding max and redistribute excess proportionally."""
    result = dict(weights)
    cash_key = "_cash"

    for _ in range(10):
        excess = 0.0
        capped_keys = set()
        uncapped = {}

        for k, v in result.items():
            if k == cash_key:
                continue
            if v > max_position_pct:
                excess += v - max_position_pct
                result[k] = max_position_pct
                capped_keys.add(k)
            else:
                uncapped[k] = v

        if excess == 0:
            break

        uncapped_total = sum(uncapped.values())
        for k, orig_val in uncapped.items():
            if uncapped_total > 0:
                result[k] = orig_val + excess * (orig_val / uncapped_total)

    return result


def check_rebalance_needed(current: Dict[str, float], target: Dict[str, float], threshold_pp: float) -> bool:
    """Check if any position deviates more than threshold (absolute percentage points)."""
    all_keys = set(current) | set(target)
    for key in all_keys:
        if abs(current.get(key, 0.0) - target.get(key, 0.0)) > threshold_pp:
            return True
    return False


def check_drawdown(current_value: float, peak_value: float, threshold: float) -> bool:
    """Check if drawdown exceeds circuit breaker threshold."""
    if peak_value <= 0:
        return False
    drawdown = (peak_value - current_value) / peak_value
    return drawdown >= threshold


def check_turnover(current: Dict[str, float], target: Dict[str, float], max_turnover: float) -> Dict[str, float]:
    """Scale target weights toward current if total turnover exceeds limit."""
    all_keys = set(current) | set(target)
    total_turnover = sum(abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in all_keys)

    if total_turnover <= max_turnover:
        return dict(target)

    scale = max_turnover / total_turnover
    result = {}
    for k in all_keys:
        cur = current.get(k, 0.0)
        tgt = target.get(k, 0.0)
        result[k] = cur + scale * (tgt - cur)
    return result


def is_trading_day(d: date) -> bool:
    """Check if a given date is a NYSE trading day."""
    schedule = _nyse.valid_days(start_date=d, end_date=d)
    return len(schedule) > 0
