# HMM Stock Trader

An automated stock trading system that uses a Hidden Markov Model to detect market regimes (bull, bear, sideways) and dynamically allocates an ETF portfolio based on the current regime. Trades are executed via Interactive Brokers.

## How It Works

1. **Regime Detection** - A 3-state Gaussian HMM observes daily market features (SPY log returns, realized volatility, SPY-TLT spread) and outputs posterior probabilities for each regime
2. **Dynamic Allocation** - Portfolio weights are blended across regime-specific target allocations based on the HMM's probabilities, producing smooth transitions instead of hard switches
3. **Risk Management** - Position limits, drawdown circuit breaker, turnover caps, and a cash floor are enforced before any trade executes
4. **Trade Execution** - Market-on-Close (MOC) orders are submitted via IBKR before the 3:45 PM ET cutoff

## ETF Universe

| ETF | Role |
|-----|------|
| SPY | US large-cap equity |
| QQQ | US tech/growth |
| IWM | US small-cap |
| TLT | Long-term US treasuries (defensive) |
| GLD | Gold (defensive/inflation hedge) |
| SHY | Short-term treasuries (cash proxy) |

## Quickstart

```bash
# Clone and install
git clone https://github.com/dwagner003/hmm-trader.git
cd hmm-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run a backtest
python -m src.main backtest

# Paper trade (requires IBKR Gateway running)
IBKR_PORT=4002 python -m src.main run --config paper

# Live trade
IBKR_PORT=4001 python -m src.main run --config live
```

## Backtest Results (10yr, 2016-2026)

| Metric | Value |
|--------|-------|
| Total Return | 47.75% |
| Annualized Return | 5.90% |
| Sharpe Ratio | 0.71 |
| Max Drawdown | -19.74% |

## Project Structure

```
config/              YAML configuration (default, paper, live)
src/
  main.py            CLI entry point (run, backtest, reset-circuit-breaker)
  config.py          Config loading and merging
  data/
    fetcher.py       Yahoo Finance data fetching
    features.py      Feature computation (log returns, vol, spread)
    store.py         SQLite storage for prices, trades, predictions
  model/
    hmm.py           HMM training, state labeling, inference
    allocator.py     Regime-blended portfolio weight calculation
  trading/
    executor.py      IBKR MOC order submission
    portfolio.py     Position tracking and order computation
    risk.py          Risk checks (position limits, drawdown, turnover, holidays)
  backtest/
    engine.py        Walk-forward backtesting
    report.py        Performance metrics and chart generation
  alerts/
    email.py         Email notifications for trades, errors, weekly summaries
tests/               69 tests covering all modules
```

## Configuration

All parameters are in `config/default.yaml` - allocation weights, risk limits, HMM settings, and schedule. Override with `config/paper.yaml` or `config/live.yaml`.

Key risk controls:
- **Cash floor**: 10% always held in cash
- **Max position**: No single ETF exceeds 40%
- **Drawdown circuit breaker**: Liquidates all positions if portfolio drops 15% from peak
- **Rebalance threshold**: Only trades when positions deviate >5pp from target
- **Turnover cap**: Max 50% portfolio turnover per day

## Deployment

Designed to run on [Railway](https://railway.app) as a daily cron job (3:30 PM ET weekdays). Includes `Dockerfile` and `railway.toml`. Uses a Railway volume at `/data` for persistent SQLite and model storage.

## Requirements

- Python 3.8+
- IBKR account with API access (paper or live)
- IB Gateway or TWS running locally (or in the same container)
- `ibapi` package (install from [IBKR TWS API](https://www.interactivebrokers.com/en/trading/ib-api.php))

## Disclaimer

This software is for educational and research purposes. Automated trading involves substantial risk of financial loss. Past backtest performance does not guarantee future results. Use at your own risk.
