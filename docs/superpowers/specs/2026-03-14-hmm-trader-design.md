# HMM Stock Trader — Design Specification

## Overview

A Python application that uses a Hidden Markov Model to detect market regimes (bull, bear, sideways) and dynamically allocates a portfolio of ETFs based on the current regime. Connects to Interactive Brokers (IBKR) for trade execution. Runs a daily rebalance job after market close.

**Target account size:** Under $10K
**Brokerage:** Interactive Brokers (IBKR)
**Deployment:** Railway (production), local machine (development/testing)
**Trading frequency:** Daily rebalance at ~4:30 PM ET

## ETF Universe

| ETF | Role |
|-----|------|
| SPY | US large-cap equity exposure |
| QQQ | US tech/growth exposure |
| IWM | US small-cap exposure |
| TLT | Long-term US treasuries (defensive) |
| GLD | Gold (defensive/inflation hedge) |
| SHY | Short-term treasuries (cash proxy) |

The universe is intentionally small to keep positions meaningful on a sub-$10K account and minimize transaction costs.

## HMM Model Design

### Hidden States (3 Regimes)

| State | Interpretation | Typical Market Behavior |
|-------|---------------|------------------------|
| Bull | Risk-on | Low volatility, positive returns, strong trends |
| Bear | Risk-off | High volatility, negative returns, drawdowns |
| Sideways | Neutral/uncertain | Mixed signals, choppy price action |

Three states balances interpretability with expressiveness. Can be expanded later if needed.

### Observable Features

The model observes a daily multivariate vector consisting of:

- Daily log returns of SPY (primary market signal)
- Realized volatility (rolling 20-day standard deviation of returns)
- Return spread between SPY and TLT (risk appetite indicator)
- VIX level (if available via data feed, otherwise implied from volatility)

### Emission Distribution

Gaussian (multivariate normal). Each hidden state has its own mean vector and covariance matrix. This is the standard approach and is well-supported by the `hmmlearn` library.

### Training

- Initial training on 5-10 years of historical daily data
- Retrained periodically (monthly rolling window) to adapt to evolving market dynamics
- Uses the Baum-Welch algorithm (Expectation-Maximization) — handled by `hmmlearn`

### Inference

- Each day after market close, feed the latest observation window to the model
- Model outputs posterior probabilities for each state (e.g., 75% bull, 20% sideways, 5% bear)
- Probabilities drive allocation via soft blending — not a hard binary regime switch

## Portfolio Allocation & Rebalancing

### Regime-Specific Target Portfolios

| ETF | Bull | Sideways | Bear |
|-----|------|----------|------|
| SPY | 35% | 15% | 0% |
| QQQ | 30% | 10% | 0% |
| IWM | 15% | 5% | 0% |
| TLT | 5% | 25% | 35% |
| GLD | 5% | 20% | 30% |
| SHY | 10% | 25% | 35% |

These are configurable in a YAML config file.

### Blended Allocation

The system blends based on HMM posterior probabilities rather than picking a single regime. Example with 60% bull / 30% sideways / 10% bear:

```
SPY target = (0.60 x 35%) + (0.30 x 15%) + (0.10 x 0%) = 25.5%
```

This produces smoother transitions and avoids whipsawing.

### Rebalancing Logic (daily at ~4:30 PM ET)

1. Fetch latest market data
2. Run HMM inference to get regime probabilities
3. Compute blended target weights
4. Compare target weights to current portfolio weights
5. Only trade if any position deviates more than 5% from target (rebalance threshold)
6. Calculate orders needed to reach target weights
7. Execute orders via IBKR
8. Log results and send email alert

### Cash Floor

10% minimum always held in cash (actual cash in the account, not SHY). Target weights are applied to the remaining 90%.

## Data Pipeline

### Data Sources

- **Yahoo Finance (`yfinance`)** — free, reliable for daily EOD data. Used for historical data (backtesting/training) and daily price fetches.
- **IBKR market data** — fallback/supplement for real-time data during live trading.

### Daily Data Flow

1. At ~4:15 PM ET (after market close), fetch latest daily OHLCV for all ETFs
2. Compute derived features (log returns, rolling volatility, spread)
3. Append to local data store
4. Feed observation window to HMM

### Data Storage

SQLite database (single file, zero configuration). Stores:

- Daily price history for all ETFs
- Computed features
- HMM regime predictions (with probabilities) per day
- Trade log (what was ordered, what was filled, at what price)
- Portfolio snapshots (daily holdings and values)

### Historical Backfill

On first run, backfills 10 years of daily data from Yahoo Finance to train the initial HMM model.

## IBKR Integration

### Library

`ib_insync` — the most popular Python wrapper for IBKR's API.

### Connection Modes

- **Paper trading:** Connects to IBKR paper trading gateway (port 7497)
- **Live trading:** Connects to IBKR live gateway (port 7496)
- **Backtest:** No IBKR connection — trades simulated internally

### IBKR Gateway

- **Railway (production):** IB Gateway (headless) in the same container
- **Local (development):** Trader Workstation (TWS) with GUI

### Order Execution

- All orders are market orders at close — simple, reliable for liquid ETFs
- Orders submitted, system polls for fill confirmation
- Failed or partial fills are logged and trigger email alert
- No automatic retry for failed orders — human review required

### Account Requirements

- IBKR account (paper trading account is free)
- API access enabled in TWS/Gateway settings
- No minimum balance for paper trading; $0 minimum for IBKR Lite

## Backtesting Engine

### Methodology

Walk-forward backtesting to prevent look-ahead bias:

- Train on a rolling window (e.g., 3 years), predict the next day, step forward, repeat
- Uses the same HMM model and allocation logic as live trading — no separate code path
- Applies realistic constraints: transaction costs ($1 per trade estimate), rebalance threshold, cash floor

### Output Report

| Metric | Description |
|--------|-------------|
| Total return | Cumulative portfolio return over period |
| Annualized return | Averaged per year |
| Sharpe ratio | Risk-adjusted return |
| Max drawdown | Worst peak-to-trough decline |
| Win rate | % of rebalance days with positive return |
| Regime timeline | Chart showing detected regimes over time |
| Equity curve | Portfolio value over time vs buy-and-hold SPY |
| Trade log | Every rebalance decision and why |

### Visualization

Uses `matplotlib` for charts — regime timeline overlaid on SPY price, equity curve comparison, drawdown chart. Reports saved as HTML.

## Risk Management

### Risk Controls

| Control | Rule | Action if Triggered |
|---------|------|-------------------|
| Max position size | No single ETF > 40% of portfolio | Cap and redistribute excess |
| Cash floor | Maintain 10% cash minimum | Reduce largest position to free cash |
| Drawdown circuit breaker | Portfolio drops 15% from peak | Liquidate all, go to cash, email alert, require manual restart |
| Rebalance threshold | Skip trades if all positions within 5% of target | No action, log "no rebalance needed" |
| Max daily turnover | Don't trade more than 50% of portfolio in one day | Spread large reallocations over 2+ days |
| Sanity check | Reject orders exceeding account buying power | Cancel all pending orders, email alert |

## Alerting

### Email Alerts (via SMTP/SendGrid)

| Event | Content |
|-------|---------|
| Daily rebalance executed | Regime probabilities, trades placed, new positions |
| No rebalance needed | Current regime, why no action taken |
| Regime change detected | Old regime to new regime, allocation shift details |
| Circuit breaker triggered | Drawdown details, all positions liquidated |
| System error | What failed, stack trace, whether positions are at risk |
| Weekly summary (Sunday) | Week's performance, current regime, portfolio snapshot |

### Logging

Structured JSON logs for all decisions, trades, and errors. Stored in SQLite and output to stdout (Railway captures stdout).

## Project Structure

```
hmm-trader/
├── config/
│   ├── default.yaml          # Default configuration
│   ├── paper.yaml            # Paper trading overrides
│   └── live.yaml             # Live trading overrides
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point — daily job orchestration
│   ├── data/
│   │   ├── fetcher.py        # Yahoo Finance + IBKR data fetching
│   │   ├── features.py       # Feature computation (returns, vol, spreads)
│   │   └── store.py          # SQLite read/write operations
│   ├── model/
│   │   ├── hmm.py            # HMM training, inference, regime detection
│   │   └── allocator.py      # Regime → blended target weights
│   ├── trading/
│   │   ├── executor.py       # IBKR order execution
│   │   ├── portfolio.py      # Current holdings, position tracking
│   │   └── risk.py           # Risk checks (drawdown, position limits)
│   ├── backtest/
│   │   ├── engine.py         # Walk-forward backtesting loop
│   │   └── report.py         # Performance metrics + visualization
│   └── alerts/
│       └── email.py          # Email notification logic
├── tests/
│   ├── test_hmm.py
│   ├── test_allocator.py
│   ├── test_risk.py
│   └── test_backtest.py
├── .env.example              # Template for secrets
├── requirements.txt
├── Dockerfile                # For Railway deployment
├── railway.toml              # Railway cron schedule config
└── README.md
```

## Configuration

### YAML Config

```yaml
mode: paper  # paper | live | backtest

etf_universe:
  - SPY
  - QQQ
  - IWM
  - TLT
  - GLD
  - SHY

hmm:
  n_states: 3
  training_window_years: 3
  retrain_frequency: monthly

allocation:
  bull:  { SPY: 0.35, QQQ: 0.30, IWM: 0.15, TLT: 0.05, GLD: 0.05, SHY: 0.10 }
  sideways: { SPY: 0.15, QQQ: 0.10, IWM: 0.05, TLT: 0.25, GLD: 0.20, SHY: 0.25 }
  bear: { SPY: 0.00, QQQ: 0.00, IWM: 0.00, TLT: 0.35, GLD: 0.30, SHY: 0.35 }

risk:
  cash_floor: 0.10
  max_position_pct: 0.40
  rebalance_threshold: 0.05
  max_daily_turnover: 0.50
  drawdown_circuit_breaker: 0.15

schedule:
  rebalance_time: "16:30"
  timezone: "US/Eastern"
```

### Secrets (environment variables, never committed)

- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- `ALERT_EMAIL_TO`

## Key Dependencies

- `hmmlearn` — HMM implementation
- `ib_insync` — IBKR API client
- `yfinance` — market data
- `pandas`, `numpy` — data manipulation
- `matplotlib` — backtest visualizations
- `pyyaml` — config loading
- `schedule` — cron-like scheduling (local runs)
- `pytest` — testing

## Modes of Operation

All three modes use the same core logic with different data sources and executors:

- **Backtest** — run against historical data locally, generate performance reports
- **Paper trade** — connect to IBKR paper trading account, execute daily with simulated money
- **Live trade** — connect to IBKR live account, execute real trades

The mandatory progression is: backtest first, then paper trade, then live trade.
