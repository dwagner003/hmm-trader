# HMM Stock Trader — Design Specification

## Overview

A Python application that uses a Hidden Markov Model to detect market regimes (bull, bear, sideways) and dynamically allocates a portfolio of ETFs based on the current regime. Connects to Interactive Brokers (IBKR) for trade execution. Runs a daily rebalance job after market close.

**Target account size:** Under $10K
**Brokerage:** Interactive Brokers (IBKR)
**Deployment:** Railway (production), local machine (development/testing)
**Trading frequency:** Daily rebalance at ~3:30 PM ET (before MOC order cutoff)

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

The model always uses exactly 3 features (fixed dimensionality required by `hmmlearn`):

- Daily log returns of SPY (primary market signal)
- Realized volatility (rolling 20-day standard deviation of SPY returns)
- Return spread between SPY and TLT (risk appetite indicator)

VIX is intentionally excluded to keep the feature vector simple and avoid data source dependency issues. The rolling volatility feature captures similar information.

### Emission Distribution

Gaussian (multivariate normal). Each hidden state has its own mean vector and covariance matrix. This is the standard approach and is well-supported by the `hmmlearn` library.

### Training

- Uses a rolling 3-year training window of historical daily data
- Initial training uses the first available 3 years from the 10-year backfill
- Retrained monthly (first trading day of each month) on the trailing 3-year window
- Uses the Baum-Welch algorithm (Expectation-Maximization) — handled by `hmmlearn`

### State Label Assignment

Baum-Welch produces anonymous latent states (State 0, 1, 2). After every training or retraining, states must be labeled by sorting on the mean SPY log return of each state's emission distribution:

- **Bull:** state with the highest mean SPY return
- **Bear:** state with the lowest mean SPY return
- **Sideways:** the remaining state

This labeling step runs automatically after every fit and is critical — without it, regime labels can silently flip between training runs, inverting the portfolio allocation.

### Inference

- Each day after market close, feed the latest observation window to the model
- Model outputs posterior probabilities for each state (e.g., 75% bull, 20% sideways, 5% bear)
- Probabilities drive allocation via soft blending — not a hard binary regime switch

## Portfolio Allocation & Rebalancing

### Regime-Specific Target Portfolios

All percentages below are of the **investable balance** (total portfolio minus the 10% cash floor). Each column sums to 100% of the investable portion.

| ETF | Bull | Sideways | Bear |
|-----|------|----------|------|
| SPY | 35% | 15% | 0% |
| QQQ | 30% | 10% | 0% |
| IWM | 15% | 5% | 0% |
| TLT | 5% | 25% | 35% |
| GLD | 5% | 20% | 30% |
| SHY | 10% | 25% | 35% |

SHY holdings do **not** count toward the cash floor — the cash floor is actual uninvested cash in the IBKR account.

These are configurable in a YAML config file.

### Blended Allocation

The system blends based on HMM posterior probabilities rather than picking a single regime. Example with 60% bull / 30% sideways / 10% bear:

```
investable = total_portfolio * 0.90   # after 10% cash floor
SPY target = investable * ((0.60 x 0.35) + (0.30 x 0.15) + (0.10 x 0.00)) = investable * 0.255
```

This produces smoother transitions and avoids whipsawing.

### Rebalancing Logic (daily at ~3:30 PM ET)

1. Check market holiday calendar — skip if market is closed or was an early close day
2. Fetch latest market data (intraday snapshot for current prices)
3. Run HMM inference to get regime probabilities
4. Compute blended target weights (applied to investable balance = total - cash floor)
5. Compare target weights to current portfolio weights
6. Only trade if any position deviates more than 5 **absolute percentage points** from target (e.g., target 35%, current 29% = 6pp deviation, triggers rebalance)
7. Calculate orders needed to reach target weights
8. Submit MOC (Market-on-Close) orders via IBKR — must be submitted before 3:45 PM ET
9. Poll for fill confirmation after 4:00 PM ET
10. Log results and send email alert

### Cash Floor

10% minimum always held in cash (actual cash in the account, not SHY). Target weights are applied to the remaining 90%.

## Data Pipeline

### Data Sources

- **Yahoo Finance (`yfinance`)** — free, reliable for daily EOD data. Used for historical data (backtesting/training) and daily price fetches.
- **IBKR market data** — fallback/supplement for real-time data during live trading.

### Daily Data Flow

1. At ~3:30 PM ET (job start, before MOC order cutoff), fetch latest intraday prices for all ETFs
2. Compute derived features (log returns, rolling volatility, spread)
3. Append to local data store
4. Feed observation window to HMM

### Data Storage

SQLite database (single file, zero configuration). On Railway, the SQLite file and trained model weights are persisted using a Railway volume mount (persistent storage that survives container restarts and redeploys). Stores:

- Daily price history for all ETFs
- Computed features
- HMM regime predictions (with probabilities) per day
- Trade log (what was ordered, what was filled, at what price)
- Portfolio snapshots (daily holdings and values)

### Historical Backfill

On first run, backfills 10 years of daily data from Yahoo Finance. The HMM trains on a rolling 3-year window from this data; the full 10 years is retained for backtesting purposes.

## IBKR Integration

### Library

`ibapi` — IBKR's official Python client, actively maintained by Interactive Brokers. Wrapped with a thin convenience layer in this project for cleaner async usage. (`ib_insync` is no longer actively maintained and is not recommended for new projects.)

### Connection Modes

- **Paper trading:** Connects to IBKR paper trading gateway (port 7497)
- **Live trading:** Connects to IBKR live gateway (port 7496)
- **Backtest:** No IBKR connection — trades simulated internally

### IBKR Gateway

- **Railway (production):** IB Gateway (headless) in the same container
- **Local (development):** Trader Workstation (TWS) with GUI

### Order Execution

- All orders are **MOC (Market-on-Close)** orders — executed at the closing auction price, ensuring fills at the official close. Must be submitted before 3:45 PM ET on NYSE.
- System submits orders by 3:40 PM ET, then polls for fill confirmation after 4:00 PM ET
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
| Drawdown circuit breaker | Portfolio drops 15% from rolling 30-day peak | Liquidate all, go to cash, email alert, require manual restart (see re-entry procedure below) |
| Rebalance threshold | Skip trades if all positions within 5pp of target | No action, log "no rebalance needed" |
| Max daily turnover | Don't trade more than 50% of portfolio in one day | Spread large reallocations over 2+ days |
| Sanity check | Reject orders exceeding account buying power | Cancel all pending orders, email alert |
| Market holiday check | Skip rebalancing on market holidays and early-close days | Log "market closed", no action |

### Circuit Breaker Re-Entry Procedure

When the drawdown circuit breaker fires:
1. All positions are liquidated to cash
2. The system enters a `circuit_breaker_active` state (persisted in SQLite)
3. No trades are executed until manually restarted
4. To restart: set `circuit_breaker_active = false` via a CLI command (`python -m src.main reset-circuit-breaker`)
5. On restart, the peak value tracker resets to the current portfolio value
6. The system re-enters at the current blended allocation based on the latest HMM regime probabilities

### Market Holiday Calendar

The system uses the `pandas_market_calendars` library with the NYSE calendar to determine trading days and early closes. On non-trading days, the daily job exits immediately with a log entry.

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

### Scheduling

`main.py` is a **one-shot script** — it runs once and exits. Scheduling is handled externally:

- **Railway:** Railway cron triggers `main.py` on schedule (configured in `railway.toml`)
- **Local:** Use system cron (`crontab`) or run manually via `python -m src.main run`

There is no in-process scheduler. This avoids duplicate execution and keeps the process lifecycle simple.

### Railway Persistence

Railway containers have ephemeral filesystems. To persist the SQLite database and trained model weights between runs:

- Use a **Railway volume** mounted at `/data`
- Configure `DATA_DIR` environment variable pointing to the volume mount
- The application writes all persistent state (SQLite DB, serialized model files) to `DATA_DIR`
- Model weights are saved using `joblib` (safe serialization for scikit-learn/hmmlearn objects)
- Locally, `DATA_DIR` defaults to `./data/` in the project root
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
  rebalance_threshold_pp: 0.05  # 5 absolute percentage points
  max_daily_turnover: 0.50
  drawdown_circuit_breaker: 0.15

schedule:
  rebalance_time: "15:30"  # Must be before 3:45 PM ET MOC cutoff
  timezone: "US/Eastern"
```

### Secrets (environment variables, never committed)

- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- `ALERT_EMAIL_TO`

## Key Dependencies

- `hmmlearn` — HMM implementation
- `ibapi` — IBKR official API client
- `yfinance` — market data
- `pandas`, `numpy` — data manipulation
- `joblib` — model serialization
- `matplotlib` — backtest visualizations
- `pyyaml` — config loading
- `pandas_market_calendars` — NYSE holiday/early-close calendar
- `pytest` — testing

## Modes of Operation

All three modes use the same core logic with different data sources and executors:

- **Backtest** — run against historical data locally, generate performance reports
- **Paper trade** — connect to IBKR paper trading account, execute daily with simulated money
- **Live trade** — connect to IBKR live account, execute real trades

The mandatory progression is: backtest first, then paper trade, then live trade.
