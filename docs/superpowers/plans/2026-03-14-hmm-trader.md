# HMM Stock Trader Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python application that uses a Hidden Markov Model to detect market regimes and dynamically allocate an ETF portfolio via Interactive Brokers.

**Architecture:** Monolithic Python app with well-separated modules: config, data pipeline, HMM model, portfolio allocator, risk manager, IBKR executor, backtester, and email alerts. One-shot script triggered by external cron (Railway or system crontab). All persistent state in SQLite + joblib model files in a configurable `DATA_DIR`.

**Tech Stack:** Python 3.11+, hmmlearn, ibapi, yfinance, pandas, numpy, SQLite, matplotlib, joblib, pandas_market_calendars, pyyaml, pytest

**Spec:** `docs/superpowers/specs/2026-03-14-hmm-trader-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `config/default.yaml` | Default configuration (ETF universe, HMM params, allocation weights, risk limits, schedule) |
| `config/paper.yaml` | Paper trading overrides (mode, IBKR port) |
| `config/live.yaml` | Live trading overrides (mode, IBKR port) |
| `src/__init__.py` | Package marker |
| `src/main.py` | CLI entry point — `run` (daily job), `backtest`, `reset-circuit-breaker` commands |
| `src/config.py` | Load and merge YAML configs, env vars, provide typed access |
| `src/data/__init__.py` | Package marker |
| `src/data/fetcher.py` | Download price data from Yahoo Finance; backfill 10yr history on first run |
| `src/data/features.py` | Compute 3-feature observation vector (log returns, rolling vol, SPY-TLT spread) |
| `src/data/store.py` | SQLite schema, read/write for prices, features, predictions, trades, snapshots |
| `src/model/__init__.py` | Package marker |
| `src/model/hmm.py` | HMM wrapper: train on rolling window, label states, infer regime probabilities, save/load model |
| `src/model/allocator.py` | Blend regime-specific target weights using posterior probabilities, apply cash floor |
| `src/trading/__init__.py` | Package marker |
| `src/trading/executor.py` | IBKR connection, submit MOC orders, poll for fills |
| `src/trading/portfolio.py` | Query current holdings from IBKR, compute current weights |
| `src/trading/risk.py` | Pre-trade risk checks: position limits, cash floor, drawdown circuit breaker, turnover cap, holiday check |
| `src/backtest/__init__.py` | Package marker |
| `src/backtest/engine.py` | Walk-forward backtesting loop using same model/allocator/risk code |
| `src/backtest/report.py` | Compute metrics (return, Sharpe, drawdown, win rate) and generate matplotlib charts |
| `src/alerts/__init__.py` | Package marker |
| `src/alerts/email.py` | Send email alerts via SMTP for rebalance, regime change, errors, weekly summary |
| `tests/conftest.py` | Shared fixtures (sample price data, config, tmp data dir) |
| `tests/test_config.py` | Config loading tests |
| `tests/test_fetcher.py` | Data fetcher tests (mocked yfinance) |
| `tests/test_features.py` | Feature computation tests |
| `tests/test_store.py` | SQLite store tests |
| `tests/test_hmm.py` | HMM training, labeling, inference tests |
| `tests/test_allocator.py` | Allocation blending tests |
| `tests/test_risk.py` | Risk check tests |
| `tests/test_backtest.py` | Backtesting engine tests |
| `tests/test_email.py` | Email alert tests (mocked SMTP) |
| `.env.example` | Template for environment variables |
| `.gitignore` | Ignore data/, .env, __pycache__, etc. |
| `requirements.txt` | Pinned dependencies |
| `Dockerfile` | Container for Railway deployment |
| `railway.toml` | Railway cron config |

---

## Chunk 1: Project Scaffolding & Configuration

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/trading/__init__.py`
- Create: `src/backtest/__init__.py`
- Create: `src/alerts/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
hmmlearn==0.3.2
ibapi>=10.19.2
yfinance>=0.2.36
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
pyyaml>=6.0.1
pandas_market_calendars>=4.4.0
joblib>=1.3.0
pytest>=8.0.0
pytest-cov>=4.1.0
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
.env
data/
*.db
*.sqlite
*.joblib
.pytest_cache/
htmlcov/
*.egg-info/
dist/
build/
.venv/
venv/
```

- [ ] **Step 3: Create .env.example**

```
# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Email Alerts (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_TO=your-email@gmail.com

# Data directory (default: ./data)
DATA_DIR=./data
```

- [ ] **Step 4: Create all __init__.py files**

Create empty `__init__.py` in: `src/`, `src/data/`, `src/model/`, `src/trading/`, `src/backtest/`, `src/alerts/`

- [ ] **Step 5: Install dependencies and verify**

```bash
cd /home/dwagner003/dev/hmm-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import hmmlearn; import yfinance; import pandas; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore .env.example src/
git commit -m "feat: project scaffolding with dependencies and package structure"
```

---

### Task 2: Configuration System

**Files:**
- Create: `config/default.yaml`
- Create: `config/paper.yaml`
- Create: `config/live.yaml`
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create config/default.yaml**

```yaml
mode: backtest

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
  rebalance_threshold_pp: 0.05
  max_daily_turnover: 0.50
  drawdown_circuit_breaker: 0.15

schedule:
  rebalance_time: "15:30"
  timezone: "US/Eastern"

backtest:
  start_date: "2016-01-01"
  end_date: null
  initial_capital: 10000
  transaction_cost: 1.0
```

- [ ] **Step 2: Create config/paper.yaml**

```yaml
mode: paper
```

- [ ] **Step 3: Create config/live.yaml**

```yaml
mode: live
```

- [ ] **Step 4: Write the failing test for config loading**

```python
# tests/test_config.py
import os
import pytest
from src.config import load_config


def test_load_default_config(tmp_path):
    """Default config loads with all expected keys."""
    cfg = load_config(config_dir="config")
    assert cfg["mode"] == "backtest"
    assert cfg["hmm"]["n_states"] == 3
    assert len(cfg["etf_universe"]) == 6
    assert "SPY" in cfg["etf_universe"]


def test_load_config_with_override(tmp_path):
    """Paper override changes mode but keeps defaults."""
    cfg = load_config(config_dir="config", override="paper")
    assert cfg["mode"] == "paper"
    assert cfg["hmm"]["n_states"] == 3


def test_allocation_weights_sum_to_one():
    """Each regime's allocation weights must sum to 1.0."""
    cfg = load_config(config_dir="config")
    for regime in ["bull", "sideways", "bear"]:
        total = sum(cfg["allocation"][regime].values())
        assert abs(total - 1.0) < 1e-9, f"{regime} weights sum to {total}"


def test_data_dir_from_env(monkeypatch):
    """DATA_DIR env var overrides default."""
    monkeypatch.setenv("DATA_DIR", "/tmp/test-data")
    cfg = load_config(config_dir="config")
    assert cfg["data_dir"] == "/tmp/test-data"


def test_data_dir_default(monkeypatch):
    """Default data_dir is ./data."""
    monkeypatch.delenv("DATA_DIR", raising=False)
    cfg = load_config(config_dir="config")
    assert cfg["data_dir"] == "./data"
```

- [ ] **Step 5: Run test to verify it fails**

```bash
cd /home/dwagner003/dev/hmm-trader
source .venv/bin/activate
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 6: Implement src/config.py**

```python
# src/config.py
import os
from pathlib import Path
from copy import deepcopy

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values win."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_dir: str = "config", override: str | None = None) -> dict:
    """Load default config, optionally merge an override file, and apply env vars.

    Args:
        config_dir: Path to directory containing YAML config files.
        override: Name of override file (without .yaml extension), e.g. "paper" or "live".

    Returns:
        Merged configuration dictionary.
    """
    config_path = Path(config_dir)

    with open(config_path / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    if override:
        override_path = config_path / f"{override}.yaml"
        with open(override_path) as f:
            override_cfg = yaml.safe_load(f)
        if override_cfg:
            cfg = _deep_merge(cfg, override_cfg)

    # Apply environment variable overrides
    cfg["data_dir"] = os.environ.get("DATA_DIR", "./data")

    return cfg
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 8: Commit**

```bash
git add config/ src/config.py tests/test_config.py
git commit -m "feat: configuration system with YAML loading and env var overrides"
```

---

## Chunk 2: Data Pipeline (Fetcher, Features, Store)

### Task 3: SQLite Data Store

**Files:**
- Create: `src/data/store.py`
- Create: `tests/test_store.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create shared test fixtures in conftest.py**

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    return str(tmp_path)


@pytest.fixture
def sample_prices():
    """Sample OHLCV DataFrame for 30 trading days, 2 tickers."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    np.random.seed(42)
    records = []
    for ticker in ["SPY", "TLT"]:
        base = 450.0 if ticker == "SPY" else 100.0
        prices = base + np.cumsum(np.random.randn(30) * 2)
        for i, date in enumerate(dates):
            records.append({
                "date": date,
                "ticker": ticker,
                "open": prices[i] - 0.5,
                "high": prices[i] + 1.0,
                "low": prices[i] - 1.0,
                "close": prices[i],
                "volume": int(np.random.uniform(1e6, 5e6)),
            })
    return pd.DataFrame(records)
```

- [ ] **Step 2: Write failing tests for store**

```python
# tests/test_store.py
import pytest
import pandas as pd
from src.data.store import DataStore


def test_init_creates_tables(tmp_data_dir):
    """Store init creates all required tables."""
    store = DataStore(tmp_data_dir)
    tables = store.list_tables()
    assert "prices" in tables
    assert "features" in tables
    assert "predictions" in tables
    assert "trades" in tables
    assert "snapshots" in tables
    assert "system_state" in tables


def test_save_and_load_prices(tmp_data_dir, sample_prices):
    """Round-trip save and load of price data."""
    store = DataStore(tmp_data_dir)
    store.save_prices(sample_prices)
    loaded = store.load_prices(ticker="SPY")
    assert len(loaded) == 30
    assert loaded.iloc[0]["ticker"] == "SPY"


def test_save_prices_no_duplicates(tmp_data_dir, sample_prices):
    """Saving same data twice doesn't create duplicates."""
    store = DataStore(tmp_data_dir)
    store.save_prices(sample_prices)
    store.save_prices(sample_prices)
    loaded = store.load_prices(ticker="SPY")
    assert len(loaded) == 30


def test_load_prices_date_range(tmp_data_dir, sample_prices):
    """Load prices filtered by date range."""
    store = DataStore(tmp_data_dir)
    store.save_prices(sample_prices)
    loaded = store.load_prices(
        ticker="SPY",
        start_date="2024-01-10",
        end_date="2024-01-20",
    )
    assert len(loaded) > 0
    assert all(loaded["date"] >= "2024-01-10")
    assert all(loaded["date"] <= "2024-01-20")


def test_save_and_load_prediction(tmp_data_dir):
    """Round-trip save and load of regime prediction."""
    store = DataStore(tmp_data_dir)
    store.save_prediction(
        date="2024-01-15",
        bull_prob=0.7,
        sideways_prob=0.2,
        bear_prob=0.1,
        regime="bull",
    )
    preds = store.load_predictions()
    assert len(preds) == 1
    assert preds.iloc[0]["regime"] == "bull"
    assert abs(preds.iloc[0]["bull_prob"] - 0.7) < 1e-9


def test_circuit_breaker_state(tmp_data_dir):
    """Circuit breaker state persists correctly."""
    store = DataStore(tmp_data_dir)
    assert store.get_circuit_breaker_active() is False
    store.set_circuit_breaker_active(True)
    assert store.get_circuit_breaker_active() is True
    store.set_circuit_breaker_active(False)
    assert store.get_circuit_breaker_active() is False


def test_save_and_load_trade(tmp_data_dir):
    """Round-trip save and load of trade record."""
    store = DataStore(tmp_data_dir)
    store.save_trade(
        date="2024-01-15",
        ticker="SPY",
        action="BUY",
        shares=10,
        price=450.0,
        cost=1.0,
    )
    trades = store.load_trades()
    assert len(trades) == 1
    assert trades.iloc[0]["ticker"] == "SPY"


def test_save_and_load_snapshot(tmp_data_dir):
    """Round-trip save and load of portfolio snapshot."""
    store = DataStore(tmp_data_dir)
    store.save_snapshot(
        date="2024-01-15",
        total_value=10000.0,
        cash=1000.0,
        holdings={"SPY": 10, "TLT": 20},
    )
    snaps = store.load_snapshots()
    assert len(snaps) == 1
    assert snaps.iloc[0]["total_value"] == 10000.0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_store.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.store'`

- [ ] **Step 4: Implement src/data/store.py**

```python
# src/data/store.py
import json
import sqlite3
from pathlib import Path

import pandas as pd


class DataStore:
    """SQLite-backed storage for prices, features, predictions, trades, and snapshots."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "hmm_trader.db"
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_tables(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prices (
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (date, ticker)
                );

                CREATE TABLE IF NOT EXISTS features (
                    date TEXT NOT NULL PRIMARY KEY,
                    spy_log_return REAL,
                    realized_vol REAL,
                    spy_tlt_spread REAL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    date TEXT NOT NULL PRIMARY KEY,
                    bull_prob REAL,
                    sideways_prob REAL,
                    bear_prob REAL,
                    regime TEXT
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    shares REAL,
                    price REAL,
                    cost REAL
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    date TEXT NOT NULL PRIMARY KEY,
                    total_value REAL,
                    cash REAL,
                    holdings_json TEXT
                );

                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT NOT NULL PRIMARY KEY,
                    value TEXT
                );
            """)

    def list_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        return [r[0] for r in rows]

    # --- Prices ---

    def save_prices(self, df: pd.DataFrame):
        with self._connect() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO prices
                       (date, ticker, open, high, low, close, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(row["date"])[:10],
                        row["ticker"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        int(row["volume"]),
                    ),
                )

    def load_prices(
        self,
        ticker: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM prices WHERE 1=1"
        params: list = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # --- Features ---

    def save_features(self, df: pd.DataFrame):
        with self._connect() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO features
                       (date, spy_log_return, realized_vol, spy_tlt_spread)
                       VALUES (?, ?, ?, ?)""",
                    (
                        str(row["date"])[:10],
                        row["spy_log_return"],
                        row["realized_vol"],
                        row["spy_tlt_spread"],
                    ),
                )

    def load_features(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM features WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # --- Predictions ---

    def save_prediction(
        self,
        date: str,
        bull_prob: float,
        sideways_prob: float,
        bear_prob: float,
        regime: str,
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO predictions
                   (date, bull_prob, sideways_prob, bear_prob, regime)
                   VALUES (?, ?, ?, ?, ?)""",
                (date, bull_prob, sideways_prob, bear_prob, regime),
            )

    def load_predictions(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY date", conn
            )

    # --- Trades ---

    def save_trade(
        self,
        date: str,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        cost: float,
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trades (date, ticker, action, shares, price, cost)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (date, ticker, action, shares, price, cost),
            )

    def load_trades(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM trades WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    # --- Snapshots ---

    def save_snapshot(
        self,
        date: str,
        total_value: float,
        cash: float,
        holdings: dict[str, float],
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO snapshots
                   (date, total_value, cash, holdings_json)
                   VALUES (?, ?, ?, ?)""",
                (date, total_value, cash, json.dumps(holdings)),
            )

    def load_snapshots(self) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM snapshots ORDER BY date", conn
            )
        if not df.empty:
            df["holdings"] = df["holdings_json"].apply(json.loads)
        return df

    # --- System State ---

    def get_circuit_breaker_active(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = 'circuit_breaker_active'"
            ).fetchone()
        if row is None:
            return False
        return row[0] == "true"

    def set_circuit_breaker_active(self, active: bool):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO system_state (key, value)
                   VALUES ('circuit_breaker_active', ?)""",
                ("true" if active else "false",),
            )

    def get_system_state(self, key: str) -> str | None:
        """Get an arbitrary system state value by key."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_system_state(self, key: str, value: str):
        """Set an arbitrary system state value by key."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO system_state (key, value)
                   VALUES (?, ?)""",
                (key, value),
            )

    def get_peak_value(self) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = 'peak_value'"
            ).fetchone()
        return float(row[0]) if row else None

    def set_peak_value(self, value: float):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO system_state (key, value)
                   VALUES ('peak_value', ?)""",
                (str(value),),
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_store.py -v
```

Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/store.py tests/conftest.py tests/test_store.py
git commit -m "feat: SQLite data store with prices, features, predictions, trades, snapshots"
```

---

### Task 4: Data Fetcher

**Files:**
- Create: `src/data/fetcher.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: Write failing tests for fetcher**

```python
# tests/test_fetcher.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.fetcher import fetch_prices, backfill_history


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance download return value."""
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
    """fetch_prices returns a DataFrame with expected columns."""
    mock_download.return_value = mock_yfinance_data
    result = fetch_prices(["SPY"], period="5d")
    assert "date" in result.columns
    assert "ticker" in result.columns
    assert "close" in result.columns
    assert len(result) == 5
    assert result.iloc[0]["ticker"] == "SPY"


@patch("src.data.fetcher.yf.download")
def test_fetch_prices_multiple_tickers(mock_download):
    """fetch_prices handles multiple tickers."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fetcher.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement src/data/fetcher.py**

```python
# src/data/fetcher.py
import yfinance as yf
import pandas as pd


def fetch_prices(
    tickers: list[str],
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV price data from Yahoo Finance.

    Args:
        tickers: List of ticker symbols.
        period: yfinance period string (e.g. "5d", "10y"). Used if start/end not provided.
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume.
    """
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

    # Normalize columns: yfinance may return MultiIndex or flat columns
    if isinstance(raw.columns, pd.MultiIndex):
        # MultiIndex: (metric, ticker) — standard for multi-ticker downloads
        pass
    else:
        # Flat columns for single ticker: convert to MultiIndex
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, tickers[0]) for col in raw.columns]
        )

    records = []
    for ticker in tickers:
        try:
            ticker_data = raw.xs(ticker, level=1, axis=1)
        except KeyError:
            continue

        for date, row in ticker_data.iterrows():
            if pd.isna(row.get("Close")):
                continue
            records.append({
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })

    return pd.DataFrame(records)


def backfill_history(tickers: list[str], years: int = 10) -> pd.DataFrame:
    """Download full historical data for initial setup.

    Args:
        tickers: List of ticker symbols.
        years: Number of years of history to fetch.

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume.
    """
    return fetch_prices(tickers, period=f"{years}y")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_fetcher.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/fetcher.py tests/test_fetcher.py
git commit -m "feat: Yahoo Finance data fetcher with backfill support"
```

---

### Task 5: Feature Computation

**Files:**
- Create: `src/data/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing tests for feature computation**

```python
# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from src.data.features import compute_features


@pytest.fixture
def price_data_for_features():
    """50 days of SPY and TLT price data for feature computation."""
    dates = pd.bdate_range("2024-01-02", periods=50)
    np.random.seed(42)
    spy_prices = 450.0 + np.cumsum(np.random.randn(50) * 2)
    tlt_prices = 100.0 + np.cumsum(np.random.randn(50) * 0.5)
    records = []
    for i, date in enumerate(dates):
        for ticker, prices in [("SPY", spy_prices), ("TLT", tlt_prices)]:
            records.append({
                "date": date,
                "ticker": ticker,
                "open": prices[i] - 0.5,
                "high": prices[i] + 1.0,
                "low": prices[i] - 1.0,
                "close": prices[i],
                "volume": 1000000,
            })
    return pd.DataFrame(records)


def test_compute_features_returns_expected_columns(price_data_for_features):
    """Feature DataFrame has the 3 required feature columns plus date."""
    features = compute_features(price_data_for_features)
    assert "date" in features.columns
    assert "spy_log_return" in features.columns
    assert "realized_vol" in features.columns
    assert "spy_tlt_spread" in features.columns


def test_compute_features_drops_nan_rows(price_data_for_features):
    """First 20 rows are NaN due to rolling window; result has no NaNs."""
    features = compute_features(price_data_for_features)
    assert not features["realized_vol"].isna().any()
    assert not features["spy_log_return"].isna().any()
    assert len(features) == 30  # 50 - 20 rolling window


def test_compute_features_log_return_range(price_data_for_features):
    """Log returns should be small daily values, roughly -0.1 to 0.1."""
    features = compute_features(price_data_for_features)
    assert features["spy_log_return"].abs().max() < 0.2


def test_compute_features_realized_vol_positive(price_data_for_features):
    """Realized volatility must always be positive."""
    features = compute_features(price_data_for_features)
    assert (features["realized_vol"] > 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement src/data/features.py**

```python
# src/data/features.py
import numpy as np
import pandas as pd


def compute_features(prices: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Compute the 3-feature observation vector from price data.

    Features:
        1. spy_log_return: Daily log return of SPY
        2. realized_vol: Rolling 20-day standard deviation of SPY log returns
        3. spy_tlt_spread: Daily log return of SPY minus daily log return of TLT

    Args:
        prices: DataFrame with columns [date, ticker, close] (at minimum).
        vol_window: Rolling window size for volatility (default 20).

    Returns:
        DataFrame with columns [date, spy_log_return, realized_vol, spy_tlt_spread].
        Rows with NaN (from rolling window warmup) are dropped.
    """
    # Pivot to get close prices by ticker
    pivot = prices.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index()

    spy_close = pivot["SPY"]
    tlt_close = pivot["TLT"]

    # Compute log returns
    spy_log_ret = np.log(spy_close / spy_close.shift(1))
    tlt_log_ret = np.log(tlt_close / tlt_close.shift(1))

    features = pd.DataFrame({
        "date": pivot.index,
        "spy_log_return": spy_log_ret.values,
        "realized_vol": spy_log_ret.rolling(window=vol_window).std().values,
        "spy_tlt_spread": (spy_log_ret - tlt_log_ret).values,
    })

    # Drop NaN rows from rolling window warmup and first log return
    features = features.dropna().reset_index(drop=True)

    return features
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/features.py tests/test_features.py
git commit -m "feat: feature computation (log returns, rolling vol, SPY-TLT spread)"
```

---

## Chunk 3: HMM Model & Allocator

### Task 6: HMM Model Wrapper

**Files:**
- Create: `src/model/hmm.py`
- Create: `tests/test_hmm.py`

- [ ] **Step 1: Write failing tests for HMM**

```python
# tests/test_hmm.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.model.hmm import RegimeHMM


@pytest.fixture
def synthetic_features():
    """Synthetic 3-feature data with obvious regime structure.

    First 200 days: bull-like (positive returns, low vol)
    Next 200 days: bear-like (negative returns, high vol)
    Next 200 days: sideways (near-zero returns, medium vol)
    """
    np.random.seed(42)
    n = 200

    bull = np.column_stack([
        np.random.normal(0.001, 0.005, n),   # positive returns
        np.random.normal(0.008, 0.001, n),    # low vol
        np.random.normal(0.0005, 0.003, n),   # positive spread
    ])
    bear = np.column_stack([
        np.random.normal(-0.001, 0.015, n),   # negative returns
        np.random.normal(0.020, 0.003, n),     # high vol
        np.random.normal(-0.001, 0.005, n),    # negative spread
    ])
    sideways = np.column_stack([
        np.random.normal(0.0, 0.008, n),      # near-zero returns
        np.random.normal(0.012, 0.002, n),     # medium vol
        np.random.normal(0.0, 0.004, n),       # neutral spread
    ])

    data = np.vstack([bull, bear, sideways])
    dates = pd.bdate_range("2022-01-03", periods=len(data))

    return pd.DataFrame({
        "date": dates[:len(data)],
        "spy_log_return": data[:, 0],
        "realized_vol": data[:, 1],
        "spy_tlt_spread": data[:, 2],
    })


def test_hmm_train_produces_3_states(synthetic_features):
    """Training produces a model with 3 states."""
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    assert model.is_trained


def test_hmm_state_labels_assigned(synthetic_features):
    """After training, states are labeled bull/bear/sideways."""
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    assert model.state_labels is not None
    assert set(model.state_labels.values()) == {"bull", "bear", "sideways"}


def test_hmm_bull_state_has_highest_mean_return(synthetic_features):
    """Bull state has the highest mean SPY return in emission distribution."""
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    bull_idx = [k for k, v in model.state_labels.items() if v == "bull"][0]
    bear_idx = [k for k, v in model.state_labels.items() if v == "bear"][0]
    bull_mean = model.model.means_[bull_idx, 0]
    bear_mean = model.model.means_[bear_idx, 0]
    assert bull_mean > bear_mean


def test_hmm_predict_returns_probabilities(synthetic_features):
    """Predict returns regime probabilities that sum to 1."""
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    probs = model.predict(synthetic_features)
    assert "bull" in probs
    assert "bear" in probs
    assert "sideways" in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_hmm_predict_bull_data_returns_high_bull_prob(synthetic_features):
    """Predicting on bull-like data should return high bull probability."""
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    bull_data = synthetic_features.iloc[:200]
    probs = model.predict(bull_data)
    assert probs["bull"] > 0.5


def test_hmm_save_and_load(synthetic_features, tmp_data_dir):
    """Model can be saved and loaded with identical predictions."""
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    probs_before = model.predict(synthetic_features)

    model.save(tmp_data_dir)

    loaded = RegimeHMM(n_states=3)
    loaded.load(tmp_data_dir)
    probs_after = loaded.predict(synthetic_features)

    for key in probs_before:
        assert abs(probs_before[key] - probs_after[key]) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hmm.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement src/model/hmm.py**

```python
# src/model/hmm.py
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeHMM:
    """Hidden Markov Model wrapper for market regime detection.

    Wraps hmmlearn's GaussianHMM with automatic state labeling
    after training. States are labeled by sorting on mean SPY
    log return: highest = bull, lowest = bear, middle = sideways.
    """

    FEATURE_COLS = ["spy_log_return", "realized_vol", "spy_tlt_spread"]

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model: GaussianHMM | None = None
        self.state_labels: dict[int, str] | None = None
        self.is_trained = False

    def train(self, features: pd.DataFrame, n_iter: int = 100):
        """Train the HMM on feature data and assign state labels.

        Args:
            features: DataFrame with columns matching FEATURE_COLS.
            n_iter: Max EM iterations.
        """
        X = features[self.FEATURE_COLS].values

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self._assign_labels()
        self.is_trained = True

    def _assign_labels(self):
        """Label states by sorting on mean SPY log return (feature index 0).

        Bull = highest mean return, Bear = lowest, Sideways = middle.
        """
        mean_returns = self.model.means_[:, 0]
        sorted_indices = np.argsort(mean_returns)

        self.state_labels = {
            int(sorted_indices[0]): "bear",
            int(sorted_indices[1]): "sideways",
            int(sorted_indices[2]): "bull",
        }

    def predict(self, features: pd.DataFrame) -> dict[str, float]:
        """Predict current regime probabilities from feature data.

        Uses the last observation's posterior state probabilities
        after filtering through the full observation sequence.

        Args:
            features: DataFrame with columns matching FEATURE_COLS.

        Returns:
            Dict mapping regime name to probability, e.g.
            {"bull": 0.7, "sideways": 0.2, "bear": 0.1}
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X = features[self.FEATURE_COLS].values
        posteriors = self.model.predict_proba(X)
        last_probs = posteriors[-1]

        return {
            self.state_labels[i]: float(last_probs[i])
            for i in range(self.n_states)
        }

    def save(self, data_dir: str):
        """Save trained model and labels to disk."""
        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "state_labels": self.state_labels},
            path / "hmm_model.joblib",
        )

    def load(self, data_dir: str):
        """Load trained model and labels from disk."""
        path = Path(data_dir) / "hmm_model.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.state_labels = data["state_labels"]
        self.is_trained = True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hmm.py -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/hmm.py tests/test_hmm.py
git commit -m "feat: HMM regime detector with auto state labeling and save/load"
```

---

### Task 7: Portfolio Allocator

**Files:**
- Create: `src/model/allocator.py`
- Create: `tests/test_allocator.py`

- [ ] **Step 1: Write failing tests for allocator**

```python
# tests/test_allocator.py
import pytest
from src.model.allocator import compute_target_weights


@pytest.fixture
def config():
    return {
        "etf_universe": ["SPY", "QQQ", "IWM", "TLT", "GLD", "SHY"],
        "allocation": {
            "bull":     {"SPY": 0.35, "QQQ": 0.30, "IWM": 0.15, "TLT": 0.05, "GLD": 0.05, "SHY": 0.10},
            "sideways": {"SPY": 0.15, "QQQ": 0.10, "IWM": 0.05, "TLT": 0.25, "GLD": 0.20, "SHY": 0.25},
            "bear":     {"SPY": 0.00, "QQQ": 0.00, "IWM": 0.00, "TLT": 0.35, "GLD": 0.30, "SHY": 0.35},
        },
        "risk": {"cash_floor": 0.10},
    }


def test_pure_bull_allocation(config):
    """100% bull probability produces bull weights scaled by investable balance."""
    probs = {"bull": 1.0, "sideways": 0.0, "bear": 0.0}
    weights = compute_target_weights(probs, config)
    cash_floor = config["risk"]["cash_floor"]
    investable = 1.0 - cash_floor  # 0.90
    assert abs(weights["SPY"] - 0.35 * investable) < 1e-9
    assert abs(weights["QQQ"] - 0.30 * investable) < 1e-9
    assert abs(weights["_cash"] - cash_floor) < 1e-9


def test_pure_bear_allocation(config):
    """100% bear probability produces bear weights, no equity exposure."""
    probs = {"bull": 0.0, "sideways": 0.0, "bear": 1.0}
    weights = compute_target_weights(probs, config)
    assert weights["SPY"] == 0.0
    assert weights["QQQ"] == 0.0
    assert weights["IWM"] == 0.0
    assert weights["TLT"] > 0.0


def test_blended_allocation(config):
    """Blended probabilities produce weighted combination."""
    probs = {"bull": 0.6, "sideways": 0.3, "bear": 0.1}
    weights = compute_target_weights(probs, config)
    cash_floor = config["risk"]["cash_floor"]
    investable = 1.0 - cash_floor
    expected_spy = investable * (0.6 * 0.35 + 0.3 * 0.15 + 0.1 * 0.0)
    assert abs(weights["SPY"] - expected_spy) < 1e-9


def test_weights_sum_to_one(config):
    """All weights including cash must sum to 1.0."""
    probs = {"bull": 0.5, "sideways": 0.3, "bear": 0.2}
    weights = compute_target_weights(probs, config)
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9


def test_cash_floor_always_present(config):
    """Cash floor is always at least the configured minimum."""
    for probs in [
        {"bull": 1.0, "sideways": 0.0, "bear": 0.0},
        {"bull": 0.0, "sideways": 0.0, "bear": 1.0},
        {"bull": 0.33, "sideways": 0.34, "bear": 0.33},
    ]:
        weights = compute_target_weights(probs, config)
        assert weights["_cash"] >= config["risk"]["cash_floor"] - 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_allocator.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement src/model/allocator.py**

```python
# src/model/allocator.py


def compute_target_weights(
    regime_probs: dict[str, float],
    config: dict,
) -> dict[str, float]:
    """Compute blended target portfolio weights from regime probabilities.

    Blends regime-specific allocation tables using the posterior probabilities,
    then scales by the investable balance (total - cash floor).

    Args:
        regime_probs: Dict mapping regime name to probability.
            E.g. {"bull": 0.6, "sideways": 0.3, "bear": 0.1}
        config: Full application config dict.

    Returns:
        Dict mapping ticker to target weight (fraction of total portfolio),
        plus "_cash" key for the cash floor.
        All values sum to 1.0.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_allocator.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/allocator.py tests/test_allocator.py
git commit -m "feat: portfolio allocator with regime blending and cash floor"
```

---

## Chunk 4: Risk Management

### Task 8: Risk Checks

**Files:**
- Create: `src/trading/risk.py`
- Create: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests for risk checks**

```python
# tests/test_risk.py
import pytest
from datetime import date
from src.trading.risk import (
    check_position_limits,
    check_rebalance_needed,
    check_drawdown,
    check_turnover,
    is_trading_day,
)


def test_check_position_limits_caps_excess():
    """Positions exceeding max_position_pct are capped and excess redistributed."""
    weights = {"SPY": 0.50, "QQQ": 0.20, "TLT": 0.10, "GLD": 0.10, "_cash": 0.10}
    capped = check_position_limits(weights, max_position_pct=0.40)
    assert capped["SPY"] <= 0.40
    total = sum(capped.values())
    assert abs(total - 1.0) < 1e-9


def test_check_position_limits_no_change_if_within():
    """Weights within limits are unchanged."""
    weights = {"SPY": 0.30, "QQQ": 0.25, "TLT": 0.25, "_cash": 0.20}
    capped = check_position_limits(weights, max_position_pct=0.40)
    assert capped == weights


def test_rebalance_needed_when_deviation_exceeds_threshold():
    """Rebalance is needed when any position deviates more than threshold."""
    current = {"SPY": 0.35, "QQQ": 0.25, "TLT": 0.20, "_cash": 0.20}
    target = {"SPY": 0.28, "QQQ": 0.25, "TLT": 0.27, "_cash": 0.20}
    assert check_rebalance_needed(current, target, threshold_pp=0.05) is True


def test_rebalance_not_needed_when_within_threshold():
    """No rebalance when all positions within threshold."""
    current = {"SPY": 0.30, "QQQ": 0.25, "TLT": 0.25, "_cash": 0.20}
    target = {"SPY": 0.32, "QQQ": 0.24, "TLT": 0.24, "_cash": 0.20}
    assert check_rebalance_needed(current, target, threshold_pp=0.05) is False


def test_drawdown_triggers_circuit_breaker():
    """15% drawdown from peak triggers circuit breaker."""
    peak = 10000.0
    current = 8400.0  # 16% drawdown
    assert check_drawdown(current, peak, threshold=0.15) is True


def test_drawdown_does_not_trigger_within_threshold():
    """10% drawdown does not trigger 15% circuit breaker."""
    peak = 10000.0
    current = 9100.0  # 9% drawdown
    assert check_drawdown(current, peak, threshold=0.15) is False


def test_turnover_within_limit():
    """Turnover within limit passes check."""
    current = {"SPY": 0.30, "QQQ": 0.30, "TLT": 0.20, "_cash": 0.20}
    target = {"SPY": 0.25, "QQQ": 0.35, "TLT": 0.20, "_cash": 0.20}
    capped = check_turnover(current, target, max_turnover=0.50)
    # Only 10% total turnover, well within 50% limit
    assert capped == target


def test_turnover_exceeding_limit_is_scaled():
    """Turnover exceeding limit is scaled down toward current weights."""
    current = {"SPY": 0.40, "QQQ": 0.00, "TLT": 0.40, "_cash": 0.20}
    target = {"SPY": 0.00, "QQQ": 0.40, "TLT": 0.00, "_cash": 0.60}
    capped = check_turnover(current, target, max_turnover=0.50)
    # Total turnover would be 120%, should be scaled to 50%
    turnover = sum(abs(capped[k] - current[k]) for k in current)
    assert turnover <= 0.50 + 1e-9


def test_is_trading_day_weekday():
    """A normal weekday should be a trading day (assuming not a holiday)."""
    # 2024-01-03 is a Wednesday, not a holiday
    assert is_trading_day(date(2024, 1, 3)) is True


def test_is_trading_day_weekend():
    """Weekends are not trading days."""
    # 2024-01-06 is a Saturday
    assert is_trading_day(date(2024, 1, 6)) is False


def test_is_trading_day_holiday():
    """NYSE holidays are not trading days."""
    # 2024-01-15 is MLK Day
    assert is_trading_day(date(2024, 1, 15)) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_risk.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement src/trading/risk.py**

```python
# src/trading/risk.py
from datetime import date

import pandas_market_calendars as mcal


_nyse = mcal.get_calendar("NYSE")


def check_position_limits(
    weights: dict[str, float], max_position_pct: float
) -> dict[str, float]:
    """Cap any position exceeding the max and redistribute excess proportionally.

    Args:
        weights: Dict of ticker -> weight (including "_cash").
        max_position_pct: Maximum allowed weight for any single ETF.

    Returns:
        Adjusted weights that sum to 1.0 with no position exceeding the cap.
    """
    result = dict(weights)
    cash_key = "_cash"

    for _ in range(10):  # iterate to handle cascading redistributions
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

        # Redistribute excess proportionally among uncapped positions
        # using snapshot of pre-redistribution values
        uncapped_total = sum(uncapped.values())
        for k, orig_val in uncapped.items():
            if uncapped_total > 0:
                result[k] = orig_val + excess * (orig_val / uncapped_total)

    return result


def check_rebalance_needed(
    current: dict[str, float],
    target: dict[str, float],
    threshold_pp: float,
) -> bool:
    """Check if any position deviates more than threshold (absolute percentage points).

    Args:
        current: Current portfolio weights.
        target: Target portfolio weights.
        threshold_pp: Threshold in absolute percentage points (e.g., 0.05 = 5pp).

    Returns:
        True if rebalancing is needed.
    """
    all_keys = set(current) | set(target)
    for key in all_keys:
        cur = current.get(key, 0.0)
        tgt = target.get(key, 0.0)
        if abs(cur - tgt) > threshold_pp:
            return True
    return False


def check_drawdown(
    current_value: float, peak_value: float, threshold: float
) -> bool:
    """Check if portfolio drawdown exceeds the circuit breaker threshold.

    Args:
        current_value: Current portfolio value.
        peak_value: Peak portfolio value (rolling 30-day).
        threshold: Drawdown threshold (e.g., 0.15 = 15%).

    Returns:
        True if circuit breaker should fire.
    """
    if peak_value <= 0:
        return False
    drawdown = (peak_value - current_value) / peak_value
    return drawdown >= threshold


def check_turnover(
    current: dict[str, float],
    target: dict[str, float],
    max_turnover: float,
) -> dict[str, float]:
    """Scale target weights toward current if total turnover exceeds the limit.

    Turnover = sum of absolute weight changes across all positions.

    Args:
        current: Current portfolio weights.
        target: Target portfolio weights.
        max_turnover: Maximum allowed total turnover (e.g., 0.50 = 50%).

    Returns:
        Adjusted target weights with turnover within the limit.
    """
    all_keys = set(current) | set(target)
    total_turnover = sum(
        abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in all_keys
    )

    if total_turnover <= max_turnover:
        return dict(target)

    # Scale changes toward current weights
    scale = max_turnover / total_turnover
    result = {}
    for k in all_keys:
        cur = current.get(k, 0.0)
        tgt = target.get(k, 0.0)
        result[k] = cur + scale * (tgt - cur)

    return result


def is_trading_day(d: date) -> bool:
    """Check if a given date is a NYSE trading day.

    Args:
        d: Date to check.

    Returns:
        True if the NYSE is open on this date.
    """
    schedule = _nyse.valid_days(start_date=d, end_date=d)
    return len(schedule) > 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_risk.py -v
```

Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/trading/risk.py tests/test_risk.py
git commit -m "feat: risk management (position limits, drawdown, turnover, holiday calendar)"
```

---

## Chunk 5: Backtesting Engine

### Task 9: Backtesting Engine

**Files:**
- Create: `src/backtest/engine.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests for backtester**

```python
# tests/test_backtest.py
import pytest
import numpy as np
import pandas as pd
from src.backtest.engine import run_backtest


@pytest.fixture
def backtest_config():
    """Minimal config for backtesting."""
    return {
        "mode": "backtest",
        "etf_universe": ["SPY", "TLT"],
        "hmm": {
            "n_states": 3,
            "training_window_years": 1,
            "retrain_frequency": "monthly",
        },
        "allocation": {
            "bull": {"SPY": 0.80, "TLT": 0.20},
            "sideways": {"SPY": 0.50, "TLT": 0.50},
            "bear": {"SPY": 0.20, "TLT": 0.80},
        },
        "risk": {
            "cash_floor": 0.10,
            "max_position_pct": 0.80,
            "rebalance_threshold_pp": 0.05,
            "max_daily_turnover": 0.50,
            "drawdown_circuit_breaker": 0.15,
        },
        "backtest": {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "initial_capital": 10000,
            "transaction_cost": 1.0,
        },
        "data_dir": None,  # will use tmp_path
    }


@pytest.fixture
def backtest_prices():
    """3 years of synthetic SPY and TLT prices for backtesting.

    1 year training + 1 year test period.
    """
    dates = pd.bdate_range("2018-01-02", "2020-12-31")
    np.random.seed(42)
    spy_prices = 270.0 + np.cumsum(np.random.randn(len(dates)) * 1.5)
    tlt_prices = 120.0 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    records = []
    for i, date in enumerate(dates):
        for ticker, prices in [("SPY", spy_prices), ("TLT", tlt_prices)]:
            records.append({
                "date": date,
                "ticker": ticker,
                "open": prices[i],
                "high": prices[i] + 1.0,
                "low": prices[i] - 1.0,
                "close": prices[i],
                "volume": 1000000,
            })
    return pd.DataFrame(records)


def test_backtest_returns_results(backtest_config, backtest_prices, tmp_data_dir):
    """Backtest returns a results dict with expected keys."""
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert "total_return" in results
    assert "annualized_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert "equity_curve" in results
    assert "trade_log" in results
    assert "regime_history" in results
    assert "win_rate" in results


def test_backtest_equity_curve_starts_at_initial_capital(
    backtest_config, backtest_prices, tmp_data_dir
):
    """Equity curve starts at the initial capital."""
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert abs(results["equity_curve"].iloc[0] - 10000.0) < 1.0


def test_backtest_max_drawdown_is_negative(
    backtest_config, backtest_prices, tmp_data_dir
):
    """Max drawdown should be a negative number (or zero)."""
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert results["max_drawdown"] <= 0.0


def test_backtest_trade_log_not_empty(
    backtest_config, backtest_prices, tmp_data_dir
):
    """Backtest should produce at least some trades."""
    backtest_config["data_dir"] = tmp_data_dir
    results = run_backtest(backtest_config, backtest_prices)
    assert len(results["trade_log"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_backtest.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement src/backtest/engine.py**

```python
# src/backtest/engine.py
import numpy as np
import pandas as pd

from src.data.features import compute_features
from src.model.hmm import RegimeHMM
from src.model.allocator import compute_target_weights
from src.trading.risk import check_rebalance_needed, check_position_limits, check_turnover


def run_backtest(config: dict, prices: pd.DataFrame) -> dict:
    """Run a walk-forward backtest using the HMM strategy.

    Walk-forward: train on a rolling window, predict next day, step forward.
    Uses the same model/allocator/risk code as live trading.

    Args:
        config: Full application config.
        prices: DataFrame with columns [date, ticker, open, high, low, close, volume].

    Returns:
        Dict with keys: total_return, annualized_return, sharpe_ratio,
        max_drawdown, equity_curve, trade_log, regime_history.
    """
    etf_universe = config["etf_universe"]
    initial_capital = config["backtest"]["initial_capital"]
    transaction_cost = config["backtest"]["transaction_cost"]
    training_years = config["hmm"]["training_window_years"]
    risk_cfg = config["risk"]
    backtest_start = pd.Timestamp(config["backtest"]["start_date"])
    backtest_end = config["backtest"].get("end_date")
    if backtest_end:
        backtest_end = pd.Timestamp(backtest_end)

    # Compute features from full price history
    features = compute_features(prices)
    features["date"] = pd.to_datetime(features["date"])

    # Build price lookup: date -> ticker -> close
    price_lookup = {}
    for _, row in prices.iterrows():
        d = pd.Timestamp(row["date"])
        if d not in price_lookup:
            price_lookup[d] = {}
        price_lookup[d][row["ticker"]] = row["close"]

    # Get trading dates in the backtest window
    feature_dates = features["date"].sort_values().unique()
    test_dates = [d for d in feature_dates if d >= backtest_start]
    if backtest_end:
        test_dates = [d for d in test_dates if d <= backtest_end]

    training_days = int(training_years * 252)

    # Portfolio state
    cash = initial_capital
    holdings: dict[str, float] = {t: 0.0 for t in etf_universe}  # shares held

    equity_curve = []
    trade_log = []
    regime_history = []
    last_train_month = None

    model = RegimeHMM(n_states=config["hmm"]["n_states"])

    for test_date in test_dates:
        # Get feature index for this date
        date_mask = features["date"] <= test_date
        available = features[date_mask]

        if len(available) < training_days + 1:
            continue

        # Retrain monthly
        current_month = test_date.month
        if last_train_month != current_month:
            train_data = available.iloc[-training_days:]
            model.train(train_data)
            last_train_month = current_month

        if not model.is_trained:
            continue

        # Get regime probabilities
        window = available.iloc[-training_days:]
        probs = model.predict(window)
        dominant_regime = max(probs, key=probs.get)
        regime_history.append({"date": test_date, "regime": dominant_regime, **probs})

        # Compute target weights
        target_weights = compute_target_weights(probs, config)
        target_weights = check_position_limits(target_weights, risk_cfg["max_position_pct"])

        # Current portfolio value
        day_prices = price_lookup.get(test_date, {})
        if not day_prices:
            continue

        portfolio_value = cash
        for ticker in etf_universe:
            if ticker in day_prices:
                portfolio_value += holdings[ticker] * day_prices[ticker]

        # Current weights
        current_weights = {"_cash": cash / portfolio_value if portfolio_value > 0 else 1.0}
        for ticker in etf_universe:
            if ticker in day_prices and portfolio_value > 0:
                current_weights[ticker] = (holdings[ticker] * day_prices[ticker]) / portfolio_value
            else:
                current_weights[ticker] = 0.0

        # Check if rebalance is needed
        if not check_rebalance_needed(current_weights, target_weights, risk_cfg["rebalance_threshold_pp"]):
            equity_curve.append({"date": test_date, "value": portfolio_value})
            continue

        # Apply turnover cap
        target_weights = check_turnover(current_weights, target_weights, risk_cfg["max_daily_turnover"])

        # Execute trades (simulated)
        day_trades = []
        for ticker in etf_universe:
            if ticker not in day_prices:
                continue
            price = day_prices[ticker]
            target_value = target_weights.get(ticker, 0.0) * portfolio_value
            current_value = holdings[ticker] * price
            diff_value = target_value - current_value

            if abs(diff_value) < 1.0:  # skip tiny trades
                continue

            shares_delta = diff_value / price
            holdings[ticker] += shares_delta
            cost = transaction_cost if abs(shares_delta) > 0 else 0
            cash -= diff_value + cost

            action = "BUY" if shares_delta > 0 else "SELL"
            day_trades.append({
                "date": test_date,
                "ticker": ticker,
                "action": action,
                "shares": abs(shares_delta),
                "price": price,
                "cost": cost,
            })

        trade_log.extend(day_trades)

        # Recalculate portfolio value after trades
        portfolio_value = cash
        for ticker in etf_universe:
            if ticker in day_prices:
                portfolio_value += holdings[ticker] * day_prices[ticker]

        equity_curve.append({"date": test_date, "value": portfolio_value})

    # Compute metrics
    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": pd.Series(dtype=float),
            "trade_log": [],
            "regime_history": pd.DataFrame(),
        }

    equity_series = eq_df.set_index("date")["value"]
    total_return = (equity_series.iloc[-1] / initial_capital) - 1.0

    n_days = len(equity_series)
    annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1.0

    daily_returns = equity_series.pct_change().dropna()

    # Sharpe ratio (excess return with 0% risk-free rate assumption)
    sharpe_ratio = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    rolling_max = equity_series.cummax()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Win rate: % of days with positive returns on rebalance days
    rebalance_dates = {t["date"] for t in trade_log}
    rebalance_returns = daily_returns[daily_returns.index.isin(rebalance_dates)]
    win_rate = 0.0
    if len(rebalance_returns) > 0:
        win_rate = (rebalance_returns > 0).mean()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "equity_curve": equity_series,
        "trade_log": trade_log,
        "regime_history": pd.DataFrame(regime_history),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_backtest.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/engine.py tests/test_backtest.py
git commit -m "feat: walk-forward backtesting engine with realistic constraints"
```

---

### Task 10: Backtest Report & Visualization

**Files:**
- Create: `src/backtest/report.py`

- [ ] **Step 1: Implement src/backtest/report.py**

```python
# src/backtest/report.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def print_summary(results: dict):
    """Print backtest summary metrics to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:      {results['total_return']:>10.2%}")
    print(f"  Annualized Return: {results['annualized_return']:>10.2%}")
    print(f"  Sharpe Ratio:      {results['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown:      {results['max_drawdown']:>10.2%}")
    print(f"  Win Rate:          {results.get('win_rate', 0):>10.2%}")
    print(f"  Total Trades:      {len(results['trade_log']):>10d}")
    print("=" * 60 + "\n")


def generate_charts(results: dict, output_dir: str) -> list[str]:
    """Generate backtest visualization charts and save as PNG files.

    Args:
        results: Backtest results dict from run_backtest.
        output_dir: Directory to save chart files.

    Returns:
        List of file paths to generated charts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    equity = results["equity_curve"]
    regime_df = results["regime_history"]

    if equity.empty:
        return paths

    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label="HMM Strategy", linewidth=1.5)
    ax.axhline(y=equity.iloc[0], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = str(out / "equity_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 2. Drawdown chart
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color="red")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = str(out / "drawdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 3. Regime timeline
    if not regime_df.empty:
        regime_colors = {"bull": "green", "sideways": "gold", "bear": "red"}
        fig, ax = plt.subplots(figsize=(12, 3))
        dates = pd.to_datetime(regime_df["date"])
        colors = [regime_colors.get(r, "gray") for r in regime_df["regime"]]
        ax.bar(dates, [1] * len(dates), color=colors, width=1.5)
        ax.set_title("Detected Market Regimes")
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=r) for r, c in regime_colors.items()]
        ax.legend(handles=legend_elements, loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        path = str(out / "regime_timeline.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # Generate HTML report wrapping all charts and metrics
    if paths:
        html_path = str(out / "report.html")
        html_lines = [
            "<html><head><title>HMM Trader Backtest Report</title></head><body>",
            "<h1>Backtest Report</h1>",
            f"<p>Total Return: {results['total_return']:.2%}</p>",
            f"<p>Annualized Return: {results['annualized_return']:.2%}</p>",
            f"<p>Sharpe Ratio: {results['sharpe_ratio']:.2f}</p>",
            f"<p>Max Drawdown: {results['max_drawdown']:.2%}</p>",
            f"<p>Win Rate: {results.get('win_rate', 0):.2%}</p>",
            f"<p>Total Trades: {len(results['trade_log'])}</p>",
        ]
        for p in paths:
            fname = Path(p).name
            html_lines.append(f'<img src="{fname}" style="max-width:100%"><br>')
        html_lines.append("</body></html>")
        with open(html_path, "w") as f:
            f.write("\n".join(html_lines))
        paths.append(html_path)

    return paths
```

- [ ] **Step 2: Verify it imports without error**

```bash
python -c "from src.backtest.report import print_summary, generate_charts; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/backtest/report.py
git commit -m "feat: backtest report with metrics summary and chart generation"
```

---

## Chunk 6: Email Alerts

### Task 11: Email Alerting

**Files:**
- Create: `src/alerts/email.py`
- Create: `tests/test_email.py`

- [ ] **Step 1: Write failing tests for email alerts**

```python
# tests/test_email.py
import pytest
from unittest.mock import patch, MagicMock
from src.alerts.email import send_alert, format_rebalance_email, format_error_email


def test_format_rebalance_email():
    """Rebalance email contains regime and trade info."""
    body = format_rebalance_email(
        regime_probs={"bull": 0.7, "sideways": 0.2, "bear": 0.1},
        trades=[
            {"ticker": "SPY", "action": "BUY", "shares": 5, "price": 450.0},
        ],
        portfolio_value=10000.0,
    )
    assert "bull" in body.lower()
    assert "SPY" in body
    assert "BUY" in body


def test_format_error_email():
    """Error email contains error message."""
    body = format_error_email(
        error_msg="Connection timeout",
        context="Daily rebalance job",
    )
    assert "Connection timeout" in body
    assert "Daily rebalance" in body


@patch("src.alerts.email.smtplib.SMTP")
def test_send_alert_connects_and_sends(mock_smtp_class):
    """send_alert connects to SMTP server and sends message."""
    mock_smtp = MagicMock()
    mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

    send_alert(
        subject="Test Alert",
        body="Test body",
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_user="user@test.com",
        smtp_password="password",
        to_email="recipient@test.com",
    )

    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("user@test.com", "password")
    mock_smtp.send_message.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_email.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement src/alerts/email.py**

```python
# src/alerts/email.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_alert(
    subject: str,
    body: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
):
    """Send an email alert via SMTP.

    Args:
        subject: Email subject line.
        body: Email body (plain text).
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        smtp_user: SMTP login username.
        smtp_password: SMTP login password.
        to_email: Recipient email address.
    """
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = f"[HMM Trader] {subject}"
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def format_rebalance_email(
    regime_probs: dict[str, float],
    trades: list[dict],
    portfolio_value: float,
) -> str:
    """Format a rebalance notification email body."""
    lines = ["Daily Rebalance Executed", "=" * 40, ""]

    lines.append("Regime Probabilities:")
    for regime, prob in sorted(regime_probs.items()):
        lines.append(f"  {regime:>10s}: {prob:.1%}")

    lines.append(f"\nPortfolio Value: ${portfolio_value:,.2f}")
    lines.append(f"\nTrades ({len(trades)}):")

    if trades:
        for t in trades:
            lines.append(
                f"  {t['action']:>4s} {t.get('shares', 0):>8.2f} {t['ticker']}"
                f" @ ${t.get('price', 0):>10.2f}"
            )
    else:
        lines.append("  No trades executed.")

    return "\n".join(lines)


def format_error_email(error_msg: str, context: str) -> str:
    """Format an error notification email body."""
    lines = [
        "SYSTEM ERROR",
        "=" * 40,
        "",
        f"Context: {context}",
        "",
        "Error:",
        error_msg,
        "",
        "Please check the system logs for details.",
    ]
    return "\n".join(lines)


def format_weekly_summary_email(
    week_return: float,
    total_value: float,
    regime: str,
    regime_probs: dict[str, float],
    holdings: dict[str, float],
) -> str:
    """Format a weekly summary email body (sent on Fridays, end of trading week)."""
    lines = [
        "Weekly Summary",
        "=" * 40,
        "",
        f"Portfolio Value: ${total_value:,.2f}",
        f"Week Return:     {week_return:.2%}",
        f"Current Regime:  {regime}",
        "",
        "Regime Probabilities:",
    ]
    for r, p in sorted(regime_probs.items()):
        lines.append(f"  {r:>10s}: {p:.1%}")
    lines.append("")
    lines.append("Holdings:")
    for ticker, shares in sorted(holdings.items()):
        lines.append(f"  {ticker}: {shares:.2f} shares")
    return "\n".join(lines)


def format_circuit_breaker_email(
    current_value: float,
    peak_value: float,
    drawdown_pct: float,
) -> str:
    """Format a circuit breaker triggered email body."""
    lines = [
        "CIRCUIT BREAKER TRIGGERED",
        "=" * 40,
        "",
        f"Peak Value:    ${peak_value:,.2f}",
        f"Current Value: ${current_value:,.2f}",
        f"Drawdown:      {drawdown_pct:.1%}",
        "",
        "All positions have been liquidated.",
        "Manual restart required: python -m src.main reset-circuit-breaker",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_email.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/alerts/email.py tests/test_email.py
git commit -m "feat: email alerting for rebalance, errors, and circuit breaker events"
```

---

## Chunk 7: IBKR Integration & Portfolio Tracking

### Task 12: Portfolio Tracking

**Files:**
- Create: `src/trading/portfolio.py`

- [ ] **Step 1: Implement src/trading/portfolio.py**

```python
# src/trading/portfolio.py


def compute_current_weights(
    holdings: dict[str, float],
    prices: dict[str, float],
    cash: float,
) -> dict[str, float]:
    """Compute current portfolio weights from holdings, prices, and cash.

    Args:
        holdings: Dict of ticker -> number of shares held.
        prices: Dict of ticker -> current price per share.
        cash: Cash balance in the account.

    Returns:
        Dict of ticker -> weight (fraction of total portfolio), plus "_cash" key.
        All values sum to 1.0.
    """
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


def compute_orders(
    current_holdings: dict[str, float],
    target_weights: dict[str, float],
    prices: dict[str, float],
    total_value: float,
) -> list[dict]:
    """Compute the orders needed to move from current holdings to target weights.

    Args:
        current_holdings: Dict of ticker -> shares currently held.
        target_weights: Dict of ticker -> target weight (fraction of total portfolio).
        prices: Dict of ticker -> current price.
        total_value: Total portfolio value (holdings + cash).

    Returns:
        List of order dicts with keys: ticker, action, shares, price.
        Only includes orders where shares > 0.
    """
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
        if shares < 0.01:  # skip negligible trades
            continue

        orders.append({
            "ticker": ticker,
            "action": "BUY" if diff_value > 0 else "SELL",
            "shares": round(shares, 4),
            "price": prices[ticker],
        })

    return orders
```

- [ ] **Step 2: Verify it imports without error**

```bash
python -c "from src.trading.portfolio import compute_current_weights, compute_orders; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/trading/portfolio.py
git commit -m "feat: portfolio weight computation and order calculation"
```

---

### Task 13: IBKR Executor

**Files:**
- Create: `src/trading/executor.py`

- [ ] **Step 1: Implement src/trading/executor.py**

This module wraps the `ibapi` client. It cannot be unit tested without a running IBKR gateway, so it is tested via paper trading integration tests manually.

```python
# src/trading/executor.py
import logging
import time
import threading

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

logger = logging.getLogger(__name__)


class IBKRApp(EWrapper, EClient):
    """Thin wrapper around ibapi for submitting MOC orders and querying positions."""

    def __init__(self):
        EClient.__init__(self, self)
        self.positions: dict[str, float] = {}
        self.cash: float = 0.0
        self.order_fills: dict[int, dict] = {}
        self._next_order_id: int | None = None
        self._position_done = threading.Event()
        self._account_done = threading.Event()

    def nextValidId(self, orderId: int):
        self._next_order_id = orderId

    def position(self, account, contract, pos, avgCost):
        self.positions[contract.symbol] = float(pos)

    def positionEnd(self):
        self._position_done.set()

    def updateAccountValue(self, key, val, currency, accountName):
        if key == "CashBalance" and currency == "USD":
            self.cash = float(val)

    def accountDownloadEnd(self, accountName):
        self._account_done.set()

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.order_fills[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avg_fill_price": avgFillPrice,
        }

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in (2104, 2106, 2158):  # informational codes
            logger.error(f"IBKR Error {errorCode}: {errorString}")


def connect_ibkr(host: str, port: int, client_id: int) -> IBKRApp:
    """Connect to IBKR gateway and return the app instance."""
    app = IBKRApp()
    app.connect(host, port, client_id)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    # Wait for connection
    timeout = 10
    while app._next_order_id is None and timeout > 0:
        time.sleep(0.5)
        timeout -= 0.5

    if app._next_order_id is None:
        raise ConnectionError("Failed to connect to IBKR gateway")

    return app


def get_positions_and_cash(app: IBKRApp) -> tuple[dict[str, float], float]:
    """Query current positions and cash balance from IBKR."""
    app.positions = {}
    app.cash = 0.0
    app._position_done.clear()
    app._account_done.clear()

    app.reqPositions()
    app._position_done.wait(timeout=10)

    app.reqAccountUpdates(True, "")
    app._account_done.wait(timeout=10)
    app.reqAccountUpdates(False, "")

    return app.positions, app.cash


def submit_moc_orders(
    app: IBKRApp,
    orders: list[dict],
) -> list[dict]:
    """Submit Market-on-Close orders to IBKR.

    Args:
        app: Connected IBKRApp instance.
        orders: List of order dicts with keys: ticker, action, shares.

    Returns:
        List of submitted order dicts with added order_id.
    """
    submitted = []
    for order_spec in orders:
        contract = Contract()
        contract.symbol = order_spec["ticker"]
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        order = Order()
        order.action = order_spec["action"]
        order.totalQuantity = int(order_spec["shares"])  # whole shares for ETFs
        order.orderType = "MOC"
        order.tif = "DAY"

        order_id = app._next_order_id
        app._next_order_id += 1

        app.placeOrder(order_id, contract, order)
        logger.info(
            f"Submitted MOC order #{order_id}: {order_spec['action']} "
            f"{order_spec['shares']} {order_spec['ticker']}"
        )

        submitted.append({**order_spec, "order_id": order_id})

    return submitted


def check_fills(app: IBKRApp, order_ids: list[int], timeout: int = 60) -> dict[int, dict]:
    """Wait for order fills and return fill status.

    Args:
        app: Connected IBKRApp instance.
        order_ids: List of order IDs to check.
        timeout: Max seconds to wait for fills.

    Returns:
        Dict mapping order_id to fill info dict.
    """
    end_time = time.time() + timeout
    while time.time() < end_time:
        all_filled = all(
            app.order_fills.get(oid, {}).get("status") == "Filled"
            for oid in order_ids
        )
        if all_filled:
            break
        time.sleep(1)

    return {oid: app.order_fills.get(oid, {"status": "Unknown"}) for oid in order_ids}
```

- [ ] **Step 2: Verify it imports without error**

```bash
python -c "from src.trading.executor import IBKRApp, connect_ibkr; print('OK')"
```

Note: `ibapi` must be installed separately — it's not on PyPI. Install via:
```bash
pip install ibapi
```
If unavailable, download from IBKR's website and install manually. For now, verify the import works or skip if ibapi is not yet installed.

- [ ] **Step 3: Commit**

```bash
git add src/trading/executor.py
git commit -m "feat: IBKR executor with MOC order submission and fill checking"
```

---

## Chunk 8: Main Entry Point & Deployment

### Task 14: Main Entry Point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Implement src/main.py**

```python
# src/main.py
"""HMM Trader — main entry point.

Commands:
    run                    Run the daily rebalance job
    backtest               Run a backtest against historical data
    reset-circuit-breaker  Reset the circuit breaker after manual review
"""
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime

from src.config import load_config
from src.data.fetcher import fetch_prices, backfill_history
from src.data.features import compute_features
from src.data.store import DataStore
from src.model.hmm import RegimeHMM
from src.model.allocator import compute_target_weights
from src.trading.portfolio import compute_current_weights, compute_orders
from src.trading.risk import (
    check_position_limits,
    check_rebalance_needed,
    check_drawdown,
    check_turnover,
    is_trading_day,
)
from src.alerts.email import (
    send_alert,
    format_rebalance_email,
    format_error_email,
    format_circuit_breaker_email,
    format_weekly_summary_email,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("hmm_trader")


def _send_email(config: dict, subject: str, body: str):
    """Send an email alert if SMTP is configured."""
    smtp_host = os.environ.get("SMTP_HOST")
    if not smtp_host:
        logger.warning("SMTP not configured, skipping email alert")
        return
    try:
        send_alert(
            subject=subject,
            body=body,
            smtp_host=smtp_host,
            smtp_port=int(os.environ.get("SMTP_PORT", 587)),
            smtp_user=os.environ["SMTP_USER"],
            smtp_password=os.environ["SMTP_PASSWORD"],
            to_email=os.environ["ALERT_EMAIL_TO"],
        )
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def cmd_run(config: dict):
    """Run the daily rebalance job."""
    today = date.today()

    # Check if market is open
    if not is_trading_day(today):
        logger.info(f"Market closed on {today}, skipping rebalance")
        return

    store = DataStore(config["data_dir"])

    # Check circuit breaker
    if store.get_circuit_breaker_active():
        logger.warning("Circuit breaker is active. Run reset-circuit-breaker to re-enable trading.")
        return

    etf_universe = config["etf_universe"]

    # Check if historical data exists; backfill on first run
    existing_prices = store.load_prices()
    if existing_prices.empty:
        logger.info("No historical data found. Backfilling 10 years...")
        history = backfill_history(etf_universe, years=10)
        if history.empty:
            _send_email(config, "Error: Backfill Failed", format_error_email("Historical data fetch returned empty", "Backfill"))
            return
        store.save_prices(history)

    # Fetch latest prices
    logger.info("Fetching latest prices...")
    prices_df = fetch_prices(etf_universe, period="5d")
    if prices_df.empty:
        _send_email(config, "Error: No Price Data", format_error_email("Price fetch returned empty", "Daily run"))
        return

    store.save_prices(prices_df)

    # Compute features from full price history
    all_prices = store.load_prices()

    features = compute_features(all_prices)

    # Load or train model
    model = RegimeHMM(n_states=config["hmm"]["n_states"])
    try:
        model.load(config["data_dir"])
        logger.info("Loaded existing model")
    except FileNotFoundError:
        logger.info("No existing model found, training from scratch...")

    # Retrain if needed (first trading day of month or no model)
    training_window = config["hmm"]["training_window_years"] * 252
    last_retrain_month = store.get_system_state("last_retrain_month")
    current_month_key = f"{today.year}-{today.month:02d}"
    needs_retrain = not model.is_trained or last_retrain_month != current_month_key
    if needs_retrain:
        if len(features) >= training_window:
            train_data = features.iloc[-training_window:]
            model.train(train_data)
            model.save(config["data_dir"])
            store.set_system_state("last_retrain_month", current_month_key)
            logger.info("Model trained and saved")
        else:
            logger.error(f"Not enough data to train: {len(features)} < {training_window}")
            return

    # Predict regime
    window = features.iloc[-training_window:]
    regime_probs = model.predict(window)
    dominant = max(regime_probs, key=regime_probs.get)
    logger.info(f"Regime: {dominant} | Probs: {json.dumps({k: f'{v:.2%}' for k, v in regime_probs.items()})}")

    # Save prediction
    store.save_prediction(
        date=str(today),
        bull_prob=regime_probs["bull"],
        sideways_prob=regime_probs["sideways"],
        bear_prob=regime_probs["bear"],
        regime=dominant,
    )

    # Compute target weights
    target_weights = compute_target_weights(regime_probs, config)
    target_weights = check_position_limits(target_weights, config["risk"]["max_position_pct"])

    if config["mode"] == "backtest":
        logger.info("Backtest mode — skipping trade execution")
        return

    # Connect to IBKR
    from src.trading.executor import connect_ibkr, get_positions_and_cash, submit_moc_orders

    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.environ.get("IBKR_PORT", 7497))
    ibkr_client_id = int(os.environ.get("IBKR_CLIENT_ID", 1))

    try:
        app = connect_ibkr(ibkr_host, ibkr_port, ibkr_client_id)
    except ConnectionError as e:
        _send_email(config, "Error: IBKR Connection Failed", format_error_email(str(e), "IBKR connect"))
        return

    try:
        holdings, cash = get_positions_and_cash(app)
        total_value = cash + sum(
            holdings.get(t, 0) * float(prices_df[prices_df["ticker"] == t]["close"].iloc[-1])
            for t in etf_universe
            if t in holdings and len(prices_df[prices_df["ticker"] == t]) > 0
        )

        # Get latest prices as dict
        latest_prices = {}
        for t in etf_universe:
            t_prices = prices_df[prices_df["ticker"] == t]
            if not t_prices.empty:
                latest_prices[t] = float(t_prices.iloc[-1]["close"])

        # Check drawdown
        peak = store.get_peak_value() or total_value
        if total_value > peak:
            peak = total_value
            store.set_peak_value(peak)

        if check_drawdown(total_value, peak, config["risk"]["drawdown_circuit_breaker"]):
            logger.warning("CIRCUIT BREAKER TRIGGERED")
            store.set_circuit_breaker_active(True)
            drawdown_pct = (peak - total_value) / peak
            _send_email(
                config,
                "CIRCUIT BREAKER TRIGGERED",
                format_circuit_breaker_email(total_value, peak, drawdown_pct),
            )
            # Liquidate all positions
            liquidation_orders = []
            for ticker, shares in holdings.items():
                if shares > 0 and ticker in latest_prices:
                    liquidation_orders.append({
                        "ticker": ticker,
                        "action": "SELL",
                        "shares": shares,
                        "price": latest_prices[ticker],
                    })
            if liquidation_orders:
                submit_moc_orders(app, liquidation_orders)
                logger.info(f"Liquidated {len(liquidation_orders)} positions")
                for order in liquidation_orders:
                    store.save_trade(
                        date=str(today),
                        ticker=order["ticker"],
                        action="SELL",
                        shares=order["shares"],
                        price=order["price"],
                        cost=1.0,
                    )
            return

        # Compute current weights
        current_weights = compute_current_weights(holdings, latest_prices, cash)

        # Check rebalance needed
        if not check_rebalance_needed(current_weights, target_weights, config["risk"]["rebalance_threshold_pp"]):
            logger.info("No rebalance needed — all positions within threshold")
            _send_email(
                config,
                "No Rebalance Needed",
                format_rebalance_email(regime_probs, [], total_value),
            )
            return

        # Apply turnover cap
        target_weights = check_turnover(current_weights, target_weights, config["risk"]["max_daily_turnover"])

        # Compute and submit orders
        orders = compute_orders(holdings, target_weights, latest_prices, total_value)
        if orders:
            submitted = submit_moc_orders(app, orders)
            logger.info(f"Submitted {len(submitted)} orders")
            _send_email(
                config,
                "Rebalance Executed",
                format_rebalance_email(regime_probs, orders, total_value),
            )

            # Log trades
            for order in orders:
                store.save_trade(
                    date=str(today),
                    ticker=order["ticker"],
                    action=order["action"],
                    shares=order["shares"],
                    price=order["price"],
                    cost=1.0,
                )

        # Save snapshot
        store.save_snapshot(
            date=str(today),
            total_value=total_value,
            cash=cash,
            holdings=holdings,
        )

        # Send weekly summary on Fridays (covers the week)
        if today.weekday() == 4:  # Friday
            snapshots = store.load_snapshots()
            if len(snapshots) >= 5:
                week_ago_value = snapshots.iloc[-5]["total_value"]
                week_return = (total_value - week_ago_value) / week_ago_value
            else:
                week_return = 0.0
            _send_email(
                config,
                "Weekly Summary",
                format_weekly_summary_email(week_return, total_value, dominant, regime_probs, holdings),
            )

    finally:
        app.disconnect()


def cmd_backtest(config: dict):
    """Run a backtest against historical data."""
    from src.backtest.engine import run_backtest
    from src.backtest.report import print_summary, generate_charts

    etf_universe = config["etf_universe"]

    logger.info("Fetching historical data for backtest...")
    prices = backfill_history(etf_universe, years=10)

    logger.info("Running backtest...")
    results = run_backtest(config, prices)

    print_summary(results)

    output_dir = os.path.join(config["data_dir"], "backtest_results")
    charts = generate_charts(results, output_dir)
    if charts:
        logger.info(f"Charts saved to {output_dir}")


def cmd_reset_circuit_breaker(config: dict):
    """Reset the circuit breaker after manual review."""
    store = DataStore(config["data_dir"])
    store.set_circuit_breaker_active(False)

    # Reset peak to current value (will be updated on next run)
    store.set_peak_value(0.0)

    logger.info("Circuit breaker reset. Trading will resume on next run.")


def main():
    parser = argparse.ArgumentParser(description="HMM Stock Trader")
    parser.add_argument(
        "command",
        choices=["run", "backtest", "reset-circuit-breaker"],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config override name (e.g. 'paper' or 'live')",
    )
    args = parser.parse_args()

    config = load_config(config_dir="config", override=args.config)

    commands = {
        "run": cmd_run,
        "backtest": cmd_backtest,
        "reset-circuit-breaker": cmd_reset_circuit_breaker,
    }

    try:
        commands[args.command](config)
    except Exception as e:
        logger.exception(f"Fatal error in {args.command}")
        _send_email(config, f"Fatal Error: {args.command}", format_error_email(str(e), args.command))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses help without error**

```bash
python -m src.main --help
```

Expected: Shows usage with run/backtest/reset-circuit-breaker commands

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat: CLI entry point with run, backtest, and reset-circuit-breaker commands"
```

---

### Task 15: Deployment Files

**Files:**
- Create: `Dockerfile`
- Create: `railway.toml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command — overridden by Railway cron
CMD ["python", "-m", "src.main", "run", "--config", "paper"]
```

- [ ] **Step 2: Create railway.toml**

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
# Railway cron: run at 3:30 PM ET on weekdays
# Use 19:30 UTC (correct for EDT, Mar-Nov). During EST (Nov-Mar) this is 2:30 PM ET,
# which is still before the 3:45 PM MOC cutoff — safe year-round.
# For exact 3:30 PM ET year-round, use Railway's timezone support or two schedules.
cronSchedule = "30 19 * * 1-5"
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile railway.toml
git commit -m "feat: Dockerfile and Railway cron config for deployment"
```

---

### Task 16: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
cd /home/dwagner003/dev/hmm-trader
source .venv/bin/activate
pytest tests/ -v --tb=short
```

Expected: All tests PASS

- [ ] **Step 2: Fix any failures and re-run**

If any tests fail, fix the root cause and re-run until all pass.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: test suite fixes"
```

---

### Task 17: Run First Backtest

- [ ] **Step 1: Run a backtest to validate the full pipeline end-to-end**

```bash
cd /home/dwagner003/dev/hmm-trader
source .venv/bin/activate
python -m src.main backtest
```

Expected: Downloads historical data, runs walk-forward backtest, prints summary metrics, generates charts in `data/backtest_results/`.

- [ ] **Step 2: Review the output**

Check that:
- Summary metrics are printed (total return, Sharpe, max drawdown)
- Charts are generated in `data/backtest_results/`
- No errors in output

- [ ] **Step 3: Commit the working state**

```bash
git add -A
git commit -m "chore: verify end-to-end backtest pipeline works"
```
