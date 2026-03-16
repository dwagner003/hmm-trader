# src/data/store.py
from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, List, Optional

import pandas as pd


DB_FILENAME = "hmm_trader.db"

_CREATE_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    date      TEXT NOT NULL,
    ticker    TEXT NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    INTEGER,
    PRIMARY KEY (date, ticker)
)
"""

_CREATE_FEATURES = """
CREATE TABLE IF NOT EXISTS features (
    date            TEXT PRIMARY KEY,
    spy_log_return  REAL,
    realized_vol    REAL,
    spy_tlt_spread  REAL
)
"""

_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT NOT NULL,
    bull_prob     REAL,
    sideways_prob REAL,
    bear_prob     REAL,
    regime        TEXT
)
"""

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    date    TEXT NOT NULL,
    ticker  TEXT NOT NULL,
    action  TEXT NOT NULL,
    shares  REAL,
    price   REAL,
    cost    REAL
)
"""

_CREATE_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    total_value REAL,
    cash        REAL,
    holdings    TEXT
)
"""

_CREATE_SYSTEM_STATE = """
CREATE TABLE IF NOT EXISTS system_state (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


class DataStore:
    """SQLite-backed data store for the HMM trader."""

    def __init__(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        self._db_path = os.path.join(data_dir, DB_FILENAME)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            for ddl in [
                _CREATE_PRICES,
                _CREATE_FEATURES,
                _CREATE_PREDICTIONS,
                _CREATE_TRADES,
                _CREATE_SNAPSHOTS,
                _CREATE_SYSTEM_STATE,
            ]:
                conn.execute(ddl)
            conn.commit()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def list_tables(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [r["name"] for r in rows]

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def save_prices(self, df: pd.DataFrame) -> None:
        """Insert or replace OHLCV rows."""
        with self._connect() as conn:
            for _, row in df.iterrows():
                date_str = (
                    row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"])[:10]
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO prices
                        (date, ticker, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        date_str,
                        row["ticker"],
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        int(row["volume"]),
                    ),
                )
            conn.commit()

    def load_prices(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        query = "SELECT date, ticker, open, high, low, close, volume FROM prices WHERE 1=1"
        params: list = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date[:10])
        if end_date:
            query += " AND date <= ?"
            params.append(end_date[:10])
        query += " ORDER BY date"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def save_features(self, df: pd.DataFrame) -> None:
        with self._connect() as conn:
            for _, row in df.iterrows():
                date_str = (
                    row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"])[:10]
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO features
                        (date, spy_log_return, realized_vol, spy_tlt_spread)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        date_str,
                        float(row["spy_log_return"]),
                        float(row["realized_vol"]),
                        float(row["spy_tlt_spread"]),
                    ),
                )
            conn.commit()

    def load_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        query = "SELECT date, spy_log_return, realized_vol, spy_tlt_spread FROM features WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND date >= ?"
            params.append(start_date[:10])
        if end_date:
            query += " AND date <= ?"
            params.append(end_date[:10])
        query += " ORDER BY date"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "spy_log_return", "realized_vol", "spy_tlt_spread"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def save_prediction(
        self,
        date: str,
        bull_prob: float,
        sideways_prob: float,
        bear_prob: float,
        regime: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (date, bull_prob, sideways_prob, bear_prob, regime)
                VALUES (?, ?, ?, ?, ?)
                """,
                (date[:10], bull_prob, sideways_prob, bear_prob, regime),
            )
            conn.commit()

    def load_predictions(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, bull_prob, sideways_prob, bear_prob, regime FROM predictions ORDER BY date"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "bull_prob", "sideways_prob", "bear_prob", "regime"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def save_trade(
        self,
        date: str,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        cost: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (date, ticker, action, shares, price, cost)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (date[:10], ticker, action, shares, price, cost),
            )
            conn.commit()

    def load_trades(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        query = "SELECT date, ticker, action, shares, price, cost FROM trades WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND date >= ?"
            params.append(start_date[:10])
        if end_date:
            query += " AND date <= ?"
            params.append(end_date[:10])
        query += " ORDER BY date"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "action", "shares", "price", "cost"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        date: str,
        total_value: float,
        cash: float,
        holdings_dict: Dict[str, float],
    ) -> None:
        holdings_json = json.dumps(holdings_dict)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO snapshots (date, total_value, cash, holdings)
                VALUES (?, ?, ?, ?)
                """,
                (date[:10], total_value, cash, holdings_json),
            )
            conn.commit()

    def load_snapshots(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, total_value, cash, holdings FROM snapshots ORDER BY date"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "total_value", "cash", "holdings"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # System state
    # ------------------------------------------------------------------

    def get_system_state(self, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_system_state(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def get_circuit_breaker_active(self) -> bool:
        val = self.get_system_state("circuit_breaker_active")
        return val == "true"

    def set_circuit_breaker_active(self, active: bool) -> None:
        self.set_system_state("circuit_breaker_active", "true" if active else "false")

    def get_peak_value(self) -> Optional[float]:
        val = self.get_system_state("peak_value")
        return float(val) if val is not None else None

    def set_peak_value(self, value: float) -> None:
        self.set_system_state("peak_value", str(value))
