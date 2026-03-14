# tests/test_store.py
import pytest
import pandas as pd
from datetime import date, datetime

from src.data.store import DataStore


class TestDataStoreInit:
    def test_creates_tables(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        tables = store.list_tables()
        for expected in ["prices", "features", "predictions", "trades", "snapshots", "system_state"]:
            assert expected in tables

    def test_db_file_created(self, tmp_data_dir):
        import os
        DataStore(tmp_data_dir)
        assert os.path.exists(os.path.join(tmp_data_dir, "hmm_trader.db"))

    def test_idempotent_init(self, tmp_data_dir):
        """Creating store twice should not error (tables already exist)."""
        DataStore(tmp_data_dir)
        DataStore(tmp_data_dir)


class TestPrices:
    def test_save_and_load_prices(self, tmp_data_dir, sample_prices):
        store = DataStore(tmp_data_dir)
        store.save_prices(sample_prices)
        df = store.load_prices("SPY")
        assert len(df) == 30
        assert set(df.columns) >= {"date", "ticker", "open", "high", "low", "close", "volume"}
        assert (df["ticker"] == "SPY").all()

    def test_load_prices_date_filter(self, tmp_data_dir, sample_prices):
        store = DataStore(tmp_data_dir)
        store.save_prices(sample_prices)
        dates = sorted(sample_prices[sample_prices["ticker"] == "SPY"]["date"].unique())
        start = dates[5]
        end = dates[15]
        df = store.load_prices("SPY", start_date=str(pd.Timestamp(start).date()), end_date=str(pd.Timestamp(end).date()))
        assert len(df) == 11  # inclusive on both ends
        assert df["date"].min() >= pd.Timestamp(start)
        assert df["date"].max() <= pd.Timestamp(end)

    def test_no_duplicate_prices(self, tmp_data_dir, sample_prices):
        store = DataStore(tmp_data_dir)
        store.save_prices(sample_prices)
        store.save_prices(sample_prices)  # insert again
        df = store.load_prices("SPY")
        assert len(df) == 30  # still 30, not 60

    def test_load_prices_no_data(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        df = store.load_prices("XYZ")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_prices_multiple_tickers_stored(self, tmp_data_dir, sample_prices):
        store = DataStore(tmp_data_dir)
        store.save_prices(sample_prices)
        spy = store.load_prices("SPY")
        tlt = store.load_prices("TLT")
        assert len(spy) == 30
        assert len(tlt) == 30
        assert (spy["ticker"] == "SPY").all()
        assert (tlt["ticker"] == "TLT").all()


class TestFeatures:
    def _make_features_df(self):
        dates = pd.bdate_range("2024-01-02", periods=10)
        return pd.DataFrame({
            "date": dates,
            "spy_log_return": [0.001 * i for i in range(10)],
            "realized_vol": [0.01 + 0.001 * i for i in range(10)],
            "spy_tlt_spread": [0.0005 * i for i in range(10)],
        })

    def test_save_and_load_features(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        df = self._make_features_df()
        store.save_features(df)
        loaded = store.load_features()
        assert len(loaded) == 10
        assert set(loaded.columns) >= {"date", "spy_log_return", "realized_vol", "spy_tlt_spread"}

    def test_features_date_filter(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        df = self._make_features_df()
        store.save_features(df)
        dates = sorted(df["date"].unique())
        start = str(pd.Timestamp(dates[2]).date())
        end = str(pd.Timestamp(dates[7]).date())
        loaded = store.load_features(start_date=start, end_date=end)
        assert len(loaded) == 6

    def test_features_empty(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        loaded = store.load_features()
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 0


class TestPredictions:
    def test_save_and_load_prediction(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.save_prediction("2024-01-15", 0.6, 0.3, 0.1, "bull")
        df = store.load_predictions()
        assert len(df) == 1
        row = df.iloc[0]
        assert abs(row["bull_prob"] - 0.6) < 1e-9
        assert abs(row["sideways_prob"] - 0.3) < 1e-9
        assert abs(row["bear_prob"] - 0.1) < 1e-9
        assert row["regime"] == "bull"

    def test_load_predictions_empty(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        df = store.load_predictions()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_multiple_predictions(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.save_prediction("2024-01-15", 0.6, 0.3, 0.1, "bull")
        store.save_prediction("2024-01-16", 0.2, 0.5, 0.3, "sideways")
        df = store.load_predictions()
        assert len(df) == 2


class TestTrades:
    def test_save_and_load_trade(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.save_trade("2024-01-15", "SPY", "BUY", 10, 450.0, 1.0)
        df = store.load_trades()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["ticker"] == "SPY"
        assert row["action"] == "BUY"
        assert abs(row["shares"] - 10) < 1e-9
        assert abs(row["price"] - 450.0) < 1e-9
        assert abs(row["cost"] - 1.0) < 1e-9

    def test_load_trades_date_filter(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.save_trade("2024-01-10", "SPY", "BUY", 5, 448.0, 1.0)
        store.save_trade("2024-01-15", "SPY", "SELL", 3, 452.0, 1.0)
        store.save_trade("2024-01-20", "TLT", "BUY", 10, 100.0, 1.0)
        df = store.load_trades(start_date="2024-01-12", end_date="2024-01-18")
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "SPY"

    def test_load_trades_empty(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        df = store.load_trades()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestSnapshots:
    def test_save_and_load_snapshot(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        holdings = {"SPY": 10, "TLT": 5}
        store.save_snapshot("2024-01-15", 10000.0, 3000.0, holdings)
        df = store.load_snapshots()
        assert len(df) == 1
        row = df.iloc[0]
        assert abs(row["total_value"] - 10000.0) < 1e-9
        assert abs(row["cash"] - 3000.0) < 1e-9

    def test_snapshot_holdings_roundtrip(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        holdings = {"SPY": 10, "TLT": 5, "GLD": 3}
        store.save_snapshot("2024-01-15", 10000.0, 2000.0, holdings)
        df = store.load_snapshots()
        import json
        loaded_holdings = json.loads(df.iloc[0]["holdings"])
        assert loaded_holdings == holdings

    def test_load_snapshots_empty(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        df = store.load_snapshots()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestSystemState:
    def test_circuit_breaker_default_false(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        assert store.get_circuit_breaker_active() is False

    def test_circuit_breaker_set_true(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.set_circuit_breaker_active(True)
        assert store.get_circuit_breaker_active() is True

    def test_circuit_breaker_set_false(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.set_circuit_breaker_active(True)
        store.set_circuit_breaker_active(False)
        assert store.get_circuit_breaker_active() is False

    def test_peak_value_default_none(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        assert store.get_peak_value() is None

    def test_peak_value_set_and_get(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.set_peak_value(12345.67)
        assert abs(store.get_peak_value() - 12345.67) < 1e-6

    def test_generic_key_value(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.set_system_state("last_retrain", "2024-01-01")
        assert store.get_system_state("last_retrain") == "2024-01-01"

    def test_generic_key_missing_returns_none(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        assert store.get_system_state("nonexistent") is None

    def test_system_state_upsert(self, tmp_data_dir):
        store = DataStore(tmp_data_dir)
        store.set_system_state("key", "value1")
        store.set_system_state("key", "value2")
        assert store.get_system_state("key") == "value2"

    def test_circuit_breaker_persists_across_instances(self, tmp_data_dir):
        store1 = DataStore(tmp_data_dir)
        store1.set_circuit_breaker_active(True)
        store2 = DataStore(tmp_data_dir)
        assert store2.get_circuit_breaker_active() is True
