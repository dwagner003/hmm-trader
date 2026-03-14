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
