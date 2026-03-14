# tests/test_hmm.py
import pytest
import numpy as np
import pandas as pd
from src.model.hmm import RegimeHMM


@pytest.fixture
def synthetic_features():
    """Synthetic 3-feature data with obvious regime structure."""
    np.random.seed(42)
    n = 200
    bull = np.column_stack([
        np.random.normal(0.001, 0.005, n),
        np.random.normal(0.008, 0.001, n),
        np.random.normal(0.0005, 0.003, n),
    ])
    bear = np.column_stack([
        np.random.normal(-0.001, 0.015, n),
        np.random.normal(0.020, 0.003, n),
        np.random.normal(-0.001, 0.005, n),
    ])
    sideways = np.column_stack([
        np.random.normal(0.0, 0.008, n),
        np.random.normal(0.012, 0.002, n),
        np.random.normal(0.0, 0.004, n),
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
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    assert model.is_trained


def test_hmm_state_labels_assigned(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    assert model.state_labels is not None
    assert set(model.state_labels.values()) == {"bull", "bear", "sideways"}


def test_hmm_bull_state_has_highest_mean_return(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    bull_idx = [k for k, v in model.state_labels.items() if v == "bull"][0]
    bear_idx = [k for k, v in model.state_labels.items() if v == "bear"][0]
    assert model.model.means_[bull_idx, 0] > model.model.means_[bear_idx, 0]


def test_hmm_predict_returns_probabilities(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    probs = model.predict(synthetic_features)
    assert "bull" in probs and "bear" in probs and "sideways" in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_hmm_predict_bull_data_returns_high_bull_prob(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    bull_data = synthetic_features.iloc[:200]
    probs = model.predict(bull_data)
    assert probs["bull"] > 0.5


def test_hmm_save_and_load(synthetic_features, tmp_data_dir):
    model = RegimeHMM(n_states=3)
    model.train(synthetic_features)
    probs_before = model.predict(synthetic_features)
    model.save(tmp_data_dir)
    loaded = RegimeHMM(n_states=3)
    loaded.load(tmp_data_dir)
    probs_after = loaded.predict(synthetic_features)
    for key in probs_before:
        assert abs(probs_before[key] - probs_after[key]) < 1e-9
