# src/model/hmm.py
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeHMM:
    """HMM wrapper for market regime detection with automatic state labeling."""

    FEATURE_COLS = ["spy_log_return", "realized_vol", "spy_tlt_spread"]

    def __init__(self, n_states: int = 3, random_state: int = 10):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None  # type: Optional[GaussianHMM]
        self.state_labels = None  # type: Optional[Dict[int, str]]
        self.is_trained = False

    def train(self, features: pd.DataFrame, n_iter: int = 100):
        """Train the HMM on feature data and assign state labels."""
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
        Bull = highest, Bear = lowest, Sideways = middle."""
        mean_returns = self.model.means_[:, 0]
        sorted_indices = np.argsort(mean_returns)
        self.state_labels = {
            int(sorted_indices[0]): "bear",
            int(sorted_indices[1]): "sideways",
            int(sorted_indices[2]): "bull",
        }

    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """Predict current regime probabilities. Returns dict like {"bull": 0.7, "sideways": 0.2, "bear": 0.1}"""
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
        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "state_labels": self.state_labels}, path / "hmm_model.joblib")

    def load(self, data_dir: str):
        path = Path(data_dir) / "hmm_model.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.state_labels = data["state_labels"]
        self.is_trained = True
