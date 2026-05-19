"""Shared model classes so pickle can find them across scripts."""

import numpy as np


class NumpyRidge:
    """Ridge regression on standardized features — runs anywhere numpy runs."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.mu_ = None
        self.sigma_ = None
        self.beta_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyRidge":
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-9
        Xz = (X - self.mu_) / self.sigma_
        Xz = np.hstack([np.ones((Xz.shape[0], 1)), Xz])
        n_features = Xz.shape[1]
        reg = self.alpha * np.eye(n_features)
        reg[0, 0] = 0
        coef = np.linalg.solve(Xz.T @ Xz + reg, Xz.T @ y)
        self.intercept_ = float(coef[0])
        self.beta_ = coef[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xz = (X - self.mu_) / self.sigma_
        return self.intercept_ + Xz @ self.beta_
