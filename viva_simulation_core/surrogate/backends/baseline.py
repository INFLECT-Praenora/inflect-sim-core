"""Trivial baseline backends — the floor the harness scores against, and the
pluggability proof (more than one method behind the same interface).

Rollout convention (framework-wide): ``rollout(x0, n)`` returns ``n`` rows, row 0
is ``x0``, each subsequent row is one applied step.
"""
from __future__ import annotations

import numpy as np


class _PersistenceFitted:
    def __init__(self, observable_names: list[str]):
        self.observable_names = observable_names

    def rollout(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        return np.tile(np.asarray(x0, float), (n_steps, 1))


class PersistenceBackend:
    """Holds the initial condition constant. Responsive to the IC, useless on
    dynamics — the honest floor."""
    name = "persistence"

    def fit(self, trajectories, observable_names, dt) -> _PersistenceFitted:
        return _PersistenceFitted(list(observable_names))


class _LinearFitted:
    def __init__(self, observable_names: list[str], A: np.ndarray, b: np.ndarray):
        self.observable_names = observable_names
        self._A, self._b = A, b

    def rollout(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        out = [np.asarray(x0, float)]
        for _ in range(n_steps - 1):
            out.append(self._A @ out[-1] + self._b)
        return np.array(out)


class LinearBackend:
    """Fits a global linear map x_{t+1} = A x_t + b by least squares on the
    transition pairs across all trajectories."""
    name = "linear"

    def fit(self, trajectories, observable_names, dt) -> _LinearFitted:
        X, Y = [], []
        for t in trajectories:
            t = np.asarray(t, float)
            X.append(t[:-1]); Y.append(t[1:])
        X = np.concatenate(X); Y = np.concatenate(Y)
        n = X.shape[1]
        # augment with a bias column, solve [X 1] W = Y  ->  W = [A^T; b]
        Xa = np.hstack([X, np.ones((len(X), 1))])
        W, *_ = np.linalg.lstsq(Xa, Y, rcond=None)   # (n+1, n)
        A = W[:n].T
        b = W[n]
        return _LinearFitted(list(observable_names), A, b)
