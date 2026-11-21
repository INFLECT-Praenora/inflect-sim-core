"""Neural backend — wrap an existing viva-torch surrogate behind the framework
interface. Engine-agnostic: the rollout seeds the net's features from ``x0`` (via
``overrides``) and uses a caller-supplied ``core_provider`` (default
``viva_torch.build_core``), so this backend is not coupled to any one model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _overrides_from_x0(feature_paths, x0) -> dict:
    """Nest the x0 vector at the surrogate's feature paths (the seed state)."""
    x0 = np.asarray(x0, float)
    ov: dict = {}
    for p, v in zip(feature_paths, x0):
        d = ov
        for key in p[:-1]:
            d = d.setdefault(key, {})
        d[p[-1]] = float(v)
    return ov


class _NeuralFitted:
    def __init__(self, *, spec, feature_paths, observable_names, timestep,
                 ckpt_path, core_provider: Callable, rollout_fn: Callable):
        self.spec = spec
        self.feature_paths = feature_paths
        self.observable_names = list(observable_names)
        self.timestep = timestep
        self._ckpt = ckpt_path
        self._core_provider = core_provider
        self._rollout_fn = rollout_fn

    def rollout(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        x0 = np.asarray(x0, float)
        if n_steps <= 1:
            return x0.reshape(1, -1)
        overrides = _overrides_from_x0(self.feature_paths, x0)
        traj = self._rollout_fn(self.spec, self._ckpt, self._core_provider(),
                                overrides, n_steps - 1, self.timestep)
        return np.vstack([x0, np.asarray(traj, float)])


class NeuralBackend:
    """Wrap a trained viva-torch checkpoint as a FittedSurrogate (via from_checkpoint)."""
    name = "neural"

    def fit(self, trajectories, observable_names, dt):  # pragma: no cover - v1 path is from_checkpoint
        raise NotImplementedError(
            "NeuralBackend v1 wraps a trained checkpoint — use NeuralBackend.from_checkpoint(path). "
            "Training lives in viva_simulation_core.api.surrogate_curate / viva_torch.train_surrogate."
        )

    @staticmethod
    def from_checkpoint(path: str, *, core_provider: Callable | None = None,
                        rollout_fn: Callable | None = None) -> _NeuralFitted:
        import torch
        from viva_torch.spec import SurrogateSpec
        if core_provider is None:
            from viva_torch import build_core as core_provider  # type: ignore
        if rollout_fn is None:
            from viva_torch.evaluate import _rollout_surrogate as rollout_fn  # type: ignore
        spec = SurrogateSpec(**torch.load(path, map_location="cpu", weights_only=False)["spec"])
        obs = [p[-1] for p in spec.target_paths_t]
        return _NeuralFitted(spec=spec, feature_paths=spec.feature_paths,
                             observable_names=obs, timestep=spec.timestep,
                             ckpt_path=path, core_provider=core_provider, rollout_fn=rollout_fn)

    @staticmethod
    def _fitted_for_test(*, feature_paths, target_names, timestep, ckpt_path,
                         core_provider, rollout_fn) -> _NeuralFitted:
        """Construct a _NeuralFitted with an injected rollout_fn (no real net) — for wiring tests."""
        return _NeuralFitted(spec=None, feature_paths=feature_paths,
                             observable_names=target_names, timestep=timestep,
                             ckpt_path=ckpt_path, core_provider=core_provider, rollout_fn=rollout_fn)
