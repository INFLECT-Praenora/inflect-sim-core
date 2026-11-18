"""Real viva-torch neural-surrogate backend (import-guarded; torch stays optional).

Operator-curated only: nets are loaded exclusively from ``directory`` (the
``checkpoint_ref`` in a sidecar is resolved *inside* that directory); caller
bytes are never passed to ``torch.load``. The checkpoint directory is TRUSTED:
loading uses ``torch.load(weights_only=False)``, which executes pickled code, so
only operator-written files may live there. Loaded nets are cached by path and
NOT invalidated if a checkpoint is re-curated in place (restart to refresh).

The surrogate runs at its TRAINED timestep (``NeuralProcess`` refuses any other
interval). The rollout uses the trained step and the series is linearly resampled
onto the exact engine's grid (``round(h/interval)+1`` points from 0 to ``horizon_s``).
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

from .surrogate_backend import CheckpointStore, SurrogateEntry, SurrogateUnavailable

try:  # guarded: base / [api] installs must not need torch
    from viva_torch import SurrogateNet, neural_process_node
    HAS_TORCH_BACKEND = True
except ImportError:
    HAS_TORCH_BACKEND = False


def require_torch_backend() -> None:
    if not HAS_TORCH_BACKEND:
        raise SurrogateUnavailable("viva-torch not installed (pip install '.[surrogate]')")


def species_name(path) -> str:
    return str(path[-1])


class TorchCheckpointBackend(CheckpointStore):
    """``CheckpointStore`` sidecar lookup + real torch inference."""

    def __init__(self, directory: str | Path):
        require_torch_backend()
        super().__init__(directory)
        self._nets: dict[str, Any] = {}
        self._core = None
        self._lock = threading.Lock()

    def _resolve(self, ref: str) -> Path:
        root = self.directory.resolve()
        path = (root / ref).resolve()  # absolute refs resolve to themselves, then fail below
        if root not in path.parents or not path.is_file():
            raise SurrogateUnavailable("checkpoint not found in curated directory")
        return path

    def _net(self, ref: str):
        path = self._resolve(ref)
        key = str(path)
        with self._lock:
            net = self._nets.get(key)
            if net is None:
                net = SurrogateNet.load(key)
                self._nets[key] = net
            return net, key

    def _get_core(self):
        if self._core is None:
            from viva_simulation_core.core import build_core
            self._core = build_core()
        return self._core

    def predict(self, entry: SurrogateEntry, request: Any, seed_state: dict | None = None) -> dict:
        import numpy as np
        from process_bigraph import Composite

        t_start = time.perf_counter()
        net, ckpt = self._net(entry.checkpoint_ref)
        spec = net.spec
        dt = float(spec.timestep)
        names = [species_name(p) for p in spec.target_paths]
        want = [o for o in request.observables if o in entry.targets and o in names]
        if len(want) != len(request.observables):
            raise SurrogateUnavailable("requested observables not covered by the surrogate")
        seed_state = seed_state or {}
        missing = [n for n in names if n not in seed_state]
        if missing:
            raise SurrogateUnavailable(f"seed state missing {missing}")

        horizon = float(request.horizon_s)
        n = max(1, math.ceil(horizon / dt - 1e-9))  # rollout covers the whole horizon
        stores: dict = {}
        for p, name in zip(spec.target_paths, names):
            node = stores
            for k in p[:-1]:
                node = node.setdefault(k, {})
            node[p[-1]] = float(seed_state[name])
        state = {**stores, "surrogate": neural_process_node(checkpoint=ckpt, spec=spec, interval=dt)}
        comp = Composite({"state": state}, core=self._get_core())

        def read() -> list[float]:
            vals = []
            for p in spec.target_paths:
                node = comp.state
                for k in p:
                    node = node[k]
                vals.append(float(node))
            return vals

        traj = [read()]
        for _ in range(n):
            comp.run(dt)
            traj.append(read())
        arr = np.asarray(traj)
        grid = dt * np.arange(n + 1)
        # same grid as the exact engine: round(h/interval)+1 points from 0 to horizon
        n_out = max(1, int(round(horizon / float(request.interval_s))))
        out_t = np.linspace(0.0, horizon, n_out + 1)
        series = {name: np.interp(out_t, grid, arr[:, i]).tolist()
                  for i, name in enumerate(names) if name in want}
        return {
            "observables": series,
            "times": [float(t) for t in out_t],
            "provenance": {"integrator": "viva-torch", "n_steps": n, "seed": None,
                           "wall_ms": (time.perf_counter() - t_start) * 1000.0,
                           "lib_versions": {}},
        }
