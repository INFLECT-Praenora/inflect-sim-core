"""Operator workflow: train a viva-torch surrogate of a model and curate it.

Writes ``<model_hash>.pt`` (the net) and ``<model_hash>.json`` (the envelope
sidecar read by ``CheckpointStore``) into ``directory``. Species are modelled
as scalar paths ``stores/species/<id>``. Import-guarded: needs the
``[surrogate]`` extra.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .modeling import model_hash
from .surrogate_backend import SurrogateEntry, SurrogateUnavailable

try:
    from viva_torch import (SamplingPlan, SurrogateSpec, TrajectorySampler,
                            evaluate_surrogate, train_surrogate)
    from viva_torch.dataset import TransitionDataset
    HAS_CURATE = True
except ImportError:
    HAS_CURATE = False


@dataclass
class CurationResult:
    model_hash: str
    entry: SurrogateEntry
    checkpoint_path: Path
    sidecar_path: Path
    train_s: float
    sample_s: float


def _all_species(source: str, model_format: str) -> dict[str, float]:
    from .modeling import load_engine

    rr = load_engine(source, model_format)
    return {i: float(rr[i]) for i in rr.model.getFloatingSpeciesIds()}


def _document(source: str, model_format: str, species: dict[str, float], interval: float) -> dict:
    from viva_tellurium import make_tellurium_document

    # Seed EVERY species: a partial seed is dropped when the process's species map is
    # larger than the seed (the store is then rebuilt from the schema defaults).
    species = {**_all_species(source, model_format), **species}

    d = make_tellurium_document(model=source, model_format=model_format,
                                species_overrides=dict(species), interval=interval)
    d["tellurium"]["config"]["bulk_species"] = False  # scalar leaves at stores/species/<id>
    d.setdefault("stores", {})["species"] = dict(species)  # seed t=0 for the sampler
    return {"state": d}


def _default_species(source: str, model_format: str, observables: list[str]) -> dict[str, float]:
    from .modeling import load_engine

    rr = load_engine(source, model_format)
    return {o: float(rr[o]) for o in observables}


def train_and_curate(*, source: str, model_format: str, observables: list[str],
                     directory: str | Path, timestep_s: float, n_trajectories: int = 16,
                     n_steps: int = 30, ic_spread: float = 0.5, hidden=(32, 32),
                     epochs: int = 300, lr: float = 2e-3, seed: int = 0) -> CurationResult:
    """Sample trajectories (initial species scaled by U(1-s, 1+s)), train, evaluate
    on a held-out IC, and write checkpoint + sidecar into ``directory``."""
    if not HAS_CURATE:
        raise SurrogateUnavailable("viva-torch not installed (pip install '.[surrogate]')")
    import numpy as np

    from viva_simulation_core.core import build_core

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mh = model_hash(source, model_format)
    base = _default_species(source, model_format, observables)
    targets = [["stores", "species", o] for o in observables]
    spec = SurrogateSpec(target_paths=targets, driver_paths=[], timestep=timestep_s,
                         parameterization="delta")
    core = build_core()
    rng = np.random.default_rng(seed)

    def draw() -> dict[str, float]:
        # species starting at 0 stay at 0 (scale-free) so the envelope is well defined
        f = rng.uniform(1 - ic_spread, 1 + ic_spread)
        return {k: v * f for k, v in base.items()}

    ics = [draw() for _ in range(n_trajectories)]
    t0 = time.perf_counter()
    Xs, Ys, DTs, ids = [], [], [], []
    lo = np.full(len(observables), np.inf)
    hi = np.full(len(observables), -np.inf)
    for tid, ic in enumerate(ics):
        ds = TrajectorySampler(_document(source, model_format, ic, timestep_s), spec, core).sample(
            SamplingPlan(n_steps, timestep_s, [{}]))
        k = len(observables)  # features are exactly the targets (no drivers); Y is the raw next state
        both = np.concatenate([ds.X[:, :k], ds.Y[:, :k]])
        lo, hi = np.minimum(lo, both.min(axis=0)), np.maximum(hi, both.max(axis=0))
        Xs.append(ds.X); Ys.append(ds.Y); DTs.append(ds.DT); ids.append(ds.traj_id + tid)
    data = TransitionDataset(X=np.concatenate(Xs), Y=np.concatenate(Ys), DT=np.concatenate(DTs),
                             traj_id=np.concatenate(ids), spec=spec)
    sample_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    net, hist = train_surrogate(data, hidden=hidden, epochs=epochs, lr=lr, seed=seed)
    train_s = time.perf_counter() - t0

    ckpt = directory / f"{mh}.pt"
    net.save(str(ckpt))

    held = draw()
    metrics = evaluate_surrogate(
        _document(source, model_format, held, timestep_s), spec, str(ckpt), build_core,
        initial_states=[{"stores": {"species": held}}], n_steps=n_steps, timestep=timestep_s)

    state_ranges = {o: (min(ic[o] for ic in ics), max(ic[o] for ic in ics)) for o in observables}
    entry = SurrogateEntry(
        checkpoint_ref=ckpt.name, targets=list(observables), horizon_s=n_steps * timestep_s,
        param_ranges={}, state_ranges=state_ranges,
        target_bounds={o: (float(lo[i]), float(hi[i])) for i, o in enumerate(observables)},
        val_error={"overall_nrmse": float(metrics["overall_nrmse"]),
                   "per_target": {k: {kk: float(vv) for kk, vv in m.items()}
                                  for k, m in metrics["per_target"].items()},
                   "val_loss": float(hist["val_loss"][-1])})
    sidecar = directory / f"{mh}.json"
    sidecar.write_text(json.dumps({
        "checkpoint_ref": entry.checkpoint_ref, "targets": entry.targets,
        "horizon_s": entry.horizon_s, "param_ranges": {},
        "state_ranges": {k: list(v) for k, v in state_ranges.items()},
        "target_bounds": {k: list(v) for k, v in entry.target_bounds.items()},
        "val_error": entry.val_error}, indent=2))
    return CurationResult(mh, entry, ckpt, sidecar, train_s, sample_s)
