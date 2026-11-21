"""The validate harness — score any ``FittedSurrogate`` honestly.

Four metrics, two gates:
  - rollout_nrmse   : normalized RMSE of a rollout vs its matched truth trajectory.
  - physicality     : fraction of rollout values within declared physical bounds. (gate)
  - responsiveness  : does the output actually vary when the driver (initial condition)
                      varies? — the min over observables of the coefficient of variation
                      of per-driver rollout endpoints. A degenerate surrogate scores ~0. (gate)
  - speedup         : wall-clock vs the exact engine (passed in; reporting only).

verdict = PASS iff BOTH gates hold. Speed never buys a pass.
"""
from __future__ import annotations

import numpy as np

from viva_simulation_core.surrogate.interface import FittedSurrogate, Scorecard

_EPS = 1e-9


def _rollout_nrmse(fitted: FittedSurrogate, truths: list[np.ndarray]) -> float:
    errs, scales = [], []
    for t in truths:
        pred = np.asarray(fitted.rollout(t[0], len(t)), float)
        t = np.asarray(t, float)
        n = min(len(pred), len(t))
        errs.append(pred[:n] - t[:n])
        scales.append(t[:n])
    e = np.concatenate([x.ravel() for x in errs])
    s = np.concatenate([x.ravel() for x in scales])
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return rmse / (float(np.std(s)) + _EPS)


def _physicality(fitted: FittedSurrogate, driver_ics: list[np.ndarray],
                 bounds: dict[str, tuple[float, float]], n_steps: int) -> float:
    names = list(fitted.observable_names)
    lo = np.array([bounds.get(k, (-np.inf, np.inf))[0] for k in names])
    hi = np.array([bounds.get(k, (-np.inf, np.inf))[1] for k in names])
    inside, total = 0, 0
    for ic in driver_ics:
        r = np.asarray(fitted.rollout(ic, n_steps), float)
        ok = (r >= lo) & (r <= hi)
        inside += int(np.count_nonzero(ok))
        total += int(ok.size)
    return inside / total if total else 0.0


def _responsiveness(fitted: FittedSurrogate, driver_ics: list[np.ndarray],
                    n_steps: int) -> float:
    ends = np.array([np.asarray(fitted.rollout(ic, n_steps), float)[-1] for ic in driver_ics])
    # per-observable coefficient of variation across drivers; min over observables
    cv = np.std(ends, axis=0) / (np.abs(np.mean(ends, axis=0)) + _EPS)
    return float(np.min(cv))


def validate_surrogate(fitted: FittedSurrogate, truth_trajs: list[np.ndarray],
                       driver_ics: list[np.ndarray], bounds: dict[str, tuple[float, float]],
                       dt: float, *, speedup: float | None = None,
                       resp_gate: float = 0.02, phys_gate: float = 0.95) -> Scorecard:
    if not truth_trajs:
        raise ValueError("truth_trajs is empty — need at least one held-out truth trajectory")
    if not driver_ics:
        raise ValueError("driver_ics is empty — need ≥2 driver initial conditions to test responsiveness")
    if len(driver_ics) < 2:
        raise ValueError("responsiveness needs ≥2 driver initial conditions")

    n_steps = len(np.asarray(truth_trajs[0]))
    nrmse = _rollout_nrmse(fitted, truth_trajs)
    phys = _physicality(fitted, driver_ics, bounds, n_steps)
    resp = _responsiveness(fitted, driver_ics, n_steps)
    gates = {"responsive": resp >= resp_gate, "physical": phys >= phys_gate}
    name = getattr(fitted, "name", type(fitted).__name__)
    return Scorecard(backend=name, rollout_nrmse=nrmse, physicality=phys,
                     responsiveness=resp, speedup=speedup, gates=gates)
