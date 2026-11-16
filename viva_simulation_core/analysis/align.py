"""Align BioGears-engine and SBML sepsis observable series onto a common clock.

Input shape (each side): a sequence of samples
``[{'time': <seconds>, <observable>: <float>, ...}, ...]``.

Grid: linear interpolation onto the union of both sides' sample times that fall
inside the overlap ``[max(t0), min(t1)]`` (no extrapolation). Using the union
keeps every native sample of both runs, so no feature is smoothed away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES

SCENARIO_FIELDS = ('severity', 'patient', 'horizon_s', 'infection_start_s')


class ScenarioMismatch(ValueError):
    """Raised when two runs are not the same scenario."""


def assert_scenario_equal(scenario_a: Mapping, scenario_b: Mapping) -> None:
    diffs = {f: (scenario_a.get(f), scenario_b.get(f)) for f in SCENARIO_FIELDS
             if scenario_a.get(f) != scenario_b.get(f)}
    if diffs:
        raise ScenarioMismatch(f'scenarios differ (field: (a, b)): {diffs}')


@dataclass
class ObservableDivergence:
    max_abs: float
    max_rel: float
    passed: bool


@dataclass
class AlignedComparison:
    time: np.ndarray
    engine: dict = field(default_factory=dict)   # observable -> array on `time`
    sbml: dict = field(default_factory=dict)
    per_observable: dict = field(default_factory=dict)  # name -> ObservableDivergence
    only_engine: list = field(default_factory=list)
    only_sbml: list = field(default_factory=list)
    missing_both: list = field(default_factory=list)
    passed: bool = False


def _columns(samples: Sequence[Mapping]):
    rows = sorted(samples, key=lambda r: r['time'])
    t = np.array([r['time'] for r in rows], dtype=float)
    names = {k for r in rows for k in r if k != 'time'}
    return t, rows, names


def _col(t, rows, name):
    pts = [(tt, r[name]) for tt, r in zip(t, rows) if r.get(name) is not None]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts], dtype=float)


def align_observables(engine: Sequence[Mapping], sbml: Sequence[Mapping],
                      observables: Iterable[str] = SEPSIS_OBSERVABLES, *,
                      rtol: float = 0.05, atol: float = 1e-6) -> AlignedComparison:
    """Resample both sides to a shared grid and gate per-observable divergence.

    An observable passes when ``max|e-s| <= atol + rtol * max|e|`` over the grid.
    ``passed`` is True only if every observable present on both sides passes and
    at least one was compared.
    """
    te, re_, ne = _columns(engine)
    ts, rs, ns = _columns(sbml)
    if not len(te) or not len(ts):
        raise ValueError('empty series')
    lo, hi = max(te[0], ts[0]), min(te[-1], ts[-1])
    if lo > hi:
        raise ValueError(f'no time overlap: engine [{te[0]}, {te[-1]}], sbml [{ts[0]}, {ts[-1]}]')
    grid = np.union1d(te, ts)
    grid = grid[(grid >= lo) & (grid <= hi)]

    out = AlignedComparison(time=grid)
    for name in observables:
        in_e, in_s = name in ne, name in ns
        if in_e and in_s:
            xe, ye = _col(te, re_, name)
            xs, ys = _col(ts, rs, name)
            e, s = np.interp(grid, xe, ye), np.interp(grid, xs, ys)
            out.engine[name], out.sbml[name] = e, s
            diff = np.abs(e - s)
            max_abs = float(diff.max())
            scale = float(np.abs(e).max())
            max_rel = float((diff / np.maximum(np.abs(e), 1e-12)).max())
            out.per_observable[name] = ObservableDivergence(
                max_abs, max_rel, bool(max_abs <= atol + rtol * scale))
        elif in_e:
            out.only_engine.append(name)
        elif in_s:
            out.only_sbml.append(name)
        else:
            out.missing_both.append(name)
    out.passed = bool(out.per_observable) and all(d.passed for d in out.per_observable.values())
    return out
