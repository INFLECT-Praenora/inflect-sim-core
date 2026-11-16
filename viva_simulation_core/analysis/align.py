"""Align BioGears-engine and SBML sepsis observable series onto a common clock.

Input shape (each side): a sequence of samples
``[{'time': <seconds>, <observable>: <float>, ...}, ...]``.

Grid: union of both sides' sample times inside the global overlap
``[max(t0), min(t1)]``. Each observable is then interpolated linearly, but only
at grid points inside that observable's OWN overlap on both sides; points
outside are NaN (never extrapolated/held) and are excluded from the metrics.
An observable with no usable data on a side is treated as absent from it.
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
             if f in scenario_a and f in scenario_b and scenario_a[f] != scenario_b[f]}
    missing = [f for f in SCENARIO_FIELDS if f not in scenario_a or f not in scenario_b]
    if missing:
        raise ScenarioMismatch(f'scenario field(s) missing: {missing}')
    if diffs:
        raise ScenarioMismatch(f'scenarios differ (field: (a, b)): {diffs}')


@dataclass
class ObservableDivergence:
    max_abs: float
    max_rel: float  # informational only; NOT used in the pass/fail gate
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
    no_overlap: list = field(default_factory=list)  # present on both, ranges disjoint
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
                      rtol: float = 0.05, atol: float = 1e-6,
                      require_all: bool = True) -> AlignedComparison:
    """Resample both sides to a shared grid and gate per-observable divergence.

    An observable passes when ``max|e-s| <= atol + rtol * max|e|`` over the grid.
    ``passed`` is True only if every compared observable passes and at least one
    was compared; with ``require_all`` (default) it is also False if any requested
    observable is not comparable on both sides (only_*, missing_both, no_overlap).
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
        xe, ye = _col(te, re_, name) if name in ne else (np.array([]), np.array([]))
        xs, ys = _col(ts, rs, name) if name in ns else (np.array([]), np.array([]))
        in_e, in_s = len(xe) > 0, len(xs) > 0
        if in_e and in_s:
            clo, chi = max(xe[0], xs[0]), min(xe[-1], xs[-1])
            ok = (grid >= clo) & (grid <= chi)
            if not ok.any():
                out.no_overlap.append(name)
                continue
            e = np.where(ok, np.interp(grid, xe, ye), np.nan)
            s_ = np.where(ok, np.interp(grid, xs, ys), np.nan)
            out.engine[name], out.sbml[name] = e, s_
            e, s_ = e[ok], s_[ok]
            diff = np.abs(e - s_)
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
    incomplete = out.only_engine or out.only_sbml or out.missing_both or out.no_overlap
    out.passed = (bool(out.per_observable)
                  and all(d.passed for d in out.per_observable.values())
                  and not (require_all and incomplete))
    return out
