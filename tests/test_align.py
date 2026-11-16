import numpy as np
import pytest

from viva_simulation_core.analysis.align import (
    ScenarioMismatch, align_observables, assert_scenario_equal)


def _series(times, **fns):
    return [{'time': float(t), **{k: f(t) for k, f in fns.items()}} for t in times]


def test_align_resamples_and_gates():
    eng = _series(np.arange(0, 101, 10), MAP=lambda t: 93.0, HR=lambda t: 72.0 + 0.01 * t)
    # different clock (offset, finer) and tiny noise
    sb = _series(np.arange(0, 121, 5), MAP=lambda t: 93.0 + 1e-4, HR=lambda t: 72.0 + 0.01 * t,
                 CO=lambda t: 5.0)
    res = align_observables(eng, sb, ['MAP', 'HR', 'CO'], rtol=1e-2, atol=1e-3)
    assert res.time[0] == 0.0 and res.time[-1] == 100.0  # overlap only
    assert len(res.engine['MAP']) == len(res.sbml['MAP']) == len(res.time)
    assert res.passed
    assert res.only_sbml == ['CO'] and res.only_engine == []
    assert res.per_observable['MAP'].passed

    bad = _series(np.arange(0, 121, 5), MAP=lambda t: 93.0 + (20.0 if t > 50 else 0),
                  HR=lambda t: 72.0 + 0.01 * t)
    res2 = align_observables(eng, bad, ['MAP', 'HR'], rtol=1e-2, atol=1e-3)
    assert not res2.passed
    assert not res2.per_observable['MAP'].passed and res2.per_observable['HR'].passed
    assert res2.per_observable['MAP'].max_abs == pytest.approx(20.0)


def test_no_overlap_raises():
    with pytest.raises(ValueError):
        align_observables(_series([0, 1], MAP=lambda t: 1), _series([5, 6], MAP=lambda t: 1), ['MAP'])


def test_scenario_equality():
    a = dict(severity=0, patient='StandardMale', horizon_s=3600.0, infection_start_s=None)
    assert_scenario_equal(a, dict(a))
    for k, v in [('severity', 2), ('horizon_s', 600.0), ('patient', 'X'), ('infection_start_s', 60.0)]:
        with pytest.raises(ScenarioMismatch):
            assert_scenario_equal(a, {**a, k: v})
