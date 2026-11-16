"""S4: cross-engine (process-bigraph vs viva-core) runtime + determinism."""
import pytest

viva_core = pytest.importorskip('viva_core')

from viva_simulation_core.analysis.run_s4_engines import (  # noqa: E402
    run_under_both, run_viva_core)
from viva_simulation_core.composites.sepsis_sbml import make_sepsis_sbml_composite  # noqa: E402


def test_viva_core_can_drive_tellurium():
    doc = make_sepsis_sbml_composite(severity=0, interval_s=5.0)
    r = run_viva_core(doc, 3)
    assert r['sim_time'] == 15.0
    f = r['final']
    assert abs(f['SBP'] - 114.0) < 5 and abs(f['HR'] - 72.0) < 5
    assert f['blood_pathogen'] < 1e-6


def test_cross_engine_determinism():
    r = run_under_both(severity=0, n_ticks=6, interval_s=10.0)
    assert r['verdict'] in ('bit-identical', 'within-tolerance'), r['abs_diff']
    r1 = run_under_both(severity=1, n_ticks=6, interval_s=10.0, infection_start_s=5.0)
    assert r1['verdict'] in ('bit-identical', 'within-tolerance'), r1['abs_diff']
    assert r1['python']['final']['blood_pathogen'] > 1e-3
