"""S4: cross-engine (process-bigraph vs viva-core) runtime + determinism.

All horizons are short (<= 60 s simulated).
"""
import json
import subprocess
import sys

import pytest

viva_core = pytest.importorskip('viva_core')

pytestmark = pytest.mark.slow

from viva_simulation_core.analysis.run_s4_engines import (  # noqa: E402
    run_under_both, run_viva_core)
from viva_simulation_core.composites.sepsis_sbml import make_sepsis_sbml_composite  # noqa: E402


@pytest.fixture(scope='module')
def warm():
    """Explicitly warm roadrunner/both engines so later comparisons are warm-to-warm."""
    for kw in (dict(severity=0), dict(severity=1, infection_start_s=5.0)):
        for _ in range(2):  # same configs/horizon as the tests below, twice
            run_under_both(n_ticks=6, interval_s=10.0, **kw)


def test_viva_core_can_drive_tellurium():
    doc = make_sepsis_sbml_composite(severity=0, interval_s=5.0)
    r = run_viva_core(doc, 3)
    assert r['sim_time'] == 15.0
    f = r['final']
    assert abs(f['SBP'] - 114.0) < 5 and abs(f['HR'] - 72.0) < 5
    assert f['blood_pathogen'] < 1e-6


@pytest.mark.parametrize('kw', [
    dict(severity=0),
    dict(severity=1, infection_start_s=5.0),
])
def test_cross_engine_determinism_warm_bit_identical(warm, kw):
    r = run_under_both(n_ticks=6, interval_s=10.0, **kw)  # 60 s horizon
    assert len(r['abs_diff']) == 9
    assert max(r['abs_diff'].values()) == 0.0, r['abs_diff']
    assert r['verdict'] == 'bit-identical'
    if kw['severity'] == 1:
        assert r['python']['final']['blood_pathogen'] > 1e-3


def test_cross_engine_cold_within_tolerance():
    """COLD case: first roadrunner run in a fresh process (~1e-6 rel offset)."""
    code = ('import json;from viva_simulation_core.analysis.run_s4_engines import run_under_both;'
            'r=run_under_both(severity=0,n_ticks=6,interval_s=10.0,rtol=1e-5);'
            'print("OUT"+json.dumps({"v":r["verdict"],"d":r["abs_diff"]}))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    r = json.loads(out.split('OUT', 1)[1])
    assert r['v'] in ('bit-identical', 'within-tolerance'), r['d']
