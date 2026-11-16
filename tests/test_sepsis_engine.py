"""Engine-side sepsis baseline composite (real BioGears). Doubles as the engine smoke test."""
import pytest
from process_bigraph import Composite, gather_emitter_results

pytest.importorskip('viva_biogears')
from viva_biogears.core import build_core  # noqa: E402
from viva_biogears.runtime import BioGearsNotFoundError, import_pybiogears, runtime_dir  # noqa: E402

try:
    import_pybiogears()
    runtime_dir()
except BioGearsNotFoundError as e:  # no native build: skip, keep CI green
    pytest.skip(f'BioGears engine not available: {e}', allow_module_level=True)

from viva_simulation_core.composites.sepsis_engine import (  # noqa: E402
    extract_observables, make_sepsis_engine_composite)
from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES  # noqa: E402


def test_engine_baseline_parity():
    core = build_core()
    doc = make_sepsis_engine_composite(severity=0, horizon_s=300.0, interval_s=60.0, core=core)
    sim = Composite({'state': doc}, core=core)
    sim.run(300.0)
    obs = extract_observables(gather_emitter_results(sim)[('emitter',)])
    assert set(obs) == set(SEPSIS_OBSERVABLES)
    assert len(obs['SBP']) >= 5
    last = {k: v[-1] for k, v in obs.items()}
    print('ENGINE BASELINE', last)
    # StandardMale targets; tolerance = physiological closeness (+/-5 mmHg, 5 bpm, 0.5 L/min)
    assert abs(last['SBP'] - 114.0) < 5
    assert abs(last['DBP'] - 73.5) < 5
    assert abs(last['HR'] - 72.0) < 5
    assert abs(last['CO'] - 5.9) < 0.5
    assert last['blood_pathogen'] < 1e-6
    assert last['tissue_integrity'] > 0.99
