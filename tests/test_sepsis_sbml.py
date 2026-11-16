"""SBML-side sepsis baseline composite (Tellurium)."""
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.processes import TelluriumProcess

from viva_simulation_core.composites.sepsis_sbml import (
    SEPSIS_OBSERVABLES, make_sepsis_sbml_composite, extract_observables)


def _run(doc, t):
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    sim = Composite({'state': doc}, core=core)
    sim.run(t)
    return extract_observables(gather_emitter_results(sim)[('emitter',)])


def test_sbml_loads_and_steps():
    doc = make_sepsis_sbml_composite(severity=0, horizon_s=30.0, interval_s=5.0)
    obs = _run(doc, 30.0)
    assert obs and all(k in obs for k in SEPSIS_OBSERVABLES)
    last = {k: v[-1] for k, v in obs.items()}
    assert abs(last['SBP'] - 114.0) < 5
    assert abs(last['DBP'] - 73.5) < 5
    assert abs(last['HR'] - 72.0) < 5
    assert abs(last['CO'] - 5.9) < 0.5
