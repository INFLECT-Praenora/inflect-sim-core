"""Event-free (proto_valve) sepsis SBML composite (Tellurium).

The proto_valve model is the full BioGears sepsis SBML with the per-beat E_beat
event and the piecewise pulOpen valve removed (continuous phase oscillator +
per-artery valve gating). It must load, step, emit the same 9 shared observables,
hold the StandardMale baseline at rest, and infect under severity > 0 — exactly
like the full sepsis_sbml composite.
"""
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.processes import TelluriumProcess

from viva_simulation_core.composites.sepsis_sbml import (
    SEPSIS_OBSERVABLES, extract_observables)
from viva_simulation_core.composites.sepsis_sbml_eventfree_valve import (
    make_sepsis_sbml_eventfree_valve_composite as make_composite)


def _run(doc, t):
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    sim = Composite({'state': doc}, core=core)
    sim.run(t)
    return extract_observables(gather_emitter_results(sim)[('emitter',)])


def test_eventfree_valve_loads_and_steps():
    doc = make_composite(severity=0, horizon_s=30.0, interval_s=5.0)
    obs = _run(doc, 30.0)
    assert obs and all(k in obs for k in SEPSIS_OBSERVABLES)
    last = {k: v[-1] for k, v in obs.items()}
    # StandardMale baseline: SBP := MAP+19.1, DBP := MAP-22 are now assignment
    # rules, so the pulse pressure is a fixed offset but the band still holds.
    assert abs(last['SBP'] - 114.0) < 5
    assert abs(last['DBP'] - 73.5) < 5
    assert abs(last['HR'] - 72.0) < 5
    assert abs(last['CO'] - 5.9) < 0.5
    # healthy run must be genuinely uninfected (guards the severity-0 seeding quirk)
    assert last['blood_pathogen'] < 1e-6
    assert last['tissue_integrity'] > 0.99


def test_eventfree_valve_severity1_infects():
    doc = make_composite(severity=1, infection_start_s=5.0,
                         horizon_s=120.0, interval_s=5.0)
    obs = _run(doc, 120.0)
    assert obs
    assert max(obs['blood_pathogen']) > 1e-3
