"""Co-simulation composite (study 06): physiology engine <-> antibiotic controller.

Exercises the real closed loop on a short horizon: the controller writes the
drug-killed pathogen back into the Tellurium engine via parameters_in every tick,
so with treatment enabled the bacteremia ends well below the matched untreated
control run from the identical composite.
"""
from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from viva_simulation_core.core import build_core
from viva_simulation_core.composites.sepsis_cosim import (
    make_sepsis_cosim_composite, extract_cosim)


def _run(treat, horizon_s=300.0, interval_s=30.0):
    core = build_core()
    core.register_link("ram-emitter", RAMEmitter)
    doc = make_sepsis_cosim_composite(severity=3, treat=treat, interval_s=interval_s,
                                      horizon_s=horizon_s)
    sim = Composite({"state": doc}, core=core)
    for _ in range(int(horizon_s / interval_s)):
        sim.run(interval_s)
    return extract_cosim(gather_emitter_results(sim)[("emitter",)])


def test_controller_is_registered():
    core = build_core()
    assert "AntibioticControllerProcess" in core.link_registry


def test_closed_loop_clears_pathogen_vs_matched_control():
    untreated = _run(treat=False)
    treated = _run(treat=True)
    assert untreated and treated
    pb_u = untreated["blood_pathogen"][-1]
    pb_t = treated["blood_pathogen"][-1]
    # Untreated bacteremia rises past the trigger; the closed loop drives it far lower.
    assert pb_u > 1.0, f"untreated bacteremia should rise, got {pb_u}"
    assert pb_t < 0.5 * pb_u, f"closed loop should clear pathogen: treated {pb_t} vs untreated {pb_u}"
    # The controller actually dosed (its PK built up) in the treated arm only.
    assert treated["drug"][-1] > 0.0
    assert untreated["drug"][-1] == 0.0
