"""The sepsis SBML model's 91 rate-rule variables are exposed as PB state paths."""
import numpy as np
from process_bigraph import Composite, gather_emitter_results

from viva_simulation_core.composites.sepsis_sbml_state import (
    build_sepsis_state_composite, RATE_RULE_IDS)


def _rows(doc, core, t):
    sim = Composite({'state': doc}, core=core)
    sim.run(t)
    return sim, gather_emitter_results(sim)[('emitter',)]


def test_rate_rule_state_is_exposed_and_ordered():
    assert len(RATE_RULE_IDS) == 91 and len(set(RATE_RULE_IDS)) == 91
    doc, core = build_sepsis_state_composite(severity=3, interval_s=5.0)
    _, rows = _rows(doc, core, 10.0)
    rr_store = rows[-1]['rate_rules']
    assert set(rr_store.keys()) == set(RATE_RULE_IDS) and len(rr_store) == 91
    vals = np.array([rr_store[i] for i in RATE_RULE_IDS], dtype=float)
    assert np.all(np.isfinite(vals))
    # state actually evolves under severe infection: some variable differs first vs last row
    first = rows[0]['rate_rules']
    assert max(abs(rr_store[i] - first[i]) for i in RATE_RULE_IDS) > 1e-9


def test_rate_rules_match_roadrunner_t0_and_after_step():
    doc, core = build_sepsis_state_composite(severity=0, interval_s=5.0)
    sim, rows = _rows(doc, core, 5.0)
    t0 = rows[0]['rate_rules']
    assert set(t0) == set(RATE_RULE_IDS)
    # live roadrunner handle of the process instance
    proc = sim.state['tellurium']['instance']
    rr = proc._rr
    live = {i: float(rr[i]) for i in RATE_RULE_IDS}
    last = rows[-1]['rate_rules']
    assert len(rows) >= 2
    for i in RATE_RULE_IDS:
        assert np.isclose(last[i], live[i], rtol=1e-12, atol=1e-12)
    assert all(np.isfinite(v) for v in t0.values())
    # ids come from roadrunner's own ordering
    assert list(rr.model.getRateRuleSymbols()) == RATE_RULE_IDS


def test_rate_rules_match_independent_roadrunner():
    """Fresh RoadRunner from the same SBML, same overrides/tolerances/steps, severity 3."""
    import roadrunner
    from viva_simulation_core.composites.sepsis_sbml import SBML_PATH

    interval = 5.0
    doc, core = build_sepsis_state_composite(severity=3, interval_s=interval)
    _, rows = _rows(doc, core, 10.0)

    rr = roadrunner.RoadRunner(str(SBML_PATH))
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-10
    rr.integrator.relative_tolerance = 1e-8
    rr['severity'] = 3.0
    rr['infectionStart'] = 1.0  # builder default for severity > 0
    t = 0.0
    for _ in range(2):  # 10 s run / 5 s interval
        rr.simulate(t, t + interval, 2)
        t += interval
    got = rows[-1]['rate_rules']
    # Same solver, tolerances and step grid, so near-identical; rtol 1e-6 leaves
    # headroom for CVODE adaptive-step differences (reset/init order).
    for i in RATE_RULE_IDS:
        assert np.isclose(got[i], float(rr[i]), rtol=1e-6, atol=1e-9), i
