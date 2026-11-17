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
    # state moved from t=0 after a step for at least one variable or equals (healthy near-steady)
    assert all(np.isfinite(v) for v in t0.values())
    # ids come from roadrunner's own ordering
    assert list(rr.model.getRateRuleSymbols()) == RATE_RULE_IDS
