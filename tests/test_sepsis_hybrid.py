"""The hybrid sepsis composite: event-free SBML core + viva-impulse discrete events.

Reproduction test — lifting the infection (E_infect) and TLR hysteresis (E_tlrOn/Off)
out of the SBML into general viva-impulse Steps (ScheduledEvent + ThresholdSwitch),
coupled to the engine via TelluriumImpulseBridge, must reproduce the full event-driven
model. Gated on viva-impulse being importable (the optional `impulse` engine), exactly
like the other optional-engine tests.
"""
import pytest

pytest.importorskip("viva_impulse")
pytest.importorskip("viva_tellurium")

from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from viva_simulation_core.core import build_core
from viva_simulation_core.engines import require as require_engine
from viva_simulation_core.composites.sepsis_hybrid import make_sepsis_hybrid_composite
from viva_simulation_core.composites.sepsis_sbml import extract_observables

# Reference = the full event-driven proto_valve model, severity 3, 30 min, 10 s interval
# (committed study-04 comparison finals). The hybrid must reproduce these.
_REF = {"MAP": 95.691, "SBP": 114.79, "HR": 71.458, "SVR": 0.89057,
        "tissue_integrity": 1.0, "blood_pathogen": 10.763}


def _hybrid_core():
    core = build_core(allocate_core())
    core = require_engine("impulse").build_core(core)   # == the composite's core_extension
    core.register_link("ram-emitter", RAMEmitter)
    return core


def _run(doc, t):
    sim = Composite({"state": doc}, core=_hybrid_core())
    sim.run(t)
    return extract_observables(gather_emitter_results(sim)[("emitter",)])


def test_hybrid_reproduces_full_model_severe():
    obs = _run(make_sepsis_hybrid_composite(severity=3, infection_start_s=1.0, interval_s=10.0),
               1800.0)
    assert obs
    last = {k: v[-1] for k, v in obs.items()}
    # hemodynamics reproduce to ~1e-5; blood pathogen to ~1% (tick-quantized infection timing)
    for k in ("MAP", "SBP", "HR", "SVR", "tissue_integrity"):
        assert abs(last[k] - _REF[k]) / (abs(_REF[k]) or 1) < 1e-4, (k, last[k], _REF[k])
    assert abs(last["blood_pathogen"] - _REF["blood_pathogen"]) / _REF["blood_pathogen"] < 0.02


def test_hybrid_healthy_at_severity0():
    obs = _run(make_sepsis_hybrid_composite(severity=0, interval_s=10.0), 300.0)
    assert obs
    assert max(obs["blood_pathogen"]) < 1e-6          # no infection event fires
    assert obs["tissue_integrity"][-1] > 0.99


def test_hybrid_uses_viva_impulse_events():
    """The discrete layer must be the general viva-impulse Steps, not bespoke logic."""
    doc = make_sepsis_hybrid_composite(severity=3)
    assert doc["infection"]["address"] == "local:ScheduledEvent"
    assert doc["tlr_switch"]["address"] == "local:ThresholdSwitch"
    assert doc["bridge"]["address"] == "local:TelluriumImpulseBridge"
