"""Sepsis SBML composite variant that also exposes the rate-rule state.

The sepsis model has 0 floating species; its Markov state is the 91 variables
defined by rate rules (``x' = f(x)``), which TelluriumProcess does not expose
(its ``species`` port is empty here). ``SepsisStateTelluriumProcess`` adds a
``rate_rules`` ``map[float]`` output keyed by rate-rule id, so each variable is
an individually addressable scalar path ``["rate_rules", "<id>"]``.

Ids come from ``rr.model.getRateRuleSymbols()`` (this roadrunner build has no
``getRateRuleValues`` / ``getStateVector``), snapshotted once at import so the
order is fixed. Values are read with ``rr[id]``.
"""
from __future__ import annotations

from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.processes import TelluriumProcess

from viva_simulation_core.composites.sepsis_sbml import SBML_PATH, make_sepsis_sbml_composite


def _snapshot_rate_rule_ids() -> list[str]:
    import roadrunner
    rr = roadrunner.RoadRunner(str(SBML_PATH))
    return [str(s) for s in rr.model.getRateRuleSymbols()]


RATE_RULE_IDS: list[str] = _snapshot_rate_rule_ids()


class SepsisStateTelluriumProcess(TelluriumProcess):
    """TelluriumProcess + a ``rate_rules`` map[float] output (91 scalars)."""

    def outputs(self):
        ports = super().outputs()
        ports['rate_rules'] = 'overwrite[map[float]]'
        return ports

    def _read_state(self):
        out = super()._read_state()
        out['rate_rules'] = {i: float(self._rr[i]) for i in RATE_RULE_IDS}
        return out


def build_sepsis_state_composite(severity: int = 0, infection_start_s: float | None = None,
                                 interval_s: float = 10.0):
    """Return ``(document, core)``: the sepsis_sbml doc plus a ``rate_rules`` store."""
    doc = make_sepsis_sbml_composite(severity=severity, infection_start_s=infection_start_s,
                                     interval_s=interval_s)
    doc['tellurium']['address'] = 'local:SepsisStateTelluriumProcess'
    doc['tellurium']['outputs']['rate_rules'] = ['stores', 'rate_rules']
    doc['emitter']['config']['emit']['rate_rules'] = 'map[float]'
    doc['emitter']['inputs']['rate_rules'] = ['stores', 'rate_rules']
    # seed t=0 so the first emitter row carries the initial rate-rule values
    seed = SepsisStateTelluriumProcess(config=doc['tellurium']['config'], core=allocate_core())
    doc.setdefault('stores', {})['rate_rules'] = seed.initial_state()['rate_rules']
    core = allocate_core()
    core.register_link('SepsisStateTelluriumProcess', SepsisStateTelluriumProcess)
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    return doc, core
