"""Sepsis SBML composite + the rate-rule state vector (config-driven).

The sepsis model has 0 floating species; its Markov state is the 91 variables
defined by rate rules (``x' = f(x)``). The **base** ``TelluriumProcess`` exposes
these as a ``rate_rules`` vector output (ordered by ``getRateRuleIds``) when
configured with ``rate_rules_port: True`` — no subclass needed (this replaced the
old ``SepsisStateTelluriumProcess``). This helper builds the sepsis_sbml doc with
that port wired to a store + emitter and returns ``(document, core)`` ready to run.
"""
from __future__ import annotations

from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from viva_simulation_core.composites.sepsis_sbml import make_sepsis_sbml_composite
from viva_simulation_core.engines import require as require_engine


def build_sepsis_state_composite(severity: int = 0, infection_start_s: float | None = None,
                                 interval_s: float = 10.0):
    """Return ``(document, core)``: the sepsis_sbml doc with the base engine's
    ``rate_rules`` vector port enabled and wired to a store + emitter (no
    subclass — config-driven), plus a core with the engine + emitter registered.
    """
    TelluriumProcess = require_engine('tellurium').TelluriumProcess
    doc = make_sepsis_sbml_composite(severity=severity, infection_start_s=infection_start_s,
                                     interval_s=interval_s)
    cfg = doc['tellurium']['config']
    cfg['rate_rules_port'] = True
    doc['tellurium']['outputs']['rate_rules'] = ['stores', 'rate_rules']
    doc['emitter']['config']['emit']['rate_rules'] = 'array'
    doc['emitter']['inputs']['rate_rules'] = ['stores', 'rate_rules']
    # Seed t=0 so the first emitter row carries the initial rate-rule vector.
    seed = TelluriumProcess(config=cfg, core=allocate_core())
    doc.setdefault('stores', {})['rate_rules'] = seed.initial_state()['rate_rules']
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    return doc, core
