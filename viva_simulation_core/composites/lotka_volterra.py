"""Lotka-Volterra predator-prey composite (Tellurium / roadrunner).

A small, self-contained demo model for the model-interchange machinery. Unlike
the sepsis composites — whose observables are global parameters / assignment
rules (no floating species) — this one has two *floating species* (prey ``X``,
predator ``Y``), so it exercises the species-based paths: the generic viva-core
adapter (:class:`~viva_simulation_core.analysis.vc_adapter.VivaCoreTelluriumGeneric`)
and the exact-ODE engines' ``FullReportUTCStep`` on the engine-core contract.

The same portable SBML this composite serializes can be run under COPASI and
viva-core (Rust) with
:func:`viva_simulation_core.analysis.run_engines.run_all` — the "one spec, many
engines" demonstration on a model that is nothing like sepsis.
"""
from __future__ import annotations

from process_bigraph.composite_generator import composite_generator

from viva_simulation_core.core import build_core
from viva_simulation_core.engines import require as require_engine

# Classic Lotka-Volterra as mass-action reactions. NB: `gamma` is a reserved word
# in Antimony, so the rate constants are a (prey birth), b (predation), d
# (predator death); the composite exposes them under the standard alpha/beta/delta
# names and maps them via parameter_overrides.
LV_ANTIMONY = """model lotka_volterra
  J1: X -> 2 X;      a * X;         # prey birth
  J2: X + Y -> 2 Y;  b * X * Y;     # predation
  J3: Y -> ;         d * Y;         # predator death
  X = 10;  Y = 5;
  a = 1.1; b = 0.4; d = 0.4;
end"""


@composite_generator(
    name="lotka_volterra",
    description="Lotka-Volterra predator-prey (prey X, predator Y) as a portable SBML "
                "model, run via Tellurium / roadrunner. A small demo of the model-"
                "interchange machinery: the same SBML runs under COPASI and viva-core "
                "through the engine-core FullReportUTCStep contract (see "
                "viva_simulation_core.analysis.run_engines). Oscillates; nothing like "
                "the sepsis model, and has floating species.",
    parameters={
        "alpha": {"type": "float", "default": 1.1, "description": "prey birth rate"},
        "beta": {"type": "float", "default": 0.4, "description": "predation rate"},
        "delta": {"type": "float", "default": 0.4, "description": "predator death rate"},
        "interval_s": {"type": "float", "default": 0.1, "description": "time per update"},
    },
    core_extensions=[build_core],
    default_n_steps=300,
    fidelity="reproduction",
)
def make_lotka_volterra_composite(core=None, *, alpha: float = 1.1, beta: float = 0.4,
                                  delta: float = 0.4, interval_s: float = 0.1) -> dict:
    """Build the Tellurium composite document for Lotka-Volterra.

    ``core`` is accepted for the generator protocol but unused — Tellurium /
    roadrunner carries its own solver. ``alpha`` / ``beta`` / ``delta`` map to the
    model's ``a`` / ``b`` / ``d`` rate constants.
    """
    make_tellurium_document = require_engine('tellurium').make_tellurium_document
    overrides = {'a': float(alpha), 'b': float(beta), 'd': float(delta)}
    return make_tellurium_document(
        model=LV_ANTIMONY, model_format='antimony',
        parameter_overrides=overrides, interval=float(interval_s))
