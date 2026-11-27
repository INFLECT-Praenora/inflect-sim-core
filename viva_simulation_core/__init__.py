"""viva_simulation_core — vivarium as a simulation core.

**The simulator contract is the front door.** Hand a model in, get results and a
structured verification out, with the engine swappable behind one interface:

    from viva_simulation_core import simulate, SimulationRequest

    out = simulate(model, SimulationRequest(duration=7200, interval=60,
                                            observables=["MAP", "PB"],
                                            engine="tellurium"))
    out.results        # {"time": [...], "MAP": [...], "PB": [...]}
    out.verification   # ran / finite / admissible / (optional) cross-engine agreement
    out.verification.passed

That is the pure-simulator identity: ``simulate(model, request) -> SimulationResult``
(see :mod:`viva_simulation_core.contract`), the engine wrappers it dispatches to
(``engine=`` tellurium / copasi / julia / rust / surrogate — all lazy and optional),
and the composite registry (``sepsis_sbml``, ``sepsis_sbml_eventfree_valve``,
``sepsis_cosim``, plus every curated ``inflect-models-db`` model by slug). Importing
the package pulls in ``.composites`` so the ``@composite_generator`` decorators fire
and those composites register in process-bigraph's generator registry; internal
imports stay lazy, so ``import viva_simulation_core`` pulls in no engine.

**Orchestration is not simulator identity.** The four-verb + serve facade
(``catalog`` / ``call`` / ``connect`` / ``run`` / ``serve``) that composes modules
into workflows now lives in :mod:`viva_simulation_core.orchestration` — a workspace /
driver concern (e.g. the discovery-engine), not the simulator. It is still re-exported
here for backward compatibility, but it is deliberately demoted: use ``simulate()`` to
run a model.
"""
from __future__ import annotations

from typing import Any

from viva_simulation_core import composites  # noqa: F401 — register generators on import

# --- The simulator contract: model in, results + verification out ----------------

def simulate(model: Any, request: Any = None, **kwargs) -> Any:
    """The simulation-core contract: model in, results + verification out.

    Hand it a model (SBML/Antimony string or file, or a registered composite id),
    say what to run, and get back a ``SimulationResult`` carrying both the results and
    a structured verification (ran / finite / admissible / optional cross-engine
    agreement). See :mod:`viva_simulation_core.contract`.
    """
    from .contract import simulate as _simulate
    return _simulate(model, request, **kwargs)


# --- Legacy orchestration facade (see viva_simulation_core.orchestration) ---------
# Re-exported for backward compatibility only; NOT the simulator identity. These
# compose modules into workflows / serve them — a workspace/driver concern. Prefer
# `simulate()` above. Implementations (and the rationale) live in orchestration.py.
from viva_simulation_core.orchestration import (  # noqa: E402
    catalog, call, connect, run, serve,
)

__all__ = [
    # simulator identity
    "simulate", "SimulationRequest", "SimulationResult", "VerificationReport",
    "composites", "models_db",
    # legacy orchestration facade (demoted; see orchestration.py)
    "catalog", "call", "connect", "run", "serve",
]


def __getattr__(name: str) -> Any:  # lazy access to the contract dataclasses + models_db
    if name in ("SimulationRequest", "SimulationResult", "VerificationReport"):
        from . import contract
        return getattr(contract, name)
    if name == "models_db":
        from . import models_db
        return models_db
    raise AttributeError(name)
