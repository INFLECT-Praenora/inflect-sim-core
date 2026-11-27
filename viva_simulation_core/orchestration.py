"""Orchestration facade — the workspace/discovery-engine side, NOT simulator identity.

The **pure-simulator** surface of this package is the contract in
:mod:`viva_simulation_core.contract`: ``simulate(model, request) -> results +
verification``, engine-swappable via ``engine=``, plus the composite registry. That
is the stable "model in, results + verification out" front door.

This module is the **orchestration** layer that sits *on top* of the simulator: a
thin four-verb + serve facade (``catalog`` / ``call`` / ``connect`` / ``run`` /
``serve``) over :mod:`viva_simulation_core.api.workflow` and
:mod:`viva_simulation_core.serving`. It composes modules into workflows and serves
them — concerns that belong to a workspace or a driver (e.g. the discovery-engine),
not to the simulator's identity. It is kept here (and re-exported from the package
root for backward compatibility) but is deliberately **demoted**: reach for
``simulate()`` when you want to run a model; reach for these only when you are
orchestrating modules into a pipeline.

Internal imports stay lazy so ``import viva_simulation_core`` pulls in no engine.
"""
from __future__ import annotations

from typing import Any


def _core(core: Any = None) -> Any:
    from .core import build_core
    return core if core is not None else build_core()


def catalog(core: Any = None) -> list:
    """List every registered module as a card (name, kind, address)."""
    from .api.workflow import list_modules
    return list_modules(_core(core))


def call(name: str, config: dict | None = None, inputs: dict | None = None,
         steps: int = 10, core: Any = None) -> dict:
    """Run one module as a 1-node composite; return its output ports."""
    from .api.workflow import call_module
    return call_module(_core(core), name, config or {}, inputs or {}, steps)


def connect(spec: Any) -> Any:
    """Normalize a workflow spec (dict or WorkflowSpec) into a WorkflowSpec."""
    from .api.workflow_models import WorkflowSpec
    return spec if isinstance(spec, WorkflowSpec) else WorkflowSpec.model_validate(spec)


def run(spec: Any, steps: int = 10, overrides: dict | None = None, core: Any = None) -> dict:
    """Connect + run a workflow; return the emitted observables as ``{"node.port": value}``."""
    from .api.workflow import WorkflowRegistry
    c = _core(core)
    reg = WorkflowRegistry(lambda: c)
    wid, _graph = reg.define(connect(spec))
    return reg.run(wid, steps, overrides or {}).observables


def serve(source: str, model_format: str = "antimony") -> Any:
    """Load a model and return a stepping handle (``ServedModel``)."""
    from .serving import ServedModel
    return ServedModel(source, model_format)


__all__ = ["catalog", "call", "connect", "run", "serve"]
