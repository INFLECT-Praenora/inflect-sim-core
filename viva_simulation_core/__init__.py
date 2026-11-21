"""viva_simulation_core — vivarium as a simulation core.

Importing the package pulls in ``.composites`` so the ``@composite_generator``
decorators fire and the workspace's composites register in process-bigraph's
generator registry (the dashboard manifest reads that registry). Without this,
an editable install's empty ``top_level.txt`` hides the package from the
dist-scan and the composites never resolve.

The top-level **four-verb + serve facade** is the unified API: register modules,
adapters, and API-calling steps into one core, then drive them from one surface,
many ways —

    from viva_simulation_core import catalog, call, connect, run, serve

Each function is a thin wrapper over ``api/workflow.py`` + ``serve.py``; internal
imports stay lazy so ``import viva_simulation_core`` pulls in no engine.
"""
from __future__ import annotations

from typing import Any

from viva_simulation_core import composites  # noqa: F401 — register generators on import

__all__ = ["composites", "catalog", "call", "connect", "run", "serve"]


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
