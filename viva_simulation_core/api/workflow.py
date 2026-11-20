"""Unified vivarium-workflow API router — the four verbs over process-bigraph.

Thin by design: every endpoint delegates to the process-bigraph ``Core`` (from
``build_core``) and ``Composite``. The router adds no simulation logic. It is
mounted on the existing digital-twin FastAPI app so ``/twins``, ``/modules`` and
``/workflows`` are one service under one auth scheme.

- define / discover a module → ``GET /modules`` + ``GET /modules/{name}``
- call a module              → ``POST /modules/{name}/call``   (Task 4)
- connect a workflow         → ``POST /workflows``             (Task 5)
- call a workflow            → ``POST /workflows/{id}/run``     (Task 5)
- the graph                  → ``GET /workflows/{id}/graph``    (Task 5)
"""
from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from process_bigraph import Process, Step

from .auth import require_key
from .workflow_models import ModuleCard


# ----------------------------------------------------------------------------
# Module introspection (generic — reads the live registry, no hardcoding)
# ----------------------------------------------------------------------------
def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def display_module_names(core: Any) -> list[str]:
    """Registered module names with the dotted fully-qualified twin of any short
    alias dropped, so each module shows once."""
    names = core.list_processes()
    shorts = {n for n in names if "." not in n}
    out = []
    for n in names:
        if "." in n and _short(n) in shorts:
            continue  # the fqcn of a name already shown by its short alias
        out.append(n)
    return out


def _best_effort_ports(core: Any, cls: Any) -> tuple[dict | None, dict | None]:
    """Try to instantiate the module with an empty config and read its port
    schemas. Many modules need a real config (a model, a checkpoint) to build, so
    this is best-effort: any failure yields ``None`` rather than propagating."""
    try:
        inst = cls({}, core)
        ins = inst.inputs() if callable(getattr(inst, "inputs", None)) else None
        outs = inst.outputs() if callable(getattr(inst, "outputs", None)) else None
        return (ins if isinstance(ins, dict) else None,
                outs if isinstance(outs, dict) else None)
    except Exception:  # noqa: BLE001 — port introspection is best-effort
        return (None, None)


def module_card(core: Any, name: str, *, with_ports: bool = False) -> ModuleCard:
    """Build one module's card. Never raises on a module's own introspection
    failure — an un-importable or un-instantiable module still yields a minimal
    card so ``GET /modules`` never 500s across the whole registry."""
    label = _short(name)
    address = f"local:{name}"
    kind = "unknown"
    config_schema: dict = {}
    input_ports = output_ports = None
    try:
        cls = core.link_registry.get(name)
    except Exception:  # noqa: BLE001
        cls = None
    if isinstance(cls, type):
        try:
            if issubclass(cls, Step):
                kind = "Step"
            elif issubclass(cls, Process):
                kind = "Process"
        except Exception:  # noqa: BLE001
            pass
        cs = getattr(cls, "config_schema", None)
        if isinstance(cs, dict):
            config_schema = cs
        if with_ports:
            input_ports, output_ports = _best_effort_ports(core, cls)
    return ModuleCard(name=name, address=address, kind=kind,
                      config_schema=config_schema, input_ports=input_ports,
                      output_ports=output_ports, capability_label=label)


def list_modules(core: Any) -> list[ModuleCard]:
    cards = []
    for name in display_module_names(core):
        try:
            cards.append(module_card(core, name))
        except Exception:  # noqa: BLE001 — one bad module never drops the list
            continue
    return cards


def _resolve_name(core: Any, name: str) -> str:
    """Map a requested module name to a registered one (short or fqcn). Raises
    KeyError if unknown."""
    known = set(core.list_processes())
    if name in known:
        return name
    # allow a short name that only exists as a dotted fqcn
    for n in known:
        if _short(n) == name:
            return n
    raise KeyError(name)


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
def make_workflow_router(core_provider: Callable[[], Any]) -> APIRouter:
    router = APIRouter(dependencies=[Depends(require_key)])

    @router.get("/modules")
    def get_modules() -> list[ModuleCard]:
        return list_modules(core_provider())

    @router.get("/modules/{name}")
    def get_module(name: str) -> ModuleCard:
        core = core_provider()
        try:
            resolved = _resolve_name(core, name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown module: {name}")
        return module_card(core, resolved, with_ports=True)

    return router
