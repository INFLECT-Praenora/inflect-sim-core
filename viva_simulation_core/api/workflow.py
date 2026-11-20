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

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from process_bigraph import Composite, Process, Step

from .auth import require_key
from .workflow_models import (GraphEdge, GraphNode, GraphResponse, ModuleCallRequest,
                              ModuleCallResponse, ModuleCard, WorkflowCreateResponse,
                              WorkflowProvenance, WorkflowRunRequest, WorkflowRunResponse,
                              WorkflowSpec)

# Fixed node key for a single-module call. Must not start with "_" (process-bigraph
# treats leading-underscore top-level keys as reserved schema and never runs them)
# and must not collide with a port store name.
_CALL_NODE = "callnode"

# One shared lock serializes ALL process-bigraph engine access — module
# instantiation (building a roadrunner/BioGears engine) and every Composite.run.
# roadrunner/BioGears are not thread-safe and FastAPI runs sync endpoints in a
# threadpool, so every path that builds or runs a composite holds this.
_ENGINE_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Module introspection (generic — reads the live registry, no hardcoding)
# ----------------------------------------------------------------------------
def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _lib_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out: dict[str, str] = {}
    for pkg in ("process-bigraph", "bigraph-schema"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


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
        with _ENGINE_LOCK:  # instantiation may build a (non-thread-safe) engine
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


def _node_type(cls: Any) -> str:
    return "step" if isinstance(cls, type) and issubclass(cls, Step) else "process"


def _run_composite(comp: "Composite", n_steps: int, interval: float) -> None:
    """Advance a composite n_steps ticks of duration `interval` each. Looping at a
    fixed interval (rather than one run(n_steps) call) is required so time-based
    processes — and a NeuralProcess, which refuses any interval != its trained
    timestep — advance at the intended step size."""
    steps = n_steps if n_steps and n_steps > 0 else 1
    dt = interval if interval and interval > 0 else 1.0
    for _ in range(steps):
        comp.run(dt)


def call_module(core: Any, name: str, config: dict, inputs: dict, n_steps: int,
                interval: float = 1.0) -> dict:
    """Call one module as a 1-node composite: wire its ports to stores, run, read
    its outputs. Raises KeyError (unknown module) / ValueError (bad config or run)
    — the endpoint maps both to 422."""
    try:
        resolved = _resolve_name(core, name)
    except KeyError:
        raise
    cls = core.link_registry.get(resolved)
    if not isinstance(cls, type):
        raise ValueError(f"module {name} did not resolve to a class")
    with _ENGINE_LOCK:  # instantiation builds the engine; serialize with every other run
        inst = cls(config, core)  # raises on bad config -> 422
        in_ports = inst.inputs() or {}
        out_ports = inst.outputs() or {}
        state: dict[str, Any] = {}
        for port in in_ports:  # seed every input store; caller value overrides the 0.0 default
            state[port] = inputs.get(port, 0.0)
        state[_CALL_NODE] = {
            "_type": _node_type(cls),
            "address": f"local:{resolved}",
            "config": config,
            "inputs": {port: [port] for port in in_ports},
            "outputs": {port: [port] for port in out_ports},
        }
        comp = Composite({"state": state}, core=core)
        _run_composite(comp, n_steps, interval)
        return {port: comp.state.get(port) for port in out_ports}


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
# Workflows: spec -> composite, graph, in-process registry
# ----------------------------------------------------------------------------
def _split_ref(ref: str) -> tuple[str, str]:
    node, _, port = ref.partition(".")
    return node, port


@dataclass
class _Built:
    state: dict
    ref_path: dict  # "node.port" -> store path (inputs and outputs)
    out_refs: list  # "node.port" for every output port (default emit set)


def spec_to_composite_state(spec: WorkflowSpec, core: Any) -> _Built:
    """Translate a WorkflowSpec to a process-bigraph composite state dict, plus the
    ``"node.port" -> store_path`` map (for seeding overrides and reading emits) and
    the list of output refs (default emit set).
    Raises ValueError on an unknown module or a bad node id -> the endpoint 422s.
    """
    ids = [n.id for n in spec.nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node id")
    for nid in ids:
        if nid.startswith("_"):
            raise ValueError(f"node id may not start with '_': {nid}")

    # A wire makes a producer output port and a consumer input port share a store.
    # Fan-out (one producer output -> many consumer inputs) reuses the producer's
    # store for every consumer, so each consumer reads the produced value.
    port_store: dict[tuple[str, str], str] = {}
    for i, w in enumerate(spec.wires):
        pn, pp = _split_ref(w.from_)
        cn, cp = _split_ref(w.to)
        path = port_store.get((pn, pp))
        if path is None:
            path = f"wire{i}_{pn}_{pp}"
            port_store[(pn, pp)] = path
        port_store[(cn, cp)] = path

    state: dict[str, Any] = {}
    ref_path: dict[str, str] = {}
    out_refs: list[str] = []
    for node in spec.nodes:
        try:
            resolved = _resolve_name(core, node.module)
        except KeyError:
            raise ValueError(f"unknown module: {node.module}")
        cls = core.link_registry.get(resolved)
        if not isinstance(cls, type):
            raise ValueError(f"module {node.module} did not resolve to a class")
        inst = cls(node.config, core)  # raises on bad config -> ValueError-ish -> 422
        in_ports = inst.inputs() or {}
        out_ports = inst.outputs() or {}
        inputs_map, outputs_map = {}, {}
        for p in in_ports:
            path = port_store.get((node.id, p), f"{node.id}__{p}")
            inputs_map[p] = [path]
            ref_path[f"{node.id}.{p}"] = path
            state.setdefault(path, 0.0)  # seed input store (overridden by a wire/override)
        for p in out_ports:
            path = port_store.get((node.id, p), f"{node.id}__{p}")
            outputs_map[p] = [path]
            ref_path[f"{node.id}.{p}"] = path
            out_refs.append(f"{node.id}.{p}")
        node_dict: dict[str, Any] = {
            "_type": _node_type(cls),
            "address": f"local:{resolved}",
            "config": node.config,
            "inputs": inputs_map,
            "outputs": outputs_map,
        }
        if node.interval is not None:
            node_dict["interval"] = node.interval
        state[node.id] = node_dict
    return _Built(state=state, ref_path=ref_path, out_refs=out_refs)


def workflow_graph(spec: WorkflowSpec) -> GraphResponse:
    nodes = [GraphNode(id=n.id, module=_short(n.module), label=_short(n.module))
             for n in spec.nodes]
    edges = [GraphEdge(from_=w.from_, to=w.to, label=None) for w in spec.wires]
    return GraphResponse(nodes=nodes, edges=edges)


@dataclass
class WorkflowSession:
    spec: WorkflowSpec
    created: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class WorkflowRegistry:
    """In-process, ephemeral workflow store (design scope). A module-level lock
    serializes composite runs — process-bigraph engines (roadrunner) are not
    thread-safe and FastAPI sync endpoints run in a threadpool."""

    def __init__(self, core_provider: Callable[[], Any], max_workflows: int = 64):
        self._core_provider = core_provider
        self._sessions: dict[str, WorkflowSession] = {}
        self._lock = threading.Lock()  # guards the sessions dict only
        self._max = max_workflows

    def define(self, spec: WorkflowSpec) -> tuple[str, GraphResponse]:
        core = self._core_provider()
        with _ENGINE_LOCK:  # validation instantiates modules (builds engines)
            spec_to_composite_state(spec, core)  # raises ValueError -> 422
        wid = uuid.uuid4().hex[:12]
        with self._lock:
            if len(self._sessions) >= self._max:  # evict LRU
                oldest = min(self._sessions, key=lambda k: self._sessions[k].last_access)
                self._sessions.pop(oldest, None)
            self._sessions[wid] = WorkflowSession(spec=spec)
        return wid, workflow_graph(spec)

    def graph(self, wid: str) -> GraphResponse:
        return workflow_graph(self._get(wid).spec)

    def delete(self, wid: str) -> None:
        with self._lock:
            if self._sessions.pop(wid, None) is None:
                raise KeyError(wid)

    def _get(self, wid: str) -> WorkflowSession:
        with self._lock:
            sess = self._sessions.get(wid)
            if sess is None:
                raise KeyError(wid)
            sess.last_access = time.time()
            return sess

    def run(self, wid: str, n_steps: int, overrides: dict,
            interval: float = 1.0) -> WorkflowRunResponse:
        sess = self._get(wid)
        core = self._core_provider()
        spec = sess.spec
        t0 = time.perf_counter()
        with _ENGINE_LOCK:  # instantiation + run serialized across the threadpool
            built = spec_to_composite_state(spec, core)
            state, ref_path = built.state, built.ref_path
            for ref, val in (overrides or {}).items():  # override input stores by "node.port"
                n, p = _split_ref(ref)
                state[ref_path.get(ref, f"{n}__{p}")] = val
            comp = Composite({"state": state}, core=core)
            _run_composite(comp, n_steps, interval)
            result_state = comp.state
        emits = spec.emit or built.out_refs  # default emit: every output port
        observables = {}
        for ref in emits:
            n, p = _split_ref(ref)
            observables[ref] = result_state.get(ref_path.get(ref, f"{n}__{p}"))
        modules = sorted({_short(n.module) for n in spec.nodes})
        prov = WorkflowProvenance(modules=modules, n_steps=n_steps,
                                  wall_ms=(time.perf_counter() - t0) * 1000.0,
                                  lib_versions=_lib_versions())
        return WorkflowRunResponse(observables=observables, provenance=prov)


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

    @router.post("/modules/{name}/call")
    def post_module_call(name: str, req: ModuleCallRequest) -> ModuleCallResponse:
        core = core_provider()
        t0 = time.perf_counter()
        try:
            outputs = call_module(core, name, req.config, req.inputs, req.n_steps, req.interval)
        except KeyError:
            raise HTTPException(status_code=422, detail=f"unknown module: {name}")
        except Exception as e:  # noqa: BLE001 — bad config/inputs/run → 422
            raise HTTPException(status_code=422, detail=str(e))
        prov = WorkflowProvenance(modules=[_short(name)], n_steps=req.n_steps,
                                  wall_ms=(time.perf_counter() - t0) * 1000.0,
                                  lib_versions=_lib_versions())
        return ModuleCallResponse(outputs=outputs, provenance=prov)

    registry = WorkflowRegistry(core_provider)

    @router.post("/workflows")
    def post_workflow(spec: WorkflowSpec) -> WorkflowCreateResponse:
        try:
            wid, graph = registry.define(spec)
        except Exception as e:  # noqa: BLE001 — unknown module / bad wiring / config → 422
            raise HTTPException(status_code=422, detail=str(e))
        return WorkflowCreateResponse(workflow_id=wid, graph=graph)

    @router.post("/workflows/{wid}/run")
    def post_workflow_run(wid: str, req: WorkflowRunRequest) -> WorkflowRunResponse:
        try:
            return registry.run(wid, req.n_steps, req.overrides, req.interval)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown workflow: {wid}")
        except Exception as e:  # noqa: BLE001 — a bad run → 422
            raise HTTPException(status_code=422, detail=str(e))

    @router.get("/workflows/{wid}/graph")
    def get_workflow_graph(wid: str) -> GraphResponse:
        try:
            return registry.graph(wid)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown workflow: {wid}")

    @router.delete("/workflows/{wid}")
    def delete_workflow(wid: str) -> dict:
        try:
            registry.delete(wid)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown workflow: {wid}")
        return {"deleted": wid}

    return router
