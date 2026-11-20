"""Pydantic models for the unified vivarium-workflow API (modules + workflows).

These are the typed request/response surface for the four verbs — define/call a
module, connect/call a workflow — over the process-bigraph primitives. The router
(``workflow.py``) translates these to composite specs and back.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------------------------------------------------------
# Provenance (workflow-specific; the twin API's Provenance hardcodes
# engine_used/integrator, which do not apply to arbitrary module composition).
# ----------------------------------------------------------------------------
class WorkflowProvenance(BaseModel):
    modules: list[str] = Field(default_factory=list)
    n_steps: int = 0
    wall_ms: float = 0.0
    lib_versions: dict[str, str] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Modules (define / discover / call)
# ----------------------------------------------------------------------------
class ModuleCard(BaseModel):
    """One registered module's interface — the 'capability' card."""

    name: str
    address: str
    kind: str  # "Process" | "Step" | "unknown"
    config_schema: dict = Field(default_factory=dict)
    input_ports: dict | None = None
    output_ports: dict | None = None
    capability_label: str


class ModuleCallRequest(BaseModel):
    config: dict = Field(default_factory=dict)
    inputs: dict = Field(default_factory=dict)
    n_steps: int = 1


class ModuleCallResponse(BaseModel):
    outputs: dict = Field(default_factory=dict)
    provenance: WorkflowProvenance


# ----------------------------------------------------------------------------
# Workflows (connect / call / graph)
# ----------------------------------------------------------------------------
class WorkflowNode(BaseModel):
    id: str
    module: str
    config: dict = Field(default_factory=dict)
    interval: float | None = None


class WorkflowWire(BaseModel):
    """A wire from one node's output to another's input.

    ``from`` is a Python keyword, so the field is ``from_`` with a ``from`` JSON
    alias (``populate_by_name`` lets either spelling construct it).
    """

    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str


class WorkflowSpec(BaseModel):
    nodes: list[WorkflowNode] = Field(min_length=1)
    wires: list[WorkflowWire] = Field(default_factory=list)
    emit: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    module: str
    label: str


class GraphEdge(BaseModel):
    from_: str = Field(serialization_alias="from")
    to: str
    label: str | None = None


class GraphResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class WorkflowCreateResponse(BaseModel):
    workflow_id: str
    graph: GraphResponse


class WorkflowRunRequest(BaseModel):
    n_steps: int = 1
    overrides: dict = Field(default_factory=dict)


class WorkflowRunResponse(BaseModel):
    observables: dict = Field(default_factory=dict)
    provenance: WorkflowProvenance
