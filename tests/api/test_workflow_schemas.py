"""Task 2: workflow API pydantic models."""
import pytest

pydantic = pytest.importorskip("pydantic")
from pydantic import ValidationError  # noqa: E402

from viva_simulation_core.api.workflow_models import (  # noqa: E402
    WorkflowNode, WorkflowSpec, WorkflowWire)


def test_workflow_spec_round_trips():
    spec = WorkflowSpec(
        nodes=[WorkflowNode(id="sim", module="TelluriumUTCStep"),
               WorkflowNode(id="derive", module="MathExpressionStep")],
        wires=[WorkflowWire(**{"from": "sim.S1", "to": "derive.x"})],
        emit=["derive.y"],
    )
    dumped = spec.model_dump(by_alias=True)
    assert dumped["nodes"][0]["module"] == "TelluriumUTCStep"
    assert dumped["wires"][0]["from"] == "sim.S1"
    # re-parse the dumped form
    again = WorkflowSpec.model_validate(dumped)
    assert again.nodes[1].id == "derive"


def test_wire_accepts_from_json_key():
    w = WorkflowWire(**{"from": "a.out", "to": "b.in"})
    assert w.from_ == "a.out"
    assert w.to == "b.in"


def test_workflow_spec_rejects_empty_nodes():
    with pytest.raises(ValidationError):
        WorkflowSpec(nodes=[], wires=[], emit=[])
