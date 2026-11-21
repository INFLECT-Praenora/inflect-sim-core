"""Function-level coverage for the four-verb workflow API (``workflow.py``).

Ported from the deleted HTTP-transport tests: the twin service exposed these
functions over FastAPI, but the enduring surface is the functions themselves
(what the top-level facade and the ``viva-sim`` CLI wrap). HTTP-only concerns
(auth, rate-limit, thread-pool concurrency) went with the service.
"""
import pytest

from viva_simulation_core.core import build_core
from viva_simulation_core.api import workflow as wf
from viva_simulation_core.api.workflow_examples import tiered_workflow_spec
from viva_simulation_core.api.workflow_models import WorkflowSpec, WorkflowNode, WorkflowWire


@pytest.fixture(scope="module")
def core():
    return build_core()


# --- call: run one module ---------------------------------------------------

def test_call_math_expression_module(core):
    out = wf.call_module(core, "MathExpressionStep",
                         config={"expressions": [{"out": "y", "expr": "2*x + 3"}]},
                         inputs={"x": 5.0}, n_steps=1)
    assert out["y"] == 13.0


def test_call_unknown_module_raises(core):
    with pytest.raises(KeyError):
        wf.call_module(core, "NoSuchModule", config={}, inputs={}, n_steps=1)


# --- list / card: introspection ---------------------------------------------

def test_list_modules_includes_real_viva_modules(core):
    names = {c.name for c in wf.list_modules(core)}
    for expected in ("TelluriumUTCStep", "MathExpressionStep", "BioGearsProcess", "NeuralProcess"):
        assert expected in names, f"{expected} missing from {sorted(names)[:20]}"


def test_list_modules_dedups_short_name(core):
    names = [c.name for c in wf.list_modules(core)]
    assert names.count("TelluriumUTCStep") == 1
    assert not any(n.endswith(".TelluriumUTCStep") for n in names)


def test_module_card_has_config_schema(core):
    card = wf.module_card(core, "MathExpressionStep")
    assert card.name == "MathExpressionStep"
    assert "expressions" in card.config_schema


def test_resolve_unknown_module_raises(core):
    # unknown-module detection is _resolve_name (what call/card callers go through)
    with pytest.raises(KeyError):
        wf._resolve_name(core, "NoSuchModule")


# --- connect + run: a two-node chain ----------------------------------------

CHAIN = WorkflowSpec(
    nodes=[
        WorkflowNode(id="inc", module="MathExpressionStep",
                     config={"expressions": [{"out": "a", "expr": "x + 1"}]}),
        WorkflowNode(id="scale", module="MathExpressionStep",
                     config={"expressions": [{"out": "b", "expr": "a * 10"}]}),
    ],
    wires=[WorkflowWire(**{"from": "inc.a", "to": "scale.a"})],
    emit=["scale.b", "inc.a"],
)


def test_registry_define_and_run_chain(core):
    reg = wf.WorkflowRegistry(core_provider=lambda: core)
    wid, graph = reg.define(CHAIN)
    assert {n.id for n in graph.nodes} == {"inc", "scale"}
    assert len(graph.edges) == 1 and graph.edges[0].from_ == "inc.a" and graph.edges[0].to == "scale.a"
    resp = reg.run(wid, n_steps=1, overrides={"inc.x": 5.0})
    assert resp.observables["inc.a"] == 6.0
    assert resp.observables["scale.b"] == 60.0


def test_registry_define_unknown_module_raises(core):
    reg = wf.WorkflowRegistry(core_provider=lambda: core)
    bad = WorkflowSpec(nodes=[WorkflowNode(id="n", module="NoSuchModule")], wires=[], emit=[])
    with pytest.raises((KeyError, ValueError)):
        reg.define(bad)


def test_registry_delete_evicts(core):
    reg = wf.WorkflowRegistry(core_provider=lambda: core)
    wid, _ = reg.define(CHAIN)
    reg.delete(wid)
    with pytest.raises(KeyError):
        reg.run(wid, n_steps=1, overrides={})


# --- the tiered worked workflow: four real modules compose ------------------

def test_tiered_spec_composes_four_real_modules():
    spec = tiered_workflow_spec(checkpoint="unused.pt", timestep=1.0,
                                feature_keys=["S1", "S2", "S3"], include_biogears=True)
    graph = wf.workflow_graph(spec)
    assert {n.id for n in graph.nodes} == {"exact", "surrogate", "compare", "physio"}
    assert {n.module for n in graph.nodes} == {
        "TelluriumUTCStep", "NeuralProcess", "MathExpressionStep", "BioGearsProcess"}
