"""The 'tiered engines, one comparison' worked workflow — all real viva- modules.

Composes, through the unified workflow API, the exact SBML engine
(``TelluriumUTCStep``), a neural surrogate of the same model
(``viva_torch.NeuralProcess``), the BioGears physiology engine
(``BioGearsProcess``), and a derivation step (``MathExpressionStep``). Mirrors the
digital-twin API's exact<->surrogate tiering and the biogears<->SBML sepsis story.

``tiered_workflow_spec`` builds the spec; a surrogate checkpoint is supplied by the
caller (train one with ``viva_torch.train_surrogate``). The spec composes four real
viva- modules into one comparison graph.
"""
from __future__ import annotations

from .workflow_models import WorkflowNode, WorkflowSpec, WorkflowWire

# A small, non-stiff kinetic model: S1 -> S2 -> S3.
SMALL_MODEL = "S1 -> S2; k1*S1; S2 -> S3; k2*S2; k1=0.3; k2=0.12; S1=10; S2=0; S3=0;"
SPECIES = ["S1", "S2", "S3"]
TIMESTEP_S = 1.0


def tiered_workflow_spec(*, checkpoint: str, timestep: float,
                         feature_keys: list[str],
                         include_biogears: bool = True) -> WorkflowSpec:
    """Build the tiered workflow: exact SBML + neural surrogate + (physiology) +
    a MathExpression derivation summing the surrogate's species."""
    # surrogate species feed the derivation (surrogate_total = sum of species)
    syms = ["a", "b", "c"][: len(feature_keys)]
    expr = " + ".join(syms)
    nodes = [
        WorkflowNode(id="exact", module="TelluriumUTCStep",
                     config={"model": SMALL_MODEL, "model_format": "antimony"}),
        WorkflowNode(id="surrogate", module="NeuralProcess",
                     config={"checkpoint": checkpoint}, interval=timestep),
        WorkflowNode(id="compare", module="MathExpressionStep",
                     config={"expressions": [{"out": "surrogate_total", "expr": expr}]}),
    ]
    wires = [WorkflowWire(**{"from": f"surrogate.{k}", "to": f"compare.{s}"})
             for k, s in zip(feature_keys, syms)]
    emit = [f"surrogate.{feature_keys[0]}", "compare.surrogate_total"]
    if include_biogears:
        nodes.append(WorkflowNode(id="physio", module="BioGearsProcess", config={}))
        emit.append("physio.mean_arterial_pressure_mmHg")
    return WorkflowSpec(nodes=nodes, wires=wires, emit=emit)
