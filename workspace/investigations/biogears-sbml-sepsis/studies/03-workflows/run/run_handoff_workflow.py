"""ORCHESTRATION / WORKFLOWS pillar (centerpiece): compose registered modules into
a pipeline and HAND OFF an Antimony/SBML file between engines — via the four-verb
workflow API, with no new simulation code.

The BioGears sepsis core was extracted to a portable Antimony/SBML file
(sources/biogears_sepsis.ant). This workflow hands that file to the Tellurium engine,
composes it with a learned surrogate and a derivation step, and runs the whole thing
with the four verbs:

  1. LIST    the registered modules (the engine boxes available).
  2. CALL    one module — TelluriumUTCStep on the real biogears_sepsis.ant: load the
             SBML/Antimony file and run it. This is the hand-off: the artifact
             extracted from the BioGears C++ engine is consumed by the SBML integrator.
  3. CONNECT a workflow — nodes (Tellurium loading the artifact + a NeuralProcess
             surrogate tier + a MathExpressionStep derivation) plus wires.
  4. RUN     the workflow — advance the composite, read the emitted observable.

Writes run/handoff_result.json.  Run:  PYTHONPATH=<worktree> python run_handoff_workflow.py
"""
import json
import time
from pathlib import Path

import torch

from viva_simulation_core.core import build_core
from viva_simulation_core.api.workflow import WorkflowRegistry, list_modules, call_module
from viva_simulation_core.api.workflow_models import WorkflowSpec, WorkflowNode, WorkflowWire
from viva_torch.spec import SurrogateSpec

HERE = Path(__file__).parent
INV = HERE.parents[2]
ANT = (INV / "sources" / "biogears_sepsis.ant").resolve()
CKPT = (INV / "assets" / "surrogate_stable2.pt").resolve()


def main() -> None:
    core = build_core()
    spec = SurrogateSpec(**torch.load(CKPT, map_location="cpu", weights_only=False)["spec"])

    # ── Verb 1: LIST the registered modules ──────────────────────────────────
    modules = [m.name for m in list_modules(core)]
    engine_boxes = [m for m in modules
                    if m in {"TelluriumUTCStep", "TelluriumProcess", "BioGearsProcess",
                             "NeuralProcess", "MathExpressionStep"}]

    # ── Verb 2: CALL one module — Tellurium loads the real .ant and runs it ───
    # This is the hand-off: the portable Antimony/SBML artifact extracted from the
    # BioGears C++ engine is handed to the Tellurium/roadrunner integrator.
    t0 = time.perf_counter()
    one = call_module(core, "TelluriumUTCStep",
                      config={"model_file": str(ANT), "model_format": "antimony",
                              "end_time": 10.0, "n_points": 11},
                      inputs={}, n_steps=1, interval=1.0)
    call_s = time.perf_counter() - t0
    loaded_ok = one is not None

    # ── Verb 3: CONNECT a workflow — artifact-loading Tellurium + surrogate + derive
    npc = core.link_registry.get("NeuralProcess")({"checkpoint": str(CKPT)}, core)
    nkeys = list(npc.inputs().keys())[:3]
    syms = ["a", "b", "c"]
    wf = WorkflowSpec(
        nodes=[
            WorkflowNode(id="exact", module="TelluriumUTCStep",
                         config={"model_file": str(ANT), "model_format": "antimony",
                                 "end_time": 10.0, "n_points": 11}),
            WorkflowNode(id="surrogate", module="NeuralProcess",
                         config={"checkpoint": str(CKPT)}, interval=300.0),
            WorkflowNode(id="compare", module="MathExpressionStep",
                         config={"expressions": [{"out": "surrogate_total", "expr": "a+b+c"}]}),
        ],
        wires=[WorkflowWire(**{"from": f"surrogate.{nkeys[i]}", "to": f"compare.{syms[i]}"})
               for i in range(3)],
        emit=["compare.surrogate_total"],
    )
    reg = WorkflowRegistry(lambda: core)
    wid, graph = reg.define(wf)

    # ── Verb 4: RUN the workflow ─────────────────────────────────────────────
    t1 = time.perf_counter()
    res = reg.run(wid, n_steps=1, overrides={}, interval=300.0)
    run_s = time.perf_counter() - t1

    out = {
        "verbs": {
            "list": {"n_modules": len(modules), "engine_boxes": engine_boxes},
            "call": {"module": "TelluriumUTCStep", "loaded_ok": bool(loaded_ok),
                     "wall_s": round(call_s, 3)},
            "connect": {"nodes": [n.id for n in wf.nodes],
                        "edges": [(e.from_, e.to) for e in graph.edges]},
            "run": {"wall_s": round(run_s, 3)},
        },
        "handoff": {
            "artifact": str(ANT),
            "loaded_by": "TelluriumUTCStep",
            "narrative": "biogears_sepsis.ant (extracted from the BioGears C++ engine) "
                         "is handed to the Tellurium/roadrunner integrator inside the workflow; "
                         "the same portable artifact can drive any engine box over the shared ports.",
        },
        "workflow": {
            "ran": True,
            "emit": res.observables,
            "provenance": {"modules": res.provenance.modules,
                           "wall_ms": round(res.provenance.wall_ms, 1),
                           "lib_versions": res.provenance.lib_versions},
            "note": "The compare step derives from the surrogate tier (a fast APPROXIMATE "
                    "engine, study 04/05). The point here is orchestration: list -> call -> "
                    "connect -> run, composing registered modules with no new simulation code.",
        },
    }
    (HERE / "handoff_result.json").write_text(json.dumps(out, indent=2))
    print(f"four verbs OK | hand-off: {ANT.name} loaded by TelluriumUTCStep | "
          f"workflow emit {res.observables}")


if __name__ == "__main__":
    main()
