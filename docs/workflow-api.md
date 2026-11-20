# Unified vivarium-workflow API

One clean HTTP surface over process-bigraph's four verbs — **define a module, call
a module, connect modules into a workflow, call the workflow** — delegating
straight to `build_core()` + `Composite`. It is mounted on the **same FastAPI app
as the digital-twin API**, so `/twins`, `/modules` and `/workflows` are one
service under one bearer-auth scheme (see [`api.md`](api.md)). Set `<key>` to one
of your configured API keys; the examples assume `BASE=http://localhost:8000`.

All endpoints need `Authorization: Bearer <key>`.

Run it with no server (in-process, every request real):

```bash
uv run python examples/vivarium_workflow_demo.py
```

## 1. Define / discover a module

Every registered process-bigraph module (149+ in this workspace) is listed from
the live registry — no hardcoding.

```bash
curl -s $BASE/modules -H "Authorization: Bearer <key>"
# [{"name": "TelluriumUTCStep", "address": "local:TelluriumUTCStep", "kind": "Step",
#   "config_schema": {...}, "input_ports": null, "output_ports": null,
#   "capability_label": "TelluriumUTCStep"}, ...]

curl -s $BASE/modules/MathExpressionStep -H "Authorization: Bearer <key>"
# one module's full card, incl. best-effort input_ports / output_ports
```

## 2. Call a module

A single module run as a 1-node composite. `config` configures it, `inputs` seeds
its input ports, `n_steps` × `interval` is how far it advances.

```bash
curl -s $BASE/modules/MathExpressionStep/call -H "Authorization: Bearer <key>" \
  -H 'Content-Type: application/json' \
  -d '{"config": {"expressions": [{"out": "y", "expr": "2*x + 3"}]}, "inputs": {"x": 5}, "n_steps": 1}'
# {"outputs": {"y": 13.0}, "provenance": {"modules": ["MathExpressionStep"], ...}}
```

## 3. Connect a workflow

A workflow is a process-bigraph composite: `nodes` reference registered modules,
`wires` connect a producer's output port to a consumer's input port
(`"<node>.<port>"`), and `emit` names the `"<node>.<port>"` values to read back.
Returns a `workflow_id` and the graph (nodes + edges — the same structure the
workbench renders).

```bash
curl -s $BASE/workflows -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"nodes": [
         {"id": "inc",   "module": "MathExpressionStep", "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
         {"id": "scale", "module": "MathExpressionStep", "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}}],
       "wires": [{"from": "inc.a", "to": "scale.a"}],
       "emit": ["scale.b"]}'
# {"workflow_id": "ab12cd34ef56", "graph": {"nodes": [...], "edges": [{"from": "inc.a", "to": "scale.a"}]}}
```

## 4. Call the workflow

Runs the composite `n_steps` ticks of `interval` each, seeding input stores from
`overrides` (keyed `"<node>.<port>"`). Returns the emitted observables.

```bash
curl -s $BASE/workflows/$WID/run -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"n_steps": 1, "interval": 1.0, "overrides": {"inc.x": 5}}'
# {"observables": {"scale.b": 60.0}, "provenance": {"modules": ["MathExpressionStep"], ...}}

curl -s $BASE/workflows/$WID/graph -H "Authorization: Bearer <key>"   # nodes + edges
curl -s -X DELETE $BASE/workflows/$WID -H "Authorization: Bearer <key>"
```

## The tiered worked workflow

`viva_simulation_core/api/workflow_examples.py` builds a workflow from four real
viva- modules — mirroring the biogears↔SBML sepsis story and the digital-twin
API's exact↔surrogate tiering:

```
BioGearsProcess  — physiology (MAP, HR, SBP) ┐
TelluriumUTCStep — exact SBML/kinetic model  ┼─► MathExpressionStep ─► derived observable
NeuralProcess    — neural surrogate (viva_torch) ┘   (physiology vs exact vs surrogate)
```

`train_demo_surrogate` trains + curates a tiny surrogate so the `NeuralProcess`
node has a real checkpoint (it runs at its trained timestep — pass that as the
run `interval`). The surrogate and exact engine always run; the BioGears node
needs a compiled `pybiogears` engine to execute.

## Notes

- **Execution:** workflows run in-process via `Composite.run()` (the same way the
  workbench runs composites), serialized by a run lock. This is distinct from the
  digital-twin API's per-twin worker subprocesses, which exist for warm,
  concurrently-stepped roadrunner instances.
- **Lifetime:** workflows live in an in-process registry (ephemeral); a restart
  forgets them. Unknown module / bad wiring → `422`; unknown workflow → `404`.
- **Single worker:** the registry requires `uvicorn --workers 1` (already enforced
  by the app).
