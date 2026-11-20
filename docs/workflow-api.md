# Workflow API

Build simulations by snapping together ready-made parts, over HTTP.

- A **module** is one simulation building block — an SBML model runner, a neural
  surrogate, a physiology engine, a math step. (These are process-bigraph
  `Process`/`Step` classes; ~150 are registered.)
- A **workflow** wires modules together and runs them as one.

The API has exactly **four verbs**:

| Verb | Endpoint |
|---|---|
| **list** the modules you can use | `GET /modules` |
| **run one** module on its own | `POST /modules/{name}/call` |
| **connect** modules into a workflow | `POST /workflows` |
| **run** the workflow | `POST /workflows/{id}/run` |

It runs on the same server as the [digital-twin API](api.md) — same base URL, same
key. Every request needs `Authorization: Bearer <key>`; examples use
`BASE=http://localhost:8000`.

## See it work in 10 seconds

No server needed — this runs the whole API in-process and prints every real
request and response:

```bash
uv run python examples/vivarium_workflow_demo.py
```

## The four verbs

### 1. List modules

```bash
curl -s $BASE/modules -H "Authorization: Bearer <key>"
```
Each entry is a card: `name`, `kind` (`Process`/`Step`), and `config_schema` (how
to configure it). `GET /modules/{name}` returns one card with its input/output
ports.

### 2. Run one module

Give it a config and inputs; get its outputs back. Here a math step computes
`y = 2x + 3` at `x = 5`:

```bash
curl -s $BASE/modules/MathExpressionStep/call -H "Authorization: Bearer <key>" \
  -H 'Content-Type: application/json' \
  -d '{"config": {"expressions": [{"out": "y", "expr": "2*x + 3"}]}, "inputs": {"x": 5}}'

# → {"outputs": {"y": 13.0}, "provenance": {...}}
```

### 3. Connect a workflow

A workflow is a list of `nodes` (each is a module) plus `wires` that connect one
node's output to another's input (`"node.port"`). `emit` names what to read back.

This chains two steps — `a = x + 1`, then `b = a * 10`:

```bash
curl -s $BASE/workflows -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{
    "nodes": [
      {"id": "inc",   "module": "MathExpressionStep", "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
      {"id": "scale", "module": "MathExpressionStep", "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}}
    ],
    "wires": [{"from": "inc.a", "to": "scale.a"}],
    "emit":  ["scale.b"]
  }'

# → {"workflow_id": "ab12cd34ef56", "graph": {"nodes": [...], "edges": [...]}}
```

The returned `graph` (nodes + edges) is the diagram of your workflow.

### 4. Run the workflow

`overrides` seed inputs (`"node.port"`); `n_steps` × `interval` is how far it runs:

```bash
curl -s $BASE/workflows/$WID/run -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"n_steps": 1, "overrides": {"inc.x": 5}}'

# → {"observables": {"scale.b": 60.0}, "provenance": {...}}
```

Also: `GET /workflows/{id}/graph` (the diagram) and `DELETE /workflows/{id}`.

## A real example: three engines at once

`viva_simulation_core/api/workflow_examples.py` builds one workflow from four real
modules — the same comparison at the heart of the biogears↔SBML sepsis work, and
the exact↔surrogate tiering of the twin API:

```
TelluriumUTCStep  — exact SBML model         ┐
NeuralProcess     — fast neural surrogate     ┤─► MathExpressionStep ─► compare
BioGearsProcess   — physiology engine         ┘
```

`train_demo_surrogate()` trains a tiny surrogate first so the `NeuralProcess` node
has a real checkpoint. The exact engine and the surrogate run anywhere; the
BioGears node needs a compiled `pybiogears` to execute.

## Good to know

- **It's thin on purpose.** Every endpoint just calls process-bigraph
  (`build_core()` + `Composite`) — the API adds no simulation logic of its own.
- **Errors:** unknown module or bad wiring → `422`; unknown workflow → `404`.
- **Lifetime:** workflows live in memory and are forgotten on restart. Run one
  server process (`uvicorn --workers 1`); runs are serialized internally.
