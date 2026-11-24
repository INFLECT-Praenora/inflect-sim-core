# The simulation core — one registry, one API, many ways to run

Vivarium as a simulation core: you **register** simulation modules, engine
adapters, and API-calling steps into one registry, then drive them from **one
unified API** that adapts to the task. The same four verbs are available as a
Python import and as the `viva-sim` CLI — both thin wrappers over
`viva_simulation_core/api/workflow.py` and `serving.py`.

```python
from viva_simulation_core import catalog, call, connect, run, serve
```

```
viva-sim list | call <module> | workflow <spec.json> | serve <model> | run <composite_id>
```

## The registry

Everything registers into `build_core()` (`core.py`): the sepsis composites, the
engine adapters (`TelluriumUTCStep`, `BioGearsProcess`, `NeuralProcess`,
`SurrogateProcess`), and the API-calling `ApiCallStep`. A Process/Step subclass
defined in this package is discovered and registered automatically — importing
the module is enough.

`catalog()` (CLI `viva-sim list`) lists every registered module as a card:
`name`, `kind` (process/step), and `config_schema`.

## Switching engines — one contract, many model types

Every engine adapter exposes the **same typed ports** — the observable contract.
So the engines are interchangeable: the exact Tellurium model, the BioGears C++
engine, and a learned surrogate are each a box over the same ports, and switching
model types is a one-node change. This is what makes `call`, `connect`, and `run`
engine-agnostic: the verbs address ports, never a specific engine.

## The four verbs

### call — run one module

```python
call("MathExpressionStep",
     config={"expressions": [{"out": "y", "expr": "2*x + 3"}]},
     inputs={"x": 5.0}, steps=1)        # -> {"y": 13.0}
```

### connect + run — compose modules into a workflow

A workflow spec is `nodes` (each a registered module), `wires` (`node.port ->
node.port`), and `emit` (the observables to read back):

```python
spec = {
  "nodes": [
    {"id": "inc",   "module": "MathExpressionStep", "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
    {"id": "scale", "module": "MathExpressionStep", "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}},
  ],
  "wires": [{"from": "inc.a", "to": "scale.a"}],
  "emit": ["scale.b", "inc.a"],
}
run(spec, steps=1, overrides={"inc.x": 5.0})   # -> {"scale.b": 60.0, "inc.a": 6.0}
```

`connect(spec)` normalizes a dict (or `WorkflowSpec`) if you want to inspect the
graph before running. From the shell: `viva-sim workflow spec.json --steps 1
--override inc.x=5.0`.

## serve — a model as a stepping API

`serve` loads an SBML/Antimony model and returns a `ServedModel` you can step:

```python
m = serve(open("model.ant").read(), "antimony")
m.observables()           # readable ids (species, non-constant globals, rule targets)
m.step({"S1": 20.0})      # seed overrides, advance one step, return the new state
m.predict(10)             # a per-observable trajectory of length 10
m.reset()                 # back to initial conditions
```

`serving.model_server(m)` wraps one instance in a single-route ASGI app
(`POST /step`) so a model can be served over HTTP with no production-service
machinery. CLI: `viva-sim serve model.ant`.

## api-call — orchestrate a remote simulation

`ApiCallStep` is a Process whose update is a *remote model call*: it POSTs its
declared observables to a model-serving URL (a `model_server`, or a remote
viva-api) and applies the stepped state. A vivarium composite can therefore
orchestrate a remote simulation as one interchangeable node.

## run a registered composite

`viva-sim run <composite_id>` builds and runs any catalog composite directly via
process-bigraph (no server). This is the path a workflow engine calls — see the
Nextflow example below.

## Runnable examples

The contract end to end: [`examples/contract_demo.py`](../examples/contract_demo.py)
(`python -m examples.contract_demo`). Workflow-engine interop:
[`examples/nextflow/`](../examples/nextflow/).
