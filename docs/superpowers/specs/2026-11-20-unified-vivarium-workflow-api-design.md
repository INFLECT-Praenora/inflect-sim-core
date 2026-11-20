# Unified Vivarium Workflow API — Design

**Status:** approved
**Date:** 2026-11-20
**Repo:** `INFLECT-Praenora/inflect-sim-core` (`simulation-core`)
**Author:** Eran Agmon

---

## 1. Intent and why

We want `simulation-core` to be more polished and user-friendly around one
story: **define a module → call a module → connect modules into a workflow →
call the workflow — all through a single, clean API.** The goal is not to build
new simulation machinery; it is to **show that process-bigraph + vivarium
already do this cleanly**, by putting a thin, honest FastAPI surface over the
primitives that already exist.

The capability is already present but split across two surfaces that don't tell
one story:

- the **digital-twin API** (`viva_simulation_core/api/`, FastAPI) — model-centric
  (`/twins`, step/predict/ensemble); and
- the **vivarium-workbench API** (FastAPI, ~24 `/api/*` routes) — which already
  runs composites over HTTP (`/api/composite-test-run`), lists modules
  (`/api/catalog`, `/api/registry`), and renders the composite graph
  (`/api/composite-layout`), but wrapped in a large dashboard server.

This design adds a **small unified facade** that names the four verbs plainly and
delegates straight to the process-bigraph `Core` + `Composite` + `Visualization`.
The thinness of the facade is the proof: the primitives carry the weight.

## 2. The four verbs → primitives (what we are surfacing)

| Verb | Already exists as | Facade endpoint |
|---|---|---|
| define / discover a module | a `Process`/`Step` registered in the `Core` registry (149 registered today), addressed `local:Name` / `capability@version` | `GET /modules`, `GET /modules/{name}` |
| call a module | build a 1-node `Composite` and `.run()` | `POST /modules/{name}/call` |
| connect into a workflow | a `Composite` spec — bigraph wiring (incl. feedback edges) | `POST /workflows` |
| call the workflow | `Composite.run()` / `.state()` | `POST /workflows/{id}/run` |
| the graph render | `process_bigraph.Visualization` (what produced the `pmp` screenshot) | `GET /workflows/{id}/graph` |

## 3. Goals

- **One unified API.** The new `/modules` + `/workflows` router is **mounted on the
  existing digital-twin FastAPI app**, so `/twins`, `/modules`, and `/workflows`
  are one service — a single base URL, one auth scheme, one docs page.
- **Thin and delegating.** The router is a typed translation layer over
  `build_core()`, `Composite`, and `Visualization` — target ≈150 LOC. It adds no
  simulation logic of its own.
- **Real viva- modules only.** The worked workflow composes
  `BioGearsProcess`, `TelluriumUTCStep`, `viva_torch`'s `NeuralProcess`
  (neural surrogate), and `MathExpressionStep` — all real, registered viva-
  modules. No invented/mock steps.
- **Generic module introspection.** `GET /modules` reads the live registry
  (`core.list_processes()`) and each module's declared interface (config + input/
  output ports) generically — no hardcoded per-module knowledge, so every one of
  the 149 modules is reachable, not just the three in the demo.
- **The graph matches the screenshot.** `GET /workflows/{id}/graph` returns the
  same layout/visualization data the workbench renders (nodes = Steps, edges =
  wiring, capability labels), so the demonstration looks like the `pmp` figure.
- **Runnable proof (then A).** A narrated `examples/vivarium_workflow_demo.py`
  (same `TestClient` style as `digital_twin_demo.py`) walks all four verbs with
  real printed requests/responses.

## 4. Non-goals

- **Not** replacing or re-implementing the workbench's `/api/composite-*`
  endpoints — this is a clean *facade/demonstration*, not a second dashboard.
- **Not** workflow persistence or a durable workflow store. Workflows live in a
  small in-process registry (keyed by `workflow_id`) for the life of the server,
  mirroring the twin registry. A restart forgets them; the demo re-defines them.
- **Not** a composite-authoring UI, and **not** arbitrary user Python — a workflow
  spec references registered module names + wiring only; no `eval`.
- **Not** full-horizon scientific runs inside the live API (BioGears + stiff SBML
  are slow — see §7). The demo proves the plumbing at a short horizon, not a 48 h
  clinical result.

## 5. Anti-goals (failure even if it "works")

- A facade thick enough to look like new machinery — it must obviously delegate,
  or it fails to *show pb/vivarium already do this*.
- A `GET /modules` that only knows the three demo modules — it must be generic.
- A demo that silently blocks for minutes because it ran BioGears/stiff SBML
  full-horizon in-process.
- Hardcoded per-module config/port knowledge that drifts from the real registry.

## 6. Constraints

- **No AI attribution** in any commit or PR (overrides harness default).
- **Worktree discipline** — work in `simulation-core--unified-workflow-api`;
  verify branch + HEAD before every commit/push; explicit-refspec push.
- **`viva-torch` moves into base `dependencies`** (author's call), joining its
  siblings `viva-core` / `viva-tellurium` / `viva-biogears` / `viva-emitters`
  which are already base deps — so the neural-surrogate Step is *always* present
  and first-class, not an opt-in `[surrogate]` extra. **Tradeoff (author
  accepted):** torch is heavy, so the base install gets larger and CI a bit
  slower; the payoff is a surrogate tier that is always available and always
  discoverable via `GET /modules`. The `[surrogate]` extra is kept as an empty
  back-compat alias so `pip install .[surrogate]` still resolves. `uv.lock` is
  re-locked in the same task.
- **`[api]` optional extra** holds the FastAPI deps (`fastapi`, `uvicorn`,
  `pydantic`); the new router is import-guarded and never imported by
  `viva_simulation_core/__init__.py` or the CLI, so a base (`[api]`-less) install
  still imports fine.
- **Reuse existing machinery:** `build_core()` (the canonical workspace core),
  `process_bigraph.Composite`, `process_bigraph.Visualization`. The router builds
  composites against the core from `build_core()`, never a bare core.
- **Bearer auth**, reusing the digital-twin API's existing auth dependency and key
  config — one scheme across the unified app.
- **Reproducibility / honesty:** every run response carries provenance
  (engine/library versions, n_steps, wall-ms) and says which modules ran.

## 7. The design

### 7.1 Execution model — in-process `Composite.run()`

The facade runs composites **in-process** via `Composite.run(n_steps)` — exactly
how the workbench's own `/api/composite-test-run` runs them. This is deliberately
*different* from the digital-twin API's worker-subprocess isolation: that
isolation exists for **warm, persistent, concurrently-stepped** roadrunner twins
(thread-safety + segfault blast-radius). A one-shot composite run is the
ecosystem's normal mode and needs no worker. This distinction is documented in
the code so the two execution models are not confused.

### 7.2 Module endpoints (define / call)

- `GET /modules` → `[{name, address, kind (Process|Step), config_schema,
  input_ports, output_ports, capability_label}]`, read generically from the live
  registry. De-duplicates the fully-qualified / short-name pairs
  (`viva_tellurium.processes.TelluriumUTCStep` ↔ `TelluriumUTCStep`).
- `GET /modules/{name}` → one module's interface card (the "capability" view).
  404 on unknown name.
- `POST /modules/{name}/call` → body `{config?, inputs?, n_steps?}`; builds a
  1-node `Composite` wiring the given inputs to the module and an emitter to its
  outputs, runs it, returns `{outputs, provenance}`. 422 on bad config/inputs.

### 7.3 Workflow endpoints (connect / call / graph)

- `POST /workflows` → body is a **workflow spec**: `{nodes: [{id, module,
  config?}], wires: [{from, to}], emit?: [...]}`. The router translates it to a
  process-bigraph composite spec, resolves it against the core (validating module
  names + wiring), stores a `WorkflowSession` in the in-process registry, and
  returns `{workflow_id, graph}`. 422 on unknown module / invalid wiring.
- `POST /workflows/{id}/run` → body `{n_steps, overrides?}`; `Composite.run()`;
  returns `{observables, provenance}`. 404 unknown id.
- `GET /workflows/{id}/graph` → the `Visualization` layout data (nodes, edges,
  capability labels) — the Graph render from the screenshot. 404 unknown id.
- `DELETE /workflows/{id}` → evict.

### 7.4 The worked workflow (real viva- modules)

"Tiered engines, one comparison" — mirrors the real biogears↔SBML sepsis
investigation *and* the digital-twin API's exact↔surrogate tiering:

```
BioGearsProcess  ─ physiology (MAP, HR, SBP) ─┐
TelluriumUTCStep ─ exact SBML/kinetic model ──┼─► MathExpressionStep ─► derived/comparison observable
NeuralProcess    ─ neural surrogate (viva_torch) ┘   (physiology vs exact vs surrogate)
```

- **`BioGearsProcess`** — physiology engine as a Step.
- **`TelluriumUTCStep`** — exact SBML (or small kinetic) model as a Step.
- **`NeuralProcess`** (`viva_torch`) — the neural-surrogate Step, built from a
  freshly trained + curated checkpoint via `register_neural_process` +
  `neural_process_node(checkpoint, spec, interval)` (reusing the existing
  `surrogate_curate.train_and_curate` machinery). Runs at its trained timestep.
- **`MathExpressionStep`** — derives a clinical/comparison observable across the
  engines' outputs (e.g. exact-vs-surrogate delta).

### 7.5 Result + provenance contract

Run responses return only emitted observables (targeted, not full state) plus
`provenance {module_versions, lib_versions, n_steps, wall_ms}`. Module-call
responses return the module's declared outputs.

## 8. Approaches considered

- **A — Narrated demo script only**, over the existing workbench FastAPI. Lightest,
  but there is no single clean four-verb surface to point at — the "unified API"
  stays implicit.
- **B — Thin unified FastAPI facade + demo (CHOSEN).** A ~150 LOC router naming the
  four verbs, mounted on the existing twin app, delegating to the primitives;
  then the narrated demo drives it. Gives an actual unified surface users can call
  and reviewers can read, while staying obviously thin.
- **C — Browsable workbench showcase** (build the composite as a published
  investigation). Most polished on-site, but heaviest, and depends on B's story
  being settled first. Natural follow-on, out of scope here.

**Chosen: B, then its demo (A on top of B).** The facade is the "unified API" the
vision names; the demo is the runnable proof.

## 9. Decisions settled with the author

1. **Mount on the existing digital-twin app** — one unified service, not a second
   server.
2. **Use real viva- modules** — `BioGearsProcess`, `TelluriumUTCStep`,
   `viva_torch.NeuralProcess`, `MathExpressionStep` — as much as possible; no
   invented `pmp` steps.
3. **Demo workflow = "tiered engines, one comparison"** (BioGears + Tellurium SBML
   + neural surrogate → MathExpression), short-horizon.
4. **`viva-torch` is a base dependency**, not an opt-in extra — the surrogate tier
   is always installed and first-class (tradeoff accepted: heavier base/CI).

## 10. Decisions settled by the author (builder's-eye calls)

- **In-process execution** for the facade (§7.1), not worker isolation.
- **Workflows are in-process / ephemeral** (no persistence) — demonstration scope.
- **Fast CI vs slow showcase split:** the **fast** test set uses a light
  sub-composite (`TelluriumUTCStep` on a tiny decay model → `MathExpressionStep`)
  so CI stays quick — it does **not** train a surrogate or run BioGears (both
  slow), even though torch is now always installed; a `@pytest.mark.slow` test
  exercises the full tiered workflow (BioGears + Tellurium + `NeuralProcess`)
  end-to-end at a short horizon; the narrated demo runs the full workflow at a
  short horizon (training + curating a tiny surrogate first).
- **`NeuralProcess` is registered in `build_core()`** so it appears in
  `GET /modules` as a first-class module (now that viva-torch is a base dep),
  reachable by the facade without any per-call install.
- **Generic registry introspection** for `GET /modules` — no hardcoded module
  knowledge; de-dup fqcn/short-name pairs.
- **Reuse the twin API's bearer auth + key config** for the whole unified app.
- **Graph data source:** `process_bigraph.Visualization` (same as the workbench
  render), returned as structured layout JSON (nodes/edges/labels), not an image.

## 11. What's left to the builder (feeds the plan)

- **Dependency move:** `viva-torch` from the `[surrogate]` extra into base
  `dependencies`; keep `[surrogate]` as an empty back-compat alias; re-lock
  `uv.lock`; ensure `build_core()` registers `viva_torch`'s `NeuralProcess`
  (via `register_neural_process`) so it lists in `GET /modules`.
- The `/modules` + `/workflows` FastAPI router + pydantic request/response models,
  import-guarded under `viva_simulation_core/api/`.
- Generic registry introspection (name, kind, config schema, input/output ports,
  capability label; fqcn/short-name de-dup).
- Single-module call: 1-node composite build + run + output extraction.
- Workflow spec → process-bigraph composite-spec translation + core resolution +
  in-process `WorkflowSession` registry (define/run/graph/delete).
- `Visualization`-backed graph endpoint.
- Mount the router on the existing digital-twin app under the shared auth.
- The "tiered engines, one comparison" worked workflow (BioGears + Tellurium +
  `NeuralProcess`), incl. training + curating a tiny surrogate for the demo.
- `examples/vivarium_workflow_demo.py` — narrated `TestClient` walkthrough of all
  four verbs, ending on the tiered workflow + its graph.
- Tests — fast: module list/introspect, single-module call, workflow
  define/run/graph on the light composite, auth, 404/422 paths; slow
  (`@pytest.mark.slow`): the full tiered workflow (BioGears + Tellurium +
  `NeuralProcess`) end-to-end at a short horizon.
- A short `docs/` section (sibling to `docs/api.md`) documenting the four verbs.
