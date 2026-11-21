# Unified Simulation-Core API — Design

**Status:** approved (Approach A), 2026-11-21
**Worktree/branch:** `simulation-core--reframe` / `reframe/unified-sim-core-api`

## Intent & why

Make the repo *itself* demonstrate vivarium as a **simulation core**: you register
simulation modules, engine adapters, and API-calling steps into one registry, then
drive them from **one unified API** that adapts to the task — call one engine,
compose several and hand a model file between them, ensemble a sweep, serve a model
as a stepping API, call out to a remote model, or let an external workflow engine
(Nextflow) orchestrate vivarium composites.

The audience is the INFLECT team: show, don't tell. The repo should read as a clean,
honest proof that vivarium is sufficient scaffolding for composition, ensembles,
orchestration, and serving — without a wall of production-service machinery obscuring
the point.

## Goals

1. **One registry, one API, many ways to run.** Everything registers into
   `build_core()`; the surface is the four verbs exposed as (a) a top-level import
   facade and (b) a `viva-sim` CLI, both thin over the existing `workflow.py`.
2. **Six runnable "ways to run," one per task shape:** `call`, `connect`+`run`,
   `sweep`, `serve`, `api-call`, and Nextflow interop — each with a small, real example.
3. **Trim the twin service** to one lean "serve a model as a stepping API" path
   (reusing `modeling.load_engine`), deleting ~1.8 kLOC of production-service code.
4. **Repo-wide cleanup:** remove dead code and AI-slop across `api/`, `examples/`, `docs/`.
5. **Honest Nextflow interop:** vivarium *inside* a Nextflow pipeline process. No
   "faster than Nextflow" claim unless a benchmark in-repo actually demonstrates it.
6. CI stays green; the existing workflow/surrogate/composite tests keep passing.

## Non-goals

- No new simulation science; the sepsis composites, engines, and surrogate framework
  are reused as-is.
- Not touching the `biogears-sbml-sepsis` investigation (separate, already shipped).
- No distributed scheduler, no container orchestration of our own — Nextflow is the
  interop demo, not a thing we reimplement.

## Anti-goals (would make this a failure even if it "works")

- Any performance claim we can't back with an in-repo measurement (echoes the
  "bit-for-bit" overclaim already cut).
- Re-growing a heavyweight serving service; `serve` must stay lean.
- A facade/CLI that reimplements composition logic instead of delegating to
  `workflow.py` — one source of truth for the four verbs.

## Constraints

- **No AI attribution** in any commit/PR (overrides harness reminders).
- Dedicated worktree; verify branch/HEAD and only-our-commits before every push.
- Lean: delete > add; the facade and CLI are thin.
- `build_core()` import-walk must stay import-error-tolerant (a missing optional dep
  — torch, nextflow — never breaks registration).

## Approaches considered

- **A — facade + CLI (chosen).** Add top-level exports (`catalog/call/connect/run/serve`)
  and `viva-sim list|call|connect|run|serve`, both delegating to `workflow.py` +
  the lean serve path. Shows the unified API in code *and* from the shell; little new
  code because the verbs already exist.
- **B — docs + cleanup only.** Keep `workflow.py` as the surface, just document and
  trim. Rejected: "a unified API that adapts to the task" is the headline; leaving six
  task shapes as scattered function calls tells rather than shows.
- **C — CLI only.** A `viva-sim` CLI, no import facade. Rejected: half the audience
  reads the Python surface; the import facade is the thing they'd embed.

## The design

### Spine
`build_core()` already walks the package and registers every `Process`/`Step`/generator
via `__init_subclass__`. Three *kinds* of registered thing make the story:
1. **simulation modules** — the sepsis composites + engines (exist);
2. **adapters** — engine adapters wrapping a real simulator as a Process (exist:
   tellurium/biogears/surrogate);
3. **API-calling steps** — a `Process` whose update is a remote model call (new, small).

### The facade — `viva_simulation_core/__init__.py`
Export five names, each a thin wrapper over existing functions:
- `catalog(core=None)` → `workflow.list_modules` (cards: name, kind, ports).
- `call(name, config=None, inputs=None, steps=..., core=None)` → `workflow.call_module`.
- `connect(spec)` / `run(spec, steps)` → `workflow.spec_to_composite_state` + run.
- `serve(source, model_format=...)` → a `ServedModel` handle (below).
`build_core` stays the registration entry; the facade imports lazily to keep import light.

### `serve` — lean stepping handle, `viva_simulation_core/serve.py`
- `ServedModel`: `load_engine(source, fmt)` (reused from `modeling.py`) → object with
  `.observables()`, `.step(overrides) -> state`, `.predict(n, overrides) -> trajectory`,
  `.reset()`. In-process, no workers/ownership/TTL/persistence.
- `model_server(served) -> ASGI app` with **one** endpoint `POST /step` (state in →
  stepped state out), ~40 lines, for the api-call demo. Optional import of starlette;
  absent → clear error, never breaks `build_core`.

### `api-call` step — `viva_simulation_core/steps/api_call.py`
- `ApiCallStep(Process)`: config `{url}`; each update POSTs the current state to a
  model-serving `url` and returns the response as its update. Demonstrates vivarium
  orchestrating a *remote* simulation inside a composite. Registered via subclass hook.
- The api-call example wires `serve` → `model_server` (locally) ← `ApiCallStep` so it
  runs with no external creds and shows the full serve/consume loop.

### Nextflow interop — `examples/nextflow/`
- `main.nf`: a 2–3 process pipeline where one process shells `viva-sim run <spec>`
  (or `python -m viva_simulation_core.cli run`) on a vivarium composite, and a
  downstream process consumes its emitted output. A `README.md` states the honest
  framing: vivarium slots into existing workflow tooling and owns the fine-grained
  in-process coupling Nextflow's task model can't express. Guarded so the repo test
  suite doesn't require a `nextflow` binary (the example is documented + a smoke test
  that runs the underlying `viva-sim run` directly).

### CLI — `viva_simulation_core/cli/`
Add verbs to the existing typer app (and a `viva-sim` console-scripts alias):
`list` (catalog), `call <module>`, `connect <spec>` + `run`, `serve <model>` (prints
observables / steps once), mirroring the facade. One source of truth: CLI calls the
same functions as the facade.

### Cleanup — delete-list (verified no live importers outside the service)
Delete: `api/{app,auth,config,heavy_plane,models,persistence,registry,worker}.py`
(the twin service) and `api/{surrogate_backend,surrogate_torch,surrogate_curate}.py`
(superseded by `surrogate/`). **Keep** `api/modeling.py` (engine-load helpers reused by
`serve`), `api/workflow*.py`, `surrogate/`, `composites/`, `analysis/`,
`capability_cards.py`. Fix the one stale docstring reference in
`surrogate/backends/neural.py:62`. Rework the three `examples/*.py` (all import the
deleted service) into: `serve_demo.py`, `workflow_tour.py` (the six-ways tour), and
keep a surrogate demo repointed at `surrogate/`. Prune AI-slop prose in `docs/`
(consolidate `api.md` + `workflow-api.md` + `first-run-agent-guide.md` into one clean
`docs/simulation-core.md`; keep `superseded-*.md` as the record).

### README
Rewrite around the spine: the registry, the one API, the six ways (each a copy-pasteable
snippet), the Nextflow note. Precise and plain; no slop.

## What's left to the builder

- Exact `ServedModel` method bodies (reuse `modeling.py` helpers; TDD each).
- Exact typer command signatures (follow the existing CLI's patterns).
- The Nextflow `main.nf` channel wiring (standard DSL2; the smoke test exercises the
  underlying CLI call, not the `nextflow` binary).
- Which `docs/` files fold where (consolidate per the cleanup note; keep records).
