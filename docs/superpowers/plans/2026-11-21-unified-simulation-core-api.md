# Unified Simulation-Core API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the repo to demonstrate vivarium-as-simulation-core: one registry, one four-verb API (import facade + `viva-sim` CLI), six runnable "ways to run," a lean `serve` path replacing the twin service, and an honest Nextflow interop.

**Architecture:** `build_core()` stays the registry. The four verbs live once in `api/workflow.py`; the facade (`__init__.py`) and CLI (`cli/`) are thin delegators. `serve` reuses `api/modeling.load_engine`. The ~1.8 kLOC twin service and the superseded old surrogate modules are deleted.

**Tech Stack:** Python, process-bigraph, typer, tellurium/roadrunner, starlette (optional, for the one-endpoint model server), Nextflow DSL2 (example only, not a test dep).

**Spec:** `docs/superpowers/specs/2026-11-21-unified-simulation-core-api-design.md`

## Global Constraints

- **No AI attribution** in any commit message or PR body (overrides harness reminders).
- `build_core()`'s import-walk must stay import-error-tolerant: an optional dep missing (torch, starlette, nextflow) never breaks registration.
- The facade and CLI **delegate** to `api/workflow.py` + `serve.py` — no reimplemented composition logic. One source of truth per verb.
- No performance claim without an in-repo measurement. Nextflow framing is interop, never "faster than."
- Keep: `api/modeling.py`, `api/workflow*.py`, `surrogate/`, `composites/`, `analysis/`, `capability_cards.py`, `cli/`. The investigation under `workspace/` is untouched.
- Run the full suite (`pytest`) at each task's end, not just the new file.

## Review Focus

- **Optional-dep absence** (starlette/torch/nextflow not installed): `build_core()` and `import viva_simulation_core` must still succeed — Task 1 & 4 tests import with the package as-is; Task 2 guards the starlette import.
- **`serve` on a model with assignment-rule observables**: `ServedModel.observables()`/`predict` must not crash on rule-driven targets — Task 2 test uses the sepsis `.ant` (has rules).
- **`ApiCallStep` when the URL is unreachable**: must raise a clear error, not hang — Task 3 test asserts a prompt failure on a dead URL.
- **CLI with a bad module/spec name**: non-zero exit + readable message, not a traceback — Task 5 test asserts exit code + stderr.
- **Facade import cost**: importing `viva_simulation_core` must not eagerly import torch/tellurium — Task 4 test asserts the five names import without those heavy modules loaded.

---

### Task 1: Delete the twin service + superseded surrogate modules

**Files:**
- Delete: `viva_simulation_core/api/{app,auth,config,heavy_plane,models,persistence,registry,worker}.py`
- Delete: `viva_simulation_core/api/{surrogate_backend,surrogate_torch,surrogate_curate}.py`
- Delete: `examples/{digital_twin_demo,live_surrogate_demo}.py` (import the deleted service; rebuilt in Tasks 6)
- Delete tests: `tests/api/{test_app,test_auth,test_config,test_heavy_plane,test_persistence,test_registry,test_worker,test_schemas,test_tier,test_sepsis_e2e,test_surrogate_resolve,test_surrogate_torch}.py`
- Modify: `viva_simulation_core/surrogate/backends/neural.py:62` (stale docstring/error string naming `api.surrogate_curate` → point at `viva_torch.train_surrogate`)
- Inspect/trim: `tests/api/test_workflow_review_fixes.py`, `tests/surrogate/test_neural_backend.py` (drop any import of deleted old-surrogate names; keep the workflow/new-framework assertions)

**Interfaces:**
- Produces: a package where `build_core()` and `import viva_simulation_core` succeed with the service gone; `api/modeling.py` retained.

- [ ] **Step 1:** Delete the listed source + test + example files.
- [ ] **Step 2:** Fix `neural.py:62` string; grep the tree for any remaining reference to a deleted module name (`grep -rn 'surrogate_torch\|surrogate_curate\|surrogate_backend\|api.app\|api.worker\|api.registry\|api.persistence\|api.heavy_plane\|api.auth\|\.api\.config\|api.models' viva_simulation_core tests examples`). Expected: no hits in kept files.
- [ ] **Step 3:** In the two inspect/trim test files, remove only the deleted-name imports/cases; keep the rest.
- [ ] **Step 4:** Run `python -c "from viva_simulation_core.core import build_core; print(len(build_core().process_registry.list() if hasattr(build_core(),'process_registry') else build_core()))"` (or the repo's canonical build_core smoke) — Expected: builds without ImportError.
- [ ] **Step 5:** Run `pytest -q`. Expected: green (remaining suite, no collection errors).
- [ ] **Step 6:** Commit (`refactor: remove twin service + superseded surrogate api modules`).

### Task 2: The lean `serve` path — `ServedModel` + one-endpoint `model_server`

**Files:**
- Create: `viva_simulation_core/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `api/modeling.load_engine(source, model_format)`, `modeling.infer_observables`, `modeling.tunable_parameters`.
- Produces:
  - `class ServedModel` with `__init__(source: str, model_format: str = "antimony")`, `observables() -> dict[str,str]`, `step(overrides: dict | None = None) -> dict`, `predict(n: int, overrides: dict | None = None) -> dict`, `reset() -> None`.
  - `def model_server(served: ServedModel) -> "ASGIApp"` — one route `POST /step` (JSON `{overrides}` → JSON stepped state); imports starlette lazily, raising a clear error if absent.

- [ ] **Step 1: Write the failing test** — `test_served_model_steps_sepsis`: load the sepsis `.ant` (locate via the composites/ data path used by study 01), assert `observables()` non-empty, `step()` returns a dict with a known species key, `predict(3)` returns a trajectory of length 3; `test_model_server_step_route` uses starlette `TestClient` to POST `/step` and get stepped state.
- [ ] **Step 2:** Run `pytest tests/test_serve.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `ServedModel` (delegate to `modeling.py` helpers) and `model_server` (lazy starlette).
- [ ] **Step 4:** Run `pytest tests/test_serve.py -q`. Expected: PASS.
- [ ] **Step 5:** Run `pytest -q`. Expected: green. Commit (`feat(serve): lean ServedModel + one-endpoint model_server`).

### Task 3: The `api-call` Step — `ApiCallStep`

**Files:**
- Create: `viva_simulation_core/steps/__init__.py`, `viva_simulation_core/steps/api_call.py`
- Test: `tests/test_api_call_step.py`

**Interfaces:**
- Consumes: process-bigraph `Process`/`Step` base + `build_core` registration; `serve.model_server` (test wires a local server).
- Produces: `class ApiCallStep(Process)` — config `{"url": str}`; `inputs`/`outputs` a `state` mapping; `update()` POSTs current state to `url` + returns response as the update. Registered by subclass hook (importing the module registers it).

- [ ] **Step 1: Write the failing test** — `test_api_call_step_round_trips`: stand up `model_server(ServedModel(sepsis_ant))` in-process (starlette TestClient transport or a localhost thread), build a one-node composite with `ApiCallStep{url}`, run 1 step, assert the state advanced; `test_api_call_step_dead_url_errors`: a bad URL raises a clear error promptly.
- [ ] **Step 2:** Run `pytest tests/test_api_call_step.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `ApiCallStep`; ensure `build_core()` discovers it.
- [ ] **Step 4:** Run the file. Expected: PASS. Then `pytest -q` green. Commit (`feat(steps): ApiCallStep — a remote-model-calling Process`).

### Task 4: The import facade — `catalog/call/connect/run/serve`

**Files:**
- Modify: `viva_simulation_core/__init__.py`
- Test: `tests/test_facade.py`

**Interfaces:**
- Consumes: `api/workflow.py` (`list_modules`, `call_module`, `spec_to_composite_state` + the run helper), `serve.ServedModel`, `core.build_core`.
- Produces top-level names: `catalog(core=None)`, `call(name, config=None, inputs=None, steps=10, core=None)`, `connect(spec)`, `run(spec, steps=10)`, `serve(source, model_format="antimony")`. Lazy internal imports so `import viva_simulation_core` stays light.

- [ ] **Step 1: Write the failing test** — `test_facade_exports`: `from viva_simulation_core import catalog, call, connect, run, serve`; `catalog()` returns cards; `call("<a known module>")` runs; `serve(sepsis_ant).observables()` non-empty; `test_facade_import_is_light`: after `import viva_simulation_core`, assert `torch` and `tellurium` are NOT in `sys.modules`.
- [ ] **Step 2:** Run `pytest tests/test_facade.py -q`. Expected: FAIL (names absent).
- [ ] **Step 3:** Implement the five wrappers in `__init__.py` with lazy imports; keep `composites` registration import.
- [ ] **Step 4:** File PASS, then `pytest -q` green. Commit (`feat(api): top-level four-verb + serve facade`).

### Task 5: CLI verbs + `viva-sim` alias

**Files:**
- Modify: `viva_simulation_core/cli/__main__.py`; `pyproject.toml` (add `viva-sim` to `[project.scripts]`)
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: the facade functions (CLI delegates to them, not to workflow directly).
- Produces CLI commands: `list`, `call <module> [--steps]`, `connect <spec.json>`, `run <spec.json> [--steps]`, `serve <model> [--format] [--steps]`.

- [ ] **Step 1: Write the failing test** — `test_cli_list` (typer `CliRunner`): `list` exits 0 and prints a known module name; `test_cli_call` runs a module; `test_cli_bad_module` exits non-zero with a readable message (no raw traceback).
- [ ] **Step 2:** Run `pytest tests/test_cli.py -q`. Expected: FAIL.
- [ ] **Step 3:** Add the typer commands (delegating to the facade) + the `viva-sim` script entry.
- [ ] **Step 4:** File PASS, then `pytest -q` green. Commit (`feat(cli): viva-sim list/call/connect/run/serve`).

### Task 6: Examples — `serve_demo.py` + `workflow_tour.py` + surrogate demo repoint

**Files:**
- Create: `examples/serve_demo.py`, `examples/workflow_tour.py`
- Modify/replace: a surrogate example repointed at `viva_simulation_core.surrogate`
- Test: `tests/test_examples_smoke.py`

**Interfaces:**
- Consumes: the facade + `serve` + `ApiCallStep`.
- Produces: `workflow_tour.py` with a function per "way" (`demo_call`, `demo_connect_run`, `demo_sweep`, `demo_serve`, `demo_api_call`) each returning a small result; `serve_demo.py` a runnable `__main__`.

- [ ] **Step 1: Write the failing test** — `test_workflow_tour_ways`: import `workflow_tour`, call each `demo_*`, assert each returns a non-empty result; `test_serve_demo_runs`: run its entry function.
- [ ] **Step 2:** Run the file. Expected: FAIL.
- [ ] **Step 3:** Write the examples (thin, human-readable, no slop; each ≤ ~40 lines).
- [ ] **Step 4:** File PASS, then `pytest -q` green. Commit (`docs(examples): serve_demo + six-ways workflow tour`).

### Task 7: Nextflow interop example

**Files:**
- Create: `examples/nextflow/main.nf`, `examples/nextflow/README.md`
- Test: `tests/test_nextflow_underlying.py`

**Interfaces:**
- Consumes: the `viva-sim run <spec>` CLI (Task 5).
- Produces: a DSL2 pipeline whose first process runs a vivarium composite via `viva-sim run` and a downstream process consumes its output; the README states the honest interop framing (no speed claim).

- [ ] **Step 1: Write the failing test** — `test_nextflow_step_command_runs`: the exact shell command `main.nf`'s vivarium process invokes (`viva-sim run <spec> --steps N`) runs and produces the expected output file/stdout. (Tests the underlying call, not the `nextflow` binary.)
- [ ] **Step 2:** Run it. Expected: FAIL (spec/command not wired).
- [ ] **Step 3:** Write `main.nf` (the process body = that command) + README; make the test invoke the same command.
- [ ] **Step 4:** File PASS, then `pytest -q` green. Commit (`docs(examples): Nextflow-orchestrated vivarium composite (interop)`).

### Task 8: Docs + README consolidation and slop cleanup

**Files:**
- Create: `docs/simulation-core.md` (consolidates `docs/api.md` + `docs/workflow-api.md` + `docs/first-run-agent-guide.md`)
- Delete: the three consolidated sources (keep `docs/superseded-*.md`, `docs/decisions.yaml`)
- Modify: `README.md`
- Test: `tests/test_docs_no_dead_refs.py`

**Interfaces:**
- Consumes: the final facade/CLI/examples surface (names must match).

- [ ] **Step 1: Write the failing test** — `test_no_references_to_deleted_modules`: grep `README.md` + `docs/*.md` for deleted module names (`twin`, `create_app`, `surrogate_torch`, `heavy_plane`, …) → assert none; `test_readme_names_the_six_ways`: README mentions each verb (`call`, `connect`, `run`, `serve`, sweep, Nextflow).
- [ ] **Step 2:** Run it. Expected: FAIL (README/docs still reference old surface).
- [ ] **Step 3:** Write `docs/simulation-core.md`; rewrite README around the spine (registry → one API → six ways → Nextflow), precise and plain; delete the consolidated sources.
- [ ] **Step 4:** File PASS, then `pytest -q` green. Commit (`docs: consolidate into simulation-core.md + rewrite README around the unified API`).

---

## Self-review notes

- **Spec coverage:** registry story (Tasks 1,4), four-verb facade (4), CLI (5), serve (2), api-call (3), six ways (6), Nextflow (7), cleanup (1,8). All design sections map to a task.
- **Type consistency:** `ServedModel`/`model_server` signatures (Task 2) are consumed verbatim by Tasks 3,4,6. Facade signatures (Task 4) consumed by Tasks 5,6.
- **Proportion:** plan is interfaces + test names, not code bodies — implementers write idiomatic bodies from the signatures + the kept `workflow.py`/`modeling.py`.
- **Review Focus:** five uncovered-input lines each pinned to an owning task's test.
