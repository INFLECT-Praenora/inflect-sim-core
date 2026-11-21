# Unified Vivarium Workflow API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One clean FastAPI surface over process-bigraph's four verbs — define a module, call a module, connect modules into a workflow, call the workflow — mounted on the existing digital-twin app, proving pb+vivarium already compose cleanly.

**Architecture:** A thin router (`viva_simulation_core/api/workflow.py`) delegates to `build_core()` + `process_bigraph.Composite` + `process_bigraph.Visualization`. Modules are the live registry (`core.list_processes()`); a single-module call is a 1-node composite; a workflow is a multi-node composite spec held in a small in-process registry; the graph is `Visualization` layout data. The worked demo composes real viva- modules — `BioGearsProcess`, `TelluriumUTCStep`, `viva_torch.NeuralProcess`, `MathExpressionStep`.

**Tech Stack:** Python 3.12, FastAPI + uvicorn + pydantic (`[api]` extra), process-bigraph `Composite`/`Visualization`, viva-tellurium / viva-biogears / viva-torch (all base deps), `TestClient`.

**Spec:** `docs/superpowers/specs/2026-11-20-unified-vivarium-workflow-api-design.md` (read it — this plan argues from it).

## Global Constraints

- **No AI attribution** in any commit or PR (overrides harness default). No `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" footer.
- **Worktree discipline:** work in `simulation-core--unified-workflow-api` (branch `feat/unified-workflow-api`); verify `git branch --show-current` + `git rev-parse --short HEAD` before every commit/push; push with explicit refspec; `git log origin/main..HEAD` shows only your commits before pushing.
- **`viva-torch` is a BASE dependency** (not the `[surrogate]` extra); `[surrogate]` is kept as an empty back-compat alias; `uv.lock` is re-locked in the same task that moves it (CI runs `uv lock --check`).
- **The new router is import-guarded** and never imported by `viva_simulation_core/__init__.py` or the CLI — a base (`[api]`-less) install still imports fine. Mirror the existing `pytest.importorskip("fastapi")` test pattern.
- **Reuse, don't reinvent:** build every composite against a core from `build_core()`; use `process_bigraph.Composite` + `Visualization`. The facade adds no simulation logic.
- **Reuse the twin API's auth:** the workflow router is `APIRouter(dependencies=[Depends(require_key)])`, mounted in `create_app`; bearer keys from the same `ApiConfig`.
- **In-process, single-worker execution** (design §7.1): the facade runs one-shot `Composite.run()` in-process (NOT the twin API's worker-subprocess isolation). The in-process module/workflow registries require `uvicorn --workers 1` (the app already enforces single-worker).

## Review Focus

- **`GET /modules` must never 500 on any of the 149 registered modules.** Some processes' `inputs()`/`outputs()` need an instance or raise; port introspection is best-effort and each module's failure is caught so the list always returns. → Task 3 test lists *every* module without error.
- **Unknown/misspelled module in a workflow or module-call → clean 422, not a 500** bubbling up from `Composite` construction. → Tasks 4 & 5 tests assert 422 on an unknown module name.
- **Concurrent composite runs touching roadrunner (Tellurium/BioGears) can hit roadrunner thread-safety.** The app is single-worker, but FastAPI runs sync endpoints in a threadpool; a per-process lock must serialize `Composite.run()` in the facade. → Task 5 holds a module-level run lock; a concurrency test runs two workflow runs and both return correct results.
- **`NeuralProcess` refuses any interval ≠ its trained `spec.timestep`.** A workflow including the surrogate must run at that timestep or fail with a clean 422. → Task 6 slow test uses the trained timestep; a fast unit asserts the mismatch path returns 422, not 500.
- **Emptying the `[surrogate]` extra must not break anything that references `.[surrogate]`** (CI, docs, scripts). → Task 1 greps for `[surrogate]` usages and keeps the alias resolvable.

---

## Task 1: Move viva-torch to base deps + register NeuralProcess

**Files:**
- Modify: `pyproject.toml` (dependencies + optional-dependencies)
- Modify: `uv.lock` (re-lock)
- Modify: `viva_simulation_core/core.py:137` (`build_core`)
- Test: `tests/api/test_workflow_registry.py`

**Interfaces:**
- Consumes: `viva_torch.register_neural_process(core, name="NeuralProcess") -> bool`; `viva_simulation_core.core.build_core`.
- Produces: `build_core()` returns a core whose `list_processes()` contains `"NeuralProcess"`.

- [ ] **Step 1: Write the failing test** — `tests/api/test_workflow_registry.py`:
```python
def test_neural_process_registered_in_core():
    from viva_simulation_core.core import build_core
    assert "NeuralProcess" in build_core().list_processes()

def test_surrogate_extra_still_resolvable():
    # back-compat alias: importing viva_torch must work (base dep now)
    import importlib; assert importlib.util.find_spec("viva_torch") is not None
```
- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/api/test_workflow_registry.py -v` → FAIL (`NeuralProcess` not registered).
- [ ] **Step 3: Move the dependency** in `pyproject.toml`: add `"viva-torch"` to base `dependencies` (next to `viva-biogears`); change `surrogate = ["viva-torch"]` to `surrogate = []` with a comment `# viva-torch is now a base dep; kept as an empty back-compat alias`. Grep first: `grep -rn '\[surrogate\]' . --include=*.toml --include=*.yml --include=*.md | grep -v '\.venv'` — keep every such reference resolvable.
- [ ] **Step 4: Register NeuralProcess in `build_core`** — in `core.py`, after `_chain_installed_module_cores(core)`, best-effort call:
```python
try:
    from viva_torch import register_neural_process
    register_neural_process(core)
except Exception:  # noqa: BLE001 — torch optional at runtime; never break build_core
    pass
```
- [ ] **Step 5: Re-lock** — `uv lock` (then `uv lock --check` must pass). Install: `uv pip install -e '.[dev,api]'`.
- [ ] **Step 6: Run tests to verify they pass** — `uv run pytest tests/api/test_workflow_registry.py -v` → PASS.
- [ ] **Step 7: Commit** — `git add pyproject.toml uv.lock viva_simulation_core/core.py tests/api/test_workflow_registry.py && git commit -m "feat(deps): viva-torch to base dep; register NeuralProcess in build_core"`

---

## Task 2: Workflow pydantic models

**Files:**
- Create: `viva_simulation_core/api/workflow_models.py`
- Test: `tests/api/test_workflow_schemas.py`

**Interfaces:**
- Produces (pydantic models, consumed by Tasks 3-6):
  - `ModuleCard{name, address, kind, config_schema: dict, input_ports: dict|None, output_ports: dict|None, capability_label}`
  - `ModuleCallRequest{config: dict = {}, inputs: dict = {}, n_steps: int = 1}` / `ModuleCallResponse{outputs: dict, provenance}`
  - `WorkflowNode{id, module, config: dict = {}, interval: float|None}`; `WorkflowWire{from_: str (alias "from"), to: str}`; `WorkflowSpec{nodes: list[WorkflowNode], wires: list[WorkflowWire] = [], emit: list[str] = []}`
  - `WorkflowCreateResponse{workflow_id, graph: dict}`; `WorkflowRunRequest{n_steps: int, overrides: dict = {}}` / `WorkflowRunResponse{observables: dict, provenance}`; `GraphResponse{nodes: list, edges: list}`
  - Reuse existing `Provenance` from `viva_simulation_core/api/models.py` (extend if a `modules: list[str]` field is needed).

- [ ] **Step 1: Write the failing test** — `test_workflow_schemas.py`: a valid `WorkflowSpec` round-trips; `WorkflowWire` accepts the JSON key `"from"` (alias) → `.from_`; a `WorkflowSpec` with an empty `nodes` list raises `ValidationError`.
- [ ] **Step 2: Run to verify it fails** — module not found.
- [ ] **Step 3: Implement the models** in `workflow_models.py` (pydantic v2 `BaseModel`; `WorkflowWire` uses `Field(alias="from")` + `model_config = ConfigDict(populate_by_name=True)`; `WorkflowSpec.nodes` min length 1).
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `git commit -m "feat(api): workflow API pydantic models"`

---

## Task 3: Module introspection + `/modules` endpoints (+ mount the router)

**Files:**
- Create: `viva_simulation_core/api/workflow.py` (introspection helpers + router factory, partial)
- Modify: `viva_simulation_core/api/app.py:63` (`create_app`) — mount the workflow router
- Test: `tests/api/test_workflow_modules.py`

**Interfaces:**
- Consumes: `build_core`, `core.list_processes()`, `core.link_registry` (name → class), `ModuleCard`.
- Produces: `make_workflow_router(core_provider: Callable[[], Core]) -> APIRouter` (dependencies `[Depends(require_key)]`); `list_modules(core) -> list[ModuleCard]`; `module_card(core, name) -> ModuleCard` (raises `KeyError` on unknown). `create_app` gains a cached `build_core()` and calls `app.include_router(make_workflow_router(...))`.

- [ ] **Step 1: Write the failing test** — `test_workflow_modules.py` (mirror `test_app.py` style: `importorskip("fastapi")`, `TestClient(create_app(make_cfg()))`, bearer header):
  - `GET /modules` (200, authed) returns a list whose `name`s include `TelluriumUTCStep`, `MathExpressionStep`, `BioGearsProcess`, `NeuralProcess`; the fqcn/short-name pair is de-duplicated (short name present once).
  - **`GET /modules` returns without error across ALL registered modules** (assert `len(resp.json()) >= 1` and status 200 — the handler must catch per-module introspection failures).
  - `GET /modules/MathExpressionStep` returns a `ModuleCard` with a non-null `config_schema` key (may be `{}`).
  - `GET /modules/NoSuchModule` → 404.
  - `GET /modules` without a key → 401.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement introspection + endpoints.** In `workflow.py`: `list_modules` iterates `core.list_processes()`, drops the dotted fqcn when its short name is also present, and builds each `ModuleCard` via a helper that reads `cls.config_schema` (default `{}`) and **best-effort** `input_ports`/`output_ports` (try instantiating with an empty/default config and calling `inputs()`/`outputs()`; on any exception set them to `None` — never propagate). `capability_label` = `f"{short_name}"` (version suffix if the class exposes one, else bare). Add `GET /modules` and `GET /modules/{name}` to the router. In `app.py`, build `core = build_core()` once in `create_app`, and `app.include_router(make_workflow_router(lambda: core))`.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `git commit -m "feat(api): /modules introspection endpoints on unified app"`

---

## Task 4: Single-module call — `POST /modules/{name}/call`

**Files:**
- Modify: `viva_simulation_core/api/workflow.py` (add the compose-and-run helper + endpoint)
- Test: `tests/api/test_workflow_call.py`

**Interfaces:**
- Consumes: `core`, `ModuleCallRequest`, `process_bigraph.Composite`.
- Produces: `call_module(core, name, config, inputs, n_steps) -> dict (outputs)`; `POST /modules/{name}/call`.

- [ ] **Step 1: Write the failing test** — call `MathExpressionStep` (pure, fast, no roadrunner) with a config + inputs that compute a known expression; assert the returned `outputs` contain the expected value. Assert `POST /modules/NoSuchModule/call` → 422. (Pick the exact `MathExpressionStep` config/expression by reading its `config_schema`/`inputs` in this task.)
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `call_module`** — build a 1-node state dict `{name: {"_type": "process", "address": f"local:{name}", "config": config, "inputs": {...}, "outputs": {...}}}` with the supplied `inputs` seeded as stores (idiom from `viva_torch/evaluate.py:_rollout_surrogate` — `Composite({"state": state}, core=core)`), `comp.run()` `n_steps` times, read outputs from `comp.state`. Map bad config/inputs/unknown-name to `HTTPException(422)`. Add the endpoint under `rate_limit("light")` or similar.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `git commit -m "feat(api): POST /modules/{name}/call (1-node composite run)"`

---

## Task 5: Workflow define / run / graph / delete

**Files:**
- Modify: `viva_simulation_core/api/workflow.py` (WorkflowSession registry, spec→composite, Visualization graph, endpoints + run lock)
- Test: `tests/api/test_workflow_run.py`

**Interfaces:**
- Consumes: `WorkflowSpec`, `WorkflowRunRequest`, `process_bigraph.Composite`, `process_bigraph.Visualization`, `core`.
- Produces: `WorkflowSession{spec, composite_state, created, last_access}`; an in-process `{workflow_id: WorkflowSession}` dict guarded by a `threading.Lock`; `spec_to_composite(spec, core) -> dict`; `workflow_graph(state, core) -> dict (nodes, edges)`; endpoints `POST /workflows`, `POST /workflows/{id}/run`, `GET /workflows/{id}/graph`, `DELETE /workflows/{id}`.

- [ ] **Step 1: Write the failing test** — a **light** composite (no roadrunner needed where possible; if a simulate node is required use `TelluriumUTCStep` on the tiny decay model `S1 -> S2; k*S1; k=0.1; S1=10; S2=0` at a short horizon): `POST /workflows` with a 2-node spec (simulate → `MathExpressionStep`) → 200 `{workflow_id, graph}` where `graph.nodes` has 2 entries and `graph.edges` has the wire. `POST /workflows/{id}/run {n_steps}` → 200 with `observables` containing the derived value. `GET /workflows/{id}/graph` → nodes+edges. `POST /workflows` with an unknown module → 422. `GET /workflows/UNKNOWN/graph` → 404. A concurrency test: two sequential-in-threads runs of the same workflow both return correct observables (the run lock serializes them).
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** `spec_to_composite` translates `WorkflowSpec` nodes to `{_type: process, address: local:<module>, config, interval}` and `wires` to shared store paths between a producer's output and a consumer's input; validate every `node.module` against `core.list_processes()` → 422 if missing. `POST /workflows` resolves + builds the `Composite` once (store the state), returns `workflow_id` (uuid) + `workflow_graph`. `workflow_graph` uses `process_bigraph.Visualization` (or the composite's node/edge structure) to emit `{nodes:[{id,module,label}], edges:[{from,to,label}]}`. `run` acquires the module-level `threading.Lock`, `Composite({"state": state}, core).run(n_steps)`, reads `emit`/output stores into `observables`. TTL/LRU optional; a `max_workflows` cap evicting LRU is enough. Map unknown id → 404.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `git commit -m "feat(api): workflow define/run/graph/delete endpoints"`

---

## Task 6: The tiered worked workflow + slow e2e

**Files:**
- Create: `viva_simulation_core/api/workflow_examples.py` (importable spec builder + surrogate trainer)
- Test: `tests/api/test_workflow_tiered.py` (`@pytest.mark.slow`)

**Interfaces:**
- Consumes: `surrogate_curate.train_and_curate(source, model_format, observables, directory, timestep_s, ...)`, `viva_torch.neural_process_node`, the Task-2 models.
- Produces: `tiered_workflow_spec(*, surrogate_dir: str|None) -> WorkflowSpec` (BioGears + Tellurium + optional `NeuralProcess` → `MathExpressionStep`); `train_demo_surrogate(directory) -> CurationResult` (thin wrapper over `train_and_curate` on the small Tellurium model at its timestep).

- [ ] **Step 1: Write the failing slow test** — `@pytest.mark.slow`: `train_demo_surrogate(tmp)`, then `POST /workflows` with `tiered_workflow_spec(surrogate_dir=tmp)`, then `run` at a **short horizon** using the surrogate's trained timestep; assert 200 and that `observables` carry the comparison output and the surrogate ran at its trained interval (no 422). Also a **fast** unit: running the tiered workflow at an interval ≠ the surrogate timestep returns 422 (clean), not 500.
- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/api/test_workflow_tiered.py -v -m slow`.
- [ ] **Step 3: Implement `workflow_examples.py`** — build the spec from real module names; `train_demo_surrogate` calls `train_and_curate` (tiny: few trajectories/steps/epochs) and returns the dir + spec so `tiered_workflow_spec` can emit the `NeuralProcess` node via `neural_process_node`. Ensure the workflow's run interval = the surrogate `spec.timestep`.
- [ ] **Step 4: Run to verify it passes** (slow + fast).
- [ ] **Step 5: Commit** — `git commit -m "feat(api): tiered BioGears+Tellurium+surrogate worked workflow + slow e2e"`

---

## Task 7: Narrated demo script

**Files:**
- Create: `examples/vivarium_workflow_demo.py`
- Test: (smoke) run it; no pytest file — it is an example, but it must execute cleanly.

**Interfaces:**
- Consumes: `create_app`, `TestClient`, `workflow_examples`.

- [ ] **Step 1: Write the demo** — same narrated `TestClient` style as `examples/digital_twin_demo.py` (docstring with the run command; real printed requests/responses). Walk all four verbs: `GET /modules` (show the four real modules) → `POST /modules/MathExpressionStep/call` → `POST /workflows` with the tiered spec (train the tiny surrogate first) → `POST /workflows/{id}/run` (short horizon) → `GET /workflows/{id}/graph` (print the node/edge graph). End pointing at `docs/workflow-api.md`.
- [ ] **Step 2: Run it** — `uv run python examples/vivarium_workflow_demo.py` → exits 0, prints the four verbs end to end.
- [ ] **Step 3: Commit** — `git commit -m "docs(api): narrated unified vivarium-workflow demo"`

---

## Task 8: Docs

**Files:**
- Create: `docs/workflow-api.md`
- Modify: `docs/api.md` (one cross-link line) — optional

**Interfaces:** none.

- [ ] **Step 1: Write `docs/workflow-api.md`** — the four verbs, each with a real `curl` example (mirror `docs/api.md` format: `BASE`, bearer key), the tiered worked-workflow diagram from the spec, and a note that it is mounted on the same app as `/twins`.
- [ ] **Step 2: Verify** the curl snippets match the implemented request/response shapes (read the models).
- [ ] **Step 3: Commit** — `git commit -m "docs(api): unified vivarium-workflow API reference"`

---

## Final

- [ ] Run the whole fast suite: `uv run pytest tests/ -q` (and `-m slow` for Task 6) — green.
- [ ] `uv lock --check` passes.
- [ ] `git log origin/main..HEAD` shows only your commits; branch + HEAD verified.
- [ ] Then use superpowers:finishing-a-development-branch (push + open a PR; **do not merge** — the user approves merges).
