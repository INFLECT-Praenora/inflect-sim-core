# INFLECT Digital-Twin Simulation API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public FastAPI service where a caller registers an Antimony/SBML model once (warm per-twin worker subprocess) and streams targeted, tiered (exact ↔ surrogate) prediction / step / ensemble requests for live digital twins.

**Architecture:** Pure-Python crash-safe API process holds a twin registry; each twin owns a worker subprocess that compiles the model once and holds the warm roadrunner. Exact by default, operator-curated neural surrogate when a checkpoint matches. Phase-1 live plane only; a typed HeavyPlane seam is defined for later sms-api/Ray offload.

**Tech Stack:** Python 3.12, FastAPI + uvicorn + pydantic, viva-tellurium / roadrunner, SQLite (stdlib), `multiprocessing` (spawn). Surrogate tier behind a protocol, real adapter gated on `neural-surrogate-sepsis`.

**Spec:** `docs/superpowers/specs/2026-11-17-digital-twin-api-design.md` (read it — this plan argues from it).

## Global Constraints

- **No AI attribution** in any commit or PR (overrides harness default).
- **Worktree discipline:** work in `simulation-core--digital-twin-api`; verify `git branch --show-current` + `git rev-parse --short HEAD` before every commit/push; push with explicit refspec.
- **`[api]` optional extra** holds `fastapi`, `uvicorn[standard]`, `pydantic`; `httpx` goes under `[dev]`. Re-lock `uv.lock` in the same task (CI runs `uv lock --check`). CI must install `.[api]` or API tests import-skip.
- **`viva_simulation_core/api/` is import-guarded** and never imported by `viva_simulation_core/__init__.py` or the CLI — base install stays lean.
- **Roadrunner lives only in worker subprocesses** (`spawn` start method), never the API process. One model's crash must not take the server down.
- **Surrogates are operator-curated**, loaded from a server-side checkpoint store keyed by model hash. Never `torch.load` caller-supplied bytes.
- **Reject Antimony `import` / SBML external-model/file/URL references** at registration.
- **Per-key twin ownership:** `step`/`predict`/`reset`/`DELETE`/`GET /twins/{id}` require the bearer key that created the twin.
- **Caps are env-configurable** with defaults: model 1 MiB; species+reactions 2000; horizon 24 h; steps/call 10 000; ensemble N 256; observables/call 64; request timeout 30 s; register compile timeout 60 s. Live target < 1 s for non-stiff exact + all surrogate calls.
- **Reproducible:** seeds echoed in provenance (null when unused); `engine_used` always set.
- **Slow tests** (`@pytest.mark.slow`) for the stiff sepsis model; fast tests use a tiny Antimony model.
- **Single-worker only:** a startup guard refuses `uvicorn --workers > 1`.

## Review Focus

- **`predict` must not mutate live `step` state.** Snapshot/restore (`saveState`/`loadState`) must leave the twin's time + state identical after a what-if. *Test in Task 4:* step N times, snapshot `t` + observables, run predict, assert twin state byte-identical after.
- **Roadrunner never in the API process.** A model that faults kills only its worker; the registry marks the twin `crashed` and rebuilds. *Test in Task 5:* kill a twin's worker, assert API stays up and next call rebuilds.
- **Ensembles reuse the compiled model.** No per-member recompile (that defeats "live"). *Test in Task 4:* an ensemble on a warm twin compiles the model at most once per worker, not once per member.
- **Surrogate never served out of envelope.** Params/state outside the checkpoint envelope → `auto` falls back to exact; forced surrogate flags `in_distribution=false`. *Test in Task 8.*
- **Per-key ownership enforced on every twin-scoped route**, not just register. *Test in Task 7 + Task 10:* a second key gets 403 on step/predict/delete/get.
- **Caps return the right status and never hang.** Oversize model→413, bad model/unknown observable→422, over-cap ensemble→503, timeout→504. *Tests in Tasks 2, 8/9, 10.*
- **Restart re-warms to t=0 (ephemeral live state)** — the API must not imply resumed mid-stream state. *Test in Task 6.*
- **Eviction never kills an in-flight request;** single-worker guard present. *Tests in Task 5 + Task 10.*

---

### Task 1: `[api]` extra, package skeleton, import guard, config

**Files:**
- Create: `viva_simulation_core/api/__init__.py`, `viva_simulation_core/api/config.py`
- Modify: `pyproject.toml` (add `[api]` extra + `httpx` to `[dev]`)
- Regenerate: `uv.lock`
- Test: `tests/api/test_import_guard.py`, `tests/api/test_config.py`

**Interfaces:**
- Produces: `HAS_API: bool`, `require_api() -> None` (raises `ImportError` naming `.[api]`), `ApiConfig` dataclass with the Global-Constraints caps as fields + `ApiConfig.from_env() -> ApiConfig`.

- [ ] **Step 1: Write failing guard + config tests.**
```python
def test_require_api_raises_when_fastapi_missing(monkeypatch): ...   # simulate absent fastapi -> ImportError mentions "[api]"
def test_config_defaults(): assert ApiConfig.from_env().max_horizon_s == 24*3600
def test_config_env_override(monkeypatch):
    monkeypatch.setenv("INFLECT_API_MAX_ENSEMBLE_N", "8"); assert ApiConfig.from_env().max_ensemble_n == 8
```
- [ ] **Step 2: Run — expect ImportError (module absent).** `pytest tests/api -q`
- [ ] **Step 3: Implement** `api/__init__.py` (guarded import of fastapi → `HAS_API`/`require_api`, `__all__` from bound names; do NOT import from package `__init__`/CLI) and `api/config.py` (`ApiConfig` + `from_env`, env prefix `INFLECT_API_`, all caps from Global Constraints).
- [ ] **Step 4: Add `[api]` extra + `httpx` dev dep to `pyproject.toml`; `uv lock`; `uv pip install -e '.[api,dev]'`.**
- [ ] **Step 5: Run tests — PASS;** confirm `python -c "import viva_simulation_core"` does not import fastapi.
- [ ] **Step 6: Commit** `feat(api): package skeleton, [api] extra, import guard, config`.

---

### Task 2: Model ingest, validation, observable inference (`modeling.py`)

Pure, server-independent. The registration + worker both use it.

**Files:**
- Create: `viva_simulation_core/api/modeling.py`
- Test: `tests/api/test_modeling.py`

**Interfaces:**
- Consumes: `viva_tellurium` / roadrunner.
- Produces:
  - `model_hash(source: str, model_format: str) -> str` (sha256 of normalized source+format).
  - `reject_unsafe(source: str, model_format: str) -> None` — raises `UnsafeModelError` on Antimony `import`/SBML external-model/file/URL refs.
  - `load_engine(source: str, model_format: str) -> RoadRunner` — compile via tellurium (`te.loada` antimony / `te.loadSBMLModel` sbml).
  - `infer_observables(rr) -> dict[str, str]` (non-constant globals + assignment-rule targets + floating species → `{id: id}`).
  - `validate_observables(rr, requested: list[str] | dict[str,str]) -> dict[str,str]` — raise `UnknownObservableError` (→ 422) on ids not in `rr`.
  - `tunable_parameters(rr) -> list[str]` (non-constant globals).

- [ ] **Step 1: Write failing tests** on a tiny Antimony string (`S1 -> S2; k*S1; k=0.1; S1=10; S2=0`):
```python
def test_load_and_infer(): rr = load_engine(ANTIMONY,"antimony"); assert {"S1","S2"} <= set(infer_observables(rr))
def test_validate_unknown_raises(): with pytest.raises(UnknownObservableError): validate_observables(rr, ["nope"])
def test_reject_import(): with pytest.raises(UnsafeModelError): reject_unsafe("import 'x.txt'\n"+ANTIMONY, "antimony")
def test_hash_stable_and_distinct(): assert model_hash(A,"antimony")==model_hash(A,"antimony") != model_hash(A+" ","antimony")
```
- [ ] **Step 2: Run — FAIL (module absent).**
- [ ] **Step 3: Implement `modeling.py`** with the signatures above; selection list for observables must include parameters, not only species.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): model ingest, validation, observable inference`.

---

### Task 3: Request/response schemas (`models.py`)

**Files:**
- Create: `viva_simulation_core/api/models.py`
- Test: `tests/api/test_schemas.py`

**Interfaces:**
- Produces pydantic models: `RegisterRequest{model, model_format, metadata?, observables?, defaults?}`, `RegisterResponse{twin_id, observables[], parameters[], engines_available[]}`, `PredictRequest{observables[], horizon_s, interval_s, initial_state?, parameters?, ensemble?, engine="auto", return_="trajectory|endpoint|stats"}`, `EnsembleSpec{members?, draws?, n?, stochastic?, seed?, quantiles?, per_member?}`, `StepRequest{dt_s?, parameters?, species?, observables?, seq?}`, `Provenance{engine_used, seed, integrator, wall_ms, n_steps, lib_versions}`, `PredictResponse`, `StepResponse`, `ErrorResponse`.

- [ ] **Step 1: Write failing tests:** defaults (`engine=="auto"`), enum rejection (`return_="bogus"` → ValidationError), ensemble cross-field (both `members` and `draws` allowed; `n` required when `draws` set), cap-independent shape round-trip.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `models.py`** (pydantic v2; `model_format`/`engine`/`return_` as `Literal`s).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): request/response schemas`.

---

### Task 4: Twin worker subprocess (`worker.py`)

The engine core. Runs in a `spawn`ed process; compiles once; serves commands.

**Files:**
- Create: `viva_simulation_core/api/worker.py`
- Test: `tests/api/test_worker.py`

**Interfaces:**
- Consumes: `modeling.load_engine`, schemas.
- Produces:
  - A worker entrypoint `run_worker(source, model_format, observables, conn)` (top-level, `spawn`-picklable) that loops on commands.
  - A parent-side handle `TwinWorker` with methods `start()`, `step(**fields)->StepResult`, `predict(**fields)->PredictResult`, `ensemble(**fields)->EnsembleStats`, `reset()`, `close()`, `is_alive()`, and a test affordance `snapshot_probe()->bytes` (current `saveState` blob, for the non-mutation assertion). Methods accept the schema fields as kwargs. Each call is one IPC round-trip with a timeout; on timeout it kills the worker and raises `WorkerTimeout`.
  - Compile happens exactly once at `start()`.

- [ ] **Step 1: Write failing tests** (spawn a real worker on the tiny model):
```python
def test_step_advances_and_persists(): w=TwinWorker(...); w.start(); t1=w.step(dt_s=1).t; assert w.step(dt_s=1).t > t1
def test_step_param_persists(): w.step(parameters={"k":0.5}); assert w.step().provenance... # k stays 0.5 next step
def test_predict_non_mutating(): w.step(dt_s=5); s=w.snapshot_probe(); w.predict(horizon_s=100, observables=["S2"]); assert w.snapshot_probe()==s
def test_ensemble_stats_shape(): r=w.ensemble(draws={"k":{"dist":"uniform","lo":.05,"hi":.2}}, n=4, observables=["S2"]); assert {"mean","p05","p95"} <= r.keys()
def test_ensemble_compiles_once(monkeypatch): # count load_engine calls across a warm ensemble; assert <= worker_pool_size, not == n
def test_seq_idempotent(): a=w.step(seq=7); b=w.step(seq=7); assert a==b   # retried step is a no-op
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `worker.py`.** `predict` uses roadrunner `saveState`/`loadState` to snapshot before and restore after (preserves time + event state). `step` applies `parameters` persistently and `species` as a one-step overwrite. Ensemble reuses the compiled engine (reset+override per member; Gillespie replicates vary seed); extra workers (if N > pool size) each compile once. Use `multiprocessing.get_context("spawn")`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): per-twin worker subprocess (compile-once, snapshot-restore predict)`.

---

### Task 5: Twin registry (`registry.py`)

**Files:**
- Create: `viva_simulation_core/api/registry.py`
- Test: `tests/api/test_registry.py`

**Interfaces:**
- Consumes: `TwinWorker`, `modeling`, `ApiConfig`.
- Produces: `TwinSession{twin_id, worker, source, model_format, observables, owner_key, model_hash, created, last_access, lock}`; `TwinRegistry` with `register(source, fmt, owner, observables?) -> TwinSession`, `get(twin_id, owner) -> TwinSession` (raises `NotOwner`→403 / `UnknownTwin`→404), `evict(twin_id)`, TTL+LRU sweep under capacity cap, `rebuild_if_crashed(session)`. The per-session lock serializes **both** step and predict.

- [ ] **Step 1: Write failing tests:**
```python
def test_register_returns_warm_twin(): s=reg.register(ANTIMONY,"antimony","k1"); assert s.worker.is_alive()
def test_get_wrong_owner_403(): with pytest.raises(NotOwner): reg.get(s.twin_id, "k2")
def test_lru_eviction_at_capacity(): # cap=2, register 3 -> oldest evicted (worker closed)
def test_ttl_eviction(): # with injected clock past ttl -> swept
def test_evict_waits_on_inflight_lock(): # holding session.lock, evict blocks/refuses until released
def test_crashed_worker_rebuilds(): s.worker._proc.kill(); reg.rebuild_if_crashed(s); assert s.worker.is_alive()  # resets to t=0
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `registry.py`** (injectable clock for tests; capacity + ttl from `ApiConfig`).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): twin registry with TTL/LRU eviction, ownership, crash-rebuild`.

---

### Task 6: Persistence + lazy re-warm (`persistence.py`)

**Files:**
- Create: `viva_simulation_core/api/persistence.py`
- Modify: `viva_simulation_core/api/registry.py` (call the store on register; `load_all` + lazy re-warm hook)
- Test: `tests/api/test_persistence.py`

**Interfaces:**
- Produces: `TwinStore(db_path)` with `save(session_meta)`, `delete(twin_id)`, `load_all() -> list[TwinMeta]` (source+format+metadata+owner+hash+twin_id). Registry re-warms a persisted twin **lazily on first call**, resetting to t=0.

- [ ] **Step 1: Write failing tests:**
```python
def test_save_load_roundtrip(tmp_path): st=TwinStore(tmp_path/"t.db"); st.save(meta); assert st.load_all()[0].owner_key==meta.owner_key
def test_restart_rewarms_to_t0(tmp_path): # new registry from same db, first call -> worker alive, t==0 (ephemeral live state)
def test_delete_removes(tmp_path): ...
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `persistence.py`** (stdlib `sqlite3`, WAL) and wire into the registry (save on register, delete on evict-by-request only — TTL/LRU eviction keeps the row for lazy re-warm).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): sqlite persistence + lazy re-warm (ephemeral live state)`.

---

### Task 7: Auth + rate limit (`auth.py`)

**Files:**
- Create: `viva_simulation_core/api/auth.py`
- Test: `tests/api/test_auth.py`

**Interfaces:**
- Produces: `require_key` (FastAPI dependency; bearer, constant-time compare vs env keys; 401 on miss), `TokenBucket` limiter (per key, weighted: register/ensemble cost > step; 429 + `Retry-After`), `is_exempt(path)` (`/healthz`,`/version`). Per-key identity is returned for the ownership check in the registry.

- [ ] **Step 1: Write failing tests** (pure, no server): valid/invalid key; bucket trips after N weighted tokens and reports `Retry-After`; register weighted heavier than step; exempt paths skip auth.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `auth.py`** (injectable clock; keys + rates from `ApiConfig`/env).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): bearer auth + per-key token-bucket rate limit`.

---

### Task 8: Surrogate backend + tier selector (`surrogate_backend.py`)

**Files:**
- Create: `viva_simulation_core/api/surrogate_backend.py`
- Test: `tests/api/test_tier.py`

**Interfaces:**
- Consumes: `model_hash`.
- Produces:
  - `SurrogateBackend` protocol: `lookup(model_hash) -> SurrogateEntry | None`, `predict(entry, request) -> dict`.
  - `SurrogateEntry{checkpoint_ref, targets, horizon_s, param_ranges, state_ranges, val_error}` + sidecar loader.
  - `CheckpointStore(dir)` — operator-curated, model-hash-keyed (loads sidecar JSON; defers actual torch load to the gated adapter).
  - `FakeBackend` for tests.
  - `select_engine(request, entry, requested_engine) -> ("exact"|"surrogate", in_distribution: bool)` — surrogate iff observables ⊆ targets ∧ horizon ≤ trained ∧ params/state ∈ ranges; else exact; forced-missing → `SurrogateUnavailable` (409).

- [ ] **Step 1: Write failing tests:**
```python
def test_auto_in_envelope_picks_surrogate(): eng,ind=select_engine(req_in, entry,"auto"); assert eng=="surrogate" and ind
def test_auto_out_of_range_falls_back_exact(): eng,_=select_engine(req_bigk, entry,"auto"); assert eng=="exact"
def test_force_surrogate_out_flags_not_in_dist(): eng,ind=select_engine(req_bigk, entry,"surrogate"); assert eng=="surrogate" and not ind
def test_force_missing_raises_409(): with pytest.raises(SurrogateUnavailable): select_engine(req, None, "surrogate")
def test_fake_backend_predict(): ...
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `surrogate_backend.py`** (real `neural-surrogate-sepsis` adapter left as a documented TODO behind the protocol; phase-1 default backend is `CheckpointStore` over an empty/operator dir → `NullBackend` when unset).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): surrogate backend protocol + tier selector (envelope-gated)`.

---

### Task 9: HeavyPlane seam (`heavy_plane.py`)

**Files:**
- Create: `viva_simulation_core/api/heavy_plane.py`
- Test: `tests/api/test_heavy_plane.py`

**Interfaces:**
- Produces: `HeavyPlane` protocol (`submit_ensemble(twin, request) -> HeavyResult`), `LocalHeavyPlane` (phase-1: runs it locally up to a hard ceiling), `OverCapacity` exception mapped to **503 `use batch`** when ensemble N exceeds the live cap and no remote plane is configured.

- [ ] **Step 1: Write failing tests:** under-cap → runs locally; over-cap with no remote → `OverCapacity`; protocol is satisfied by `LocalHeavyPlane`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `heavy_plane.py`** (sms-api/Ray impl is a documented TODO; the live plane depends only on the protocol).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(api): HeavyPlane seam + over-cap 503`.

---

### Task 10: FastAPI app wiring + server entrypoint (`app.py`, `__main__.py`)

Integration task — wires every prior piece behind HTTP.

**Files:**
- Create: `viva_simulation_core/api/app.py`, `viva_simulation_core/api/__main__.py`
- Test: `tests/api/test_app.py`

**Interfaces:**
- Consumes: registry, auth, tier selector, heavy plane, persistence, schemas, config.
- Produces: `create_app(config, registry, backend, ...) -> FastAPI`; routes `POST /twins`, `GET /twins/{id}`, `POST /twins/{id}/predict`, `POST /twins/{id}/step`, `POST /twins/{id}/reset`, `DELETE /twins/{id}`, `GET /healthz`, `GET /version`. `__main__` launches uvicorn single-worker with a guard that refuses `--workers > 1`.

- [ ] **Step 1: Write failing `TestClient` tests** (tiny Antimony model, caps shrunk via a test config):
```python
def test_register_predict_step_flow(): # 200s; engine_used=="exact"; step advances t
def test_targeted_observables_only(): # response carries only requested observables
def test_ensemble_stats_endpoint(): # stats shape mean/p05/p95
def test_owner_isolation_403(): # key2 step/predict/delete/get on key1's twin -> 403
def test_unknown_twin_404(): ...
def test_oversize_model_413(): ...
def test_unknown_observable_422(): ...
def test_over_cap_ensemble_503(): ...
def test_request_timeout_504(): # a deliberately slow step past the timeout -> 504, server still up
def test_healthz_version_exempt_200(): ...
def test_multi_worker_guard_refused(): # the single-worker guard raises on workers>1
```
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `app.py`** (dependency-wire; map each exception to its Global-Constraints status; `GET /twins/{id}` and all twin-scoped routes go through `registry.get(twin_id, owner)`; ensemble over cap → HeavyPlane → 503) and `__main__.py` (uvicorn single-worker guard).
- [ ] **Step 4: Run — PASS** (`pytest tests/api -q`).
- [ ] **Step 5: Commit** `feat(api): FastAPI app, routes, error contract, single-worker entrypoint`.

---

### Task 11: Slow sepsis end-to-end + usage doc

**Files:**
- Create: `tests/api/test_sepsis_e2e.py` (`@pytest.mark.slow`), `docs/api.md`
- Test: the slow test itself.

**Interfaces:**
- Consumes: the running app via `TestClient`, the vendored sepsis SBML (`workspace/investigations/biogears-sbml-sepsis/sources/biogears_sepsis.xml`).

- [ ] **Step 1: Write the slow test:** register the sepsis SBML (as `model_format="sbml"` source string), `step` and `predict` a few short intervals at `severity`, request `MAP`/`CO`; assert finite values and `engine_used=="exact"` (no surrogate present). Horizon ≤ 60 s simulated to stay bounded.
- [ ] **Step 2: Run** `pytest -m slow tests/api/test_sepsis_e2e.py -q` — PASS.
- [ ] **Step 3: Write `docs/api.md`** — register→step→predict→ensemble with `curl` examples, the tier rule, the caps table, and the operator-curated-surrogate note.
- [ ] **Step 4: Commit** `test(api): slow sepsis end-to-end + usage doc`.

---

## Notes for the executor

- Fast tests (Tasks 1–10) must pass with `uv pip install -e '.[api,dev]'` and run in the default (non-slow) set; the CI workflow must install `.[api]`.
- The real surrogate adapter (Task 8 TODO) and the sms-api/Ray HeavyPlane (Task 9 TODO) are intentionally deferred; their seams are what phase 1 delivers.
- Public hosting is a deploy-time decision (out of scope here); keep everything 12-factor (config from env).
