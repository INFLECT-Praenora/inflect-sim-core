# INFLECT Digital-Twin Simulation API — Design

**Status:** approved-pending-review
**Date:** 2026-11-17
**Repo:** `INFLECT-Praenora/inflect-sim-core` (`simulation-core`)
**Author:** Eran Agmon

---

## 1. Intent and why

INFLECT needs a **public HTTP API that serves live digital-twin simulation**. A
caller registers an **Antimony (or SBML) model plus metadata once**, then streams
**fast, targeted prediction requests** — including **ensembles** — against a
**warm, persistent** model instance. The defining requirement is *live
performance*: a digital twin is stepped forward as new data arrives, so the
latency budget is sub-second-to-seconds, not a batch job.

This collides with mechanistic stiffness. Roadrunner (compiled C) is fast for
small/non-stiff models — milliseconds — but stiff models like INFLECT's own
sepsis core cost ~1.8 h per 48 h simulated, and an ensemble multiplies that. Exact
integration simply cannot be "live" for that class. The resolution is a **tiered
engine**: exact roadrunner by default; a **millisecond neural surrogate**
(`viva-torch`, built in the in-flight `neural-surrogate-sepsis` work) when a
trained checkpoint exists for the twin. Every response says which engine answered.

Today the repo is a process-bigraph library + CLI + read-only workbench, with **no
HTTP server**. This design adds one.

## 2. Goals

- **Register-once / stream-many.** `POST` an Antimony/SBML model + metadata → a
  `twin_id`. The server builds and **warms** a roadrunner instance and keeps it
  alive; subsequent calls never reload or recompile the model.
- **Targeted prediction.** The caller names specific observables over specific
  horizon(s); the API returns **only those** (trajectory, endpoint, or ensemble
  stats) — never a full-state dump.
- **Ensembles.** Per-request parameter sweeps and/or stochastic (Gillespie)
  replicates, returned as aggregate stats (mean / quantiles) and optionally
  per-member.
- **Tiered engine, reported.** `engine=auto` serves the surrogate when a validated
  checkpoint exists and the request is inside its trained envelope, else exact.
  Every response carries `engine_used` + provenance (seed, integrator, wall-ms,
  n-steps).
- **Live stepping.** A stateful `step` primitive advances the warm instance by one
  interval with new inputs — the live-twin hot path — distinct from the
  non-mutating what-if `predict`.
- **Bounded latency, never hang.** Input caps (model size, horizon, step count,
  ensemble N) + per-request timeout guarantee a stiff model is rejected or routed
  to the surrogate, never silently blocked for minutes.
- **Staged (plane C).** Phase 1 builds the **live in-process plane** in this repo
  (FastAPI). A typed **heavy-plane seam** is defined (not implemented) for phase 2
  offload of large ensembles and surrogate-training sweeps to sms-api / Ray.

## 3. Non-goals

- **Not** building the GovCloud/Ray heavy plane now — phase 2 defines only the
  interface the live plane calls.
- **Not** training surrogates inside the live API. Training is offline / heavy
  plane; the live API only *serves* a checkpoint if one is registered.
- **Not** a multi-node HA cluster in phase 1. The warm registry is in-process;
  horizontal scaling (sticky sessions) is a later concern.
- **Not** arbitrary-code execution. Input is Antimony/SBML only — parsed and
  compiled by libantimony/roadrunner, never `eval`'d. No user Python.

## 4. Anti-goals (failure even if it "works")

- A live call that silently blocks for minutes on a stiff model.
- Returning full state when the caller asked for three observables.
- Coupling the *public* API to GovCloud's restricted access so external callers
  can't reach it.
- A surrogate served outside its validated envelope with no in-distribution flag.

## 5. Constraints

- **No AI attribution** in any commit or PR (overrides harness default).
- **Worktree discipline** — work in `simulation-core--digital-twin-api`, verify
  branch+HEAD before every commit/push, explicit-refspec push.
- **FastAPI + uvicorn**, added behind an **`[api]` optional extra** (same pattern as
  the existing `[cli]` extra in `pyproject.toml`) so the base install and CI stay
  lean; the API package is import-guarded.
- **Reuse existing machinery:** `viva_tellurium.make_tellurium_document` (accepts
  an Antimony *source string* + `model_format`, `integrator` cvode|gillespie,
  `seed`, `parameter_overrides`, `species_overrides`, `interval`), `build_core`,
  and process-bigraph `Composite`. The surrogate tier reuses
  `viva_simulation_core/surrogate` + the `neural-surrogate-sepsis` `NeuralProcess`.
- **Roadrunner is not thread-safe** and has a known multi-instance segfault risk,
  so every roadrunner instance lives in a **worker subprocess**, never the API
  process (§7.2) — this isolates crashes and sidesteps the threadpool thread-safety
  problem.
- **Public-surface hardening:** API key (bearer) + rate limit + the input caps
  above. Antimony size and complexity are bounded at registration.
- **Reproducibility:** seeds echoed back; engine + provenance in every response.
- **Host-agnostic:** 12-factor config (bind/port/caps/keys via env) so the exact
  public host is a deploy-time choice, not baked in.

## 6. Approaches considered

- **A — Standalone FastAPI in-process.** Warm twins in the server process; local
  parallel ensembles; surrogate tier loads a per-twin checkpoint. Lowest live
  latency, self-contained; but stateful → scaling needs sticky sessions.
- **B — Build on sms-api / GovCloud Ray.** Reuses a deployed authenticated
  backend and cluster parallelism; but Ray adds a per-call scheduling+serialization
  hop (fights sub-second stepping) and GovCloud is access-restricted (fights
  *public*). Right for big offline ensembles, wrong for the live hot path.
- **C — Two-plane, staged (CHOSEN).** Live plane = A (in-process warm twins, exact
  + surrogate, light ensembles) for everything latency-critical; heavy plane =
  delegate to sms-api/Ray for large ensembles and surrogate-training sweeps. The
  live path never touches Ray; the cluster does what it is good at. Built
  incrementally: phase 1 = live plane with a clean seam; phase 2 = the offload.

**Chosen: C, staged.** It keeps live latency on the in-process path, offloads the
expensive stiff sweeps honestly, and avoids coupling a public API to restricted
GovCloud access.

## 7. The design (phase 1 — live plane)

New package `viva_simulation_core/api/` (FastAPI app), served by the existing
container. Pydantic models for every request/response. All deps under `[api]`.

### 7.1 Endpoints

| Method + path | Purpose |
|---|---|
| `POST /twins` | Register: `{model, model_format, metadata, observables?, defaults?}` → builds + warms a roadrunner instance; returns `{twin_id, observables[], parameters[], engines_available}`. |
| `GET /twins/{id}` | Twin metadata: observables, parameters, engines available, warm-state summary, last-access. |
| `POST /twins/{id}/predict` | Non-mutating what-if forward run from a given state. Body: `{observables[], horizon_s, interval_s, initial_state?, parameters?, ensemble?, engine=auto, return=trajectory\|endpoint\|stats}`. Returns `{engine_used, observables{...}, ensemble_stats?, provenance}`. |
| `POST /twins/{id}/step` | **Stateful** live step: advance the warm instance one interval with new inputs; return requested observables at the new time. The live-twin hot path. |
| `DELETE /twins/{id}` | Evict the warm instance. |
| `GET /healthz`, `GET /version` | Liveness + build info. |

### 7.2 Twin registry and per-twin worker process

The API process is **pure-Python and crash-safe**: it holds no roadrunner. Each
twin owns a **dedicated worker subprocess** that loads and compiles the model
**once** and keeps the warm roadrunner instance. This single choice resolves three
problems the in-process design had — roadrunner is not thread-safe (FastAPI sync
endpoints run in a threadpool), roadrunner has a known multi-instance segfault
risk, and a segfault in a warm instance would otherwise take down the whole server
and every twin. With per-twin workers, a crash is isolated to one worker; the API
marks that twin `crashed` and can rebuild it from stored source. The live
`step`/`predict` dispatch is one local IPC round-trip (sub-ms to low-ms —
negligible against the sub-second-to-seconds budget).

The in-process registry is `{twin_id: TwinSession}`, where a `TwinSession` holds
the worker handle (not the roadrunner itself), the model source + format, the
observable→variable map, an optional surrogate-checkpoint reference, created /
last-access timestamps, an owner key (§8.2), and a per-session lock that serializes
**both** `step` and `predict` against the twin's worker. Eviction by **TTL + LRU**
under a capacity cap; eviction waits on (or refuses while held) the session lock so
it never kills an in-flight request. Registration is the only place a model is
compiled.

### 7.3 Warm execution core

- **Exact `step`.** Advance the twin's warm instance one `dt_s` with new inputs and
  **retain** state — the stateful live primitive.
- **Exact `predict`.** Non-mutating what-if. The worker **snapshots** roadrunner
  state (`saveState`/`loadState`, which preserves time + event state), runs the
  forward integration from the supplied `initial_state` (or, if omitted, the twin's
  **current stepped state**), extracts the targets, then **restores** the snapshot
  so the live `step` timeline is untouched.
- **Ensemble.** Runs **inside the twin's worker(s)** against the already-compiled
  model — never a fresh compile per member (that would defeat "live"). Stochastic
  (Gillespie) replicates re-run the one compiled instance with different seeds;
  parameter draws reuse the compiled model via `reset` + overrides. When N exceeds
  a per-twin pool size, extra workers are forked that **reuse the parsed model**
  (compile-once per worker, not per member). `spawn` start method; each worker is
  killable on the per-request timeout. Aggregate to mean/quantiles. N above the
  live cap → phase-2 heavy-plane seam (503 `use batch`).
- **Surrogate.** Autoregressive `NeuralProcess` rollout — milliseconds. Used only
  within the trained envelope (§7.4).

### 7.4 Tier selector

`engine=auto` → surrogate iff a checkpoint is registered for the twin **and** the
request's observables ⊆ trained targets **and** horizon ≤ trained horizon **and**
every supplied parameter / `initial_state` value is inside the checkpoint's stored
envelope ranges (§9); otherwise it **falls back to exact** (never serves the
surrogate out of distribution — §4). `engine=exact|surrogate` forces one; forcing a
missing surrogate → 409; forcing a surrogate out of envelope → the result is
returned with `in_distribution=false`. Response always sets `engine_used`;
surrogate responses carry `in_distribution`. The checkpoint envelope schema
(trained targets, horizon, per-parameter and per-state ranges, validation-error
threshold) is defined in §9.

### 7.5 Result contract

Targeted and small: only requested observables, as `trajectory` (series),
`endpoint` (final values), or `stats` (ensemble aggregates). Typed JSON, bounded
by the step/observable/ensemble caps.

### 7.6 Performance levers (how "live" is met)

Warm-instance reuse (no per-call reload) is the primary lever; compiled model
cached per twin; subprocess-isolated capped ensemble parallelism; surrogate path
for stiff/hot twins; input caps + request timeout bound worst-case latency.

### 7.7 Phase-2 seam (designed, not built)

A `HeavyPlane` protocol the live plane calls for (a) ensembles above the local cap
and (b) surrogate-training sweeps. Phase 1 ships an in-process/local
implementation (or a 503 "exceeds live limits, use batch" for oversize); the
sms-api/Ray implementation lands in phase 2. The live plane depends only on the
interface.

## 8. Decisions (settled with the product owner 2026-11-17)

1. **Surrogate checkpoints are operator-curated only.** The deployment mounts a
   checkpoint store keyed by model hash; a twin auto-binds a surrogate when its
   model hash matches one. Callers never upload checkpoints (no `torch.load` of
   untrusted bytes). Caller-upload is a possible later admin-authenticated feature.
   This is the bridge to the `neural-surrogate-sepsis` work: that branch produces
   checkpoints + the envelope sidecar (§9) that an operator drops into the store.
2. **Per-key twin ownership.** Each twin records the bearer key that created it;
   `step` / `predict` / `step` / `DELETE` / `GET /twins/{id}` require the owning
   key. Twins are private to their creator.
3. **Persist model, reset live state on restart.** Model source + metadata + owner
   persist (SQLite under a configurable data dir) and re-warm lazily on first call
   after restart. Stepped live state is **ephemeral** — a restart resets twins to
   t=0 and callers replay their inputs.

**Remaining (deploy-time, not build-time): public hosting.** Where the live plane
runs so external callers reach it — a cloud host, not GovCloud (restricted) and not
the Tailscale-private mini. The service is 12-factor host-agnostic, so this is
picked when phase 1 is ready to ship, and does not block the build.

## 9. Settled by the author (builder's-eye calls)

**Sequencing & framework.**
- **Surrogate sequencing:** build the live plane **exact-first** against a
  `SurrogateBackend` protocol tested with a fake; the real adapter wires in when
  `neural-surrogate-sepsis` merges. Does not block phase 1.
- **Framework:** FastAPI + uvicorn + pydantic. Deps under an **`[api]` optional
  extra** (`fastapi`, `uvicorn`, `pydantic`; `httpx` under `[dev]` for `TestClient`).
  `viva_simulation_core/api/` is import-guarded and never imported by `__init__` or
  the CLI, so the base install stays lean. **CI must install `.[api]`** or the API
  tests import-skip and silently never run.
- **Single-worker only:** the in-process registry requires `uvicorn --workers 1`;
  a startup guard refuses `>1`. (Horizontal scaling is a later concern — §3.)

**`step` semantics.** Request `{dt_s?, parameters?: {id: v}, species?: {id: v},
observables?}`. `parameters` **persist from that step forward** (a new measurement
updates a model parameter — the twin-update semantics); `species` **overwrite**
current state for this step. `dt_s` defaults to the twin's registered interval,
capped by the step cap. Returns `{t, observables, provenance}`. An optional
client `seq` (idempotency key) makes a retried step a no-op that re-returns the
prior result (prevents double-advance). A `POST /twins/{id}/reset` returns the twin
to t=0.

**`predict` default state.** With no `initial_state`, predict starts from the
twin's **current stepped state** (what-if from "now"); snapshot/restore keeps it
non-mutating (§7.3).

**Ensemble request spec.** `ensemble: {members?: [{parameters}], draws?: {id:
{dist: uniform|normal|lognormal, ...bounds}}, n?, stochastic?: bool, seed?,
quantiles?: [...], per_member?: bool}`. `members` = explicit sets; `draws` =
sampled (seeded, seed echoed); `stochastic` = Gillespie replicates. `draws` ×
`stochastic` is a cross-product. Default aggregation = mean + p05/p50/p95;
`per_member` opt-in and bounded by the response cap.

**Observables.** `observables` optional at registration. Accepts a list of model
ids or an `{alias: id}` dict; validated against `rr.model` ids at registration
(**422** on unknown). If omitted, inferred as non-constant global parameters +
assignment-rule targets + floating species (sepsis observables are globals /
assignment rules read via roadrunner selections — the selection list must include
parameters, not only species). Registration returns `observables[]` and the
tunable `parameters[]` (non-constant globals).

**Checkpoint envelope schema.** Each operator-curated checkpoint carries a sidecar:
`{model_hash, targets: [...], horizon_s, param_ranges: {id: [lo, hi]},
state_ranges: {id: [lo, hi]}, val_error: {metric, threshold}}`. The tier selector
(§7.4) reads it; `in_distribution` = request inside all ranges.

**Caps (env-configurable; defaults are the product's "live" envelope).** Max
Antimony/SBML size (default 1 MiB); max species+reactions (default 2000); max
horizon (default 24 h); max steps/call (default 10 000); max ensemble N (default
256; above → heavy-plane 503); max observables/call (default 64); per-request
timeout (default 30 s, kills the worker). Registration compile is synchronous with
its own timeout (default 60 s). "Live" target: exact `step`/`predict` on a
non-stiff model and all surrogate calls return **< 1 s**.

**Error contract.** 401 (no/invalid key), 403 (not twin owner), 404 (unknown/evicted
twin), 409 (forced engine unavailable), 413 (model too large), 422 (bad model /
unknown observable / malformed request), 429 (`Retry-After`), 503 (ensemble over
live cap → use batch), 504 (request timeout). A crashed worker → 404-with-rebuild
or 503 on the twin until rebuilt.

**Input security.** Reject Antimony `import` statements and SBML external model
definitions / external file or URL references at registration (they can read local
files or fetch URLs) — Antimony/SBML is data, but these directives are not.

**Auth/rate-limit mechanics (given the §8.2 answer).** Bearer key loaded from env,
constant-time compared; `/healthz` and `/version` exempt. In-process token-bucket
limiter (fine under single-worker), per key, expensive calls (register, ensemble)
cost more tokens than a `step`; 429 + `Retry-After`.

**Dedup.** Re-registering byte-identical source under the same owner returns the
existing `twin_id` rather than a duplicate warm worker.

**Provenance schema.** `{engine_used, seed (null when cvode/unused), integrator,
wall_ms, n_steps, lib_versions: {roadrunner, tellurium}}`.

**Test strategy.** Fast set (CI, `.[api]` installed): `TestClient` over
register/predict/step/ensemble on a tiny Antimony model (e.g. a 2-species decay);
ensemble test N=3 with `spawn`-safe top-level worker fns; auth, caps, eviction, and
concurrent-`step` tests use injected clock + shrunk caps. Surrogate-tier tests use a
fake `SurrogateBackend` (real one gated until the branch merges). Slow set
(`@pytest.mark.slow`): the stiff sepsis model end-to-end at a short horizon (≤ 60 s
simulated).

## 10. What's left to the builder (feeds the plan)

- FastAPI app skeleton + pydantic request/response models + `[api]` extra + import
  guard + single-worker startup guard.
- Per-twin **worker subprocess** (compile-once warm roadrunner) + the API-side
  `TwinSession` registry with TTL/LRU eviction, owner keys, and per-session locking.
- Worker ops: stateful `step` (persist params, overwrite species, `seq`
  idempotency, `reset`); non-mutating `predict` via `saveState`/`loadState`
  snapshot-restore; extract targeted observables via parameter-inclusive selections.
- Capped ensemble over the twin's worker(s) with compile-once reuse + stats
  aggregation; `spawn`; timeout-kill.
- Tier selector + `SurrogateBackend` protocol (envelope check + `in_distribution`,
  fake-tested; real adapter gated on the surrogate branch).
- Registration: Antimony/SBML string ingest, observable inference/validation,
  `import`/external-ref rejection, dedup-by-hash.
- Input validation, caps, timeouts, bearer-key auth + per-key ownership, token-bucket
  rate limit, full error contract.
- The `HeavyPlane` seam interface + local phase-1 implementation (or 503 over cap).
- Tests — fast: register/predict/step/ensemble/auth/caps/eviction on a tiny Antimony
  model; slow (`@pytest.mark.slow`): the stiff sepsis model end-to-end at ≤ 60 s.
