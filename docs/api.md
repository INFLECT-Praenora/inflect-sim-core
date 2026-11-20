# Digital-twin API

HTTP API for registering an SBML or Antimony model as a "twin", stepping its live state, and running
what-if predictions and ensembles. Set `<key>` to one of your configured API keys; the examples assume
`BASE=http://localhost:8000`.

All endpoints except `/healthz` and `/version` need `Authorization: Bearer <key>`.

> The same app also serves the unified **vivarium-workflow API** (`/modules`,
> `/workflows`) — define/call modules and connect/call workflows over
> process-bigraph. See [`workflow-api.md`](workflow-api.md).

## Flow: register, step, predict, ensemble

**Register** a model (source string, `model_format` is `antimony` or `sbml`):

```bash
curl -s $BASE/twins -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"model": "S1 -> S2; k*S1; k=0.1; S1=10; S2=0;", "model_format": "antimony"}'
# {"twin_id": "...", "observables": ["S1","S2"], "parameters": ["k"], "engines_available": ["exact"]}
```

Optional fields: `observables` (list, or name-to-expression map), `defaults`, `metadata`.

**Step** the live state forward (`dt_s` of 0 reads the current state without advancing):

```bash
curl -s $BASE/twins/$TID/step -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"dt_s": 5.0, "parameters": {"k": 0.2}, "observables": ["S1"]}'
# {"t": 5.0, "observables": {"S1": 6.1}, "provenance": {...}}
```

Other step fields: `species` (set species values), `seq`. `POST /twins/$TID/reset` returns the twin
to its initial state.

**Predict** a trajectory from the current live state. Predict never changes the live state.

```bash
curl -s $BASE/twins/$TID/predict -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"observables": ["S1"], "horizon_s": 60, "interval_s": 5, "return": "trajectory"}'
# {"engine_used": "exact", "observables": {"S1": [...]}, "times": [...], "in_distribution": null,
#  "provenance": {"engine_used","seed","integrator","wall_ms","n_steps","lib_versions"}}
```

Request fields: `observables`, `horizon_s`, `interval_s` (required); `initial_state`, `parameters`
(overrides for this prediction only); `engine` (`auto` | `exact` | `surrogate`, default `auto`);
`return` (`trajectory` default, `endpoint`, or `stats`; the JSON key is `return`). `endpoint` and
`stats` collapse each series to its final value.

**Ensemble**: add an `ensemble` object to a predict request:

```bash
curl -s $BASE/twins/$TID/predict -H "Authorization: Bearer <key>" -H 'Content-Type: application/json' \
  -d '{"observables": ["S1"], "horizon_s": 60, "interval_s": 5,
       "ensemble": {"n": 50, "draws": {"k": {"dist": "uniform", "lo": 0.05, "hi": 0.2}},
                    "seed": 1, "quantiles": [0.05, 0.95]}}'
```

Draw distributions (`dist` plus its parameters): `uniform` (`lo`, `hi`), `normal` (`mean`, `sd`), `lognormal` (`mu`, `sigma`), `constant` (`value`). `ensemble` fields: `members` (explicit parameter sets) or `draws` plus `n`, `stochastic`, `seed`,
`quantiles`, `per_member`. The response carries `ensemble_stats` (`mean`, `p05`, `p95`, `quantiles`,
`members`, `n`). Ensembles run on the exact engine only (`engine: "surrogate"` is rejected with 422). An ensemble larger than `MAX_ENSEMBLE_N` returns 503 `over_capacity`.

Other twin endpoints: `GET /twins/{id}`, `DELETE /twins/{id}`, `GET /healthz`, `GET /version`.

## Tier rule

Predictions use the **exact** engine (roadrunner) by default. A **surrogate** is used only when an
operator-curated checkpoint matches the registered model and the request lies inside that
checkpoint's validated envelope. The `engine_used` field in every predict response says which ran.
`engine: "exact"` forces exact; `engine: "surrogate"` returns 409 if none is available.
`engines_available` on the twin lists what exists. Surrogates are loaded from sidecars in
`INFLECT_API_CHECKPOINT_DIR`; with that unset, only exact is available.

## Caps and configuration

All settings are environment variables with prefix `INFLECT_API_` (name upper-cased).

| Variable | Default | Meaning |
|---|---|---|
| `API_KEYS` (full name `INFLECT_API_API_KEYS`) | empty | Comma-separated bearer keys; empty rejects everything |
| `MAX_MODEL_BYTES` | 1048576 | Max model source size (413) |
| `MAX_SPECIES_REACTIONS` | 2000 | Reserved, not yet enforced (only `MAX_MODEL_BYTES` is) |
| `MAX_HORIZON_S` | 86400 | Max `horizon_s` / `dt_s` |
| `MAX_STEPS_PER_CALL` | 10000 | Max `horizon_s / interval_s` |
| `MAX_ENSEMBLE_N` | 256 | Max ensemble members |
| `MAX_OBSERVABLES_PER_CALL` | 64 | Observables per request |
| `REQUEST_TIMEOUT_S` | 30 | Per-call worker timeout (504) |
| `REGISTER_COMPILE_TIMEOUT_S` | 60 | Model compile timeout |
| `MAX_TWINS` | 64 | Live twins; the least-recently-used twin is evicted to stay under the cap |
| `TWIN_TTL_S` | 3600 | Idle eviction time (evicted twins re-warm at t=0 on next access) |
| `DB_PATH` | empty | SQLite file for twin persistence |
| `CHECKPOINT_DIR` | empty | Surrogate sidecar directory |
| `RATE_CAPACITY` / `RATE_REFILL_PER_S` | 100 / 10 | Per-key token bucket |
| `COST_STEP` / `COST_REGISTER` / `COST_ENSEMBLE` / `COST_LIGHT` | 1 / 10 / 20 / 0.25 | Token cost per call type |

Error bodies are not uniform. Errors raised by the registry, worker and surrogate layers return
`{"error": ..., "detail": ...}`: 403 `not_owner`, 404 `unknown_twin`, 409 `surrogate_unavailable`,
422 `unsafe_model` / `unknown_observable` / `invalid_request`, 502 `worker_error`, 503 `over_capacity`
(ensemble over `MAX_ENSEMBLE_N`) and `worker_unavailable`, 504 `timeout`. Other errors (401 bad key,
413 oversized model, 429 rate limited, and 422s from API limit checks or model rejection at registration)
return FastAPI's default `{"detail": "..."}`, and request-validation 422s return a `detail` list.

## Auth and ownership

Each twin is owned by the key that registered it. Registering the same model again with the same key returns the existing twin (dedup) instead of creating a new one. Another key gets 403 on that twin. Rate limits are
per key.

## Live state and restarts

Live state (current time, stepped species and parameter values) is ephemeral. With `DB_PATH` set, the
registry persists twin definitions, and after a restart a twin is rebuilt from its model; it comes back
at its initial state, not where you had stepped it. Without `DB_PATH`, twins are gone after a restart. TTL or LRU eviction behaves the same way: with `DB_PATH` set, the persisted row survives and the twin is rebuilt at t=0 on next access. An explicit `DELETE` also removes the persisted row.

## Deployment: single worker

Twins live in the memory of one process (each backed by a worker process). Run exactly one server
process, for example `uvicorn ... --workers 1`. Multiple workers would give each its own registry, so
requests for a twin would land on a process that does not hold it.
