# `viva-surrogate` framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pluggable surrogate/emulator framework whose centerpiece is a backend-agnostic **validate harness** that produces an honest PASS/FAIL scorecard — so "which surrogate works?" is answerable, and a silently-degenerate surrogate can't ship.

**Architecture:** A new module `viva_simulation_core/surrogate/` (v1 prototype home; extracts to a standalone `viva-surrogate` package once proven). A `SurrogateBackend` fits trajectories → a `FittedSurrogate` (exposes `rollout`); the `validate` harness scores any `FittedSurrogate` on four metrics; backends are neural (wrap viva-torch) + a trivial baseline; a `SurrogateProcess` wraps a fitted surrogate as an interchangeable process-bigraph box.

**Tech Stack:** numpy (harness + baselines, no new deps); viva-torch (`SurrogateSpec`, `_rollout_surrogate`, `NeuralProcess`) for the neural backend; process-bigraph `Process` for `SurrogateProcess`; pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-viva-surrogate-framework-design.md`

## Global Constraints

- **v1 home:** `viva_simulation_core/surrogate/` — import-clean (no circular deps on the workspace) so extraction to a `viva-surrogate` package is mechanical. Public API lives in `__init__.py`.
- **No new dependencies** for the core (harness + baselines) — numpy only. viva-torch is already a base dep (used by the neural backend).
- **The harness is backend-agnostic:** it scores any object satisfying the `FittedSurrogate` protocol; its core tests use SYNTHETIC fitted surrogates (constant / nonphysical / good), not viva-torch, so they are fast and deterministic.
- **Honesty gate:** a surrogate that fails responsiveness OR physicality is `verdict = "FAIL"` regardless of speed. Speed never buys a pass.
- **Process:** no AI attribution; work in this worktree (`feat/viva-surrogate-framework`); this is a **feature PR to main** (ready-for-review, not draft, per AGENTS.md) — CI must pass before merge; never auto-merge.
- **Tests** run against the worktree (`PYTHONPATH=<worktree>` prepended; the editable install points at the canonical checkout).

## Review Focus

- **A degenerate-but-fast surrogate must FAIL.** The responsiveness metric must catch a surrogate whose output barely varies across drivers even if rollout error happens to look ok on one trajectory. Covered by Task 2's degenerate-surrogate test.
- **Physicality with a partially-out-of-range surrogate:** a surrogate physical on most steps but spiking out of bounds must score physicality < 1 and FAIL if below the gate — not round up to "fine." Covered by Task 2's nonphysical test.
- **The neural backend wrapping the real sepsis checkpoint is correctly scored FAIL** (it is degenerate across severity) — the harness must flag the real known-bad surrogate, not whitewash it. Covered by Task 4's test.
- **Empty / single-trajectory / mismatched-length inputs** to the harness must raise a clear error, not divide-by-zero or silently pass. Covered by Task 2's edge-case test.
- **`SurrogateProcess` port names match the source engine's** so a fitted surrogate is actually swap-compatible (not just shaped right). Covered by Task 6's test.

---

### Task 1: `interface.py` — protocols + Scorecard

**Files:**
- Create: `viva_simulation_core/surrogate/__init__.py`, `viva_simulation_core/surrogate/interface.py`
- Test: `tests/surrogate/test_interface.py`

**Interfaces (Produces):**
- `FittedSurrogate` (typing.Protocol): attr `observable_names: list[str]`; method `rollout(x0: np.ndarray, n_steps: int) -> np.ndarray` returning shape `(n_steps, len(observable_names))`.
- `SurrogateBackend` (Protocol): attr `name: str`; method `fit(trajectories: list[np.ndarray], observable_names: list[str], dt: float) -> FittedSurrogate`.
- `@dataclass Scorecard`: fields `backend: str`, `rollout_nrmse: float`, `physicality: float`, `responsiveness: float`, `speedup: float | None`, `gates: dict[str,bool]`, and a computed `verdict: str` ("PASS"/"FAIL"); `as_dict()` and `__str__` (a small aligned table).

- [ ] **Step 1: Write failing test** — `Scorecard(gates={"responsive":True,"physical":True}, ...)` → `verdict=="PASS"`; any gate False → `"FAIL"`; `as_dict()` round-trips; `__str__` contains the backend name + verdict.
- [ ] **Step 2: Run it, verify it fails** (module absent).
- [ ] **Step 3: Implement** the two Protocols + the `Scorecard` dataclass (verdict = "PASS" iff all `gates` values True).
- [ ] **Step 4: Run tests → pass.**
- [ ] **Step 5: Commit** — `feat(surrogate): backend/fitted protocols + Scorecard`

---

### Task 2: `validate.py` — the harness (centerpiece)

**Files:**
- Create: `viva_simulation_core/surrogate/validate.py`
- Test: `tests/surrogate/test_validate.py`

**Interfaces:**
- Consumes: `FittedSurrogate`, `Scorecard` (Task 1).
- Produces: `validate_surrogate(fitted, truth_trajs: list[np.ndarray], driver_ics: list[np.ndarray], bounds: dict[str,tuple[float,float]], dt: float, *, speedup: float | None = None, resp_gate: float = 0.02, phys_gate: float = 0.95) -> Scorecard`. Helpers `_rollout_nrmse`, `_physicality`, `_responsiveness` (coefficient of variation of per-driver rollout endpoints, min across observables).

- [ ] **Step 1: Write the failing tests** (the heart of the framework):
```python
def test_degenerate_surrogate_fails_responsiveness():
    # outputs the same endpoint regardless of x0 -> responsiveness ~0 -> FAIL
    fitted = ConstantFitted(names, value=...)  # test double
    sc = validate_surrogate(fitted, truths, driver_ics=[ic_a, ic_b, ic_c], bounds=..., dt=1.0)
    assert sc.responsiveness < 0.02 and sc.verdict == "FAIL"

def test_nonphysical_surrogate_fails_physicality():
    fitted = OutOfRangeFitted(names)   # spikes outside bounds
    sc = validate_surrogate(...)
    assert sc.physicality < 0.95 and sc.verdict == "FAIL"

def test_good_surrogate_passes():
    fitted = TruthEchoFitted(truths)   # reproduces the held-out truth, physical, responsive
    sc = validate_surrogate(...)
    assert sc.verdict == "PASS" and sc.rollout_nrmse < 0.1

def test_empty_or_mismatched_inputs_raise():
    with pytest.raises(ValueError): validate_surrogate(fitted, [], [...], ...)
```
- [ ] **Step 2: Run them, verify they fail.**
- [ ] **Step 3: Implement** `validate_surrogate` + the three metric helpers. responsiveness = min over observables of the coefficient of variation (std/|mean|) of the per-driver rollout endpoints. physicality = fraction of all rollout values within `bounds`. nrmse = normalized RMSE of a rollout vs its matched truth trajectory. Raise `ValueError` on empty/mismatched inputs.
- [ ] **Step 4: Run tests → all pass.**
- [ ] **Step 5: Commit** — `feat(surrogate): validate harness + PASS/FAIL scorecard`

---

### Task 3: `backends/baseline.py` — persistence + linear baselines

**Files:**
- Create: `viva_simulation_core/surrogate/backends/__init__.py`, `viva_simulation_core/surrogate/backends/baseline.py`
- Test: `tests/surrogate/test_baseline_backends.py`

**Interfaces:**
- Consumes: `SurrogateBackend`, `FittedSurrogate` (Task 1).
- Produces: `PersistenceBackend` (rollout holds `x0` constant — responsive to IC, poor accuracy; the floor) and `LinearBackend` (fit `x_{t+1} = A x_t + b` by least squares on the transition pairs; rollout iterates).

- [ ] **Step 1: Write failing test** — `PersistenceBackend().fit(trajs,names,dt).rollout(x0,5)` is `x0` repeated 5×; `LinearBackend` fits a trivial linear system (`x_{t+1}=0.5 x_t`) and its rollout matches to tight tolerance; both satisfy the `FittedSurrogate` protocol.
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Implement** both backends.
- [ ] **Step 4: Run tests → pass.**
- [ ] **Step 5: Commit** — `feat(surrogate): persistence + linear baseline backends`

---

### Task 4: `backends/neural.py` — wrap viva-torch

**Files:**
- Create: `viva_simulation_core/surrogate/backends/neural.py`
- Test: `tests/surrogate/test_neural_backend.py` (uses the sepsis checkpoint; `@skipif` viva-torch/ckpt absent)

**Interfaces:**
- Consumes: `SurrogateBackend`/`FittedSurrogate` (Task 1); viva-torch `SurrogateSpec`, `_rollout_surrogate`.
- Produces: `NeuralBackend` — **engine-agnostic**: `from_checkpoint(path, *, core_provider)` wraps an existing `.pt`; the returned `FittedSurrogate.rollout` calls `_rollout_surrogate(spec, ckpt, core_provider(), …)`, where `core_provider` is a caller-supplied callable that builds the stepping core (so the backend is not sepsis-coupled). `fit` may train via viva-torch; `from_checkpoint` is the v1 path.
- **Note:** `build_sepsis_state_composite` (the sepsis rollout core) lives on the investigation branch, not `main`. The sepsis integration test is therefore gated: skip unless `viva_simulation_core.composites.sepsis_sbml_state` AND the committed checkpoint are importable. The backend itself stays general.

- [ ] **Step 1: Write failing test** — if the sepsis state builder + a sepsis checkpoint are importable (`@pytest.mark.skipif` otherwise), wrap the checkpoint via `NeuralBackend.from_checkpoint(path, core_provider=lambda: build_sepsis_state_composite(sev,0.0,dt)[1])`, score it across severity-0/1/3 driver ICs, and assert `verdict == "FAIL"` with `responsiveness < resp_gate` (the real degenerate surrogate is correctly flagged). Also add a non-skipped test with a tiny stub `_rollout_surrogate`-shaped callable to exercise `NeuralBackend` wiring without viva-torch.
- [ ] **Step 2: Run it, verify it fails** (NeuralBackend absent).
- [ ] **Step 3: Implement** `NeuralBackend` + `from_checkpoint`, adapting the `SurrogateSpec(**ckpt["spec"])` + `_rollout_surrogate` path (as used in the investigation).
- [ ] **Step 4: Run tests → pass** (the harness flags the real surrogate FAIL).
- [ ] **Step 5: Commit** — `feat(surrogate): neural backend wrapping viva-torch; harness flags the degenerate sepsis net`

---

### Task 5: `fit_and_score.py` + CLI

**Files:**
- Create: `viva_simulation_core/surrogate/fit_and_score.py`
- Modify: `viva_simulation_core/surrogate/__init__.py` (export the public API)
- Test: `tests/surrogate/test_fit_and_score.py`

**Interfaces:**
- Consumes: all of Tasks 1–4.
- Produces: `fit_and_score(backend, trajectories, observable_names, dt, bounds, driver_ics, truth_trajs) -> Scorecard`; a `__main__` CLI (`python -m viva_simulation_core.surrogate.fit_and_score --backend persistence|linear|neural ...`) printing the scorecard table.

- [ ] **Step 1: Write failing test** — `fit_and_score(LinearBackend(), ...)` on a trivial linear system → `verdict=="PASS"`; on the persistence baseline over a system with real dynamics → poor nrmse (floor); the public API imports from `viva_simulation_core.surrogate`.
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Implement** `fit_and_score` + CLI + `__init__` exports.
- [ ] **Step 4: Run tests → pass.**
- [ ] **Step 5: Commit** — `feat(surrogate): fit_and_score orchestration + CLI + public API`

---

### Task 6: `SurrogateProcess` — the interchangeable box

**Files:**
- Create: `viva_simulation_core/surrogate/process.py`
- Modify: `viva_simulation_core/surrogate/__init__.py`
- Test: `tests/surrogate/test_surrogate_process.py`

**Interfaces:**
- Consumes: a `FittedSurrogate` (Task 1); process-bigraph `Process`.
- Produces: `SurrogateProcess(Process)` — config carries a fitted surrogate (or a backend + checkpoint); `inputs()`/`outputs()` expose one flat port per observable (same convention as viva-torch `NeuralProcess`); `update(state, interval)` advances one rollout step. A `register(core)` helper.

- [ ] **Step 1: Write failing test** — build a `SurrogateProcess` over a `LinearBackend` fit, run it a few steps inside a `process_bigraph.Composite`, assert it emits finite values and that its port names equal the given `observable_names` (swap-compatible).
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Implement** `SurrogateProcess` + `register`.
- [ ] **Step 4: Run tests → pass.**
- [ ] **Step 5: Commit** — `feat(surrogate): SurrogateProcess — fitted surrogate as an interchangeable box`

---

## Post-plan (not tasks)
- Open a **feature PR to `main`** (ready-for-review): `feat(surrogate): pluggable surrogate framework + validate harness`. Body notes v1 lives in `simulation-core` and extracts to a `viva-surrogate` package once the interface is proven; names the honest-scorecard motivation.
- Follow-on (separate, scored research loop, NOT this plan): reduced-SINDy backend (POD/DMD reduction), a hand-derived mechanistic reduced ODE, and a PDMP backend — each scored by this harness.
