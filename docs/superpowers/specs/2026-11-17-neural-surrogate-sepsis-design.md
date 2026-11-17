# Neural Surrogate of the Sepsis SBML Model — Design

**Status:** draft for approval (2026-11-17)
**Workspace:** `simulation-core` (public: `INFLECT-Praenora/inflect-sim-core`)
**Investigation slug:** `neural-surrogate-sepsis`
**Depends on:** `biogears-sbml-sepsis` (the target model + its committed CVODE runs), `viva-torch` (`~/code/viva-torch`)

---

## Intent & why

The `biogears-sbml-sepsis` investigation established that the sepsis SBML model is a
**stiff 91-state ODE** integrated by libroadrunner/CVODE, and that reimplementing the
integrator (Rust/RK4) is *not* a speed lever (study 06). The remaining speed lever named
in that thread — and asked for directly by Herbert Sauro and Tien — is a **learned
surrogate**: a neural network that approximates the model's dynamics and rolls out far
faster than integrating a stiff ODE.

`viva-torch` is purpose-built for this: it trains a residual-MLP (`SurrogateNet`) to
approximate any target `Process`/`Composite`'s per-step dynamics over a declared panel of
state paths, and emits a drop-in `NeuralProcess` that rolls out autoregressively. This
investigation puts that pipeline to work on the sepsis model and answers, with evidence:
**can a surrogate reproduce the sepsis dynamics faithfully enough to be a usable fast
stand-in, and where does it break?**

The scientific hook ties back to study 06: a per-step learned surrogate **sidesteps
integration entirely** (no stiffness), but the fast pulsatile cardiovascular states are the
hardest to learn, while the slow inflammation core should be easy. **Hypothesis:** the
surrogate is accurate *and* fastest on the slow inflammation states and degrades on the fast
CV states — the same slow/fast split that defines the stiffness, now seen from the ML side.

## Goals

1. **Train** a viva-torch surrogate of `sepsis_sbml` across a severity range.
2. **Run** it as a drop-in `NeuralProcess` and roll out 48 h trajectories.
3. **Compare** surrogate vs real CVODE model on **accuracy** (per-observable trajectory
   error) *and* **speed** (wall-clock per 48 h sim + virtual-patient batch throughput).
4. Characterize **where it breaks** — slow vs fast states, and out-of-distribution severity.
5. Ship as a rigorous investigation in `simulation-core` (studies, figures, report), and
   give Herbert/Tien a concrete surrogate result.

## Non-goals

- Not replacing the mechanistic model — the surrogate is a coarse-grained fast stand-in,
  evaluated honestly, not a validated clinical model.
- Not a GPU/throughput-engineering effort — single-machine CPU/GPU training is enough to
  answer the accuracy+speed question. (GPU-roadrunner is a separate thread.)
- Not beat-to-beat pulsatile fidelity — we compare the coarse-grained observables the
  committed runs already sample (~60/h), not within-beat waveforms.
- Not touching the `biogears-sbml-sepsis` investigation's conclusions.

## Anti-goals (would make this a failure even if it "works")

- A surrogate that only memorizes one severity and is presented as general.
- A speed number without a paired accuracy number (or vice versa) — a fast wrong model is
  worthless; the comparison must carry both.
- Hiding the fast-CV-state degradation if it occurs — that *is* a headline finding, not a
  defect to bury.
- Fabricated or hand-waved metrics — every number comes from a real `evaluate_surrogate`
  run against the real composite.

## Constraints

- **No AI attribution** in commits/PRs (standing rule).
- **Worktree discipline** — this investigation gets its own branch/worktree
  (`neural-surrogate-sepsis`); do not build on a shared `main` checkout.
- **torch is a heavy dep** and is NOT in simulation-core's venv today. Add `viva-torch`
  (which pulls torch) as an **optional extra** (`[surrogate]` or `[ml]`), and mark all
  surrogate tests `@pytest.mark.slow` so the default CI stays light — the exact pattern the
  BioGears engine tests already use.
- **Public repo** — everything shipped is public; keep it clean and reproducible.
- Surrogates **interpolate within their training distribution** (viva-torch README) — the
  sampling plan must cover the severities/initial conditions we later evaluate; OOD is a
  deliberate study (#4), not an accident.

## Approaches considered

**A. Full-state autoregressive surrogate (chosen).** `target_paths` = the model's state
vector (the ~91 rate-rule variables), so the surrogate learns a **Markov-closed** map and
rolls out faithfully; the 9 comparison observables are derived/compared on top. This is
viva-torch's intended use and the only approach that supports honest multi-step rollout.

**B. Observable-only surrogate (rejected as the primary; retained as fallback).**
`target_paths` = just the 9 observables. Simpler, but the 9 observables are *functions* of
the 91 states and are **not Markov-closed**, so autoregressive rollout can drift. Rejected
as the primary — but see the Key Risk: if exposing the 91-state vector proves infeasible,
B becomes the v1 and "is the observable panel Markov-enough?" becomes an explicit finding.

**C. Reduced slow-states-only surrogate (rejected).** Surrogate only the non-stiff
inflammation subsystem. Cleaner result but dodges the whole point (the CV states are where
the interesting answer is). Kept only as a possible ablation inside study #4.

## KEY RISK — state-vector exposure (resolve first)

The `sepsis_sbml` composite exposes the **9 observables** (SBML global params / assignment
rules) in the `TelluriumProcess` `parameters` store as PB-addressable paths. The raw
~91-element roadrunner **state vector** (the independent rate-rule variables) may **not**
currently be addressable as process-bigraph paths. Approach A needs it to be.

Resolution order (this is **Task 1** of the plan — a feasibility spike, before any training):
1. Probe whether the roadrunner state / floating variables are reachable as PB paths from
   `TelluriumProcess` (or via a thin read-through of `rr.model` state ids).
2. If a small, clean addition to `viva-tellurium`/the composite exposes them → do that (it
   is reusable infrastructure → **feature PR against `main`**, per AGENTS.md, not carried on
   the investigation branch).
3. If exposing the full vector is hard/ugly → fall back to Approach B for v1 (observable
   panel), and make rollout-drift a measured finding; note the full-state surrogate as a
   follow-up. **Either way the investigation ships** — the fallback is a real result, not a
   failure.

## Decided "how"

- **Investigation:** `neural-surrogate-sepsis` in `simulation-core`, own branch/worktree.
- **Target:** `viva_simulation_core.composites.sepsis_sbml` (registered; CVODE under the hood).
- **Surrogate spec:** `parameterization="delta"` (per-step residual), coarse `timestep`
  (candidate 10–60 s; pinned in the plan after the spike), targets per Approach A (fallback B).
- **Training distribution:** a **severity sweep** (0/1/2/3 + jittered initial conditions)
  via `TrajectorySampler` + `SamplingPlan`; hold out initial conditions for evaluation.
- **Comparison:** `evaluate_surrogate` for per-observable nRMSE over 48 h; a paired
  wall-clock/throughput harness (surrogate rollout vs the real composite) reusing the
  `biogears-sbml-sepsis` runtime-figure pattern.
- **Dependency:** `viva-torch` as optional extra; surrogate tests `@pytest.mark.slow`.
- **Four studies:**
  1. `01-sample-and-train` — dataset + trained `SurrogateNet` + train/val loss. *(train)*
  2. `02-accuracy` — surrogate vs CVODE per-observable nRMSE on held-out ICs; which states
     track vs drift. *(compare — accuracy)*
  3. `03-speed` — surrogate rollout vs stiff ODE integration, per-48 h and batch. *(compare — speed)*
  4. `04-where-it-breaks` — accuracy stratified slow-inflammation vs fast-CV; OOD severity
     (adversarial / rigor). Ties to `biogears-sbml-sepsis` study 06.
- **Report:** rendered via `/viva-report`; each study carries findings + ≥1 figure
  (accuracy-vs-state bars, surrogate-vs-CVODE overlays, speed bars, slow/fast scatter).

## What's left to the builder (plan will pin)

- Exact `timestep`, network size (`hidden`, `epochs`, `lr`), and sampling `n_steps` /
  severity grid — chosen empirically in Task 1–2, not guessed here.
- The precise set of state paths for Approach A (resolved by the Task 1 spike).
- The state→observable derivation (recompute the 9 from predicted states, vs. also predict
  the 9 directly as extra targets for a direct comparison) — plan picks one with a one-line
  rationale.
- Whether the batch-throughput number uses a real virtual-patient sweep or a timed loop.

## Success criteria

- A trained surrogate checkpoint, reproducibly regenerated from committed scripts.
- Study 02 reports honest per-observable accuracy vs the real model on held-out ICs.
- Study 03 reports a real, paired speed number (surrogate vs CVODE), accuracy-gated.
- Study 04 answers the slow/fast hypothesis with evidence.
- Investigation report renders with findings + figures; DE audit clean; CI green.
