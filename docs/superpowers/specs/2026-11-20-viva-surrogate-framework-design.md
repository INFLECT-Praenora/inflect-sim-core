# Design — `viva-surrogate`: a pluggable surrogate/emulator framework

**Status:** draft for approval
**Date:** 2026-11-20
**Branch:** `feat/viva-surrogate-framework` (off `main` @ 4a1e659)

## Intent & why

Make surrogates/emulators a **first-class, pluggable capability** in the viva
ecosystem: fit a surrogate to any engine with **any method**, wrap it as an
**interchangeable engine box** over the same typed ports, and **validate it
honestly** with a shared scorecard.

The motivation is concrete and recent: the sepsis neural surrogate *ran fast but
was silently degenerate* — identical nonphysical outputs across severities. Nothing
in `viva-torch` / `surrogate_curate` caught that; it was caught by an ad-hoc check.
The single most valuable thing this framework provides is the **validate harness**
that makes that failure impossible to miss — and makes "which surrogate method
works?" an answerable, scored question instead of a guess. This also upgrades the
INFLECT story from "one brittle net" to "scaffolding for surrogates, with an honest
scorecard."

## Goals (v1)

1. **One interface.** A `SurrogateBackend` protocol: `fit(trajectories | engine) →`
   a `SurrogateProcess` that exposes the **same typed ports** as the source engine,
   so a fitted surrogate is a drop-in interchangeable box (the composition thesis).
2. **A backend-agnostic validate harness** that produces an honest **scorecard**:
   - **rollout error** (NRMSE vs a held-out truth trajectory),
   - **physicality** (fraction of outputs within declared physical bounds),
   - **input-responsiveness** (does the output actually vary when a known driver
     varies — e.g. severity? the exact check that would have flagged the degenerate
     net),
   - **speedup** (wall-clock vs the exact engine).
   A surrogate that fails responsiveness or physicality is marked FAILED regardless
   of speed.
3. **Neural backend #1** — wrap the existing `viva-torch` path behind the interface
   (no reimplementation; it becomes one scored backend).
4. **Prove pluggability** with one more trivial backend (a persistence/linear
   baseline) so the harness has a floor and the plug-in path is exercised by >1 method.

## Non-goals (v1)

- **Not** solving the sepsis surrogate-accuracy research problem. v1 delivers the
  framework + harness; accurate backends (better SINDy with POD/DMD reduction, a
  hand-derived reduced mechanistic ODE, a PDMP) are a **scored research loop on top**,
  not v1 deliverables. (A quick reduced-SINDy spike already showed the fast
  hemodynamic states need more than a 5-observable reduction — this is real research.)
- **Not** a standalone published repo in v1. v1 lives as a module in `simulation-core`
  to prove it against the real sepsis engines/data; it **extracts to a standalone
  `viva-surrogate` package once proven** (feature-first; the module is import-clean so
  extraction is mechanical).
- **Not** a UQ framework; it composes with `viva-uq` (GP/PCE emulators can later be
  backends) but does not duplicate it.

## Approaches considered

- **(A) New `viva-surrogate` repo now.** Rejected for v1 — heavy (repo, pyproject, CI,
  publish) before the design is proven; extraction later is cheap once the interface
  is settled.
- **(B) Module in `simulation-core`, extract later — CHOSEN.** Fastest path to a
  provable framework against the real sepsis surrogate + engines; import-clean so the
  later extraction to a package is mechanical.
- **(C) Extend `viva-torch`.** Rejected — couples the framework to the NN backend; the
  whole point is backend-agnosticism.

## Scope (v1 — what gets built)

A new module `viva_simulation_core/surrogate/` (prototype home):

- `interface.py` — `SurrogateBackend` protocol (`fit(...) -> SurrogateSpec/artifact`),
  and `SurrogateProcess` (a process-bigraph Process wrapping a fitted surrogate over
  the source engine's ports). Reuses the existing `SurrogateSpec` shape where it fits.
- `validate.py` — `validate_surrogate(surrogate, truth_trajectories, driver, bounds)
  -> Scorecard` computing the four metrics above; `Scorecard` renders to dict/table
  and a PASS/FAIL verdict.
- `backends/neural.py` — the viva-torch backend behind the interface.
- `backends/baseline.py` — a trivial persistence/linear backend (floor + pluggability
  proof).
- `fit_and_score.py` — a function (and thin CLI) that fits a backend to an engine's
  trajectories and prints/returns the scorecard — the honest "does this surrogate
  work?" entry point.
- Tests: the harness catches (a) a degenerate surrogate (fails responsiveness), (b) a
  nonphysical surrogate (fails physicality), (c) a good baseline on a trivial system
  (passes); the neural backend wraps + scores end-to-end.

## What's left to the builder

1. Exact metric thresholds + the PASS/FAIL gating rule.
2. The baseline backend (persistence vs 1-step linear) choice.
3. How `SurrogateProcess` maps a fitted model back onto the source engine's exact port
   names (reuse the viva-torch flat-port convention).
4. Scorecard rendering (dict + a small table; later a workbench viz).
5. The extraction boundary (which symbols are the stable public API for the eventual
   `viva-surrogate` package).

## Open questions for approval

- **Home:** v1 as `viva_simulation_core/surrogate/` (prototype, extract later) — OK, or
  scaffold the standalone `viva-surrogate` repo now?
- **v1 backend count:** neural + one trivial baseline (recommended), or also attempt a
  first real research backend (reduced-SINDy) in v1?
