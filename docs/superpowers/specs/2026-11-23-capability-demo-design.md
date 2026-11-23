# Capability-demo integration — design

**Date:** 2026-11-23
**Branch:** `capability-demo-cosim-nf-rust`
**Investigation:** `biogears-sbml-sepsis`

## Intent / why

The `biogears-sbml-sepsis` investigation is the simulation core's **general
capabilities demo**. Three vivarium capabilities it should show are currently
thin or missing:

1. **Co-simulation** — not shown. Study 01/03 run engines in *parallel* over the
   same spec (comparison), or as parallel tiers. No composite where two processes
   **exchange state every tick**.
2. **Nextflow interop** — the committed pipeline runs a toy `MathExpressionStep`
   chain, not the real sepsis composite.
3. **Rust execution backend** — `viva_core` is wrapped + has a cross-engine test,
   but appears in the showcase only as a "doesn't help (stiff ODE)" footnote in
   study 04; no actual run artifact.

Goal: make all three first-class, runnable, honestly-captioned arms of the
showcase. Audience: general capabilities demo (not a single hypothesis test).

## Goals

- A genuine, tick-coupled **co-simulation** study with a runnable artifact + figure.
- Nextflow pipeline running the **real** sepsis composite, committed + figure.
- An honest **Rust-backend** arm with a real `viva_core` run + measurement.
- Every new claim honestly captioned (no overclaiming; where a tier is approximate
  or a backend doesn't help, say so).

## Non-goals / anti-goals

- No native BioGears build required (Tellurium/COPASI/surrogate/rust only).
- Do **not** present the surrogate or the Rust backend as fidelity-grade where they
  are not. A demo that quietly overclaims is a failure even if it runs.
- Do not break the existing fidelity backbone (studies 01/05) or the committed
  comparison data.

## Constraints

- Worktree build (`inflect-sim-core--capability-demo`), canonical `.venv` with
  `PYTHONPATH=<worktree>` so edits resolve. Engines installed: tellurium, copasi,
  surrogate, rust (not julia; biogears wrapper imports but native engine optional).
- Study on-disk shape: `study.yaml` schema v3 (question/objective/claim/
  expected_behavior/findings/visualizations/baseline/variants), `charts/*.svg`
  (+ `.meta.json`), `runs.db` for the published gallery.
- No AI attribution in commits/PRs. Never auto-merge.

## Approach (chosen)

Add **one new study** for the genuinely-new pillar, and **upgrade two existing
studies** where the capability already has a home:

- **`06-cosimulation` (new):** a closed-loop composite — the sepsis physiology
  engine (Tellurium `sepsis_sbml`) coupled to a **separate controller process**
  that reads a physiology observable each tick and writes an actuation back via
  the Tellurium `parameters_in` port (confirmed to apply a settable global before
  each step). Clinically framed (closed-loop treatment vs untreated septic shock).
  The exact lever is chosen empirically (sensitivity probe) so the controller has a
  real physiological effect. Stretch: the controller's PK is a second engine
  (COPASI) → true engine↔engine exchange.
- **`03-workflows` (upgrade):** add a committed Nextflow pipeline whose stage runs
  `viva-sim run sepsis_sbml` (real composite), with a second stage consuming it;
  add a DAG/caption figure and a finding. Keep the toy chain as the minimal example.
- **`04-performance` (upgrade):** add a real `viva_core` (Rust) run arm driving a
  composite; record where it helps (graph/step execution overhead on a non-stiff
  composite) and restate honestly where it cannot (the stiff 91-state sepsis solve
  needs CVODE). Replaces the test-only + footnote treatment with a run artifact.

Then **integrate:** update `investigation.yaml` (question/lead name co-sim +
execution backends), the README composites/pillars tables (also fix the stale
engine table missing `julia`), regenerate figures + the report/dashboard.

### Approaches considered, not taken

- *Co-sim by coupling BioGears↔Tellurium on the same spec.* Rejected: same model
  under two engines is comparison (already studies 01/05), not co-simulation.
- *A new standalone demo outside the investigation.* Rejected: the ask is to
  integrate into the showcase so the four→six pillars read as one story.
- *Rust as its own new study.* Rejected: its honest story is execution cost, which
  belongs with performance (04); a standalone study would overstate it.

## Left to the builder

- Final lever/observable for the controller (from the sensitivity probe): a
  vasopressor-like global that raises MAP, or a pathogen kill-rate for an
  antibiotic-PK loop — whichever gives a clean, correct closed-loop signal.
- Whether the COPASI-as-second-engine stretch lands in this pass or is noted as a
  follow-up.
