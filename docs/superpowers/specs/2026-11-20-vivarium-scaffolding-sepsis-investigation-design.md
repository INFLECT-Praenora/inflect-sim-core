# Design — Simulation-core capabilities sepsis investigation (INFLECT demonstration)

**Status:** draft for approval
**Date:** 2026-11-20
**Author/driver:** eagmon (via Claude)
**Workspace:** `simulation-core` · investigation `biogears-sbml-sepsis` (slug kept; title reframed)
**Branch:** `investigation/simcore-scaffolding` (off `main` @ 4a1e659)

## Intent & why

Convince the **INFLECT team** (ARPA-H CIRCLE) that vivarium / process-bigraph is the
**simulation scaffolding they need**: it handles **composition**, **ensembles**,
**orchestration & workflows**, and it is **performant**. The governing principle is
**show, don't tell** — every capability claim is backed by a *runnable* artifact in
the investigation (a composite you can open, a workflow you can call, a live
benchmark, an interactive figure), never prose assertion.

Today the investigation answers a narrow validation question — *"does the
Antimony/SBML extraction reproduce the BioGears engine run-for-run?"* — across 6
studies. We consolidate that into **one comprehensive capabilities demonstration**
built on the same real sepsis model, reorganized so each study demonstrates a
capability pillar, with the honest engine-comparison evidence retained as a
credibility backbone.

## Audience

INFLECT technical evaluators deciding whether vivarium is the right substrate.
Technically strong but **not vivarium-fluent** — hence a `how_to_read` and
`glossary` carried on the investigation, and plain-language study framings.

## Goals

1. **One comprehensive investigation** demonstrating four pillars: **Composition**,
   **Ensembles**, **Orchestration/Workflows**, **Performance**.
2. **Each study leads with a runnable artifact**, not prose: a composite card, a
   callable workflow, an ensemble sweep, a live micro-benchmark, an interactive viz.
3. **The orchestration centerpiece**: a workflow that **passes an Antimony/SBML
   file between engines** (BioGears core → its SBML/Antimony extraction → Tellurium
   runs it → surrogate/compare), demonstrating module hand-offs on a portable
   artifact.
4. **Fold the PyTorch surrogate fully in** as the third engine throughout
   (Composition, Ensembles, Performance); **retire** the separate
   `neural-surrogate-sepsis` investigation (supersede, with provenance).
5. **Keep the honest fidelity/divergence evidence** — the engines do *not* agree
   perfectly — so the demonstration is credible, not marketing.

## Non-goals / anti-goals

- **Non-goal:** re-running the ~3.4 h BioGears severe jobs. Reuse committed
  `meta.wall_s` for heavy runs; run only the cheap new things live.
- **Anti-goal (overclaiming):** trimming the divergence evidence to make the
  engines look more interchangeable than they are. Credibility *is* the product;
  the fidelity backbone stays and stays honest.
- **Non-goal:** a slide deck. It must remain a real, reproducible investigation
  rendered by the workbench (and publishable read-only).
- **Non-goal:** renaming the investigation slug (`biogears-sbml-sepsis`) — only the
  title/framing change, to avoid churning the published URL, git history, and
  cross-references.

## Constraints (global — every task inherits these)

- **Run budget:** reuse committed `wall_s` for BioGears + Tellurium full runs; run
  live only: the handoff workflow, the surrogate, the ensemble sweep, and
  per-step micro-benchmarks. No 3 h re-runs.
- **Orchestration surface:** the unified **four-verb workflow API** (PR #9, merged):
  list modules → call one → connect a workflow → run it
  (`viva_simulation_core/api/workflow*.py`, `workflow_examples.py`).
- **Ensembles:** the study-schema `sweep` primitive + the workbench
  `lib/ensemble_config.py`.
- **Engines available:** `BioGearsProcess` (viva-biogears, C++), `TelluriumProcess`
  / `TelluriumUTCStep` (viva-tellurium, SBML/CVODE), `NeuralProcess` (viva-torch
  surrogate), `MathExpressionStep` (viva-basic-processes), viva-core (Rust
  composition engine). Composites: `sepsis_engine` (BioGears), `sepsis_sbml`
  (Tellurium). Handoff artifacts: `sources/biogears_sepsis.ant` / `.xml`.
- **Reuse simcore bits:** cherry-pick the already-built pieces from branch
  `biogears-sbml-sepsis-simcore`: the `00-simulation-core` study, the investigation
  `how_to_read` + `glossary`, and the live engine micro-benchmark
  (`bench_live.py` / `make_runtime_compare.py`).
- **Process:** no AI attribution; dedicated worktree; investigation PR opens as
  **draft** with an `investigation:` title (per `AGENTS.md`); never auto-merge.
- **Honesty:** the surrogate is a fast *approximate* tier (bounded, not yet
  clinically accurate) — labeled as such everywhere it appears.

## Approaches considered

- **(A) Pillar-mapped capability tour — CHOSEN.** 6 studies, each demonstrating one
  pillar, plus a fidelity backbone. Maps the four capabilities 1:1 to what INFLECT
  asked to see; each study is an openable, runnable demonstration; narrative reads
  as a guided tour.
- **(B) Engine-centric** (one study per engine). Rejected — repeats structure, buries
  orchestration and ensembles (the two pillars INFLECT most needs), and never shows
  composition or the hand-off.
- **(C) One mega-study.** Rejected — too dense; loses the "open each capability"
  affordance that makes show-don't-tell work.

## The study map (decided)

Six existing studies → reorganized around four pillars + a fidelity backbone. Each
row's deliverable is the *runnable artifact*, with prose kept minimal.

| # | Study (new name) | Pillar | What it SHOWS (runnable) | Built from |
|---|---|---|---|---|
| 00 | The simulation core | framing | `how_to_read` + `glossary` + at-a-glance of the 4 pillars; the one spec and the engines | simcore `00` (cherry-pick) |
| 01 | One spec, interchangeable engines | **Composition** | the same sepsis core run by BioGears (C++), Tellurium (SBML/CVODE), and the surrogate — swap one node address, same 9 observables; overlay trajectories | consolidates 01+02+03 |
| 02 | A severity × engine sweep in one batch | **Ensembles** | one orchestrated `sweep` over severities {0,1,3} × engines {Tellurium, surrogate} (BioGears cited), endpoints gathered into one grid/heatmap | **new** (elevates 02/03 arms) |
| 03 | Compose modules, hand off an SBML file | **Orchestration / Workflows** | the four verbs end-to-end; a workflow where `biogears_sepsis.ant` flows into `TelluriumProcess` → `MathExpressionStep` compare (+ `NeuralProcess` tier), built from registered modules alone | **new — centerpiece** |
| 04 | Where compute goes, and the 40,000× surrogate | **Performant** | compute profile across engines (committed `wall_s`), the stiff-ODE bottleneck, "a faster integrator can't help," the surrogate speedup; a live per-step micro-benchmark | consolidates 04+06 + surrogate |
| 05 | Do the engines agree, and where not | fidelity backbone | parity (healthy/mild/severe) + the divergence mechanism — honest "where and why they differ" | keeps 05 + parity detail from 01/02/03 |

Net change: the three severity/parity studies (01/02/03) consolidate into
**Composition** + **Fidelity**; the two runtime studies (04/06) consolidate into
**Performance**; **two new capability studies** (Ensembles, Orchestration/Workflows)
carry the pillars INFLECT most needs; the surrogate folds in as the third engine
throughout and the separate `neural-surrogate-sepsis` investigation is superseded.

## Per-study detail (decided vs. left-to-builder)

### 00 — The simulation core (framing)
Cherry-pick the simcore `00-simulation-core` study, `how_to_read`, and `glossary`.
Reframe the investigation `title`, `question`, `hypothesis`/thesis, `executive`,
and `at_a_glance` to the four-pillar capabilities tour. Decided. Builder fills the
exact at-a-glance ordering + the one-paragraph "what you're looking at."

### 01 — Composition: one spec, interchangeable engines
Baseline composite `sepsis_sbml` (Tellurium) is the anchor; show the SAME spec under
`sepsis_engine` (BioGears) and the surrogate by swapping one node. Lead artifact: the
Model-tab composite card(s) + an overlay of the 9 observables across engines at a
matched severity. Reuse committed parity trajectories; the Tellurium + surrogate
overlays can be (re)rendered cheaply. Builder: pick the severity to headline and the
overlay viz.

### 02 — Ensembles: a severity × engine sweep in one batch (NEW)
Use the study `sweep` primitive to run severities {0,1,3} × engines {Tellurium,
surrogate} as one orchestrated ensemble (BioGears severe cited from committed runs,
not swept live). Gather an endpoint (e.g., min MAP, time-to-threshold) per cell into
one figure (grid/heatmap). Cheap to run live (Tellurium seconds/run, surrogate
sub-second). Builder: exact sweep config shape (`sweep:` in `study.yaml`), the
gathered endpoint(s), and the ensemble figure.

### 03 — Orchestration/Workflows: compose + SBML hand-off (NEW, centerpiece)
Build on `viva_simulation_core/api/workflow_examples.py` (`tiered_workflow_spec`): a
workflow whose nodes are registered modules only — `TelluriumProcess` loading
`sources/biogears_sepsis.ant` (the portable artifact the BioGears core was extracted
to), its outputs wired into a `MathExpressionStep` compare, with a `NeuralProcess`
surrogate tier. Narrate the four verbs against the live API (list → call one →
connect → run) and show the artifact hand-off explicitly (the `.ant`/`.xml` file is
the interface between the C++ engine and the SBML integrator). Runs live (cheap).
Builder: the exact node/wire spec, what the compare step emits, and how the hand-off
is visualized (a small DAG + the shared-artifact callout).

### 04 — Performance: compute profile + surrogate (consolidate 04+06 + surrogate)
Reuse the simcore performance study + live micro-benchmark (`bench_live.py`,
`make_runtime_compare.py`): full-run `wall_s` across BioGears (~3.38 h) / Tellurium
(~2.81 h) / surrogate (~0.2 s), the stiff-ODE bottleneck (06's finding: CVODE
required, a faster/Rust integrator can't help), and the live per-step ratio
(~528 ms exact vs ~0.042 ms surrogate). Builder: merge 04 + 06 findings without
duplication; keep the two-panel figure.

### 05 — Fidelity & divergence (keep 05 + parity)
Keep `05-divergence-mechanism` (where/why engines differ) and fold the parity
evidence (healthy/mild/severe match + lag) in as the "do they agree?" first half, so
the demonstration stays honest. Builder: decide whether parity + divergence are one
study or parity lives under 01 with divergence as 05.

## What's left to the builder (collected)

1. Exact `sweep` config + gathered endpoint(s) + ensemble figure (study 02).
2. Exact workflow node/wire spec + compare-step output + hand-off viz (study 03).
3. How much surrogate-accuracy methods detail to migrate vs. reference when
   superseding `neural-surrogate-sepsis` (study 04 / provenance).
4. Whether parity sits under 01 (Composition) or 05 (Fidelity).
5. Per-study interactive viz specifics (overlays, heatmap, DAG).
6. Final investigation title + at-a-glance ordering.

## Supersession / cleanup

- `neural-surrogate-sepsis` (PR #8, branch `neural-surrogate-sepsis`): its surrogate
  engine + headline result migrate here; mark it superseded by this investigation
  with a pointer, and close PR #8 (do not merge it separately). The surrogate's
  deep-accuracy study content migrates per builder decision (3).
- The reframe keeps the slug `biogears-sbml-sepsis`; the published read-only
  workbench re-publishes from `main` after the investigation PR merges.

## Open questions for approval

- Title wording — set to *"Simulation core — one sepsis model, every engine:
  composed, ensembled, orchestrated, benchmarked"* (simulation-core-forward; no
  "vivarium" in the reader-facing title). Reader-facing study names and framings
  likewise lead with the capability, not the framework.
- Study-05 shape (parity+divergence together, or parity under 01).
- Depth of surrogate-accuracy migration.
