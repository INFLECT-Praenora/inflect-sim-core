# simulation-core: BioGears ↔ SBML sepsis comparison — design

**Date:** 2026-11-15
**Org:** `INFLECT-Praenora` (ARPA-H CIRCLE program "INFLECT")
**Author/owner:** eagmon
**Status:** awaiting approval (brainstorming → design gate)

## Intent & why

INFLECT needs a shared **API + runtime layer** for the program, built on
`process-bigraph` + `viva-core`. `simulation-core` is that layer. Its first
concrete job is a scientific comparison that is newly possible:

The BioGears sepsis core was extracted by Herbert Sauro into an Antimony/SBML
model (`biogears_sepsis.ant` / `.xml`, 91 state variables, from
`/Users/eranagmon/Downloads/Tien/`). That extraction was validated **only
against the McDaniel et al. 2019 paper milestones**, because the author could
not run BioGears itself. We can run BioGears: `viva-biogears` drives the **real**
BioGears 8.2.0 engine via `pybiogears` (nothing mocked) and already ships a
`biogears_sepsis` composite.

So the first investigation is the **run-for-run comparison the extraction report
could not do**: run the BioGears sepsis core as the real engine (`viva-biogears`)
and as the SBML/Antimony extraction (`viva-tellurium`) side by side, and quantify
agreement and divergence.

## Goals

1. A **public** repo `INFLECT-Praenora/simulation-core`, a viva workspace
   (`workspace.yaml` + `pbg_simulation_core/`), that is the program's API/runtime
   layer and depends on `process-bigraph`, `viva-core`, `viva-tellurium`,
   `viva-biogears`.
2. A first **investigation** ("BioGears sepsis core: engine vs SBML extraction")
   with studies that run both sides for the **mild** and **severe** infection
   arms and produce a quantitative, reviewer-readable comparison, plus a
   **runtime axis** that runs the same composite under the Python
   `process-bigraph` engine and the Rust `viva-core` engine to compare wall-clock
   and confirm cross-engine determinism.
3. The Antimony/SBML model + extraction report brought into the repo as the
   investigation's source artifacts, with provenance (Sauro, Apache-2.0 BioGears,
   McDaniel 2019).
4. A comparison report: aligned time-series (SBP, MAP, HR, CO, SVR, blood volume,
   tissue integrity, pathogen load) + milestone table (the report's measures:
   critical bacterial load, SBP<100 timing, MAP<65, SVR-vs-baseline), with
   divergence called out (notably the severe-arm ~8–10 h delay).

## Non-goals

- Not forking `viva-tellurium`/`viva-biogears` into INFLECT now (Approach B, later
  if governance requires).
- Not fixing or re-deriving the extraction (not patching the `(1-PT)` quirk,
  adding urine/fever/lactate, etc.) — we **measure** divergence, we don't correct
  the model in this first pass.
- Not a general multi-model API surface beyond what this comparison needs; the
  runtime API grows from real use, not up front.
- Not reproducing BioGears' full physiology — only the sepsis core already in
  scope on both sides.

## Anti-goals (failure even if it "works")

- A comparison that silently compares mismatched configs (different patient,
  severity, start time, units, or horizon) and reports false agreement.
- Presenting SBML-side numbers as "BioGears" or vice versa.
- A green report produced without the real BioGears engine actually having run
  (mock/placeholder masquerading as a run-for-run result).
- Fabricated or hand-waved time-series where a real run was too slow to finish.

**Fallback if the engine can't be built (builder's call, no fabrication):** if
neither laptop nor mini can build BioGears/pybiogears in time, the first pass
ships the **SBML/Tellurium runs compared against the McDaniel-2019 paper
milestones** (what Sauro did) with the engine side **explicitly marked pending** —
never a stand-in or invented engine trace. The run-for-run engine comparison then
lands in a follow-up once the build exists.

## Constraints

- **Native build + private dep.** `viva-biogears` requires BioGears 8.2.0 +
  `pybiogears` built from source (not on PyPI). Confirmed **absent** on this
  laptop; mini unverified (Tailscale not resolving at design time). `pybiogears`
  is now **public** (flipped 2026-11-15, with `viva-biogears`), so the whole
  engine→SBML stack is publicly installable; the engine still needs a native
  source build on the run host. `viva-biogears` CI skips engine tests when no
  build is present, so a public `simulation-core` CI can still stay green without
  the engine.
- **Compute cost.** Tellurium side ≈ 2.5 h per 48 h run (per report); BioGears
  sepsis ≈ 8 h/arm (per `viva-biogears` README). Four arms (2 severities × 2
  engines) is long-running → mini / parallel sweeps, not interactive local runs.
- **Alignment risk.** The two sides must be driven from the *same* scenario:
  StandardMale patient, matched `severity`/`infectionStart`, same horizon, unit
  conventions reconciled (Antimony runs in seconds, inflammation /3600; BioGears
  records per beat). Alignment is the core engineering risk.
- **Working mode:** dedicated worktree per session; AI-free dashboard split;
  no AI attribution in commits/PRs; squash-merge; the user approves all merges.

## Approaches

- **A — depend on the vivarium-collective repos as-is (CHOSEN).**
  `simulation-core` declares `viva-tellurium` + `viva-biogears` as dependencies;
  they stay in vivarium-collective. Fastest, zero duplication, upstream keeps CI.
  Cost: INFLECT depends on another org's repos.
- **B — fork both into INFLECT-Praenora (deferred).** Program owns its stack;
  revisit if ARPA-H governance needs sovereign, pinned copies. Cost: +2 repos to
  maintain, fork drift.
- **C — git submodules (rejected).** Fights the viva-workspace dependency model;
  native BioGears build is separate regardless; worst DX.

## The how (decided)

- **Repo:** `INFLECT-Praenora/simulation-core`, public, MIT or Apache-2.0
  (builder to match program norm; Apache-2.0 aligns with BioGears provenance).
  Scaffolded via `/viva-workspace` (standalone mode), package `pbg_simulation_core`.
- **Dependencies:** `process-bigraph`, `viva-core`, `viva-tellurium`,
  `viva-biogears`, `viva-emitters` (as needed). Editable installs from `~/code`
  checkouts where present (`viva-tellurium` already cloned), GitHub otherwise.
  BioGears native runtime installed via `pybiogears` installer on the run host.
- **Investigation (`/viva-investigation`):** "BioGears sepsis core: engine vs SBML
  extraction." Member studies:
  - **S1 — healthy baseline parity:** both sides at rest, confirm matched
    baseline (≈114/73.5, HR 72, CO 5.9 L/min) before any infection. Gate: within
    a stated tolerance, else alignment is wrong and later arms are meaningless.
  - **S2 — mild infection (severity 1):** both sides, compare resolution.
  - **S3 — severe infection (severity 3):** both sides, compare the collapse
    sequence and quantify the ~8–10 h delay.
  - Each study: a baseline composite (viva-biogears `biogears_sepsis`) and a
    variant (viva-tellurium running `biogears_sepsis.xml`), matched scenario,
    emitted observables aligned on a common clock.
  - **Severity reconciliation (resolved by recon):** both sides drive the same
    McDaniel-2019 AIR model from the same `severity` 1/2/3 enum. SBML seeds initial
    pathogen load (1e6/5e6/2.5e7); `viva-biogears` `infection_severity` 1/2/3 sets
    the initial pathogen load once at t=0 (MIC inert without antibiotics). Build
    step verifies BioGears' per-severity initial load equals the Antimony values
    (report notes BioGears code uses 2.5e7 for severe, which the extraction copied).
  - **Scenario-match caveat:** `viva-biogears`' sepsis runs BioGears'
    `InfectionModerate` nutrition schedule (hourly water + 5 meals/48 h) for every
    arm; the Antimony extraction does not model meals/energy. S1 baseline-parity
    gate catches whether this matters; documented as a known scenario delta.
  - **Observables (Q5):** blood/local pathogen load and SVR exist on both sides;
    "tissue integrity" is an Antimony-side scalar with no direct BioGears
    counterpart — mapped to a BioGears tissue-fluid/inflammation proxy or shown
    SBML-only, scoped per observable in the report (builder's call).
  - **S4 — engine runtime + cross-engine determinism.** Run the *same* composite
    (the SBML/Tellurium side, as the portable `.pbg` document) under both the
    Python `process-bigraph` engine and the Rust `viva-core` engine. Measure
    wall-clock per horizon and confirm results are **bit-identical / within
    tolerance** across engines (viva-core's interchangeability claim on a real
    physiology workload). **Scope note:** for a single heavy libroadrunner-backed
    process the engine swap measures framework/tick overhead, not solver time, so
    expect a modest delta — not viva-core's ~160× fine-grained figure; reported
    honestly as overhead, with a note that fine-grained decomposition is where the
    speedup would appear. **Gate:** a quick smoke-test that `viva-core` (Milestone
    1 MVP) can drive the Python-callback `TelluriumProcess` at all before
    committing S4; if it can't yet, S4 is deferred, not faked.
- **Source artifacts** committed under the investigation (decided: model files +
  report both go in the public repo): `biogears_sepsis.ant`, `.xml`, the extraction
  report (pdf/docx, Herbert Sauro), with a provenance note crediting Sauro,
  Apache-2.0 BioGears, and McDaniel 2019. These are untrusted downloads → copied
  in, not executed from their download dir.
- **Comparison viz (`/viva-viz`):** overlaid time-series per observable (engine vs
  SBML, mild vs severe) + a milestone-divergence table mirroring the report's
  validation table, extended with the engine's own numbers.
- **Runtime API:** thin — a documented entry point to construct and run either
  side of the comparison as a `process_bigraph.Composite`, and to pull aligned
  observables. Grows only as the investigation needs it.
- **Run host:** mini preferred for the long arms (Ray/parallel); baseline/mild may
  run locally once BioGears is built. Confirm mini build before committing arms.

## What's left to the builder

- License choice (Apache-2.0 recommended for BioGears-derived provenance).
- Exact observable set and alignment tolerances for the S1 baseline gate.
- Common-clock resampling scheme for overlaying per-beat (BioGears) vs
  continuous (Antimony) outputs.
- Whether to run arms on laptop vs mini, and whether to shorten the horizon for a
  first smoke-test pass before the full 48 h arms.
- Standard viva-workspace scaffolding details (CI, README gen, read-only workbench
  publish) per workspace conventions.
