# simulation-core: BioGears ↔ SBML sepsis comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the public `INFLECT-Praenora/simulation-core` viva workspace and run its first investigation — the BioGears sepsis core as the real BioGears engine vs the SBML/Antimony extraction, side by side, plus a Python-vs-Rust runtime axis.

**Architecture:** `simulation-core` is a viva workspace (`workspace.yaml` + `viva_simulation_core/`) depending on `process-bigraph`, `viva-core`, `viva-tellurium`, `viva-biogears`. The science comes from two existing composites: `viva-biogears` `biogears_sepsis` (real engine) and a `viva-tellurium` composite running `biogears_sepsis.xml`. The workbench hosts the investigation + studies; the runtime axis swaps the Python `process-bigraph` engine for the Rust `viva-core` engine on the same `.pbg` document.

**Tech Stack:** Python 3.12, process-bigraph, viva-core (Rust/PyO3), viva-tellurium (libroadrunner), viva-biogears (pybiogears → BioGears 8.2.0 C++), vivarium-workbench, uv.

**Spec:** `docs/superpowers/specs/2026-11-15-simulation-core-sepsis-sbml-design.md` (travels with this plan; executors read both).

## Global Constraints

- Repo is **public**, org `INFLECT-Praenora`, name `simulation-core`. License **Apache-2.0** (BioGears-derived provenance).
- Dependencies: `process-bigraph`, `viva-core`, `viva-tellurium`, `viva-biogears`, `viva-emitters`. Approach A — depend on the vivarium-collective repos; do **not** fork them into INFLECT now.
- **No fabrication.** Every reported number comes from a real run. If the BioGears engine can't be built, the engine side is marked **pending**, never stubbed (spec Fallback). Same for S4 if `viva-core` can't drive the process.
- **No AI attribution** in commits or PR bodies. End commits with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. PR body footer `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. (User CLAUDE.md forbids the Co-Authored-By trailer — **omit it**; keep commits clean.)
- **Never auto-merge**; the user approves merges.
- Severity maps directly: `severity`/`infection_severity` 1/2/3 = mild/moderate/severe, both seeding the McDaniel-2019 AIR initial pathogen load; no antibiotics so MIC is inert.
- Source artifacts in `~/Downloads/Tien/` are untrusted downloads: **copy** into the repo, never run an interpreter from that directory.
- Working checkout: `~/code/simulation-core` (brand-new repo, single session — no shared-worktree hazard). Long arms run on the **mini** (`mct`/Ray); verify the mini is reachable before committing to engine/severe arms.

## Review Focus

- **Mismatched scenario reported as agreement** — if the two sides run different patient/severity/start/horizon, the comparison is meaningless. → S1 baseline-parity gate (Task 7) and an explicit scenario-equality assertion before S2/S3.
- **Antimony model fails to load/step in Tellurium** (SBML L3V2, 91 states) — a load error must surface loudly, not yield an empty trace. → Task 5 smoke test asserts non-empty emitted observables.
- **BioGears engine absent** — import/build missing must mark engine side pending, not silently pass. → Task 4 asserts `pybiogears` import + a real short engine run; Task 6 skips-with-notice if absent.
- **viva-core can't drive a Python-callback process** — S4 must gate on a smoke test, not assume. → Task 9 Step 1 is the gate.
- **Cross-engine results differ** — a timing win with divergent numbers is a correctness bug, not a speedup. → Task 9 asserts within-tolerance equality before reporting any timing.

---

### Task 1: Create repo + scaffold workspace

**Files:**
- Create: `~/code/simulation-core/` (via `/viva:workspace` standalone)
- Create: `~/code/simulation-core/docs/superpowers/specs/2026-11-15-simulation-core-sepsis-sbml-design.md` (copy of approved spec)
- Create: `~/code/simulation-core/docs/superpowers/plans/2026-11-15-simulation-core-sepsis-sbml-plan.md` (this plan)

**Interfaces:**
- Produces: a viva workspace at `~/code/simulation-core` with `workspace.yaml`, package `viva_simulation_core/`, git initialized, remote `origin` → `git@github.com:INFLECT-Praenora/simulation-core.git`.

- [ ] **Step 1:** Create the public repo: `gh repo create INFLECT-Praenora/simulation-core --public --license apache-2.0 -d "API + runtime for INFLECT (ARPA-H CIRCLE): process-bigraph / viva-core, with the BioGears↔SBML sepsis investigation"`.
- [ ] **Step 2:** Scaffold the workspace with `/viva:workspace` (standalone mode), package name `viva_simulation_core`, into `~/code/simulation-core`. Point `origin` at the repo from Step 1.
- [ ] **Step 3:** Copy the approved spec + this plan into `docs/superpowers/{specs,plans}/`.
- [ ] **Step 4: Verify** — run `/viva-status` in `~/code/simulation-core`; expect it to detect `workspace.yaml`, report the package and a clean/initialized git state.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "chore: scaffold simulation-core workspace + design/plan"`; push `main`.

---

### Task 2: Declare and install dependencies

**Files:**
- Modify: `~/code/simulation-core/pyproject.toml` (add deps)
- Modify: `~/code/simulation-core/workspace.yaml` (register module deps if the workspace tracks them)

**Interfaces:**
- Consumes: Task 1 workspace.
- Produces: importable `process_bigraph`, `viva_core`, `viva_tellurium`, `viva_biogears`, `viva_emitters` in the workspace venv.

- [ ] **Step 1:** Add the five dependencies. Prefer editable installs from existing `~/code` checkouts where present (`viva-core`, `viva-tellurium` are cloned); clone the others from `github.com/vivarium-collective` into `~/code` and install editable. Use the workspace's uv venv.
- [ ] **Step 2: Write the failing test** `tests/test_deps_import.py::test_core_deps_import` — imports all five modules.
- [ ] **Step 3:** Run it; expect FAIL (modules not yet installed).
- [ ] **Step 4:** Install the deps into the venv.
- [ ] **Step 5:** Run the test; expect PASS. (`viva_biogears` imports without a native build; engine calls are what need it.)
- [ ] **Step 6: Verify** — `/viva:catalog list` shows the installed modules.
- [ ] **Step 7: Commit** — `deps: add process-bigraph, viva-core, viva-tellurium, viva-biogears, viva-emitters`.

---

### Task 3: Vendor the sepsis source artifacts + provenance

**Files:**
- Create: `~/code/simulation-core/workspace/investigations/biogears-sbml-sepsis/sources/biogears_sepsis.ant`
- Create: `.../sources/biogears_sepsis.xml`
- Create: `.../sources/BioGears sepsis core in Antimony extraction report.pdf` (+ `.docx`)
- Create: `.../sources/PROVENANCE.md`

**Interfaces:**
- Produces: the canonical in-repo path to `biogears_sepsis.xml` used by Tasks 5, 8, 9.

- [ ] **Step 1:** Copy the four artifacts from `~/Downloads/Tien/` into `sources/` (copy, do not execute from the download dir).
- [ ] **Step 2:** Write `PROVENANCE.md`: authored by Herbert Sauro; model translated from BioGears (Apache-2.0, `BioGearsEngine/core`); references McDaniel et al. 2019 (Front Physiol 10:1321); note the known quirks from the report header (`(1-PT)`, merged gut node, 2.5e7 severe seed).
- [ ] **Step 3: Verify** — files present, `biogears_sepsis.xml` is valid XML (`python -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('.../biogears_sepsis.xml')"`).
- [ ] **Step 4: Commit** — `data: add BioGears sepsis Antimony/SBML extraction + provenance (Sauro)`.

---

### Task 4: Build the BioGears native engine (prerequisite for the engine side)

**Files:**
- Host build only (no repo files). Target install: `~/biogears-8.2.0-install`.

**Interfaces:**
- Produces: importable `pybiogears` driving BioGears 8.2.0; `$BIOGEARS_RUNTIME` set.

- [ ] **Step 1:** Decide the run host. Default **mini** (long arms live there); confirm it is reachable (`ssh mini true`). If the mini is down, build on the laptop for short arms.
- [ ] **Step 2:** On the chosen host, clone `vivarium-collective/pybiogears`, run `scripts/install_biogears.sh` (BioGears 8.2.0 → `~/biogears-8.2.0-install`), then install `pybiogears` into the simulation-core venv with `CMAKE_PREFIX_PATH=~/biogears-8.2.0-install`.
- [ ] **Step 3: Write the failing test** `tests/test_biogears_engine.py::test_engine_short_run` — constructs a `viva_biogears` process and steps a short run, asserting a plausible vital (e.g. MAP in a physiological range). Mark it to **skip with a clear reason** when `pybiogears`/runtime is absent (mirror viva-biogears' own skip pattern), so CI stays green.
- [ ] **Step 4:** Run it; expect FAIL before the build, SKIP on a host without the engine.
- [ ] **Step 5:** With the engine built, run it; expect PASS on the run host.
- [ ] **Step 6: Commit** — `test: BioGears engine smoke test (skips without native build)`.

---

### Task 5: SBML-side baseline composite (Tellurium)

**Files:**
- Create: `viva_simulation_core/composites/sepsis_sbml.py` — builds a `viva_tellurium` composite loading `sources/biogears_sepsis.xml`, emitting the comparison observables.
- Test: `tests/test_sepsis_sbml.py`

**Interfaces:**
- Consumes: `sources/biogears_sepsis.xml` (Task 3).
- Produces: `make_sepsis_sbml_composite(severity: int = 0, infection_start_s: float | None = None, horizon_s: float, ...) -> dict` (a `.pbg` document) and `SEPSIS_OBSERVABLES` (the shared observable list: SBP, DBP, MAP, HR, CO, SVR, blood_volume, tissue_integrity, blood_pathogen).

- [ ] **Step 1: Write the failing test** `test_sbml_loads_and_steps` — build the composite at `severity=0` (healthy), run a short horizon, assert emitted observables are non-empty and the healthy baseline sits near the spec targets (SBP≈114, DBP≈73.5, HR≈72, CO≈5.9 L/min) within a stated tolerance.
- [ ] **Step 2:** Run it; expect FAIL (composite not defined).
- [ ] **Step 3:** Implement `make_sepsis_sbml_composite` + `SEPSIS_OBSERVABLES` using `viva_tellurium`'s process + `make_*_document` helpers (map `severity`→initial PT, `infectionStart`, `infScale`). Observable name mapping lives here.
- [ ] **Step 4:** Run it; expect PASS.
- [ ] **Step 5: Commit** — `feat: Tellurium SBML sepsis baseline composite`.

---

### Task 6: BioGears-side baseline composite (engine)

**Files:**
- Create: `viva_simulation_core/composites/sepsis_engine.py` — thin wrapper selecting `viva_biogears`' `biogears_sepsis` composite with the matched scenario + the same `SEPSIS_OBSERVABLES` (mapped to BioGears request names).
- Test: `tests/test_sepsis_engine.py`

**Interfaces:**
- Consumes: `viva_biogears.biogears_sepsis`, `SEPSIS_OBSERVABLES` (Task 5).
- Produces: `make_sepsis_engine_composite(severity, infection_start_s, horizon_s, ...) -> dict`, with observable names reconciled to BioGears (tissue_integrity → documented proxy or marked engine-absent).

- [ ] **Step 1: Write the failing test** `test_engine_baseline_parity` — healthy (`severity=0`) short run, assert baseline near the same spec targets as Task 5. **Skip with notice** when the engine is absent (Task 4 pattern).
- [ ] **Step 2:** Run; expect FAIL/SKIP.
- [ ] **Step 3:** Implement `make_sepsis_engine_composite` wrapping `biogears_sepsis` (StandardMale, matched severity, `InfectionModerate` nutrition schedule per viva-biogears), reconciling observable names.
- [ ] **Step 4:** Run; expect PASS on the engine host.
- [ ] **Step 5: Commit** — `feat: BioGears engine sepsis baseline composite`.

---

### Task 7: Investigation + S1 baseline-parity study (workbench)

**Files:**
- Workbench state (investigation + study), written via `/viva:investigate` and `/viva:study`.
- Create: `viva_simulation_core/analysis/align.py` — `align_observables(engine_df, sbml_df, observables) -> AlignedComparison` (common-clock resample + per-observable tolerance check).
- Test: `tests/test_align.py`

**Interfaces:**
- Consumes: Tasks 5 & 6 composites.
- Produces: `/api` investigation `biogears-sbml-sepsis`; `align_observables(...)` returning aligned series + a per-observable pass/fail against tolerance; the **scenario-equality assertion** both arms must satisfy.

- [ ] **Step 1:** `/viva:investigate new` → "BioGears sepsis core: engine vs SBML extraction", overview from the spec intent.
- [ ] **Step 2: Write the failing test** `test_align_resamples_and_gates` — feed two short baseline runs; assert `align_observables` returns a common time grid and flags the baseline within tolerance (the S1 gate). Also `test_scenario_equality` — mismatched severity/patient/horizon must be rejected.
- [ ] **Step 3:** Run; expect FAIL.
- [ ] **Step 4:** Implement `align.py` (resample to a shared grid; per-observable relative tolerance; scenario-equality guard that raises on mismatch).
- [ ] **Step 5:** Run; expect PASS.
- [ ] **Step 6:** `/viva:study` — create **S1** with the two baseline composites as baseline+variant, record the parity conclusion (gate must pass before S2/S3 are trusted). If the engine is absent, S1 records the SBML baseline and marks the engine arm pending.
- [ ] **Step 7: Commit** — `feat: investigation + S1 baseline parity (alignment + scenario gate)`.

---

### Task 8: S2 (mild) and S3 (severe) infection arms

**Files:**
- Create: `workspace/investigations/biogears-sbml-sepsis/run_arms.py` — runs severity 1 and 3 on both sides over the 48 h horizon (resumable; mini/Ray-friendly).
- Workbench: S2, S3 studies + conclusions.
- Test: `tests/test_milestones.py`

**Interfaces:**
- Consumes: Tasks 5, 6, 7.
- Produces: per-arm aligned result tables + a milestone table mirroring the report's validation measures (critical bacterial load, SBP<100 timing, MAP<65, SVR-vs-baseline), extended with the engine's own numbers.

- [ ] **Step 1:** Implement `run_arms.py` (severity, horizon, host params; checkpoint/resume; writes results under the investigation `out/`). Long runs go on the **mini** (`mct`/Ray); laptop only for a shortened smoke horizon.
- [ ] **Step 2: Write the failing test** `test_milestone_extraction` — against a **short recorded fixture** (not a full 48 h run), assert the milestone extractor computes SBP<100 crossing time, MAP<65, SVR ratio from an aligned result. (Full arms are run as an operation, not a CI test.)
- [ ] **Step 3:** Run; expect FAIL; implement the milestone extractor in `analysis/`; run; expect PASS.
- [ ] **Step 4:** Launch the full arms on the run host (SBML side always; engine side when built). Verify progress via commits/`out/` artifacts, not just "running".
- [ ] **Step 5:** `/viva:study` — record **S2** and **S3**, including the severe-arm ~8–10 h delay and the untranslated-process caveats from the spec.
- [ ] **Step 6: Commit** — `feat: S2/S3 infection arms + milestone extraction`.

---

### Task 9: S4 runtime + cross-engine determinism

**Files:**
- Create: `workspace/investigations/biogears-sbml-sepsis/run_s4_engines.py` — runs the SBML `.pbg` composite under `process-bigraph` (Python) and `viva-core` (Rust), timing each.
- Test: `tests/test_cross_engine.py`

**Interfaces:**
- Consumes: Task 5 composite (the portable `.pbg` document).
- Produces: per-engine wall-clock + a determinism verdict (bit-identical / within tolerance).

- [ ] **Step 1 (GATE): Write the failing test** `test_viva_core_can_drive_tellurium` — build the SBML composite, run a handful of steps under `viva-core`. If `viva-core` (Milestone-1 MVP) cannot drive the Python-callback `TelluriumProcess`, this **fails/skips with a clear reason** and S4 is **deferred** (spec), not faked.
- [ ] **Step 2:** Run it; record whether S4 proceeds.
- [ ] **Step 3 (if gate passes): Write the failing test** `test_cross_engine_determinism` — run the same short composite under both engines, assert the observable series match within tolerance.
- [ ] **Step 4:** Implement `run_s4_engines.py` (same document, two engines via the interchangeable runtime; time each horizon; return results + timings).
- [ ] **Step 5:** Run; expect determinism PASS; capture wall-clock for a representative horizon.
- [ ] **Step 6:** `/viva:study` — record **S4**: wall-clock Python vs Rust + the determinism verdict, with the spec's honest scope note (single heavy process → overhead, modest delta; fine-grained decomposition is where viva-core's speedup appears).
- [ ] **Step 7: Commit** — `feat: S4 cross-engine runtime + determinism`.

---

### Task 10: Comparison visualization, report, publish, PR

**Files:**
- Create: `viva_simulation_core/viz/sepsis_comparison.py` (via `/viva:viz`) — overlaid time-series (engine vs SBML, mild vs severe) + the milestone-divergence table.
- Modify: README (badges, read-only workbench link) via `gen-readme`.

**Interfaces:**
- Consumes: S1–S4 results.
- Produces: the investigation report + published read-only workbench.

- [ ] **Step 1:** `/viva:viz` — build the overlay figure(s) and the milestone-divergence table (report's measures + engine numbers + the ~8–10 h delay called out).
- [ ] **Step 2:** `/viva:report` — run Pass A (reviewer-readiness) + Pass B (lint), fix findings, render the investigation report. Any pending engine/S4 arm is shown as pending, not green.
- [ ] **Step 3: Verify** — report renders; `/viva:explore decisions biogears-sbml-sepsis` shows no unresolved blockers (or only expected pending items).
- [ ] **Step 4:** Publish the read-only workbench; regenerate README badges.
- [ ] **Step 5: Commit + PR** — push; `gh pr create` into `main` with a body summarizing the investigation and its current state. **Do not merge** (user approves).

---

## Notes on execution ordering

- Tasks 1→2→3 are strictly sequential (repo → deps → data).
- Task 4 (native build) can proceed in parallel with Tasks 5/7 (SBML side needs no engine), but Task 6 and the engine arms of Tasks 8 depend on it.
- Tasks 8 and 9 involve **long real runs** best driven on the mini; they are operations with recorded fixtures for CI, not full-run unit tests.
