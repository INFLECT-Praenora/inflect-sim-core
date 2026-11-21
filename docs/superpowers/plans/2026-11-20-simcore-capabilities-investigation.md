# Simulation-core capabilities investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the `biogears-sbml-sepsis` investigation into one comprehensive, INFLECT-facing demonstration that *shows* (not tells) simulation-core's four capability pillars — composition, ensembles, orchestration/workflows, performance — on one real sepsis model across interchangeable engines.

**Architecture:** The investigation is workspace content: one `investigation.yaml` + per-study `study.yaml` + chart scripts/artifacts, rendered by the workbench and publishable read-only. Each study leads with a *runnable* artifact (a composite, a live ensemble grid, a callable workflow, a live benchmark). New runnable demos are small Python scripts under each study's `charts/` or `run/` dir that execute live (cheap engines) and bake JSON + figures; heavy BioGears runs are cited from committed `meta.wall_s`.

**Tech Stack:** process-bigraph `Composite` + `build_core()`; the unified workflow API (`viva_simulation_core/api/workflow*.py`); engines `TelluriumProcess`/`TelluriumUTCStep`, `BioGearsProcess`, `NeuralProcess` (viva-torch), `MathExpressionStep`; matplotlib for figures; pytest for verification; `scripts/lint-workspace.py` + the workbench schema validators.

**Spec:** `docs/superpowers/specs/2026-11-20-vivarium-scaffolding-sepsis-investigation-design.md`

## Global Constraints

- **Reader-facing title (exact):** `Simulation core — one sepsis model, every engine: composed, ensembled, orchestrated, benchmarked`. No "vivarium" in any reader-facing title/study-name/framing; lead with the capability.
- **Investigation slug stays** `biogears-sbml-sepsis` (dir names, URL, cross-refs unchanged).
- **Run budget:** cite committed `meta.wall_s` for the ~3.38 h BioGears + ~2.81 h Tellurium full runs; run live only the cheap new work (Tellurium short runs, surrogate inference, the handoff workflow, per-step micro-benchmarks). No 3 h re-runs.
- **No `sweep:` schema field exists.** Ensembles are expressed via `baseline[]` engine arms + `variants[]` severities, plus a live gather-script that runs the cheap engine×severity cells and bakes the grid.
- **Surrogate honesty:** label the surrogate a fast *approximate* tier (bounded, not yet clinically accurate) everywhere it appears.
- **Surrogate asset:** fold `surrogate_stable2.pt` from branch `neural-surrogate-sepsis` into the workspace; downstream studies load it by path via `NeuralProcess({"checkpoint": <path>})`.
- **study.yaml shape (v3):** `schema_version: 3`, `name`, `investigation`, `status`, `title`, `confidence`, `parent_studies`, `question`, `objective`, `claim`, `expected_behavior[] {name,statement,status}`, `findings[] {summary,detail,status}`, `visualizations[] {name,address,display}`, `baseline[] {name,composite,config,note}`, `variants`, `interventions`, `runs`.
- **Process:** no AI attribution on commits/PRs; work only in this worktree (`investigation/simcore-scaffolding`); the investigation PR opens as **draft** with an `investigation:` title; never auto-merge.
- **Verification floor:** every task ends with `python scripts/lint-workspace.py` → `workspace lint: OK` AND the task's own pytest green. Tests run with the worktree on `PYTHONPATH` (editable install points at the canonical checkout).

## Review Focus

- **Ensemble grid with a dead engine cell** (study 02): a surrogate or Tellurium cell that errors/NaNs must render as a visible failed cell, not silently drop — the grid must always show all 2×3 cells. Covered by Task 4's test asserting finite values in every cell + an explicit failure marker path.
- **Workflow handoff on the real artifact** (study 03): the workflow must load the real `sources/biogears_sepsis.ant` (not the toy `SMALL_MODEL`), or the "SBML file passed between engines" claim is false. Covered by Task 5's test asserting the Tellurium node's config references the real `.ant` and the run emits real sepsis observables.
- **Cited-vs-measured provenance** (study 04): every performance number is tagged as committed-run (BioGears/Tellurium full) or live-measured (micro-benchmark), never blurred. Covered by Task 6's test asserting the chart `meta.json` provenance names the source for each bar.
- **Superseded investigation left runnable** (Task 8): retiring `neural-surrogate-sepsis` must not break its branch/PR #8 history; supersession is a pointer + status, not a deletion. Covered by Task 8 keeping the branch intact and only adding a supersede marker.
- **Reader-facing framework leakage:** a study title or `at_a_glance` role that says "vivarium"/"process-bigraph" violates the title constraint. Covered by Task 1 + Task 8 tests grepping reader-facing fields for banned tokens.

---

### Task 1: Reframe the investigation shell + fold in framing (00) + surrogate asset

**Files:**
- Modify: `workspace/investigations/biogears-sbml-sepsis/investigation.yaml`
- Create (cherry-pick, then reframe): `workspace/investigations/biogears-sbml-sepsis/studies/00-simulation-core/study.yaml` from branch `biogears-sbml-sepsis-simcore`, **reframed to framing-only** (at-a-glance of the four pillars; a light overview). **Do NOT carry 00's runtime-compare charts or `bench_live.py`/`make_runtime_compare.py` here** — those belong to study 04 (Task 5 cherry-picks them straight from the simcore branch). 00 owns no heavy figure; if it needs a visual, a small 4-pillar overview graphic.
- Create: `workspace/investigations/biogears-sbml-sepsis/assets/surrogate_stable2.pt` (copied from branch `neural-surrogate-sepsis:workspace/investigations/neural-surrogate-sepsis/studies/01-sample-and-train/out/surrogate_stable2.pt`) + `assets/surrogate_meta.json` (copied `eval_metrics.json`)
- Test: `tests/investigation/test_investigation_shell.py`

**Interfaces:**
- Produces: the reframed `investigation.yaml` with `title` (exact, from Global Constraints), `studies: [00-simulation-core, 01-composition, 02-ensembles, 03-workflows, 04-performance, 05-fidelity]`, reframed `at_a_glance[]` (role text per the design's pillar mapping), retained `how_to_read` + `glossary` (cherry-picked), updated `acceptance_criteria[]`. The surrogate checkpoint path `workspace/investigations/biogears-sbml-sepsis/assets/surrogate_stable2.pt` (consumed by Tasks 4/5/6).

- [ ] **Step 1: Write the failing test** `tests/investigation/test_investigation_shell.py`:
```python
import yaml, pathlib
INV = pathlib.Path("workspace/investigations/biogears-sbml-sepsis/investigation.yaml")
BANNED = ("vivarium", "process-bigraph", "process bigraph")
def test_title_and_members():
    d = yaml.safe_load(INV.read_text())
    assert d["title"] == "Simulation core — one sepsis model, every engine: composed, ensembled, orchestrated, benchmarked"
    assert d["studies"] == ["00-simulation-core","01-composition","02-ensembles","03-workflows","04-performance","05-fidelity"]
    assert "how_to_read" in d and len(d["glossary"]) >= 7
def test_no_framework_in_reader_facing_text():
    d = yaml.safe_load(INV.read_text())
    blob = (d["title"] + " " + " ".join(a["role"] for a in d["at_a_glance"])).lower()
    for b in BANNED: assert b not in blob
def test_surrogate_asset_present():
    assert pathlib.Path("workspace/investigations/biogears-sbml-sepsis/assets/surrogate_stable2.pt").exists()
```
- [ ] **Step 2: Run it, verify it fails** — `pytest tests/investigation/test_investigation_shell.py -q` → FAIL (members/title mismatch, asset missing).
- [ ] **Step 3: Cherry-pick 00 + surrogate asset** — `git checkout biogears-sbml-sepsis-simcore -- workspace/investigations/biogears-sbml-sepsis/studies/00-simulation-core`; copy the surrogate `.pt` + `eval_metrics.json` out of `neural-surrogate-sepsis` (`git show neural-surrogate-sepsis:…/out/surrogate_stable2.pt > assets/surrogate_stable2.pt` via `git cat-file` for the binary).
- [ ] **Step 4: Rewrite `investigation.yaml`** — new `title`, `question`/thesis (four-pillar tour), `studies[]` (6 new names), `at_a_glance[]` roles mapped to pillars (00 framing, 01 Composition, 02 Ensembles, 03 Orchestration/Workflows, 04 Performance, 05 Fidelity), retain cherry-picked `how_to_read`+`glossary`, update `acceptance_criteria[]` to the new study names. No banned tokens in reader-facing fields.
- [ ] **Step 5: Run tests + lint** — `pytest tests/investigation/test_investigation_shell.py -q` PASS; `python scripts/lint-workspace.py` → `workspace lint: OK`.
- [ ] **Step 6: Commit** — `git commit -m "investigation(simcore): reframe shell to four-pillar capabilities tour + fold in 00 + surrogate asset"`

---

### Task 2: Study 01 — Composition (one spec, interchangeable engines)

**Files:**
- Create: `workspace/investigations/biogears-sbml-sepsis/studies/01-composition/study.yaml`
- Create: `.../studies/01-composition/charts/make_engine_overlay.py` (+ emits `01_engine_overlay.{svg,png,meta.json}`)
- Reuse (read-only): committed parity trajectories from old `01/02/03` charts for reference numbers
- Test: `tests/investigation/test_study_01_composition.py`

**Interfaces:**
- Consumes: surrogate asset (Task 1), `sepsis_sbml` + `sepsis_engine` composites.
- Produces: study `01-composition` with `baseline[]` anchored on `sepsis_sbml` and a `claim` that the same spec runs under three engines by one node swap; the overlay figure.

- [ ] **Step 1: Write failing test** — assert `study.yaml` loads, `name=="01-composition"`, `baseline[0].composite` endswith `sepsis_sbml`, and `charts/01_engine_overlay.svg` exists after the chart script runs.
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Write `make_engine_overlay.py`** — run `sepsis_sbml` (Tellurium) and the surrogate at a matched severity for a short horizon via `build_core()` + `Composite`, overlay the shared observables (MAP/HR at minimum) on one figure; BioGears trace cited from committed parity meta (annotated, not re-run). Write `.svg/.png/.meta.json` with per-trace provenance.
- [ ] **Step 4: Write `study.yaml`** — question/objective/claim per design §01; `expected_behavior` = "same-spec-three-engines", "swap-is-one-node"; `findings` lead with the overlay; `visualizations` → the overlay (display: primary); `baseline` = sepsis_sbml (+ note on the engine-swap arms).
- [ ] **Step 5: Run chart + tests + lint** → all green.
- [ ] **Step 6: Commit** — `investigation(simcore): study 01 composition — one spec, interchangeable engines`

---

### Task 3: Study 02 — Ensembles (severity × engine in one batch) [NEW]

**Files:**
- Create: `.../studies/02-ensembles/study.yaml`
- Create: `.../studies/02-ensembles/charts/make_ensemble_grid.py` (+ emits `01_ensemble_grid.{svg,png,meta.json}` and `ensemble_results.json`)
- Test: `tests/investigation/test_study_02_ensembles.py`

**Interfaces:**
- Consumes: surrogate asset; `sepsis_sbml` composite; `build_core()`.
- Produces: `ensemble_results.json` = `{engine: {severity: {endpoint: value}}}` for engines {tellurium, surrogate} × severities {0,1,3}; the grid figure. BioGears severe cell cited from committed endpoint (flagged `source:"committed"`).

- [ ] **Step 1: Write failing test** — after running `make_ensemble_grid.py`, assert `ensemble_results.json` has exactly engines {"tellurium","surrogate"} each with severities {0,1,3}, every live cell a finite number, and every cell carries a `source` tag (`"live"` or `"committed"`); assert `01_ensemble_grid.svg` exists.
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Write `make_ensemble_grid.py`** — loop engines×severities; for each live cell build the composite at that severity, run a short horizon, extract one endpoint (e.g., min MAP or final tissue integrity); catch per-cell errors into a visible `{"value":None,"error":...}` marker (Review Focus); render a heatmap/grid; write `ensemble_results.json` + figure. BioGears severe = cited constant.
- [ ] **Step 4: Write `study.yaml`** — Ensembles framing; `baseline[]` = engine arms (tellurium, surrogate); `variants[]` = severities {1,3}; `findings` lead with the grid ("one orchestrated batch over severity × engine"); `visualizations` → grid (primary).
- [ ] **Step 5: Run chart + tests + lint** → green.
- [ ] **Step 6: Commit** — `investigation(simcore): study 02 ensembles — severity × engine sweep in one batch`

---

### Task 4: Study 03 — Orchestration/Workflows + SBML hand-off (centerpiece) [NEW]

**Files:**
- Create: `.../studies/03-workflows/study.yaml`
- Create: `.../studies/03-workflows/run/run_handoff_workflow.py` (+ emits `handoff_result.json`)
- Create: `.../studies/03-workflows/charts/make_handoff_dag.py` (+ emits `01_handoff_dag.{svg,png,meta.json}`)
- Test: `tests/investigation/test_study_03_workflows.py`

**Interfaces:**
- Consumes: the four-verb workflow API (`viva_simulation_core/api/workflow.py`, `workflow_models.py`); `sources/biogears_sepsis.ant`; surrogate asset; `TelluriumProcess`/`TelluriumUTCStep`, `MathExpressionStep`, `NeuralProcess`.
- Produces: `handoff_result.json` = the workflow run output (the compared/derived observable + the surrogate tier); the hand-off DAG figure.

- [ ] **Step 1: Write failing test** — assert `run_handoff_workflow.py` builds a workflow whose Tellurium node `config.model_file`/`model` references the real `biogears_sepsis.ant` (NOT `SMALL_MODEL`), runs it through the workflow API, and `handoff_result.json` holds a finite derived observable; assert the DAG svg exists.
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Write `run_handoff_workflow.py`** — narrate + exercise the four verbs against the real spec: (list modules) → (call one: Tellurium on `biogears_sepsis.ant`) → (connect: a workflow where the Antimony/SBML artifact feeds Tellurium whose outputs wire into `MathExpressionStep`, with a `NeuralProcess` surrogate tier) → (run). Base it on `tiered_workflow_spec` but swap the toy `SMALL_MODEL` for the real `.ant`. Write `handoff_result.json`.
- [ ] **Step 4: Write `make_handoff_dag.py`** — render the workflow graph (nodes + wires) with the shared `.ant` artifact called out as the inter-engine interface.
- [ ] **Step 5: Write `study.yaml`** — Orchestration/Workflows framing; findings lead with "the four verbs" + "an Antimony/SBML file passed between engines"; visualizations → DAG (primary); link to `handoff_result.json`.
- [ ] **Step 6: Run both scripts + tests + lint** → green.
- [ ] **Step 7: Commit** — `investigation(simcore): study 03 orchestration/workflows — compose modules, hand off an SBML file`

---

### Task 5: Study 04 — Performance (consolidate 04+06 + surrogate)

**Files:**
- Create: `.../studies/04-performance/study.yaml`
- Cherry-pick from `biogears-sbml-sepsis-simcore:.../studies/00-simulation-core/charts/`: `bench_live.py`, `make_runtime_compare.py` → `.../studies/04-performance/charts/` (these live on study 00 on the simcore branch; they belong to Performance here — 00 is framing-only per Task 1)
- Create/refresh: `.../studies/04-performance/charts/01_runtime_compare.{svg,png,meta.json}` + `bench_live.json`
- Test: `tests/investigation/test_study_04_performance.py`

**Interfaces:**
- Consumes: committed `wall_s` from old `04-engine-runtime` + the `06-ode-integrator` stiffness finding; surrogate asset; `bench_live.py`.
- Produces: `bench_live.json` (live per-step ratio) + the two-panel runtime figure with per-bar provenance.

- [ ] **Step 1: Write failing test** — run `bench_live.py`; assert `bench_live.json` has `exact.per_step_ms` and `surrogate.per_step_ms` finite with exact ≫ surrogate; assert the figure `meta.json` provenance names each bar's source (committed vs live).
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Cherry-pick + adapt** `bench_live.py` + `make_runtime_compare.py` (fix the `parents[...]` path for the new study location); re-run to regenerate `bench_live.json` + figure.
- [ ] **Step 4: Write `study.yaml`** — merge 04 (full-run wall_s, cross-engine determinism) + 06 (stiff ODE ⇒ a faster/Rust integrator can't help) + the surrogate 40,000× into ONE performance narrative; findings: full-run profile, stiff-ODE bottleneck, live per-step ratio, surrogate speedup (approximate tier); visualizations → two-panel figure (primary).
- [ ] **Step 5: Run + tests + lint** → green.
- [ ] **Step 6: Commit** — `investigation(simcore): study 04 performance — compute profile + the 40,000x surrogate`

---

### Task 6: Study 05 — Fidelity & divergence (keep 05 + fold parity)

**Files:**
- Create: `.../studies/05-fidelity/study.yaml`
- Cherry-pick (read-only reuse): committed charts from old `05-divergence-mechanism` + `01/02/03` parity charts into `.../studies/05-fidelity/charts/`
- Test: `tests/investigation/test_study_05_fidelity.py`

**Interfaces:**
- Consumes: committed divergence + parity chart artifacts (no re-run).
- Produces: `05-fidelity` study presenting "do the engines agree, and where not": parity (healthy/mild/severe match + lag) then the divergence mechanism.

- [ ] **Step 1: Write failing test** — assert `study.yaml` loads, `name=="05-fidelity"`, references the divergence + parity charts that exist on disk, and `claim` is honest (contains both agreement and divergence).
- [ ] **Step 2: Run it, verify it fails.**
- [ ] **Step 3: Cherry-pick charts** from old 05 + 01/02/03; **Step 4: Write `study.yaml`** folding parity + divergence into one honest fidelity study (keeps the credibility backbone).
- [ ] **Step 5: Tests + lint** → green.
- [ ] **Step 6: Commit** — `investigation(simcore): study 05 fidelity — do the engines agree, and where not`

---

### Task 7: Remove the old studies + wire the new study set

**Files:**
- Delete: old `studies/{01-baseline-parity,02-mild-infection,03-severe-infection,04-engine-runtime,05-divergence-mechanism,06-ode-integrator-runtime}` (after their reusable charts are cherry-picked into the new studies in Tasks 2/5/6)
- Modify: any `parent_studies`/cross-refs pointing at old slugs
- Test: `tests/investigation/test_no_orphans.py`

**Interfaces:** Consumes the six new studies; produces a clean on-disk study set matching `investigation.yaml:studies[]`.

- [ ] **Step 1: Write failing test** — assert the set of `studies/*/` dirs on disk == `investigation.yaml:studies[]` exactly (no old slugs, no orphans), and every `at_a_glance.study` + `acceptance_criteria.study` resolves to an existing dir.
- [ ] **Step 2: Run it, verify it fails** (old dirs still present).
- [ ] **Step 3: Delete old study dirs** (only after confirming their reused charts were copied).
- [ ] **Step 4: Fix any dangling cross-refs.**
- [ ] **Step 5: Tests + lint** → green; **Step 6: Commit** — `investigation(simcore): retire old validation studies; study set matches the four-pillar map`

---

### Task 8: Supersede neural-surrogate-sepsis + render + finalize

**Files:**
- Modify: `investigation.yaml` (final `at_a_glance` ordering, executive/thesis polish)
- Create: a supersede marker for `neural-surrogate-sepsis` (a `SUPERSEDED.md` note on THIS branch referencing where its surrogate content now lives; do NOT touch the other branch)
- Test: full workspace render/lint

**Interfaces:** Consumes all six studies; produces the finished, render-clean investigation.

- [ ] **Step 1: Write failing test** `tests/investigation/test_renders_end_to_end.py` — build the read-only bundle for this investigation (`vivarium-workbench-publish --workspace . --out <tmp>`), assert exit 0 and that each of the 6 studies produced a shell + the baked `composite-resolve/<id>.json` for study 01's baseline (snapshot parity).
- [ ] **Step 2: Run it, verify it fails** (if any study is unrenderable).
- [ ] **Step 3: Finalize `investigation.yaml`** (at_a_glance order, executive verdict framed as the capabilities demonstration); add the `neural-surrogate-sepsis` supersede pointer; run `/viva-report` (or `vivarium-workbench` render) to regenerate reports.
- [ ] **Step 4: Full verification** — `pytest tests/investigation -q` all green; `python scripts/lint-workspace.py` OK; publish-render test green.
- [ ] **Step 5: Commit** — `investigation(simcore): finalize capabilities tour + supersede neural-surrogate-sepsis`

---

## Post-plan (not tasks)
- Open the investigation PR as **draft** with title `investigation: simulation core — one sepsis model, every engine (composition · ensembles · orchestration · performance)`; body per `AGENTS.md` (not a merge target; names the merged feature PRs it depends on — the workflow API #9). Close PR #8 (neural-surrogate-sepsis) as superseded (do not merge). Re-publish the read-only workbench from `main` only after any infrastructure feature PRs land.
