# Superseded: `neural-surrogate-sepsis` investigation

The standalone **`neural-surrogate-sepsis`** investigation (branch
`neural-surrogate-sepsis`, draft PR #8) is **superseded** by this comprehensive
simulation-core capabilities investigation.

Its surrogate is **folded in here** as the third interchangeable engine:

- The stabilized checkpoint (`surrogate_stable2.pt`) and its eval metadata now live at
  `workspace/investigations/biogears-sbml-sepsis/assets/`.
- The surrogate **state builder** (`viva_simulation_core/composites/sepsis_sbml_state.py`)
  was migrated from that branch. **This is reusable package code and should be carved
  into a standalone feature PR to `main`** (per `AGENTS.md`'s feature-first rule), since
  PR #8 is being closed rather than merged.
- The surrogate appears as the third engine in **study 01 (composition)**, the live
  orchestration/speed tier in **study 02 (ensembles)** and **study 03 (workflows)**, and
  its headline **speed** result (~40,000× faster; live per-step ~13,000×) in
  **study 04 (performance)**. Its honest accuracy picture (a fast *approximate* tier —
  degenerate across severity, data-limited) is stated wherever it appears.

**Action at PR time:** close draft PR #8 as superseded (do **not** merge it separately);
open a small feature PR to `main` for `sepsis_sbml_state.py` + the checkpoint asset so the
investigation consumes merged infrastructure rather than carrying it.
