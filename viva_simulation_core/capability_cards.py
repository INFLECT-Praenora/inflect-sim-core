"""Report-card Tests that earn each simulation-core study its verdict by grading the
study's actual capability claim against its committed run/artifact.

Lightweight + honest: each card checks what its study *demonstrates* (the engines
construct and run, the ensemble grid is complete and shows the severity response, the
hand-off emits a physical result, the surrogate speedup, the inflammation agreement) —
graded with a real band, not an invented clinical threshold. Imported automatically by
``build_core()`` (which walks this package), so declaring a card in a study's
``tests:`` is enough to wire it.
"""
from __future__ import annotations

import json
import sqlite3

from viva_superpowers import TestStep, TestBuilder, check, band


def _json(study, rel: str) -> dict:
    return json.loads((study.study_dir / rel).read_text(encoding="utf-8"))


def _run_count(study) -> int:
    db = study.study_dir / "runs.db"
    if not db.exists():
        return 0
    try:
        return sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
    except Exception:
        return 0


class _StudyCard(TestStep):
    slug: str = ""

    def applies(self, study) -> bool:
        return study.study_name == self.slug


class SimulationCoreCard(_StudyCard):
    name = "simulation_core_runs"
    slug = "00-simulation-core"

    def build(self, study):
        v = (TestBuilder(model_ref=study.study_name)
             .add("Simulation core",
                  check("spec_runs", "The one sepsis spec runs under the core (recorded run)",
                        float(_run_count(study)), band(1, 1e9), severity="hard"))
             .build())
        return v, "<p>One sepsis spec, run under the simulation core.</p>"


class CompositionCard(_StudyCard):
    name = "composition_demonstrated"
    slug = "01-composition"

    def build(self, study):
        d = _json(study, "charts/engines_live_check.json")
        tell = 1.0 if d.get("tellurium_builds") else 0.0
        sur = 1.0 if d.get("surrogate", {}).get("finite") else 0.0
        v = (TestBuilder(model_ref=study.study_name)
             .add("Composition",
                  check("tellurium_builds", "Tellurium engine constructs over the real 91-state model",
                        tell, band(0.5, 1.5), severity="hard"))
             .add("Composition",
                  check("surrogate_runs", "Surrogate (third interchangeable engine) runs finite",
                        sur, band(0.5, 1.5), severity="hard"))
             .build())
        return v, "<p>Same spec, three interchangeable engines (swap one box).</p>"


class EnsemblesCard(_StudyCard):
    name = "ensembles_demonstrated"
    slug = "02-ensembles"

    def build(self, study):
        cells = _json(study, "charts/ensemble_results.json")["cells"]
        n_cells = sum(len(cells[e]) for e in cells)                   # 3 engines x 3 severities = 9
        ti0 = cells["tellurium"]["0"]["tissue_integrity_min"]
        ti3 = cells["tellurium"]["3"]["tissue_integrity_min"]
        drop = ti0 - ti3                                             # severity response (full-physics engine)
        v = (TestBuilder(model_ref=study.study_name)
             .add("Ensembles",
                  check("grid_complete", "Every severity x engine cell is present",
                        float(n_cells), band(9, 9), severity="hard"))
             .add("Ensembles",
                  check("severity_response", "Tissue integrity falls with severity (full-physics engine)",
                        float(drop), band(0.1, 1.5), severity="hard"))
             .build())
        return v, "<p>Severity x engine batch, gathered into one grid.</p>"


class WorkflowsCard(_StudyCard):
    name = "handoff_demonstrated"
    slug = "03-workflows"

    def build(self, study):
        d = _json(study, "run/handoff_result.json")
        ran = 1.0 if d["workflow"]["ran"] else 0.0
        em = float(d["result"]["exact_map_final"])
        v = (TestBuilder(model_ref=study.study_name)
             .add("Orchestration",
                  check("workflow_ran", "Four-verb workflow builds + runs from registered modules",
                        ran, band(0.5, 1.5), severity="hard"))
             .add("Orchestration",
                  check("exact_map_physical", "Handed-off .ant -> exact Tellurium MAP is physical (mmHg)",
                        em, band(30, 160), severity="hard"))
             .build())
        return v, "<p>Modules composed; an Antimony/SBML file handed between engines.</p>"


class PerformanceCard(_StudyCard):
    name = "performance_profiled"
    slug = "04-performance"

    def build(self, study):
        ratio = float(_json(study, "charts/bench_live.json")["exact_vs_surrogate_per_step"])
        v = (TestBuilder(model_ref=study.study_name)
             .add("Performance",
                  check("surrogate_speedup", "Exact per-step stiff solve >> surrogate forward pass (x)",
                        ratio, band(100, 1e9), severity="hard"))
             .build())
        return v, "<p>Compute profile across engines; the surrogate speedup.</p>"


class FidelityCard(_StudyCard):
    name = "fidelity_measured"
    slug = "05-fidelity"

    def build(self, study):
        # committed severe-arm milestones (copied into the ensembles study's data/)
        data = study.study_dir.parent / "02-ensembles" / "data"
        e = json.loads((data / "engine-sev3.milestones.json").read_text(encoding="utf-8"))["tissue_integrity_min"]
        s = json.loads((data / "sbml-sev3.milestones.json").read_text(encoding="utf-8"))["tissue_integrity_min"]
        disagree_pct = abs(e - s) / abs(e) * 100.0
        v = (TestBuilder(model_ref=study.study_name)
             .add("Fidelity",
                  check("inflammation_agreement", "Inflammation core agrees engine vs SBML (|Δ| %)",
                        float(disagree_pct), band(0.0, 0.5), severity="hard"))
             .build())
        return v, "<p>Engines agree on the inflammation core (~0.1%); hemodynamics diverge.</p>"


# Physical-admissibility bounds for the sepsis observables (used by the discriminating
# control: the exact/hybrid methods stay inside these; the surrogate leaves 8 of 9).
_BOUNDS = {
    "SBP": (40.0, 250.0), "DBP": (20.0, 200.0), "MAP": (30.0, 200.0), "HR": (20.0, 250.0),
    "CO": (0.5, 20.0), "SVR": (0.05, 5.0), "blood_volume": (2000.0, 8000.0),
    "tissue_integrity": (-1e-9, 1.0001), "blood_pathogen": (-1e-6, 1e12),
}


class MethodsComparisonCard(_StudyCard):
    """Grades the cross-method comparison: the exact engines agree, the event-driven
    hybrid reproduces them AND stays physically admissible (the discriminating control
    the surrogate fails), and it does so at a competitive runtime. Reads the study's
    committed comparison artifacts (charts/*.json)."""
    name = "methods_comparison"
    slug = "10-methods-comparison"

    def build(self, study):
        fm = _json(study, "charts/fidelity_matrix.json")["hybrid_per_obs"]
        sc = _json(study, "charts/methods_scalars.json")
        traj = _json(study, "charts/hybrid_traj.json")
        surr = json.loads((study.study_dir.parent / "04-performance" / "charts"
                           / "surrogate_admissibility.json").read_text(encoding="utf-8"))

        hemo = max(fm[o] for o in ("SBP", "MAP", "HR"))              # hybrid hemodynamic error
        n_admissible = sum(1 for o, (lo, hi) in _BOUNDS.items()
                           if o in traj and min(traj[o]) >= lo and max(traj[o]) <= hi)
        frac_admissible = n_admissible / len(_BOUNDS)
        speedup = sc["full_s"] / sc["hyb_s"]

        v = (TestBuilder(model_ref=study.study_name)
             .add("Cross-engine agreement",
                  check("exact_engines_agree",
                        "Two independent exact SBML engines (COPASI vs Tellurium) agree (max rel)",
                        float(sc["fid_copasi"]), band(0.0, 1e-3), severity="hard",
                        knob=["engine"]))
             .add("Reproduction",
                  check("hybrid_hemodynamic_fidelity",
                        "Event-driven hybrid reproduces the exact engine's hemodynamics (max rel)",
                        float(hemo), band(0.0, 1e-2), severity="hard",
                        knob=["interval_s"]))
             .add("Admissibility (discriminating control)",
                  check("hybrid_admissible",
                        "Every hybrid observable stays inside physical bounds (fraction)",
                        float(frac_admissible), band(1.0, 1.0), severity="hard"))
             .add("Admissibility (discriminating control)",
                  check("surrogate_fails_admissibility",
                        "Control: the learned surrogate leaves physical bounds on most observables (count)",
                        float(surr["n_inadmissible"]), band(5.0, 9.0), severity="soft"))
             .add("Runtime",
                  check("runtime_competitive",
                        "Hybrid matches the fastest exact option (speedup vs full event-driven model)",
                        float(speedup), band(1.0, 100.0), severity="directional",
                        knob=["model_reduction"]))
             .add("Event-relocation cost",
                  check("infection_timing_residual",
                        "Blood-pathogen transient from the tick-quantized infection start (max rel)",
                        float(sc["fid_hyb_bp"]), band(0.0, 0.5), severity="soft",
                        knob=["interval_s"]))
             .build())
        html = ("<p>Exact engines agree (~5e-4); the event-driven hybrid reproduces them to "
                "~1e-3 on the hemodynamics, stays fully admissible (the surrogate leaves "
                f"{surr['n_inadmissible']}/9 bounds), and runs at {speedup:.2f}x the full model. "
                "The blood-pathogen transient is the stated tick-quantization cost.</p>")
        return v, html
