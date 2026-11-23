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
