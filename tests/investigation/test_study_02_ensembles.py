"""Task 3 — Study 02 Ensembles (severity × engine batch)."""
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = ROOT / "workspace/investigations/biogears-sbml-sepsis/studies/02-ensembles"

ENGINES = {"biogears", "tellurium", "surrogate"}
SEVS = {"0", "1", "3"}


def test_study_yaml_shape():
    d = yaml.safe_load((S / "study.yaml").read_text(encoding="utf-8"))
    assert d["name"] == "02-ensembles"
    assert "ensemble" in d["claim"].lower()


def test_ensemble_results_grid():
    j = json.loads((S / "charts/ensemble_results.json").read_text(encoding="utf-8"))
    cells = j["cells"]
    assert set(cells) == ENGINES
    for eng in ENGINES:
        assert set(cells[eng]) == SEVS
        for sev, cell in cells[eng].items():
            assert cell["source"] in ("committed", "live", "healthy-baseline")
            assert "tissue_integrity_min" in cell
    # the full-physics engines carry real, finite severity-response endpoints
    for eng in ("biogears", "tellurium"):
        vals = [cells[eng][s]["tissue_integrity_min"] for s in ("0", "1", "3")]
        assert all(isinstance(v, (int, float)) for v in vals)
        assert vals[0] > vals[-1]  # tissue integrity falls as severity rises
    # the surrogate ran live across every cell (orchestration proof), flagged approximate
    for s in SEVS:
        assert cells["surrogate"][s]["ran"] is True
        assert cells["surrogate"][s]["source"] == "live"


def test_grid_chart_present():
    assert (S / "charts/01_ensemble_grid.svg").exists()
    meta = json.loads((S / "charts/01_ensemble_grid.meta.json").read_text(encoding="utf-8"))
    assert meta.get("provenance")
