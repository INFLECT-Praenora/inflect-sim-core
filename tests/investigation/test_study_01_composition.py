"""Task 2 — Study 01 Composition (one spec, interchangeable engines)."""
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = ROOT / "workspace/investigations/biogears-sbml-sepsis/studies/01-composition"


def test_study_yaml_shape():
    d = yaml.safe_load((S / "study.yaml").read_text())
    assert d["name"] == "01-composition"
    assert d["baseline"][0]["composite"].endswith("sepsis_sbml")
    # the composition claim names all three engines
    claim = d["claim"].lower()
    assert "biogears" in claim and "tellurium" in claim and "surrogate" in claim


def test_overlay_chart_present():
    assert (S / "charts/01_engine_overlay.svg").exists()
    meta = json.loads((S / "charts/01_engine_overlay.meta.json").read_text())
    assert meta.get("provenance")


def test_live_engine_check_finite():
    """The cheap live check proves the engines construct + run here (the surrogate
    full rollout is finite/bounded — an approximate tier, not asserted accurate)."""
    j = json.loads((S / "charts/engines_live_check.json").read_text())
    assert j["surrogate"]["finite"] is True
    assert j["surrogate"]["n_steps"] >= 500
    assert j["tellurium_builds"] is True
