"""Task 4 — Study 03 Orchestration/Workflows + SBML hand-off (centerpiece)."""
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = ROOT / "workspace/investigations/biogears-sbml-sepsis/studies/03-workflows"


def test_study_yaml_shape():
    d = yaml.safe_load((S / "study.yaml").read_text())
    assert d["name"] == "03-workflows"
    c = d["claim"].lower()
    assert "workflow" in c and ("antimony" in c or "sbml" in c)


def test_handoff_used_the_real_artifact_and_ran():
    j = json.loads((S / "run/handoff_result.json").read_text())
    # the hand-off loaded the REAL biogears_sepsis.ant (not a toy model)
    assert j["handoff"]["artifact"].endswith("biogears_sepsis.ant")
    assert "TelluriumUTCStep" in j["handoff"]["loaded_by"]
    # the four verbs were exercised
    assert set(j["verbs"]) >= {"list", "call", "connect", "run"}
    # the workflow ran and emitted a finite derived observable
    assert j["workflow"]["ran"] is True
    val = j["workflow"]["emit"]["compare.surrogate_total"]
    assert isinstance(val, (int, float)) and val == val  # finite, not NaN


def test_dag_chart_present():
    assert (S / "charts/01_handoff_dag.svg").exists()
