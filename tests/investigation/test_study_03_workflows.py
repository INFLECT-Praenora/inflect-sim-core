"""Task 4 — Study 03 Orchestration/Workflows + SBML hand-off (centerpiece)."""
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = ROOT / "workspace/investigations/biogears-sbml-sepsis/studies/03-workflows"


def test_study_yaml_shape():
    d = yaml.safe_load((S / "study.yaml").read_text(encoding="utf-8"))
    assert d["name"] == "03-workflows"
    c = d["claim"].lower()
    assert "workflow" in c and ("antimony" in c or "sbml" in c)


def test_handoff_used_the_real_artifact_and_ran():
    j = json.loads((S / "run/handoff_result.json").read_text(encoding="utf-8"))
    # the hand-off loaded the REAL biogears_sepsis.ant (not a toy model)
    assert j["handoff"]["artifact"].endswith("biogears_sepsis.ant")
    assert "TelluriumUTCStep" in j["handoff"]["loaded_by"]
    # the four verbs were exercised
    assert set(j["verbs"]) >= {"list", "call", "connect", "run"}
    assert j["workflow"]["ran"] is True


def test_exact_engine_drives_the_output_not_just_the_surrogate():
    """Critical review fix: the handed-off exact engine must produce the workflow's
    result, not be a dead-end node while a surrogate-only sum is emitted."""
    j = json.loads((S / "run/handoff_result.json").read_text(encoding="utf-8"))
    # the exact Tellurium run of the .ant produces a REAL, physical MAP
    ex = j["result"]["exact_map_final"]
    assert isinstance(ex, (int, float)) and 30.0 < ex < 160.0, "exact MAP must be physical"
    # a genuine exact-vs-surrogate comparison is reported (not an arbitrary sum)
    assert "exact_minus_surrogate_map" in j["result"]
    # the exact node is wired into the emitted outputs
    assert any(k.startswith("exact.") for k in j["workflow"]["emit"])


def test_handoff_chart_present():
    assert (S / "charts/01_handoff_output.svg").exists()
