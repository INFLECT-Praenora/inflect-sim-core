"""Task 6 — Study 05 Fidelity & divergence (the honest backbone)."""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = ROOT / "workspace/investigations/biogears-sbml-sepsis/studies/05-fidelity"


def test_study_yaml_shape():
    d = yaml.safe_load((S / "study.yaml").read_text(encoding="utf-8"))
    assert d["name"] == "05-fidelity"
    c = d["claim"].lower()
    # honest: both agreement AND divergence
    assert ("agree" in c or "bit-for-bit" in c or "match" in c)
    assert ("diverge" in c or "differ" in c)


def test_fidelity_charts_present():
    for name in ("00_baseline_parity", "01_divergence_heatmap", "02_inflammation_match"):
        assert (S / f"charts/{name}.svg").exists(), name
