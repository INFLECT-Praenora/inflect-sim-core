"""Task 8 — every declared study is render-ready: valid v3 shape + its primary viz on disk."""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
INV_DIR = ROOT / "workspace/investigations/biogears-sbml-sepsis"
STUDIES = yaml.safe_load((INV_DIR / "investigation.yaml").read_text())["studies"]

REQUIRED = {"schema_version", "name", "investigation", "status", "title", "claim"}


def test_every_study_is_valid_and_renderable():
    for slug in STUDIES:
        sy = INV_DIR / "studies" / slug / "study.yaml"
        assert sy.exists(), f"missing {slug}/study.yaml"
        d = yaml.safe_load(sy.read_text())
        assert REQUIRED <= set(d), f"{slug} missing fields {REQUIRED - set(d)}"
        assert d["name"] == slug
        # every study that declares a primary visualization has that chart on disk
        for v in d.get("visualizations", []):
            if v.get("display") == "primary":
                addr = v["address"].split(":", 1)[1]  # image:charts/x.svg -> charts/x.svg
                assert (INV_DIR / "studies" / slug / addr).exists(), f"{slug}: missing {addr}"


def test_no_framework_token_in_study_titles():
    banned = ("vivarium", "process-bigraph")
    for slug in STUDIES:
        d = yaml.safe_load((INV_DIR / "studies" / slug / "study.yaml").read_text())
        assert not any(b in d["title"].lower() for b in banned), f"{slug} title leaks framework"
