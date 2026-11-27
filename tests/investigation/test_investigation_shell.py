"""Task 1 — the reframed investigation shell + framing (00) + surrogate asset."""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
INV = ROOT / "workspace/investigations/biogears-sbml-sepsis/investigation.yaml"
ASSET = ROOT / "workspace/investigations/biogears-sbml-sepsis/assets/surrogate_stable2.pt"

TITLE = (
    "Simulation core — one sepsis model, every engine: "
    "composed, ensembled, orchestrated, benchmarked"
)
STUDIES = [
    "00-simulation-core",
    "01-composition",
    "02-ensembles",
    "03-workflows",
    "04-performance",
    "05-fidelity",
    "06-cosimulation",
    "07-evidence",
    "08-reduced-model",
    "09-event-architecture",
    "10-methods-comparison",
]
BANNED = ("vivarium", "process-bigraph", "process bigraph")


def _inv():
    return yaml.safe_load(INV.read_text(encoding="utf-8"))


def test_title_and_members():
    d = _inv()
    assert d["title"] == TITLE
    assert d["studies"] == STUDIES
    assert "how_to_read" in d and d["how_to_read"].strip()
    assert isinstance(d.get("glossary"), list) and len(d["glossary"]) >= 7


def test_no_framework_in_reader_facing_text():
    d = _inv()
    blob = (d["title"] + " " + " ".join(a["role"] for a in d["at_a_glance"])).lower()
    for b in BANNED:
        assert b not in blob, f"reader-facing text leaks framework token {b!r}"


def test_at_a_glance_matches_members():
    d = _inv()
    assert [a["study"] for a in d["at_a_glance"]] == STUDIES


def test_surrogate_asset_present():
    assert ASSET.exists() and ASSET.stat().st_size > 0


def test_framing_study_00_present():
    p = ROOT / "workspace/investigations/biogears-sbml-sepsis/studies/00-simulation-core/study.yaml"
    assert p.exists()
