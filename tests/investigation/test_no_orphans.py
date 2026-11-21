"""Task 7 — the on-disk study set matches investigation.yaml; no orphans, no stragglers."""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
INV_DIR = ROOT / "workspace/investigations/biogears-sbml-sepsis"
INV = INV_DIR / "investigation.yaml"


def _d():
    return yaml.safe_load(INV.read_text())


def test_disk_matches_declared_studies():
    declared = set(_d()["studies"])
    on_disk = {p.parent.name for p in (INV_DIR / "studies").glob("*/study.yaml")}
    assert on_disk == declared, f"on_disk={sorted(on_disk)} declared={sorted(declared)}"


def test_no_old_validation_slugs_remain():
    old = {"01-baseline-parity", "02-mild-infection", "03-severe-infection",
           "04-engine-runtime", "05-divergence-mechanism", "06-ode-integrator-runtime"}
    on_disk = {p.name for p in (INV_DIR / "studies").iterdir() if p.is_dir()}
    assert not (on_disk & old), f"old study dirs still present: {sorted(on_disk & old)}"


def test_cross_refs_resolve():
    d = _d()
    dirs = {p.parent.name for p in (INV_DIR / "studies").glob("*/study.yaml")}
    for a in d["at_a_glance"]:
        assert a["study"] in dirs, f"at_a_glance references missing study {a['study']}"
    for ac in d["acceptance_criteria"]:
        assert ac["study"] in dirs, f"acceptance_criteria references missing study {ac['study']}"
