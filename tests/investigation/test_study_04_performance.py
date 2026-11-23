"""Task 5 — Study 04 Performance (compute profile + surrogate speedup)."""
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
S = ROOT / "workspace/investigations/biogears-sbml-sepsis/studies/04-performance"


def test_study_yaml_shape():
    d = yaml.safe_load((S / "study.yaml").read_text(encoding="utf-8"))
    assert d["name"] == "04-performance"
    c = d["claim"].lower()
    assert "surrogate" in c and ("stiff" in c or "ode" in c)


def test_bench_live_finite_and_exact_dominates():
    b = json.loads((S / "charts/bench_live.json").read_text(encoding="utf-8"))
    ex = b["exact"]["per_step_ms"]
    su = b["surrogate"]["per_step_ms"]
    assert ex == ex and su == su and ex > 0 and su > 0  # finite, positive
    assert ex / su > 100  # the stiff ODE solve dwarfs the surrogate forward pass


def test_runtime_figure_provenance():
    assert (S / "charts/01_runtime_compare.svg").exists()
    meta = json.loads((S / "charts/01_runtime_compare.meta.json").read_text(encoding="utf-8"))
    blob = json.dumps(meta).lower()
    # provenance distinguishes committed full-run wall-clock from the live micro-benchmark
    assert "committed" in blob and ("live" in blob or "per-step" in blob or "per step" in blob)
