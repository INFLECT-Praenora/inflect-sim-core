"""The simulation-core contract: model in, results + verification out.

Engine-exercising cases run in a subprocess: tellurium (roadrunner) and COPASI are
native extensions that can crash the interpreter when churned in one pytest process,
so — like test_nextflow_underlying — we drive them out-of-process and read JSON back.
The pure verify() logic is tested in-process.
"""
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from viva_simulation_core.contract import verify, SimulationRequest

ROOT = Path(__file__).resolve().parents[1]
SBML = ROOT / "workspace/investigations/biogears-sbml-sepsis/sources/biogears_sepsis.xml"


def _simulate_subprocess(snippet: str) -> dict:
    """Run `snippet` (which must build `out` via simulate) and return out as JSON."""
    code = textwrap.dedent(f"""
        import json
        from viva_simulation_core import simulate, SimulationRequest
        SBML = {str(SBML)!r}
        {snippet}
        v = out.verification
        print("RESULT" + json.dumps({{
            "engine": out.engine, "n": len(out.results.get("time", [])),
            "obs": [k for k in out.results if k != "time"],
            "passed": v.passed, "ran": v.ran, "finite": v.finite,
            "admissible": v.admissible, "cross_engine": v.cross_engine,
        }}))
    """)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, f"subprocess failed: {p.stderr[-800:]}"
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def test_contract_sbml_model_in_results_and_verification_out():
    out = _simulate_subprocess(
        'out = simulate(SBML, duration=120, interval=60, observables=["MAP","SVR","TI"])')
    assert out["engine"] == "tellurium"
    assert set(["MAP", "SVR", "TI"]).issubset(out["obs"]) and out["n"] >= 2
    assert out["ran"] and out["finite"] and out["passed"]
    assert out["admissible"] == {"MAP": True, "SVR": True, "TI": True}


def test_contract_cross_engine_verification():
    out = _simulate_subprocess(
        'out = simulate(SBML, duration=120, interval=60, observables=["MAP","SVR"], cross_engine="copasi")')
    ce = out["cross_engine"]
    assert ce is not None and ce["engine"] == "copasi"
    assert ce["within_tolerance"] is True and ce["max_rel_diff"] < 1e-2


def test_contract_composite_id_path():
    out = _simulate_subprocess(
        'out = simulate("viva_simulation_core.composites.sepsis_sbml.sepsis_sbml", '
        'SimulationRequest(duration=100, interval=20, observables=["MAP","TI"], parameters={"severity":1}))')
    assert out["n"] >= 2 and out["passed"]


def test_verify_flags_inadmissible_values():
    bad = {"time": [0, 1, 2], "MAP": [95.0, 10.0, -5.0]}
    rep = verify(bad, SimulationRequest(observables=["MAP"], verify=("ran", "finite", "admissible")))
    assert rep.admissible == {"MAP": False}
    assert rep.passed is False


def test_verify_flags_nonfinite():
    bad = {"time": [0, 1], "MAP": [95.0, math.nan]}
    rep = verify(bad, SimulationRequest(observables=["MAP"], verify=("ran", "finite")))
    assert rep.finite is False
    assert rep.passed is False
