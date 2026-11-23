#!/usr/bin/env python
"""Study 07: evaluate pre-declared acceptance criteria against committed evidence.

Turns the study-05 engine-vs-SBML comparison into traceable pass / fail / pending
verdicts. Each criterion declares a tolerance up front; the evaluator reads the
committed milestone artifacts (the exact evidence source), computes the quantity,
and records the verdict plus the artifact it came from. A criterion whose required
test has not been run is reported as PENDING rather than silently passed.

Reads:  studies/02-ensembles/data/{engine,sbml}-sev3.milestones.json
Writes: studies/07-evidence/out/acceptance_eval.json

Run from the investigation dir:
    python studies/07-evidence/evaluate_acceptance.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INV = HERE.parents[1]
DATA = INV / "studies" / "02-ensembles" / "data"
ENGINE = DATA / "engine-sev3.milestones.json"
SBML = DATA / "sbml-sev3.milestones.json"


def rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main() -> None:
    eng = json.loads(ENGINE.read_text())
    sb = json.loads(SBML.read_text())
    src = {"engine": str(ENGINE.relative_to(INV)), "sbml": str(SBML.relative_to(INV))}

    criteria = []

    # PASS — inflammation core agreement (tissue integrity minimum).
    ti_e, ti_s = eng["tissue_integrity_min"], sb["tissue_integrity_min"]
    e = rel_err(ti_e, ti_s)
    criteria.append({
        "id": "INFLAM-01", "observable": "tissue_integrity_min", "tolerance_rel": 0.01,
        "engine": ti_e, "sbml": ti_s, "rel_error": e,
        "verdict": "pass" if e <= 0.01 else "fail", "evidence": src,
        "statement": "The two full-physics engines agree on minimum tissue integrity to within 1%."})

    # PASS — inflammation core agreement (peak blood pathogen).
    pb_e, pb_s = eng["peak_blood_pathogen"], sb["peak_blood_pathogen"]
    e = rel_err(pb_e, pb_s)
    criteria.append({
        "id": "INFLAM-02", "observable": "peak_blood_pathogen", "tolerance_rel": 0.01,
        "engine": pb_e, "sbml": pb_s, "rel_error": e,
        "verdict": "pass" if e <= 0.01 else "fail", "evidence": src,
        "statement": "The engines agree on peak blood pathogen to within 1%."})

    # FAIL — hemodynamic agreement (time to MAP < 65 mmHg, septic shock onset).
    map_e, map_s = eng["map_below_65_h"], sb["map_below_65_h"]
    dh = abs(map_e - map_s)
    criteria.append({
        "id": "HEMO-01", "observable": "map_below_65_h", "tolerance_hours": 6.0,
        "engine_h": map_e, "sbml_h": map_s, "abs_diff_h": dh,
        "verdict": "pass" if dh <= 6.0 else "fail", "evidence": src,
        "statement": "Time to septic shock (MAP < 65 mmHg) agrees within 6 h."})

    # PENDING — surrogate clinical-accuracy criterion: its validator has not been run.
    criteria.append({
        "id": "SURR-01", "observable": "surrogate_hemodynamic_trajectory_error",
        "tolerance_rel": 0.10, "verdict": "pending",
        "evidence": None, "required_test": "held-out multi-seed surrogate-vs-exact trajectory comparison",
        "statement": "Surrogate hemodynamic error within 10% of the exact engine (validator not yet run)."})

    counts = {k: sum(1 for c in criteria if c["verdict"] == k) for k in ("pass", "fail", "pending")}
    out = {"meta": {"source_study": "05-fidelity", "severity": 3},
           "criteria": criteria, "counts": counts}
    (HERE / "out" / "acceptance_eval.json").write_text(json.dumps(out, indent=1))
    print("counts:", counts)
    for c in criteria:
        print(f"  {c['id']:9s} {c['verdict']:7s} {c['observable']}")


if __name__ == "__main__":
    main()
