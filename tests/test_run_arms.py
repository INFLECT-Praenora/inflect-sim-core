"""run_arms analyze aggregation on a synthetic fixture (no engines)."""
import importlib.util
import json
from pathlib import Path

from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES

P = Path(__file__).resolve().parents[1] / 'workspace/investigations/biogears-sbml-sepsis/run_arms.py'
spec = importlib.util.spec_from_file_location('run_arms', P)
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)


def _arm(path, off):
    s = [{'time': t * 60.0, **{k: 100.0 + off - t for k in SEPSIS_OBSERVABLES}} for t in range(0, 120)]
    path.write_text(json.dumps({'meta': {'horizon_s': 7140.0, 'complete': True}, 'samples': s}))


def test_analyze_synthetic(tmp_path):
    for e, off in (('engine', 0), ('sbml', 1)):
        _arm(ra.arm_paths(tmp_path, e, 1)[0], off)
    res = ra.analyze(1, tmp_path)
    assert (tmp_path / 'compare-sev1.json').exists()
    assert len(res['aligned']['per_observable']) == 9
    assert res['milestones']['sbp_below_100_h']['engine'] is not None
