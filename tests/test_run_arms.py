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
    path.write_text(json.dumps({'meta': {'horizon_s': 7140.0, 'complete': True, 'severity': 1,
                                           'interval_s': 60.0, 'infection_start_s': 0}, 'samples': s}))


def test_analyze_synthetic(tmp_path):
    for e, off in (('engine', 0), ('sbml', 1)):
        _arm(ra.arm_paths(tmp_path, e, 1)[0], off)
    res = ra.analyze(1, tmp_path)
    assert (tmp_path / 'compare-sev1.json').exists()
    assert len(res['aligned']['per_observable']) == 9
    assert res['milestones']['sbp_below_100_h']['engine'] is not None


# ---- strengthened tests ----
import pytest  # noqa: E402


def test_rows_to_samples_both_shapes():
    obs = list(SEPSIS_OBSERVABLES)

    def mk(shape):
        def extract(rows):
            out = {k: [] for k in obs}
            for r in rows:
                c = r.get(shape)
                if not c:
                    continue
                for i, k in enumerate(obs):
                    out[k].append(float(c['v']) + i)
            return out if out['SBP'] else {}
        return extract

    for shape, tkey in (('parameters', 'time'), ('circuits', 'global_time')):
        rows = [{shape: None, tkey: 0.0}, {shape: {'v': 1}, tkey: 60.0}, {shape: {'v': 2}, tkey: 120.0}]
        s = ra.rows_to_samples(rows, mk(shape), obs)
        assert [x['time'] for x in s] == [60.0, 120.0]
        assert s[1]['SBP'] == 2.0 and s[1][obs[1]] == 3.0


def test_rows_to_samples_missing_time_raises():
    ex = lambda rows: {k: [1.0] for k in SEPSIS_OBSERVABLES}
    with pytest.raises(ValueError):
        ra.rows_to_samples([{'parameters': {'x': 1}}], ex, list(SEPSIS_OBSERVABLES))


def test_analyze_mismatched_horizon_refuses(tmp_path):
    for e, off in (('engine', 0), ('sbml', 1)):
        _arm(ra.arm_paths(tmp_path, e, 1)[0], off)
    p = ra.arm_paths(tmp_path, 'sbml', 1)[0]
    d = json.loads(p.read_text()); d['meta'].update(horizon_s=3600.0); d['samples'] = d['samples'][:60]
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        ra.analyze(1, tmp_path)
    assert not (tmp_path / 'compare-sev1.json').exists()


def test_analyze_incomplete_refuses(tmp_path):
    for e, off in (('engine', 0), ('sbml', 1)):
        _arm(ra.arm_paths(tmp_path, e, 1)[0], off)
    p = ra.arm_paths(tmp_path, 'engine', 1)[0]
    d = json.loads(p.read_text()); d['meta']['complete'] = False; p.write_text(json.dumps(d))
    with pytest.raises(SystemExit):
        ra.analyze(1, tmp_path)


def test_skip_requires_matching_settings_and_version(tmp_path):
    sp, mp = ra.arm_paths(tmp_path, 'sbml', 1)
    meta = dict(engine='sbml', severity=1, interval_s=60.0, horizon_s=120.0,
                infection_start_s=0, version='sbml:abc')
    sp.write_text(json.dumps({'meta': {**meta, 'complete': True},
                              'samples': [{'time': 0.0}, {'time': 120.0}]}))
    assert not ra.arm_is_finished(sp, mp, meta)          # milestones missing
    mp.write_text('{}')
    assert ra.arm_is_finished(sp, mp, meta)              # exact match
    for k, v in (('version', 'sbml:zzz'), ('infection_start_s', 5.0), ('interval_s', 10.0),
                 ('horizon_s', 240.0)):
        assert not ra.arm_is_finished(sp, mp, {**meta, k: v}), k
    sp.write_text(json.dumps({'meta': {**meta, 'complete': True}, 'samples': [{'time': 0.0}, {'time': 60.0}]}))
    assert not ra.arm_is_finished(sp, mp, meta)          # last sample short of horizon
