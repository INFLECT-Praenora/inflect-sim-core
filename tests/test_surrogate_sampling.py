"""Sampling the sepsis transition dataset (real roadrunner; slow)."""
import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.slow

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / "workspace/investigations/neural-surrogate-sepsis/sample_dataset.py")


def _load():
    # must be importable by name: spawned workers re-import the module to unpickle _run_combo
    import sys
    sys.path.insert(0, str(SCRIPT.parent))
    return importlib.import_module("sample_dataset")


def test_dataset_has_expected_shape_and_split(tmp_path):
    sd = _load()
    ds, meta = sd.sample(severities=(0, 2), jitters=(0.0, 30.0), eval_jitters=(30.0,),
                         n_steps=4, interval_s=30.0, workers=2, out_dir=tmp_path)
    assert ds.X.shape[1] == 98 and ds.Y.shape[1] == 98
    assert ds.spec.n_features == 98 and ds.spec.n_targets == 98
    train = {tuple(i) for i in meta["train_ics"]}
    ev = {tuple(i) for i in meta["eval_ics"]}
    assert len(train) + len(ev) == meta["n_planned_ics"] == 3
    assert train.isdisjoint(ev) and ev
    assert (tmp_path / "dataset_meta.json").exists()


def test_empty_eval_split_raises(tmp_path):
    sd = _load()
    with pytest.raises(ValueError, match="empty split"):
        sd.sample(severities=(0, 2), jitters=(0.0, 30.0), eval_jitters=(999.0,),
                  n_steps=1, interval_s=30.0, workers=1, out_dir=tmp_path)
