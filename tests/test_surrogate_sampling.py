"""Sampling the sepsis transition dataset (real roadrunner; slow)."""
import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.slow

SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
          / "workspace/investigations/neural-surrogate-sepsis/sample_dataset.py")


def _load():
    spec = importlib.util.spec_from_file_location("sample_dataset", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dataset_has_expected_shape_and_split(tmp_path):
    sd = _load()
    ds, meta = sd.sample(severities=(0, 2), jitters=(0.0, 30.0), eval_jitters=(30.0,),
                         n_steps=4, out_dir=tmp_path)
    assert ds.X.shape[1] == 100 and ds.Y.shape[1] == 100
    assert ds.spec.n_features == 100 and ds.spec.n_targets == 100
    train = {tuple(i) for i in meta["train_ics"]}
    ev = {tuple(i) for i in meta["eval_ics"]}
    assert len(train) + len(ev) == meta["n_planned_ics"] == 3
    assert train.isdisjoint(ev) and ev
    assert (tmp_path / "dataset_meta.json").exists()
