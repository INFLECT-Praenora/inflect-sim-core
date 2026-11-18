import pytest
from pydantic import ValidationError

from viva_simulation_core.api.models import (
    EnsembleSpec,
    PredictRequest,
    RegisterRequest,
)


def _pr(**kw):
    return dict(observables=["x"], horizon_s=1, interval_s=1, **kw)


def test_predict_defaults():
    r = PredictRequest(**_pr())
    assert r.engine == "auto"
    assert r.return_ == "trajectory"
    assert r.ensemble is None


def test_enum_rejection():
    with pytest.raises(ValidationError):
        PredictRequest(**_pr(return_="bogus"))
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**_pr(), "return": "bogus"})
    with pytest.raises(ValidationError):
        PredictRequest(**_pr(engine="nope"))
    with pytest.raises(ValidationError):
        RegisterRequest(model="m", model_format="xml")


def test_alias_round_trip():
    r = PredictRequest.model_validate({**_pr(), "return": "stats"})
    assert r.return_ == "stats"
    assert r.model_dump(by_alias=True)["return"] == "stats"
    assert PredictRequest(**_pr(return_="endpoint")).return_ == "endpoint"


def test_ensemble_spec():
    e = EnsembleSpec(members=[{"k": 1}], draws={"k": {"dist": "uniform"}})
    assert e.stochastic is False
    assert e.members and e.draws
