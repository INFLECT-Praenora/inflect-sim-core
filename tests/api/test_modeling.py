import pytest

from viva_simulation_core.api.modeling import (
    UnknownObservableError,
    UnsafeModelError,
    infer_observables,
    load_engine,
    model_hash,
    reject_unsafe,
    tunable_parameters,
    validate_observables,
)

ANTIMONY = "S1 -> S2; k*S1; k = 0.1; S1 = 10; S2 = 0;"
ANTIMONY_RULE = "S1 -> S2; k*S1; var k = 0.1; S1 = 10; S2 = 0; total := S1 + S2; c = 3; const c;"


def test_load_and_infer():
    rr = load_engine(ANTIMONY, "antimony")
    obs = infer_observables(rr)
    assert {"S1", "S2"} <= set(obs)
    assert obs["S1"] == "S1"


def test_rules_and_tunables():
    rr = load_engine(ANTIMONY_RULE, "antimony")
    assert "total" in infer_observables(rr)
    assert "k" in tunable_parameters(rr)
    assert "c" not in tunable_parameters(rr)
    assert "c" not in infer_observables(rr)


def test_validate_unknown_raises():
    rr = load_engine(ANTIMONY, "antimony")
    with pytest.raises(UnknownObservableError):
        validate_observables(rr, ["nope"])
    assert validate_observables(rr, ["S1"]) == {"S1": "S1"}
    assert validate_observables(rr, {"a": "S2"}) == {"a": "S2"}
    with pytest.raises(UnknownObservableError):
        validate_observables(rr, {"a": "nope"})


def test_roundtrip_sbml():
    import tellurium as te

    sbml = te.loada(ANTIMONY).getCurrentSBML()
    rr = load_engine(sbml, "sbml")
    assert "S1" in infer_observables(rr)


def test_reject_import():
    with pytest.raises(UnsafeModelError):
        reject_unsafe("import 'x.txt'\n" + ANTIMONY, "antimony")
    with pytest.raises(UnsafeModelError):
        reject_unsafe("  IMPORT \"http://x/y.sbml\"\n" + ANTIMONY, "antimony")
    reject_unsafe(ANTIMONY, "antimony")


def test_reject_sbml_external():
    bad = (
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1" '
        'xmlns:comp="http://www.sbml.org/sbml/level3/version1/comp/version1">'
        '<model><comp:listOfExternalModelDefinitions>'
        '<comp:externalModelDefinition comp:id="e" comp:source="file:///etc/passwd"/>'
        "</comp:listOfExternalModelDefinitions></model></sbml>"
    )
    with pytest.raises(UnsafeModelError):
        reject_unsafe(bad, "sbml")
    with pytest.raises(UnsafeModelError):
        reject_unsafe('<sbml><model><x href="http://evil/x"/></model></sbml>', "sbml")
    with pytest.raises(UnsafeModelError):
        reject_unsafe("<sbml><model><x src='file:///x'/></model></sbml>", "sbml")


def test_unknown_format():
    with pytest.raises(ValueError):
        load_engine(ANTIMONY, "cellml")


def test_hash_stable_and_distinct():
    a = model_hash(ANTIMONY, "antimony")
    assert a == model_hash(ANTIMONY, "antimony")
    assert a != model_hash(ANTIMONY + " ", "antimony")
    assert a != model_hash(ANTIMONY, "sbml")
