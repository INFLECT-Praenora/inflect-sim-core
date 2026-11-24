"""The inflect-models-db bridge: discover the curated database and run any model
by slug through the simulation-core contract.

Most of the bridge is engine-free and tested in-process against a tiny temporary
database (a `card.yaml` + `.ant`) pointed at via INFLECT_MODELS_DB_ROOT, so these
run without the curated `models/` dir being present (e.g. a pure wheel install).
The one end-to-end `simulate("<slug>")` run goes through a subprocess like the rest
of test_contract (roadrunner is a native extension).
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("inflect_models_db")

ROOT = Path(__file__).resolve().parents[1]

# A minimal exponential-decay model: X' = -k X, X(0) set per scenario. Analytic,
# fast, engine-light — enough to exercise discovery + resolution + a real run.
_ANT = textwrap.dedent("""
    model toy
      X' = -k * X
      X = 1.0
      k = 1.0
    end
""").strip()

_CARD = textwrap.dedent("""
    slug: toy-decay
    title: Toy exponential decay
    source: {authors: "Test", year: 2026}
    model:
      file: model.ant
      format: antimony
      species: [X]
    validation:
      time: {start: 0, end: 5, n_points: 6, units: s}
      scenarios:
        - name: baseline
          initials: {X: 1.0}
          expect: {X: {target: 0.0067, tol: 0.01}}
        - name: hot-start
          initials: {X: 3.0}
          expect: {X: {target: 0.0202, tol: 0.03}}
""").strip()


@pytest.fixture()
def toy_db(tmp_path, monkeypatch):
    """A one-model temporary curated database, selected via INFLECT_MODELS_DB_ROOT."""
    d = tmp_path / "models" / "toy-decay"
    d.mkdir(parents=True)
    (d / "model.ant").write_text(_ANT)
    (d / "card.yaml").write_text(_CARD)
    monkeypatch.setenv("INFLECT_MODELS_DB_ROOT", str(tmp_path / "models"))
    return tmp_path / "models"


def test_list_and_is_slug(toy_db):
    from viva_simulation_core import models_db
    assert models_db.available()
    assert "toy-decay" in models_db.list_models()
    assert models_db.is_slug("toy-decay")
    assert not models_db.is_slug("does-not-exist")
    assert not models_db.is_slug(12345)


def test_resolve_carries_card_runspec(toy_db):
    from viva_simulation_core import models_db
    spec = models_db.resolve("toy-decay", "hot-start")
    assert spec["scenario"] == "hot-start"
    assert spec["observables"] == ["X"]
    assert spec["initials"] == {"X": 3.0}
    assert spec["duration"] == 5.0 and spec["interval"] == 1.0
    assert "X' = -k * X" in spec["antimony"]


def test_resolve_defaults_to_first_scenario(toy_db):
    from viva_simulation_core import models_db
    assert models_db.resolve("toy-decay")["scenario"] == "baseline"


def test_apply_to_request_kwargs_fills_without_overriding(toy_db):
    from viva_simulation_core import models_db
    ant, kw = models_db.apply_to_request_kwargs(
        "toy-decay", "hot-start", {"interval": 0.5, "engine": "tellurium"})
    assert "X' = -k * X" in ant
    assert kw["model_format"] == "antimony"
    assert kw["observables"] == ["X"]
    assert kw["interval"] == 0.5          # caller's value wins
    assert kw["duration"] == 5.0          # from the card
    assert kw["parameters"]["X"] == 3.0   # scenario initial merged in
    assert kw["engine"] == "tellurium"


def test_resolve_unknown_slug_raises(toy_db):
    from viva_simulation_core import models_db
    with pytest.raises(LookupError):
        models_db.resolve("nope")


def test_register_all_registers_toy_model(toy_db):
    from process_bigraph.composite_spec import all_specs
    from viva_simulation_core import models_db
    names = models_db.register_all()
    assert "modelsdb_toy_decay" in names
    assert any(sid.endswith(".modelsdb_toy_decay") for sid in all_specs())


def test_simulate_by_slug_runs_through_contract(tmp_path):
    """End-to-end: simulate("<slug>", scenario=...) runs the card's model via the
    tellurium engine and returns the card's observable. Subprocess (roadrunner)."""
    pytest.importorskip("viva_tellurium")
    d = tmp_path / "models" / "toy-decay"
    d.mkdir(parents=True)
    (d / "model.ant").write_text(_ANT)
    (d / "card.yaml").write_text(_CARD)

    code = textwrap.dedent("""
        import json
        from viva_simulation_core import simulate
        out = simulate("toy-decay", scenario="hot-start", engine="tellurium")
        r = out.results
        print("RESULT" + json.dumps({
            "obs": [k for k in r if k != "time"],
            "n": len(r.get("time", [])),
            "x0": r["X"][0], "xend": r["X"][-1],
            "passed": out.verification.passed,
        }))
    """)
    pp = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env = {**os.environ, "PYTHONPATH": pp,
           "INFLECT_MODELS_DB_ROOT": str(tmp_path / "models")}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=600)
    assert p.returncode == 0, f"subprocess failed: {p.stderr[-800:]}"
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("RESULT"))
    out = json.loads(line[len("RESULT"):])
    assert out["obs"] == ["X"] and out["n"] >= 2
    assert abs(out["x0"] - 3.0) < 1e-6          # scenario initial applied
    assert out["xend"] < 0.1                     # decayed by t=5 (3*e^-5 ≈ 0.02)
    assert out["passed"]
