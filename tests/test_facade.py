"""Task 4: the top-level four-verb + serve facade.

``from viva_simulation_core import catalog, call, connect, run, serve`` — one
import surface, thin over api/workflow.py + serve.py. Importing the package must
stay light (no torch/tellurium pulled eagerly).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SMALL = "S1 -> S2; k*S1; k = 0.3; S1 = 10; S2 = 0"

CHAIN = {
    "nodes": [
        {"id": "inc", "module": "MathExpressionStep",
         "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
        {"id": "scale", "module": "MathExpressionStep",
         "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}},
    ],
    "wires": [{"from": "inc.a", "to": "scale.a"}],
    "emit": ["scale.b", "inc.a"],
}


def test_facade_exports():
    import viva_simulation_core as v
    for name in ("catalog", "call", "connect", "run", "serve"):
        assert callable(getattr(v, name)), name


def test_catalog_lists_modules():
    import viva_simulation_core as v
    assert "MathExpressionStep" in {c.name for c in v.catalog()}


def test_call_runs_a_module():
    import viva_simulation_core as v
    out = v.call("MathExpressionStep",
                 config={"expressions": [{"out": "y", "expr": "2*x + 3"}]},
                 inputs={"x": 5.0}, steps=1)
    assert out["y"] == 13.0


def test_connect_and_run_chain():
    import viva_simulation_core as v
    obs = v.run(CHAIN, steps=1, overrides={"inc.x": 5.0})
    assert obs["inc.a"] == 6.0
    assert obs["scale.b"] == 60.0


def test_serve_returns_a_handle():
    import viva_simulation_core as v
    m = v.serve(SMALL, "antimony")
    assert m.observables()


def test_import_is_light():
    # fresh interpreter: this test session already has torch loaded by other tests.
    code = ("import sys, viva_simulation_core; "
            "print('torch' in sys.modules, 'tellurium' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout
