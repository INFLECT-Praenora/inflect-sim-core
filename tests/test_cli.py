"""Task 5: the viva-sim CLI verbs (thin over the facade).

list / call / workflow / serve mirror the facade; the existing ``run
<composite_id>`` (catalog-composite run) is kept untouched.
"""
import json

from typer.testing import CliRunner

from viva_simulation_core.cli.__main__ import app

runner = CliRunner()


def test_cli_list_shows_a_known_module():
    r = runner.invoke(app, ["list"])
    assert r.exit_code == 0, r.output
    assert "MathExpressionStep" in r.output


def test_cli_call_runs_a_module():
    r = runner.invoke(app, [
        "call", "MathExpressionStep",
        "--config", json.dumps({"expressions": [{"out": "y", "expr": "2*x + 3"}]}),
        "--input", "x=5.0", "--steps", "1",
    ])
    assert r.exit_code == 0, r.output
    assert "y = 13.0" in r.output


def test_cli_call_bad_module_exits_nonzero_cleanly():
    r = runner.invoke(app, ["call", "NoSuchModule"])
    assert r.exit_code != 0
    assert "unknown module" in r.output.lower()
    # a readable message, not a raw traceback
    assert "Traceback" not in r.output


def test_cli_workflow_runs_a_spec(tmp_path):
    spec = {
        "nodes": [
            {"id": "inc", "module": "MathExpressionStep",
             "config": {"expressions": [{"out": "a", "expr": "x + 1"}]}},
            {"id": "scale", "module": "MathExpressionStep",
             "config": {"expressions": [{"out": "b", "expr": "a * 10"}]}},
        ],
        "wires": [{"from": "inc.a", "to": "scale.a"}],
        "emit": ["scale.b", "inc.a"],
    }
    f = tmp_path / "wf.json"
    f.write_text(json.dumps(spec))
    r = runner.invoke(app, ["workflow", str(f), "--steps", "1", "--override", "inc.x=5.0"])
    assert r.exit_code == 0, r.output
    assert "scale.b = 60.0" in r.output


def test_cli_serve_prints_observables(tmp_path):
    f = tmp_path / "m.ant"
    f.write_text("S1 -> S2; k*S1; k = 0.3; S1 = 10; S2 = 0")
    r = runner.invoke(app, ["serve", str(f), "--format", "antimony"])
    assert r.exit_code == 0, r.output
    assert "S1" in r.output


def _assert_clean_failure(r):
    """Non-zero exit with a readable message, never a leaked (traceback-producing)
    exception — only typer's own SystemExit is acceptable."""
    assert r.exit_code != 0
    assert r.exception is None or isinstance(r.exception, SystemExit), repr(r.exception)


def test_cli_call_bad_config_json_fails_cleanly():
    r = runner.invoke(app, ["call", "MathExpressionStep", "--config", "{not json"])
    _assert_clean_failure(r)


def test_cli_workflow_unknown_module_fails_cleanly(tmp_path):
    f = tmp_path / "wf.json"
    f.write_text(json.dumps({"nodes": [{"id": "n", "module": "NoSuchModule"}], "wires": [], "emit": []}))
    r = runner.invoke(app, ["workflow", str(f)])
    _assert_clean_failure(r)


def test_cli_workflow_malformed_json_fails_cleanly(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not valid json")
    r = runner.invoke(app, ["workflow", str(f)])
    _assert_clean_failure(r)


def test_cli_serve_bad_model_fails_cleanly(tmp_path):
    f = tmp_path / "bad.ant"
    f.write_text("this is not antimony @@@")
    r = runner.invoke(app, ["serve", str(f), "--format", "antimony"])
    _assert_clean_failure(r)
