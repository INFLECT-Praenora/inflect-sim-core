"""Task 7: the vivarium command the Nextflow pipeline runs.

Exercises the exact command examples/nextflow/main.nf's VIVA_RUN process invokes
(viva-sim == python -m viva_simulation_core.cli) — not the nextflow binary, so
the suite needs no nextflow install.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NF = REPO / "examples" / "nextflow"


def test_viva_run_command_produces_scale_b():
    cmd = [sys.executable, "-m", "viva_simulation_core.cli", "workflow",
           str(NF / "chain.json"), "--steps", "1", "--override", "inc.x=5.0"]
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "scale.b = 60.0" in out.stdout
