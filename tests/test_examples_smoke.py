"""Smoke test: the contract demo (the one example) runs end to end.

It drives real engines (Tellurium + COPASI), so it is slow and skips cleanly when
either exact-SBML engine is absent — the fast, engine-free suite stays green.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


@pytest.mark.slow
@pytest.mark.skipif(not (_have("tellurium") and _have("basico")),
                    reason="contract_demo needs the tellurium + copasi engines")
def test_contract_demo_runs():
    p = subprocess.run([sys.executable, "-m", "examples.contract_demo"],
                       cwd=REPO, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-800:]
    assert "PASSED" in p.stdout
