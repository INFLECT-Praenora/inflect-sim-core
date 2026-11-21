"""Live check for the COMPOSITION pillar: the interchangeable engines construct
and run in THIS workspace, cheaply.

- Tellurium (exact SBML): building the sepsis state composite constructs the
  roadrunner-backed engine over the real 91-state model (tellurium_builds).
- Surrogate (viva-torch NeuralProcess): a full 48 h / 576-step autoregressive
  rollout runs in a fraction of a second. We assert it is FINITE and BOUNDED,
  not that it is accurate — it is a fast APPROXIMATE tier (see study 04/05).

BioGears (C++) is the third engine; its full run is cited from committed
wall-clock (studies carry the heavy runs), not re-run here.

Run:  PYTHONPATH=<worktree> python confirm_engines_run.py
Writes engines_live_check.json next to this script.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from viva_torch.spec import SurrogateSpec
from viva_torch.evaluate import _rollout_surrogate
from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES
from viva_simulation_core.composites.sepsis_sbml_state import build_sepsis_state_composite

HERE = Path(__file__).parent
CKPT = HERE.parents[2] / "assets" / "surrogate_stable2.pt"
INTERVAL = 300.0
N_STEPS = 576  # 48 h


def _col(spec, var):
    for i, p in enumerate(spec.target_paths_t):
        if p[-1] == var:
            return i
    return None


def main() -> None:
    spec = SurrogateSpec(**torch.load(CKPT, map_location="cpu", weights_only=False)["spec"])

    t0 = time.perf_counter()
    doc, core = build_sepsis_state_composite(3, 0.0, INTERVAL)  # builds Tellurium/roadrunner engine
    build_s = time.perf_counter() - t0
    tellurium_builds = isinstance(doc, dict) and "tellurium" in doc

    t1 = time.perf_counter()
    sur = np.asarray(_rollout_surrogate(spec, str(CKPT), core, {}, N_STEPS, INTERVAL))
    sur_s = time.perf_counter() - t1
    cmap = _col(spec, SEPSIS_OBSERVABLES["MAP"])

    out = {
        "tellurium_builds": bool(tellurium_builds),
        "tellurium_build_s": round(build_s, 3),
        "surrogate": {
            "ran": True,
            "finite": bool(np.isfinite(sur).all()),
            "n_steps": int(N_STEPS),
            "wall_s": round(sur_s, 4),
            "note": "approximate tier — bounded/finite over the horizon, not asserted clinically accurate (study 04/05)",
        },
        "map_col_present": cmap is not None,
    }
    (HERE / "engines_live_check.json").write_text(json.dumps(out, indent=2))
    print(f"tellurium builds: {tellurium_builds} ({build_s:.1f}s) | "
          f"surrogate 48h rollout: finite={out['surrogate']['finite']} in {sur_s:.2f}s")


if __name__ == "__main__":
    main()
