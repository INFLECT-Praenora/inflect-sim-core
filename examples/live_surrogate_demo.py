"""Neural-surrogate tier of the digital-twin API: train a surrogate, serve it, and
compare it to the exact engine on the same twin.

Runs in-process (no server). Needs the ``[api]`` extra for FastAPI; viva-torch is
a base dependency, so the surrogate works out of the box.

    uv run --extra api python examples/live_surrogate_demo.py

Reference: docs/api.md (the tier rule)
"""
from __future__ import annotations

import tempfile
import time

import numpy as np
from fastapi.testclient import TestClient

from viva_simulation_core.api.app import create_app
from viva_simulation_core.api.config import ApiConfig
from viva_simulation_core.api.surrogate_curate import train_and_curate
from viva_simulation_core.api.surrogate_torch import TorchCheckpointBackend

# Observed core S1 -> S2 -> S3 (closed dynamics, what the surrogate learns) plus a
# large, stiff, UNOBSERVED downstream cascade driven by S2 that the exact engine must
# still integrate. This stands in for a big mechanistic model with a few outputs of
# interest; the surrogate skips the unobserved stiff part entirely.
_N = 150
_cascade = "\n".join(f"  X{i} -> X{i+1}; {10.0 ** (1 + (i % 4))}*X{i};" for i in range(1, _N))
MODEL = f"""
model toy
  S1 -> S2; k1*S1;
  S2 -> S3; k2*S2;
  S2 -> X1; 0.5*S2;
{_cascade}
  k1 = 0.30; k2 = 0.12;
  S1 = 10; S2 = 0; S3 = 0;
end
"""
DT, N_STEPS, SPECIES = 1.0, 30, ["S1", "S2", "S3"]
H = {"Authorization": "Bearer demo-key"}


def timed(c, body, reps=1):
    best, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = c.post(f"/twins/{TID}/predict", json=body, headers=H)
        best = min(best, time.perf_counter() - t0)
    assert r.status_code == 200, r.text
    return r.json(), best


def main():
    global TID
    ckdir = tempfile.mkdtemp(prefix="inflect_surrogate_")
    print("== 1. train + curate a surrogate (operator workflow)")
    res = train_and_curate(source=MODEL, model_format="antimony", observables=SPECIES,
                           directory=ckdir, timestep_s=DT, n_steps=N_STEPS)
    nrmse = res.entry.val_error["overall_nrmse"]
    print(f"sampling {res.sample_s:.1f}s, training {res.train_s:.1f}s, held-out nrmse={nrmse:.4f}")
    print("envelope: horizon_s =", res.entry.horizon_s, " state_ranges =",
          {k: tuple(round(x, 2) for x in v) for k, v in res.entry.state_ranges.items()})

    cfg = ApiConfig(checkpoint_dir=ckdir, api_keys="demo-key", rate_capacity=1e6, rate_refill_per_s=1e6)
    with TestClient(create_app(cfg, backend=TorchCheckpointBackend(ckdir))) as c:
        print("\n== 2. register the same model over HTTP")
        reg = c.post("/twins", json={"model": MODEL, "model_format": "antimony",
                                     "observables": SPECIES}, headers=H)
        assert reg.status_code == 200, reg.text
        TID = reg.json()["twin_id"]
        info = c.get(f"/twins/{TID}", headers=H).json()
        print("engines_available:", info["engines_available"])
        assert "surrogate" in info["engines_available"]

        body = {"observables": SPECIES, "horizon_s": 25.0, "interval_s": DT}
        print("\n== 3. exact vs auto (surrogate) over 25 s")
        ex, t_ex = timed(c, {**body, "engine": "exact"}, reps=3)
        timed(c, {**body, "engine": "auto"})  # warm the net cache
        au, t_au = timed(c, {**body, "engine": "auto"}, reps=3)
        print(f"engine_used: exact -> {ex['engine_used']}, auto -> {au['engine_used']} "
              f"(in_distribution={au['in_distribution']})")
        print(f"{'t':>4} | " + " | ".join(f"{s}: exact / surrogate" for s in SPECIES))
        for i in range(0, len(ex["times"]), 5):
            print(f"{ex['times'][i]:4.0f} | " + " | ".join(
                f"{ex['observables'][s][i]:7.3f} / {au['observables'][s][i]:7.3f}   " for s in SPECIES))
        diffs = {s: np.abs(np.array(ex["observables"][s]) - np.array(au["observables"][s])) for s in SPECIES}
        max_abs = max(float(d.max()) for d in diffs.values())
        allx = np.concatenate([ex["observables"][s] for s in SPECIES])
        allq = np.concatenate([au["observables"][s] for s in SPECIES])
        rng_ = float(allx.max() - allx.min())
        nrmse_rt = float(np.sqrt(np.mean((allx - allq) ** 2)) / rng_)
        print(f"max abs diff = {max_abs:.3f}, nrmse (range-normalized) = {nrmse_rt:.4f}")
        print(f"wall time (HTTP request, best of 3): exact {t_ex*1000:.1f} ms | surrogate {t_au*1000:.1f} ms "
              f"| speedup x{t_ex/t_au:.1f}")
        print(f"engine-side wall_ms: exact {ex['provenance']['wall_ms']:.1f} | "
              f"surrogate {au['provenance']['wall_ms']:.1f}")

        print("\n== 4. out-of-envelope request (horizon 200 s > trained 30 s) -> exact fallback")
        far, _ = timed(c, {**body, "horizon_s": 200.0, "engine": "auto"})
        print("engine_used:", far["engine_used"])

        assert au["engine_used"] == "surrogate" and au["in_distribution"] is True
        assert ex["engine_used"] == "exact"
        assert nrmse_rt < 0.15, nrmse_rt
        speed = ex["provenance"]["wall_ms"] / au["provenance"]["wall_ms"]
        assert speed > 1, f"surrogate not faster engine-side (x{speed:.2f})"
        assert far["engine_used"] == "exact"
    print("\nLIVE SURROGATE DEMO OK")


if __name__ == "__main__":
    main()
