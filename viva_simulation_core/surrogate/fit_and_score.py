"""Fit a surrogate with a backend and score it with the validate harness — the
honest "does this surrogate work?" entry point.

CLI:  python -m viva_simulation_core.surrogate.fit_and_score --backend linear|persistence
      (runs a self-contained demo on a toy decay system and prints the scorecard).
"""
from __future__ import annotations

import numpy as np

from viva_simulation_core.surrogate.interface import Scorecard, SurrogateBackend
from viva_simulation_core.surrogate.validate import validate_surrogate


def fit_and_score(backend: SurrogateBackend, trajectories, observable_names, dt,
                  bounds, driver_ics, truth_trajs, *, speedup=None) -> Scorecard:
    fitted = backend.fit(trajectories, observable_names, dt)
    sc = validate_surrogate(fitted, truth_trajs, driver_ics, bounds, dt, speedup=speedup)
    sc.backend = backend.name
    return sc


def _demo(backend_name: str) -> Scorecard:
    from viva_simulation_core.surrogate.backends.baseline import (
        PersistenceBackend, LinearBackend,
    )
    backends = {"persistence": PersistenceBackend(), "linear": LinearBackend()}
    backend = backends[backend_name]
    names = ["a", "b"]
    trajs = []
    for ic in ([9.0, 3.0], [2.0, 8.0], [5.0, 5.0], [7.0, 1.0]):
        xs = [np.array(ic, float)]
        for _ in range(7):
            xs.append(xs[-1] * 0.6)
        trajs.append(np.array(xs))
    ics = [t[0] for t in trajs]
    bounds = {k: (-1.0, 100.0) for k in names}
    return fit_and_score(backend, trajs, names, 1.0, bounds, ics, trajs)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="fit + score a surrogate (demo on a toy system)")
    ap.add_argument("--backend", choices=["persistence", "linear"], default="linear")
    args = ap.parse_args(argv)
    print(_demo(args.backend))


if __name__ == "__main__":
    main()
