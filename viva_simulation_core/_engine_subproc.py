"""Run one engine on one model in an isolated subprocess.

Tellurium (roadrunner) and COPASI (basico) each statically link their own libSBML,
so loading both in one process can segfault. The contract's cross-engine verification
therefore runs the second engine here, in its own process, and reads the results back
as JSON. Invoked as ``python -m viva_simulation_core._engine_subproc`` with a JSON
spec {kind, payload, request} on stdin; writes {results, wall} to stdout.
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    from .contract import SimulationRequest, _run

    spec = json.load(sys.stdin)
    req = SimulationRequest(**spec["request"])
    results, wall = _run(spec["kind"], spec["payload"], req.engine, req)
    json.dump({"results": results, "wall": wall}, sys.stdout)


if __name__ == "__main__":
    main()
