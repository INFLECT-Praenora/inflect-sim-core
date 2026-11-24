"""The simulation-core contract in one screen: model in, results + verification out.

One interface — ``simulate(model, request)`` — hands you results AND a structured
verification, with the engine swappable behind it. Run:

    python -m examples.contract_demo
"""
from __future__ import annotations

from pathlib import Path

from viva_simulation_core import simulate, SimulationRequest

SBML = (Path(__file__).resolve().parents[1]
        / "workspace/investigations/biogears-sbml-sepsis/sources/biogears_sepsis.xml")


def _show(title: str, out) -> None:
    v = out.verification
    print(f"\n{title}")
    print(f"  engine={out.engine} wall={out.wall_s:.2f}s points={len(out.results.get('time', []))}")
    for k, series in out.results.items():
        if k != "time" and series:
            print(f"  {k}: {series[0]:.4g} -> {series[-1]:.4g}")
    print(f"  VERIFICATION: {'PASSED' if v.passed else 'FAILED'} "
          f"(ran={v.ran}, finite={v.finite}, admissible={v.admissible})")
    if v.cross_engine:
        print(f"  cross-engine {v.cross_engine['engine']}: rel diff "
              f"{v.cross_engine['max_rel_diff']:.1e}, within tol={v.cross_engine['within_tolerance']}")


def main() -> None:
    print("The simulation-core contract: simulate(model, request) -> results + verification")

    # 1. A model file (SBML) in -> results + verification out.
    _show("[1] SBML model file, Tellurium engine",
          simulate(str(SBML), duration=300, interval=60, observables=["MAP", "SVR", "PB", "TI"]))

    # 2. The SAME model, with cross-engine verification (re-run under COPASI, compare).
    _show("[2] same model + cross-engine verification (COPASI)",
          simulate(str(SBML), duration=300, interval=60, observables=["MAP", "SVR", "PB", "TI"],
                   cross_engine="copasi"))

    # 3. A registered composite id in -> same contract out.
    _show("[3] registered composite id, severity override",
          simulate("viva_simulation_core.composites.sepsis_sbml.sepsis_sbml",
                   SimulationRequest(duration=120, interval=20, observables=["MAP", "SVR", "TI"],
                                     parameters={"severity": 1})))

    print("\nSame interface, any model, any engine — every run carries its own verification.")


if __name__ == "__main__":
    main()
