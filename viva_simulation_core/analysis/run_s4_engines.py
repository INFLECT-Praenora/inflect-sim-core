"""S4: run the SBML sepsis composite under process-bigraph and viva-core.

Same Tellurium/roadrunner computation under both engines; compares final
observables (determinism) and wall-clock. For a single heavy process the
engine swap measures framework overhead, not viva-core's fine-grained speedup.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np

from viva_simulation_core.composites.sepsis_sbml import (
    SEPSIS_OBSERVABLES, make_sepsis_sbml_composite, extract_observables)

ADAPTER_ADDRESS = 'viva_simulation_core.analysis.vc_adapter.VivaCoreTelluriumProcess'


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def to_viva_core_spec(doc: dict) -> str:
    """Serialize the composite document to a viva-core JSON spec.

    Swaps in the viva-core adapter address, drops the empty species/rates
    ports, restricts ``parameters`` to the shared observables, and adds the
    ``global_time`` store the Rust engine requires.
    """
    d = copy.deepcopy(doc)
    d['tellurium']['address'] = ADAPTER_ADDRESS
    for k in ('species', 'rates'):
        d['tellurium']['outputs'].pop(k, None)
        d['stores'].pop(k, None)
        d['emitter']['inputs'].pop(k, None)
        d['emitter']['config']['emit'].pop(k, None)
    ids = list(SEPSIS_OBSERVABLES.values())
    d['stores']['parameters'] = {i: d['stores']['parameters'][i] for i in ids}
    d['global_time'] = 0.0
    return json.dumps(d, default=_json_default)


def run_python(doc: dict, horizon_s: float) -> dict:
    from process_bigraph import Composite, allocate_core, gather_emitter_results
    from process_bigraph.emitter import RAMEmitter
    from viva_tellurium.processes import TelluriumProcess
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    t0 = time.perf_counter()
    sim = Composite({'state': copy.deepcopy(doc)}, core=core)
    sim.run(horizon_s)
    wall = time.perf_counter() - t0
    obs = extract_observables(gather_emitter_results(sim)[('emitter',)])
    return {'final': {k: v[-1] for k, v in obs.items()}, 'wall_s': wall}


def run_viva_core(doc: dict, n_ticks: int) -> dict:
    import viva_core
    spec = to_viva_core_spec(doc)
    t0 = time.perf_counter()
    rep = viva_core.run_report(spec, n_ticks)
    wall = time.perf_counter() - t0
    params = rep['state']['stores']['parameters']
    return {'final': {k: float(params[v]) for k, v in SEPSIS_OBSERVABLES.items()},
            'wall_s': wall, 'sim_time': rep['state']['global_time']}


def run_under_both(severity: int = 0, n_ticks: int = 60, interval_s: float = 10.0,
                   infection_start_s: float | None = None,
                   rtol: float = 1e-5, atol: float = 1e-9) -> dict:
    # NB: the first roadrunner run in a fresh process differs from every later run
    # by ~1e-6 relative (cold-start effect); warm runs are bit-identical across engines.
    doc = make_sepsis_sbml_composite(severity=severity, infection_start_s=infection_start_s,
                                     horizon_s=n_ticks * interval_s, interval_s=interval_s)
    py = run_python(doc, n_ticks * interval_s)
    rs = run_viva_core(doc, n_ticks)
    diffs = {k: abs(py['final'][k] - rs['final'][k]) for k in py['final']}
    ok_tol = all(np.isclose(py['final'][k], rs['final'][k], rtol=rtol, atol=atol)
                 for k in py['final'])
    bit = all(py['final'][k] == rs['final'][k] for k in py['final'])
    verdict = 'bit-identical' if bit else ('within-tolerance' if ok_tol else 'DIVERGENT')
    return {'python': py, 'viva_core': rs, 'abs_diff': diffs, 'verdict': verdict,
            'rtol': rtol, 'atol': atol,
            'wall_ratio_python_over_rust': py['wall_s'] / rs['wall_s'],
            'n_ticks': n_ticks, 'interval_s': interval_s, 'severity': severity}


if __name__ == '__main__':
    r = run_under_both()
    print(json.dumps({k: v for k, v in r.items()}, indent=1, default=float))
