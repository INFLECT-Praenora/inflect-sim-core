"""Multi-engine runner + Lotka-Volterra composite — the model-interchange machinery.

The Lotka-Volterra composite is a small model with floating species (unlike the
sepsis composites), run under every available exact-ODE engine through the
engine-core FullReportUTCStep contract plus the viva-core (Rust) arm.
"""
import copy

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.processes import TelluriumProcess

from viva_simulation_core.analysis import run_engines
from viva_simulation_core.composites.lotka_volterra import (
    LV_ANTIMONY, make_lotka_volterra_composite)


def _final_species(doc):
    import tellurium as te
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    sim = Composite({'state': doc}, core=core)
    sim.run(30.0)
    order = list(te.loada(LV_ANTIMONY).getFloatingSpeciesIds())
    rows = gather_emitter_results(sim)[('emitter',)]
    sp = [r['species'] for r in rows if r.get('species') is not None and len(r['species'])]
    return {order[i]: float(sp[-1][i]) for i in range(len(order))}


# --- the composite ---------------------------------------------------------

def test_lotka_volterra_composite_runs():
    """The composite builds a Tellurium document that integrates to the known LV
    attractor-approach state at t=30 with the default parameters."""
    final = _final_species(copy.deepcopy(make_lotka_volterra_composite(interval_s=0.1)))
    assert abs(final['X'] - 0.0491) < 5e-3
    assert abs(final['Y'] - 0.1595) < 5e-3


def test_lotka_volterra_params_change_trajectory():
    """alpha/beta/delta map to the model's rate constants and move the state."""
    base = _final_species(copy.deepcopy(make_lotka_volterra_composite(beta=0.4)))
    hi = _final_species(copy.deepcopy(make_lotka_volterra_composite(beta=0.9)))
    assert abs(base['X'] - hi['X']) > 1e-3 or abs(base['Y'] - hi['Y']) > 1e-3


# --- the multi-engine runner ----------------------------------------------

def test_run_all_engines_agree_on_species():
    """Every available exact-ODE engine agrees with Tellurium on the final
    floating-species values (the model-interchange guarantee). At minimum
    Tellurium itself runs; COPASI / viva-core join when installed."""
    res = run_engines.run_all(LV_ANTIMONY, model_format='antimony', time=30.0, n_points=301)
    ran = {k: v for k, v in res.items() if 'final_all' in v}
    assert 'tellurium' in ran, f"tellurium did not run: {res.get('tellurium')}"
    disagree = run_engines.compare(res, ref='tellurium')
    for engine, rel in disagree.items():
        assert rel < 1e-2, f"{engine} diverged from tellurium on final species: {rel}"


def test_run_all_skips_unavailable_gracefully():
    """An unknown engine name is reported as skipped, never raised."""
    res = run_engines.run_all(LV_ANTIMONY, model_format='antimony', engines=['tellurium'],
                              include_viva_core=False)
    assert set(res) == {'tellurium'}
    assert 'final_all' in res['tellurium']


def test_species_comparison_ignores_reaction_fluxes():
    """compare() restricts to floating species, not the reaction-flux columns the
    FullReport step also returns."""
    res = run_engines.run_all(LV_ANTIMONY, model_format='antimony', time=5.0, n_points=51,
                              engines=['tellurium'], include_viva_core=False)
    r = res['tellurium']
    # FullReport reports species AND reaction fluxes; the species subset is just X,Y
    assert set(r['species_ids']) == {'X', 'Y'}
    assert len(r['columns']) > 2  # species + reaction fluxes present in the raw report


# --- the generic viva-core (Rust) adapter ----------------------------------

@pytest.mark.slow
def test_generic_viva_core_adapter_emits_species_leaves():
    """VivaCoreTelluriumGeneric emits floating species as per-id overwrite[float]
    leaves (what viva-core's Rust engine requires) and drops rates/parameters."""
    pytest.importorskip('viva_core')
    from viva_simulation_core.analysis.vc_adapter import VivaCoreTelluriumGeneric
    p = VivaCoreTelluriumGeneric(
        {'model': LV_ANTIMONY, 'model_format': 'antimony', 'interval': 0.1},
        core=allocate_core())
    outs = p.outputs()
    assert set(outs['species']) == {'X', 'Y'}
    assert all(v == 'overwrite[float]' for v in outs['species'].values())
    assert 'rates' not in outs and 'parameters' not in outs


@pytest.mark.slow
def test_run_all_includes_viva_core_and_agrees():
    """The viva-core (Rust) arm runs the same model and agrees with Tellurium."""
    pytest.importorskip('viva_core')
    res = run_engines.run_all(LV_ANTIMONY, model_format='antimony', time=30.0, n_points=301)
    assert 'viva-core' in res and 'final_all' in res['viva-core'], res.get('viva-core')
    assert run_engines.compare(res, ref='tellurium')['viva-core'] < 1e-2


@pytest.mark.slow
def test_julia_arm_isolated_agrees(tmp_path):
    """Julia (DifferentialEquations.jl) reads the SAME SBML and reaches the LV final
    state. Run in a SUBPROCESS: importing juliacall loads a Julia native runtime that
    segfaults alongside roadrunner/COPASI in one process. The SBML is pre-built here
    (roadrunner in this process) and the julia-only subprocess never imports it."""
    import json
    import subprocess
    import sys

    pytest.importorskip('viva_julia_diffeqs')
    from viva_simulation_core.analysis.run_engines import _to_sbml
    sbml_path = tmp_path / "lv.xml"
    sbml_path.write_text(_to_sbml(LV_ANTIMONY, "antimony"))
    code = (
        "import json, sys\n"
        "from viva_simulation_core.analysis.run_engines import run_julia\n"
        "r = run_julia(open(sys.argv[1]).read(), 30.0, 301)\n"
        "print('OUT' + json.dumps(r['final_all']))\n"
    )
    out = subprocess.run([sys.executable, "-c", code, str(sbml_path)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    final = json.loads(out.stdout.split("OUT", 1)[1])
    assert abs(final['X'] - 0.0491) < 5e-3 and abs(final['Y'] - 0.1595) < 5e-3
