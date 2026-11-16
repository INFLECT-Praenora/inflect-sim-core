"""SBML-side sepsis baseline composite (Tellurium / roadrunner).

Loads the vendored BioGears-sepsis SBML and emits the shared comparison
observables. All observables are global parameters / assignment rules of the
SBML model, so they appear in the Tellurium process' ``parameters`` store.

Scenario knobs ``severity`` (0 none, 1 mild, 2 moderate, 3 severe) and
``infectionStart`` (s) are model parameters. ``TelluriumProcess`` re-applies
``parameter_overrides`` after its init reset(), so they take effect; the event
``E_infect`` fires at t >= infectionStart and seeds tissue pathogen PT from
``severity`` (1e6 / 5e6 / 2.5e7). Healthy baseline = severity 0; there ``infectionStart`` is deliberately left at
its model default (1e12) so the infection event cannot fire.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator
from viva_tellurium import make_tellurium_document

from viva_simulation_core.core import build_core

SBML_PATH = (Path(__file__).resolve().parents[2] / 'workspace' / 'investigations'
             / 'biogears-sbml-sepsis' / 'sources' / 'biogears_sepsis.xml')

# shared observable name -> SBML variable id
SEPSIS_OBSERVABLES = {
    'SBP': 'SBP',                    # mmHg
    'DBP': 'DBP',                    # mmHg
    'MAP': 'MAP',                    # mmHg
    'HR': 'HR',                      # 1/min
    'CO': 'COmeas',                  # L/min (SV*HR/1000)
    'SVR': 'SVR',                    # mmHg min/L-ish: (MAP-CVP)/(CO*1000/60)
    'blood_volume': 'Vblood',        # mL, sum of node q + U
    'tissue_integrity': 'TI',        # 0..1
    'blood_pathogen': 'PB',          # PB = blood pathogen; PT = tissue pathogen
}


@composite_generator(
    name="sepsis_sbml",
    description="The BioGears sepsis core as its SBML/Antimony extraction (Sauro), run via "
                "Tellurium / roadrunner. Emits the 9 shared comparison observables (SBP, DBP, MAP, "
                "HR, CO, SVR, blood volume, tissue integrity, blood pathogen). Infection is seeded "
                "from `severity` at t = infection_start_s. Healthy at severity 0.",
    parameters={
        "severity": {"type": "integer", "default": 0,
                     "description": "0 none / 1 mild / 2 moderate / 3 severe (seeds tissue pathogen)"},
        "infection_start_s": {"type": "float", "default": 0.0,
                              "description": "Infection start time in seconds (applied only when severity > 0)"},
        "interval_s": {"type": "float", "default": 10.0, "description": "Seconds per update"},
    },
    core_extensions=[build_core],
    default_n_steps=360,
    fidelity="reproduction",
)
def make_sepsis_sbml_composite(core=None, *, severity: int = 0,
                               infection_start_s: float | None = None,
                               horizon_s: float = 3600.0,
                               interval_s: float = 10.0) -> dict:
    """Build the Tellurium composite document (healthy at severity=0).

    Registered as the ``sepsis_sbml`` composite generator (the SBML counterpart
    to viva-biogears' ``biogears_sepsis``). ``core`` is accepted for the
    generator protocol but unused — Tellurium/roadrunner carries its own solver.
    ``horizon_s`` is advisory (run the Composite for that long); the document
    itself only fixes the process interval.
    """
    overrides = {'severity': float(severity)}
    # NB: the model's piecewise seed maps severity 0 to the *mild* dose, so a
    # healthy run must never fire E_infect: keep the default infectionStart=1e12.
    if severity > 0:
        overrides['infectionStart'] = float(infection_start_s) if infection_start_s is not None else 1.0
    doc = make_tellurium_document(
        model_file=str(SBML_PATH), model_format='sbml',
        parameter_overrides=overrides, interval=float(interval_s))
    return doc


def extract_observables(rows) -> dict[str, list[float]]:
    """Emitter rows -> {observable: [values over time]} (rows lacking parameters skipped)."""
    out = {k: [] for k in SEPSIS_OBSERVABLES}
    for r in rows:
        p = r.get('parameters')
        if not p:
            continue
        for k, v in SEPSIS_OBSERVABLES.items():
            out[k].append(float(p[v]))
    return out if out['SBP'] else {}
