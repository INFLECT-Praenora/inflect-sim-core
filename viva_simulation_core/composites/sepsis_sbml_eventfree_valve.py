"""Event-free (``proto_valve``) sepsis SBML composite (Tellurium / roadrunner).

A near-exact *reduction* of the full :mod:`sepsis_sbml` model: the same 91-state
BioGears sepsis core, but with the per-beat heartbeat event (``E_beat``) and the
piecewise pulmonic-valve gate (``pulOpen``) removed — the two sources of CVODE
root returns and reinitialization in the heartbeat. It replaces them with:

- a continuous phase oscillator driving the heartbeat shape ``eshape``;
- per-artery pulmonic-valve gating instead of the piecewise ``pulOpen``;
- ``SBP := MAP + 19.1`` / ``DBP := MAP - 22`` and 5 s filters for MAP/CVP/COmeas
  in place of the per-beat latched pressures;
- immediate (rather than next-beat) heart-rate changes.

It keeps the three infection/TLR events (``E_tlrOn``, ``E_tlrOff``, ``E_infect``)
unchanged, so the ``severity`` / ``infection_start_s`` scenario knobs behave
exactly as in :mod:`sepsis_sbml`, and it emits the **same 9 shared comparison
observables**. This makes it a drop-in faster arm for the engine/performance and
fidelity comparison studies: on the roadrunner 2.12.0 wheel it runs a severe 48 h
course in ~26 min vs ~34 min for the full model (~1.3x), while per-beat
hemodynamics stay within ~0.2 mmHg and the immune variables within ~1e-6 (``Alb_v``
within ~1e-4). See ``sources/proto_valve.{ant,xml}`` and the extraction report.

Fidelity caveat (per the prototype report): the per-artery pulmonic valve, the
``SBP``/``DBP`` offsets, and the 5 s filters are physiological stand-ins, not values
taken from the full model; heart-rate changes act immediately (~0.1% HR shift); and
instantaneous values do not line up because the beat phase differs — compare
*per-beat* summaries, not single time points. This is a speed experiment, not a
validated replacement for ``E_beat``.

Because the model has no heartbeat event, CVODE takes no root returns; over a very
long *single* integration call this can hit ``CV_TOO_MUCH_WORK``. The Composite
drives the model in short ``interval_s`` chunks (CVODE's step counter resets each
interval), so the default ``maximum_num_steps`` is sufficient here; raise the
interval count rather than the per-call step budget for long horizons.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from viva_simulation_core.core import build_core
from viva_simulation_core.engines import require as require_engine
# Shared with the full model — same observables, same emitter extraction.
from viva_simulation_core.composites.sepsis_sbml import (  # noqa: F401 (re-export)
    SEPSIS_OBSERVABLES,
    extract_observables,
)

SBML_PATH = (Path(__file__).resolve().parents[2] / 'workspace' / 'investigations'
             / 'biogears-sbml-sepsis' / 'sources' / 'proto_valve.xml')


@composite_generator(
    name="sepsis_sbml_eventfree_valve",
    description="Event-free (proto_valve) reduction of the BioGears sepsis core: the same SBML "
                "model with the per-beat E_beat event and piecewise pulOpen valve replaced by a "
                "continuous phase oscillator + per-artery valve gating. Runs via Tellurium / "
                "roadrunner, emits the same 9 shared comparison observables (SBP, DBP, MAP, HR, "
                "CO, SVR, blood volume, tissue integrity, blood pathogen), and keeps the infection "
                "events — a ~1.3x faster, per-beat-faithful (~0.2 mmHg) arm for the comparison "
                "studies. Healthy at severity 0.",
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
def make_sepsis_sbml_eventfree_valve_composite(core=None, *, severity: int = 0,
                                               infection_start_s: float | None = None,
                                               horizon_s: float = 3600.0,
                                               interval_s: float = 10.0) -> dict:
    """Build the Tellurium composite document for the event-free ``proto_valve`` model.

    Mirrors :func:`sepsis_sbml.make_sepsis_sbml_composite` exactly — same scenario
    knobs and the same ``infectionStart`` handling (severity 0 must never fire
    ``E_infect``, so its default 1e12 is kept) — but points roadrunner at
    ``sources/proto_valve.xml`` instead of the full ``biogears_sepsis.xml``. ``core``
    is accepted for the generator protocol but unused (roadrunner carries its solver).
    """
    make_tellurium_document = require_engine('tellurium').make_tellurium_document
    overrides = {'severity': float(severity)}
    if severity > 0:
        overrides['infectionStart'] = float(infection_start_s) if infection_start_s is not None else 1.0
    doc = make_tellurium_document(
        model_file=str(SBML_PATH), model_format='sbml',
        parameter_overrides=overrides, interval=float(interval_s))
    return doc
