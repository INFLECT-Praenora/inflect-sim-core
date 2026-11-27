"""Hybrid sepsis composite: event-free SBML core + viva-impulse discrete events.

The same BioGears sepsis core, but with its discrete events **lifted out of the SBML
into the vivarium layer**. The continuous core is the fully event-free SBML
(``proto_valve_eventfree.xml`` — proto_valve with the per-beat events AND the infection/
TLR events removed), run by the Tellurium engine. The discrete events are
general viva-impulse Steps:

- ``ScheduledEvent`` — the timed infection (``E_infect``): at ``infection_start_s`` it
  raises ``infOn`` and arms the tissue-pathogen seed.
- ``ThresholdSwitch`` — the TLR hysteresis (``E_tlrOn``/``E_tlrOff``): watches the
  sensed ``PT`` and toggles ``TLR`` at ``pUpper``/``pLower``.

A :class:`~viva_simulation_core.steps.tellurium_impulse_bridge.TelluriumImpulseBridge`
couples the engine-agnostic events to the engine's ``parameters_in`` override channel.
The composite emits the same 9 shared comparison observables, and reproduces the full
event-driven model to within the tick-quantization of the infection timing — the honest
cost of lifting the events to the composition layer, where they are now explicit and
swappable (change the protocol, chain a controller) without touching the SBML.

Requires the ``impulse`` engine (viva-impulse) and ``tellurium``.
"""
from __future__ import annotations

from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from viva_simulation_core.core import build_core
from viva_simulation_core.engines import require as require_engine

EVENTFREE_SBML_PATH = (Path(__file__).resolve().parents[2] / 'workspace' / 'investigations'
                       / 'biogears-sbml-sepsis' / 'sources' / 'proto_valve_eventfree.xml')

# tissue-pathogen seed by severity (the piecewise in the full model's E_infect)
_SEED = {1: 1.0e6, 2: 5.0e6, 3: 2.5e7}
# TLR hysteresis band (the full model's pUpper / pLower)
_P_UPPER, _P_LOWER = 2.0e6, 1.0e3


def _register_impulse(core):
    """Deferred core-extension: register viva-impulse's processes. Imported lazily (only
    when a composite is actually built) so a bare package import pulls in no engine."""
    return require_engine('impulse').build_core(core)


@composite_generator(
    name="sepsis_hybrid",
    description="Hybrid sepsis: the event-free BioGears SBML core (Tellurium) with its "
                "discrete events lifted into the vivarium layer as viva-impulse Steps — a "
                "ScheduledEvent for the timed infection (E_infect) and a ThresholdSwitch for "
                "the TLR hysteresis (E_tlrOn/Off), coupled via TelluriumImpulseBridge. Emits "
                "the 9 shared observables and reproduces the full event-driven model (to the "
                "tick-quantization of the infection timing). Healthy at severity 0.",
    parameters={
        "severity": {"type": "integer", "default": 3,
                     "description": "0 none / 1 mild / 2 moderate / 3 severe (seeds PT)"},
        "infection_start_s": {"type": "float", "default": 1.0,
                              "description": "Infection start time (s); the ScheduledEvent fires at the first tick >= this"},
        "interval_s": {"type": "float", "default": 10.0, "description": "Seconds per update"},
    },
    core_extensions=[build_core, _register_impulse],
    default_n_steps=360,
    fidelity="reproduction",
)
def make_sepsis_hybrid_composite(core=None, *, severity: int = 3,
                                 infection_start_s: float = 1.0,
                                 interval_s: float = 10.0) -> dict:
    """Build the hybrid document: event-free Tellurium core + viva-impulse events + bridge."""
    make_tellurium_document = require_engine('tellurium').make_tellurium_document
    iv = float(interval_s)

    doc = make_tellurium_document(
        model_file=str(EVENTFREE_SBML_PATH), model_format='sbml',
        parameter_overrides={}, interval=iv)

    # Rename the engine node and wire its override input.
    phys = doc.pop('tellurium')
    phys['inputs'] = {'parameters_in': ['stores', 'parameters_in']}
    doc['physiology'] = phys

    # The event-free SBML has zero floating species; drop the species/rates emitter
    # ports (an empty array[(0,)] store clashes with the emitter's map schema).
    emit = doc['emitter']['config']['emit']
    for k in ('species', 'rates'):
        emit.pop(k, None)
        doc['emitter']['inputs'].pop(k, None)

    seed = _SEED.get(int(severity), 0.0)

    # --- The discrete layer: general viva-impulse events (only when infecting) ---
    if severity > 0:
        doc['infection'] = {
            '_type': 'step', 'address': 'local:ScheduledEvent',
            'config': {'time': float(infection_start_s),
                       'actions': {'infOn': '1', 'pt_seed': repr(float(seed))}},
            'inputs': {'global_time': ['global_time'], 'infOn': ['infOn'], 'pt_seed': ['pt_seed']},
            'outputs': {'infOn': ['infOn'], 'pt_seed': ['pt_seed']},
        }
        doc['tlr_switch'] = {
            '_type': 'step', 'address': 'local:ThresholdSwitch',
            'config': {'watch': 'PT_sensed', 'upper': _P_UPPER, 'lower': _P_LOWER,
                       'up_actions': {'TLR': '1'}, 'down_actions': {'TLR': '0'}},
            'inputs': {'global_time': ['global_time'], 'PT_sensed': ['PT_sensed'], 'TLR': ['TLR']},
            'outputs': {'TLR': ['TLR']},
        }

    # --- The engine bridge: events <-> parameters_in override, PT sensing ---
    doc['bridge'] = {
        '_type': 'process', 'address': 'local:TelluriumImpulseBridge', 'config': {},
        'interval': iv,
        'inputs': {'infOn': ['infOn'], 'TLR': ['TLR'], 'pt_seed': ['pt_seed'],
                   'parameters': ['stores', 'parameters']},
        'outputs': {'parameters_in': ['stores', 'parameters_in'], 'PT_sensed': ['PT_sensed']},
    }

    doc.update({'infOn': 0.0, 'TLR': 0.0, 'pt_seed': 0.0, 'PT_sensed': 0.0})
    return doc
