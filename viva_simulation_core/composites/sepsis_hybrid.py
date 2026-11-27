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

The discrete layer is **extracted from the model, not hand-authored**: both the event-free
SBML and the event specs this composite builds from (``sources/sepsis_events.json`` — the
thresholds and the severity-dependent seed formula) are generated from ``proto_valve.xml``
by ``sources/extract_hybrid_events.py`` (``viva_impulse.extract``). The composite supplies
only the engine-coupling routing (one-shot PT seed, TLR owned by the switch, PT sensed back).

Requires the ``impulse`` engine (viva-impulse) and ``tellurium``.
"""
from __future__ import annotations

import json
from pathlib import Path

from process_bigraph.composite_generator import composite_generator

from viva_simulation_core.core import build_core
from viva_simulation_core.engines import require as require_engine

_SOURCES = (Path(__file__).resolve().parents[2] / 'workspace' / 'investigations'
            / 'biogears-sbml-sepsis' / 'sources')
EVENTFREE_SBML_PATH = _SOURCES / 'proto_valve_eventfree.xml'
# The discrete layer is GENERATED from the model: extract_hybrid_events.py runs
# viva_impulse.extract over proto_valve.xml and writes these specs (thresholds, the
# severity-dependent seed formula). Re-run that script if the model changes.
EVENTS_SPEC_PATH = _SOURCES / 'sepsis_events.json'


def _lifted_spec(kind: str) -> dict:
    """The extracted spec for the one lifted event of the given viva-impulse kind."""
    lifted = json.loads(EVENTS_SPEC_PATH.read_text())["lifted"]
    return next(s for s in lifted if s["kind"] == kind)


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

    # --- The discrete layer: EXTRACTED from the model (sepsis_events.json), not hardcoded ---
    # The specs carry the triggers/thresholds/seed formula; the composite supplies only the
    # engine-coupling routing: the ScheduledEvent's one-shot PT seed goes to the `pt_seed`
    # store (injected once by the bridge), TLR is owned by the ThresholdSwitch (so dropped
    # from the infection event), and the switch watches the engine-sensed PT (`PT_sensed`).
    if severity > 0:
        infect = _lifted_spec("ScheduledEvent")["config"]["actions"]
        doc['infection'] = {
            '_type': 'step', 'address': 'local:ScheduledEvent',
            'config': {'time': float(infection_start_s),
                       # held infection flag + the extracted severity-dependent seed formula
                       'actions': {'infOn': infect['infOn'], 'pt_seed': infect['PT']},
                       'params': {'severity': float(severity)}},
            'inputs': {'global_time': ['global_time'], 'infOn': ['infOn'], 'pt_seed': ['pt_seed']},
            'outputs': {'infOn': ['infOn'], 'pt_seed': ['pt_seed']},
        }
        tlr = _lifted_spec("ThresholdSwitch")["config"]
        doc['tlr_switch'] = {
            '_type': 'step', 'address': 'local:ThresholdSwitch',
            'config': {'watch': 'PT_sensed', 'upper': tlr['upper'], 'lower': tlr['lower'],
                       'up_actions': tlr['up_actions'], 'down_actions': tlr['down_actions']},
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
