"""Bridge: viva-impulse discrete events <-> the Tellurium engine's parameters_in.

viva-impulse events are engine-agnostic — they write discrete state (flags, a seed)
to plain stores. This Process couples that discrete layer to a ``TelluriumProcess``
running the *event-free* SBML, through the engine's ``parameters_in`` override channel:

- **Held flags** (``infOn``, ``TLR``) are forwarded every tick as an
  ``overwrite[map[float]]`` so the engine holds them at their current value — exactly
  the semantics an SBML held-mode flag has.
- **The one-shot seed** (``pt_seed``, the tissue pathogen ``PT`` bolus) is injected into
  ``parameters_in`` for a *single* tick when it first arrives, then omitted. Because the
  override map is replace-semantics, omitting ``PT`` afterwards leaves the engine's own
  rate-rule integration of ``PT`` untouched — the bolus jumps once and then evolves, as
  a true impulsive state jump should (a *persisted* override would wrongly pin it).
- **Sensing**: the engine's current ``PT`` is read back out as ``PT_sensed`` so a
  viva-impulse ``ThresholdSwitch`` can watch it and toggle ``TLR`` with hysteresis.

This is the only sepsis-specific glue the hybrid needs; the discrete logic lives in the
general viva-impulse ScheduledEvent / ThresholdSwitch.
"""
from __future__ import annotations

from process_bigraph import Process


class TelluriumImpulseBridge(Process):
    """Couple viva-impulse event stores to a Tellurium engine's parameters_in override.

    Inputs
    ------
    infOn, TLR : float
        Held discrete flags written by the viva-impulse events.
    pt_seed : float
        The tissue-pathogen seed value (0 until the infection event fires); injected once.
    parameters : map[float]
        The engine's current global-parameter map (read to sense ``PT``).

    Outputs
    -------
    parameters_in : overwrite[map[float]]
        Override map applied by the engine before its next step (replace semantics).
    PT_sensed : overwrite[float]
        The engine's current ``PT``, for a ThresholdSwitch to watch.
    """

    config_schema: dict = {}

    def describe(self):
        return {
            "summary": "Bridges viva-impulse discrete events to the Tellurium parameters_in "
                       "override: forwards held flags, injects the one-shot PT seed once, "
                       "and senses PT back out for a ThresholdSwitch.",
            "description": (
                "Couples the engine-agnostic viva-impulse discrete layer to a TelluriumProcess "
                "running the event-free sepsis SBML. Held flags (infOn, TLR) are forwarded every "
                "tick; the PT seed is injected for a single tick then omitted so the engine's "
                "rate-rule integration of PT is not pinned; PT is sensed back out as PT_sensed."),
            "ports": {
                "infOn": "held infection flag (dimensionless), forwarded to the engine",
                "TLR": "held TLR mode flag (dimensionless), forwarded to the engine",
                "pt_seed": "tissue-pathogen seed bolus (bacteria count), injected once",
                "parameters": "engine global-parameter map (read to sense PT)",
                "parameters_in": "override map applied before the engine's next step (replace)",
                "PT_sensed": "engine tissue pathogen PT (bacteria count), for a ThresholdSwitch",
            },
        }

    def inputs(self):
        return {
            "infOn": "float",
            "TLR": "float",
            "pt_seed": "float",
            "parameters": "map[float]",
        }

    def outputs(self):
        return {
            "parameters_in": "overwrite[map[float]]",
            "PT_sensed": "overwrite[float]",
        }

    def update(self, state, interval):
        params = state.get("parameters") or {}
        pt = float(params.get("PT", 0.0))
        override = {
            "infOn": float(state.get("infOn", 0.0)),
            "TLR": float(state.get("TLR", 0.0)),
        }
        seed = float(state.get("pt_seed", 0.0))
        if seed > 0.0 and not getattr(self, "_seeded", False):
            self._seeded = True
            override["PT"] = seed          # one-shot: include PT this tick only
        return {"parameters_in": override, "PT_sensed": pt}
