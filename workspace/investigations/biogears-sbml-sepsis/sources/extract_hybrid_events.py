"""Generate the hybrid's event-free SBML + extracted event specs from proto_valve.

Closes the loop: instead of hand-authoring the sepsis_hybrid composite's discrete layer,
we EXTRACT it from the model with ``viva_impulse.extract``. The extractor lifts
proto_valve's discrete events (E_infect, E_tlrOn/E_tlrOff) into viva-impulse specs and
produces the event-free SBML core. Writes:

  - proto_valve_eventfree.xml  — the continuous core (events removed)
  - sepsis_events.json         — the lifted event specs the composite builds from
                                 (thresholds, the severity-dependent seed formula, etc.)

The sepsis_hybrid composite reads sepsis_events.json, so its event thresholds and seed
are DERIVED from the SBML, not hand-copied. Re-run this after the model changes.

Run from this dir:  python extract_hybrid_events.py   (needs viva-impulse[sbml] + libsbml)
"""
from __future__ import annotations

import json
from pathlib import Path

from viva_impulse.extract import extract_events

HERE = Path(__file__).parent


def main():
    res = extract_events(str(HERE / "proto_valve.xml"))   # lifts all 3 events (E_beat already gone)
    (HERE / "proto_valve_eventfree.xml").write_text(res.eventfree_sbml)
    payload = {
        "source": "proto_valve.xml",
        "generated_by": "viva_impulse.extract.extract_events",
        "lifted": [{"name": s.name, "kind": s.kind, "config": s.config, "sbml_ids": s.sbml_ids}
                   for s in res.lifted],
        "kept": [{"name": s.name, "kind": s.kind, "sbml_ids": s.sbml_ids, "note": s.note}
                 for s in res.kept],
        "model_values": res.model_values,
    }
    (HERE / "sepsis_events.json").write_text(json.dumps(payload, indent=1))
    print(f"lifted {len(res.lifted)} event group(s); event-free SBML + sepsis_events.json written")
    for s in res.lifted:
        print(f"  {s.name} -> {s.kind}: {s.config}")
    for s in res.kept:
        print(f"  KEPT {s.name}: {s.note}")


if __name__ == "__main__":
    main()
