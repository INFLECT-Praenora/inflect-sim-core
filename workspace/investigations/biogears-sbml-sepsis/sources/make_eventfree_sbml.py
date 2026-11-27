"""Generate the fully event-free SBML from proto_valve.

proto_valve already removed the per-beat heartbeat events (E_beat, pulOpen). This
removes the remaining three discrete events — E_tlrOn, E_tlrOff, E_infect — so the
continuous core has NO events at all. The infection (E_infect) and TLR hysteresis
(E_tlrOn/Off) are then supplied from the vivarium layer by viva-impulse Steps (see
the `sepsis_hybrid` composite), reproducing the full model with the discrete events
made explicit and swappable.

PT, TLR, infOn remain as model variables (initial 0), driven externally via the
engine's parameters_in override channel.

Run from this dir:  python make_eventfree_sbml.py
"""
from pathlib import Path

import tellurium as te

HERE = Path(__file__).parent
DROP_PREFIXES = ("E_tlrOn:", "E_tlrOff:", "E_infect:")


def main():
    ant = (HERE / "proto_valve.ant").read_text().splitlines()
    kept = [ln for ln in ant if not ln.strip().startswith(DROP_PREFIXES)]
    removed = len(ant) - len(kept)
    assert removed == 3, f"expected to remove 3 event lines, removed {removed}"
    ef_ant = "\n".join(kept) + "\n"
    (HERE / "proto_valve_eventfree.ant").write_text(ef_ant)

    rr = te.loada(ef_ant)
    assert rr.getNumEvents() == 0, f"model still has {rr.getNumEvents()} events"
    (HERE / "proto_valve_eventfree.xml").write_text(rr.getSBML())
    print(f"wrote proto_valve_eventfree.{{ant,xml}} (removed {removed} events; 0 remain)")


if __name__ == "__main__":
    main()
