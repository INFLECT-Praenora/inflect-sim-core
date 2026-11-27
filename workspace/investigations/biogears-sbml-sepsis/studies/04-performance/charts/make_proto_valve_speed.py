"""Study-04 arm: event-free proto_valve vs the full sepsis_sbml model — speed.

Reads proto_valve_compare.json (written by run_proto_valve_compare.py: both models
run through the same Composite harness on this machine) and draws the measured
wall-clock for the matched severity-3 run, so the proto_valve speedup sits in the
performance study on an apples-to-apples basis.

Run: python make_proto_valve_speed.py   (writes 05_proto_valve_speed.{svg,png})
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
GREEN, TEAL = "#55A868", "#2a9d8f"

d = json.loads((HERE / "proto_valve_compare.json").read_text())
sc = d["scenario"]
tm = d["timing"]
hours = sc["horizon_s"] / 3600.0

fig, ax = plt.subplots(figsize=(6.2, 4.8))
labels = ["sepsis_sbml\n(full, event-driven)", "proto_valve\n(event-free)"]
walls = [tm["full_wall_s"], tm["proto_valve_wall_s"]]
bars = ax.bar(labels, walls, color=[GREEN, TEAL], width=0.6)
ax.set_ylabel(f"wall-clock, one {hours:.0f} h severity-{sc['severity']} run  (s)")
ax.set_title("Event-free reduction (same exact SBML engine, same harness)\n"
             f"measured this machine — proto_valve is {tm['speedup']:.2f}x faster")
for bar, s in zip(bars, walls):
    ax.text(bar.get_x() + bar.get_width() / 2, s, f"{s:.1f} s",
            ha="center", va="bottom", fontweight="bold")
ax.annotate(f"{tm['speedup']:.2f}x\n(no E_beat root returns\n/ reinitializations)",
            xy=(1, walls[1]), xytext=(0.5, max(walls) * 0.6),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))
ax.margins(y=0.18)
ax.grid(axis="y", alpha=0.3)

cap = (f"Event-free proto_valve vs the full event-driven sepsis_sbml model, both run "
       f"through the identical process-bigraph Composite harness on this machine "
       f"(severity {sc['severity']}, {hours:.0f} h model time, {sc['interval_s']:.0f} s "
       f"interval). Removing the per-beat E_beat event and the piecewise pulOpen valve "
       f"removes all CVODE root returns and reinitializations, giving a measured "
       f"{tm['speedup']:.2f}x speedup ({tm['full_wall_s']:.1f} s -> "
       f"{tm['proto_valve_wall_s']:.1f} s) within the same exact-SBML engine. The "
       f"absolute seconds are harness-driven (per-interval reset + parameter re-apply) "
       f"and are higher than the prototype report's raw-wheel single-call timings; only "
       f"the ratio is comparable across arms, and both arms pay the overhead identically. "
       f"The prototype report's raw-wheel measurements give a consistent 1.24-1.44x over "
       f"30 min to 48 h.")

fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(HERE / f"05_proto_valve_speed.{ext}", dpi=140, bbox_inches="tight")
json.dump({"title": "05_proto_valve_speed", "interpretation": cap, "caption": cap},
          open(HERE / "05_proto_valve_speed.meta.json", "w"), indent=1)
print("wrote 05_proto_valve_speed.svg / .png / .meta.json")
