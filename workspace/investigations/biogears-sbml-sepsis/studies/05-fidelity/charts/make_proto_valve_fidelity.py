"""Study-05 arm: event-free proto_valve vs the full sepsis_sbml model — fidelity.

Reads the same-harness comparison data (produced by study 04's
run_proto_valve_compare.py) and shows how closely the event-free reduction tracks
the full event-driven model within the *exact SBML engine*. This is a different
fidelity axis from the engine-vs-engine (BioGears C++ vs SBML) comparison that is
the rest of study 05: here both arms are the same roadrunner engine, so any
difference is the event-free structural change alone.

Run: python make_proto_valve_fidelity.py   (writes 04_eventfree_valve_fidelity.{svg,png})
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "04-performance" / "charts" / "proto_valve_compare.json"
FULL, PV = "#2f6fb3", "#2a9d8f"

d = json.loads(DATA.read_text())
sc = d["scenario"]
t = np.array(d["time_h"])
OBS = ["SBP", "DBP", "MAP", "HR", "CO", "SVR",
       "blood_volume", "tissue_integrity", "blood_pathogen"]
LAB = {"SBP": "Systolic BP (mmHg)", "DBP": "Diastolic BP (mmHg)", "MAP": "MAP (mmHg)",
       "HR": "Heart rate (/min)", "CO": "Cardiac output (L/min)", "SVR": "SVR",
       "blood_volume": "Blood volume (mL)", "tissue_integrity": "Tissue integrity",
       "blood_pathogen": "Blood pathogen"}

fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True)
for ax, o in zip(axes.flat, OBS):
    a = np.array(d["full"][o]); b = np.array(d["proto_valve"][o])
    ax.plot(t, a, color=FULL, lw=1.8, label="full (event-driven)")
    ax.plot(t, b, color=PV, lw=1.4, ls="--", label="proto_valve (event-free)")
    ag = d["agreement"][o]
    ax.set_title(f"{LAB[o]}\nmax |Δ| {ag['max_abs_diff']:.2g} (rel {ag['max_rel_diff']:.1g})",
                 fontsize=10)
    ax.grid(alpha=0.25)
axes.flat[0].legend(fontsize=9, loc="best")
for ax in axes[-1]:
    ax.set_xlabel("time (h)")
fig.suptitle("Event-free proto_valve vs full sepsis_sbml — same exact-SBML engine, same harness\n"
             f"(severity {sc['severity']}, {sc['horizon_s']/3600:.0f} h): the reduction tracks the "
             f"full model's filtered hemodynamics and immune core",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

ag = d["agreement"]
cap = (f"Event-free proto_valve vs the full event-driven sepsis_sbml model, both run "
       f"through the same roadrunner/CVODE engine and the same Composite harness "
       f"(severity {sc['severity']}, {sc['horizon_s']/3600:.0f} h). Removing E_beat and the "
       f"piecewise pulOpen valve leaves the filtered hemodynamics and the immune core "
       f"almost unchanged: MAP tracks within {ag['MAP']['max_abs_diff']:.2g} mmHg, HR within "
       f"{ag['HR']['max_abs_diff']:.2g} /min, and blood pathogen / tissue integrity agree to "
       f"{max(ag['blood_pathogen']['max_rel_diff'], ag['tissue_integrity']['max_rel_diff']):.1g} "
       f"relative. Cardiac output and SVR differ most (~{ag['CO']['max_rel_diff']*100:.0f}% / "
       f"~{ag['SVR']['max_rel_diff']*100:.0f}%) because the 5 s filters and the continuous beat "
       f"phase replace the per-beat latched values — instantaneous points are not meant to line "
       f"up (compare per-beat summaries). This is the fidelity cost of the event-free speedup, "
       f"measured within one engine so it isolates the structural change from the engine-vs-engine "
       f"divergence that is the rest of this study.")
json.dump({"title": "04_eventfree_valve_fidelity", "interpretation": cap, "caption": cap},
          open(HERE / "04_eventfree_valve_fidelity.meta.json", "w"), indent=1)
for ext in ("svg", "png"):
    fig.savefig(HERE / f"04_eventfree_valve_fidelity.{ext}", dpi=140, bbox_inches="tight")
print("wrote 04_eventfree_valve_fidelity.svg / .png / .meta.json")
