"""00 overview — REAL output of the sepsis core (exact SBML engine).

One spec, run under the exact Tellurium/roadrunner engine at healthy baseline
(severity 0) and severe sepsis (severity 3): four key observables over 48 h,
straight from the simulation. No schematic.
"""
from pathlib import Path
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = Path(__file__).parents[3] / "data" / "real_trajectories.json"
d = json.load(open(DATA))
h0, h3 = d["exact_sev0"], d["exact_sev3"]
panels = [("MAP", "MAP (mmHg)", 65, "shock < 65"),
          ("HR", "Heart rate (/min)", 90, "tachycardia > 90"),
          ("tissue_integrity", "Tissue integrity (0–1)", None, None),
          ("blood_pathogen", "Blood pathogen (a.u.)", None, None)]
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
for ax, (key, lab, thr, thrlab) in zip(axes.ravel(), panels):
    ax.plot(h0["t_h"], h0["obs"][key], color="#0ea5e9", lw=2.2, label="healthy (sev 0)")
    ax.plot(h3["t_h"], h3["obs"][key], color="#dc2626", lw=2.2, label="severe sepsis (sev 3)")
    if thr is not None:
        ax.axhline(thr, ls=":", color="#64748b", lw=1)
        ax.text(0.98, thr, " "+thrlab, transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#64748b")
    ax.set_title(lab, fontsize=11, fontweight="bold")
    ax.set_xlabel("hours"); ax.grid(alpha=0.25); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes.ravel()[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("The sepsis core — one spec, real output under the exact SBML engine",
             fontsize=13, fontweight="bold", y=1.0)
fig.tight_layout()
dd = Path(__file__).parent
fig.savefig(dd/"00_core_output.svg", bbox_inches="tight"); fig.savefig(dd/"00_core_output.png", bbox_inches="tight")
ti3 = min(h3["obs"]["tissue_integrity"]); map3 = min(h3["obs"]["MAP"])
cap = (f"Real output of the one sepsis spec under the exact Tellurium/roadrunner engine "
       f"(real_trajectories.json): four observables over 48 h at healthy baseline vs severe sepsis. "
       f"The severe run drops MAP to {map3:.0f} mmHg (below the 65 shock line) and tissue integrity to "
       f"{ti3:.2f} as blood pathogen climbs — the physiology the whole investigation studies.")
json.dump({"title": "The sepsis core — real output", "interpretation": cap, "caption": cap,
           "provenance": {"generator": "charts/make_core_output.py", "data": "data/real_trajectories.json"}},
          open(dd/"00_core_output.meta.json", "w"), indent=1)
print("wrote 00_core_output")
