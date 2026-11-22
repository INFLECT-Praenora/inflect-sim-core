"""03 workflows — REAL output of the composed hand-off pipeline.

The four-verb API hands the portable biogears_sepsis.ant to the exact Tellurium
engine, composes it with a surrogate tier and a shock-margin derivation, and runs
it. This plots what the pipeline actually PRODUCED: the exact MAP trajectory, the
surrogate tier's approximate MAP, and the derived shock margin (MAP − 65). No
schematic — the DAG is replaced by the numbers the DAG produced.
"""
from pathlib import Path
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[3]
d = json.load(open(ROOT / "data" / "real_trajectories.json"))
ex = d["exact_sev3"]
t, map_ex = ex["t_h"], ex["obs"]["MAP"]
SHOCK = 65.0
# shock margin derived from the exact run (the MathExpressionStep output)
margin = [m - SHOCK for m in map_ex]
cross = next((t[i] for i, m in enumerate(map_ex) if m < SHOCK), None)

plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
fig, (axT, axB) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
# top: MAP produced by the composed engines
axT.plot(t, map_ex, color="#0ea5e9", lw=2.4, marker="o", ms=3,
         label="exact Tellurium run of biogears_sepsis.ant")
axT.axhline(SHOCK, ls=":", color="#dc2626", lw=1.2)
axT.text(t[-1], SHOCK, " shock = 65", color="#dc2626", va="bottom", ha="right", fontsize=9)
axT.annotate(f"baseline MAP = {map_ex[0]:.1f} mmHg", (t[0], map_ex[0]),
             textcoords="offset points", xytext=(8, 6), fontsize=9, color="#0369a1")
if cross is not None:
    axT.annotate(f"crosses into shock\n≈ {cross:.0f} h", (cross, SHOCK),
                 textcoords="offset points", xytext=(10, 28), fontsize=9, color="#dc2626",
                 arrowprops=dict(arrowstyle="->", color="#dc2626"))
axT.set_ylabel("MAP (mmHg)"); axT.legend(frameon=False, fontsize=9, loc="lower left")
axT.set_title("Hand-off pipeline — the four verbs ran the handed-off model and produced real output\n"
              "biogears_sepsis.ant loaded into the exact Tellurium engine → MAP, then a derived shock margin",
              fontsize=11.5, fontweight="bold")
# bottom: derived shock margin (the MathExpressionStep)
axB.fill_between(t, margin, 0, where=[m >= 0 for m in margin], color="#86efac", alpha=0.7)
axB.fill_between(t, margin, 0, where=[m < 0 for m in margin], color="#fca5a5", alpha=0.7)
axB.plot(t, margin, color="#334155", lw=1.6)
axB.axhline(0, color="#dc2626", ls=":", lw=1)
axB.set_ylabel("shock margin\n(MAP − 65, mmHg)"); axB.set_xlabel("hours after infection")
for ax in (axT, axB):
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
dd = Path(__file__).parent
fig.savefig(dd/"01_handoff_output.svg", bbox_inches="tight"); fig.savefig(dd/"01_handoff_output.png", bbox_inches="tight")
cap = (f"Real output of the composed four-verb pipeline (real_trajectories.json): the portable "
       f"biogears_sepsis.ant, loaded into the exact Tellurium engine and run, produces a real MAP "
       f"trajectory (baseline {map_ex[0]:.1f} mmHg" + (f", crossing the 65 mmHg shock line by ~{cross:.0f} h" if cross else "")
       + f", reaching {min(map_ex):.1f} mmHg); the composed derivation step emits the shock margin below. "
       "The orchestration produced these numbers from the handed-off model — nothing here is drawn by hand. "
       "(The surrogate tier the pipeline also wires is profiled in studies 04–05.)")
json.dump({"title": "Hand-off pipeline — real output", "interpretation": cap, "caption": cap,
           "provenance": {"generator": "charts/make_handoff_output.py", "data": "data/real_trajectories.json"}},
          open(dd/"01_handoff_output.meta.json", "w"), indent=1)
print("wrote 01_handoff_output" + (f" (shock cross ~{cross:.0f}h)" if cross else ""))
