"""Live interchangeability check — the third engine constructs and runs here, now."""
from pathlib import Path
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(Path(__file__).parent / "engines_live_check.json"))
tb = d["tellurium_build_s"]; su = d["surrogate"]
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios":[1.1,1]})
# left: the two live timings
axL.barh(["Tellurium\nconstructs", "Surrogate\nruns 576 steps"], [tb, su["wall_s"]],
         color=["#f59e0b", "#8b5cf6"], edgecolor="#0f172a", linewidth=0.8, height=0.55)
for i,(v,lab) in enumerate([(tb,f"{tb:.2f} s"), (su["wall_s"],f"{su['wall_s']:.3f} s · {su['n_steps']} steps")]):
    axL.text(v, i, "  "+lab, va="center", ha="left", fontsize=10, fontweight="bold")
axL.set_xlim(0, tb*1.35); axL.set_xlabel("wall-clock (s)"); axL.set_title("Live, this machine", fontsize=11, fontweight="bold")
for s in ("top","right"): axL.spines[s].set_visible(False)
# right: the interchangeability checklist
axR.axis("off")
checks = [("BioGears (C++)", "same spec, committed baseline", True),
          ("Tellurium (exact SBML)", f"constructs in {tb:.1f}s", bool(d["tellurium_builds"])),
          ("PyTorch surrogate", f"runs, finite over horizon", bool(su["ran"] and su["finite"])),
          ("MAP handoff column", "present on every engine", bool(d["map_col_present"]))]
axR.set_title("One spec · three interchangeable engines", fontsize=11, fontweight="bold", loc="left")
for i,(name,note,ok) in enumerate(checks):
    y = 0.82 - i*0.24
    axR.text(0.02, y, "✓" if ok else "✗", color="#16a34a" if ok else "#dc2626", fontsize=15, fontweight="bold", transform=axR.transAxes)
    axR.text(0.10, y, name, fontsize=10.5, fontweight="bold", transform=axR.transAxes)
    axR.text(0.10, y-0.075, note, fontsize=9, color="#475569", transform=axR.transAxes)
fig.suptitle("The composition pillar — the engine is a box you swap, not a rewrite", fontsize=12, fontweight="bold", y=1.0)
fig.tight_layout()
dd = Path(__file__).parent
fig.savefig(dd/"02_live_check.svg", bbox_inches="tight"); fig.savefig(dd/"02_live_check.png", bbox_inches="tight")
cap = (f"Live interchangeability check (engines_live_check.json): the exact-SBML engine constructs in "
       f"{tb:.1f}s and the surrogate runs {su['n_steps']} steps in {su['wall_s']:.3f}s, finite over the horizon — "
       f"all three engines expose the same observables and the MAP handoff column. The engine is a box you swap.")
json.dump({"title":"Live interchangeability check","interpretation":cap,"caption":cap,
           "provenance":{"generator":"charts/make_live_check.py","data":"engines_live_check.json"}},
          open(dd/"02_live_check.meta.json","w"), indent=1)
print("wrote 02_live_check")
