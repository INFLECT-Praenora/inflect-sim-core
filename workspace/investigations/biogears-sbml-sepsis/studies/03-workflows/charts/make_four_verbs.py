"""The four-verb workflow API — list, call, connect, run — on one handed-off model."""
from pathlib import Path
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
d = json.load(open(Path(__file__).parent.parent / "run" / "handoff_result.json"))["verbs"]
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3)
verbs = [("list", f"{d['list']['n_modules']} modules\n{len(d['list']['engine_boxes'])} engine boxes", "#0ea5e9"),
         ("call", f"{d['call']['module']}\nloaded in {d['call']['wall_s']:.2f}s", "#14b8a6"),
         ("connect", f"{len(d['connect']['nodes'])} nodes · {len(d['connect']['edges'])} edge\nMAP → shock.map", "#f59e0b"),
         ("run", f"pipeline runs\n{d['run']['wall_s']:.1f}s wall", "#8b5cf6")]
w=2.0; gap=0.5; x=0.25
for i,(verb,note,c) in enumerate(verbs):
    box = FancyBboxPatch((x,0.6), w, 1.8, boxstyle="round,pad=0.04,rounding_size=0.12",
                         fc=c, ec="#0f172a", lw=1.0, alpha=0.92); ax.add_patch(box)
    ax.text(x+w/2, 2.05, verb+"()", ha="center", va="center", fontsize=13, fontweight="bold", color="white", family="monospace")
    ax.text(x+w/2, 1.25, note, ha="center", va="center", fontsize=9.5, color="white")
    if i < 3:
        ax.add_patch(FancyArrowPatch((x+w,1.5),(x+w+gap,1.5), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#334155"))
    x += w+gap
ax.set_title("The orchestration pillar — four verbs compose and run a handed-off model",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
dd = Path(__file__).parent
fig.savefig(dd/"02_four_verbs.svg", bbox_inches="tight"); fig.savefig(dd/"02_four_verbs.png", bbox_inches="tight")
cap = (f"The four-verb workflow API on one handed-off model (handoff_result.json): list surfaces "
       f"{d['list']['n_modules']} modules, call loads {d['call']['module']} in {d['call']['wall_s']:.2f}s, "
       f"connect wires the surrogate's MAP into the shock node, and run executes the pipeline in {d['run']['wall_s']:.1f}s.")
json.dump({"title":"Four-verb workflow","interpretation":cap,"caption":cap,
           "provenance":{"generator":"run/make_four_verbs.py","data":"handoff_result.json"}},
          open(dd/"02_four_verbs.meta.json","w"), indent=1)
print("wrote 02_four_verbs")
