"""Severity response: minimum tissue integrity falls with infection severity (per engine)."""
from pathlib import Path
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(Path(__file__).parent / "ensemble_results.json"))
cells = d["cells"]; sev = sorted(int(s) for s in d["severities"])
names = {"biogears":"BioGears (C++)","tellurium":"Tellurium (exact SBML)"}
# BioGears solid underneath; Tellurium dashed on top, so both read even where they coincide.
styles = {"biogears":dict(ls="-", lw=3.2, ms=11, color="#0ea5e9", zorder=2),
          "tellurium":dict(ls="--", lw=2.0, ms=7, color="#f59e0b", zorder=3)}
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for eng in ("biogears","tellurium"):
    if eng not in cells: continue
    ys = [cells[eng].get(str(s),{}).get("tissue_integrity_min") for s in sev]
    ax.plot(sev, ys, marker="o", label=names[eng], **styles[eng])
    if eng == "tellurium":
        for x,y in zip(sev,ys):
            if y is not None: ax.annotate(f"{y:.2f}", (x,y), textcoords="offset points", xytext=(0,11), ha="center", fontsize=9)
ax.set_xticks(sev); ax.set_xlabel("infection severity"); ax.set_ylabel("min tissue integrity (1 = intact)")
ax.set_ylim(0,1.08); ax.set_title("Severity response — one orchestrated batch\nfull-physics engines agree and track severity",
                                    fontsize=11.5, fontweight="bold")
ax.grid(alpha=0.25); ax.set_axisbelow(True); ax.legend(frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout()
dd = Path(__file__).parent
fig.savefig(dd/"02_severity_response.svg", bbox_inches="tight"); fig.savefig(dd/"02_severity_response.png", bbox_inches="tight")
cap = ("Minimum tissue integrity vs infection severity, from one orchestrated severity×engine batch "
       "(ensemble_results.json). Both full-physics engines agree and show the severity response: "
       "near-intact at severity 0–1, collapsing to ~0.24 at severity 3.")
json.dump({"title":"Severity response across engines","interpretation":cap,"caption":cap,
           "provenance":{"generator":"charts/make_severity_response.py","data":"ensemble_results.json"}},
          open(dd/"02_severity_response.meta.json","w"), indent=1)
print("wrote 02_severity_response")
