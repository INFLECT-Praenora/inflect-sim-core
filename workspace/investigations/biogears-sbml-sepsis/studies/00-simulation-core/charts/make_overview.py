"""Overview figure for study 00 — the simulation core at a glance.

One sepsis spec → three interchangeable engines (swap one node) → four runnable
pillars. A schematic, not a data plot: it states the scope the rest of the
investigation demonstrates. Regenerate: python charts/make_overview.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 11, "svg.fonttype": "none",
                     "font.family": "DejaVu Sans", "figure.dpi": 140, "savefig.dpi": 140})

TEAL, SLATE, INK = "#14b8a6", "#64748b", "#0f172a"
ENG = "#0ea5e9"; PILL = "#8b5cf6"

fig, ax = plt.subplots(figsize=(11, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec, tc="white", fs=11, weight="bold"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.6,rounding_size=2.2",
                 linewidth=1.6, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight=weight, zorder=3, linespacing=1.3)


def arrow(x0, y0, x1, y1, color=SLATE):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=16, linewidth=1.5, color=color, zorder=1,
                 shrinkA=2, shrinkB=2))


# Tier 1 — one spec
box(50, 90, 52, 11, "BioGears sepsis core  —  ONE specification\n(SBML / Antimony, nine shared observables)",
    INK, INK, fs=12)

# Tier 2 — three interchangeable engines
engines = [
    (20, "BioGears 8.2.0\nC++ engine", "full physics"),
    (50, "Exact SBML\nTellurium / CVODE", "same equations"),
    (80, "PyTorch\nsurrogate", "learned, fast"),
]
for x, name, sub in engines:
    arrow(50, 84, x, 66)
    box(x, 59, 24, 15, name, ENG, "#0369a1", fs=11)
    ax.text(x, 50.5, sub, ha="center", va="center", color=SLATE, fontsize=9, style="italic")
ax.text(50, 71.5, "same typed ports → swap one node", ha="center", color=TEAL,
        fontsize=10.5, fontweight="bold")

# Tier 3 — four runnable pillars
pillars = [
    (14, "Composition", "01 · three engines,\none node swapped"),
    (38, "Ensembles", "02 · severity × engine\nbatch, one sweep"),
    (62, "Orchestration", "03 · a model file handed\nbetween engines"),
    (86, "Performance", "04 · compute profile,\nseconds → hours"),
]
for x, name, sub in pillars:
    arrow(50, 44, x, 27)
    box(x, 19, 21, 15, name + "\n", PILL, "#6d28d9", fs=11)
    ax.text(x, 17.5, sub, ha="center", va="center", color="white", fontsize=8.3, linespacing=1.25)
ax.text(50, 32, "four pillars — each a study you can open and run", ha="center",
        color=SLATE, fontsize=10.5, fontweight="bold")

fig.suptitle("The simulation core: one model, every engine, four capabilities",
             fontsize=15, fontweight="bold", y=0.99, color=INK)

d = Path(__file__).resolve().parent
fig.savefig(d / "00_overview.svg", bbox_inches="tight")
fig.savefig(d / "00_overview.png", bbox_inches="tight")
cap = ("One BioGears sepsis specification runs under three interchangeable engines "
       "(a C++ physiology engine, an exact SBML extraction, and a learned surrogate) "
       "by swapping a single node over shared typed ports — and that one model drives "
       "four runnable capabilities: composition, ensembles, orchestration, and performance.")
json.dump({"title": "The simulation core at a glance", "interpretation": cap, "caption": cap,
           "provenance": {"generator": "charts/make_overview.py", "kind": "schematic"}},
          open(d / "00_overview.meta.json", "w"), indent=1)
print("wrote 00_overview.svg/.png/.meta.json")
