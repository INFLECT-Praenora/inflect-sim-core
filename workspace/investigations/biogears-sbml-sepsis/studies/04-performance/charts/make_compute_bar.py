"""Per-step compute: the stiff exact solve vs the surrogate forward pass (log scale)."""
from pathlib import Path
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(Path(__file__).parent / "bench_live.json"))
ex, su = d["exact"]["per_step_ms"], d["surrogate"]["per_step_ms"]
ratio = d["exact_vs_surrogate_per_step"]
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar(["Exact SBML\n(CVODE, 91 states)", "PyTorch\nsurrogate"], [ex, su],
              color=["#0ea5e9", "#8b5cf6"], width=0.6, edgecolor="#0f172a", linewidth=0.8)
ax.set_yscale("log"); ax.set_ylabel("wall-clock per step (ms, log scale)")
ax.set_title(f"Per-step compute — the stiff ODE solve is the cost\nexact ≈ {ex:.0f} ms/step · surrogate ≈ {su:.3f} ms/step · {ratio:,.0f}× faster",
             fontsize=11.5, fontweight="bold")
for b, v in zip(bars, [ex, su]):
    ax.text(b.get_x()+b.get_width()/2, v*1.25, f"{v:.3g} ms", ha="center", fontweight="bold", fontsize=10)
ax.grid(axis="y", alpha=0.25, which="both"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout()
dd = Path(__file__).parent
fig.savefig(dd/"02_compute_per_step.svg", bbox_inches="tight"); fig.savefig(dd/"02_compute_per_step.png", bbox_inches="tight")
cap = (f"Wall-clock per step, measured live (bench_live.json, warm): one exact CVODE step of the "
       f"real 91-state sepsis model is ~{ex:.0f} ms, versus ~{su:.3f} ms for a surrogate forward pass "
       f"— {ratio:,.0f}× faster. This is where the compute goes: the stiff ODE solve.")
json.dump({"title":"Per-step compute — exact vs surrogate","interpretation":cap,"caption":cap,
           "provenance":{"generator":"charts/make_compute_bar.py","data":"bench_live.json"}},
          open(dd/"02_compute_per_step.meta.json","w"), indent=1)
print("wrote 02_compute_per_step")
