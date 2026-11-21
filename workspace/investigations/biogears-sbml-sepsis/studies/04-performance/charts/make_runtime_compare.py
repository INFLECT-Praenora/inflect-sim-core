"""Three-engine compute-time comparison for the sepsis core — same model spec run
by BioGears, Tellurium (exact SBML), and a PyTorch surrogate.

Numbers are REAL and cited, not re-measured here:
- BioGears 8.2.0 (C++) and Tellurium/roadrunner-CVODE wall-clock for one matched
  48 h severe-sepsis run come from study 04-engine-runtime (committed meta.wall_s).
- The PyTorch surrogate figure (one 576-step / 48 h autoregressive rollout) comes
  from the neural-surrogate-sepsis investigation (~0.2 s; ~40,000x faster than the
  exact SBML path on the same rollout).

Run: python make_runtime_compare.py   (writes 01_runtime_compare.{svg,png})
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLUE, GREEN, RED = "#4C72B0", "#55A868", "#C44E52"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.9))

# -- Left: full 48 h run (committed numbers) --------------------------------
engines = ["BioGears\n(C++ physiology)", "Tellurium\n(exact SBML)", "PyTorch surrogate\n(learned)"]
wall_s = [3.38 * 3600, 2.81 * 3600, 0.2]
where = ["whole-body physiology\n+ homeostatic control",
         "stiff 91-state ODE\nintegration (CVODE)",
         "one forward pass / step\n(no ODE solve)"]
bars = axL.bar(engines, wall_s, color=[BLUE, GREEN, RED], width=0.6, log=True)
axL.set_ylabel("wall-clock, one 48 h sepsis run  (s, log)")
axL.set_title("Full 48 h run, same spec, three engines\n(committed runs; study 04 + surrogate inv.)")
axL.set_ylim(0.05, 5e4)
for bar, s, w in zip(bars, wall_s, where):
    axL.text(bar.get_x() + bar.get_width() / 2, s * 1.4,
             f"{s/3600:.2f} h" if s >= 60 else f"{s:.2g} s",
             ha="center", va="bottom", fontweight="bold")
    axL.text(bar.get_x() + bar.get_width() / 2, 0.075, w, ha="center", va="bottom",
             fontsize=7.5, color="white")
axL.annotate("~40,000x faster\nthan exact SBML\n(approximate)", xy=(2, 0.2), xytext=(1.4, 300),
             fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
axL.grid(axis="y", alpha=0.3, which="both")

# -- Right: live per-step cost (measured this machine) ----------------------
b = json.loads((HERE / "bench_live.json").read_text())
ex, su = b["exact"]["per_step_ms"], b["surrogate"]["per_step_ms"]
pbars = axR.bar(["Tellurium exact\n(CVODE / step)", "PyTorch surrogate\n(forward pass / step)"],
                [ex, su], color=[GREEN, RED], width=0.55, log=True)
axR.set_ylabel("per-step wall-clock  (ms, log)")
axR.set_title(f"Live per-step cost, real sepsis model\n(measured; exact is {b['exact_vs_surrogate_per_step']:,.0f}x the surrogate)")
axR.set_ylim(0.01, 3e3)
for bar, v in zip(pbars, [ex, su]):
    axR.text(bar.get_x() + bar.get_width() / 2, v * 1.4,
             f"{v:.3g} ms", ha="center", va="bottom", fontweight="bold")
axR.grid(axis="y", alpha=0.3, which="both")

fig.suptitle("Same sepsis model spec, three interchangeable engines — where the compute time goes",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("svg", "png"):
    fig.savefig(HERE / f"01_runtime_compare.{ext}", dpi=140, bbox_inches="tight")
print("wrote 01_runtime_compare.svg / .png")
