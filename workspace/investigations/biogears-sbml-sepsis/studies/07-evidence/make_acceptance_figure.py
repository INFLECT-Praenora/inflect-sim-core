#!/usr/bin/env python3
"""Study 07 figure: the acceptance gates as a pass / fail / pending status panel.

Reads out/acceptance_eval.json and renders charts/01_acceptance_gates.svg|png
(+ .meta.json). Run after evaluate_acceptance.py, from the investigation dir:
    PYTHONPATH=<worktree> python studies/07-evidence/make_acceptance_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLOR = {"pass": "#2e8b57", "fail": "#c0392b", "pending": "#d4952a"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def measured(c):
    if "rel_error" in c:
        return f"rel err {c['rel_error']*100:.3g}%  (tol {c['tolerance_rel']*100:.0f}%)"
    if "abs_diff_h" in c:
        return f"Δ {c['abs_diff_h']:.1f} h  (tol {c['tolerance_hours']:.0f} h)"
    return f"not evaluated  (tol {c.get('tolerance_rel',0)*100:.0f}%)"


def main() -> None:
    data = json.loads((HERE / "out" / "acceptance_eval.json").read_text())
    crit = data["criteria"]
    n = len(crit)
    fig, ax = plt.subplots(figsize=(8.4, 0.9 + 0.8 * n))
    ax.axis("off")
    ax.set_title("Acceptance gates — evidence turned into pass / fail / pending verdicts",
                 fontsize=12, fontweight="bold", loc="left")

    for i, c in enumerate(crit):
        y = n - 1 - i
        col = COLOR.get(c["verdict"], "#888")
        ax.add_patch(plt.Rectangle((0.0, y + 0.08), 1.30, 0.84, color=col, alpha=0.14, lw=0))
        ax.add_patch(plt.Rectangle((0.0, y + 0.08), 0.045, 0.84, color=col, lw=0))
        ax.text(0.09, y + 0.60, f"{c['id']}  ·  {c['observable']}", fontsize=10.5, fontweight="bold")
        ax.text(0.09, y + 0.26, c["statement"], fontsize=8.6, color="#333", wrap=True)
        ax.text(1.95, y + 0.52, c["verdict"].upper(), fontsize=11, fontweight="bold",
                color=col, ha="right")
        ax.text(1.95, y + 0.22, measured(c), fontsize=8.6, color="#555", ha="right")

    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, n)
    counts = data["counts"]
    fig.text(0.012, 0.012,
             f"{counts['pass']} pass · {counts['fail']} fail · {counts['pending']} pending   "
             "— each linked to its evidence artifact (study 05 milestones)",
             fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = HERE / "charts" / "01_acceptance_gates"
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"))
    caption = (
        "Pre-declared acceptance criteria evaluated against committed study-05 evidence. "
        "INFLAM-01/02 pass: the BioGears engine and the SBML extraction agree on minimum "
        "tissue integrity (rel err 0.0044%) and peak blood pathogen (0.12%), within a 1% "
        "tolerance. HEMO-01 fails: time to septic shock (MAP < 65 mmHg) differs by 15.3 h "
        "(engine 69.3 h vs SBML 54.1 h), outside the 6 h tolerance — consistent with the "
        "known hemodynamic divergence. SURR-01 is pending: the surrogate clinical-accuracy "
        "validator is declared but not yet run. Each verdict links to its evidence artifact; "
        "the pending gate is reported honestly rather than silently passed.")
    out.with_name(out.name + ".meta.json").write_text(json.dumps({"caption": caption}, indent=1))
    print("wrote", out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
