#!/usr/bin/env python3
"""Regenerate the biogears-sbml-sepsis comparison figures from the committed run data.

Reproducible: reads the per-arm sample JSON under each study's out/ dir and the
compare-sevN.json divergence summaries, and writes SVG+PNG (+ .meta.json caption)
into the member studies' charts/ dirs. Run from the investigation dir:

    .venv/bin/python workspace/investigations/biogears-sbml-sepsis/make_figures.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
STUDIES = HERE / "studies"
ENGINE = "#b3462f"   # BioGears engine (brown-red, matches existing charts)
SBML = "#2f6fb3"     # SBML extraction (steel blue)
OMIT = "#c0392b"; SIMP = "#d4952a"; KEEP = "#2e8b57"  # matrix colors

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

OBS = ["SBP","DBP","MAP","HR","CO","SVR","blood_volume","tissue_integrity","blood_pathogen"]
LABELS = {"SBP":"Systolic BP (mmHg)","DBP":"Diastolic BP (mmHg)","MAP":"MAP (mmHg)",
          "HR":"Heart rate (/min)","CO":"Cardiac output (L/min)","SVR":"SVR (mmHg·s/mL)",
          "blood_volume":"Blood volume (mL)","tissue_integrity":"Tissue integrity",
          "blood_pathogen":"Blood pathogen"}

def load(study_rel, name):
    p = STUDIES / study_rel / "out" / name
    s = json.load(open(p))["samples"]
    t = np.array([x["time"] for x in s]) / 3600.0
    return t, {o: np.array([x[o] for x in s]) for o in OBS}

def save(fig, study_rel, base, caption):
    d = STUDIES / study_rel / "charts"; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{base}.svg", bbox_inches="tight")
    fig.savefig(d / f"{base}.png", bbox_inches="tight")
    json.dump({"title": base, "interpretation": caption, "caption": caption},
              open(d / f"{base}.meta.json", "w"), indent=1)
    plt.close(fig); print("wrote", d / base)

# ---------------------------------------------------------------- severe 9-panel
te,E = load("03-severe-infection","engine-sev3.json")
ts,S = load("03-severe-infection","sbml-sev3.json")
fig,axes = plt.subplots(3,3, figsize=(13,9.5), sharex=True)
thr = {"SBP":100,"MAP":65,"HR":90}
for ax,o in zip(axes.flat, OBS):
    ax.plot(te,E[o],color=ENGINE,lw=1.6,label="engine")
    ax.plot(ts,S[o],color=SBML,lw=1.6,label="SBML")
    if o in thr: ax.axhline(thr[o],ls=":",color="0.5",lw=1)
    ax.set_title(LABELS[o]); ax.margins(x=0)
for ax in axes[-1]: ax.set_xlabel("Hours after infection")
axes[0,0].legend(loc="upper right",fontsize=10)
fig.suptitle("Severe sepsis (severity 3): BioGears engine vs SBML extraction — all 9 observables",
             fontsize=14,fontweight="bold",y=1.005)
fig.tight_layout()
save(fig,"03-severe-infection","03_severe_9panel_engine_vs_sbml",
     "Severe arm, all 9 observables, engine vs SBML extraction over 48 h. Inflammation "
     "(tissue integrity, blood pathogen) overlaps bit-for-bit; every divergence is in the "
     "hemodynamic layer. Dotted lines mark the SBP<100 / MAP<65 / HR>90 thresholds.")

# -------------------------------------------------- severe mechanism (CO vs SVR)
fig,(a1,a2) = plt.subplots(2,1, figsize=(9,7), sharex=True)
a1.plot(te,E["CO"],color=ENGINE,lw=2,label="engine"); a1.plot(ts,S["CO"],color=SBML,lw=2,label="SBML")
a1.set_ylabel("Cardiac output (L/min)"); a1.set_title("Cardiac output — engine falls and stays low; extraction rebounds")
a1.legend(loc="lower left",fontsize=10); a1.margins(x=0)
a1.annotate("engine: output failure\n(5.8 → 4.5)",(44,4.5),color=ENGINE,fontsize=9,ha="right")
a1.annotate("extraction: hyperdynamic\nrebound (→ 6.0)",(44,5.9),color=SBML,fontsize=9,ha="right")
a2.plot(te,E["SVR"],color=ENGINE,lw=2); a2.plot(ts,S["SVR"],color=SBML,lw=2)
a2.axhline(1.0,ls=":",color="0.6",lw=1)
a2.set_ylabel("SVR (mmHg·s/mL)"); a2.set_xlabel("Hours after infection")
a2.set_title("Vascular tone — engine barely moves; extraction swings up then collapses")
a2.margins(x=0)
a2.annotate("extraction: vasoconstrict → fatigue\n(0.93 → 1.13 → 0.67)",(37,1.1),color=SBML,fontsize=9,ha="center")
a2.annotate("engine: SVR ~flat (0.93)",(30,0.90),color=ENGINE,fontsize=9,ha="center")
fig.suptitle("Same hypotension, opposite mechanism (MAP ≈ CO × SVR)",fontsize=14,fontweight="bold")
fig.tight_layout()
save(fig,"03-severe-infection","04_severe_mechanism_co_svr",
     "The two drive MAP down to ≈70 mmHg (still above the MAP<65 shock threshold at 48 h) by "
     "different routes: the engine declines via cardiac-output failure with near-constant SVR "
     "(hypodynamic/low-output), while the extraction holds output and swings SVR — vasoconstriction "
     "then sympathetic-fatigue vasodilation (hyperdynamic/warm). The extraction's single AIR→sympathetic "
     "arc overshoots the vasomotor response the engine's full cardiovascular control keeps muted.")

# --------------------------------------------------------- mild fluctuation
tem,Em = load("02-mild-infection","engine-sev1.json")
tsm,Sm = load("02-mild-infection","sbml-sev1.json")
fig,(a1,a2) = plt.subplots(2,1, figsize=(9,7), sharex=True)
a1.plot(tem,Em["SBP"],color=ENGINE,lw=1.3,label="engine"); a1.plot(tsm,Sm["SBP"],color=SBML,lw=2,label="SBML")
a1.set_ylabel("Systolic BP (mmHg)"); a1.legend(loc="upper right",fontsize=10); a1.margins(x=0)
a1.set_title("Systolic BP — engine fluctuates (range ~11 mmHg), extraction near-flat (~0.6 mmHg)")
a1.annotate("meal / nutrition-schedule\nand homeostatic fluctuation",(24,124),color=ENGINE,fontsize=9,ha="center")
a2.plot(tem,Em["blood_volume"],color=ENGINE,lw=1.3,label="engine"); a2.plot(tsm,Sm["blood_volume"],color=SBML,lw=2,label="SBML")
a2.set_ylabel("Blood volume (mL)"); a2.set_xlabel("Hours after infection"); a2.margins(x=0)
a2.set_title("Blood volume — the engine's fluid pools swing with meals; the extraction drifts smoothly")
fig.suptitle("Mild infection (severity 1): the engine lives, the extraction idles",fontsize=14,fontweight="bold")
fig.tight_layout()
save(fig,"02-mild-infection","02_mild_fluctuation",
     "Even in the agreement case, the engine carries continuous physiological variation — the "
     "InfectionModerate nutrition schedule plus full homeostatic control drive SBP over an ~11 mmHg "
     "range and swing the fluid pools — while the autonomous SBML extraction (a closed ODE with no "
     "meals, respiration, or temperature inputs) sits nearly flat. Same clinical outcome, no ambient dynamics.")

# ============================================= study 05: divergence-mechanism
S5 = "05-divergence-mechanism"
def perobs(study_rel, name):
    return json.load(open(STUDIES/study_rel/"out"/name))["aligned"]["per_observable"]
mild = perobs("02-mild-infection","compare-sev1.json")
sev  = perobs("03-severe-infection","compare-sev3.json")

# ---- heatmap: max relative divergence (%) per observable per condition
M = np.array([[mild[o]["max_rel"]*100, sev[o]["max_rel"]*100] for o in OBS])
fig,ax = plt.subplots(figsize=(6.2,7))
im = ax.imshow(M, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=max(20,np.nanmax(M)))
ax.set_xticks([0,1]); ax.set_xticklabels(["Mild (sev 1)","Severe (sev 3)"])
ax.set_yticks(range(len(OBS))); ax.set_yticklabels([LABELS[o] for o in OBS])
for i in range(len(OBS)):
    for j in range(2):
        v=M[i,j]; ax.text(j,i,f"{v:.1f}%",ha="center",va="center",
                           color="white" if v>20 else "black",fontsize=9)
ax.set_title("Peak relative divergence by observable",fontweight="bold")
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label("max |engine−SBML| / |engine|  (%)")
fig.tight_layout()
save(fig,S5,"01_divergence_heatmap",
     "Peak relative divergence between engine and extraction, per observable and condition. "
     "The inflammation states (tissue integrity, blood pathogen) agree to well under 0.1% in both "
     "arms; the hemodynamic states carry all the divergence — small but nonzero at mild (the "
     "engine's fluctuation) and large at severe (the different decompensation mechanism and ~11 h lag).")

# ---- milestone dumbbell (severe timing lag)
ms = json.load(open(STUDIES/"03-severe-infection"/"out"/"compare-sev3.json"))["milestones"]
rows = [("SBP < 100 mmHg","sbp_below_100_h"),("HR > 90 /min","hr_above_90_h")]
fig,ax = plt.subplots(figsize=(8.5,3.4))
for i,(lab,key) in enumerate(rows):
    e=ms[key]["engine"]; s=ms[key]["sbml"]
    ax.plot([e,s],[i,i],color="0.6",lw=3,zorder=1)
    ax.scatter([e],[i],color=ENGINE,s=120,zorder=2,label="engine" if i==0 else None)
    ax.scatter([s],[i],color=SBML,s=120,zorder=2,label="SBML" if i==0 else None)
    ax.text((e+s)/2,i+0.18,f"+{s-e:.1f} h lag",ha="center",fontsize=9,color="0.3")
    ax.text(e,i-0.22,f"{e:.1f} h",ha="center",fontsize=8,color=ENGINE)
    ax.text(s,i-0.22,f"{s:.1f} h",ha="center",fontsize=8,color=SBML)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.set_ylim(-0.6,len(rows)-0.4); ax.set_xlabel("Hours after infection (severe arm)")
ax.set_title("Hemodynamic milestones arrive ~11 h late in the extraction",fontweight="bold")
ax.legend(loc="lower right",fontsize=10); ax.margins(x=0.08)
ax.text(0.01,0.02,"Mild arm: neither milestone reached on either side (infection controlled).",
        transform=ax.transAxes,fontsize=8,color="0.4")
fig.tight_layout()
save(fig,S5,"02_milestone_dumbbell",
     "Severe-arm crossing times, engine vs extraction: SBP<100 at 26.6 h (engine) vs 37.5 h "
     "(extraction) and HR>90 at 26.8 h vs 37.6 h — a consistent ~11 h lag, measured run-for-run "
     "against the real engine rather than against the McDaniel 2019 paper milestones.")

# ---- functional-difference matrix (the heart of the report)
feat = [
 # (BioGears subsystem / functionality, status, affected observables, what we see)
 ("Acute Inflammatory Response (21-state AIR)", KEEP, "pathogen, tissue integrity", "identical (<0.1%)"),
 ("Cardiovascular circuit + beat-resolved heart", SIMP, "SBP, DBP, MAP, CO", "baseline matches; dynamics differ"),
 ("Baroreflex (afferent/efferent, sympathetic)", SIMP, "HR, SVR", "set-point adaptation omitted"),
 ("Capillary fluid / albumin exchange", SIMP, "blood volume", "single plasma pool; ~matches"),
 ("Nitric-oxide vasodilation", OMIT, "SVR, MAP", "inert in code → muted vasodilation"),
 ("Fever-driven skin vasodilation (temperature)", OMIT, "SVR, MAP", "absent → SVR mismatch"),
 ("Renal module / urine output", OMIT, "blood volume, MAP", "no urine → fluid handling differs"),
 ("Lactate / metabolic acidosis", OMIT, "SVR, HR", "absent → vasomotor drive differs"),
 ("Intracellular-water shifts", OMIT, "blood volume", "absent → volume distribution differs"),
 ("Respiration / gas exchange / chemoreceptors", OMIT, "HR, SVR", "held at baseline"),
 ("Nutrition schedule / meals / exertion", OMIT, "SBP, CO, blood volume", "no inputs → extraction is flat"),
]
fig,ax = plt.subplots(figsize=(12.5,6.2)); ax.axis("off")
ax.set_title("What BioGears does that the SBML extraction does not — and where it shows",
             fontweight="bold",fontsize=13,loc="left")
col_x=[0.0,0.46,0.60,0.80]; hdr=["BioGears functionality","Status","Observables affected","Run-for-run result"]
for x,h in zip(col_x,hdr): ax.text(x,0.96,h,fontweight="bold",fontsize=10,transform=ax.transAxes)
n=len(feat)
for i,(name,color,obs,res) in enumerate(feat):
    y=0.90-(i+1)*(0.86/n)
    ax.add_patch(plt.Rectangle((col_x[1],y-0.012),0.11,0.028,color=color,alpha=0.85,transform=ax.transAxes))
    status={KEEP:"translated",SIMP:"simplified",OMIT:"omitted/inert"}[color]
    ax.text(col_x[1]+0.055,y,status,ha="center",va="center",fontsize=8.5,color="white",fontweight="bold",transform=ax.transAxes)
    ax.text(col_x[0],y,name,va="center",fontsize=9.3,transform=ax.transAxes)
    ax.text(col_x[2],y,obs,va="center",fontsize=8.8,color="0.25",transform=ax.transAxes)
    ax.text(col_x[3],y,res,va="center",fontsize=8.8,color="0.25",transform=ax.transAxes)
leg=[Patch(color=KEEP,label="translated faithfully"),Patch(color=SIMP,label="simplified"),
     Patch(color=OMIT,label="omitted or inert")]
ax.legend(handles=leg,loc="lower left",ncol=3,fontsize=9,bbox_to_anchor=(0,-0.04))
fig.tight_layout()
save(fig,S5,"03_functional_difference_matrix",
     "Map of BioGears functionality to its fate in the extraction and to the run-for-run result. "
     "The inflammation core is translated faithfully (identical output); the cardiovascular control "
     "and fluid systems are simplified; and a cluster of vasomotor/fluid/metabolic subsystems "
     "(nitric oxide, fever vasodilation, renal/urine, lactate, intracellular water, respiration, "
     "nutrition) is omitted or inert — exactly the pathways that would couple into SVR, CO and "
     "volume, which is where every divergence lands.")

# ================================================= study 04: runtime bars (S4)
horizons=["600 s sim","3600 s sim"]; pb=[25.3,139.9]; vc=[25.9,125.8]
x=np.arange(len(horizons)); w=0.35
fig,ax=plt.subplots(figsize=(7,4.5))
b1=ax.bar(x-w/2,pb,w,label="process-bigraph (Python)",color="#6a51a3")
b2=ax.bar(x+w/2,vc,w,label="viva-core (Rust)",color="#2a924a")
ax.set_xticks(x); ax.set_xticklabels(horizons); ax.set_ylabel("Wall-clock (s)")
ax.set_title("S4 cross-engine runtime — framework overhead, not a solver speedup",fontweight="bold")
ax.legend(fontsize=10)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f"{b.get_height():.0f}s",ha="center",fontsize=9)
ax.text(0.5,0.92,"Results bit-identical (max abs diff 0.0 across all 9 observables)",
        transform=ax.transAxes,ha="center",fontsize=9,color="0.3")
fig.tight_layout()
save(fig,"04-engine-runtime","01_runtime_bars",
     "Wall-clock for the same SBML composite under the Python process-bigraph engine vs the Rust "
     "viva-core engine, at two simulated horizons. A single heavy libroadrunner process dominates, "
     "so the engine swap moves only framework overhead (within ~10%); the two engines produce "
     "bit-identical results (max abs diff 0.0 across all 9 observables).")

print("\nall figures written.")
