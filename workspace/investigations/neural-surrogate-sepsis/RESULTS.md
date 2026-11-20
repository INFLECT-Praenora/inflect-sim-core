# Neural surrogate for the sepsis SBML composite — results (partial)

**Status:** pipeline complete; stabilization solved; accuracy plateaus (data-limited).
Held-out eval of the *stable* model and a denser sweep are pending dedicated compute.

## Question

Can a neural surrogate learn the 98-dimensional sepsis SBML dynamics
(Tellurium/roadrunner) well enough to replace the stiff integrator for fast
autonomous 48 h rollouts, across infection severities and infection timings it
was not trained on?

## Pipeline (real, reproducible)

| Script | Role |
|---|---|
| `sample_dataset.py` | Parallel multi-process sweep. Severities {0,1,2,3} × infection starts {0, 3, 6 h}; 48 h / 300 s = 576 macro-steps per trajectory. Held-out infection start = 3 h across infected severities → **7 train ICs + 3 held-out ICs, 5760 transitions** (`out/dataset.npz`). |
| `train_and_eval.py` | Baseline **one-step (teacher-forced)** delta-MLP surrogate (`viva_torch` `SurrogateNet`) + held-out eval that rolls out the real SBML vs the surrogate. |
| `train_stable.py` | **Unrolled (K-step) training** for rollout stability — curriculum K = 16→64→192→576, normalized-space clamp + out-of-distribution barrier + truncated BPTT. Same architecture, so eval loads it unchanged. |
| `eval_parallel.py` | Checkpoint-reusing **parallel held-out eval** — one real-SBML rollout subprocess per held-out IC (~2 h vs ~7 h serial). |

Checkpoints: `out/surrogate.pt` (baseline), `out/surrogate_stable.pt` (128×128),
`out/surrogate_stable2.pt` (256×256, best stable).

## Findings

### 1. Baseline one-step surrogate is fast but diverges — honest negative

Trained teacher-forced, the model is accurate one-step but, fed its own
predictions autonomously for 576 steps, compounding error explodes on the stiff
positive-feedback dynamics.

- **Held-out rollout NRMSE ≈ 7.5 × 10³¹** (mean over the 3 held-out infection
  timings); ~2.8 % of steps go non-finite.
- **Speed: real SBML 8064 s vs surrogate 0.2 s** for a 576-step rollout →
  **40,463× faster**.
- Figure: `studies/01-sample-and-train/charts/01_surrogate_vs_real_heldout.*`
  (super-exponential blow-up after ~40 h).

### 2. Unrolled training solves stability — `surrogate_stable2.pt`

K-step unrolled training makes the model contractive rather than merely
one-step accurate. The surrogate-only 576-step rollout stays **bounded**
(max|state| ~10⁵–10⁶ vs data max 2.5 × 10⁷, **zero non-finite**) across all ICs.

Accuracy (training-IC free-rollout probe, mean over 10 ICs):

| Observable | Stable NRMSE | Baseline NRMSE |
|---|---|---|
| SBP | 11.5 | ~5.9 × 10²⁸ |
| MAP | 17.6 | ~1.6 × 10²⁹ |
| HR | 14.8 | ~7.3 × 10²⁸ |
| SVR | 7.7 | ~2.3 × 10²⁹ |
| tissue_integrity | 4.2 | ~1.1 × 10²⁸ |
| blood_pathogen | 2.9 | ~2.7 × 10²⁹ |

Stable and physiologically bounded, but **not yet clinically accurate** (< 1
needed) and still ~10¹² on the full 98-observable set — a few internal columns
drift.

### 3. Accuracy is data-limited

One-step loss floors ~0.09 while the full-horizon (K = 576) rollout loss floors
~0.95. Diagnosed as too few training trajectories (7 ICs) for 98-dimensional
stiff dynamics.

## Pending / not done

- **Held-out real-SBML eval of the *stable* model.** Only the baseline was run
  through `eval_parallel.py`; the stable-model accuracy numbers above are
  training-IC probes, not held-out real-SBML rollouts.
- **Denser sweep to break the plateau.** Attempted (severities × 11 infection
  starts ≈ 4× the training ICs) but could not complete: the shared Mac mini was
  saturated by another session's v2ecoli ensemble, starving every roadrunner
  worker past the 3 h per-combo timeout. Needs dedicated compute.

## Verdict

The pipeline is real and the **stabilization result is solid** — 40,000× faster
than the integrator, with bounded full-horizon rollouts where the baseline
diverges. The surrogate is **not yet accurate enough to replace the integrator**;
closing that gap needs a denser dataset and a held-out eval of the stabilized
model, both gated on dedicated compute.
