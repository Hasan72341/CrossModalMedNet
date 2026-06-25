# AI Fluency — How AI Was Used to Build CrossModalMedNet

This document is an honest account of how I used AI coding assistants (primarily Claude /
LLM-based tools) while building this CT→MRI translation framework. The goal is not to claim the
project is "AI-generated" — every architectural decision, every loss-function choice, and every
reported metric was reviewed, run, and validated by me. AI was a **force multiplier on a human-owned
engineering process**, and this page documents exactly where and how.

It is organised around four competencies: **driving productivity**, **prompt engineering**,
**critical assessment**, and **workflow integration**.

---

## 1. Driving Productivity — where AI measurably enhanced the workflow

AI was used for the *high-volume, low-ambiguity* work so I could spend my time on the parts that
need judgement (architecture, normalization strategy, evaluation methodology).

Concrete instances in this repo:

- **Boilerplate across 4 model families.** CycleGAN, Pix2Pix, paired diffusion and unpaired
  diffusion share a lot of scaffolding (config dataclasses, `create_dataloaders`, checkpoint
  save/load, TensorBoard/W&B logging, CLI arg parsing). I generated the *first draft* of each
  scaffold from the working version of the previous model, then adapted it. This turned ~4× the
  repetitive typing into ~4× review passes.
- **The preprocessing pipeline** (`src/data/preprocess.py`). The 9-step SimpleITK flow
  (reorient → resample → crop-to-mask → HU clip → percentile/z-score normalise → manifest) is
  standard medical-imaging plumbing. AI produced a structured skeleton; I supplied the
  domain-specific constants (HU window `[-1000, 1000]`, brain `1×1×1 mm` vs pelvis `1×1×2.5 mm`
  spacing, foreground percentile clipping for MRI) and the patient-wise split logic.
- **Documentation generation.** The per-model reports under `docs/` and their LaTeX/PDF build
  scripts (`compile_reports.py`, `improve_latex.py`, …) were drafted with AI, then fact-checked
  against the actual code.
- **Repository hygiene.** `pyproject.toml`, `Makefile` targets, issue/PR templates, and
  `.env.example` were scaffolded quickly so the project reads as engineering-grade from day one.

Net effect: the bottleneck moved from "writing plumbing" to "deciding what is correct" — which is
where it should be.

---

## 2. Prompt Engineering — structured context for high-quality results

The quality of AI output is dominated by the quality of the context it is given. My prompting
followed a few rules that consistently produced usable code:

- **Ground every request in the repo's invariants.** Instead of "write a metrics module", the
  effective prompt was: *"Images are normalized to `[-1, 1]`; foreground masks are available;
  metrics must be masked (foreground-only) because background dominates the frame. Compute
  PSNR/SSIM with the correct `data_range`, and LPIPS-VGG which needs 3-channel input."* The
  constraints (`[-1, 1]`, `data_range=2.0`, masking, channel count) are exactly the details that
  separate a correct medical-imaging metric from a plausible-looking wrong one.
- **Provide the contract, not just the task.** For each model I specified input/output tensor
  shapes (`(B, 1, 256, 256)`), the activation (`Tanh` → `[-1, 1]`), and the loss terms with their
  weights (e.g. CycleGAN: cycle + identity + WGAN-GP + LPIPS). This let the model fill in a body
  that already matched the rest of the system.
- **Few-shot from my own code.** The strongest results came from showing a *working* sibling file
  (e.g. the Pix2Pix trainer) and asking for the diffusion variant "in the same structure, same
  logging, same config style." Consistency across the 4 model families came largely from this.
- **Ask for the reasoning, then the code.** For non-trivial decisions (paired vs unpaired data
  handling, 2.5D slice stacking, LoRA rank for the diffusion UNet) I had the model explain the
  trade-offs *first*, which surfaced assumptions I could accept or reject before any code existed.

---

## 3. Critical Assessment — verifying and polishing AI output to production standards

This is the most important section: **AI output is a hypothesis, not an answer.** Everything was
put through a verification gate before it was trusted. Real examples from this codebase:

- **Metric correctness is checked by hand, not assumed.** AI-suggested metric code is exactly
  where subtle bugs hide. During review I caught and fixed:
  - **LPIPS on single-channel input** — `lpips.LPIPS(net='vgg')` requires 3-channel RGB; the
    grayscale tensors had to be repeated to 3 channels or the call crashes. (`src/utils/metrics.py`)
  - **Masked PSNR inflation** — computing PSNR on mask-multiplied images lets the (large) zeroed
    background drive the score up. Fixed to compute PSNR from the *foreground-only* MSE.
  - **`data_range`** is pinned to `2.0` for `[-1, 1]` tensors, with a comment, because a silent
    `data_range` mismatch produces metrics that look fine but are wrong.
- **Reproducibility is enforced.** AI's first dataset draft sampled *random* slices for every
  split. I changed validation/test to a deterministic centre slice so reported numbers are
  comparable across runs (`src/data/dataset.py`). Determinism is a correctness property, not a
  nicety.
- **Train/eval state discipline.** Evaluation helpers called `model.eval()` but never restored
  `model.train()`, which silently disables dropout for the rest of training. Caught in review and
  fixed with save/restore (`models/cyclegan/evaluation.py`).
- **Logic that only breaks off the happy path.** The CycleGAN trainer logged a *stale* `d_loss`
  whenever `d_update_every > 1` (the discriminator isn't stepped every iteration). Guarded so only
  real discriminator steps are logged (`models/cyclegan/train.py`).
- **Rejecting confident-but-wrong suggestions.** Not every AI proposal was accepted. For example,
  a suggested "fix" to the CT normalization formula was *incorrect* — the existing
  `2·(x − min)/(max − min) − 1` correctly maps `[-1000, 1000] HU → [-1, 1]`; "fixing" it would have
  broken every downstream loss and metric. The bug report was discarded after I re-derived the math.

The verification toolkit: re-deriving formulas, running `make syntax` / smoke evals, checking tensor
shapes and value ranges at boundaries, and reading the *actual* library API rather than trusting the
generated call. See `docs/KNOWN_ISSUES.md` for the running list of what was found and fixed.

---

## 4. Workflow Integration — where AI sits in the problem-solving toolkit

AI is one tool among several, used at specific stages and deliberately *not* used at others:

| Stage | AI role | Human role |
|-------|---------|-----------|
| Literature / approach scoping | Summarise CycleGAN / Pix2Pix / LDM trade-offs, surface options | Choose the 4 architectures, define the comparison |
| Scaffolding & boilerplate | Draft configs, trainers, dataloaders, CLI | Specify contracts, integrate, own structure |
| Domain logic (normalization, masking, registration) | Draft skeletons | Supply HU windows, spacing, split policy, masking strategy |
| Debugging | Generate hypotheses for a failing run | Reproduce, bisect, confirm root cause, accept/reject |
| Metrics & evaluation | Draft implementations | **Verify every formula and value range by hand** |
| Documentation | Draft prose + LaTeX | Fact-check against code, edit for accuracy |
| Final correctness | — | Run, measure, sign off |

Principles that kept AI useful rather than dangerous:
- **AI accelerates typing and exploration; humans own correctness.** Anything that affects a
  reported number (loss, metric, normalization) gets manual verification regardless of how
  confident the suggestion looks.
- **Small, reviewable diffs.** Generated code goes in as focused changes that can be read in full,
  not large opaque dumps.
- **The model never has the last word on a bug.** Every AI-reported issue is reproduced or
  re-derived before it is "fixed" — which is how a wrong "fix" to the CT normalization was caught.

---

### TL;DR
AI roughly doubled my throughput on plumbing and documentation and was a strong debugging partner,
but the project's correctness — normalization ranges, masked metrics, training dynamics, and
reproducibility — was owned, verified, and signed off by me. The fixes documented in
`docs/KNOWN_ISSUES.md` and the `CHANGELOG.md` are concrete evidence of that verification loop.
