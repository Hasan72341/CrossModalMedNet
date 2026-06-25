# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- **Metrics:** LPIPS now repeats single-channel scans to 3 channels (LPIPS-VGG requires RGB) —
  previously crashed on grayscale input. (`src/utils/metrics.py`)
- **Metrics:** Masked PSNR is computed from the foreground-only MSE instead of mask-multiplied
  images, so the zeroed background no longer inflates the score. (`src/utils/metrics.py`)
- **Data:** Validation/test now use a deterministic centre slice and deterministic unpaired
  pairing; randomization is restricted to `split='train'`. Makes reported metrics reproducible.
  (`src/data/dataset.py`)
- **Training:** CycleGAN evaluation helpers restore `model.train()` after `eval()`
  (dropout was being left disabled mid-training). (`models/cyclegan/evaluation.py`)
- **Training:** CycleGAN no longer logs a stale discriminator loss when `d_update_every > 1`.
  (`models/cyclegan/train.py`)
- **Inference:** Checkpoints load with `strict=False` and report key mismatches; `F.interpolate`
  calls pin `align_corners=False`. (`inference.py`)

### Added
- `docs/AI_FLUENCY.md` — how AI was used in development (productivity, prompt engineering,
  critical assessment, workflow integration).
- `docs/KNOWN_ISSUES.md` — engineering audit trail of issues found, fixed, and documented.
- `CHANGELOG.md` — this file.

## [0.1.0]
- Initial public release: CT→MRI translation across CycleGAN, Pix2Pix, paired diffusion and
  unpaired diffusion, for brain and pelvis; SynthRAD 2023 preprocessing pipeline; masked
  evaluation; Streamlit demo.
