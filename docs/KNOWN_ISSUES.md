# Known Issues & Engineering Audit

A running log of correctness issues found during code review, what was fixed, and what is a
documented limitation (intentionally left, with rationale). This is part of the project's
verification trail — see `docs/AI_FLUENCY.md` §3.

## Fixed

| Area | Issue | Fix | File |
|------|-------|-----|------|
| Metrics | **LPIPS crashed on single-channel input** — `lpips.LPIPS(net='vgg')` requires 3-channel RGB. | Repeat the grey channel to 3 channels before the LPIPS call. | `src/utils/metrics.py` |
| Metrics | **Masked PSNR was inflated** — computing PSNR on mask-multiplied images let the zeroed background dominate and raise the score. | Compute PSNR from the foreground-only MSE (`10·log10(data_range² / mse_fg)`, `data_range=2`). | `src/utils/metrics.py` |
| Data | **Non-deterministic validation** — random slice (and random unpaired CT pairing) were used for every split, so val/test metrics changed run-to-run. | Random only for `split='train'`; deterministic centre slice + deterministic pairing for val/test. | `src/data/dataset.py` |
| Training | **`model.eval()` not restored** — CycleGAN eval helpers left the model in eval mode, disabling dropout for the rest of training. | Save and restore `model.training` around evaluation. | `models/cyclegan/evaluation.py` |
| Training | **Stale discriminator loss logged** — with `d_update_every > 1`, `d_loss` from a previous iteration was logged/averaged on non-D steps. | Only update the D-loss metric / progress bar when a discriminator step actually ran (`d_results is not None`). | `models/cyclegan/train.py` |
| Inference | **Silent/strict checkpoint load** — key mismatches surfaced as a hard failure with no diagnostics. | Load with `strict=False` and print missing/unexpected key counts. | `inference.py` |
| Inference | `F.interpolate(..., mode='bilinear')` without `align_corners` (deprecation warning + ambiguous behaviour). | Pin `align_corners=False`. | `inference.py` |

## Verified-correct (reported as bugs, but were not)

- **CT normalization formula** `2·(x − CT_MIN)/(CT_MAX − CT_MIN) − 1` with `CT_MIN=-1000, CT_MAX=1000`
  correctly maps `[-1000, 1000] HU → [-1, 1]` (max → `2·(2000/2000) − 1 = +1`). An automated review
  flagged this as mapping max→0; that derivation was wrong and the suggested change was rejected.
  (`src/data/preprocess.py`)

## Documented limitations (intentional, not yet changed)

- **2.5D coronal/sagittal slice extraction** (`src/data/dataset.py::_extract_slices`): the default
  `slice_axis='axial'` path is correct and is what all shipped configs use. The coronal/sagittal
  permutations are less-tested; if you switch axes, verify the output orientation before training.
- **SSIM under masking** is computed on background-zeroed images (SSIM is spatial and cannot be
  trivially restricted to arbitrary foreground pixels). MAE/MSE/PSNR are foreground-only. Interpret
  masked SSIM accordingly.
- **Dependency pinning** in `requirements.txt` is loose (`torch>=2.0.0`, unpinned
  `diffusers`/`transformers`). Pin exact versions for a fully reproducible environment.
