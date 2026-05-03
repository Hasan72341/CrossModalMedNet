# Reproducibility

Use this checklist when adding a new experiment or result table.

## Record

- Commit hash and branch.
- Dataset source, preprocessing configuration, and `manifest.csv` checksum.
- Train/validation/test split file.
- Model family, cohort, checkpoint path, and image resolution.
- GPU model, CUDA version, PyTorch version, and random seed.
- Metric implementation versions for LPIPS, SSIM, MS-SSIM, PSNR, FID, and Dice.

## Artifact Policy

Track source code, lightweight configs, final summaries, report tables, and selected publication-quality figures. Do not track raw medical data, checkpoints, tensor dumps, virtual environments, full training logs, or per-epoch generated images.

## Recommended Result Layout

```text
project-group-5/evaluation/
├── <model>_<region>_metrics.csv
├── <model>_<region>_summary.json
└── plots/analysis/
    ├── REPORT.md
    ├── summary_table.csv
    └── *.png
```
