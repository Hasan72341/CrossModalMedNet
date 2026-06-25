# CrossModalMedNet: High-Fidelity Medical Image Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Git LFS](https://img.shields.io/badge/Git-LFS-orange.svg)](https://git-lfs.github.com/)

CrossModalMedNet is a comprehensive, research-grade framework for high-fidelity **CT-to-MRI medical image translation**. Optimized for the [SynthRAD 2023](https://synthrad2023.grand-challenge.org/) dataset, it provides modular implementations of 8 specialized models across 4 state-of-the-art architectures (GANs and Diffusion) for both **Brain** and **Pelvis** anatomy.

## 🔬 Project Overview

Synthetic MRI generation from CT scans is critical for MRI-only radiotherapy planning, providing superior soft-tissue contrast without the ionizing radiation or cost of additional MRI scans. This repository standardizes the entire research lifecycle: from rigorous SimpleITK-based preprocessing to masked evaluation metrics.

### 🌟 Key Features
-   **8 Specialized Models**: Architecture-region specific models (CycleGAN, Pix2Pix, Paired LDM, Unpaired Cycle-Diffusion).
-   **Research-Grade Preprocessing**: Multi-threaded SimpleITK pipeline for resampling, RAS reorientation, and mask-aware cropping.
-   **Masked Evaluation**: SSIM, PSNR, LPIPS, and MAE computed exclusively on anatomical foregrounds to ensure clinical relevance.
-   **Portable Design**: Unified configuration system (YAML) and portable dataset loaders (manifest-based).
-   **Interactive Ecosystem**: Integrated Streamlit dashboard and CLI tools for real-time inference and benchmarking.

---

## 🏗️ Architecture Suite (8 Models)

| Architecture | Paradigm | Region | Training Mode | Configuration File |
| :--- | :--- | :--- | :--- | :--- |
| **CycleGAN 2D** | Unpaired GAN | Brain | Unpaired | `configs/cyclegan2d_brain.yaml` |
| **CycleGAN 2D** | Unpaired GAN | Pelvis | Unpaired | `configs/cyclegan2d_pelvis.yaml` |
| **Pix2Pix 2D** | Paired GAN | Brain | Paired | `configs/pix2pix2d_brain.yaml` |
| **Pix2Pix 2D** | Paired GAN | Pelvis | Paired | `configs/pix2pix2d_pelvis.yaml` |
| **Diffusion 2.5D** | Paired LDM | Brain | Paired | `configs/paired_diffusion2.5d_brain.yaml` |
| **Diffusion 2.5D** | Paired LDM | Pelvis | Paired | `configs/paired_diffusion2.5d_pelvis.yaml` |
| **Diffusion 2.5D** | Cycle-Diffusion | Brain | Unpaired | `configs/unpaired_diffusion2.5d_brain.yaml` |
| **Diffusion 2.5D** | Cycle-Diffusion | Pelvis | Unpaired | `configs/unpaired_diffusion2.5d_pelvis.yaml` |

---

## 🚀 Installation

### Troubleshooting

- Ensure Git LFS is installed before downloading checkpoints.
- Use Python 3.10 or newer.
- It is recommended to use a virtual environment to avoid dependency conflicts.

### Prerequisites
- **Git LFS**: Required to download pre-trained model checkpoints.
  ```bash
  git lfs install
  ```

### Environment Setup
```bash
git clone https://github.com/Hasan72341/CrossModalMedNet.git
cd CrossModalMedNet
pip install -r requirements.txt
pip install -e .
```

---

## 📦 Data Source

This project uses the **SynthRAD 2023 Grand Challenge** dataset (paired CT and MR volumes with
anatomical masks, brain & pelvis):

- 🔗 **Challenge:** https://synthrad2023.grand-challenge.org/
- 📥 **Dataset (Task 1):** https://synthrad2023.grand-challenge.org/data/ — access requires a free
  Grand Challenge account; the data is released for research use under the challenge's terms.
- 📄 **Dataset citation:** Thummerer A. *et al.*, "SynthRAD2023 Grand Challenge dataset: Generating
  synthetic CT for radiotherapy," *Medical Physics*, 2023. Please cite the dataset authors if you
  use this data.

> ⚠️ Raw imaging data is **not** redistributed in this repository — download it from the Grand
> Challenge above and run the preprocessing pipeline below.

## 📊 Data Lifecycle

### 1. Raw Data Structure
Download the [SynthRAD 2023](https://synthrad2023.grand-challenge.org/) Task 1 data and organize it as follows:
```text
data/raw/
├── brain/
│   ├── 1BA001/ {ct.nii.gz, mr.nii.gz, mask.nii.gz}
│   └── ...
└── pelvis/
    ├── 1PA001/ {ct.nii.gz, mr.nii.gz, mask.nii.gz}
    └── ...
```

### 2. Standardized Preprocessing
Our pipeline ensures all research data is clinically comparable:
1.  **Resampling**: To isotropic spacing (Brain: 1mm, Pelvis: 1.0x1.0x2.5mm).
2.  **Reorientation**: Fixed to RAS (Right-Anterior-Superior).
3.  **Cropping**: Automated mask bounding-box cropping with 5-voxel margins.
4.  **Normalization**: HU-aware scaling (CT: [-1000, 1000] $\rightarrow$ [-1, 1]).

Run the pipeline (writes processed `.pt` volumes + `manifest.csv` to `--output`):
```bash
python src/data/preprocess.py --data2023 data/raw --output ./SynthRAD2023_Dataset --workers 4
# add --dry_run to validate without writing, or --patient_id 1BA001 for a single case
```

---

## 🏋️ Training & Inference

### Model Training
Training is configured through environment variables / `Settings` (pydantic), not a `--config`
flag. Copy `.env.example` to `.env` and set `DATA_ROOT` / `MANIFEST_PATH` (and region/hyper-params
as needed), then launch the trainer module:
```bash
cp .env.example .env            # then edit DATA_ROOT, MANIFEST_PATH, region, ...
python -m models.pix2pix.train --num_epochs 200   # --num_epochs overrides the configured value
```
For a quick verified end-to-end smoke test (eval path on a tiny test list):
```bash
make smoke-pix2pix-brain
```
> The per-architecture YAMLs in `configs/` (e.g. `pix2pix2d_pelvis.yaml`) drive the Streamlit app
> and inference model selection.

### Dashboard Inference
Launch the interactive Streamlit interface to run cross-model comparisons:
```bash
streamlit run app.py
```

### CLI Benchmarking
Run inference and generate research metrics (SSIM/PSNR) for specific slices:
```bash
python inference.py --architecture cyclegan --region brain --input input.pt --target target.pt --output result.png
```

---

## 📂 Project Structure

```text
CrossModalMedNet/
├── models/             # Specialized DL Architectures
│   ├── cyclegan/       # 2D Unpaired GANs
│   ├── cyclegan_3d/    # Volumetric 3D CycleGAN
│   ├── pix2pix/        # 2D Paired GANs
│   └── diffusion/      # LDM-based Paired/Unpaired Translation
├── src/                # Core Research Engine
│   ├── data/           # Preprocessing (SimpleITK) & Portable Datasets
│   └── utils/          # Masked Metrics & High-res Visualization
├── configs/            # Modular Experiment Configuration (10 YAMLs)
├── checkpoints/        # Pre-trained SOTA Weights (Git LFS)
├── presentation/       # Final LaTeX presentation and assets
├── scripts/            # Dataset Validation & Evaluation Scripts
├── docs/               # Technical Reports & Comparative Studies
├── app.py              # Unified Research Dashboard
└── inference.py        # CLI Inference & Metrics Engine
```

---

## 🔬 Evaluation & Reproducibility

Metrics are calculated using the `src.utils.metrics` module. We report **Masked Metrics** (MAE, MSE, SSIM, PSNR, LPIPS) to prevent background padding from inflating performance scores, focusing purely on anatomical translation fidelity.

> **Note on masked metrics:** MAE/MSE/PSNR are computed on the anatomical foreground only; SSIM is
> a spatial metric and is computed on background-zeroed images. `data_range=2.0` (tensors are in
> `[-1, 1]`). See `src/utils/metrics.py`.

Detailed comparative analysis reports are available in `docs/`:
- [CycleGAN vs Diffusion Analysis](docs/all_models_comparison/all_models_comparison.md)
- [Brain Optimization Study](docs/cyclegan_brain/cyclegan_brain_documentation.md)

---

## 🧠 Engineering & AI-Assisted Development
- **[docs/AI_FLUENCY.md](docs/AI_FLUENCY.md)** — how AI tools were used in development: driving
  productivity, prompt engineering, critical assessment of AI output, and workflow integration.
- **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** — engineering audit trail: correctness issues
  found, fixed, and documented (e.g. LPIPS channel handling, masked-PSNR, reproducible validation).
- **[CHANGELOG.md](CHANGELOG.md)** — versioned change history.

---

## 📜 Citation
If you use this framework in your research, please cite:
```bibtex
@software{CrossModalMedNet2026,
  author = {CrossModalMedNet Authors},
  title = {CrossModalMedNet: High-Fidelity Medical Image Translation},
  url = {https://github.com/Hasan72341/CrossModalMedNet},
  year = {2026}
}
```

## ⚖️ License
Licensed under the [MIT License](LICENSE).
