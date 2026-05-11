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

Run the pipeline:
```bash
python src/data/preprocess.py --config configs/preprocess.yaml
```

---

## 🏋️ Training & Inference

### Model Training
Reference any of the 8 model configs to start training:
```bash
# Example: Train Pix2Pix for Pelvis
python models/pix2pix/train.py --config configs/pix2pix_pelvis.yaml
```

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

Detailed comparative analysis reports are available in `docs/`:
- [CycleGAN vs Diffusion Analysis](docs/all_models_comparison/all_models_comparison.md)
- [Brain Optimization Study](docs/cyclegan_brain/cyclegan_brain_documentation.md)

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
