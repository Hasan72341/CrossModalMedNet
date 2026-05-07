# CrossModalMedNet: High-Fidelity Medical Image Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

CrossModalMedNet is a comprehensive research-grade repository for high-fidelity CT-to-MRI medical image translation, specifically targeting the [SynthRAD 2023](https://synthrad2023.grand-challenge.org/) dataset. This repository implements 8 specialized models across 4 state-of-the-art architectures, optimized for CT-to-MRI synthesis in both Brain and Pelvis regions.

## 🔬 Overview

Synthetic MRI generation from CT scans is a critical task in radiotherapy planning, providing essential soft-tissue contrast without additional ionizing radiation. CrossModalMedNet provides a standardized framework to benchmark GAN and Diffusion-based methods for CT-to-MRI translation.

### Key Features
- **Multi-Architecture Support**: CycleGAN, Pix2Pix, Paired LDM, and Unpaired Cycle-Diffusion.
- **Region Optimization**: Specialized models for both **Brain** and **Pelvis** anatomy.
- **Research Utility**: Unified data pipeline, standard evaluation metrics (SSIM, PSNR, LPIPS), and HU-aware normalization.
- **Interactive Interface**: Streamlit-based web dashboard for real-time inference and visualization.

## 🏗️ Architecture details

| Architecture | Paradigm | Loss Functions | Best For |
| :--- | :--- | :--- | :--- |
| **CycleGAN** | Unpaired GAN | Cycle-Consistency, Identity, WGAN-GP | Unlabeled data scenarios |
| **Pix2Pix** | Paired GAN | Conditional Adversarial + L1 Reconstruction | Structural precision |
| **Diffusion (Paired)** | Latent Diffusion | Noise Prediction MSE + LDM | High perceptual quality |
| **Diffusion (Unpaired)** | Cycle-Diffusion | Variational Inference + Structural Bottleneck | High-fidelity unpaired |

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Hasan72341/CrossModalMedNet.git
cd CrossModalMedNet
```

2. Set up the environment:
```bash
pip install -r requirements.txt
pip install -e .
```

### Quick Start: Web Interface
Launch the interactive demo to visualize results:
```bash
streamlit run app.py
```

### CLI Inference
Run translation on a specific image via command line:
```bash
python inference.py --architecture cyclegan --region brain --input path/to/ct_slice.pt --output result.png
```

## 📊 Evaluation & Reproducibility

Detailed evaluation reports for each architecture are available in the `docs/` directory:
- [CycleGAN Report](docs/cyclegan_brain/cyclegan_brain_report.pdf)
- [Pix2Pix Report](docs/pix2pix_brain/pix2pix_brain_report.pdf)
- [Diffusion Report](docs/paired_diffusion_brain/paired_diffusion_brain_report.pdf)

Metrics are computed using the `src.utils.metrics` module, ensuring consistent benchmarking.

## 📂 Project Structure
```text
CrossModalMedNet/
├── models/             # Architecture-specific implementations (2D & 3D)
│   ├── cyclegan/       # 2D CycleGAN
│   ├── cyclegan_3d/    # 3D CycleGAN (Research grade)
│   ├── pix2pix/        # 2D Pix2Pix
│   └── diffusion/      # LDM-based models
├── src/                # Shared research logic
│   ├── data/           # Advanced SimpleITK-based preprocessing & Dataset
│   └── utils/          # Masked Metrics (MAE, SSIM, PSNR, LPIPS), Visualization
├── checkpoints/        # Pre-trained SOTA weights (Git LFS)
├── configs/            # Experiment YAML configurations
│   ├── preprocess.yaml # Global preprocessing settings
│   ├── dataset.yaml    # Shared dataset parameters
│   ├── cyclegan_*.yaml # Model-specific configs (Brain/Pelvis)
│   ├── pix2pix_*.yaml
│   └── diffusion_*.yaml
├── scripts/            # Data inspection and validation tools
├── docs/               # Detailed research reports and documentation
├── app.py              # Streamlit dashboard
└── inference.py        # CLI inference tool
```

## 📊 Data Preparation (Research Grade)

The repository provides a high-fidelity preprocessing pipeline using **SimpleITK** for rigorous medical image handling.

1.  **Preprocessing**: Resample volumes to isotropic spacing, reorient to RAS, and crop to mask bounding boxes. Configuration is managed via `configs/preprocess.yaml`:
    ```bash
    python src/data/preprocess.py --config configs/preprocess.yaml
    ```
2.  **Dataset**: Our `SynthRADDataset` (in `src/data/dataset.py`) is portable and supports:
    -   **3D Mode**: Returns full volumetric patches.
    -   **2.5D Mode**: Returns multi-slice stacks (e.g., 3-slice) for context-aware 2D training.
    -   **2D Mode**: Standard single-slice translation.

## 🔬 Evaluation Metrics

We implement **Masked Metrics** (MAE, MSE, SSIM, PSNR) as the primary benchmark. By focusing only on the anatomical foreground, we ensure that evaluation is not biased by background zero-padding, providing a true measure of clinical utility.

## 🚀 Getting Started

If you use this code in your research, please cite:
```bibtex
@software{CrossModalMedNet2026,
  author = {CrossModalMedNet Authors},
  title = {CrossModalMedNet: High-Fidelity Medical Image Translation},
  url = {https://github.com/Hasan72341/CrossModalMedNet},
  year = {2026}
}
```

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
