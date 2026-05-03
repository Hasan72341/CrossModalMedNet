# CrossModalMedNet: Medical Image Translation for SynthRAD 2023

CrossModalMedNet is a comprehensive research repository focused on high-fidelity medical image translation, specifically targeting CT-to-MRI synthesis using the SynthRAD 2023 dataset. It implements three state-of-the-art architectures optimized for both Brain and Pelvis regions.

## Architectures

1.  **CycleGAN**: Unpaired image-to-image translation using cycle-consistency loss and WGAN-GP.
2.  **Pix2Pix**: Paired image-to-image translation using conditional GANs with L1 reconstruction loss.
3.  **Diffusion (Paired)**: Latent Diffusion Model (LDM) fine-tuned for precise paired translation with structural guidance.

## Project Structure

```text
CrossModalMedNet/
├── models/
│   ├── cyclegan/       # CycleGAN implementation
│   ├── pix2pix/        # Pix2Pix implementation
│   └── diffusion/      # Diffusion-based translation
├── checkpoints/        # Pre-trained weights for all 6 models
├── src/                # Shared source code
│   ├── data/           # Data loading and preprocessing
│   └── utils/          # Common utilities
├── app.py              # Streamlit-based web interface
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## Getting Started

### Installation

```bash
git clone https://github.com/[YOUR_USERNAME]/CrossModalMedNet.git
cd CrossModalMedNet
pip install -r requirements.txt
```

### Running the Web Interface

Launch the interactive demo to visualize CT-to-MRI translation:

```bash
streamlit run app.py
```

## Model Training & Evaluation

Each model directory contains dedicated training and evaluation scripts. 

-   **CycleGAN**: `python models/cyclegan/train.py`
-   **Pix2Pix**: `python models/pix2pix/train.py`
-   **Diffusion**: `python models/diffusion/train.py`

## Checkpoints

The repository includes the latest checkpoints for:
- CycleGAN (Brain/Pelvis)
- Pix2Pix (Brain/Pelvis)
- Diffusion (Brain/Pelvis)

## Dataset

This project utilizes the [SynthRAD 2023](https://synthrad2023.grand-challenge.org/) dataset.

## License

MIT License
