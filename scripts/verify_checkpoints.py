import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

models = [
    "pix2pix/brain",
    "pix2pix/pelvis",
    "cyclegan/brain",
    "cyclegan/pelvis",
    "paired_diffusion/brain",
    "paired_diffusion/pelvis",
    "unpaired_diffusion/brain",
    "unpaired_diffusion/pelvis"
]

all_passed = True

print("Verifying model checkpoints...")
for model in models:
    path = PROJECT_ROOT / "models" / model
    if not path.exists():
        print(f"FAIL: {model} (Directory missing)")
        all_passed = False
        continue

    # Check for weights/ or checkpoints/
    has_ckpts = (path / "checkpoints").is_dir() or (path / "weights").is_dir()
    
    # Check for any weight
    has_epoch_500 = False
    ckpt_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path / "weights"
    if ckpt_dir.exists():
        for root, _, files in os.walk(ckpt_dir):
            if any(f.endswith(".pth") or f.endswith(".pt") for f in files):
                has_epoch_500 = True
                break

    # Check config
    has_config = (path / "config.py").is_file() or \
                 (path / "options.py").is_file() or \
                 (path / "configs").is_dir() or \
                 (path / "config.yaml").is_file()

    if has_ckpts and has_epoch_500 and has_config:
        print(f"PASS: {model}")
    else:
        missing = []
        if not has_ckpts: missing.append("checkpoints/weights dir")
        if not has_epoch_500: missing.append("weights")
        if not has_config: missing.append("config file")
        print(f"FAIL: {model} (Missing: {', '.join(missing)})")
        all_passed = False

if not all_passed:
    sys.exit(1)
sys.exit(0)
