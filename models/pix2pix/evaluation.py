"""
pix2pix_standard/evaluation.py
-----------------------------
Evaluation logic for Pix2Pix CT → MRI translation.
Mirroring the CycleGAN implementation style.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import lpips
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Using torchmetrics where possible (consistent with reference)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def save_test_visuals(
    model, 
    test_loader, 
    device: torch.device, 
    save_dir: Path, 
    epoch: int
) -> None:
    """Save a visualization: Real CT, Fake MRI, Real MRI"""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab one batch
    batch = next(iter(test_loader))
    ct, real_mri, mask, meta = batch
    ct, real_mri, mask = ct.to(device), real_mri.to(device), mask.to(device)
    
    with torch.no_grad():
        fake_mri = model.G_CT2MRI(ct)
        
    # Apply mask for clean visualization
    fake_mri = fake_mri * mask
    real_mri = real_mri * mask
        
    # Denormalize [-1, 1] -> [0, 1]
    def denorm(x): return (x + 1) / 2
    
    ct_val = denorm(ct)[0, 0].cpu().numpy()
    real_mri_val = denorm(real_mri)[0, 0].cpu().numpy()
    fake_mri_val = denorm(fake_mri)[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ct_val, cmap='gray'); axes[0].set_title("Input CT")
    axes[1].imshow(fake_mri_val, cmap='gray'); axes[1].set_title("Fake MRI")
    axes[2].imshow(real_mri_val, cmap='gray'); axes[2].set_title("Real MRI")
    
    for ax in axes:
        ax.axis('off')
        
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(save_dir / f"epoch_{epoch:03d}_visuals.png")
    plt.close()


def run_evaluation(
    model,
    test_loader,
    device: torch.device,
    save_dir: Path,
    epoch: int,
) -> dict:
    """Run metrics (LPIPS, MSE, SSIM, PSNR) on the test set."""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    print(f"Initialized LPIPS with net='vgg' for epoch {epoch}")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Patient-wise metrics log
    patient_log_path = save_dir / f"patient_metrics_epoch_{epoch:03d}.csv"
    headers = ["patient_id", "ssim", "psnr", "mse", "lpips"]
    
    metrics_accumulator = {k: [] for k in headers[1:]}
    
    print(f"Evaluating epoch {epoch} on {len(test_loader)} batches...")
    
    with open(patient_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for batch in tqdm(test_loader, desc=f"Eval Epoch {epoch}"):
            ct, real_mri, mask, meta = batch
            ct, real_mri, mask = ct.to(device), real_mri.to(device), mask.to(device)
            
            ct_paths = meta.get('ct_path', [])
            patient_ids = [Path(str(cp)).name.split('_')[0] for cp in ct_paths]
            if not patient_ids:
                patient_ids = ['unknown'] * ct.shape[0]
            
            with torch.no_grad():
                fake_mri = model.G_CT2MRI(ct)
                
            # Denormalize [0, 1] for metrics
            real_01 = ((real_mri[:, 0:1, :, :] + 1) / 2) * mask
            fake_01 = ((fake_mri[:, 0:1, :, :] + 1) / 2) * mask
            
            for b in range(ct.shape[0]):
                # Metrics using torchmetrics
                s_val = ssim_metric(fake_01[b:b+1], real_01[b:b+1]).item()
                p_val = psnr_metric(fake_01[b:b+1], real_01[b:b+1]).item()
                mse_val = torch.mean((fake_01[b:b+1] - real_01[b:b+1]) ** 2).item()
                
                f_lp = fake_01[b:b+1].expand(-1, 3, -1, -1)
                r_lp = real_01[b:b+1].expand(-1, 3, -1, -1)
                lp_val = lpips_model(f_lp, r_lp).mean().item()
                
                patient_id = patient_ids[b]
                row = {
                    "patient_id": patient_id,
                    "ssim": s_val,
                    "psnr": p_val,
                    "mse": mse_val,
                    "lpips": lp_val
                }
                writer.writerow(row)
                
                for k in metrics_accumulator:
                    metrics_accumulator[k].append(row[k])
        
    summary = {k: float(np.mean(v)) for k, v in metrics_accumulator.items()}
    summary["epoch"] = epoch
    summary["timestamp"] = datetime.now().isoformat()
    
    with open(save_dir / f"summary_epoch_{epoch:03d}.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    history_path = save_dir.parent / "evaluation_history.csv"
    h_headers = ["epoch"] + list(summary.keys())
    write_header = not history_path.exists()
    with open(history_path, "a", newline="") as f:
        h_writer = csv.DictWriter(f, fieldnames=h_headers)
        if write_header:
            h_writer.writeheader()
        h_writer.writerow(summary)
        
    return summary
