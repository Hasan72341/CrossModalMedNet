from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm

from .config import Settings
from .data import create_dataloaders
from .utils import unwrap

import matplotlib.pyplot as plt


def save_test_visuals(
    model, 
    test_loader, 
    device: torch.device, 
    save_dir: Path, 
    epoch: int
) -> None:
    """Save a 4-panel visualization: Real MRI, Fake CT, Real CT, Fake MRI"""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab one batch
    batch = next(iter(test_loader))
    ct, mri, mask, meta = batch
    ct, mri, mask = ct.to(device), mri.to(device), mask.to(device)
    
    with torch.no_grad():
        fake_ct = model.G_MRI2CT(mri)
        fake_mri = model.G_CT2MRI(ct)
        
    # Apply mask for clean visualization
    if mask.shape[2:] != fake_mri.shape[2:]:
        mask = F.interpolate(mask, size=fake_mri.shape[2:], mode='nearest')
        
    fake_ct = fake_ct * mask
    fake_mri = fake_mri * mask
    ct = ct * mask
    mri = mri * mask
        
    # Denormalize [-1, 1] -> [0, 1]
    def denorm(x): return (x + 1) / 2
    
    ct_val = denorm(ct)[0, 0].cpu().numpy()
    mri_val = denorm(mri)[0, 0].cpu().numpy()
    fake_ct_val = denorm(fake_ct)[0, 0].cpu().numpy()
    fake_mri_mval = denorm(fake_mri)[0, 0].cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(mri_val, cmap='gray'); axes[0].set_title("Real MRI")
    axes[1].imshow(fake_ct_val, cmap='gray'); axes[1].set_title("Fake CT")
    axes[2].imshow(ct_val, cmap='gray'); axes[2].set_title("Real CT")
    axes[3].imshow(fake_mri_mval, cmap='gray'); axes[3].set_title("Fake MRI")
    
    for ax in axes:
        ax.axis('off')
        
    ct_paths = meta.get('ct_path', ['unknown'])
    patient_id = "unknown"
    if ct_paths and str(ct_paths[0]) != 'unknown':
        patient_id = Path(str(ct_paths[0])).name.split('_')[0]
        
    plt.suptitle(f"Epoch {epoch} | Patient {patient_id}")
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
    """Run metrics (LPIPS, MSE, SSIM, PSNR) on the test set and log patient-wise."""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    
    # Patient-wise metrics log
    patient_log_path = save_dir / f"patient_metrics_epoch_{epoch:03d}.csv"
    headers = ["patient_id", "ssim", "psnr", "mse", "lpips"]
    
    metrics_accumulator = {k: [] for k in headers[1:]}
    
    print(f"Evaluating epoch {epoch} on {len(test_loader)} batches...")
    
    with open(patient_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for batch in tqdm(test_loader, desc=f"Eval Epoch {epoch}"):
            ct, mri, mask, meta = batch
            ct, mri, mask = ct.to(device), mri.to(device), mask.to(device)
            ct_paths = meta.get('ct_path', [])
            patient_ids = [Path(str(cp)).name.split('_')[0] for cp in ct_paths]
            if not patient_ids:
                patient_ids = ['unknown'] * ct.shape[0]
            
            with torch.no_grad():
                fake_mri = model.G_CT2MRI(ct)
                
            # Denormalize (use only origin resolution channel 0)
            real_01 = ((mri[:, 0:1, :, :] + 1) / 2) * mask
            fake_01 = ((fake_mri[:, 0:1, :, :] + 1) / 2) * mask
            
            # Resize if needed
            if fake_01.shape[2:] != real_01.shape[2:]:
                import torch.nn.functional as F
                fake_01 = F.interpolate(fake_01, size=real_01.shape[2:], mode='bilinear', align_corners=False)
                fake_mri = F.interpolate(fake_mri, size=mri.shape[2:], mode='bilinear', align_corners=False)

            # Metrics
            # For simplicity, handle batch_size=1 or average across batch if batch_size > 1
            # But the user asked for "patient wise", so let's iterate batch
            for b in range(ct.shape[0]):
                s_val = ssim(fake_01[b:b+1], real_01[b:b+1], data_range=1.0, size_average=True).item()
                mse_val = torch.mean((fake_01[b:b+1] - real_01[b:b+1]) ** 2).item()
                psnr_val = 10 * torch.log10(1.0 / (torch.tensor(mse_val) + 1e-8)).item()
                
                f_lp = fake_mri[b:b+1, 0:1, :, :].expand(-1, 3, -1, -1)
                r_lp = mri[b:b+1, 0:1, :, :].expand(-1, 3, -1, -1)
                lp_val = lpips_model(f_lp, r_lp).mean().item()
                
                patient_id = patient_ids[b]
                row = {
                    "patient_id": patient_id,
                    "ssim": s_val,
                    "psnr": psnr_val,
                    "mse": mse_val,
                    "lpips": lp_val
                }
                writer.writerow(row)
                
                for k in metrics_accumulator:
                    metrics_accumulator[k].append(row[k])
        
    summary = {k: float(np.mean(v)) for k, v in metrics_accumulator.items()}
    summary["epoch"] = epoch
    summary["num_patients"] = len(metrics_accumulator["ssim"])
    summary["timestamp"] = datetime.now().isoformat()
    
    # Save summary JSON
    with open(save_dir / f"summary_epoch_{epoch:03d}.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    # Append to global history
    history_path = save_dir.parent / "evaluation_history.csv"
    h_headers = ["epoch"] + list(summary.keys())
    write_h = not history_path.exists()
    with open(history_path, "a", newline="") as f:
        h_writer = csv.DictWriter(f, fieldnames=h_headers)
        if write_h:
            h_writer.writeheader()
        h_writer.writerow(summary)
        
    return summary
