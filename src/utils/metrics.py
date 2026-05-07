import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import lpips
import numpy as np

class MetricsCalculator:
    """
    Research-grade metrics calculator for medical image translation.
    Focuses on masked evaluation (foreground only) which is the gold standard.
    """
    def __init__(self, device='cuda'):
        self.device = device
        # Use data_range=2.0 because our tensors are normalized to [-1, 1]
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        self.lpips_vgg.eval()

    @torch.no_grad()
    def calculate_metrics(self, fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """
        Calculate metrics. If mask is provided, MAE/MSE are computed foreground-only.
        Spatial metrics (SSIM/PSNR) are computed on masked images.
        """
        fake = fake.to(self.device)
        real = real.to(self.device)
        
        results = {}
        
        if mask is not None:
            mask = mask.to(self.device).bool()
            # 1. Masked MAE / MSE
            diff = (fake - real)
            masked_diff = diff[mask]
            
            if masked_diff.numel() > 0:
                results['mae_masked'] = masked_diff.abs().mean().item()
                results['mse_masked'] = (masked_diff**2).mean().item()
            else:
                results['mae_masked'] = 0.0
                results['mse_masked'] = 0.0

            # 2. SSIM / PSNR on masked images
            # Multiply by mask to zero out background
            results['ssim'] = self.ssim_metric(fake * mask, real * mask).item()
            results['psnr'] = self.psnr_metric(fake * mask, real * mask).item()
        else:
            # Fallback to full-image metrics
            results['mae'] = torch.abs(fake - real).mean().item()
            results['mse'] = ((fake - real)**2).mean().item()
            results['ssim'] = self.ssim_metric(fake, real).item()
            results['psnr'] = self.psnr_metric(fake, real).item()

        # 3. LPIPS (always global or on masked - usually global is fine for perceptual)
        results['lpips'] = self.lpips_vgg(fake, real).mean().item()
        
        return results

def compute_hu_metrics(fake_hu: np.ndarray, real_hu: np.ndarray, mask: np.ndarray = None) -> dict:
    """Compute metrics in Hounsfield Units (HU)."""
    if mask is not None:
        mask = mask.astype(bool)
        fake_hu = fake_hu[mask]
        real_hu = real_hu[mask]
        
    mae = np.mean(np.abs(fake_hu - real_hu))
    rmse = np.sqrt(np.mean((fake_hu - real_hu)**2))
    return {"mae_hu": mae, "rmse_hu": rmse}
