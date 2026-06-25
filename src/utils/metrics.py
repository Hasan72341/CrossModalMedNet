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

            # 2. SSIM is a spatial metric -> compute on masked images (background zeroed).
            mask_f = mask.float()
            results['ssim'] = self.ssim_metric(fake * mask_f, real * mask_f).item()
            # PSNR from the FOREGROUND-only MSE so the (large) zeroed background does not
            # inflate the score. data_range = 2.0 for [-1, 1] -> data_range**2 = 4.0.
            if masked_diff.numel() > 0:
                results['psnr'] = float(10.0 * np.log10(4.0 / (results['mse_masked'] + 1e-8)))
            else:
                results['psnr'] = 0.0
        else:
            # Fallback to full-image metrics
            results['mae'] = torch.abs(fake - real).mean().item()
            results['mse'] = ((fake - real)**2).mean().item()
            results['ssim'] = self.ssim_metric(fake, real).item()
            results['psnr'] = self.psnr_metric(fake, real).item()

        # 3. LPIPS (perceptual, global). LPIPS-VGG expects 3-channel RGB in [-1, 1];
        # our scans are single-channel, so repeat the grey channel to 3 channels.
        fake_rgb = fake.repeat(1, 3, 1, 1) if fake.shape[1] == 1 else fake
        real_rgb = real.repeat(1, 3, 1, 1) if real.shape[1] == 1 else real
        results['lpips'] = self.lpips_vgg(fake_rgb, real_rgb).mean().item()
        
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
