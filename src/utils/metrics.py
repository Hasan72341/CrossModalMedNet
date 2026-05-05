import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim
import lpips
import numpy as np

class MetricsCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        self.lpips_vgg.eval()

    @torch.no_grad()
    def calculate_metrics(self, fake: torch.Tensor, real: torch.Tensor) -> dict:
        """
        Calculate standard medical imaging metrics.
        Expects tensors in range [-1, 1] with shape [B, C, H, W]
        """
        # Map to [0, 1] for SSIM and PSNR
        fake_01 = (fake + 1.0) / 2.0
        real_01 = (real + 1.0) / 2.0
        
        # Clamp to avoid precision issues
        fake_01 = torch.clamp(fake_01, 0, 1)
        real_01 = torch.clamp(real_01, 0, 1)

        # SSIM
        ssim_val = ssim(fake_01, real_01, data_range=1.0, size_average=True).item()
        
        # PSNR
        mse = torch.mean((fake_01 - real_01) ** 2)
        if mse == 0:
            psnr_val = 100.0
        else:
            psnr_val = (10 * torch.log10(1.0 / mse)).item()
            
        # LPIPS (expects [-1, 1])
        lpips_val = self.lpips_vgg(fake, real).mean().item()
        
        # MAE
        mae_val = torch.mean(torch.abs(fake_01 - real_01)).item()
        
        return {
            "ssim": ssim_val,
            "psnr": psnr_val,
            "lpips": lpips_val,
            "mae": mae_val
        }

def compute_hu_metrics(fake_hu: np.ndarray, real_hu: np.ndarray) -> dict:
    """Compute metrics in Hounsfield Units (HU)."""
    mae = np.mean(np.abs(fake_hu - real_hu))
    rmse = np.sqrt(np.mean((fake_hu - real_hu)**2))
    return {"mae_hu": mae, "rmse_hu": rmse}
