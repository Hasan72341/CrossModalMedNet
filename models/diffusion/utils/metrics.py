import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict

try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

try:
    import lpips
    HAS_LPIPS = True
    _LPIPS_VGG = None
except ImportError:
    HAS_LPIPS = False
    _LPIPS_VGG = None

def compute_metrics(
    real: torch.Tensor, 
    fake: torch.Tensor, 
    mask: torch.Tensor | None = None,
    data_range: float = 2.0,
    device: torch.device = torch.device("cpu")
) -> Dict[str, float]:
    
    real = real.to(device)
    fake = fake.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    if real.shape[1] == 3:
        real_eval = real[:, 1:2, :, :]
        fake_eval = fake[:, 1:2, :, :]
        if mask is not None and mask.shape[1] == 3:
            mask_eval = mask[:, 1:2, :, :]
        else:
            mask_eval = mask
    else:
        real_eval = real
        fake_eval = fake
        mask_eval = mask

    if mask_eval is not None:
        # STRICTLY WITH MASK
        mask_bin = (mask_eval > 0.5).float()
        mask_sum = mask_bin.sum().clamp(min=1.0)
        
        # Masked MSE and MAE
        mse = (F.mse_loss(real_eval, fake_eval, reduction='none') * mask_bin).sum() / mask_sum
        mae = (F.l1_loss(real_eval, fake_eval, reduction='none') * mask_bin).sum() / mask_sum
        
        # Apply mask to images for PSNR, SSIM, LPIPS
        real_eval = real_eval * mask_bin
        fake_eval = fake_eval * mask_bin
        mse_val = mse.item()
        mae_val = mae.item()
    else:
        mse_val = F.mse_loss(real_eval, fake_eval).item()
        mae_val = F.l1_loss(real_eval, fake_eval).item()
        mask_bin = torch.ones_like(real_eval)

    metrics = {"mse": mse_val, "mae": mae_val}

    if mse_val > 1e-10:
        psnr = 10 * np.log10((data_range ** 2) / mse_val)
    else:
        psnr = 100.0
    metrics["psnr"] = psnr

    if HAS_TORCHMETRICS:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        metrics["ssim"] = ssim_metric(fake_eval, real_eval).item()
    else:
        metrics["ssim"] = 0.0
        
    if HAS_LPIPS:
        global _LPIPS_VGG
        if _LPIPS_VGG is None:
            _LPIPS_VGG = lpips.LPIPS(net="vgg").to(device)
        else:
            _LPIPS_VGG = _LPIPS_VGG.to(device)
        
        real_rgb = real_eval.expand(-1, 3, -1, -1)
        fake_rgb = fake_eval.expand(-1, 3, -1, -1)
        with torch.no_grad():
            metrics["lpips"] = _LPIPS_VGG(fake_rgb, real_rgb).mean().item()
    else:
        metrics["lpips"] = 0.0

    # Dice (pseudo-dice on foreground thresholded at 0)
    real_fg = (real_eval > 0).float() * mask_bin
    fake_fg = (fake_eval > 0).float() * mask_bin
    intersection = (real_fg * fake_fg).sum()
    union = real_fg.sum() + fake_fg.sum()
    dice = (2. * intersection / union.clamp(min=1e-8)).item()
    metrics["dice"] = dice
        
    return metrics

def get_error_map(real: torch.Tensor, fake: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if real.shape[1] == 3:
        real = real[:, 1:2, :, :]
        fake = fake[:, 1:2, :, :]
        if mask is not None and mask.shape[1] == 3:
            mask = mask[:, 1:2, :, :]
            
    err = torch.abs(real - fake)
    if mask is not None:
        err = err * mask
    return err
