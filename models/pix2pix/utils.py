"""
pix2pix_standard/utils.py
------------------------
Small helper utilities shared across train / eval scripts.
Mirroring the CycleGAN implementation style.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import time
from datetime import datetime
from pathlib import Path


def unwrap(m: nn.Module) -> nn.Module:
    """Strip ``nn.DataParallel`` wrapper if present."""
    return m.module if isinstance(m, nn.DataParallel) else m


def adapt_state_dict(state_dict: dict, use_multi_gpu: bool) -> dict:
    """
    Convert a checkpoint's model ``state_dict`` between single-GPU and
    multi-GPU (DataParallel) key formats.
    """
    prefixes = ("G_CT2MRI.", "D_MRI.")

    out: dict = {}

    if use_multi_gpu:
        for k, v in state_dict.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p) and not nk.startswith(p + "module."):
                    nk = p + "module." + nk[len(p):]
                    break
            out[nk] = v
    else:
        for k, v in state_dict.items():
            nk = (
                k.replace("G_CT2MRI.module.", "G_CT2MRI.")
                 .replace("D_MRI.module.", "D_MRI.")
            )
            out[nk] = v

    return out


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    run_name: str,
    model: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
) -> Path:
    """
    Save model + optimiser states to ``path``.
    """
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "timestamp": time.time(),
        "run_name": run_name,
        "G_CT2MRI_state": unwrap(model.G_CT2MRI).state_dict(),
        "D_MRI_state": unwrap(model.D_MRI).state_dict(),
        "opt_G_state": opt_G.state_dict(),
        "opt_D_state": opt_D.state_dict(),
    }
    torch.save(payload, path)
    torch.save(payload, path.parent / "latest.pth")
    return path


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    device: torch.device,
    use_multi_gpu: bool,
) -> tuple[int, int, str]:
    """
    Load states from ``ckpt_path`` into ``model``, ``opt_G``, ``opt_D``.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if "G_CT2MRI_state" in ckpt:
        unwrap(model.G_CT2MRI).load_state_dict(ckpt["G_CT2MRI_state"])
        unwrap(model.D_MRI).load_state_dict(ckpt["D_MRI_state"])
    else:
        # Legacy/Full model format
        adapted = adapt_state_dict(ckpt.get("model_state", ckpt), use_multi_gpu)
        model.load_state_dict(adapted, strict=False)

    if "opt_G_state" in ckpt:
        opt_G.load_state_dict(ckpt["opt_G_state"])
    if "opt_D_state" in ckpt:
        opt_D.load_state_dict(ckpt["opt_D_state"])

    resume_epoch = int(ckpt.get("epoch", 0))
    resume_global_step = int(ckpt.get("global_step", 0))
    run_name = str(ckpt.get("run_name", f"pix2pix_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    return resume_epoch, resume_global_step, run_name
