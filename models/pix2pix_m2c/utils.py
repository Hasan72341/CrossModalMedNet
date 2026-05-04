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
    prefixes = ("G_MRI2CT.", "D_CT.")

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
                k.replace("G_MRI2CT.module.", "G_MRI2CT.")
                 .replace("D_CT.module.", "D_CT.")
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
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "run_name": run_name,
        "G_MRI2CT_state": unwrap(model.G_MRI2CT).state_dict(),
        "D_CT_state": unwrap(model.D_CT).state_dict(),
        "opt_G_state": opt_G.state_dict(),
        "opt_D_state": opt_D.state_dict(),
    }
    torch.save(checkpoint, path)
    torch.save(checkpoint, path.parent / "latest.pth")
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

    if "G_MRI2CT_state" in ckpt:
        unwrap(model.G_MRI2CT).load_state_dict(ckpt["G_MRI2CT_state"])
    if "D_CT_state" in ckpt:
        unwrap(model.D_CT).load_state_dict(ckpt["D_CT_state"])
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
