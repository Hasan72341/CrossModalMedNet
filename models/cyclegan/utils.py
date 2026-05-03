"""
cyclegan/utils.py
-----------------
Small helper utilities shared across train / eval scripts.
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

    Single → Multi
        ``G_CT2MRI.layer`` → ``G_CT2MRI.module.layer``

    Multi → Single
        ``G_CT2MRI.module.layer`` → ``G_CT2MRI.layer``
    """
    prefixes = ("G_CT2MRI.", "G_MRI2CT.", "D_CT.", "D_MRI.")

    out: dict = {}

    if use_multi_gpu:
        # old single-GPU checkpoint → load into multi-GPU model
        for k, v in state_dict.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p) and not nk.startswith(p + "module."):
                    nk = p + "module." + nk[len(p):]
                    break
            out[nk] = v
    else:
        # old multi-GPU checkpoint → load into single-GPU model
        for k, v in state_dict.items():
            nk = (
                k.replace("G_CT2MRI.module.", "G_CT2MRI.")
                 .replace("G_MRI2CT.module.", "G_MRI2CT.")
                 .replace("D_CT.module.", "D_CT.")
                 .replace("D_MRI.module.", "D_MRI.")
            )
            out[nk] = v

    return out


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    run_name: str,
    model: nn.Module,  # Using nn.Module here for flexibility, or CycleGAN if we want to be specific
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
) -> Path:
    """
    Save model + optimiser states to ``path``.
    Also writes ``latest.pth`` alongside ``path``.
    """
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "timestamp": time.time(),
        "run_name": run_name,
        "G_CT2MRI_state": unwrap(model.G_CT2MRI).state_dict(),
        "G_MRI2CT_state": unwrap(model.G_MRI2CT).state_dict(),
        "D_CT_state": unwrap(model.D_CT).state_dict(),
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

    Returns
    -------
    (resume_epoch, resume_global_step, run_name)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if "G_CT2MRI_state" in ckpt:
        # Modern checkpoint format — individual component states
        unwrap(model.G_CT2MRI).load_state_dict(ckpt["G_CT2MRI_state"])
        unwrap(model.G_MRI2CT).load_state_dict(ckpt["G_MRI2CT_state"])
        unwrap(model.D_CT).load_state_dict(ckpt["D_CT_state"])
        unwrap(model.D_MRI).load_state_dict(ckpt["D_MRI_state"])
    else:
        # Legacy format — full model state dict
        adapted = adapt_state_dict(ckpt["model_state"], use_multi_gpu)
        missing, unexpected = model.load_state_dict(adapted, strict=False)
        print(f"Legacy checkpoint load → missing: {len(missing)}, unexpected: {len(unexpected)}")

    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])

    resume_epoch = int(ckpt.get("epoch", 0))
    resume_global_step = int(ckpt.get("global_step", 0))
    run_name = str(ckpt.get("run_name", f"cyclegan_brain_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    return resume_epoch, resume_global_step, run_name
