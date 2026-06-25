"""
cyclegan_pelvis/train.py
------------------------
Training loop and checkpoint helpers for the CycleGAN MRI ↔ CT model.

Entry points
------------
build_model()       Build model + optimisers and optionally wrap with DataParallel.
load_checkpoint()   Resume from a ``.pth`` file produced by ``save_checkpoint()``.
save_checkpoint()   Persist model + optimiser states to disk.
train()             Full training loop; call from ``main.py``.
"""
from __future__ import annotations
import argparse

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

import os
import random
import time
from datetime import datetime
from pathlib import Path

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from torch.optim import lr_scheduler

from .config import Settings, DataConfig, TrainingConfig
from .data import create_dataloaders
from .models import CycleGAN, build_cyclegan, build_cyclegan_2d_friendly
from .utils import adapt_state_dict, unwrap, save_checkpoint, load_checkpoint
from .evaluation import run_evaluation, save_test_visuals


def _init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] > 3:
        return x[:, :3, :, :]
    return x[:, :1, :, :].repeat(1, 3, 1, 1)


def _match_spatial(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if x.shape[2:] == target_hw:
        return x
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


def _settings_to_dict(settings: Settings) -> dict:
    if hasattr(settings, "model_dump"):
        return settings.model_dump()
    return settings.dict()
# ──────────────────────────────────────────────────────────────────────────────
# Model + optimiser construction
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    settings: Settings,
    device: torch.device,
    use_multi_gpu: bool,
) -> tuple[CycleGAN, torch.optim.Adam, torch.optim.Adam]:
    """
    Instantiate generators + discriminators, wrap with DataParallel if needed,
    and create Adam optimisers.

    Returns
    -------
    model, opt_G, opt_D
    """
    model_variant = getattr(settings, "model_variant", "standard").lower()
    gan_mode = getattr(settings, "gan_mode", "lsgan")
    input_nc = getattr(settings, "input_nc", 1)
    output_nc = getattr(settings, "output_nc", 1)
    dropout = 0.5 if getattr(settings, "use_dropout", False) else 0.0

    if model_variant in ("2d", "2d_friendly", "synthrad_2d"):
        model = build_cyclegan_2d_friendly(
            use_attention=settings.use_attention,
            use_transformer_attention=settings.use_transformer_attention,
            use_multiscale=settings.use_multiscale,
            lambda_cycle=settings.lambda_cycle,
            lambda_identity=settings.lambda_identity,
            lambda_feature=settings.lambda_feature,
            lambda_volume=settings.lambda_volume,
            lambda_edge=settings.lambda_edge,
            num_res_blocks=6,
            num_discriminators=settings.num_discriminators,
            dropout=dropout,
        ).to(device)
    else:
        model = build_cyclegan(
            use_attention=settings.use_attention,
            use_transformer_attention=settings.use_transformer_attention,
            use_multiscale=settings.use_multiscale,
            lambda_cycle=settings.lambda_cycle,
            lambda_identity=settings.lambda_identity,
            lambda_feature=settings.lambda_feature,
            lambda_volume=settings.lambda_volume,
            lambda_edge=settings.lambda_edge,
            adv_mode=gan_mode,
            num_res_blocks=9,
            num_discriminators=settings.num_discriminators,
            dropout=dropout,
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=settings.ngf,
            ndf=settings.ndf,
            n_layers_d=settings.n_layers_d,
            norm=settings.norm,
        ).to(device)
    init_type = getattr(settings, "init_type", "normal")
    init_gain = float(getattr(settings, "init_gain", 0.02))
    _init_weights(model, init_type=init_type, init_gain=init_gain)
    model.train()

    if use_multi_gpu:
        model.G_CT2MRI = nn.DataParallel(model.G_CT2MRI)
        model.G_MRI2CT = nn.DataParallel(model.G_MRI2CT)
        model.D_CT = nn.DataParallel(model.D_CT)
        model.D_MRI = nn.DataParallel(model.D_MRI)

    opt_G = torch.optim.Adam(
        list(model.G_CT2MRI.parameters()) + list(model.G_MRI2CT.parameters()),
        lr=settings.lr_g,
        betas=(settings.beta1, settings.beta2),
    )
    opt_D = torch.optim.Adam(
        list(model.D_CT.parameters()) + list(model.D_MRI.parameters()),
        lr=settings.lr_d * settings.lr_d_scale,
        betas=(settings.beta1, settings.beta2),
    )

    return model, opt_G, opt_D


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    run_name: str,
    model: CycleGAN,
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
    model: CycleGAN,
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


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────
def get_lr_lambda(epoch):
    """
    Returns a multiplier for the learning rate:
    - 1.0 for epochs 0-25 (Constant)
    - Linearly decreases to 0.0 for epochs 26-50 (Decay)
    """
    n_epochs = 50
    decay_start_epoch = 25
    
    if epoch < decay_start_epoch:
        return 1.0
    else:
        # Linearly decay from 1.0 to 0.0
        return 1.0 - (epoch - decay_start_epoch) / (n_epochs - decay_start_epoch)
def _parse_patch_size(value: str | None) -> tuple[int, int, int] | None:
    if not value or not value.strip():
        return None
    raw = value.replace("x", ",").replace(" ", "")
    parts = [int(p) for p in raw.split(",") if p]
    if len(parts) != 3:
        raise ValueError(f"PATCH_SIZE must have 3 values, got: {value}")
    return parts[0], parts[1], parts[2]


class ImagePool:
    """Replay buffer for GAN stabilization (CycleGAN paper)."""

    def __init__(self, size: int = 50) -> None:
        self.size = size
        self.pool: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size <= 0:
            return images
        out = []
        for img in images:
            img = img.detach().unsqueeze(0)
            if len(self.pool) < self.size:
                self.pool.append(img)
                out.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.size - 1)
                tmp = self.pool[idx]
                self.pool[idx] = img
                out.append(tmp)
            else:
                out.append(img)
        return torch.cat(out, dim=0)


def _setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return False, rank, local_rank, world_size, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return True, rank, local_rank, world_size, device


# Deprecated: _run_periodic_eval is replaced by direct calls in the training loop.


def train(settings: Settings, device: torch.device) -> None:
    """
    Run the CycleGAN training loop.

    Reads all hyper-parameters from ``settings``, resumes from the latest
    checkpoint if one exists, and writes TensorBoard logs + per-epoch
    checkpoints.
    """
    from torchmetrics import MeanMetric  # optional heavy import

    # ── distributed setup ───────────────────────────────────────────────────
    is_distributed, rank, local_rank, world_size, ddp_device = _setup_distributed()
    if is_distributed:
        device = ddp_device
    is_main = rank == 0

    # ── reproducibility ───────────────────────────────────────────────────────
    seed = settings.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if settings.seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── device / multi-GPU ───────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_multi_gpu = (num_gpus > 1) and not is_distributed
    if is_main:
        print(
            f"device: {device}  |  num_gpus: {num_gpus}  |  "
            f"multi_gpu: {use_multi_gpu}  |  distributed: {is_distributed}"
        )

    # ── data ──────────────────────────────────────────────────────────────────
    # ── data ──────────────────────────────────────────────────────────────────
    # The new data loader uses DataConfig and TrainingConfig (inherited by Settings)
    loader, val_loader, test_loader = create_dataloaders(settings, settings)
    if is_main:
        print(
            f"Dataset size: {len(loader.dataset)}  |  loader length: {len(loader)}  |  "
            f"Test size: {len(test_loader.dataset) if test_loader else 0}"
        )

    # ── model + optimisers ────────────────────────────────────────────────────
    model, opt_G, opt_D = build_model(settings, device, use_multi_gpu)
    ddp_model = model
    if is_distributed:
        ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    model_ref = ddp_model.module if is_distributed else ddp_model

    # ── checkpoint dir ────────────────────────────────────────────────────────
    gan_mode = getattr(settings, "gan_mode", "lsgan")
    module_name = Path(__file__).resolve().parent.name
    ckpt_dir = Path(settings.checkpoint_dir) / gan_mode
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pth"
    
    run_tag = f"{settings.data_mode}_{settings.slice_axis}_{gan_mode}"
    run_name = f"{module_name}_{run_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Evaluation directories
    eval_dir = Path(settings.log_dir).parent / "eval"
    visuals_dir = Path(settings.log_dir).parent / "visuals"
    if is_main:
        eval_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)

    resume_epoch = 0
    global_step = 0

    if latest_ckpt.exists():
        resume_epoch, global_step, loaded_run_name = load_checkpoint(
            latest_ckpt, model_ref, opt_G, opt_D, device, use_multi_gpu
        )
        run_name = loaded_run_name
        if is_main:
            print(f"Resuming from epoch {resume_epoch}, step {global_step}, run={run_name}")
    else:
        if is_main:
            print(f"No checkpoint found at {latest_ckpt}. Starting fresh.")

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if settings.use_wandb and is_main:
        config_dict = _settings_to_dict(settings)
        config_dict.update({
            "run_name": run_name,
            "gan_mode": gan_mode,
            "module": module_name,
        })
        wandb_run = wandb.init(
            project=settings.wandb_project_name,
            entity=settings.wandb_entity,
            name=run_name,
            tags=[settings.data_mode, settings.slice_axis, gan_mode],
            config=config_dict,
            mode=settings.wandb_mode,
            dir=str(Path(settings.log_dir).parent),
        )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if settings.use_tensorboard and is_main:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(settings.log_dir)

    # ── output images dir ─────────────────────────────────────────────────────
    module_root = Path(__file__).resolve().parent
    output_dir = module_root / "outputs"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── schedulers ───────────────────────────────────────────────────────────
    n_epochs = getattr(settings, "n_epochs", None)
    n_epochs_decay = getattr(settings, "n_epochs_decay", None)
    if n_epochs is None or n_epochs_decay is None:
        total_epochs = int(settings.num_epochs)
        if n_epochs is None:
            n_epochs = max(1, total_epochs // 2)
        if n_epochs_decay is None:
            n_epochs_decay = max(0, total_epochs - n_epochs)
    else:
        total_epochs = int(n_epochs) + int(n_epochs_decay)
    n_epochs = int(n_epochs)
    n_epochs_decay = int(n_epochs_decay)

    def linear_lr_lambda(ep: int) -> float:
        if ep < n_epochs:
            return 1.0
        if n_epochs_decay <= 0:
            return 1.0
        return max(0.0, 1.0 - (ep - n_epochs) / float(n_epochs_decay + 1))

    scheduler_G = lr_scheduler.LambdaLR(opt_G, lr_lambda=linear_lr_lambda)
    scheduler_D = lr_scheduler.LambdaLR(opt_D, lr_lambda=linear_lr_lambda)

    # ── torchmetrics ─────────────────────────────────────────────────────────
    g_loss_mean = MeanMetric().to(device) if settings.use_torchmetrics and is_main else None
    d_loss_mean = MeanMetric().to(device) if settings.use_torchmetrics and is_main else None

    # ── LPIPS (VGG16) ────────────────────────────────────────────────────────
    lpips_weight = float(getattr(settings, "lambda_lpips", 0.0))
    lpips_model = None
    if lpips_weight > 0:
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad_(False)

    # ── replay buffers ───────────────────────────────────────────────────────
    pool_size = getattr(settings, "pool_size", 50)
    pool_mri = ImagePool(size=pool_size)
    pool_ct = ImagePool(size=pool_size)

    max_steps = settings.max_steps_per_epoch if settings.max_steps_per_epoch is not None else -1
    last_epoch = resume_epoch

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        for epoch in range(resume_epoch, resume_epoch + total_epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", disable=not is_main)
            for i, batch in enumerate(pbar):
                if max_steps > 0 and i >= max_steps:
                    break

                # Hierarchical Data Loader returns (ct, mri, mask, meta)
                if isinstance(batch, (list, tuple)):
                    ct, mri, mask, _ = batch
                else:
                    ct, mri, mask = batch["ct"], batch["mri"], batch["mask"]
                
                ct = ct.to(device, non_blocking=True)
                mri = mri.to(device, non_blocking=True)
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)

                # ── Normalisation ──────────────────────────────────────────
                # (Strict [-1, 1] normalization is now handled in data.py)
                pass

                if i == 0 and epoch == resume_epoch and is_main:
                    print(f"ct range: {ct.min().item():.3f}..{ct.max().item():.3f}")
                    print(f"mri range: {mri.min().item():.3f}..{mri.max().item():.3f}")

                # ── Generator step ────────────────────────────────────────────
                model_ref.D_CT.requires_grad_(False)
                model_ref.D_MRI.requires_grad_(False)
                opt_G.zero_grad(set_to_none=True)
                g_loss, g_parts = model_ref.generator_loss(ct, mri, return_parts=True)
                if lpips_model is not None:
                    fake_mri_lp = _to_lpips_input(model_ref.G_CT2MRI(ct))
                    fake_ct_lp = _to_lpips_input(model_ref.G_MRI2CT(mri))
                    real_mri_lp = _to_lpips_input(mri)
                    real_ct_lp = _to_lpips_input(ct)
                    fake_mri_lp = _match_spatial(fake_mri_lp, real_mri_lp.shape[2:])
                    fake_ct_lp = _match_spatial(fake_ct_lp, real_ct_lp.shape[2:])
                    lpips_loss = (
                        lpips_model(fake_mri_lp, real_mri_lp).mean()
                        + lpips_model(fake_ct_lp, real_ct_lp).mean()
                    )
                    g_loss = g_loss + lpips_weight * lpips_loss
                    g_parts["lpips"] = lpips_loss.detach()
                g_loss.backward()
                opt_G.step()

                with torch.no_grad():
                    fake_mri = model_ref.G_CT2MRI(ct)
                    fake_ct = model_ref.G_MRI2CT(mri)

                # ── Save sample images ────────────────────────────────────────
                if i % 200 == 0 and is_main:

                    save_image((fake_mri[:, 0:1, :, :] + 1) / 2, output_dir / f"epoch{epoch}_iter{i}_fake_mri.png")
                    save_image((fake_ct[:, 0:1, :, :] + 1) / 2, output_dir / f"epoch{epoch}_iter{i}_fake_ct.png")
                    save_image((ct[:, 0:1, :, :] + 1) / 2, output_dir / f"epoch{epoch}_iter{i}_real_ct.png")
                    save_image((mri[:, 0:1, :, :] + 1) / 2, output_dir / f"epoch{epoch}_iter{i}_real_mri.png")

                    if writer is not None:
                        def norm(x: torch.Tensor) -> torch.Tensor:
                            return (x + 1) / 2

                        writer.add_image("Images/Fake_MRI", norm(fake_mri[0, 0:1, :, :]), global_step)
                        writer.add_image("Images/Fake_CT", norm(fake_ct[0, 0:1, :, :]), global_step)
                        writer.add_image("Images/Real_CT", norm(ct[0, 0:1, :, :]), global_step)
                        writer.add_image("Images/Real_MRI", norm(mri[0, 0:1, :, :]), global_step)

                        vis_ct = norm(ct[:2, 0:1, :, :])
                        vis_fake_mri = norm(fake_mri[:2, 0:1, :, :])
                        vis_mri = norm(mri[:2, 0:1, :, :])
                        vis_fake_ct = norm(fake_ct[:2, 0:1, :, :])

                        target_size = vis_ct.shape[2:]
                        if vis_fake_mri.shape[2:] != target_size:
                            vis_fake_mri = torch.nn.functional.interpolate(
                                vis_fake_mri,
                                size=target_size,
                                mode="bilinear",
                                align_corners=False,
                            )
                        if vis_mri.shape[2:] != target_size:
                            vis_mri = torch.nn.functional.interpolate(
                                vis_mri,
                                size=target_size,
                                mode="bilinear",
                                align_corners=False,
                            )
                        if vis_fake_ct.shape[2:] != target_size:
                            vis_fake_ct = torch.nn.functional.interpolate(
                                vis_fake_ct,
                                size=target_size,
                                mode="bilinear",
                                align_corners=False,
                            )

                        grid = vutils.make_grid(
                            torch.cat(
                                [
                                    vis_ct,
                                    vis_fake_mri,
                                    vis_mri,
                                    vis_fake_ct,
                                ],
                                dim=0,
                            ),
                            nrow=2,
                        )
                        writer.add_image("Comparison/CT_to_MRI_to_CT", grid, global_step)

                # ── Discriminator step ────────────────────────────────────────
                do_d_step = (i % max(1, settings.d_update_every)) == 0
                d_results = None
                if do_d_step:
                    model_ref.D_CT.requires_grad_(True)
                    model_ref.D_MRI.requires_grad_(True)
                    opt_D.zero_grad(set_to_none=True)
                    
                    # WGAN-GP requires fresh fakes and possibly pooled fakes
                    with torch.no_grad():
                        fake_mri = model_ref.G_CT2MRI(ct)
                        fake_ct = model_ref.G_MRI2CT(mri)
                    
                    q_fake_mri = pool_mri.query(fake_mri)
                    q_fake_ct = pool_ct.query(fake_ct)
                    
                    # Use the correct API name _discriminator_loss
                    d_mri = model_ref._discriminator_loss(model_ref.D_MRI, mri, q_fake_mri)
                    d_ct = model_ref._discriminator_loss(model_ref.D_CT, ct, q_fake_ct)
                    d_loss = d_mri + d_ct
                    d_loss.backward()
                    opt_D.step()
                    d_results = (d_loss, d_mri, d_ct)

                if settings.use_torchmetrics and g_loss_mean is not None:
                    g_loss_mean.update(g_loss)
                    # Only log the discriminator loss on iterations where D was actually
                    # updated (d_update_every > 1 skips D steps); otherwise d_loss is stale.
                    if d_results is not None:
                        d_loss_mean.update(d_results[0])  # type: ignore[union-attr]

                    if is_main:
                        pbar.set_postfix({
                            "G": f"{g_loss.item():.4f}",
                            "D": f"{d_results[0].item():.4f}" if d_results is not None else "—",
                        })

                    if writer is not None:
                        writer.add_scalar("Loss/Generator/Total", g_loss.item(), global_step)
                        for k, v in g_parts.items():
                            writer.add_scalar(f"Loss/Generator/{k}", v.item(), global_step)

                        if d_results:
                            d_loss_val, d_mri_val, d_ct_val = d_results
                            writer.add_scalar("Loss/Discriminator/Total", d_loss_val.item(), global_step)
                            writer.add_scalar("Loss/Discriminator/d_ct", d_ct_val.item(), global_step)
                            writer.add_scalar("Loss/Discriminator/d_mri", d_mri_val.item(), global_step)

                        writer.add_scalar("LR/Generator", opt_G.param_groups[0]["lr"], global_step)
                        writer.add_scalar("LR/Discriminator", opt_D.param_groups[0]["lr"], global_step)

                if wandb_run is not None:
                    log_payload = {
                        "loss/G_total": g_loss.item(),
                        "lr/G": opt_G.param_groups[0]["lr"],
                        "lr/D": opt_D.param_groups[0]["lr"],
                    }
                    for key, value in g_parts.items():
                        log_payload[f"loss/G_{key}"] = value.item()
                    if d_results:
                        d_loss_val, d_mri_val, d_ct_val = d_results
                        log_payload.update({
                            "loss/D_total": d_loss_val.item(),
                            "loss/D_mri": d_mri_val.item(),
                            "loss/D_ct": d_ct_val.item(),
                        })
                    wandb.log(log_payload, step=global_step)

                if writer is not None and global_step % 500 == 0:
                    for name, param in model_ref.named_parameters():
                        writer.add_histogram(f"Weights/{name}", param, global_step)
                        if param.grad is not None:
                            writer.add_histogram(f"Gradients/{name}", param.grad, global_step)

                if wandb_run is not None and i % 200 == 0:
                    def _norm_vis(x: torch.Tensor) -> torch.Tensor:
                        return (x + 1) / 2

                    wandb.log({
                        "images/fake_mri": wandb.Image(_norm_vis(fake_mri[0, 0:1]).detach().cpu().numpy()),
                        "images/fake_ct": wandb.Image(_norm_vis(fake_ct[0, 0:1]).detach().cpu().numpy()),
                        "images/real_ct": wandb.Image(_norm_vis(ct[0, 0:1]).detach().cpu().numpy()),
                        "images/real_mri": wandb.Image(_norm_vis(mri[0, 0:1]).detach().cpu().numpy()),
                    }, step=global_step)

                global_step += 1

            last_epoch = epoch + 1  # mark completed

            if writer is not None and g_loss_mean is not None:
                writer.add_scalar("Epoch/Generator_Loss_Mean", g_loss_mean.compute().item(), epoch)
                writer.add_scalar("Epoch/Discriminator_Loss_Mean", d_loss_mean.compute().item(), epoch)
            if wandb_run is not None and g_loss_mean is not None:
                wandb.log({
                    "epoch": last_epoch,
                    "epoch/G_loss_mean": g_loss_mean.compute().item(),
                    "epoch/D_loss_mean": d_loss_mean.compute().item(),
                }, step=global_step)

            scheduler_G.step()
            scheduler_D.step()

            # ── Evaluation ──────────────────────────────────────────────────
            if is_main:
                # Qualitative visuals every epoch
                if settings.eval_visuals_every_epochs > 0 and last_epoch % settings.eval_visuals_every_epochs == 0:
                    if test_loader:
                        save_test_visuals(model_ref, test_loader, device, visuals_dir, last_epoch)
                
                # Quantitative metrics every 5 epochs
                if settings.eval_metrics_every_epochs > 0 and last_epoch % settings.eval_metrics_every_epochs == 0:
                    if test_loader:
                        # run_evaluation handles mask application internally during its loop over test_loader
                        stats = run_evaluation(model_ref, test_loader, device, eval_dir, last_epoch)
                        if writer is not None:
                            for k, v in stats.items():
                                if isinstance(v, (int, float)):
                                    writer.add_scalar(f"Eval/{k}", v, last_epoch)
                        if wandb_run is not None:
                            eval_payload = {f"eval/{k}": v for k, v in stats.items() if isinstance(v, (int, float))}
                            eval_payload["epoch"] = last_epoch
                            wandb.log(eval_payload, step=global_step)

            if settings.save_every_epochs > 0 and last_epoch % settings.save_every_epochs == 0 and is_main:
                p = save_checkpoint(
                    ckpt_dir / f"{run_name}_epoch{last_epoch}.pth",
                    last_epoch, global_step, run_name, model_ref, opt_G, opt_D,
                )
                print(f"Saved checkpoint: {p}")

    except KeyboardInterrupt:
        if is_main:
            p = save_checkpoint(
                ckpt_dir / f"{run_name}_interrupt.pth",
                last_epoch, global_step, run_name, model_ref, opt_G, opt_D,
            )
            print(f"Interrupted. Saved: {p}")

    finally:
        if is_main:
            p = save_checkpoint(
                ckpt_dir / f"{run_name}_final.pth",
                last_epoch, global_step, run_name, model_ref, opt_G, opt_D,
            )
            print(f"Saved final checkpoint: {p}")
            if writer is not None:
                writer.flush()
                writer.close()
            if wandb_run is not None:
                wandb_run.finish()
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN Pelvis Training")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--n_epochs_decay", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--eval_metrics_every", type=int, default=None)
    parser.add_argument("--eval_visuals_every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load settings
    settings = Settings()
    if args.num_epochs is not None:
        settings.num_epochs = args.num_epochs
    if args.n_epochs is not None:
        settings.n_epochs = args.n_epochs
    if args.n_epochs_decay is not None:
        settings.n_epochs_decay = args.n_epochs_decay
    if args.max_steps_per_epoch is not None:
        settings.max_steps_per_epoch = args.max_steps_per_epoch
    if args.eval_metrics_every is not None:
        settings.eval_metrics_every_epochs = args.eval_metrics_every
    if args.eval_visuals_every is not None:
        settings.eval_visuals_every_epochs = args.eval_visuals_every
    if args.device is not None:
        settings.device = args.device.lower()

    # Device setup
    if settings.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(settings.device)

    # Start training
    train(settings, device)
