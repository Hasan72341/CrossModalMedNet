"""
pix2pix_standard/train.py
------------------------
Training loop and checkpoint helpers for the Pix2Pix CT → MRI model.
Mirroring the CycleGAN implementation style.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.utils as vutils
try:
    import wandb
except ImportError:
    wandb = None

from tqdm import tqdm
from torchvision.utils import save_image
from torch.optim import lr_scheduler

from .config import Settings, DataConfig, TrainingConfig
from .data import create_dataloaders
from .models import Pix2Pix, build_pix2pix, init_weights
from .utils import adapt_state_dict, unwrap, save_checkpoint, load_checkpoint
from .evaluation import run_evaluation, save_test_visuals


def build_model(
    settings: Settings,
    device: torch.device,
    use_multi_gpu: bool,
) -> tuple[Pix2Pix, torch.optim.Adam, torch.optim.Adam]:
    """
    Instantiate generator + discriminator, wrap with DataParallel if needed,
    and create Adam optimisers.
    """
    input_nc = getattr(settings, "input_nc", 1)
    output_nc = getattr(settings, "output_nc", 1)
    
    model = build_pix2pix(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=settings.ngf,
        ndf=settings.ndf,
        n_layers_d=settings.n_layers_d,
        norm=settings.norm,
        use_dropout=settings.use_dropout,
        lambda_l1=settings.lambda_l1,
        lambda_identity=settings.lambda_identity,
        gan_mode=settings.gan_mode,
    ).to(device)
    
    init_type = getattr(settings, "init_type", "normal")
    init_gain = float(getattr(settings, "init_gain", 0.02))
    init_weights(model, init_type=init_type, init_gain=init_gain)
    model.train()

    if use_multi_gpu:
        model.G_CT2MRI = nn.DataParallel(model.G_CT2MRI)
        model.D_MRI = nn.DataParallel(model.D_MRI)

    opt_G = torch.optim.Adam(
        model.G_CT2MRI.parameters(),
        lr=settings.lr_g,
        betas=(settings.beta1, settings.beta2),
    )
    opt_D = torch.optim.Adam(
        model.D_MRI.parameters(),
        lr=settings.lr_d * settings.lr_d_scale,
        betas=(settings.beta1, settings.beta2),
    )

    return model, opt_G, opt_D


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


def train(settings: Settings, device: torch.device) -> None:
    """
    Run the Pix2Pix training loop.
    """
    from torchmetrics import MeanMetric

    is_distributed, rank, local_rank, world_size, ddp_device = _setup_distributed()
    if is_distributed:
        device = ddp_device
    is_main = rank == 0

    # Reproducibility
    seed = settings.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_multi_gpu = (num_gpus > 1) and not is_distributed
    
    if is_main:
        print(f"device: {device} | num_gpus: {num_gpus} | multi_gpu: {use_multi_gpu}")

    # Data
    loader, _, test_loader = create_dataloaders(settings, settings)
    if is_main:
        print(f"Dataset size: {len(loader.dataset)} | loader length: {len(loader)}")

    # Model + Optimisers
    model, opt_G, opt_D = build_model(settings, device, use_multi_gpu)
    model_ref = model

    # Paths
    ckpt_dir = Path(settings.checkpoint_dir)
    log_dir = Path(settings.log_dir)
    eval_dir = Path(settings.eval_dir)
    visuals_dir = Path(settings.visuals_dir)
    
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
    
    latest_ckpt = ckpt_dir / "latest.pth"
    
    run_name = f"pix2pix_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # (Old evaluation directories removed as they are now handled above)

    resume_epoch = 0
    global_step = 0

    if latest_ckpt.exists():
        resume_epoch, global_step, loaded_run_name = load_checkpoint(
            latest_ckpt, model_ref, opt_G, opt_D, device, use_multi_gpu
        )
        run_name = loaded_run_name
        if is_main:
            print(f"Resuming from epoch {resume_epoch}, step {global_step}, run={run_name}")

    # W&B
    wandb_run = None
    if settings.use_wandb and is_main:
        if wandb is None:
            print("Warning: wandb not installed. Disabling wandb logging.")
        else:
            wandb_run = wandb.init(
                project=settings.wandb_project_name,
                entity=settings.wandb_entity,
                name=run_name,
                config=settings.model_dump() if hasattr(settings, "model_dump") else settings.dict(),
                mode=settings.wandb_mode,
                dir=str(settings.project_root),
            )

    # TensorBoard
    writer = None
    if settings.use_tensorboard and is_main:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir))

    # Schedulers
    total_epochs = settings.num_epochs
    n_epochs = settings.n_epochs
    n_epochs_decay = settings.n_epochs_decay

    def linear_lr_lambda(ep: int) -> float:
        if ep < n_epochs:
            return 1.0
        return max(0.0, 1.0 - (ep - n_epochs) / float(n_epochs_decay + 1))

    scheduler_G = lr_scheduler.LambdaLR(opt_G, lr_lambda=linear_lr_lambda)
    scheduler_D = lr_scheduler.LambdaLR(opt_D, lr_lambda=linear_lr_lambda)

    g_loss_mean = MeanMetric().to(device) if is_main else None
    d_loss_mean = MeanMetric().to(device) if is_main else None

    max_steps = settings.max_steps_per_epoch
    last_epoch = resume_epoch

    try:
        for epoch in range(resume_epoch, total_epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", disable=not is_main)
            model.train()
            
            for i, batch in enumerate(pbar):
                if max_steps > 0 and i >= max_steps:
                    break

                ct, real_mri, mask, meta = batch
                ct = ct.to(device, non_blocking=True)
                real_mri = real_mri.to(device, non_blocking=True)

                # ── Generator step ────────────────────────────────────────────
                model_ref.D_MRI.requires_grad_(False)
                opt_G.zero_grad(set_to_none=True)
                g_loss, g_parts = model_ref.generator_loss(ct, real_mri, return_parts=True)
                g_loss.backward()
                opt_G.step()

                # ── Discriminator step ────────────────────────────────────────
                model_ref.D_MRI.requires_grad_(True)
                opt_D.zero_grad(set_to_none=True)
                
                with torch.no_grad():
                    fake_mri = model_ref.G_CT2MRI(ct)
                
                d_loss = model_ref._discriminator_loss(model_ref.D_MRI, ct, real_mri, fake_mri)
                d_loss.backward()
                opt_D.step()

                # Logging
                if is_main:
                    g_loss_mean.update(g_loss)
                    d_loss_mean.update(d_loss)
                    pbar.set_postfix({"G": f"{g_loss.item():.4f}", "D": f"{d_loss.item():.4f}"})

                    if global_step % settings.log_every == 0:
                        if writer:
                            writer.add_scalar("Loss/G_total", g_loss.item(), global_step)
                            writer.add_scalar("Loss/D_total", d_loss.item(), global_step)
                            for k, v in g_parts.items():
                                writer.add_scalar(f"Loss/G_{k}", v.item(), global_step)
                        if wandb_run and wandb:
                            wandb.log({
                                "loss/G_total": g_loss.item(),
                                "loss/D_total": d_loss.item(),
                                **{f"loss/G_{k}": v.item() for k, v in g_parts.items()}
                            }, step=global_step)

                    if global_step % settings.image_log_every == 0:
                        with torch.no_grad():
                            vis_fake = (fake_mri[0:1] + 1) / 2
                            vis_real = (real_mri[0:1] + 1) / 2
                            vis_ct = (ct[0:1] + 1) / 2
                            
                        save_image(vis_fake, visuals_dir / f"step_{global_step}_fake.png")
                        if writer:
                            writer.add_image("Images/Fake", vis_fake[0], global_step)
                            writer.add_image("Images/Real", vis_real[0], global_step)
                            writer.add_image("Images/CT", vis_ct[0], global_step)
                        if wandb_run and wandb:
                            wandb.log({
                                "images/fake": wandb.Image(vis_fake[0].cpu().numpy()),
                                "images/real": wandb.Image(vis_real[0].cpu().numpy()),
                                "images/ct": wandb.Image(vis_ct[0].cpu().numpy()),
                            }, step=global_step)

                global_step += 1

            last_epoch = epoch + 1
            scheduler_G.step()
            scheduler_D.step()

            if is_main:
                # Per-epoch summary
                if writer:
                    writer.add_scalar("Epoch/G_loss", g_loss_mean.compute().item(), epoch)
                if wandb_run and wandb:
                    wandb.log({"epoch/G_loss": g_loss_mean.compute().item()}, step=global_step)
                g_loss_mean.reset()
                d_loss_mean.reset()

                # Evaluation
                if last_epoch % settings.eval_visuals_every_epochs == 0:
                    save_test_visuals(model_ref, test_loader, device, visuals_dir, last_epoch)
                if last_epoch % settings.eval_metrics_every_epochs == 0:
                    run_evaluation(model_ref, test_loader, device, eval_dir, last_epoch)

                # Save
                if last_epoch % settings.save_every_epochs == 0:
                    save_checkpoint(
                        ckpt_dir / f"epoch_{last_epoch}.pth",
                        last_epoch, global_step, run_name, model_ref, opt_G, opt_D
                    )

    except KeyboardInterrupt:
        if is_main:
            print("Interrupted. Saving...")
            save_checkpoint(ckpt_dir / "interrupt.pth", last_epoch, global_step, run_name, model_ref, opt_G, opt_D)
    finally:
        if is_main:
            save_checkpoint(ckpt_dir / "final.pth", last_epoch, global_step, run_name, model_ref, opt_G, opt_D)
            if writer: writer.close()
            if wandb_run: wandb_run.finish()
        if is_distributed: dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    if args.num_epochs: settings.num_epochs = args.num_epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(settings, device)
