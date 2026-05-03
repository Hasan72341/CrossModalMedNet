import os
import sys
import torch
import json
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import argparse

# Allow relative imports if run as a module
from .configs.config import Settings, get_settings
from .datasets.data import create_dataloader
from .models.model import (
    VAEEncode,
    VAEDecode,
    forward_with_networks,
    initialize_unet,
    initialize_vae,
    load_lora_checkpoint,
    make_1step_sched,
)
from .utils.metrics import compute_metrics, get_error_map
from .utils.utils import save_batch_images

from transformers import AutoTokenizer, CLIPTextModel

def _prompt_to_emb(
    tokenizer: AutoTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return text_encoder(tokens.to(device))[0].detach()


def _center_slice(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x[:, 1:2, :, :]
    return x


def _compute_tissue_dice(
    real: torch.Tensor,
    fake: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    real_eval = _center_slice(real)
    fake_eval = _center_slice(fake)

    if mask is not None:
        mask_eval = (_center_slice(mask) > 0.5).float()
    else:
        mask_eval = torch.ones_like(real_eval)

    real_01 = torch.clamp((real_eval + 1.0) / 2.0, 0.0, 1.0) * mask_eval
    fake_01 = torch.clamp((fake_eval + 1.0) / 2.0, 0.0, 1.0) * mask_eval

    def _dice(min_v: float, max_v: float) -> float:
        real_bin = ((real_01 > min_v) & (real_01 <= max_v)).float() * mask_eval
        fake_bin = ((fake_01 > min_v) & (fake_01 <= max_v)).float() * mask_eval
        inter = (real_bin * fake_bin).sum()
        union = real_bin.sum() + fake_bin.sum()
        return (2.0 * inter / union).item() if union.item() > 0 else 1.0

    return {
        "dice_soft_tissue": _dice(0.0, 0.3),
        "dice_dense_tissue": _dice(0.3, 1.0),
    }

def run_patient_eval(settings: Settings, checkpoint_path: str, output_root: Path, split: str = "val", gpu_id: int = 0) -> dict:
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    scheduler = make_1step_sched(settings.base_model, device)

    unet = initialize_unet(settings.base_model, settings.lora_rank_unet, add_lora=False)
    vae = initialize_vae(settings.base_model, settings.lora_rank_vae, add_lora=False)
    
    vae_enc, vae_dec, _ = load_lora_checkpoint(
        checkpoint_path,
        unet,
        vae,
        device,
    )

    unet.to(device)
    vae.to(device)
    vae_enc.to(device)
    vae_dec.to(device)
    unet.eval()
    vae.eval()
    vae_enc.eval()
    vae_dec.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        settings.base_model,
        subfolder="tokenizer",
        use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        settings.base_model,
        subfolder="text_encoder",
    ).to(device)
    text_encoder.requires_grad_(False)

    fixed_tgt_emb = _prompt_to_emb(tokenizer, text_encoder, settings.prompt_target, device)

    loader = create_dataloader(settings, unpaired=False, split=split, shuffle=False, drop_last=False, augment=False)
    
    checkpoint_name = Path(checkpoint_path).stem
    out_dir = output_root / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)

    src_label = settings.source_modality.upper()
    tgt_label = settings.target_modality.upper()

    # patient_metrics[patient_id][metric_name] = [values_per_slice]
    patient_metrics = defaultdict(lambda: defaultdict(list))
    
    print(f"Evaluating {checkpoint_name} on {len(loader.dataset)} samples...")

    for batch in tqdm(loader, desc=f"Eval {checkpoint_name}"):
        img_src = batch["pixel_values_src"].to(device)
        img_tgt = batch["pixel_values_tgt"].to(device)
        img_mask = batch.get("mask")
        if img_mask is not None:
            img_mask = img_mask.to(device)
        
        meta = batch.get("meta", {})
        patient_ids = meta.get("id", ["unknown"] * img_src.shape[0])

        bsz = img_src.shape[0]
        timesteps = torch.tensor(
            [scheduler.config.num_train_timesteps - 1] * bsz,
            device=device,
        ).long()

        with torch.no_grad():
            fake_tgt = forward_with_networks(
                img_src,
                vae_enc,
                unet,
                vae_dec,
                scheduler,
                timesteps,
                fixed_tgt_emb.repeat(bsz, 1, 1),
            )

        # Compute Metrics slice by slice in the batch
        for i in range(bsz):
            p_id = patient_ids[i]
            mask_i = img_mask[i:i+1] if img_mask is not None else None
            # Since compute_metrics expects [B, C, H, W], we unsqueeze
            m_src_to_tgt = compute_metrics(
                img_tgt[i:i+1], fake_tgt[i:i+1], 
                mask=mask_i,
                device=device
            )
            d_src_to_tgt = _compute_tissue_dice(img_tgt[i:i+1], fake_tgt[i:i+1], mask=mask_i)

            for k, v in {**m_src_to_tgt, **d_src_to_tgt}.items():
                patient_metrics[p_id][f"{src_label}_to_{tgt_label}/{k}"].append(v)

    # Aggregate patient-wise
    patient_summaries = {}
    for p_id, metrics in patient_metrics.items():
        p_summary = {}
        for k, v in metrics.items():
            p_summary[k] = float(np.mean(v))
        patient_summaries[p_id] = p_summary

    # Save patient-wise CSV
    pd.DataFrame.from_dict(patient_summaries, orient="index").to_csv(out_dir / "patient_metrics.csv")

    # Compute global mean of patient means
    global_metrics = defaultdict(list)
    for p_id, p_summary in patient_summaries.items():
        for k, v in p_summary.items():
            global_metrics[k].append(v)
    
    final_summary = {}
    for k, v in global_metrics.items():
        final_summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

    with open(out_dir / "summary.json", "w") as f:
        pd.DataFrame(final_summary).T.to_csv(out_dir / "summary.csv")

    return final_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pix2pix_diffusion_standard")
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--subset", type=int, default=None, help="Evaluate only every N-th checkpoint")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--epoch_interval", type=int, default=1)
    args = parser.parse_args()

    # Set CUDA device
    gpu_id = args.gpu_id if args.gpu_id is not None else args.rank
    
    settings = get_settings()
    
    checkpoints = sorted(list(Path(args.checkpoint_dir).glob("*.pt")))
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    # Filter by epoch interval
    if args.epoch_interval > 1:
        checkpoints = checkpoints[args.epoch_interval - 1 :: args.epoch_interval]

    # Distribute by rank
    if args.num_gpus > 1:
        checkpoints = checkpoints[args.rank :: args.num_gpus]

    if args.subset:
        checkpoints = checkpoints[::args.subset]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    history = []
    history_file = output_root / f"patient_evaluation_history_rank{args.rank}.csv"

    for cp in checkpoints:
        res = run_patient_eval(settings, str(cp), output_root, split=args.split, gpu_id=gpu_id)
        
        # Flatten for CSV
        row = {"checkpoint": cp.name}
        try:
            step = int(cp.stem.split("_")[1])
            row["step"] = step
        except:
            row["step"] = 0
            
        for k, v in res.items():
            row[k] = v["mean"]
        history.append(row)
        
        # Incremental save
        pd.DataFrame(history).to_csv(history_file, index=False)
        print(f"Updated history: {history_file}")

    print("\n" + "="*50)
    print("FINAL PATIENT-WISE EVALUATION HISTORY")
    print("="*50)
    print(pd.DataFrame(history).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
