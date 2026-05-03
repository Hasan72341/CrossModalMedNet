from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def normalize_tensor(
    x: torch.Tensor,
    method: str = "minmax",
    shift: float = 0.0,
    scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = x.float()
    if method == "shift_scale":
        x = (x - shift) / (scale + eps)
        return torch.clamp(x, -1.0, 1.0)
    min_val = x.min()
    max_val = x.max()
    if (max_val - min_val) < eps:
        return x * 0.0 - 1.0
    x = (x - min_val) / (max_val - min_val + eps)
    return x * 2.0 - 1.0


def ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.shape[1] == 1:
            x = x.squeeze(1)
    if x.dim() == 3:
        c = x.shape[0]
        if c == 1:
            x = x.repeat(3, 1, 1)
        elif c == 2:
            x = torch.cat([x, x[:1]], dim=0)
        elif c > 3:
            center = c // 2
            start = max(0, center - 1)
            if start + 3 > c:
                start = c - 3
            x = x[start : start + 3]
        elif c == 3:
            x = x
    elif x.dim() == 2:
        x = x.unsqueeze(0).repeat(3, 1, 1)
    else:
        raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")
    return x


def maybe_resize(x: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    if image_size <= 0 or mode == "none":
        return x
    if mode == "center_crop":
        return TF.center_crop(x, [image_size, image_size])
    return TF.resize(x, [image_size, image_size], interpolation=InterpolationMode.BICUBIC)


def prepare_output_dirs(output_dir: str, checkpoint_dir: str, log_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1) * 0.5, 0.0, 1.0)


def save_batch_images(
    base_dir: Path,
    tag: str,
    images: torch.Tensor,
    step: int,
    max_items: int,
    save_25d_slices: bool,
) -> list[Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    tag_dir = base_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    if max_items <= 0:
        count = images.shape[0]
    else:
        count = min(images.shape[0], max_items)
    images_cpu = images.detach().cpu()

    for idx in range(count):
        img = images_cpu[idx]
        out_path = tag_dir / f"step_{step:07d}_{idx:03d}.png"

        if img.dim() == 3 and img.shape[0] == 3:
            vis_img = img[1:2]
        else:
            vis_img = img

        save_image(_to_01(vis_img), out_path)
        saved_paths.append(out_path)

        if save_25d_slices and img.dim() == 3 and img.shape[0] == 3:
            for slice_idx in range(3):
                slice_dir = tag_dir / f"slice_{slice_idx}"
                slice_dir.mkdir(parents=True, exist_ok=True)
                slice_path = slice_dir / f"step_{step:07d}_{idx:03d}.png"
                save_image(_to_01(img[slice_idx : slice_idx + 1]), slice_path)
                saved_paths.append(slice_path)

    return saved_paths


def save_lora_checkpoint(
    path: Path,
    unet: torch.nn.Module,
    vae_enc: torch.nn.Module,
    vae_dec: torch.nn.Module,
    unet_modules: dict,
    vae_lora_modules: list[str],
    lora_rank_unet: int,
    lora_rank_vae: int,
) -> None:
    from peft.utils import get_peft_model_state_dict

    payload = {
        "l_target_modules_encoder": unet_modules["encoder"],
        "l_target_modules_decoder": unet_modules["decoder"],
        "l_modules_others": unet_modules["others"],
        "rank_unet": lora_rank_unet,
        "sd_unet": get_peft_model_state_dict(unet, adapter_name="default"),
        "rank_vae": lora_rank_vae,
        "vae_lora_target_modules": vae_lora_modules,
        "sd_vae_enc": vae_enc.state_dict(),
        "sd_vae_dec": vae_dec.state_dict(),
    }
    torch.save(payload, path)

def save_comparison_grid(
    eval_dir: Path,
    epoch: int,
    step: int,
    real_src: torch.Tensor,
    fake_tgt: torch.Tensor,
    real_tgt: torch.Tensor,
    src_label: str,
    tgt_label: str,
) -> None:
    import matplotlib.pyplot as plt
    comp_dir = eval_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    def to_numpy(x):
        x = x[0].detach().cpu()
        if x.shape[0] == 3:
            x = x[1]
        else:
            x = x[0]
        return (x.numpy() + 1) / 2

    images = [to_numpy(real_src), to_numpy(fake_tgt), to_numpy(real_tgt)]
    titles = [f"Real {src_label}", f"Fake {tgt_label}", f"Real {tgt_label}"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(comp_dir / f"comparison_step_{step:07d}.png", dpi=150)
    plt.close(fig)
