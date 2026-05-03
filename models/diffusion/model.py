from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig


class VAEEncode(nn.Module):
    def __init__(self, vae: AutoencoderKL) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor


class VAEDecode(nn.Module):
    def __init__(self, vae: AutoencoderKL) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        decoded = self.vae.decode(x / self.vae.config.scaling_factor).sample
        return decoded.clamp(-1, 1)


def make_1step_sched(base_model: str, device: torch.device) -> DDPMScheduler:
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    scheduler.set_timesteps(1, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return scheduler


def _my_vae_encoder_fwd(self, sample: torch.Tensor) -> torch.Tensor:
    sample = self.conv_in(sample)
    blocks = []
    for down_block in self.down_blocks:
        blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = blocks
    return sample


def _my_vae_decoder_fwd(self, sample: torch.Tensor, latent_embeds: torch.Tensor | None = None) -> torch.Tensor:
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def initialize_unet(
    base_model: str,
    rank: int,
    return_lora_module_names: bool = False,
    add_lora: bool = True,
) -> tuple[UNet2DConditionModel, list[str], list[str], list[str]] | UNet2DConditionModel:
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder: list[str] = []
    l_target_modules_decoder: list[str] = []
    l_modules_others: list[str] = []
    patterns = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv_out",
        "proj_out",
        "proj_in",
        "ff.net.2",
        "ff.net.0.proj",
    ]
    for name, _ in unet.named_parameters():
        if "bias" in name or "norm" in name:
            continue
        for pattern in patterns:
            if pattern in name and ("down_blocks" in name or "conv_in" in name):
                l_target_modules_encoder.append(name.replace(".weight", ""))
                break
            if pattern in name and "up_blocks" in name:
                l_target_modules_decoder.append(name.replace(".weight", ""))
                break
            if pattern in name:
                l_modules_others.append(name.replace(".weight", ""))
                break

    if add_lora:
        all_target_modules = l_target_modules_encoder + l_target_modules_decoder + l_modules_others
        lora_conf = LoraConfig(
            r=rank,
            init_lora_weights="gaussian",
            target_modules=all_target_modules,
            lora_alpha=rank,
        )
        unet.add_adapter(lora_conf, adapter_name="default")
        unet.set_adapters(["default"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    return unet


def initialize_vae(
    base_model: str,
    rank: int,
    return_lora_module_names: bool = False,
    add_lora: bool = True,
) -> tuple[AutoencoderKL, list[str]] | AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = _my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = _my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
    vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
    vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False)
    vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
    nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "skip_conv_1",
        "skip_conv_2",
        "skip_conv_3",
        "skip_conv_4",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]
    if add_lora:
        vae_lora_config = LoraConfig(
            r=rank,
            init_lora_weights="gaussian",
            target_modules=l_vae_target_modules,
        )
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    return vae


def forward_with_networks(
    x: torch.Tensor,
    vae_enc: VAEEncode,
    unet: UNet2DConditionModel,
    vae_dec: VAEDecode,
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    text_emb: torch.Tensor,
) -> torch.Tensor:
    batch = x.shape[0]
    x_enc = vae_enc(x).to(x.dtype)
    model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb).sample
    x_out = scheduler.step(model_pred, timesteps, x_enc, return_dict=True).prev_sample
    return vae_dec(x_out)


def get_trainable_params(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen_ids = set()

    def add_unique(new_params: Iterable[nn.Parameter]) -> None:
        for p in new_params:
            if id(p) not in seen_ids:
                params.append(p)
                seen_ids.add(id(p))

    add_unique(unet.conv_in.parameters())
    unet.conv_in.requires_grad_(True)
    unet.set_adapters(["default"])
    for name, param in unet.named_parameters():
        if "lora" in name and "default" in name:
            add_unique([param])

    for name, param in vae.named_parameters():
        if "lora" in name and "vae_skip" in name:
            add_unique([param])
    add_unique(vae.decoder.skip_conv_1.parameters())
    add_unique(vae.decoder.skip_conv_2.parameters())
    add_unique(vae.decoder.skip_conv_3.parameters())
    add_unique(vae.decoder.skip_conv_4.parameters())

    return params


def load_lora_checkpoint(
    checkpoint_path: str,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    device: torch.device,
) -> tuple[VAEEncode, VAEDecode, dict]:
    sd = torch.load(checkpoint_path, map_location=device)

    all_target_modules = sd["l_target_modules_encoder"] + sd["l_target_modules_decoder"] + sd["l_modules_others"]
    lora_conf = LoraConfig(
        r=sd["rank_unet"],
        init_lora_weights="gaussian",
        target_modules=all_target_modules,
        lora_alpha=sd["rank_unet"],
    )
    unet.add_adapter(lora_conf, adapter_name="default")
    unet.load_state_dict(sd.get("sd_unet", {}), strict=False)
    unet.set_adapters(["default"])

    vae_lora_config = LoraConfig(
        r=sd["rank_vae"],
        init_lora_weights="gaussian",
        target_modules=sd["vae_lora_target_modules"],
    )
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    vae.decoder.gamma = 1
    vae_enc = VAEEncode(vae)
    vae_dec = VAEDecode(vae)
    vae_enc.load_state_dict(sd["sd_vae_enc"], strict=False)
    vae_dec.load_state_dict(sd["sd_vae_dec"], strict=False)

    return vae_enc, vae_dec, sd
