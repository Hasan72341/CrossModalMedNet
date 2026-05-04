"""
pix2pix_mri_to_ct/models.py
-------------------------
Neural network architecture for Pix2Pix MRI → CT translation.
Mask-aware implementation with LSGAN (MSE) loss.
"""
from __future__ import annotations

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_norm_layer(norm_type: str = "batch"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "syncbatch":
        return functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        def _identity(_: int) -> nn.Module:
            return Identity()
        return _identity
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
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
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


# ──────────────────────────────────────────────────────────────────────────────
# Pix2Pix U-Net Generator
# ──────────────────────────────────────────────────────────────────────────────

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = (norm_layer.func == nn.InstanceNorm2d) if isinstance(norm_layer, functools.partial) else (norm_layer == nn.InstanceNorm2d)
        if input_nc is None:
            input_nc = outer_nc
        downconv  = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu  = nn.LeakyReLU(0.2, True)
        downnorm  = norm_layer(inner_nc)
        uprelu    = nn.ReLU(True)
        upnorm    = norm_layer(outer_nc)
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model  = [downconv] + [submodule] + [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model  = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model  = [downrelu, downconv, downnorm] + [submodule] + [uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, submodule=None,
                                              norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf*8, ngf*8, input_nc=None, submodule=unet_block,
                                                  norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf*4, ngf*8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf*2, ngf*4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf,   ngf*2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                              outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# PatchGAN Discriminator
# ──────────────────────────────────────────────────────────────────────────────

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = (norm_layer.func == nn.InstanceNorm2d) if isinstance(norm_layer, functools.partial) else (norm_layer == nn.InstanceNorm2d)
        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2,
                                   padding=padw, bias=use_bias),
                         norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=1,
                               padding=padw, bias=use_bias),
                     norm_layer(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────────────────────────────────────
# Pix2Pix Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class Pix2Pix(nn.Module):
    def __init__(
        self,
        G_MRI2CT: nn.Module,
        D_CT: nn.Module,
        lambda_l1: float = 100.0,
        lambda_identity: float = 0.0,
        gan_mode: str = "lsgan",
    ) -> None:
        super().__init__()
        self.G_MRI2CT = G_MRI2CT
        self.D_CT = D_CT
        self.lambda_l1 = lambda_l1
        self.lambda_identity = lambda_identity
        self.gan_mode = gan_mode.lower()
        
        # Standard GAN using BCEWithLogitsLoss can exceed [0, 1].
        # LSGAN using MSELoss is naturally bounded within [0, 1] for targets [0, 1].
        if self.gan_mode == "vanilla":
            self.criterion_gan = nn.BCEWithLogitsLoss()
        elif self.gan_mode == "lsgan":
            self.criterion_gan = nn.MSELoss()
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] not implemented")
            
        self.criterion_l1  = nn.L1Loss()

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        return self.G_MRI2CT(mri)

    def generator_loss(
        self,
        mri: torch.Tensor,
        real_ct: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fake_ct = self.G_MRI2CT(mri)
        
        # 1. Adversarial loss (Discriminator sees MRI + Fake CT)
        pred_fake = self.D_CT(torch.cat([mri, fake_ct], dim=1))
        # Note: LSGAN with target 1.0 is in [0, 1]
        loss_g_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
        
        # 2. L1 loss
        loss_g_l1 = self.criterion_l1(fake_ct, real_ct)
        
        # 3. Identity loss (optional)
        loss_g_id = torch.tensor(0.0, device=mri.device)
        if self.lambda_identity > 0:
            loss_g_id = self.criterion_l1(self.G_MRI2CT(real_ct), real_ct)

        total_loss = loss_g_gan + (self.lambda_l1 * loss_g_l1) + (self.lambda_identity * loss_g_id)
        
        parts = {
            "gan": loss_g_gan.detach(),
            "l1": loss_g_l1.detach(),
            "identity": loss_g_id.detach() if self.lambda_identity > 0 else loss_g_id
        }
        
        if return_parts:
            return total_loss, parts
        return total_loss

    def _discriminator_loss(
        self,
        D: nn.Module,
        mri: torch.Tensor,
        real_ct: torch.Tensor,
        fake_ct: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adversarial loss for the discriminator.
        For LSGAN, loss = 0.5 * (MSE(pred_real, 1) + MSE(pred_fake, 0))
        This loss value resides in the range [0, 0.5] if outputs are in [0, 1].
        """
        # Real
        pred_real = D(torch.cat([mri, real_ct], dim=1))
        loss_d_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))
        
        # Fake
        pred_fake = D(torch.cat([mri, fake_ct.detach()], dim=1))
        loss_d_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        return (loss_d_real + loss_d_fake) * 0.5


def build_pix2pix(
    input_nc: int = 1,
    output_nc: int = 1,
    ngf: int = 64,
    ndf: int = 64,
    n_layers_d: int = 3,
    norm: str = "batch",
    use_dropout: bool = True,
    lambda_l1: float = 100.0,
    lambda_identity: float = 0.0,
    gan_mode: str = "lsgan",
) -> Pix2Pix:
    norm_layer = get_norm_layer(norm)
    G = UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=8, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    D = NLayerDiscriminator(input_nc=input_nc + output_nc, ndf=ndf, n_layers=n_layers_d, norm_layer=norm_layer)
    return Pix2Pix(G, D, lambda_l1=lambda_l1, lambda_identity=lambda_identity, gan_mode=gan_mode)
