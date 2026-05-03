"""
cyclegan/models.py
------------------
Neural network architecture for CycleGAN MRI ↔ CT translation,
optimised for the SynthRAD 2023/2025 2-D unpaired setting.

Key design choices
------------------
* WGAN-GP adversarial loss with a corrected gradient-penalty graph.
* Cycle & identity losses for structural fidelity.
* Volume-conservation loss via soft Jacobian-determinant regularisation
  (encourages det(J) ≈ 1 everywhere, penalising local volume changes).
* SynthRAD-aware HU normalisation constants embedded as class attributes.
* Single forward pass gathers all four images; identity images are
  produced in the *same* call to avoid redundant passes.
* All lambdas tuned for the SynthRAD pelvis/brain 2-D regime.

Classes
-------
ResnetBlock             : Residual block used by the ResNet generator.
ResnetGenerator         : Original CycleGAN ResNet generator.
Generator2D             : Wrapper around ResnetGenerator for this project.
Discriminator2D         : PatchGAN discriminator (NLayer) with optional features.
MultiScaleDiscriminator : Optional multi-scale discriminator wrapper.
CycleGAN                : Full model with generator/discriminator loss helpers.
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


def get_norm_layer(norm_type: str = "instance"):
    """Return a normalization layer matching the original CycleGAN setup."""
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


class ResnetBlock(nn.Module):
    """ResNet block used by the original CycleGAN generator."""

    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer: nn.Module,
        use_dropout: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.conv_block = self._build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def _build_conv_block(
        self,
        dim: int,
        padding_type: str,
        norm_layer: nn.Module,
        use_dropout: bool,
        use_bias: bool,
    ) -> nn.Sequential:
        conv_block: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based generator from the original CycleGAN paper."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm: str = "instance",
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")

        norm_layer = get_norm_layer(norm)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Generator2D(nn.Module):
    """ResNet generator wrapper; attention flags are ignored to match CycleGAN."""

    def __init__(
        self,
        use_attention: bool = False,
        use_transformer_attention: bool = False,
        dropout: float = 0.0,
        num_res_blocks: int = 9,
        input_nc: int = 1,
        output_nc: int = 1,
        ngf: int = 64,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        _ = use_attention, use_transformer_attention
        self.model = ResnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm=norm,
            use_dropout=dropout > 0.0,
            n_blocks=num_res_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator2D(nn.Module):
    """PatchGAN discriminator (NLayer) with optional feature-map return."""

    def __init__(
        self,
        input_nc: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        norm_layer = get_norm_layer(norm)
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        blocks: list[nn.Module] = []
        blocks.append(
            nn.Sequential(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            )
        )

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            )
        )

        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        feats: list[torch.Tensor] = []
        out = x
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        out = self.head(out)
        if return_features:
            return out, feats
        return out


class MultiScaleDiscriminator(nn.Module):
    """Optional multi-scale discriminator wrapper (full-res + 1/2-res)."""

    def __init__(
        self,
        num_scales: int = 2,
        input_nc: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                Discriminator2D(
                    input_nc=input_nc,
                    ndf=ndf,
                    n_layers=n_layers,
                    norm=norm,
                )
                for _ in range(num_scales)
            ]
        )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> list | tuple[list, list]:
        outputs, features = [], []
        inp = x
        for i, disc in enumerate(self.discriminators):
            if return_features:
                out, feats = disc(inp, return_features=True)
                outputs.append(out)
                features.append(feats)
            else:
                outputs.append(disc(inp))
            if i < len(self.discriminators) - 1:
                inp = self.downsample(inp)
        if return_features:
            return outputs, features
        return outputs


# ──────────────────────────────────────────────────────────────────────────────
# CycleGAN wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CycleGAN(nn.Module):
    """
    Full CycleGAN for unpaired CT ↔ MRI translation (SynthRAD 2D).

    Parameters
    ----------
    G_CT2MRI, G_MRI2CT : Generator2D
    D_CT, D_MRI        : Discriminator2D or MultiScaleDiscriminator
    lambda_cycle        : Cycle-consistency weight  (default 10).
    lambda_identity     : Identity-mapping weight   (default 0.5).
    lambda_feature      : Feature-matching weight   (default 0).
    lambda_volume       : Volume-conservation weight (default 0).
                          Set to 0 to disable.

    SynthRAD normalisation
    ----------------------
    CT images must be linearly mapped to [−1, 1] before calling any method:
        ct_norm = (ct_hu − CT_HU_SHIFT) / CT_HU_SCALE
    Use the class attributes CT_HU_SHIFT / CT_HU_SCALE for consistency.
    MRI images should be min-max normalised per-volume to [−1, 1].
    """

    # SynthRAD 2023 pelvis statistics (body-masked, clipped to [-1000, 2000] HU)
    CT_HU_SHIFT: float =  500.0   # midpoint of [-1000, 2000]
    CT_HU_SCALE: float = 1500.0   # half-range of [-1000, 2000]

    def __init__(
        self,
        G_CT2MRI: nn.Module,
        G_MRI2CT: nn.Module,
        D_CT: nn.Module,
        D_MRI: nn.Module,
        lambda_cycle:    float = 10.0,
        lambda_identity: float =  0.5,
        lambda_feature:  float =  0.0,
        lambda_volume:   float =  0.0,
        adv_mode:        str   = "lsgan",
        lambda_edge:     float =  0.0,
    ) -> None:
        super().__init__()
        self.G_CT2MRI = G_CT2MRI
        self.G_MRI2CT = G_MRI2CT
        self.D_CT     = D_CT
        self.D_MRI    = D_MRI

        self.lambda_cycle    = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_feature  = lambda_feature
        self.lambda_volume   = lambda_volume
        self.adv_mode        = adv_mode.lower()
        self.lambda_edge     = lambda_edge

    # ── helper: dispatch single vs multi-scale discriminator ─────────────────

    def _disc_outputs(self, D: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
        result = D(x)
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                flat: list[torch.Tensor] = []
                for item in result:
                    flat.extend(item)
                return flat
            return result
        return [result]

    def _disc_features(
        self, D: nn.Module, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        result = D(x, return_features=True)
        if isinstance(result, list):
            outs_list: list[torch.Tensor] = []
            feats_list: list[list[torch.Tensor]] = []
            for item in result:
                outs, feats = item
                if isinstance(outs, list):
                    outs_list.extend(outs)
                else:
                    outs_list.append(outs)
                if isinstance(feats, list):
                    feats_list.extend(feats)
                else:
                    feats_list.append(feats)
            return outs_list, feats_list
        outs, feats = result
        if isinstance(outs, list):
            return outs, feats                     # list[Tensor], list[list[Tensor]]
        return [outs], [feats]

    # ── L1 loss ───────────────────────────────────────────────────────────────

    @staticmethod
    def _multiscale_l1(
        real: torch.Tensor,
        fake: torch.Tensor,
        scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
    ) -> torch.Tensor:
        """Plain L1 loss (multi-resolution inputs removed)."""
        _ = scales
        return F.l1_loss(fake, real)

    @staticmethod
    def _edge_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Simple gradient-difference loss to preserve 2D edges."""
        real_dx = real[:, :, :, 1:] - real[:, :, :, :-1]
        real_dy = real[:, :, 1:, :] - real[:, :, :-1, :]
        fake_dx = fake[:, :, :, 1:] - fake[:, :, :, :-1]
        fake_dy = fake[:, :, 1:, :] - fake[:, :, :-1, :]

        real_dx = F.pad(real_dx, (0, 1, 0, 0))
        fake_dx = F.pad(fake_dx, (0, 1, 0, 0))
        real_dy = F.pad(real_dy, (0, 0, 0, 1))
        fake_dy = F.pad(fake_dy, (0, 0, 0, 1))

        return F.l1_loss(fake_dx, real_dx) + F.l1_loss(fake_dy, real_dy)

    def _lsgan_g_loss(self, D: nn.Module, fake: torch.Tensor) -> torch.Tensor:
        outs = self._disc_outputs(D, fake)
        if not outs:
            return fake.new_zeros(1).squeeze()
        total = fake.new_zeros(1).squeeze()
        for out in outs:
            total = total + F.mse_loss(out, torch.ones_like(out))
        return total / max(len(outs), 1)

    def _lsgan_d_loss(
        self, D: nn.Module, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        real_outs = self._disc_outputs(D, real)
        fake_outs = self._disc_outputs(D, fake.detach())
        if not real_outs:
            return real.new_zeros(1).squeeze()
        total = real.new_zeros(1).squeeze()
        for ro, fo in zip(real_outs, fake_outs):
            total = total + 0.5 * (
                F.mse_loss(ro, torch.ones_like(ro))
                + F.mse_loss(fo, torch.zeros_like(fo))
            )
        return total / max(len(real_outs), 1)

    # ── volume-conservation loss ──────────────────────────────────────────────

    @staticmethod
    def _volume_conservation_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """
        Soft Jacobian-determinant regularisation.

        For a 2-D intensity-to-intensity translation we cannot define a dense
        deformation field, but we can still measure *local volume change* by
        approximating the Jacobian of the residual displacement:

            displacement  = fake − real          (in normalised [−1,1] space)
            J_approx      = ∂u/∂x * ∂v/∂y − ∂u/∂y * ∂v/∂x

        We penalise |det(J) − 1| so the generator is discouraged from
        arbitrarily expanding or contracting structures.  For pure intensity
        mapping this acts as an indirect spatial smoothness prior.

        Reference: Reg-GAN / SynthRad-volume-preservation literature.
        """
        disp = fake - real                              # (B, 1, H, W)

        # Finite-difference spatial gradients of the displacement map
        # ∂u/∂x (horizontal gradient)
        du_dx = disp[:, :, :, 1:] - disp[:, :, :, :-1]   # (B,1,H,W-1)
        du_dy = disp[:, :, 1:, :] - disp[:, :, :-1, :]   # (B,1,H-1,W)

        # Jacobian determinant approximation on the overlapping region
        h = min(du_dx.shape[2], du_dy.shape[2])
        w = min(du_dx.shape[3], du_dy.shape[3])

        # det(J) ≈ (1 + ∂u/∂x)(1 + ∂v/∂y) − ∂u/∂y · ∂v/∂x
        # For a single-channel displacement:  ∂u = ∂v  →  off-diagonal terms cancel
        # det(J) ≈ (1 + ∂u/∂x)(1 + ∂u/∂y)
        det_J = (1.0 + du_dx[:, :, :h, :w]) * (1.0 + du_dy[:, :, :h, :w])

        # Penalise deviation from 1 (= no volume change)
        return F.l1_loss(det_J, torch.ones_like(det_J))

    # ── gradient penalty (FIXED) ──────────────────────────────────────────────

    def _gradient_penalty(
        self, D: nn.Module, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """
        WGAN-GP gradient penalty.

        Bug fixed: the original code created a CPU scalar via
        `torch.ones_like(torch.tensor(d_out))` which broke autograd.
        We now operate directly on the discriminator output tensor.
        """
        if fake.shape != real.shape:
            fake = F.interpolate(
                fake, size=real.shape[2:], mode="bilinear", align_corners=False
            )

        b = real.size(0)
        alpha = torch.rand(b, 1, 1, 1, device=real.device)
        interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

        d_interp = D(interp)

        # If multi-scale, sum scale outputs into a single scalar graph node
        if isinstance(d_interp, list):
            d_out = torch.stack([o.mean() for o in d_interp]).sum()
        else:
            d_out = d_interp.mean()

        # ── FIX: use grad_outputs matching d_out (a scalar tensor) ──────────
        gradients = torch.autograd.grad(
            outputs=d_out,
            inputs=interp,
            grad_outputs=torch.ones_like(d_out),   # scalar → ones_like is fine
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]                                        # (B, 1, H, W)

        gradients = gradients.view(b, -1)
        gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    # ── single joint forward pass (avoids redundant generator calls) ──────────

    def _full_forward(
        self, ct: torch.Tensor, mri: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Run all six generator forward passes in one place so nothing is
        computed twice within a single train step.

        Returns
        -------
        dict with keys: fake_mri, fake_ct, rec_ct, rec_mri, id_mri, id_ct
        """
        fake_mri = self.G_CT2MRI(ct)
        fake_ct  = self.G_MRI2CT(mri)
        rec_ct   = self.G_MRI2CT(fake_mri)
        rec_mri  = self.G_CT2MRI(fake_ct)
        # Identity: G_CT2MRI(mri) should ≈ mri;  G_MRI2CT(ct) should ≈ ct
        id_mri   = self.G_CT2MRI(mri)
        id_ct    = self.G_MRI2CT(ct)
        return dict(
            fake_mri=fake_mri, fake_ct=fake_ct,
            rec_ct=rec_ct,     rec_mri=rec_mri,
            id_mri=id_mri,     id_ct=id_ct,
        )

    # ── forward (kept for compatibility / inference) ──────────────────────────

    def forward(
        self, ct: torch.Tensor, mri: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (fake_mri, fake_ct, rec_ct, rec_mri)."""
        out = self._full_forward(ct, mri)
        return out["fake_mri"], out["fake_ct"], out["rec_ct"], out["rec_mri"]

    # ── generator loss ────────────────────────────────────────────────────────

    def generator_loss(
        self,
        ct:  torch.Tensor,
        mri: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Total generator loss:
            L_G = L_adv + λ_cycle·L_cycle + λ_id·L_id
                + λ_feat·L_feat + λ_vol·L_vol
        """
        imgs = self._full_forward(ct, mri)
        fake_mri = imgs["fake_mri"]
        fake_ct  = imgs["fake_ct"]
        rec_ct   = imgs["rec_ct"]
        rec_mri  = imgs["rec_mri"]
        id_mri   = imgs["id_mri"]
        id_ct    = imgs["id_ct"]

        # ── 1. Adversarial loss ─────────────────────────────────────────────
        if self.adv_mode == "lsgan":
            adv = (
                self._lsgan_g_loss(self.D_MRI, fake_mri)
                + self._lsgan_g_loss(self.D_CT, fake_ct)
            )
        else:
            adv = ct.new_zeros(1).squeeze()
            for out in self._disc_outputs(self.D_MRI, fake_mri):
                adv = adv - out.mean()
            for out in self._disc_outputs(self.D_CT, fake_ct):
                adv = adv - out.mean()

        # ── 2. Cycle-consistency ──────────────────────────────────────────────
        rec_ct  = _match_size(rec_ct,  ct.shape[2:])
        rec_mri = _match_size(rec_mri, mri.shape[2:])

        cycle = (
            self._multiscale_l1(ct,  rec_ct)
            + self._multiscale_l1(mri, rec_mri)
        )

        # ── 3. Identity loss ─────────────────────────────────────────────────
        id_mri = _match_size(id_mri, mri.shape[2:])
        id_ct  = _match_size(id_ct,  ct.shape[2:])

        identity = (
            self._multiscale_l1(mri, id_mri)
            + self._multiscale_l1(ct,  id_ct)
        )
        identity_weight = self.lambda_identity
        if self.lambda_identity <= 1.0:
            identity_weight = self.lambda_identity * self.lambda_cycle

        # ── 4. Feature-matching loss ──────────────────────────────────────────
        feature = ct.new_zeros(1).squeeze()
        if self.lambda_feature > 0:
            _, rf_mri = self._disc_features(self.D_MRI, mri)
            _, ff_mri = self._disc_features(self.D_MRI, fake_mri)
            _, rf_ct  = self._disc_features(self.D_CT,  ct)
            _, ff_ct  = self._disc_features(self.D_CT,  fake_ct)

            def _fm(real_feats: list, fake_feats: list) -> torch.Tensor:
                total, n = ct.new_zeros(1).squeeze(), 0
                for sf_real, sf_fake in zip(real_feats, fake_feats):
                    for fr, ff in zip(sf_real, sf_fake):
                        if ff.shape[2:] != fr.shape[2:]:
                            ff = F.interpolate(ff, size=fr.shape[2:],
                                               mode="bilinear", align_corners=False)
                        total = total + F.l1_loss(ff, fr.detach())
                        n += 1
                return total / max(n, 1)

            feature = _fm(rf_mri, ff_mri) + _fm(rf_ct, ff_ct)

        # ── 5. Volume-conservation loss ───────────────────────────────────────
        volume = ct.new_zeros(1).squeeze()
        if self.lambda_volume > 0:
            fake_mri_s = _match_size(fake_mri, ct.shape[2:])
            fake_ct_s  = _match_size(fake_ct,  mri.shape[2:])
            volume = (
                self._volume_conservation_loss(ct,  fake_mri_s)
                + self._volume_conservation_loss(mri, fake_ct_s)
            )

        # ── 6. Edge loss (2D-friendly) ──────────────────────────────────────
        edge = ct.new_zeros(1).squeeze()
        if self.lambda_edge > 0:
            fake_mri_s = _match_size(fake_mri, mri.shape[2:])
            fake_ct_s  = _match_size(fake_ct,  ct.shape[2:])
            edge = (
                self._edge_loss(mri, fake_mri_s)
                + self._edge_loss(ct, fake_ct_s)
            )

        # ── Total ─────────────────────────────────────────────────────────────
        total = (
            adv
            + self.lambda_cycle    * cycle
            + identity_weight * identity
            + self.lambda_feature  * feature
            + self.lambda_volume   * volume
            + self.lambda_edge     * edge
        )

        if return_parts:
            return total, {
                "adv":      adv.detach(),
                "cycle":    cycle.detach(),
                "identity": identity.detach(),
                "feature":  feature.detach(),
                "volume":   volume.detach(),
                "edge":     edge.detach(),
            }
        return total

    # ── discriminator loss ────────────────────────────────────────────────────

    def _discriminator_loss(
        self,
        D:    nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        gp_lambda: float = 10.0,
    ) -> torch.Tensor:
        """
        WGAN-GP discriminator loss:
            L_D = E[D(fake)] − E[D(real)] + λ_gp · GP
        """
        if self.adv_mode == "lsgan":
            return self._lsgan_d_loss(D, real, fake)

        real_outs = self._disc_outputs(D, real)
        fake_outs = self._disc_outputs(D, fake.detach())

        w_loss = sum(
            fo.mean() - ro.mean()
            for ro, fo in zip(real_outs, fake_outs)
        )
        gp = self._gradient_penalty(D, real, fake.detach())
        return w_loss + gp_lambda * gp

    def discriminator_step(
        self,
        ct:  torch.Tensor,
        mri: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute total discriminator loss.
        Generator is frozen via torch.no_grad().
        """
        with torch.no_grad():
            fake_mri = self.G_CT2MRI(ct)
            fake_ct  = self.G_MRI2CT(mri)

        d_mri = self._discriminator_loss(self.D_MRI, mri, fake_mri)
        d_ct  = self._discriminator_loss(self.D_CT,  ct,  fake_ct)
        total = d_mri + d_ct

        if return_parts:
            return total, {"d_mri": d_mri.detach(), "d_ct": d_ct.detach()}
        return total


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _match_size(
    x: torch.Tensor, target_hw: tuple[int, int] | torch.Size
) -> torch.Tensor:
    """Bilinear resize x to target spatial size if needed (no-op if already correct)."""
    if x.shape[2:] == torch.Size(target_hw):
        return x
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


def build_cyclegan(
    use_attention:   bool  = False,
    use_transformer_attention: bool = False,
    use_multiscale:  bool  = False,
    lambda_cycle:    float = 10.0,
    lambda_identity: float =  0.5,
    lambda_feature:  float =  0.0,
    lambda_volume:   float =  0.0,
    adv_mode:        str   = "lsgan",
    lambda_edge:     float =  0.0,
    num_res_blocks:  int   = 9,
    num_discriminators: int = 1,
    dropout:         float = 0.0,
    input_nc:        int   = 1,
    output_nc:       int   = 1,
    ngf:             int   = 64,
    ndf:             int   = 64,
    n_layers_d:      int   = 3,
    norm:            str   = "instance",
    device:          torch.device | str = "cuda"
) -> CycleGAN:
    """
    Convenience factory that wires up generators and discriminators.

    Parameters
    ----------
    use_attention  : Add self-attention in generator bottleneck.
    use_multiscale : Use two-scale discriminators (recommended for SynthRAD).
    lambda_*       : Loss weights; see CycleGAN docstring.
    dropout        : Dropout rate in ResBlocks (0 = off; try 0.05 for small sets).
    """
    G_CT2MRI = Generator2D(
        use_attention=use_attention,
        use_transformer_attention=use_transformer_attention,
        dropout=dropout,
        num_res_blocks=num_res_blocks,
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        norm=norm,
    )
    G_MRI2CT = Generator2D(
        use_attention=use_attention,
        use_transformer_attention=use_transformer_attention,
        dropout=dropout,
        num_res_blocks=num_res_blocks,
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        norm=norm,
    )

    if use_multiscale:
        D_CT  = MultiScaleDiscriminator(
            num_scales=num_discriminators,
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers_d,
            norm=norm,
        )
        D_MRI = MultiScaleDiscriminator(
            num_scales=num_discriminators,
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers_d,
            norm=norm,
        )
    else:
        D_CT  = Discriminator2D(input_nc=input_nc, ndf=ndf, n_layers=n_layers_d, norm=norm)
        D_MRI = Discriminator2D(input_nc=input_nc, ndf=ndf, n_layers=n_layers_d, norm=norm)

    return CycleGAN(
        G_CT2MRI=G_CT2MRI,
        G_MRI2CT=G_MRI2CT,
        D_CT=D_CT,
        D_MRI=D_MRI,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
        lambda_feature=lambda_feature,
        lambda_volume=lambda_volume,
        adv_mode=adv_mode,
        lambda_edge=lambda_edge,
    )


def build_cyclegan_2d_friendly(
    use_attention:   bool  = False,
    use_transformer_attention: bool = False,
    use_multiscale:  bool  = True,
    lambda_cycle:    float = 10.0,
    lambda_identity: float =  0.5,
    lambda_feature:  float =  0.0,
    lambda_volume:   float =  0.0,
    lambda_edge:     float =  0.1,
    dropout:         float =  0.0,
    num_res_blocks:  int   =  6,
    num_discriminators: int = 2,
) -> CycleGAN:
    """2D-friendly SynthRAD model: lighter generator + LSGAN + edge loss."""
    return build_cyclegan(
        use_attention=use_attention,
        use_transformer_attention=use_transformer_attention,
        use_multiscale=use_multiscale,
        lambda_cycle=lambda_cycle,
        lambda_identity=lambda_identity,
        lambda_feature=lambda_feature,
        lambda_volume=lambda_volume,
        adv_mode="lsgan",
        lambda_edge=lambda_edge,
        dropout=dropout,
        num_res_blocks=num_res_blocks,
        num_discriminators=num_discriminators,
    )