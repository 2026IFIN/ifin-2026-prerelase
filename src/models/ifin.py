from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn


from utils.operators import gaus_t, generate_roi


def get_num_groups(channels: int) -> int:
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0:
            return num_groups
    return 1


class ConvG(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None, num_groups: int | None = None):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.single_conv(x)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class RB(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int | None = None, drop_prob: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()

        self.rec_norm1 = nn.GroupNorm(1, out_ch)
        self.rec_pw1 = nn.Conv2d(out_ch, out_ch, 1)
        self.rec_dw = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch)
        self.rec_pw2 = nn.Conv2d(out_ch, out_ch * 2, 1)
        self.rec_sg1 = SimpleGate()
        self.rec_se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_ch, out_ch, 1))
        self.rec_conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.rec_drop1 = nn.Dropout(drop_prob)
        self.rec_beta = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        self.rec_norm2 = nn.GroupNorm(1, out_ch)
        self.rec_pw3 = nn.Conv2d(out_ch, out_ch * 2, 1)
        self.rec_sg2 = SimpleGate()
        self.rec_pw4 = nn.Conv2d(out_ch, out_ch, 1)
        self.rec_drop2 = nn.Dropout(drop_prob)
        self.rec_gamma = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        y = self.rec_norm1(x)
        y = self.rec_pw1(y)
        y = self.rec_dw(y)
        y = self.rec_pw2(y)
        y = self.rec_sg1(y)
        y = y * self.rec_se(x)
        y = self.rec_conv3(y)
        y = self.rec_drop1(y)
        y = x + self.rec_beta * y

        z = self.rec_norm2(y)
        z = self.rec_pw3(z)
        z = self.rec_sg2(z)
        z = self.rec_pw4(z)
        z = self.rec_drop2(z)
        x = y + self.rec_gamma * z

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x


class ISO(nn.Module):
    """Inverse System Operator."""

    def __init__(self, channels: int, height: int, width: int, psf_height: int, psf_width: int, k: int = 16):
        super().__init__()
        self.psf_weights = nn.Parameter(torch.ones(k, channels, 1, 1) * 0.01)
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.alpha = nn.Parameter(torch.ones(k, 1, 1, 1) * 1, requires_grad=True)
        self.kernel_weights = nn.Parameter(generate_roi(k, height, width), requires_grad=False)
        self.relu = nn.ReLU()
        self.k = k

    def forward(self, measurement: torch.Tensor, psf: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        _, _, img_h, img_w = measurement.shape
        _, _, psf_h, psf_w = psf.shape

        psf = psf.reshape(self.k, -1, psf.size(-2), psf.size(-1))
        psf_sum = psf.sum(dim=(-2, -1), keepdim=True)
        psf_normalized = psf / (psf_sum.abs() * self.alpha + 1e-12)

        measurement_padded = F.pad(
            measurement,
            (psf_w // 2, psf_w - psf_w // 2, psf_h // 2, psf_h - psf_h // 2),
            mode="replicate",
        )
        measurement_padded = gaus_t(measurement_padded, fwhm=2)
        psf_padded = F.pad(psf_normalized, (img_w // 2, img_w - img_w // 2, img_h // 2, img_h - img_h // 2), mode="constant")

        measurement_fft = torch.fft.rfft2(measurement_padded, dim=(-2, -1))
        psf_fft = torch.fft.rfft2(psf_padded, s=(measurement_padded.size(-2), measurement_padded.size(-1)), dim=(-2, -1))

        spectral_reg = self.relu(self.psf_weights)
        wiener_filter = psf_fft.conj() / (psf_fft.abs() ** 2 + epsilon + spectral_reg)
        inverse_fft = measurement_fft.unsqueeze(0) * wiener_filter.unsqueeze(1)

        inverse_spatial = torch.fft.irfft2(inverse_fft, dim=(-2, -1))
        inverse_spatial = torch.fft.ifftshift(inverse_spatial, dim=(-2, -1))
        crop_h_start = psf_h // 2
        crop_w_start = psf_w // 2
        inverse_crop = inverse_spatial[..., crop_h_start : crop_h_start + img_h, crop_w_start : crop_w_start + img_w]
        roi_weights = self.kernel_weights.unsqueeze(1).unsqueeze(1)
        inverse_crop = (inverse_crop * roi_weights).sum(dim=0)
        return self.group_norm(inverse_crop.real)


class FSO(nn.Module):
    """Forward System Operator."""

    def __init__(self, in_channels: int, height: int, width: int, psf_height: int, psf_width: int, init_scale: float = 1.0, k: int = 16):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1) * 1)
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.k = k

    def forward(self, image: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        _, _, img_h, img_w = image.shape
        _, _, psf_h, psf_w = psf.shape

        psf = psf.reshape(self.k, -1, psf.size(-2), psf.size(-1)).mean(dim=0, keepdim=True)
        psf_sum = psf.sum(dim=(-2, -1), keepdim=True)
        psf_normalized = psf / (abs(psf_sum * self.alpha) + 1e-12)

        image_padded = F.pad(
            image,
            (psf_w // 2, psf_w - psf_w // 2, psf_h // 2, psf_h - psf_h // 2),
            mode="constant",
        )
        psf_padded = F.pad(psf_normalized, (img_w // 2, img_w - img_w // 2, img_h // 2, img_h - img_h // 2), mode="constant")

        image_fft = torch.fft.rfft2(image_padded, dim=(-2, -1))
        psf_fft = torch.fft.rfft2(psf_padded, s=(image_padded.size(-2), image_padded.size(-1)), dim=(-2, -1))
        forward_fft = image_fft * psf_fft
        forward_spatial = torch.fft.irfft2(forward_fft, s=(image_padded.size(-2), image_padded.size(-1)), dim=(-2, -1))
        forward_spatial = torch.fft.ifftshift(forward_spatial, dim=(-2, -1))

        crop_h_start = psf_h // 2
        crop_w_start = psf_w // 2
        forward_crop = forward_spatial[:, :, crop_h_start : crop_h_start + img_h, crop_w_start : crop_w_start + img_w]
        return self.group_norm(forward_crop.real).real


class IFIB(nn.Module):
    """Integrated Forward-Inverse Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        psf_height: int,
        psf_width: int,
        block_cls: type[nn.Module] = RB,
        exchange: float = 0.2,
        k: int = 16,
        pass_fso_to_inverse: bool = True,
        pass_iso_to_forward: bool = True,
    ):
        super().__init__()
        self.iso_operator = ISO(in_channels, height, width, psf_height, psf_width, k=k)
        self.fso_operator = FSO(in_channels, height, width, psf_height, psf_width, k=k)
        self.pass_fso_to_inverse = pass_fso_to_inverse
        self.pass_iso_to_forward = pass_iso_to_forward

        self.inverse_branch_block = block_cls(in_channels, out_channels, num_groups=1)
        self.forward_branch_block = block_cls(in_channels, out_channels, num_groups=1)
        self.inverse_residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.forward_residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.forward_self_mix = nn.Parameter(torch.full((1, 1, 1, 1), 1 - exchange, dtype=torch.float32), requires_grad=True)
        self.forward_cross_mix = nn.Parameter(torch.full((1, 1, 1, 1), exchange, dtype=torch.float32), requires_grad=True)
        self.inverse_self_mix = nn.Parameter(torch.full((1, 1, 1, 1), 1 - exchange, dtype=torch.float32), requires_grad=True)
        self.inverse_cross_mix = nn.Parameter(torch.full((1, 1, 1, 1), exchange, dtype=torch.float32), requires_grad=True)

    def forward(self, inverse_feature: torch.Tensor, forward_feature: torch.Tensor, psf_feature: torch.Tensor):
        inverse_cross = self.iso_operator(forward_feature, psf_feature) if self.pass_iso_to_forward else inverse_feature
        forward_cross = self.fso_operator(inverse_feature, psf_feature) if self.pass_fso_to_inverse else forward_feature

        inverse_feature = self.inverse_branch_block(
            inverse_feature * self.inverse_self_mix + inverse_cross * self.inverse_cross_mix
        )
        forward_feature = self.forward_branch_block(
            forward_feature * self.forward_self_mix + forward_cross * self.forward_cross_mix
        )
        return inverse_feature, forward_feature


class IFINUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        height: int,
        width: int,
        psf_height: int,
        psf_width: int,
        block_cls: type[nn.Module] = RB,
        exchange: float = 0.2,
        k: int = 16,
        pass_fso_to_inverse: bool = True,
        pass_iso_to_forward: bool = True,
    ):
        super().__init__()
        self.inverse_upsample = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.forward_upsample = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.ifib_block = IFIB(
            mid_channels,
            out_channels,
            height,
            width,
            psf_height,
            psf_width,
            block_cls=block_cls,
            exchange=exchange,
            k=k,
            pass_fso_to_inverse=pass_fso_to_inverse,
            pass_iso_to_forward=pass_iso_to_forward,
        )
        self.inverse_merge_projection = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.forward_merge_projection = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

    def forward(
        self,
        inverse_feature_up: torch.Tensor,
        inverse_feature_skip: torch.Tensor,
        forward_feature_up: torch.Tensor,
        forward_feature_skip: torch.Tensor,
        psf_feature: torch.Tensor,
    ):
        inverse_feature_up = self.inverse_upsample(inverse_feature_up)
        forward_feature_up = self.forward_upsample(forward_feature_up)

        diff_y = inverse_feature_skip.size()[2] - inverse_feature_up.size()[2]
        diff_x = inverse_feature_skip.size()[3] - inverse_feature_up.size()[3]
        inverse_feature_up = F.pad(inverse_feature_up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        forward_feature_up = F.pad(forward_feature_up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        inverse_feature = torch.cat([inverse_feature_skip, inverse_feature_up], dim=1)
        forward_feature = torch.cat([forward_feature_skip, forward_feature_up], dim=1)
        inverse_feature = self.inverse_merge_projection(inverse_feature)
        forward_feature = self.forward_merge_projection(forward_feature)
        return self.ifib_block(inverse_feature, forward_feature, psf_feature)


class IFINDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        psf_height: int,
        psf_width: int,
        block_cls: type[nn.Module] = RB,
        exchange: float = 0.2,
        k: int = 16,
        pass_fso_to_inverse: bool = True,
        pass_iso_to_forward: bool = True,
    ):
        super().__init__()
        self.spatial_pool = nn.AvgPool2d(2)
        self.ifib_block = IFIB(
            in_channels,
            out_channels,
            height,
            width,
            psf_height,
            psf_width,
            block_cls=block_cls,
            exchange=exchange,
            k=k,
            pass_fso_to_inverse=pass_fso_to_inverse,
            pass_iso_to_forward=pass_iso_to_forward,
        )
        self.psf_block = ConvG(in_channels * k, out_channels * k)

    def forward(self, inverse_feature: torch.Tensor, forward_feature: torch.Tensor, psf_feature: torch.Tensor):
        inverse_feature = self.spatial_pool(inverse_feature)
        forward_feature = self.spatial_pool(forward_feature)
        psf_feature = self.spatial_pool(psf_feature)
        inverse_feature, forward_feature = self.ifib_block(inverse_feature, forward_feature, psf_feature)
        psf_feature = self.psf_block(psf_feature)
        return inverse_feature, forward_feature, psf_feature


class IFINNet(nn.Module):
    """Integrated Forward-Inverse Network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        psf: torch.Tensor,
        height: int = 270,
        width: int = 480,
        dim: int = 32,
        depth: int = 3,
        block_cls: type[nn.Module] = RB,
        exchange: float = 0.2,
        k: int = 16,
        repeat: bool = True,
        random: bool = False,
    ):
        super().__init__()
        self.psf = psf
        _, _, psf_height, psf_width = psf.size()
        self.depth = depth
        self.psf = nn.Parameter(psf.repeat(1, k, 1, 1), requires_grad=True) if repeat else nn.Parameter(psf, requires_grad=True)
        if random:
            nn.init.xavier_uniform_(self.psf)

        channels = [dim * (2 ** i) for i in range(depth)]
        heights = [height // (2 ** i) for i in range(depth + 1)]
        widths = [width // (2 ** i) for i in range(depth + 1)]
        psf_heights = [psf_height // (2 ** i) for i in range(depth + 1)]
        psf_widths = [psf_width // (2 ** i) for i in range(depth + 1)]

        self.initial_iso_operator = ISO(in_channels, heights[0], widths[0], psf_heights[0], psf_widths[0], k=k)
        self.inverse_seed_block = self.create_block(block_cls, in_channels, channels[0])
        self.forward_seed_block = self.create_block(block_cls, in_channels, channels[0])
        self.psf_encoder = ConvG(self.psf.size(1), channels[0] * k)

        self.down_blocks = nn.ModuleList()
        for idx in range(depth):
            if idx < depth - 1:
                down_out_channels = channels[idx + 1]
            else:
                down_out_channels = channels[idx]
            self.down_blocks.append(
                IFINDownBlock(
                    channels[idx],
                    down_out_channels,
                    heights[idx + 1],
                    widths[idx + 1],
                    psf_heights[idx + 1],
                    psf_widths[idx + 1],
                    block_cls=block_cls,
                    exchange=exchange,
                    k=k,
                    pass_fso_to_inverse=True,
                    pass_iso_to_forward=True,
                )
            )

        self.up_blocks = nn.ModuleList()
        for idx in range(depth - 1, -1, -1):
            if idx == 0:
                self.up_blocks.append(
                    IFINUpBlock(
                        channels[0] * 2,
                        channels[0],
                        channels[0],
                        heights[0],
                        widths[0],
                        psf_heights[0],
                        psf_widths[0],
                        block_cls=block_cls,
                        exchange=exchange,
                        k=k,
                    )
                )
            else:
                self.up_blocks.append(
                    IFINUpBlock(
                        channels[idx] * 2,
                        channels[idx - 1],
                        channels[idx],
                        heights[idx],
                        widths[idx],
                        psf_heights[idx],
                        psf_widths[idx],
                        block_cls=block_cls,
                        exchange=exchange,
                        k=k,
                    )
                )

        self.inverse_refine_block = self.create_block(block_cls, channels[0], channels[0])
        self.inverse_output_head = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.forward_refine_block = self.create_block(block_cls, channels[0], channels[0])
        self.forward_output_head = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1, stride=1, bias=True)

    def create_block(self, block_cls: type[nn.Module], in_channels: int, out_channels: int) -> nn.Module:
        _ = get_num_groups(in_channels)
        return block_cls(in_channels, out_channels, num_groups=1)

    def forward(self, meas_input: torch.Tensor):
        psf_feature = self.psf
        inverse_feature = self.inverse_seed_block(torch.zeros_like(meas_input))
        forward_feature = self.forward_seed_block(meas_input)
        psf_feature = self.psf_encoder(psf_feature)

        inverse_skips = [inverse_feature]
        forward_skips = [forward_feature]
        psf_skips = [psf_feature]

        for depth_idx in range(self.depth):
            inverse_feature, forward_feature, psf_feature = self.down_blocks[depth_idx](
                inverse_skips[-1], forward_skips[-1], psf_skips[-1]
            )
            psf_skips.append(psf_feature)
            inverse_skips.append(inverse_feature)
            forward_skips.append(forward_feature)

        for up_idx in range(self.depth):
            skip_idx = -(up_idx + 2)
            psf_skip = psf_skips[skip_idx]
            inverse_skip = inverse_skips[skip_idx]
            forward_skip = forward_skips[skip_idx]
            inverse_feature, forward_feature = self.up_blocks[up_idx](
                inverse_feature,
                inverse_skip,
                forward_feature,
                forward_skip,
                psf_skip,
            )

        img_inverse = self.inverse_output_head(self.inverse_refine_block(inverse_feature))
        img_forward = self.forward_output_head(self.forward_refine_block(forward_feature))
        return img_inverse, img_forward, img_inverse


def build_ifin_model(config: Dict[str, Any], psf_tensor: torch.Tensor) -> nn.Module:
    model_cfg = config["model"]
    return IFINNet(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        psf=psf_tensor,
        height=model_cfg["height"],
        width=model_cfg["width"],
        dim=model_cfg["dim"],
        depth=model_cfg["depth"],
        block_cls=RB,
        exchange=model_cfg["exchange"],
        k=model_cfg["k"],
        repeat=model_cfg["repeat_psf"],
        random=model_cfg["random_init_psf"],
    )
