from __future__ import annotations

import torch


def gaussian_window(size: int, fwhm: float) -> torch.Tensor:
    with torch.no_grad():
        sigma = size / fwhm
        x = torch.arange(size) - (size - 1) / 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    return gauss.detach()


def gaus_t(x: torch.Tensor, fwhm: float = 3) -> torch.Tensor:
    batch, channels, width, height = x.size()
    ga_w = gaussian_window(width, fwhm)
    ga_h = gaussian_window(height, fwhm)
    ga = ga_w.unsqueeze(1) * ga_h.unsqueeze(0)
    ga = ga.unsqueeze(0).unsqueeze(0)
    ga = ga.expand(batch, channels, width, height)
    return x * ga.to(x.device)


def generate_roi(
    channels: int,
    height: int,
    width: int,
    mode: str = "grid",
    max_angle_deg: float = 60.0,
) -> torch.Tensor:
    if mode == "grid":
        max_rois = 112
        num_rois = max(min(channels, max_rois), 1)
        num_rois_tensor = torch.tensor(num_rois, dtype=torch.float32)

        num_rois_x = torch.ceil(torch.sqrt(num_rois_tensor)).int().item()
        num_rois_y = torch.ceil(num_rois_tensor / num_rois_x).int().item()
        total_rois = num_rois_x * num_rois_y

        num_rois_x = min(num_rois_x, width)
        num_rois_y = min(num_rois_y, height)
        total_rois = num_rois_x * num_rois_y

        x_coords = torch.linspace(0, width - 1, steps=num_rois_x)
        y_coords = torch.linspace(0, height - 1, steps=num_rois_y)
        xs, ys = torch.meshgrid(x_coords, y_coords, indexing="ij")
        centers_x = xs.t().flatten()
        centers_y = ys.t().flatten()

        if total_rois > channels:
            centers_x = centers_x[:channels]
            centers_y = centers_y[:channels]
        elif total_rois < channels:
            repeats = -(-channels // total_rois)
            centers_x = centers_x.repeat(repeats)[:channels]
            centers_y = centers_y.repeat(repeats)[:channels]

        y = torch.arange(0, height, dtype=torch.float32)
        x = torch.arange(0, width, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)

        centers_x = centers_x.view(-1, 1, 1)
        centers_y = centers_y.view(-1, 1, 1)

        sigma_x = width / (2 * num_rois_x) * 2
        sigma_y = height / (2 * num_rois_y) * 2
        if sigma_x == 0:
            sigma_x = 1e-6
        if sigma_y == 0:
            sigma_y = 1e-6

        dx = grid_x - centers_x
        dy = grid_y - centers_y
        weights = torch.exp(-((dx**2) / (2 * sigma_x**2) + (dy**2) / (2 * sigma_y**2)))

    elif mode == "radial":
        y = torch.arange(height, dtype=torch.float32).view(1, height, 1)
        x = torch.arange(width, dtype=torch.float32).view(1, 1, width)
        cx0, cy0 = (width - 1) / 2.0, (height - 1) / 2.0
        rr = torch.sqrt((x - cx0) ** 2 + (y - cy0) ** 2)
        max_r = rr.max()

        i = torch.arange(channels, dtype=torch.float32)
        r_centers = torch.sqrt((i + 0.5) / channels) * max_r
        r_centers = r_centers.view(channels, 1, 1)
        sigma_r = max_r / (2.0 * channels)

        dr = rr - r_centers
        weights = torch.exp(-0.5 * (dr / sigma_r) ** 2)

    else:
        raise ValueError("mode must be 'grid' or 'radial'")

    weights = weights / weights.sum(dim=0, keepdim=True)
    dev = torch.max(torch.abs(weights.sum(dim=0) - 1.0)).item()
    print(f"Max deviation after normalization: {dev:.3e}")
    return weights
