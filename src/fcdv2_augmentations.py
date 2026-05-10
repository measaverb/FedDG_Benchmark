"""Twin-view photometric augmentations for FCDv2.

The augmenter takes a normalised batch tensor ``(B, C, H, W)`` and returns
two independent stochastic views ``(x1, x2)`` in the same ImageNet-
normalised space.

Operates at the batch level rather than in the dataset's transform pipe
so that the existing dataset/split/dataloader code stays untouched.

Augmentation set (per spec):
    * ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    * RandomGrayscale(p=0.2)
    * GaussianBlur(kernel_size=3, sigma in [0.1, 2.0])  applied with p=0.5
    * RandomGamma(gamma in [0.7, 1.3])

Each transform is applied per sample so that views differ across the
batch as well as across views.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Standard ImageNet stats used by the FCD/FCDv2 dataset bundles.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * s + m


def _normalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - m) / s


class TwinViewAugmenter(nn.Module):
    """Produce two independently augmented views of a normalised batch."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        grayscale_p: float = 0.2,
        blur_p: float = 0.5,
        blur_kernel: int = 3,
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
        gamma_range: Tuple[float, float] = (0.7, 1.3),
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.color_jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.random_grayscale = T.RandomGrayscale(p=grayscale_p)
        self.gaussian_blur = T.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)
        self.blur_p = float(blur_p)
        self.gamma_range = tuple(gamma_range)
        self.mean = tuple(mean)
        self.std = tuple(std)

    # ── Single-view application ──────────────────────────────────
    def _apply_single(self, x_unnorm: torch.Tensor) -> torch.Tensor:
        """Apply the four-stage stochastic augmentation in [0, 1] space.

        ``x_unnorm`` is denormalised to [0, 1] already.
        """
        # ColorJitter is per-sample stochastic when called on a tensor;
        # apply across the batch by iterating to ensure independence.
        outs = []
        for i in range(x_unnorm.shape[0]):
            xi = x_unnorm[i]
            xi = self.color_jitter(xi)
            xi = self.random_grayscale(xi)
            if torch.rand(()) < self.blur_p:
                xi = self.gaussian_blur(xi)
            gamma = float(
                torch.empty(()).uniform_(self.gamma_range[0], self.gamma_range[1]).item()
            )
            # adjust_gamma requires non-negative inputs; clamp to be safe.
            xi = TF.adjust_gamma(xi.clamp(min=0.0, max=1.0), gamma=gamma)
            outs.append(xi)
        return torch.stack(outs, dim=0).clamp(0.0, 1.0)

    # ── Public entry point ───────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x1, x2)`` -- two ImageNet-normalised augmented views."""
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W); got {tuple(x.shape)}")
        x_unnorm = _denormalize(x, self.mean, self.std).clamp(0.0, 1.0)
        v1 = self._apply_single(x_unnorm)
        v2 = self._apply_single(x_unnorm)
        x1 = _normalize(v1, self.mean, self.std)
        x2 = _normalize(v2, self.mean, self.std)
        return x1, x2

    def two_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias kept for readability at call sites."""
        return self.forward(x)
