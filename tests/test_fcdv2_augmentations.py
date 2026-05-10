"""Tests for the FCDv2 twin-view augmentation pipeline.

The augmenter must:
  * produce two distinct, well-shaped views from a normalised input batch
  * preserve dtype/device/shape
  * leave the original tensor untouched
  * be deterministic under a fixed RNG seed
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fcdv2_augmentations import IMAGENET_MEAN, IMAGENET_STD, TwinViewAugmenter


def _make_normalised_batch(b: int = 4, h: int = 32, w: int = 32, seed: int = 0):
    """A small ImageNet-normalised batch in [0, 1]-ish space."""
    g = torch.Generator().manual_seed(seed)
    x_raw = torch.rand(b, 3, h, w, generator=g)
    mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
    return (x_raw - mean) / std


def test_two_views_have_correct_shape_and_dtype():
    aug = TwinViewAugmenter()
    x = _make_normalised_batch()
    torch.manual_seed(0)
    x1, x2 = aug(x)
    assert x1.shape == x.shape
    assert x2.shape == x.shape
    assert x1.dtype == x.dtype
    assert x2.dtype == x.dtype


def test_two_views_are_distinct():
    """The two stochastic views must differ from each other and from the input."""
    aug = TwinViewAugmenter()
    x = _make_normalised_batch()
    torch.manual_seed(0)
    x1, x2 = aug(x)
    # Both views should differ from each other on at least one element.
    assert not torch.allclose(x1, x2)
    # And both should differ from the original input (something augmented).
    assert not torch.allclose(x1, x)
    assert not torch.allclose(x2, x)


def test_input_is_not_mutated():
    aug = TwinViewAugmenter()
    x = _make_normalised_batch()
    snap = x.clone()
    torch.manual_seed(0)
    _ = aug(x)
    assert torch.allclose(x, snap)


def test_outputs_are_finite():
    aug = TwinViewAugmenter()
    x = _make_normalised_batch()
    torch.manual_seed(0)
    x1, x2 = aug(x)
    assert torch.isfinite(x1).all()
    assert torch.isfinite(x2).all()


def test_seeded_runs_are_deterministic():
    aug = TwinViewAugmenter()
    x = _make_normalised_batch(seed=7)

    torch.manual_seed(42)
    a1, a2 = aug(x)
    torch.manual_seed(42)
    b1, b2 = aug(x)

    assert torch.allclose(a1, b1)
    assert torch.allclose(a2, b2)


def test_rejects_non_4d_input():
    aug = TwinViewAugmenter()
    bad = torch.randn(3, 32, 32)
    with pytest.raises(ValueError):
        aug(bad)


def test_two_views_alias_matches_forward():
    aug = TwinViewAugmenter()
    x = _make_normalised_batch(seed=3)

    torch.manual_seed(11)
    a1, a2 = aug(x)
    torch.manual_seed(11)
    b1, b2 = aug.two_views(x)

    assert torch.allclose(a1, b1)
    assert torch.allclose(a2, b2)
