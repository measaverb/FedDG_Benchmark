"""Unit tests for FCDv2 model + client loss composition.

Covers:
  * The five FCDv2 loss terms each compute finite values on a dummy batch.
  * Round-0 behaviour: cyclic pathway is skipped when no aggregator is
    fitted, training proceeds with the four non-task losses + original CE.
  * Twin-view forward returns two distinct sets of features.
  * The model wrapper round-trips through both training and eval modes.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fcdv2_aggregators import FCDv2_Gaussian
from src.fcdv2_augmentations import IMAGENET_MEAN, IMAGENET_STD, TwinViewAugmenter
from src.models import (
    Classifier,
    FCDv2Featurizer,
    FCDv2ModelWrapper,
    StyleEncoder,
)


# ── Backbone stub ────────────────────────────────────────────────


class TinyBackbone(nn.Module):
    """Convolutional stub that returns a tiny spatial feature map."""

    def __init__(self, in_channels: int = 3, n_outputs: int = 16):
        super().__init__()
        self.n_outputs = n_outputs
        self.probabilistic = False
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, n_outputs, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def proj_dim():
    return 8


@pytest.fixture
def featurizer(proj_dim):
    return FCDv2Featurizer(TinyBackbone(n_outputs=16), proj_dim=proj_dim)


@pytest.fixture
def model(featurizer, proj_dim):
    classifier = Classifier(proj_dim, 4)
    style = StyleEncoder(z_dim=proj_dim, feat_dim=16)
    return FCDv2ModelWrapper(featurizer, classifier, style)


@pytest.fixture
def normalised_batch():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 3, 16, 16, generator=g)
    mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
    return (x - mean) / std


# ── Wrapper sanity ───────────────────────────────────────────────


def test_wrapper_train_returns_six_tuple(model, normalised_batch):
    model.train()
    out = model(normalised_batch)
    assert isinstance(out, tuple) and len(out) == 6
    logits, z_inv, z_env, H, gamma, beta = out
    assert logits.shape == (4, 4)
    assert z_inv.shape == (4, 8)
    assert z_env.shape == (4, 8)
    assert H.ndim == 4
    assert gamma.shape == (4, 16)
    assert beta.shape == (4, 16)


def test_wrapper_eval_returns_logits_only(model, normalised_batch):
    model.eval()
    out = model(normalised_batch)
    assert torch.is_tensor(out)
    assert out.shape == (4, 4)


# ── Twin views are distinct ──────────────────────────────────────


def test_twin_views_yield_different_features(model, normalised_batch):
    model.train()
    aug = TwinViewAugmenter()
    torch.manual_seed(0)
    x1, x2 = aug(normalised_batch)
    _, z_inv_1, _, _, _, _ = model(x1)
    _, z_inv_2, _, _, _, _ = model(x2)
    assert not torch.allclose(z_inv_1, z_inv_2)


# ── Five loss terms compute without NaN ──────────────────────────


def test_five_loss_terms_finite_on_dummy_batch(model, normalised_batch, proj_dim):
    """Compute each loss head directly on a dummy forward pass."""
    from src.client import FCDv2Client

    model.train()
    aug = TwinViewAugmenter()
    torch.manual_seed(0)
    x1, x2 = aug(normalised_batch)
    out_1 = model(x1)
    out_2 = model(x2)
    logits_1, z_inv_1, z_env_1, H_1, gamma_1, beta_1 = out_1
    logits_2, z_inv_2, z_env_2, H_2, gamma_2, beta_2 = out_2
    y = torch.tensor([0, 1, 2, 3])

    ce = nn.CrossEntropyLoss()
    loss_inv = FCDv2Client._invariance_loss(z_inv_1, z_inv_2)
    loss_stat = 0.5 * (
        FCDv2Client._statistical_grounding_loss(H_1, gamma_1, beta_1)
        + FCDv2Client._statistical_grounding_loss(H_2, gamma_2, beta_2)
    )
    loss_cov = 0.5 * (
        FCDv2Client._cross_covariance_loss(z_inv_1, z_env_1)
        + FCDv2Client._cross_covariance_loss(z_inv_2, z_env_2)
    )
    loss_var = (
        FCDv2Client._variance_loss(z_inv_1)
        + FCDv2Client._variance_loss(z_env_1)
        + FCDv2Client._variance_loss(z_inv_2)
        + FCDv2Client._variance_loss(z_env_2)
    )
    loss_task = 0.5 * (ce(logits_1, y) + ce(logits_2, y))

    for name, loss in [
        ("L_task", loss_task),
        ("L_inv", loss_inv),
        ("L_stat", loss_stat),
        ("L_cov_cross", loss_cov),
        ("L_var", loss_var),
    ]:
        assert torch.isfinite(loss), f"{name} not finite: {loss}"
        assert loss.item() >= 0.0, f"{name} negative: {loss}"


# ── Round-0 / no-aggregator path ─────────────────────────────────


def _build_stub(aggregator=None):
    """Build a minimal FCDv2Client without going through ERM.__init__.

    Uses __new__ so we skip the dataloader/optimizer machinery the real
    constructor builds; populate only the attributes ``step`` reads.
    """
    from src.client import FCDv2Client

    feat = FCDv2Featurizer(TinyBackbone(n_outputs=16), proj_dim=8)
    clf = Classifier(8, 4)
    style = StyleEncoder(z_dim=8, feat_dim=16)
    wrapper = FCDv2ModelWrapper(feat, clf, style)
    wrapper.train()

    stub = FCDv2Client.__new__(FCDv2Client)
    stub.lambda_task = 1.0
    stub.lambda_inv = 1.0
    stub.lambda_stat = 1.0
    stub.lambda_cov = 1.0
    stub.lambda_var = 1.0
    stub.eps = 1e-8
    stub.fcd_cf_start_round = 1
    stub.aggregator = aggregator
    stub.current_server_round = 0
    stub.device = "cpu"
    stub.model = wrapper
    stub._classifier = clf
    stub._featurizer = feat
    stub._style_encoder = style
    stub.optimizer = torch.optim.SGD(wrapper.parameters(), lr=1e-3)
    stub._last_losses = None
    return stub


def _build_results(model):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 3, 16, 16, generator=g)
    mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
    x_norm = (x - mean) / std
    aug = TwinViewAugmenter()
    torch.manual_seed(0)
    x1, x2 = aug(x_norm)
    o1 = model(x1)
    o2 = model(x2)
    return {
        "y_true": torch.tensor([0, 1, 2, 3]),
        "logits_1": o1[0], "z_inv_1": o1[1], "z_env_1": o1[2],
        "H_1": o1[3], "gamma_1": o1[4], "beta_1": o1[5],
        "logits_2": o2[0], "z_inv_2": o2[1], "z_env_2": o2[2],
        "H_2": o2[3], "gamma_2": o2[4], "beta_2": o2[5],
    }


def test_round_zero_skips_cyclic_pathway():
    """No aggregator -> step succeeds, cf loss recorded as 0."""
    stub = _build_stub(aggregator=None)
    results = _build_results(stub.model)

    total_loss = stub.step(results)
    assert isinstance(total_loss, float)
    assert stub._last_losses is not None
    # cf is exactly 0 because the cyclic pathway is gated off.
    assert stub._last_losses["cf"] == 0.0
    # All recorded loss components are finite.
    for k, v in stub._last_losses.items():
        assert v == v, f"NaN in {k}"


def test_round_zero_skipped_with_unfitted_aggregator():
    """An aggregator that was instantiated but not fit is also gated off."""
    agg = FCDv2_Gaussian(dim=8)
    assert not agg.fitted
    stub = _build_stub(aggregator=agg)
    results = _build_results(stub.model)
    _ = stub.step(results)
    assert stub._last_losses["cf"] == 0.0


def test_cyclic_pathway_active_with_fitted_aggregator():
    """A fitted aggregator + sufficient round triggers the cyclic CE term."""
    agg = FCDv2_Gaussian(dim=8)
    pseudo = torch.randn(64, 8)
    idx = torch.zeros(64, dtype=torch.long)
    agg.fit(pseudo, idx)

    stub = _build_stub(aggregator=agg)
    stub.current_server_round = 1
    results = _build_results(stub.model)
    _ = stub.step(results)
    # cf loss should be a finite, non-zero value when the path runs.
    assert stub._last_losses["cf"] > 0.0
    assert stub._last_losses["cf"] == stub._last_losses["cf"]  # not NaN
