"""End-to-end smoke test for FCDv2.

Spins up two FCDv2 clients on a tiny CIFAR-10-shaped TensorDataset and
runs two federated rounds for each of the four aggregators
(``gaussian``, ``gmm``, ``vae``, ``realnvp``). The test verifies:

  * The full client.fit -> server.aggregate_style -> transmit cycle runs
    without crashes for each aggregator.
  * Per-batch losses stay finite across both rounds.
  * Round 0 yields ``cf == 0`` (no aggregator yet); round 1 engages the
    cyclic counterfactual pathway and produces ``cf > 0``.

Bypasses WILDS / main.py and uses ``__new__`` to skip the heavy ERM
constructor; the goal here is wiring correctness, not a benchmark.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.client import FCDv2Client
from src.fcdv2_aggregators import build_aggregator
from src.fcdv2_augmentations import IMAGENET_MEAN, IMAGENET_STD, TwinViewAugmenter
from src.models import (
    Classifier,
    FCDv2Featurizer,
    FCDv2ModelWrapper,
    StyleEncoder,
)
from src.server import FCDv2Server


# ── Lightweight backbone stub ────────────────────────────────────


class TinyBackbone(nn.Module):
    def __init__(self, n_outputs: int = 16):
        super().__init__()
        self.n_outputs = n_outputs
        self.probabilistic = False
        self.net = nn.Sequential(
            nn.Conv2d(3, n_outputs, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ── Mock ds_bundle pieces ────────────────────────────────────────


class _Grouper:
    def metadata_to_group(self, metadata):
        return torch.zeros(len(metadata), dtype=torch.long)


class _DSBundle:
    def __init__(self):
        self.grouper = _Grouper()


# ── Tiny tensor dataset ──────────────────────────────────────────


def _make_dataset(n: int = 12, n_classes: int = 4, seed: int = 0):
    """ImageNet-normalised CIFAR-shaped tensors with arbitrary metadata."""
    g = torch.Generator().manual_seed(seed)
    x_raw = torch.rand(n, 3, 16, 16, generator=g)
    mean = torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
    x = (x_raw - mean) / std
    y = torch.randint(0, n_classes, (n,), generator=g)
    metadata = torch.zeros(n, dtype=torch.long)
    return TensorDataset(x, y, metadata)


# ── Build a smoke-test FCDv2Client without ERM.__init__ ──────────


def _build_client(client_id, dataset, hparam, featurizer, classifier, style_encoder):
    client = FCDv2Client.__new__(FCDv2Client)
    client.client_id = client_id
    client.device = "cpu"
    client.dataset = dataset
    client.ds_bundle = _DSBundle()
    client.hparam = hparam
    client.batch_size = 4
    client.local_epochs = 1
    client.optimizer_name = "torch.optim.SGD"
    client.optim_config = {"lr": 1e-3, "momentum": 0.0, "weight_decay": 0.0}
    client.scheduler_name = "torch.optim.lr_scheduler.ConstantLR"
    client.scheduler_config = {"factor": 1, "total_iters": 1}
    client.dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    client.opt_dict_path = f"/tmp/_fcdv2_smoke_opt_{client_id}.pt"
    client.sch_dict_path = f"/tmp/_fcdv2_smoke_sch_{client_id}.pt"
    client.saved_optimizer = False
    client.n_groups_per_batch = 1

    # FCDv2-specific attributes (would be set by __init__).
    client.lambda_task = 1.0
    client.lambda_inv = 1.0
    client.lambda_stat = 1.0
    client.lambda_cov = 1.0
    client.lambda_var = 1.0
    client.eps = 1e-8
    client.fcd_cf_start_round = 1
    client.augmenter = TwinViewAugmenter()
    client.aggregator = None
    client.local_env_stats = None
    client._last_losses = None

    # Wire model. Skip nn.DataParallel: it defaults to CUDA when available
    # and this smoke test stays on CPU.
    import copy as _copy
    feat = _copy.deepcopy(featurizer)
    clf = _copy.deepcopy(classifier)
    style = _copy.deepcopy(style_encoder)
    client._featurizer = feat
    client._classifier = clf
    client._style_encoder = style
    client.featurizer = feat
    client.classifier = clf
    client.model = FCDv2ModelWrapper(feat, clf, style)
    return client


def _build_server(aggregator_type, hparam):
    server = FCDv2Server.__new__(FCDv2Server)
    server.device = "cpu"
    server.ds_bundle = _DSBundle()
    server.hparam = hparam
    server.clients = []
    server.num_clients = 0
    server._round = 0
    server.featurizer = None
    server.classifier = None
    server.fcd_gmm_pseudo_samples = 50  # small for the smoke test
    server.aggregator_type = aggregator_type
    server.aggregator = build_aggregator(aggregator_type, dim=8, hparam=hparam)
    return server


# ── The smoke test itself ────────────────────────────────────────


@pytest.mark.parametrize("aggregator_type", ["gaussian", "gmm", "vae", "realnvp"])
def test_two_round_federated_loop(aggregator_type):
    """Run two federated rounds with two clients and verify finite losses."""
    # Tiny aggregator settings so the server fit doesn't dominate the test.
    hparam = {
        "fcd_proj_dim": 8,
        "fcd_gmm_components": 2,
        "fcdv2_vae_latent_dim": 4,
        "fcdv2_vae_epochs": 2,
        "fcdv2_realnvp_layers": 2,
        "fcdv2_realnvp_epochs": 2,
        "seed": 0,
        "wandb": False,
    }

    # Shared model components (deep-copied per client inside _build_client).
    backbone = TinyBackbone(n_outputs=16)
    featurizer = FCDv2Featurizer(backbone, proj_dim=8)
    classifier = Classifier(8, 4)
    style_encoder = StyleEncoder(z_dim=8, feat_dim=16)

    server = _build_server(aggregator_type, hparam)

    clients = []
    for cid in range(2):
        ds = _make_dataset(n=8, seed=cid)
        client = _build_client(
            cid, ds, hparam, featurizer, classifier, style_encoder
        )
        clients.append(client)
    server.clients = clients
    server.num_clients = len(clients)

    # ── Round 0: cyclic pathway is gated off -----------------------------
    server._round = 0
    for client in clients:
        client.set_aggregator(server.aggregator)  # unfitted
        losses = client.fit(server_round=0)
        assert losses is not None
        # cf must be 0 before the aggregator is fitted.
        assert losses["cf"] == 0.0, f"round-0 cf should be 0, got {losses['cf']}"
        for k, v in losses.items():
            assert v == v, f"NaN in {k}: {v}"
        assert client.local_env_stats is not None

    # Server fits the aggregator on pooled per-client pseudo-samples.
    server.aggregate_style([0, 1])
    assert server.aggregator.fitted

    # ── Round 1: cyclic pathway should now run ---------------------------
    server._round = 1
    cf_losses = []
    for client in clients:
        client.set_aggregator(server.aggregator)  # fitted snapshot
        losses = client.fit(server_round=1)
        for k, v in losses.items():
            assert v == v, f"NaN in {k}: {v}"
            assert v == v and abs(v) < float("inf"), f"Non-finite {k}: {v}"
        cf_losses.append(losses["cf"])

    # At least one client should have produced a non-zero cf loss now.
    assert any(c > 0 for c in cf_losses), (
        f"cyclic pathway did not engage in round 1 for {aggregator_type}: {cf_losses}"
    )
