"""End-of-training evaluation for FCDv2.

These helpers are invoked by ``FCDv2Server.fit`` after federated training
finishes. They produce four numbers per run that are reported to wandb:

* per-train-domain accuracy of the global model;
* four linear-probe accuracies on frozen z_inv / z_env predicting class
  and domain labels (80/20 stratified split of a held-out validation
  feature set);
* counterfactual fidelity, i.e. the mean L2 distance between (mu, sigma)
  of generated counterfactual feature maps and the empirical (mu, sigma)
  of real feature maps from the target domain, averaged over every
  ordered pair of domains.

The implementation reuses the conventions in ``linear_probing.py`` but
keeps the dependency surface minimal so this can run inside the federated
training loop with no manual checkpoint reload.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_eval_loader


# ── Feature extraction ────────────────────────────────────────────


@torch.no_grad()
def extract_features(featurizer, loader, device):
    """Return tensors (z_inv, z_env, y, d, H_mu, H_sigma) on CPU."""
    featurizer.train()  # toggle training mode so h_env is exercised
    featurizer.to(device)

    z_inv_list, z_env_list = [], []
    y_list, d_list = [], []
    h_mu_list, h_sigma_list = [], []
    eps = 1e-8

    try:
        for batch in loader:
            x, y, metadata = batch[0], batch[1], batch[2]
            if isinstance(metadata, list):
                metadata = metadata[0]
            x = x.to(device)
            H = featurizer.backbone(x)
            pooled = featurizer.gap(H).flatten(1)
            z_inv = featurizer.h_inv(pooled)
            z_env = featurizer.h_env(pooled)

            mu_H = H.mean(dim=[2, 3])
            sigma_H = (H.var(dim=[2, 3]) + eps).sqrt()

            z_inv_list.append(z_inv.detach().cpu())
            z_env_list.append(z_env.detach().cpu())
            h_mu_list.append(mu_H.detach().cpu())
            h_sigma_list.append(sigma_H.detach().cpu())
            y_list.append(y.detach().cpu())
            d_list.append(metadata[:, 0].detach().cpu())
    finally:
        featurizer.eval()

    return {
        "z_inv": torch.cat(z_inv_list, dim=0),
        "z_env": torch.cat(z_env_list, dim=0),
        "y": torch.cat(y_list, dim=0),
        "d": torch.cat(d_list, dim=0),
        "H_mu": torch.cat(h_mu_list, dim=0),
        "H_sigma": torch.cat(h_sigma_list, dim=0),
    }


# ── Linear probe ──────────────────────────────────────────────────


def _stratified_indices(labels: torch.Tensor, train_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for v in torch.unique(labels):
        mask = (labels == v).numpy()
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * train_ratio))
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _linear_probe_acc(features, labels, num_classes, train_ratio=0.8, seed=0,
                     epochs=50, lr=1e-2, batch_size=256):
    """Train a linear classifier on a stratified split and return test acc."""
    if len(torch.unique(labels)) < 2:
        return None
    train_idx, test_idx = _stratified_indices(labels, train_ratio, seed)
    if not train_idx or not test_idx:
        return None
    f_tr, f_te = features[train_idx], features[test_idx]
    y_tr, y_te = labels[train_idx].long(), labels[test_idx].long()

    probe = nn.Linear(features.shape[1], num_classes)
    opt = torch.optim.SGD(probe.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(f_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    probe.train()
    for _ in range(epochs):
        for fb, yb in loader:
            loss = F.cross_entropy(probe(fb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(f_te).argmax(dim=1)
    return float((preds == y_te).float().mean().item())


# ── Counterfactual fidelity ───────────────────────────────────────


@torch.no_grad()
def counterfactual_fidelity(style_encoder, features_by_domain, device,
                             n_samples=256, seed=0):
    """Mean L2 distance of CF (μ,σ) to real (μ,σ) over ordered domain pairs.

    For each ordered pair (A, B), we sample z_env from B's empirical
    distribution, decode to (γ̂_B, β̂_B) via the style encoder, and treat the
    resulting channel-wise (β̂, γ̂) means as the predicted (μ_cf, σ_cf) at
    the channel level. We then compare them against the empirical (μ_B, σ_B)
    averaged over B's real spatial maps.

    Returns (mean_l2_mu, mean_l2_sigma, list_of_per_pair_results).
    """
    rng = np.random.RandomState(seed)
    style_encoder = style_encoder.to(device).eval()

    domains = sorted(features_by_domain.keys())
    if len(domains) < 2:
        return None, None, []

    per_pair = []
    total_l2_mu, total_l2_sigma = 0.0, 0.0

    for A, B in permutations(domains, 2):
        data_B = features_by_domain[B]
        true_mu_B = data_B["H_mu"].mean(dim=0)
        true_sigma_B = data_B["H_sigma"].mean(dim=0)

        z_env_B = data_B["z_env"]
        n_tgt = z_env_B.size(0)
        if n_tgt == 0:
            continue
        idx = rng.choice(n_tgt, size=min(n_samples, n_tgt), replace=True)
        z_env_sample = z_env_B[idx].to(device)

        gamma_hat, beta_hat = style_encoder(z_env_sample)
        cf_mu_mean = beta_hat.mean(dim=0).cpu()
        cf_sigma_mean = gamma_hat.mean(dim=0).cpu()

        l2_mu = float(torch.norm(cf_mu_mean - true_mu_B, p=2).item())
        l2_sigma = float(torch.norm(cf_sigma_mean - true_sigma_B, p=2).item())
        total_l2_mu += l2_mu
        total_l2_sigma += l2_sigma
        per_pair.append({"src": A, "tgt": B, "l2_mu": l2_mu, "l2_sigma": l2_sigma})

    if not per_pair:
        return None, None, []
    n = len(per_pair)
    return total_l2_mu / n, total_l2_sigma / n, per_pair


# ── Per-train-domain accuracy ─────────────────────────────────────


@torch.no_grad()
def per_domain_accuracy(model, ds_bundle, dataset, device, batch_size=64):
    """Accuracy of ``model`` on the train split, broken down by domain.

    The grouper extracts the domain id from each batch's metadata; predictions
    are bucketed accordingly. Returns ``{domain_id: accuracy}``.
    """
    train_subset = dataset.get_subset("train", transform=ds_bundle.test_transform)
    loader = get_eval_loader(loader="standard", dataset=train_subset, batch_size=batch_size)

    model.eval()
    model.to(device)

    correct = {}
    total = {}
    for batch in tqdm(loader, desc="train per-domain", leave=False):
        x, y, metadata = batch[0], batch[1], batch[2]
        if isinstance(metadata, list):
            metadata = metadata[0]
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if ds_bundle.is_classification:
            preds = torch.argmax(logits, dim=-1)
        else:
            preds = logits
        for i in range(x.size(0)):
            d = int(metadata[i, 0].item())
            correct[d] = correct.get(d, 0) + int((preds[i] == y[i]).item())
            total[d] = total.get(d, 0) + 1

    model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    return {d: correct[d] / total[d] for d in total if total[d] > 0}
