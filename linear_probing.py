"""
Linear Probing & Counterfactual Fidelity Experiment for FFD / FCD
==================================================================
Post-training, freeze global Θ and train lightweight linear classifiers
on z_inv and z_env independently to verify double dissociation:

  - Probe z_inv → class y  (expect HIGH accuracy)
  - Probe z_inv → domain d (expect RANDOM chance)
  - Probe z_env → class y  (expect RANDOM chance)
  - Probe z_env → domain d (expect HIGH accuracy)

By default, features are extracted from ALL splits (train + val + test)
so that multiple domains are always represented.  Use --splits to control
which splits are included.

Usage (FFD – default):
    python linear_probing.py \
        --checkpoint resources/pacs_v1.0/models/pacs_FFDClient_ffd_pacs_10.pth \
        --dataset PACS \
        --dataset_path resources/ \
        --epochs 50 \
        --lr 0.01 \
        --batch_size 256

Usage (FCD ResNet-50):
    python linear_probing.py \
        --model_type fcd \
        --backbone resnet50 \
        --checkpoint resources/pacs_v1.0/models/pacs_FCDClient_fcd_pacs_r50_80.pth \
        --dataset PACS \
        --dataset_path resources/ \
        --proj_dim 1024

    # Only use training data (multiple source domains):
    python linear_probing.py --checkpoint ... --splits train

    # Use all splits:
    python linear_probing.py --checkpoint ... --splits train val test

    # Counterfactual fidelity analysis (FCD only):
    python linear_probing.py \
        --model_type fcd \
        --backbone resnet50 \
        --checkpoint resources/pacs_v1.0/models/pacs_FCDClient_fcd_pacs_r50_80.pth \
        --dataset PACS \
        --dataset_path resources/ \
        --proj_dim 1024 \
        --cf_fidelity
"""

import argparse
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_eval_loader

import src.datasets as my_datasets
from src.dataset_bundle import PACS as PACSBundle
from src.dataset_bundle import OfficeHome as OfficeHomeBundle
from src.models import (
    Classifier,
    FCDFeaturizer,
    FCDModelWrapper,
    FFDFeaturizer,
    FFDModelWrapper,
    PooledResNetBackbone,
    SpatialResNetBackbone,
    StyleEncoder,
)


def load_ffd_model(
    checkpoint_path, ds_bundle, device, proj_dim=128, backbone_arch="resnet50"
):
    """Reconstruct FFDModelWrapper and load checkpoint weights."""
    backbone = PooledResNetBackbone(arch=backbone_arch)
    featurizer = FFDFeaturizer(backbone, proj_dim=proj_dim)
    classifier = Classifier(featurizer.n_outputs, ds_bundle.dataset.n_classes)
    model = nn.DataParallel(FFDModelWrapper(featurizer, classifier))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, featurizer


def load_fcd_model(
    checkpoint_path, ds_bundle, device, proj_dim=1024, backbone_arch="resnet50"
):
    """Reconstruct FCDModelWrapper and load checkpoint weights."""
    backbone = SpatialResNetBackbone(arch=backbone_arch)
    featurizer = FCDFeaturizer(backbone, proj_dim=proj_dim)
    classifier = Classifier(featurizer.n_outputs, ds_bundle.dataset.n_classes)
    style_encoder = StyleEncoder(z_dim=proj_dim, feat_dim=backbone.n_outputs)
    model = nn.DataParallel(FCDModelWrapper(featurizer, classifier, style_encoder))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, featurizer


@torch.no_grad()
def extract_ffd_features(model, featurizer, dataloader, device):
    """
    Extract z_inv, z_env, class labels y, and domain labels d
    from a given dataloader using the frozen FFD model.

    FFDFeaturizer uses PooledResNetBackbone (flat vectors), so we
    pass the pooled output directly through h_inv / h_env.
    """
    all_z_inv = []
    all_z_env = []
    all_y = []
    all_d = []

    # Temporarily set featurizer to train mode so both heads are active
    featurizer.train()
    model.eval()

    for batch in tqdm(dataloader, desc="Extracting features (FFD)", leave=False):
        x, y, metadata = batch[0], batch[1], batch[2]
        if isinstance(metadata, list):
            metadata = metadata[0]
        x = x.to(device)

        backbone = featurizer.backbone
        features = backbone(x)  # (B, C_feat) — already pooled
        z_inv = featurizer.h_inv(features)
        z_env = featurizer.h_env(features)

        all_z_inv.append(z_inv.cpu())
        all_z_env.append(z_env.cpu())
        all_y.append(y)
        # metadata[:, 0] = domain, metadata[:, 1] = y label
        all_d.append(metadata[:, 0])

    featurizer.eval()

    z_inv = torch.cat(all_z_inv, dim=0)
    z_env = torch.cat(all_z_env, dim=0)
    y = torch.cat(all_y, dim=0)
    d = torch.cat(all_d, dim=0)

    return z_inv, z_env, y, d


@torch.no_grad()
def extract_fcd_features(model, featurizer, dataloader, device):
    """
    Extract z_inv, z_env, class labels y, and domain labels d
    from a given dataloader using the frozen FCD model.

    FCDFeaturizer uses SpatialResNetBackbone (spatial maps), so we
    must GAP the (B, C_feat, h, w) output before projecting through
    h_inv / h_env — exactly mirroring FCDFeaturizer.forward().
    """
    all_z_inv = []
    all_z_env = []
    all_y = []
    all_d = []

    # Temporarily set featurizer to train mode so both heads are active
    featurizer.train()
    model.eval()

    for batch in tqdm(dataloader, desc="Extracting features (FCD)", leave=False):
        x, y, metadata = batch[0], batch[1], batch[2]
        if isinstance(metadata, list):
            metadata = metadata[0]
        x = x.to(device)

        H = featurizer.backbone(x)  # (B, C_feat, h, w)
        pooled = featurizer.gap(H).flatten(1)  # (B, C_feat)
        z_inv = featurizer.h_inv(pooled)
        z_env = featurizer.h_env(pooled)

        all_z_inv.append(z_inv.cpu())
        all_z_env.append(z_env.cpu())
        all_y.append(y)
        all_d.append(metadata[:, 0])

    featurizer.eval()

    z_inv = torch.cat(all_z_inv, dim=0)
    z_env = torch.cat(all_z_env, dim=0)
    y = torch.cat(all_y, dim=0)
    d = torch.cat(all_d, dim=0)

    return z_inv, z_env, y, d


@torch.no_grad()
def extract_fcd_spatial_features(model, featurizer, style_encoder, dataloader, device):
    """Extract spatial feature maps H, z_env, channel-wise stats, per-sample
    class labels y and domain labels d from a given dataloader.

    This function is purpose-built for the counterfactual fidelity analysis:
    it returns the *spatial* tensors H rather than the pooled representations,
    together with the z_env vectors and precomputed channel-wise (μ, σ) of H
    so the caller can compare synthesised counterfactuals against true stats.

    Returns
    -------
    dict keyed by domain id, each value is a dict with:
        'H_mu'   : (N_d, C_feat)  – channel-wise means of spatial maps
        'H_sigma': (N_d, C_feat)  – channel-wise stds of spatial maps
        'z_env'  : (N_d, proj_dim)
        'y'      : (N_d,)
    """
    per_domain = {}  # domain_id → lists of tensors
    eps = 1e-8

    featurizer.eval()
    model.eval()

    for batch in tqdm(dataloader, desc="Extracting spatial features", leave=False):
        x, y, metadata = batch[0], batch[1], batch[2]
        if isinstance(metadata, list):
            metadata = metadata[0]
        x = x.to(device)

        H = featurizer.backbone(x)  # (B, C_feat, h, w)

        # Channel-wise spatial statistics
        mu_H = H.mean(dim=[2, 3])  # (B, C_feat)
        sigma_H = (H.var(dim=[2, 3]) + eps).sqrt()  # (B, C_feat)

        # Environment vector
        pooled = featurizer.gap(H).flatten(1)
        z_env = featurizer.h_env(pooled)  # (B, proj_dim)

        domains = metadata[:, 0]

        for i in range(x.size(0)):
            d = domains[i].item()
            if d not in per_domain:
                per_domain[d] = {
                    "H_mu": [],
                    "H_sigma": [],
                    "z_env": [],
                    "y": [],
                }
            per_domain[d]["H_mu"].append(mu_H[i].cpu())
            per_domain[d]["H_sigma"].append(sigma_H[i].cpu())
            per_domain[d]["z_env"].append(z_env[i].cpu())
            per_domain[d]["y"].append(y[i])

    # Stack into tensors
    for d in per_domain:
        per_domain[d] = {
            "H_mu": torch.stack(per_domain[d]["H_mu"]),
            "H_sigma": torch.stack(per_domain[d]["H_sigma"]),
            "z_env": torch.stack(per_domain[d]["z_env"]),
            "y": torch.stack(per_domain[d]["y"]),
        }

    return per_domain


@torch.no_grad()
def counterfactual_fidelity_analysis(
    featurizer,
    style_encoder,
    per_domain,
    device,
    n_samples=256,
    seed=42,
):
    """Verify that the style decoder D_φ synthesises accurate domain-specific
    spatial statistics rather than unstructured noise.

    For every ordered pair (A → B) of domains:
      1. Sample z_env vectors from B's empirical distribution.
      2. Decode them through the StyleEncoder to get (γ̂^B, β̂^B).
      3. Take spatial maps from A, instance-normalise, apply (γ̂^B, β̂^B)
         → this produces the counterfactual feature map Ĥ_{A→B}.
      4. Compute channel-wise (μ, σ) of Ĥ_{A→B}.
      5. Compare against the *true* channel-wise (μ, σ) of H_B via L2.

    Returns a list of dicts, one per pair, with L2 distances for μ and σ,
    and a "random baseline" for reference.
    """
    rng = np.random.RandomState(seed)
    domains = sorted(per_domain.keys())
    results = []

    featurizer.eval()
    style_encoder.eval()

    for dom_a, dom_b in combinations(domains, 2):
        for src, tgt in [(dom_a, dom_b), (dom_b, dom_a)]:
            data_src = per_domain[src]
            data_tgt = per_domain[tgt]

            # --- True target statistics (empirical mean of channel-wise stats) ---
            true_mu_B = data_tgt["H_mu"].mean(dim=0)  # (C_feat,)
            true_sigma_B = data_tgt["H_sigma"].mean(dim=0)  # (C_feat,)

            # --- Sample z_env from target domain B's empirical distribution ---
            n_tgt = data_tgt["z_env"].size(0)
            sample_idx = rng.choice(n_tgt, size=min(n_samples, n_tgt), replace=True)
            z_env_B = data_tgt["z_env"][sample_idx].to(device)  # (K, proj_dim)

            # Decode to affine parameters
            gamma_hat, beta_hat = style_encoder(z_env_B)  # (K, C_feat) each

            # --- Synthesise counterfactual channel-wise stats ---
            # AdaIN: Ĥ_{A→B} = γ̂^B · ((H_A - μ_A) / σ_A) + β̂^B
            # The channel-wise mean of Ĥ_{A→B} is β̂^B (since normalised
            # content has zero mean per channel) and std is γ̂^B.
            # But more precisely, we compute the resulting per-sample stats
            # after the affine transform.
            #
            # For a single spatial position: h' = γ̂ · (h - μ_A)/σ_A + β̂
            # E[h'] over spatial positions = γ̂ · (μ_A - μ_A)/σ_A + β̂ = β̂
            # Std[h'] = γ̂  (since Std[(h - μ)/σ] = 1)
            #
            # So the counterfactual channel-wise stats are exactly (β̂, γ̂).
            cf_mu = beta_hat  # (K, C_feat)
            cf_sigma = gamma_hat  # (K, C_feat)

            # Average over samples
            cf_mu_mean = cf_mu.mean(dim=0).cpu()  # (C_feat,)
            cf_sigma_mean = cf_sigma.mean(dim=0).cpu()  # (C_feat,)

            # --- L2 distances ---
            l2_mu = torch.norm(cf_mu_mean - true_mu_B, p=2).item()
            l2_sigma = torch.norm(cf_sigma_mean - true_sigma_B, p=2).item()

            # --- Random baseline: source stats vs target stats ---
            src_mu_mean = data_src["H_mu"].mean(dim=0)
            src_sigma_mean = data_src["H_sigma"].mean(dim=0)
            l2_mu_baseline = torch.norm(src_mu_mean - true_mu_B, p=2).item()
            l2_sigma_baseline = torch.norm(src_sigma_mean - true_sigma_B, p=2).item()

            results.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "l2_mu": l2_mu,
                    "l2_sigma": l2_sigma,
                    "l2_mu_baseline": l2_mu_baseline,
                    "l2_sigma_baseline": l2_sigma_baseline,
                }
            )

    return results


def _fit_umap(z_np, n_neighbors=15, min_dist=0.1):
    """Run UMAP dimensionality reduction (GPU cuML → CPU umap-learn fallback)."""
    try:
        from cuml.manifold import UMAP as cuUMAP

        reducer = cuUMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        return reducer.fit_transform(z_np), "cuML (GPU)"
    except ImportError:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            n_jobs=-1,
        )
        return reducer.fit_transform(z_np), "umap-learn (CPU)"


def visualize_umap(
    features,
    labels,
    output_prefix="umap",
    num_clusters=4,
    label_map=None,
    title="UMAP Projection",
    label_name_prefix="Cluster",
    n_neighbors=15,
    min_dist=0.1,
):
    """
    Generate and save a single UMAP visualization of generic features.
    Attempts to use GPU-accelerated cuML UMAP if available, else falls
    back to the CPU umap-learn library.
    """
    print("\n" + "═" * 60)
    print("UMAP VISUALIZATION")
    print("═" * 60)
    print(
        f"Projecting latent vectors to 2D  "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}) ..."
    )

    if isinstance(features, torch.Tensor):
        z_np = features.cpu().numpy()
    else:
        z_np = features

    if isinstance(labels, torch.Tensor):
        d_np = labels.cpu().numpy()
    else:
        d_np = labels

    z_2d, backend = _fit_umap(z_np, n_neighbors=n_neighbors, min_dist=min_dist)
    print(f"  Backend: {backend}")

    print("  Plotting clusters...")
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    # Cluster reverse mapping
    rev_label_map = {v: k for k, v in label_map.items()} if label_map else None

    for d_idx in range(num_clusters):
        mask = d_np == d_idx
        if mask.sum() > 0:
            label_name = (
                f"{label_name_prefix} {rev_label_map[d_idx]}"
                if rev_label_map
                else f"{label_name_prefix} {d_idx}"
            )
            plt.scatter(
                z_2d[mask, 0],
                z_2d[mask, 1],
                alpha=0.6,
                s=15,
                color=cmap(d_idx % 10),
                label=label_name,
            )

    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(markerscale=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = f"{output_prefix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved UMAP plot to: {os.path.abspath(out_path)}")


def visualize_umap_sweep(
    features,
    labels,
    output_prefix="umap_sweep",
    num_clusters=4,
    label_map=None,
    title_base="UMAP",
    label_name_prefix="Cluster",
    neighbors_list=None,
    min_dist_list=None,
):
    """
    Generate a grid of UMAP projections over all (n_neighbors × min_dist)
    combinations to help identify optimal hyperparameters.

    Produces a single composite figure with rows = n_neighbors values and
    columns = min_dist values.
    """
    if neighbors_list is None:
        neighbors_list = [5, 15, 30, 50, 100]
    if min_dist_list is None:
        min_dist_list = [0.0, 0.1, 0.25, 0.5, 0.8]

    n_rows = len(neighbors_list)
    n_cols = len(min_dist_list)

    print("\n" + "═" * 60)
    print("UMAP PARAMETER SWEEP")
    print("═" * 60)
    print(f"  n_neighbors : {neighbors_list}")
    print(f"  min_dist    : {min_dist_list}")
    print(f"  Grid size   : {n_rows} × {n_cols} = {n_rows * n_cols} projections")

    if isinstance(features, torch.Tensor):
        z_np = features.cpu().numpy()
    else:
        z_np = features

    if isinstance(labels, torch.Tensor):
        d_np = labels.cpu().numpy()
    else:
        d_np = labels

    rev_label_map = {v: k for k, v in label_map.items()} if label_map else None
    cmap = plt.get_cmap("tab10")

    cell_w, cell_h = 5, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w, n_rows * cell_h),
        squeeze=False,
    )

    backend_name = None
    for ri, nn in enumerate(neighbors_list):
        for ci, md in enumerate(min_dist_list):
            ax = axes[ri][ci]
            print(f"  [{ri * n_cols + ci + 1}/{n_rows * n_cols}] "
                  f"n_neighbors={nn}, min_dist={md} ...")

            z_2d, backend_name = _fit_umap(z_np, n_neighbors=nn, min_dist=md)

            for d_idx in range(num_clusters):
                mask = d_np == d_idx
                if mask.sum() > 0:
                    label_name = (
                        f"{label_name_prefix} {rev_label_map[d_idx]}"
                        if rev_label_map
                        else f"{label_name_prefix} {d_idx}"
                    )
                    ax.scatter(
                        z_2d[mask, 0],
                        z_2d[mask, 1],
                        alpha=0.5,
                        s=8,
                        color=cmap(d_idx % 10),
                        label=label_name,
                    )

            ax.set_title(f"nn={nn}  md={md}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.2)

            # Legend only on the first subplot
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, markerscale=1.5, loc="best")

    fig.suptitle(
        f"{title_base}  —  parameter sweep  ({backend_name})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.supylabel("n_neighbors →", fontsize=12)
    fig.supxlabel("min_dist →", fontsize=12)
    fig.tight_layout()

    out_path = f"{output_prefix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved UMAP sweep grid to: {os.path.abspath(out_path)}")


def _fit_tsne(z_np, perplexity=30, learning_rate=200.0, n_iter=1000):
    """Run t-SNE dimensionality reduction (GPU cuML → CPU sklearn fallback)."""
    try:
        from cuml.manifold import TSNE as cuTSNE

        reducer = cuTSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=42,
        )
        return reducer.fit_transform(z_np), "cuML (GPU)"
    except ImportError:
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,
            random_state=42,
            n_jobs=-1,
            init="pca",
        )
        return reducer.fit_transform(z_np), "sklearn (CPU)"


def visualize_tsne_sweep(
    features,
    labels,
    output_prefix="tsne_sweep",
    num_clusters=4,
    label_map=None,
    title_base="t-SNE",
    label_name_prefix="Cluster",
    perplexity_list=None,
    learning_rate_list=None,
    n_iter=1000,
):
    """
    Generate a grid of t-SNE projections over all
    (perplexity × learning_rate) combinations.

    Produces a single composite figure with rows = perplexity values
    and columns = learning_rate values.
    """
    if perplexity_list is None:
        perplexity_list = [5, 15, 30, 50, 100]
    if learning_rate_list is None:
        learning_rate_list = [10.0, 50.0, 200.0, 500.0, 1000.0]

    n_rows = len(perplexity_list)
    n_cols = len(learning_rate_list)

    print("\n" + "═" * 60)
    print("t-SNE PARAMETER SWEEP")
    print("═" * 60)
    print(f"  perplexity    : {perplexity_list}")
    print(f"  learning_rate : {learning_rate_list}")
    print(f"  n_iter        : {n_iter}")
    print(f"  Grid size     : {n_rows} × {n_cols} = {n_rows * n_cols} projections")

    if isinstance(features, torch.Tensor):
        z_np = features.cpu().numpy()
    else:
        z_np = features

    if isinstance(labels, torch.Tensor):
        d_np = labels.cpu().numpy()
    else:
        d_np = labels

    rev_label_map = {v: k for k, v in label_map.items()} if label_map else None
    cmap = plt.get_cmap("tab10")

    cell_w, cell_h = 5, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w, n_rows * cell_h),
        squeeze=False,
    )

    backend_name = None
    for ri, perp in enumerate(perplexity_list):
        for ci, lr_val in enumerate(learning_rate_list):
            ax = axes[ri][ci]
            print(f"  [{ri * n_cols + ci + 1}/{n_rows * n_cols}] "
                  f"perplexity={perp}, lr={lr_val} ...")

            z_2d, backend_name = _fit_tsne(
                z_np, perplexity=perp, learning_rate=lr_val, n_iter=n_iter
            )

            for d_idx in range(num_clusters):
                mask = d_np == d_idx
                if mask.sum() > 0:
                    label_name = (
                        f"{label_name_prefix} {rev_label_map[d_idx]}"
                        if rev_label_map
                        else f"{label_name_prefix} {d_idx}"
                    )
                    ax.scatter(
                        z_2d[mask, 0],
                        z_2d[mask, 1],
                        alpha=0.5,
                        s=8,
                        color=cmap(d_idx % 10),
                        label=label_name,
                    )

            ax.set_title(f"perp={perp}  lr={lr_val}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.2)

            # Legend only on the first subplot
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, markerscale=1.5, loc="best")

    fig.suptitle(
        f"{title_base}  —  parameter sweep  ({backend_name})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.supylabel("perplexity →", fontsize=12)
    fig.supxlabel("learning_rate →", fontsize=12)
    fig.tight_layout()

    out_path = f"{output_prefix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved t-SNE sweep grid to: {os.path.abspath(out_path)}")


def train_linear_probe(
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_classes,
    epochs=50,
    lr=0.01,
    batch_size=256,
):
    """
    Train a linear classifier on frozen features (train set) and
    evaluate on a held-out test set.
    Returns (train_accuracy, test_accuracy, probe_model).
    """
    train_ds = TensorDataset(train_features, train_labels.long())
    test_ds = TensorDataset(test_features, test_labels.long())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    probe = nn.Linear(train_features.shape[1], num_classes)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(epochs):
        for feat_batch, label_batch in train_loader:
            logits = probe(feat_batch)
            loss = criterion(logits, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on both sets
    probe.eval()

    def _eval(loader):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feat_batch, label_batch in loader:
                logits = probe(feat_batch)
                all_preds.append(logits.argmax(dim=1))
                all_labels.append(label_batch)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return (all_preds == all_labels).float().mean().item()

    train_acc = _eval(train_loader)
    test_acc = _eval(test_loader)
    return train_acc, test_acc, probe


def stratified_split(features, labels, domain_labels, train_ratio=0.8, seed=42):
    """
    Split data into train/test ensuring all domains are represented in
    both partitions (stratified by domain).
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    indices = np.arange(n)

    train_idx = []
    test_idx = []
    unique_domains = torch.unique(domain_labels).numpy()

    for dom in unique_domains:
        dom_mask = domain_labels.numpy() == dom
        dom_indices = indices[dom_mask]
        rng.shuffle(dom_indices)
        split_point = max(1, int(len(dom_indices) * train_ratio))
        train_idx.extend(dom_indices[:split_point].tolist())
        test_idx.extend(dom_indices[split_point:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return train_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="FFD / FCD Linear Probing Experiment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved FFD/FCD model checkpoint (.pth)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ffd",
        choices=["ffd", "fcd"],
        help="Model variant: 'ffd' (FFDFeaturizer) or 'fcd' (FCDFeaturizer)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="Backbone architecture (must match checkpoint)",
    )
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--dataset_path", type=str, default="resources/")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which dataset splits to pool features from (default: train val test). "
        "Using multiple splits ensures multiple domains are represented.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Fraction of pooled data used for probe training (rest for eval)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Epochs for each linear probe"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for linear probes"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=1024,
        help="Projection dim (must match checkpoint)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cf_fidelity",
        action="store_true",
        help="Run counterfactual fidelity analysis (FCD only). "
        "Verifies that the style decoder synthesises accurate "
        "domain-specific spatial statistics.",
    )
    parser.add_argument(
        "--cf_n_samples",
        type=int,
        default=256,
        help="Number of samples per domain pair for counterfactual fidelity analysis.",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Run UMAP visualization on z_env / z_inv to evaluate clustering.",
    )
    parser.add_argument(
        "--umap_output",
        type=str,
        default="umap_z_env",
        help="Output file prefix for UMAP plots.",
    )
    parser.add_argument(
        "--umap_sweep",
        action="store_true",
        help="Run a UMAP parameter sweep over n_neighbors and min_dist values. "
        "Generates a grid figure per feature type for visual comparison.",
    )
    parser.add_argument(
        "--umap_neighbors",
        nargs="+",
        type=int,
        default=[5, 15, 30, 50, 100],
        help="n_neighbors values for the UMAP sweep (default: 5 15 30 50 100).",
    )
    parser.add_argument(
        "--umap_min_dist",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 0.8],
        help="min_dist values for the UMAP sweep (default: 0.0 0.1 0.25 0.5 0.8).",
    )
    parser.add_argument(
        "--tsne_sweep",
        action="store_true",
        help="Run a t-SNE parameter sweep over perplexity and learning_rate values. "
        "Generates a grid figure per feature type for visual comparison.",
    )
    parser.add_argument(
        "--tsne_perplexity",
        nargs="+",
        type=float,
        default=[5, 15, 30, 50, 100],
        help="Perplexity values for the t-SNE sweep (default: 5 15 30 50 100).",
    )
    parser.add_argument(
        "--tsne_lr",
        nargs="+",
        type=float,
        default=[10.0, 50.0, 200.0, 500.0, 1000.0],
        help="Learning rate values for the t-SNE sweep (default: 10 50 200 500 1000).",
    )
    parser.add_argument(
        "--tsne_n_iter",
        type=int,
        default=1000,
        help="Number of iterations for each t-SNE run (default: 1000).",
    )
    parser.add_argument(
        "--tsne_output",
        type=str,
        default="tsne",
        help="Output file prefix for t-SNE sweep plots.",
    )
    args = parser.parse_args()

    if args.cf_fidelity and args.model_type != "fcd":
        parser.error("--cf_fidelity is only supported with --model_type fcd")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────
    if args.dataset.lower() == "pacs":
        dataset = my_datasets.PACS(
            version="1.0", root_dir=args.dataset_path, download=True
        )
        ds_bundle = PACSBundle(dataset, probabilistic=False)
    elif args.dataset.lower() == "officehome":
        dataset = my_datasets.OfficeHome(
            version="1.0", root_dir=args.dataset_path, download=True
        )
        ds_bundle = OfficeHomeBundle(dataset, probabilistic=False)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported yet.")

    # ── Load model ────────────────────────────────────────────────
    if args.model_type == "fcd":
        model, featurizer = load_fcd_model(
            args.checkpoint,
            ds_bundle,
            device,
            proj_dim=args.proj_dim,
            backbone_arch=args.backbone,
        )
        extract_fn = extract_fcd_features
    else:
        model, featurizer = load_ffd_model(
            args.checkpoint,
            ds_bundle,
            device,
            proj_dim=args.proj_dim,
            backbone_arch=args.backbone,
        )
        extract_fn = extract_ffd_features
    featurizer = featurizer.to(device)
    print(f"✓ Loaded {args.model_type.upper()} checkpoint: {args.checkpoint}")

    # ── Extract features from ALL requested splits ────────────────
    all_z_inv, all_z_env, all_y, all_d = [], [], [], []
    available_splits = list(dataset.split_names)

    for split in args.splits:
        if split not in available_splits:
            print(f"  ⚠ Split '{split}' not found in dataset, skipping.")
            continue

        print(f"\nExtracting features from '{split}' split...")
        ds_subset = dataset.get_subset(split, transform=ds_bundle.test_transform)
        loader = get_eval_loader(
            loader="standard", dataset=ds_subset, batch_size=args.batch_size
        )
        z_inv, z_env, y, d = extract_fn(model, featurizer, loader, device)
        all_z_inv.append(z_inv)
        all_z_env.append(z_env)
        all_y.append(y)
        all_d.append(d)
        print(
            f"  {split}: {len(y)} samples, "
            f"domains={sorted(torch.unique(d).tolist())}"
        )

    z_inv = torch.cat(all_z_inv, dim=0)
    z_env = torch.cat(all_z_env, dim=0)
    y_labels = torch.cat(all_y, dim=0)
    d_labels = torch.cat(all_d, dim=0)

    num_classes = dataset.n_classes
    unique_domains = torch.unique(d_labels)
    num_domains = len(unique_domains)
    # Remap domain labels to 0..num_domains-1
    domain_map = {v.item(): i for i, v in enumerate(sorted(unique_domains))}
    d_labels = torch.tensor([domain_map[v.item()] for v in d_labels])

    print(f"\n{'─'*60}")
    print(f"  Pooled data:  {len(y_labels)} samples")
    print(f"  z_inv shape:  {z_inv.shape}")
    print(f"  z_env shape:  {z_env.shape}")
    print(f"  Num classes:  {num_classes}")
    print(f"  Num domains:  {num_domains}  (IDs: {sorted(domain_map.keys())})")
    print(f"  Chance class:  {1.0/num_classes:.4f}")
    print(f"  Chance domain: {1.0/num_domains:.4f}")
    print(f"{'─'*60}")

    if args.umap:
        # Avoid naming collisions
        env_out = (
            args.umap_output
            if not args.umap_output.endswith("umap_z_env")
            else "umap_z_env"
        )
        inv_out = (
            args.umap_output + "_inv"
            if not args.umap_output.endswith("umap_z_env")
            else "umap_z_inv"
        )

        visualize_umap(
            features=z_env,
            labels=d_labels,
            output_prefix=env_out,
            num_clusters=num_domains,
            label_map=domain_map,
            title="UMAP Projection of z_env (Domains)",
            label_name_prefix="Domain",
        )

        visualize_umap(
            features=z_inv,
            labels=y_labels,
            output_prefix=inv_out,
            num_clusters=num_classes,
            label_map=None,
            title="UMAP Projection of z_inv (Classes)",
            label_name_prefix="Class",
        )

    if args.umap_sweep:
        env_sweep_out = args.umap_output + "_sweep_env"
        inv_sweep_out = args.umap_output + "_sweep_inv"

        visualize_umap_sweep(
            features=z_env,
            labels=d_labels,
            output_prefix=env_sweep_out,
            num_clusters=num_domains,
            label_map=domain_map,
            title_base="z_env (Domains)",
            label_name_prefix="Domain",
            neighbors_list=args.umap_neighbors,
            min_dist_list=args.umap_min_dist,
        )

        visualize_umap_sweep(
            features=z_inv,
            labels=y_labels,
            output_prefix=inv_sweep_out,
            num_clusters=num_classes,
            label_map=None,
            title_base="z_inv (Classes)",
            label_name_prefix="Class",
            neighbors_list=args.umap_neighbors,
            min_dist_list=args.umap_min_dist,
        )

    if args.tsne_sweep:
        env_sweep_out = args.tsne_output + "_sweep_env"
        inv_sweep_out = args.tsne_output + "_sweep_inv"

        visualize_tsne_sweep(
            features=z_env,
            labels=d_labels,
            output_prefix=env_sweep_out,
            num_clusters=num_domains,
            label_map=domain_map,
            title_base="z_env (Domains)",
            label_name_prefix="Domain",
            perplexity_list=args.tsne_perplexity,
            learning_rate_list=args.tsne_lr,
            n_iter=args.tsne_n_iter,
        )

        visualize_tsne_sweep(
            features=z_inv,
            labels=y_labels,
            output_prefix=inv_sweep_out,
            num_clusters=num_classes,
            label_map=None,
            title_base="z_inv (Classes)",
            label_name_prefix="Class",
            perplexity_list=args.tsne_perplexity,
            learning_rate_list=args.tsne_lr,
            n_iter=args.tsne_n_iter,
        )

    if num_domains < 2:
        print(
            "\n⚠ WARNING: Only 1 domain found in the selected splits!\n"
            "  Domain probing is meaningless with a single domain.\n"
            "  Re-run with --splits that include multiple domains, e.g.:\n"
            "    python linear_probing.py --splits train val test ...\n"
        )
        return

    # ── Stratified train/test split (balanced across domains) ─────
    train_idx, test_idx = stratified_split(
        z_inv, y_labels, d_labels, train_ratio=args.train_ratio, seed=args.seed
    )
    print(f"\n  Probe train: {len(train_idx)} samples")
    print(f"  Probe test:  {len(test_idx)} samples")

    z_inv_train, z_inv_test = z_inv[train_idx], z_inv[test_idx]
    z_env_train, z_env_test = z_env[train_idx], z_env[test_idx]
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]
    d_train, d_test = d_labels[train_idx], d_labels[test_idx]

    # ── Train 4 Linear Probes ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK A: Class Prediction (y)")
    print("=" * 60)

    print("\n[Probe 1] z_inv → y (class)")
    tr_acc, te_acc, _ = train_linear_probe(
        z_inv_train,
        y_train,
        z_inv_test,
        y_test,
        num_classes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    acc_inv_y = te_acc
    print(f"  ➜ Train: {tr_acc:.4f}  Test: {te_acc:.4f}  (expect HIGH)")

    print("\n[Probe 2] z_env → y (class)")
    tr_acc, te_acc, _ = train_linear_probe(
        z_env_train,
        y_train,
        z_env_test,
        y_test,
        num_classes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    acc_env_y = te_acc
    print(
        f"  ➜ Train: {tr_acc:.4f}  Test: {te_acc:.4f}  (expect ~{1.0/num_classes:.4f} random)"
    )

    print("\n" + "=" * 60)
    print("TASK B: Domain Prediction (d)")
    print("=" * 60)

    print("\n[Probe 3] z_inv → d (domain)")
    tr_acc, te_acc, _ = train_linear_probe(
        z_inv_train,
        d_train,
        z_inv_test,
        d_test,
        num_domains,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    acc_inv_d = te_acc
    print(
        f"  ➜ Train: {tr_acc:.4f}  Test: {te_acc:.4f}  (expect ~{1.0/num_domains:.4f} random)"
    )

    print("\n[Probe 4] z_env → d (domain)")
    tr_acc, te_acc, _ = train_linear_probe(
        z_env_train,
        d_train,
        z_env_test,
        d_test,
        num_domains,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    acc_env_d = te_acc
    print(f"  ➜ Train: {tr_acc:.4f}  Test: {te_acc:.4f}  (expect HIGH)")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DISENTANGLEMENT SUMMARY (test-set accuracy)")
    print("=" * 60)
    print(f"{'Probe':<25} {'Test Acc':>10}  {'Expected':>12}")
    print("-" * 50)
    print(f"{'z_inv → y (class)':<25} {acc_inv_y:>10.4f}  {'HIGH':>12}")
    print(
        f"{'z_inv → d (domain)':<25} {acc_inv_d:>10.4f}  {f'~{1.0/num_domains:.4f}':>12}"
    )
    print(
        f"{'z_env → y (class)':<25} {acc_env_y:>10.4f}  {f'~{1.0/num_classes:.4f}':>12}"
    )
    print(f"{'z_env → d (domain)':<25} {acc_env_d:>10.4f}  {'HIGH':>12}")
    print("=" * 60)

    # Disentanglement score
    inv_class_gap = acc_inv_y - (1.0 / num_classes)
    inv_domain_gap = acc_inv_d - (1.0 / num_domains)
    env_class_gap = acc_env_y - (1.0 / num_classes)
    env_domain_gap = acc_env_d - (1.0 / num_domains)

    disentangle_score = (
        inv_class_gap - inv_domain_gap + env_domain_gap - env_class_gap
    ) / 4

    print(
        f"\nDisentanglement Score: {disentangle_score:.4f} (higher is better, max ~0.5)"
    )

    # ── §4 Counterfactual Fidelity Analysis (FCD only) ─────────────
    if args.cf_fidelity:
        print("\n" + "═" * 60)
        print("COUNTERFACTUAL FIDELITY ANALYSIS")
        print("═" * 60)
        print("Verifying that the style decoder D_φ synthesises accurate")
        print("domain-specific spatial statistics (μ, σ) via AdaIN...\n")

        style_encoder = model.module.style_encoder
        style_encoder.to(device)
        style_encoder.eval()

        # Re-extract spatial features with per-domain grouping
        all_spatial = {}
        for split in args.splits:
            if split not in available_splits:
                continue
            print(f"  Extracting spatial features from '{split}'...")
            ds_subset = dataset.get_subset(split, transform=ds_bundle.test_transform)
            loader = get_eval_loader(
                loader="standard", dataset=ds_subset, batch_size=args.batch_size
            )
            spatial = extract_fcd_spatial_features(
                model, featurizer, style_encoder, loader, device
            )
            # Merge into all_spatial
            for d_id, d_data in spatial.items():
                if d_id not in all_spatial:
                    all_spatial[d_id] = d_data
                else:
                    for k in d_data:
                        all_spatial[d_id][k] = torch.cat(
                            [all_spatial[d_id][k], d_data[k]], dim=0
                        )

        print(f"\n  Domains found: {sorted(all_spatial.keys())}")
        for d_id in sorted(all_spatial.keys()):
            print(f"    Domain {d_id}: {all_spatial[d_id]['H_mu'].size(0)} samples")

        # Run analysis
        cf_results = counterfactual_fidelity_analysis(
            featurizer,
            style_encoder,
            all_spatial,
            device,
            n_samples=args.cf_n_samples,
            seed=args.seed,
        )

        # Display results
        print(f"\n{'─' * 72}")
        print(
            f"  {'Pair':<14} {'L2(μ) CF':>10} {'L2(μ) Base':>12} "
            f"{'L2(σ) CF':>10} {'L2(σ) Base':>12}  {'μ Fidelity':>10}"
        )
        print(f"{'─' * 72}")

        total_l2_mu = 0.0
        total_l2_sigma = 0.0
        total_l2_mu_base = 0.0
        total_l2_sigma_base = 0.0

        for r in cf_results:
            # Fidelity ratio: how much closer CF is vs baseline (1.0 = perfect)
            mu_fidelity = 1.0 - r["l2_mu"] / max(r["l2_mu_baseline"], 1e-8)
            print(
                f"  {r['src']} → {r['tgt']:<8} "
                f"{r['l2_mu']:>10.4f} {r['l2_mu_baseline']:>12.4f} "
                f"{r['l2_sigma']:>10.4f} {r['l2_sigma_baseline']:>12.4f}  "
                f"{mu_fidelity:>10.2%}"
            )
            total_l2_mu += r["l2_mu"]
            total_l2_sigma += r["l2_sigma"]
            total_l2_mu_base += r["l2_mu_baseline"]
            total_l2_sigma_base += r["l2_sigma_baseline"]

        n_pairs = len(cf_results)
        print(f"{'─' * 72}")
        avg_mu_fidelity = 1.0 - (total_l2_mu / n_pairs) / max(
            total_l2_mu_base / n_pairs, 1e-8
        )
        print(
            f"  {'Average':<14} "
            f"{total_l2_mu / n_pairs:>10.4f} "
            f"{total_l2_mu_base / n_pairs:>12.4f} "
            f"{total_l2_sigma / n_pairs:>10.4f} "
            f"{total_l2_sigma_base / n_pairs:>12.4f}  "
            f"{avg_mu_fidelity:>10.2%}"
        )
        print(f"{'═' * 72}")
        print(
            "\n  Interpretation:\n"
            "    L2(CF) ≪ L2(Base)  ⇒  decoder is a faithful domain translator\n"
            "    μ Fidelity → 1.0   ⇒  perfect moment transfer\n"
            "    μ Fidelity → 0.0   ⇒  no better than using source stats directly"
        )


if __name__ == "__main__":
    main()
