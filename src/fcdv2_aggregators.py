"""Federated style aggregators for FCDv2.

Each aggregator consumes pooled pseudo-samples drawn from per-client
``(mu_i, Sigma_i)`` Gaussians on the server, fits a density model over
``z_env``, and exposes ``sample`` for the cyclic counterfactual pathway
plus ``log_prob`` for diagnostics.

Available aggregators:
    * ``FCDv2_Gaussian`` -- single diagonal multivariate Gaussian (MLE).
    * ``FCDv2_GMM``      -- diagonal Gaussian mixture model (sklearn EM).
    * ``FCDv2_VAE``      -- small MLP variational autoencoder.
    * ``FCDv2_RealNVP``  -- coupling-layer normalising flow (from scratch;
                             no flow library is in the project's deps).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Base interface ────────────────────────────────────────────────


class FederatedStyleAggregator:
    """Abstract base for federated style aggregators over ``z_env``.

    Subclasses fit a density on pooled pseudo-samples and provide
    ``sample`` / ``log_prob`` / ``state_dict`` / ``load_state_dict``.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._fitted = False

    # API ------------------------------------------------------------------
    def fit(
        self,
        pseudo_samples: torch.Tensor,
        client_indices: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def sample(
        self,
        n_samples: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError

    # Helpers --------------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return self._fitted

    def _check_input(self, pseudo_samples: torch.Tensor, client_indices: torch.Tensor):
        if pseudo_samples.ndim != 2:
            raise ValueError(
                f"pseudo_samples must be (N, d); got {tuple(pseudo_samples.shape)}"
            )
        if pseudo_samples.shape[1] != self.dim:
            raise ValueError(
                f"pseudo_samples dim {pseudo_samples.shape[1]} != aggregator dim {self.dim}"
            )
        if client_indices.shape[0] != pseudo_samples.shape[0]:
            raise ValueError(
                "client_indices length must match pseudo_samples rows"
            )


# ── 3.1 Single diagonal Gaussian ─────────────────────────────────


class FCDv2_Gaussian(FederatedStyleAggregator):
    """Maximum-likelihood diagonal multivariate Gaussian on pooled samples.

    Uses ``scale_tril`` for numerical stability when constructing the
    underlying ``torch.distributions.MultivariateNormal``.
    """

    EPS = 1e-6

    def __init__(self, dim: int):
        super().__init__(dim)
        self._mean: Optional[torch.Tensor] = None
        self._scale_tril: Optional[torch.Tensor] = None

    def fit(self, pseudo_samples, client_indices):
        self._check_input(pseudo_samples, client_indices)
        x = pseudo_samples.detach().to(torch.float64)
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False).clamp_min(self.EPS)
        self._mean = mean.to(torch.float32)
        self._scale_tril = torch.diag(var.sqrt().to(torch.float32))
        self._fitted = True

    def _dist(self) -> torch.distributions.MultivariateNormal:
        if not self._fitted:
            raise RuntimeError("FCDv2_Gaussian must be fit() before use")
        return torch.distributions.MultivariateNormal(
            loc=self._mean, scale_tril=self._scale_tril
        )

    def sample(self, n_samples, generator=None):
        if not self._fitted:
            raise RuntimeError("FCDv2_Gaussian must be fit() before use")
        # MultivariateNormal does not accept a generator; manually use
        # reparameterised sampling so the caller's generator is honoured.
        eps = torch.randn(n_samples, self.dim, generator=generator)
        return self._mean.unsqueeze(0) + eps @ self._scale_tril.T

    def log_prob(self, z):
        return self._dist().log_prob(z)

    def state_dict(self):
        return {
            "type": "gaussian",
            "dim": self.dim,
            "mean": None if self._mean is None else self._mean.cpu(),
            "scale_tril": None if self._scale_tril is None else self._scale_tril.cpu(),
            "fitted": self._fitted,
        }

    def load_state_dict(self, state):
        if state.get("type") != "gaussian":
            raise ValueError(f"State type mismatch: {state.get('type')}")
        self.dim = int(state["dim"])
        self._mean = None if state["mean"] is None else state["mean"].clone()
        self._scale_tril = (
            None if state["scale_tril"] is None else state["scale_tril"].clone()
        )
        self._fitted = bool(state["fitted"])


# ── 3.2 Diagonal Gaussian mixture model ──────────────────────────


class FCDv2_GMM(FederatedStyleAggregator):
    """Diagonal-covariance GMM fitted via sklearn EM on pooled pseudo-samples.

    Mirrors the existing FCD GMM aggregator's behaviour; the number of
    components is capped at the number of valid clients seen in the fit.
    """

    EPS = 1e-6

    def __init__(self, dim: int, n_components: int = 8, max_iter: int = 100, random_state: int = 0):
        super().__init__(dim)
        self.n_components_requested = int(n_components)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._weights: Optional[torch.Tensor] = None
        self._means: Optional[torch.Tensor] = None
        # Per-component diagonal variances stored as (M, d) for compactness;
        # expanded to scale_tril on demand.
        self._variances: Optional[torch.Tensor] = None

    def fit(self, pseudo_samples, client_indices):
        from sklearn.mixture import GaussianMixture

        self._check_input(pseudo_samples, client_indices)
        n_clients = int(torch.unique(client_indices).numel())
        n_components = max(1, min(self.n_components_requested, n_clients))

        x_np = pseudo_samples.detach().to(torch.float64).cpu().numpy()
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        gmm.fit(x_np)

        self._weights = torch.tensor(gmm.weights_, dtype=torch.float32)
        self._means = torch.tensor(gmm.means_, dtype=torch.float32)
        variances = torch.tensor(gmm.covariances_, dtype=torch.float32).clamp_min(self.EPS)
        self._variances = variances
        self._fitted = True

    def _component_dist(self, m: int) -> torch.distributions.MultivariateNormal:
        scale_tril = torch.diag(self._variances[m].sqrt())
        return torch.distributions.MultivariateNormal(
            loc=self._means[m], scale_tril=scale_tril
        )

    def sample(self, n_samples, generator=None):
        if not self._fitted:
            raise RuntimeError("FCDv2_GMM must be fit() before use")
        # Pick components according to mixing weights.
        comp_idx = torch.multinomial(
            self._weights, n_samples, replacement=True, generator=generator
        )
        out = torch.empty(n_samples, self.dim)
        for m in range(self._weights.numel()):
            mask = comp_idx == m
            count = int(mask.sum().item())
            if count == 0:
                continue
            std = self._variances[m].sqrt()
            eps = torch.randn(count, self.dim, generator=generator)
            out[mask] = self._means[m].unsqueeze(0) + eps * std.unsqueeze(0)
        return out

    def log_prob(self, z):
        if not self._fitted:
            raise RuntimeError("FCDv2_GMM must be fit() before use")
        # log sum_m w_m * N(z | mu_m, diag(var_m)) computed in log-space.
        log_w = torch.log(self._weights.clamp_min(1e-12))
        log_components = []
        for m in range(self._weights.numel()):
            log_components.append(self._component_dist(m).log_prob(z) + log_w[m])
        return torch.logsumexp(torch.stack(log_components, dim=0), dim=0)

    def state_dict(self):
        return {
            "type": "gmm",
            "dim": self.dim,
            "n_components_requested": self.n_components_requested,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "weights": None if self._weights is None else self._weights.cpu(),
            "means": None if self._means is None else self._means.cpu(),
            "variances": None if self._variances is None else self._variances.cpu(),
            "fitted": self._fitted,
        }

    def load_state_dict(self, state):
        if state.get("type") != "gmm":
            raise ValueError(f"State type mismatch: {state.get('type')}")
        self.dim = int(state["dim"])
        self.n_components_requested = int(state["n_components_requested"])
        self.max_iter = int(state["max_iter"])
        self.random_state = int(state["random_state"])
        self._weights = None if state["weights"] is None else state["weights"].clone()
        self._means = None if state["means"] is None else state["means"].clone()
        self._variances = (
            None if state["variances"] is None else state["variances"].clone()
        )
        self._fitted = bool(state["fitted"])


# ── 3.3 Variational autoencoder ──────────────────────────────────


class _VAEModule(nn.Module):
    """Encoder/decoder MLP pair for ``FCDv2_VAE``."""

    def __init__(self, dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )
        self.latent_dim = latent_dim
        self.dim = dim

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        # Soft clamp for stability of exp().
        log_var = log_var.clamp(min=-10.0, max=10.0)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)


class FCDv2_VAE(FederatedStyleAggregator):
    """Small VAE over ``z_env`` with N(0, I) prior and Gaussian likelihood.

    ``log_prob`` returns an importance-weighted (IWAE, K=10) estimate of the
    marginal likelihood -- this is an *estimate*, not the exact value, since
    the marginal of a VAE is intractable.
    """

    LOG2PI = math.log(2.0 * math.pi)

    def __init__(
        self,
        dim: int,
        latent_dim: int = 32,
        hidden: int = 256,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        log_likelihood_var: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        # Fixed observation variance for the Gaussian likelihood term.
        self.log_likelihood_var = float(log_likelihood_var)
        self.device = device
        self.module: Optional[_VAEModule] = None

    # Internal helpers -----------------------------------------------------
    def _build(self):
        self.module = _VAEModule(self.dim, self.latent_dim, self.hidden).to(self.device)

    def _gaussian_log_prob(self, x, mean, log_var):
        # Diagonal Gaussian log-density, summed over feature dim.
        return -0.5 * (
            self.LOG2PI + log_var + (x - mean) ** 2 / log_var.exp()
        ).sum(dim=-1)

    def _standard_normal_log_prob(self, z):
        return -0.5 * (self.LOG2PI + z * z).sum(dim=-1)

    def fit(self, pseudo_samples, client_indices):
        self._check_input(pseudo_samples, client_indices)
        self._build()
        x = pseudo_samples.detach().to(self.device, dtype=torch.float32)
        n = x.shape[0]

        optim = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        self.module.train()
        log_obs_var = torch.full(
            (1,), math.log(self.log_likelihood_var), device=self.device
        )

        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch = x[idx]
                mu, log_var = self.module.encode(batch)
                std = (0.5 * log_var).exp()
                eps = torch.randn_like(std)
                z = mu + eps * std
                recon = self.module.decode(z)

                # Negative ELBO with fixed-variance Gaussian likelihood.
                recon_log_prob = -0.5 * (
                    self.LOG2PI
                    + log_obs_var
                    + (batch - recon) ** 2 / log_obs_var.exp()
                ).sum(dim=-1)
                kl = 0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var).sum(dim=-1)
                neg_elbo = -(recon_log_prob - kl).mean()

                optim.zero_grad()
                neg_elbo.backward()
                optim.step()

        self.module.eval()
        self._fitted = True

    def sample(self, n_samples, generator=None):
        if not self._fitted:
            raise RuntimeError("FCDv2_VAE must be fit() before use")
        z = torch.randn(n_samples, self.latent_dim, generator=generator, device=self.device)
        with torch.no_grad():
            x = self.module.decode(z)
        return x.cpu()

    def log_prob(self, z, K: int = 10):
        """IWAE bound (K samples) on the marginal log-likelihood.

        Returned values are estimates, not exact log-probs.
        """
        if not self._fitted:
            raise RuntimeError("FCDv2_VAE must be fit() before use")
        x = z.to(self.device, dtype=torch.float32)
        n = x.shape[0]
        log_obs_var = torch.full(
            (1,), math.log(self.log_likelihood_var), device=self.device
        )
        with torch.no_grad():
            mu, log_var = self.module.encode(x)
            std = (0.5 * log_var).exp()
            # (K, n, latent_dim)
            eps = torch.randn(K, n, self.latent_dim, device=self.device)
            z_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
            recon = self.module.decode(z_samples.reshape(-1, self.latent_dim))
            recon = recon.view(K, n, self.dim)

            log_p_x_given_z = -0.5 * (
                self.LOG2PI
                + log_obs_var
                + (x.unsqueeze(0) - recon) ** 2 / log_obs_var.exp()
            ).sum(dim=-1)  # (K, n)
            log_p_z = -0.5 * (self.LOG2PI + z_samples.pow(2)).sum(dim=-1)  # (K, n)
            log_q_z_given_x = -0.5 * (
                self.LOG2PI + log_var.unsqueeze(0)
                + (z_samples - mu.unsqueeze(0)) ** 2 / log_var.exp().unsqueeze(0)
            ).sum(dim=-1)  # (K, n)

            log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
            iwae = torch.logsumexp(log_w, dim=0) - math.log(K)
        return iwae.cpu()

    def state_dict(self):
        return {
            "type": "vae",
            "dim": self.dim,
            "latent_dim": self.latent_dim,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "log_likelihood_var": self.log_likelihood_var,
            "module": None if self.module is None else self.module.state_dict(),
            "fitted": self._fitted,
        }

    def load_state_dict(self, state):
        if state.get("type") != "vae":
            raise ValueError(f"State type mismatch: {state.get('type')}")
        self.dim = int(state["dim"])
        self.latent_dim = int(state["latent_dim"])
        self.hidden = int(state["hidden"])
        self.epochs = int(state["epochs"])
        self.batch_size = int(state["batch_size"])
        self.lr = float(state["lr"])
        self.log_likelihood_var = float(state["log_likelihood_var"])
        self._build()
        if state["module"] is not None:
            self.module.load_state_dict(state["module"])
        self._fitted = bool(state["fitted"])


# ── 3.4 RealNVP normalising flow ─────────────────────────────────


class _CouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP.

    Given a binary mask m \\in {0, 1}^d, the transform is:
        y_masked   = x_masked
        y_passive  = x_passive * exp(s(x_masked)) + t(x_masked)
    where ``passive`` denotes positions where the mask is 0.
    """

    def __init__(self, dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.float().unsqueeze(0))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x):
        """x -> y, log|det(dy/dx)|."""
        x_masked = x * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=-1)
        # Tanh-bound the scale so exp(s) stays in a numerically friendly range.
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        y = x_masked + (1.0 - self.mask) * (x * s.exp() + t)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        """y -> x."""
        y_masked = y * self.mask
        st = self.net(y_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        x = y_masked + (1.0 - self.mask) * ((y - t) * (-s).exp())
        return x


class _RealNVPModule(nn.Module):
    """Stack of alternating-mask coupling layers."""

    def __init__(self, dim: int, n_layers: int, hidden: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            # Alternate halves; equivalent to the "checkerboard" mask in 1D.
            if i % 2 == 0:
                mask[: dim // 2] = 1.0
            else:
                mask[dim // 2 :] = 1.0
            layers.append(_CouplingLayer(dim, hidden, mask))
        self.layers = nn.ModuleList(layers)
        self.dim = dim

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


class FCDv2_RealNVP(FederatedStyleAggregator):
    """RealNVP normalising flow over ``z_env``.

    Implemented from scratch because no flow library (nflows / normflows /
    zuko) is in the project's requirements; the model is small enough that
    pulling in a dependency is not warranted.

    ``log_prob`` is an exact log-likelihood (no approximation).
    """

    LOG2PI = math.log(2.0 * math.pi)

    def __init__(
        self,
        dim: int,
        n_layers: int = 6,
        hidden: int = 256,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__(dim)
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device
        self.module: Optional[_RealNVPModule] = None

    def _build(self):
        self.module = _RealNVPModule(self.dim, self.n_layers, self.hidden).to(self.device)

    def _base_log_prob(self, z):
        return -0.5 * (self.LOG2PI + z * z).sum(dim=-1)

    def fit(self, pseudo_samples, client_indices):
        self._check_input(pseudo_samples, client_indices)
        self._build()
        x = pseudo_samples.detach().to(self.device, dtype=torch.float32)
        n = x.shape[0]
        optim = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        self.module.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                batch = x[idx]
                z, log_det = self.module(batch)
                log_p_z = self._base_log_prob(z)
                nll = -(log_p_z + log_det).mean()
                optim.zero_grad()
                nll.backward()
                optim.step()
        self.module.eval()
        self._fitted = True

    def sample(self, n_samples, generator=None):
        if not self._fitted:
            raise RuntimeError("FCDv2_RealNVP must be fit() before use")
        z = torch.randn(n_samples, self.dim, generator=generator, device=self.device)
        with torch.no_grad():
            x = self.module.inverse(z)
        return x.cpu()

    def log_prob(self, z):
        if not self._fitted:
            raise RuntimeError("FCDv2_RealNVP must be fit() before use")
        x = z.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            z_base, log_det = self.module(x)
            log_p = self._base_log_prob(z_base) + log_det
        return log_p.cpu()

    def state_dict(self):
        return {
            "type": "realnvp",
            "dim": self.dim,
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "module": None if self.module is None else self.module.state_dict(),
            "fitted": self._fitted,
        }

    def load_state_dict(self, state):
        if state.get("type") != "realnvp":
            raise ValueError(f"State type mismatch: {state.get('type')}")
        self.dim = int(state["dim"])
        self.n_layers = int(state["n_layers"])
        self.hidden = int(state["hidden"])
        self.epochs = int(state["epochs"])
        self.batch_size = int(state["batch_size"])
        self.lr = float(state["lr"])
        self._build()
        if state["module"] is not None:
            self.module.load_state_dict(state["module"])
        self._fitted = bool(state["fitted"])


# ── Factory ──────────────────────────────────────────────────────


_AGGREGATOR_REGISTRY = {
    "gaussian": FCDv2_Gaussian,
    "gmm": FCDv2_GMM,
    "vae": FCDv2_VAE,
    "realnvp": FCDv2_RealNVP,
}


def build_aggregator(name: str, dim: int, hparam: Optional[dict] = None) -> FederatedStyleAggregator:
    """Instantiate an aggregator by short name from the config namespace."""
    name = name.lower()
    if name not in _AGGREGATOR_REGISTRY:
        raise ValueError(
            f"Unknown aggregator '{name}'. Choose from {list(_AGGREGATOR_REGISTRY)}"
        )
    hparam = hparam or {}
    cls = _AGGREGATOR_REGISTRY[name]
    device = hparam.get("fcdv2_aggregator_device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if cls is FCDv2_Gaussian:
        return cls(dim=dim)
    if cls is FCDv2_GMM:
        return cls(
            dim=dim,
            n_components=int(hparam.get("fcd_gmm_components", 8)),
            random_state=int(hparam.get("seed", 0)),
        )
    if cls is FCDv2_VAE:
        return cls(
            dim=dim,
            latent_dim=int(hparam.get("fcdv2_vae_latent_dim", 32)),
            epochs=int(hparam.get("fcdv2_vae_epochs", 50)),
            device=device,
        )
    if cls is FCDv2_RealNVP:
        return cls(
            dim=dim,
            n_layers=int(hparam.get("fcdv2_realnvp_layers", 6)),
            epochs=int(hparam.get("fcdv2_realnvp_epochs", 100)),
            device=device,
        )
    raise RuntimeError("unreachable")
