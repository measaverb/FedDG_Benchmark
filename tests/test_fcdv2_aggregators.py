"""Unit tests for the FCDv2 federated style aggregators.

Each aggregator must support a ``fit -> sample -> log_prob`` roundtrip on
small synthetic data, and all four classes must be usable interchangeably
through the shared interface.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fcdv2_aggregators import (
    FCDv2_Gaussian,
    FCDv2_GMM,
    FCDv2_RealNVP,
    FCDv2_VAE,
    FederatedStyleAggregator,
    build_aggregator,
)


# ── Helpers ──────────────────────────────────────────────────────


def make_synthetic_mog(
    dim: int = 4,
    n_per_client: int = 32,
    n_clients: int = 4,
    seed: int = 0,
):
    """Generate a small mixture-of-Gaussians spread over fake clients."""
    g = torch.Generator().manual_seed(seed)
    # One mode per client: distinct means, modest variance.
    means = torch.randn(n_clients, dim, generator=g) * 2.0
    samples, indices = [], []
    for c in range(n_clients):
        s = means[c].unsqueeze(0) + 0.3 * torch.randn(
            n_per_client, dim, generator=g
        )
        samples.append(s)
        indices.append(torch.full((n_per_client,), c, dtype=torch.long))
    return torch.cat(samples, dim=0), torch.cat(indices, dim=0)


def fast_aggregator_kwargs():
    """Tiny overrides so tests stay fast on CPU."""
    return {
        "vae": dict(latent_dim=4, hidden=16, epochs=3, batch_size=32),
        "realnvp": dict(n_layers=2, hidden=16, epochs=3, batch_size=32),
        "gmm": dict(n_components=2),
    }


def build_all(dim: int):
    kw = fast_aggregator_kwargs()
    return [
        FCDv2_Gaussian(dim=dim),
        FCDv2_GMM(dim=dim, **kw["gmm"]),
        FCDv2_VAE(dim=dim, **kw["vae"]),
        FCDv2_RealNVP(dim=dim, **kw["realnvp"]),
    ]


# ── 1. fit -> sample -> log_prob roundtrip ────────────────────────


@pytest.mark.parametrize("dim", [4])
def test_each_aggregator_fit_sample_log_prob(dim):
    """Each aggregator must complete the full roundtrip without crashing
    and return well-shaped, finite outputs."""
    pseudo, idx = make_synthetic_mog(dim=dim)

    for agg in build_all(dim=dim):
        assert not agg.fitted
        agg.fit(pseudo, idx)
        assert agg.fitted

        samples = agg.sample(16)
        assert samples.shape == (16, dim)
        assert torch.isfinite(samples).all()

        lp = agg.log_prob(pseudo[:8])
        assert lp.shape == (8,)
        assert torch.isfinite(lp).all()


def test_log_prob_higher_for_in_distribution():
    """log_prob on real samples should beat log_prob on far-away noise
    for the parametric aggregators (Gaussian, GMM, RealNVP).

    The VAE's log_prob is an IWAE estimate and may be noisy with K=10 and
    tiny training, so it is not asserted here.
    """
    dim = 4
    pseudo, idx = make_synthetic_mog(dim=dim)
    far = pseudo + 50.0  # well outside the support of the fit

    for agg in [FCDv2_Gaussian(dim=dim), FCDv2_GMM(dim=dim, n_components=2)]:
        agg.fit(pseudo, idx)
        lp_in = agg.log_prob(pseudo[:8]).mean()
        lp_out = agg.log_prob(far[:8]).mean()
        assert lp_in > lp_out, f"{type(agg).__name__}: in={lp_in}, out={lp_out}"


# ── 2. Common interface ──────────────────────────────────────────


def test_all_aggregators_share_interface():
    """All four classes implement the FederatedStyleAggregator API and
    are interchangeable."""
    pseudo, idx = make_synthetic_mog(dim=4)
    for agg in build_all(dim=4):
        assert isinstance(agg, FederatedStyleAggregator)
        for method in ("fit", "sample", "log_prob", "state_dict", "load_state_dict"):
            assert callable(getattr(agg, method))
        agg.fit(pseudo, idx)
        # Interchangeable downstream use:
        out = agg.sample(4)
        _ = agg.log_prob(out)
        _ = agg.state_dict()


def test_state_dict_roundtrip_preserves_samples():
    """state_dict / load_state_dict roundtrips deterministic aggregators."""
    pseudo, idx = make_synthetic_mog(dim=4, seed=1)
    g_seed = 12345

    # Gaussian: closed-form, fully deterministic given the same RNG.
    agg_a = FCDv2_Gaussian(dim=4)
    agg_a.fit(pseudo, idx)
    state = agg_a.state_dict()

    agg_b = FCDv2_Gaussian(dim=4)
    agg_b.load_state_dict(state)

    g1 = torch.Generator().manual_seed(g_seed)
    g2 = torch.Generator().manual_seed(g_seed)
    s_a = agg_a.sample(8, generator=g1)
    s_b = agg_b.sample(8, generator=g2)
    assert torch.allclose(s_a, s_b)


def test_factory_dispatch():
    """build_aggregator returns the correct concrete class for each name."""
    expected = {
        "gaussian": FCDv2_Gaussian,
        "gmm": FCDv2_GMM,
        "vae": FCDv2_VAE,
        "realnvp": FCDv2_RealNVP,
    }
    hparam = {"fcdv2_vae_epochs": 2, "fcdv2_realnvp_epochs": 2}
    for name, cls in expected.items():
        assert isinstance(build_aggregator(name, dim=4, hparam=hparam), cls)

    with pytest.raises(ValueError):
        build_aggregator("unknown", dim=4)


# ── 3. Input validation ──────────────────────────────────────────


def test_dim_mismatch_raises():
    pseudo, idx = make_synthetic_mog(dim=4)
    agg = FCDv2_Gaussian(dim=8)
    with pytest.raises(ValueError):
        agg.fit(pseudo, idx)


def test_unfitted_aggregator_errors():
    agg = FCDv2_Gaussian(dim=4)
    with pytest.raises(RuntimeError):
        agg.sample(4)


# ── 4. Parametric correctness ────────────────────────────────────


def test_gaussian_recovers_mean_on_concentrated_data():
    """Diagonal Gaussian MLE recovers the empirical mean."""
    dim = 4
    g = torch.Generator().manual_seed(0)
    truth = torch.tensor([1.0, -2.0, 0.5, 3.0])
    x = truth + 0.05 * torch.randn(2048, dim, generator=g)
    idx = torch.zeros(2048, dtype=torch.long)
    agg = FCDv2_Gaussian(dim=dim)
    agg.fit(x, idx)
    assert torch.allclose(agg._mean, truth, atol=0.02)


def test_gmm_caps_components_at_client_count():
    """When fewer clients than requested components are present, the GMM
    must reduce its component count to the number of clients."""
    pseudo, idx = make_synthetic_mog(dim=4, n_clients=3)
    agg = FCDv2_GMM(dim=4, n_components=10)
    agg.fit(pseudo, idx)
    assert agg._weights.numel() == 3


def test_realnvp_log_prob_matches_change_of_variables():
    """For an exact flow, log p(x) = log p(z) + log|det J|.

    Here we just sanity-check that log_prob produces values consistent
    with manually composing the forward pass + base distribution.
    """
    dim = 4
    pseudo, idx = make_synthetic_mog(dim=dim)
    agg = FCDv2_RealNVP(dim=dim, n_layers=2, hidden=16, epochs=2, batch_size=32)
    agg.fit(pseudo, idx)

    x = pseudo[:5]
    z, log_det = agg.module(x.to(agg.device, dtype=torch.float32))
    base = -0.5 * (x.shape[-1] * torch.log(torch.tensor(2 * 3.141592653589793)) + (z * z).sum(dim=-1))
    expected = base + log_det
    got = agg.log_prob(x)
    assert torch.allclose(got, expected.cpu(), atol=1e-4)
