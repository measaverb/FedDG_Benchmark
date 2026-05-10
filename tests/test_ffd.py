"""Unit tests for FFD: model shapes, losses, and client integration."""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import Classifier, FFDFeaturizer, FFDModelWrapper

# ── Helpers ──────────────────────────────────────────────────────


class DummyBackbone(nn.Module):
    """Minimal backbone that mimics ResNet output (flat feature vector)."""

    def __init__(self, n_outputs=512):
        super().__init__()
        self.n_outputs = n_outputs
        self.fc = nn.Linear(3 * 32 * 32, n_outputs)
        self.probabilistic = False

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


@pytest.fixture
def backbone():
    return DummyBackbone(n_outputs=512)


@pytest.fixture
def proj_dim():
    return 128


# ── Model shape tests ────────────────────────────────────────────


class TestFFDFeaturizerShapes:
    def test_training_output_shapes(self, backbone, proj_dim):
        """In training mode, featurizer returns (z_inv, z_env) both of shape (B, proj_dim)."""
        feat = FFDFeaturizer(backbone, proj_dim=proj_dim)
        feat.train()
        x = torch.randn(8, 3, 32, 32)
        z_inv, z_env = feat(x)
        assert z_inv.shape == (8, proj_dim)
        assert z_env.shape == (8, proj_dim)

    def test_eval_output_shape(self, backbone, proj_dim):
        """In eval mode, featurizer returns z_inv only."""
        feat = FFDFeaturizer(backbone, proj_dim=proj_dim)
        feat.eval()
        x = torch.randn(4, 3, 32, 32)
        out = feat(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, proj_dim)

    def test_auto_detects_feature_dim(self, proj_dim):
        """FFDFeaturizer auto-detects feature_dim from backbone.n_outputs."""
        bb = DummyBackbone(n_outputs=256)
        feat = FFDFeaturizer(bb, proj_dim=proj_dim)
        # First layer of h_inv should have in_features == 256
        first_linear = feat.h_inv[0]
        assert first_linear.in_features == 256


class TestFFDModelWrapper:
    def test_training_returns_tuple(self, backbone, proj_dim):
        """Training mode returns (logits, z_inv, z_env)."""
        feat = FFDFeaturizer(backbone, proj_dim=proj_dim)
        clf = Classifier(proj_dim, 7)
        wrapper = FFDModelWrapper(feat, clf)
        wrapper.train()

        x = torch.randn(8, 3, 32, 32)
        out = wrapper(x)
        assert isinstance(out, tuple)
        assert len(out) == 3
        logits, z_inv, z_env = out
        assert logits.shape == (8, 7)
        assert z_inv.shape == (8, proj_dim)
        assert z_env.shape == (8, proj_dim)

    def test_eval_returns_tensor(self, backbone, proj_dim):
        """Eval mode returns logits tensor only (not tuple)."""
        feat = FFDFeaturizer(backbone, proj_dim=proj_dim)
        clf = Classifier(proj_dim, 7)
        wrapper = FFDModelWrapper(feat, clf)
        wrapper.eval()

        x = torch.randn(4, 3, 32, 32)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 7)


# ── VICReg loss tests ─────────────────────────────────────────────


class TestVICRegLosses:
    """Test the static loss methods on FFDClient."""

    @pytest.fixture(autouse=True)
    def import_client(self):
        from src.client import FFDClient

        self.FFDClient = FFDClient

    def test_variance_loss_zero_when_std_above_one(self):
        """If all dims have std > 1, variance loss should be 0."""
        z = torch.randn(64, 128) * 2.0  # std ≈ 2
        loss = self.FFDClient._variance_loss(z)
        assert loss.item() == pytest.approx(0.0, abs=0.05)

    def test_variance_loss_positive_when_collapsed(self):
        """If features are near-constant, variance loss should be positive."""
        z = torch.ones(32, 128) + torch.randn(32, 128) * 0.01
        loss = self.FFDClient._variance_loss(z)
        assert loss.item() > 0.5

    def test_covariance_loss_zero_for_identity(self):
        """Diagonal covariance → off-diagonal loss ≈ 0."""
        # Use orthogonal features: each sample activates one dim
        z = torch.eye(32, 128)
        loss = self.FFDClient._covariance_loss(z)
        assert loss.item() < 0.01

    def test_covariance_loss_positive_for_correlated(self):
        """Highly correlated features → positive covariance loss."""
        base = torch.randn(64, 1)
        z = base.expand(64, 128) + torch.randn(64, 128) * 0.01
        loss = self.FFDClient._covariance_loss(z)
        assert loss.item() > 0.1

    def test_cross_covariance_positive_for_correlated(self):
        """Same features → positive cross-covariance loss."""
        z = torch.randn(64, 128)
        loss = self.FFDClient._cross_covariance_loss(z, z)
        assert loss.item() > 0.0

    def test_cross_covariance_near_zero_for_independent(self):
        """Independent features → cross-covariance ≈ 0."""
        torch.manual_seed(42)
        z1 = torch.randn(1000, 128)
        z2 = torch.randn(1000, 128)
        loss = self.FFDClient._cross_covariance_loss(z1, z2)
        # With enough samples, should be close to 0
        assert loss.item() < 0.2


# ── Client step scalar return test ────────────────────────────────


class TestFFDClientStepReturn:
    def test_step_returns_scalar(self):
        """FFDClient.step must return a float/int (not dict) for ERM.fit compatibility."""
        from src.client import FFDClient

        # We can't easily create a full client, so we test the return type
        # by checking that step returns a number (via the implementation)
        # This is a design contract test
        assert hasattr(FFDClient, "step"), "FFDClient must have step method"

        # Verify the method signature doesn't return a dict by inspecting
        # the source code (static check)
        import inspect

        source = inspect.getsource(FFDClient.step)
        assert "return total_loss" in source, "step() should return scalar total_loss"
        assert "return {" not in source, "step() should NOT return a dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
