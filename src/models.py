import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn import init
from transformers import (
    BertForSequenceClassification,
    BertModel,
    DistilBertForSequenceClassification,
    DistilBertModel,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Tokenizer,
)


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(
                            bottleneck,
                            name2,
                            fuse(module2, getattr(bottleneck, bn_name)),
                        )
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(
                        bottleneck.downsample[0], bottleneck.downsample[1]
                    )
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CNN(nn.Module):
    def __init__(self, input_shape, probabilistic=False):
        super(CNN, self).__init__()
        self.n_outputs = 512
        self.probabilistic = probabilistic
        if self.probabilistic:
            self.fc = nn.Linear(in_features=7 * 7 * 64, out_features=self.n_outputs * 2)
        else:
            self.fc = nn.Linear(in_features=7 * 7 * 64, out_features=self.n_outputs)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], out_channels=16, kernel_size=5, padding=2
            ),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            self.fc,
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, feature_dimension=2048, probabilistic=False):
        super(ResNet, self).__init__()
        self.probabilistic = probabilistic

        self.network = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        self.n_outputs = feature_dimension

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.fc = nn.Linear(self.network.fc.in_features, self.n_outputs * 2)
        else:
            self.network.fc = nn.Linear(self.network.fc.in_features, self.n_outputs)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class DenseNet(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        feature_dimension=2048,
        probabilistic=False,
        pretrained=True,  # noqa: FBT002
    ):
        super(DenseNet, self).__init__()
        self.probabilistic = probabilistic

        weights = (
            torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.network = torchvision.models.densenet121(weights=weights)
        self.n_outputs = feature_dimension

        # adapt number of channels
        nc = input_shape[0]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.classifier = nn.Linear(
                self.network.classifier.in_features, self.n_outputs * 2
            )
        else:
            self.network.classifier = nn.Linear(
                self.network.classifier.in_features, self.n_outputs
            )

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))


class GPT2LMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size

    def __call__(self, x):
        outputs = super().__call__(x)
        logits = outputs[0]  # [batch_size, seqlen, vocab_size]
        return logits


class GPT2Featurizer(GPT2Model):
    def __init__(self, config, probabilistic=False):
        self.probabilistic = probabilistic
        super().__init__(config)

    def init_probablistic(self):
        d = self.embed_dim
        self.lm = nn.Linear(in_features=d, out_features=2 * d)
        weight_init = torch.cat((torch.eye(d), torch.eye(d)), dim=0)
        weight_init = nn.parameter.Parameter(weight_init, requires_grad=True)
        self.lm.weight = weight_init

    @property
    def n_outputs(self):
        return self.embed_dim

    def __call__(self, x):
        outputs = super().__call__(x)
        hidden_states = outputs[0]  # [batch_size, seqlen, n_embd]
        if self.probabilistic:
            hidden_states = self.lm(hidden_states)
        return hidden_states


class GPT2FeaturizerLMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size
        self.transformer = GPT2Featurizer(config)

    def __call__(self, x):
        hidden_states = self.transformer(x)  # [batch_size, seqlen, n_embd]
        logits = self.lm_head(hidden_states)  # [batch_size, seqlen, vocab_size]
        return logits


class GeneDistrNet(nn.Module):
    def __init__(self, num_labels, input_size, hidden_size=4096):
        super(GeneDistrNet, self).__init__()
        self.num_labels = num_labels
        self.input_size = input_size
        self.latent_size = 4096
        self.genedistri = nn.Sequential(
            nn.Linear(input_size + self.num_labels, self.latent_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size, hidden_size),
            nn.ReLU(),
        )
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.5)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.genedistri(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_labels, rp_size=1024):
        super(Discriminator, self).__init__()
        self.features_pro = nn.Sequential(
            nn.Linear(rp_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        self.optimizer = None
        self.projection = nn.Linear(hidden_size + num_labels, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(
                torch.norm(self.projection.weight, keepdim=True)
            )

    def forward(self, y, z):
        feature = z.view(z.size(0), -1)
        feature = torch.cat([feature, y], dim=1)
        feature = self.projection(feature)
        logit = self.features_pro(feature)
        return logit


def code_gpt_py(probabilistic=False):
    name = "microsoft/CodeGPT-small-py"
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    model = GPT2FeaturizerLMHeadLogit.from_pretrained(name)
    model.resize_token_embeddings(len(tokenizer))
    featurizer = model.transformer
    featurizer.probabilistic = probabilistic
    classifier = model.lm_head
    model = (featurizer, classifier)
    return model


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features),
        )
    else:
        return torch.nn.Linear(in_features, out_features)


class BertClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.num_labels

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        return outputs


class BertFeaturizer(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[
            1
        ]  # get pooled output
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config, probabilistic=False):
        super().__init__(config)
        self.probabilistic = probabilistic

    @property
    def d_out(self):
        return 768

    @property
    def n_outputs(self):
        return self.d_out

    def init_probablistic(self):
        d = self.d_out
        self.probabilistic = True
        self.lm = nn.Linear(in_features=d, out_features=2 * d)
        weight_init = torch.cat((torch.eye(d), torch.eye(d)), dim=0)
        weight_init = nn.parameter.Parameter(weight_init, requires_grad=True)
        self.lm.weight = weight_init

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        if self.probabilistic:
            pooled_output = self.lm(pooled_output)


class PooledResNetBackbone(nn.Module):
    """ResNet backbone producing pooled 1D feature vectors (B, C_feat).

    Unlike ``SpatialResNetBackbone`` (used by FCD) which returns raw spatial
    maps, this variant applies global average pooling so that downstream
    ``FFDFeaturizer`` heads receive flat vectors.

    Supports ``resnet18`` (C_feat=512) and ``resnet50`` (C_feat=2048).
    BatchNorm layers are frozen during training for consistency with the
    federated setting.
    """

    _ARCH_TABLE = {
        "resnet18": (
            torchvision.models.resnet18,
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            512,
        ),
        "resnet50": (
            torchvision.models.resnet50,
            torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            2048,
        ),
    }

    def __init__(self, arch="resnet50"):
        super().__init__()
        if arch not in self._ARCH_TABLE:
            raise ValueError(
                f"Unsupported architecture '{arch}'. Choose from {list(self._ARCH_TABLE)}"
            )

        factory, weights, self.n_outputs = self._ARCH_TABLE[arch]
        network = factory(weights=weights)

        # Keep everything up to (and including) layer4 + avgpool; discard fc.
        self.features = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Return pooled feature vector: (B, C_feat)."""
        h = self.features(x)       # (B, C_feat, h, w)
        return self.gap(h).flatten(1)  # (B, C_feat)

    def train(self, mode=True):
        """Freeze BatchNorm layers during training."""
        super().train(mode)
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


class FFDFeaturizer(nn.Module):
    """Featurizer that disentangles invariant and environment features.

    Architecture: backbone → GAP (if spatial) → 3-layer MLP for each head.
    Each MLP: Linear→BN1d→ReLU → Linear→BN1d→ReLU → Linear
    """

    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.backbone = backbone
        self.probabilistic = False
        feature_dim = backbone.n_outputs
        self.n_outputs = proj_dim

        # 3-layer MLP projection head for invariant features
        self.h_inv = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim),
        )

        # 3-layer MLP projection head for environment features
        self.h_env = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim),
        )

    def forward(self, x):
        features = self.backbone(x)

        # Handle BatchNorm1d crash when batch size is 1 during training
        is_singleton = self.training and features.size(0) == 1
        if is_singleton:
            self.h_inv.eval()
            self.h_env.eval()

        z_inv = self.h_inv(features)

        if self.training:
            z_env = self.h_env(features)
            if is_singleton:
                self.h_inv.train()
                self.h_env.train()
            return z_inv, z_env

        return z_inv


class FFDModelWrapper(nn.Module):
    """Wraps FFDFeaturizer + classifier.

    Training: forward returns (logits, z_inv, z_env) for loss computation.
    Eval:     forward returns logits only for standard evaluation.
    """

    def __init__(self, featurizer, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, x):
        if self.training:
            z_inv, z_env = self.featurizer(x)
            logits = self.classifier(z_inv)
            return logits, z_inv, z_env
        else:
            z_inv = self.featurizer(x)
            return self.classifier(z_inv)


# ═══════════════════════════════════════════════════════════════════
# Federated Cyclic Disentanglement (FCD) components
# ═══════════════════════════════════════════════════════════════════


class SpatialResNetBackbone(nn.Module):
    """ResNet backbone producing spatial feature maps (B, C_feat, h, w).

    Unlike the standard ``ResNet`` class which applies global average pooling
    and a fully connected layer, this backbone returns the raw output of the
    final convolutional block so that downstream modules can operate on the
    spatial tensor directly.

    Supports ``resnet18`` (C_feat=512) and ``resnet50`` (C_feat=2048).
    BatchNorm layers are frozen during training for consistency with the
    federated setting.
    """

    _ARCH_TABLE = {
        "resnet18": (
            torchvision.models.resnet18,
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            512,
        ),
        "resnet50": (
            torchvision.models.resnet50,
            torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
            2048,
        ),
    }

    def __init__(self, arch="resnet18"):
        super().__init__()
        if arch not in self._ARCH_TABLE:
            raise ValueError(
                f"Unsupported architecture '{arch}'. Choose from {list(self._ARCH_TABLE)}"
            )

        factory, weights, self.n_outputs = self._ARCH_TABLE[arch]
        network = factory(weights=weights)

        # Keep everything up to (and including) layer4; discard avgpool + fc.
        self.features = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )

    def forward(self, x):
        """Return spatial feature map H: (B, C_feat, h, w)."""
        return self.features(x)

    def train(self, mode=True):
        """Freeze BatchNorm layers during training."""
        super().train(mode)
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


class FCDFeaturizer(nn.Module):
    """Featurizer that decomposes spatial features into invariant and
    environment subspaces for Federated Cyclic Disentanglement.

    Architecture:
        backbone → H (spatial)
        H → GAP → MLP → z_inv  (invariant semantics)
        H → GAP → MLP → z_env  (domain style)

    Each MLP is a 3-layer network:
        Linear(C_feat, C_feat) → LayerNorm → ReLU →
        Linear(C_feat, C_feat) → LayerNorm → ReLU →
        Linear(C_feat, proj_dim)

    LayerNorm is used instead of BatchNorm1d because it normalises
    per-sample across the feature dimension, eliminating batch-size
    dependencies (no singleton crash), federated running-stat divergence,
    and counterfactual BN contamination in the cyclic pathway.
    """

    def __init__(self, backbone, proj_dim=256):
        super().__init__()
        self.backbone = backbone
        self.probabilistic = False
        feat_dim = backbone.n_outputs
        self.n_outputs = proj_dim

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 3-layer MLP projection head for invariant features
        self.h_inv = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

        # 3-layer MLP projection head for environment features
        self.h_env = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        """Forward pass.

        Training: returns (z_inv, z_env, H) where H is the spatial map.
        Eval:     returns z_inv only (environment head is discarded).
        """
        H = self.backbone(x)  # (B, C_feat, h, w)
        pooled = self.gap(H).flatten(1)  # (B, C_feat)

        z_inv = self.h_inv(pooled)  # (B, proj_dim)

        if self.training:
            z_env = self.h_env(pooled)  # (B, proj_dim)
            return z_inv, z_env, H

        return z_inv

    def forward_inv_from_spatial(self, H):
        """Project a (possibly counterfactual) spatial map through h_inv.

        Used by the cyclic invariance pathway:
            H_cf → GAP → h_inv → ẑ_inv^cf
        """
        pooled = self.gap(H).flatten(1)
        return self.h_inv(pooled)

    def extract_env(self, x):
        """Extract z_env from raw input (convenience for stats collection).

        Unlike forward(), this returns z_env regardless of train/eval mode,
        so callers don't need to manually decompose backbone → GAP → h_env.
        """
        H = self.backbone(x)
        pooled = self.gap(H).flatten(1)
        return self.h_env(pooled)


class StyleEncoder(nn.Module):
    """Lightweight affine style encoder S_φ.

    Maps the latent environment vector z_env to predicted channel-wise
    affine parameters (γ̂, β̂) that approximate the spatial statistics
    (std, mean) of the backbone feature map H.

    Architecture:
        Linear(proj_dim → hidden) → ReLU → Linear(hidden → 2 × C_feat)
        → split → (softplus(γ̂), β̂)

    A ReLU hidden layer breaks the linear collapse that would otherwise
    occur between the final bare linear layer of h_env and a single-layer
    encoder.  Softplus on γ̂ enforces the positivity constraint implied by
    its role as a predicted standard deviation.
    """

    def __init__(self, z_dim, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, feat_dim * 2),
        )

    def forward(self, z_env):
        """Return (gamma_hat, beta_hat), each (B, C_feat).

        gamma_hat is passed through softplus to ensure positivity, since it
        predicts the channel-wise standard deviation σ(H) ≥ 0.
        """
        out = self.net(z_env)
        gamma_hat, beta_hat = out.chunk(2, dim=-1)
        gamma_hat = F.softplus(gamma_hat)
        return gamma_hat, beta_hat


class FCDModelWrapper(nn.Module):
    """Wraps FCDFeaturizer + StyleEncoder + Classifier for FCD.

    Training mode:
        forward(x) → (logits, z_inv, z_env, H, gamma_hat, beta_hat)
        All six outputs are needed by the four-phase client loss.

    Eval mode:
        forward(x) → logits
        The environment head and style encoder are discarded at inference.
    """

    def __init__(self, featurizer, classifier, style_encoder):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.style_encoder = style_encoder

    def forward(self, x):
        if self.training:
            z_inv, z_env, H = self.featurizer(x)
            logits = self.classifier(z_inv)
            gamma_hat, beta_hat = self.style_encoder(z_env)
            return logits, z_inv, z_env, H, gamma_hat, beta_hat
        else:
            z_inv = self.featurizer(x)
            return self.classifier(z_inv)


# ═══════════════════════════════════════════════════════════════════
# FCDv2 (Federated Cyclic Disentanglement v2) components
#
# Architecturally identical to FCD (same backbone + h_inv + h_env +
# style encoder + classifier). The differences are:
#   * the client applies a twin-view augmentation pipeline and computes
#     a five-term loss (L_task, L_inv, L_stat, L_cov_cross, L_var)
#   * the server fits a pluggable FederatedStyleAggregator
#     (gaussian / gmm / vae / realnvp) instead of a hardcoded GMM
#   * the prototype alignment loss L_align is dropped entirely
#
# Subclasses are kept as no-op renames so the FCDv2 code path is fully
# decoupled from FCD even though the modules behave identically.
# ═══════════════════════════════════════════════════════════════════


class FCDv2Featurizer(FCDFeaturizer):
    """Same architecture as FCDFeaturizer; renamed for FCDv2 clarity."""


class FCDv2ModelWrapper(FCDModelWrapper):
    """Same forward signature as FCDModelWrapper; renamed for FCDv2 clarity.

    FCDv2 calls this wrapper twice per step (once per augmented view) and
    composes the five-term loss at the client level.
    """
