import copy
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm

import wandb

from .client import *
from .dataset_bundle import *
from .models import *
from .utils import *


class FedAvg(object):
    def __init__(self, device, ds_bundle, hparam):
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = []
        self.hparam = hparam
        self.num_rounds = hparam["num_rounds"]
        self.fraction = hparam["fraction"]
        self.num_clients = 0
        self.test_dataloader = {}
        self._round = 0
        self.featurizer = None
        self.classifier = None

    def setup_model(self, model_file=None, start_epoch=0):
        """
        The model setup depends on the datasets.
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in tqdm(self.clients):
            client.setup_model(
                copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier)
            )

    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)

    def transmit_model(self, sampled_client_indices=None):
        """
        Description: Send the updated global model to selected/all clients.
        This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):

                client.update_model(self.model.state_dict())
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):

                self.clients[idx].update_model(self.model.state_dict())

    def sample_clients(self):
        """
        Description: Sample a subset of clients.
        Could be overriden if some methods require specific ways of sampling.
        """
        # sample clients randommly
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(
            np.random.choice(
                a=[i for i in range(self.num_clients)],
                size=num_sampled_clients,
                replace=False,
            ).tolist()
        )

        return sampled_client_indices

    def update_clients(self, sampled_client_indices):
        """
        Description: This method will call the client.fit methods.
        Usually doesn't need to override in the derived class.
        """

        def update_single_client(selected_index):
            self.clients[selected_index].fit(self._round)
            client_size = len(self.clients[selected_index])
            return client_size

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size

    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True

        for idx in tqdm(sampled_client_indices):
            self.clients[idx].client_evaluate()

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [
            len(self.clients[idx]) / selected_total_size
            for idx in sampled_client_indices
        ]
        self.aggregate(sampled_client_indices, mixing_coefficients)

    def evaluate_global_model(self, dataloader):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                if self._featurizer.probabilistic:
                    features_params = self.featurizer(data)
                    z_dim = int(features_params.shape[-1] / 2)
                    if len(features_params.shape) == 2:
                        z_mu = features_params[:, :z_dim]
                        z_sigma = F.softplus(features_params[:, z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)
                    elif len(features_params.shape) == 3:
                        flattened_features_params = features_params.view(
                            -1, features_params.shape[-1]
                        )
                        z_mu = flattened_features_params[:, :z_dim]
                        z_sigma = F.softplus(flattened_features_params[:, z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)
                    features = z_dist.rsample()
                    if len(features_params.shape) == 3:
                        features = features.view(data.shape[0], -1, z_dim)
                else:
                    features = self.featurizer(data)
                prediction = self.classifier(features)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))

            metric = self.ds_bundle.dataset.eval(
                y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu")
            )
            print(metric)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]

    def fit(self):
        """
        Description: Execute the whole process of the federated learning.
        """
        best_id_val_round = 0
        best_id_val_value = 0
        best_id_val_test_value = 0
        best_lodo_val_round = 0
        best_lodo_val_value = 0
        best_lodo_val_test_value = 0

        for r in range(self.num_rounds):
            print("num of rounds: {}".format(r))

            self.train_federated_model()
            metric_dict = {}
            id_flag = False
            lodo_flag = False
            id_t_val = 0
            t_val = 0
            for name, dataloader in self.test_dataloader.items():
                metric = self.evaluate_global_model(dataloader)
                metric_dict[name] = metric

                if name == "val":
                    lodo_val = metric[self.ds_bundle.key_metric]
                    if lodo_val > best_lodo_val_value:
                        best_lodo_val_round = r
                        best_lodo_val_value = lodo_val
                        lodo_flag = True
                if name == "id_val":
                    id_val = metric[self.ds_bundle.key_metric]
                    if id_val > best_id_val_value:
                        best_id_val_round = r
                        best_id_val_value = id_val
                        id_flag = True
                if name == "test":
                    t_val = metric[self.ds_bundle.key_metric]
                if name == "id_test":
                    id_t_val = metric[self.ds_bundle.key_metric]
            if lodo_flag:
                best_lodo_val_test_value = t_val
            if id_flag:
                best_id_val_test_value = id_t_val

            print(metric_dict)
            if self.hparam["wandb"]:
                wandb.log(metric_dict, step=self._round * self.hparam["local_epochs"])
            self.save_model(r)
            self._round += 1
        if self.hparam["wandb"]:
            if best_id_val_round != 0:
                wandb.summary["best_id_round"] = best_id_val_round
                wandb.summary["best_id_val_acc"] = best_id_val_test_value
            if best_lodo_val_round != 0:
                wandb.summary["best_lodo_round"] = best_lodo_val_round
                wandb.summary["best_lodo_val_acc"] = best_lodo_val_test_value
        else:
            print(f"best_id_round: {best_id_val_round}")
            print(f"best_id_val_acc: {best_id_val_test_value}")
            print(f"best_lodo_round: {best_lodo_val_round}")
            print(f"best_lodo_val_acc: {best_lodo_val_test_value}")
        self.transmit_model()

    def save_model(self, num_epoch):
        path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['id']}_{num_epoch}.pth"
        torch.save(self.model.state_dict(), path)


class FedDG(FedAvg):
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(
                copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier)
            )
            client.set_amploader(self.amploader)
        super().register_clients(clients)

    def set_amploader(self, amp_dataset):
        self.amploader = amp_dataset


class FedADGServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.gen_input_size = int(hparam["hparam5"])

    def setup_model(self, model_file, start_epoch):
        """
        The model setup depends on the datasets.
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self._generator = GeneDistrNet(
            num_labels=self.ds_bundle.n_classes,
            input_size=self.gen_input_size,
            hidden_size=self._featurizer.n_outputs,
        )
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.generator = nn.DataParallel(self._generator)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(
                copy.deepcopy(self._featurizer),
                copy.deepcopy(self._classifier),
                copy.deepcopy(self._generator),
            )

    def transmit_model(self, sampled_client_indices=None):
        """
        Description: Send the updated global model to selected/all clients.
        This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):

                client.update_model(
                    self.model.state_dict(), self._generator.state_dict()
                )

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)

        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].update_model(
                    self.model.state_dict(), self._generator.state_dict()
                )
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)

        averaged_weights = OrderedDict()
        averaged_generator_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            local_generator_weights = self.clients[idx].generator.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
            for key in self.generator.state_dict().keys():
                if it == 0:
                    averaged_generator_weights[key] = (
                        coefficients[it] * local_generator_weights[key]
                    )

                else:
                    averaged_generator_weights[key] += (
                        coefficients[it] * local_generator_weights[key]
                    )
        self.model.load_state_dict(averaged_weights)
        self.generator.load_state_dict(averaged_generator_weights)


class FedGMA(FedAvg):
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        num_sampled_clients = len(sampled_client_indices)
        delta = []
        sign_delta = ParamDict()
        self.model.to("cpu")
        last_weights = ParamDict(self.model.state_dict())
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            self.clients[idx].model.to("cpu")
            local_weights = ParamDict(self.clients[idx].model.state_dict())
            delta.append(coefficients[it] * (local_weights - last_weights))
            if it == 0:
                sum_delta = delta[it]
                sign_delta = delta[it].sign()
            else:
                sum_delta += delta[it]
                sign_delta += delta[it].sign()

        sign_delta /= num_sampled_clients
        abs_sign_delta = sign_delta.abs()

        mask = abs_sign_delta.ge(self.hparam["hparam1"])

        final_mask = mask + (0 - mask) * abs_sign_delta
        averaged_weights = (
            last_weights + self.hparam["hparam1"] * final_mask * sum_delta
        )
        self.model.load_state_dict(averaged_weights)


class ScaffoldServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.c = None

    def transmit_model(self, sampled_client_indices=None):
        """
        Description: Send the updated global model to selected/all clients.
        This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):

                client.update_model(self.model.state_dict())
                client.c_global = copy.deepcopy(self.c)
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):

                self.clients[idx].update_model(self.model.state_dict())
                self.clients[idx].c_global = copy.deepcopy(self.c)

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            if it == 0:
                c_local = self.clients[idx].c_local
            else:
                c_local += self.clients[idx].c_local
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]

                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.c = c_local / len(sampled_client_indices)
        self.model.load_state_dict(averaged_weights)


class AFLServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.group_weights = torch.zeros(self.ds_bundle.grouper.n_groups)
        train_set = self.ds_bundle.dataset.get_subset(
            "train", transform=self.ds_bundle.train_transform
        )
        train_g = self.ds_bundle.grouper.metadata_to_group(train_set.metadata_array)
        unique_groups, unique_counts = torch.unique(
            train_g, sorted=False, return_counts=True
        )
        counts = torch.zeros(self.ds_bundle.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.is_group_in_train = is_group_in_train
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights / self.group_weights.sum()

    def transmit_lambda(self, sampled_client_indices=None):
        """
        Description: Send the updated global model to selected/all clients.
        This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):

                client.update_vector(self.group_weights)
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].update_vector(self.group_weights)

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

    def update_lambda(self, sampled_client_indices):
        self.transmit_model(sampled_client_indices)
        total_loss_per_domain = torch.zeros_like(self.group_weights)
        total_samples_per_domain = torch.zeros_like(self.group_weights)

        # send the global model to selected clients
        for idx in tqdm(sampled_client_indices, leave=False):

            loss_per_domain, samples_per_domain = self.clients[idx].gradient_lambda()
            total_loss_per_domain += loss_per_domain
            total_samples_per_domain += samples_per_domain
        self.group_weights += torch.nan_to_num(
            self.hparam["hparam1"] * total_loss_per_domain / total_samples_per_domain,
            nan=0.0,
        )

        self.group_weights = euclidean_proj_simplex(self.group_weights)

        wandb.log(
            {
                "l0_lmda": torch.count_nonzero(
                    self.group_weights[self.group_weights > 0.001]
                )
            },
            step=self._round * self.hparam["local_epochs"],
        )

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        self.transmit_lambda(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [
            len(self.clients[idx]) / selected_total_size
            for idx in sampled_client_indices
        ]
        self.aggregate(sampled_client_indices, mixing_coefficients)

        self.update_lambda(sampled_client_indices)


class FFDServer(FedAvg):
    """Server for Federated Feature Disentanglement.

    Manages: model aggregation, global prototype aggregation, warm-up scheduling.
    """

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.ffd_alpha = hparam.get("ffd_alpha", 1.0)
        self.ffd_warmup_rounds = hparam.get("ffd_warmup_rounds", 10)
        self.global_prototypes = {}

    def setup_model(self, model_file=None, start_epoch=0):
        assert self._round == 0
        from .models import Classifier, FFDFeaturizer, FFDModelWrapper, PooledResNetBackbone

        arch = self.hparam.get("ffd_backbone", "resnet50")
        proj_dim = self.hparam.get("ffd_proj_dim", 128)

        backbone = PooledResNetBackbone(arch=arch)
        self._featurizer = FFDFeaturizer(backbone, proj_dim=proj_dim)

        n_classes = self.ds_bundle.dataset.n_classes
        self._classifier = Classifier(self._featurizer.n_outputs, n_classes)

        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(
            FFDModelWrapper(self._featurizer, self._classifier)
        )

        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)


    def register_clients(self, clients):
        """Override: send FFD featurizer (not plain backbone) to clients."""
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in tqdm(self.clients):
            client.setup_model(
                copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier)
            )

    # ── Warm-up scheduling ────────────────────────────────────────

    def _compute_warmup_alpha(self):
        """Linear ramp: α(t) = min(1, t / warmup_rounds) * ffd_alpha."""
        if self.ffd_warmup_rounds <= 0:
            return self.ffd_alpha
        ramp = min(1.0, self._round / max(self.ffd_warmup_rounds, 1))
        return ramp * self.ffd_alpha

    # ── Model / prototype transmission ────────────────────────────

    def transmit_model(self, sampled_client_indices=None):
        """Send model weights + global prototypes + warm-up alpha to clients."""
        super().transmit_model(sampled_client_indices)

        current_alpha = self._compute_warmup_alpha()
        targets = (
            self.clients
            if sampled_client_indices is None
            else [self.clients[idx] for idx in sampled_client_indices]
        )
        for client in targets:
            client.set_prototypes(self.global_prototypes, current_alpha=current_alpha)

    # ── Client training ────────────────────────────────────────────

    def update_clients(self, sampled_client_indices):
        """Train sampled clients and aggregate per-component losses."""
        selected_total_size = 0
        loss_keys = ["total", "task", "align", "var", "cov"]
        agg_losses = {k: 0.0 for k in loss_keys}

        for idx in tqdm(sampled_client_indices, leave=False):
            client_losses = self.clients[idx].fit(self._round)
            client_size = len(self.clients[idx])
            selected_total_size += client_size
            if client_losses is not None:
                for k in loss_keys:
                    agg_losses[k] += client_losses.get(k, 0.0) * client_size

        # Log weighted-average losses across all sampled clients
        if selected_total_size > 0 and self.hparam.get("wandb", False):
            import wandb

            wandb.log(
                {
                    f"server/loss_{k}": agg_losses[k] / selected_total_size
                    for k in loss_keys
                },
                step=self._round,
            )

        return selected_total_size

    # ── Prototype aggregation ─────────────────────────────────────

    def aggregate_prototypes(self, sampled_client_indices):
        """Weighted average of client prototypes (by dataset size)."""
        proto_sums = {}
        proto_weights = {}
        for idx in sampled_client_indices:
            client = self.clients[idx]
            if not hasattr(client, "local_prototypes") or not client.local_prototypes:
                continue
            w = len(client)
            for c, proto in client.local_prototypes.items():
                if c not in proto_sums:
                    proto_sums[c] = proto.clone() * w
                    proto_weights[c] = w
                else:
                    proto_sums[c] += proto * w
                    proto_weights[c] += w

        self.global_prototypes = {
            c: proto_sums[c] / proto_weights[c] for c in proto_sums
        }

    # ── Overridden training loop ──────────────────────────────────

    def train_federated_model(self):
        """FedAvg + prototype aggregation after each round."""
        sampled_client_indices = self.sample_clients()
        self.transmit_model(sampled_client_indices)
        selected_total_size = self.update_clients(sampled_client_indices)

        mixing_coefficients = [
            len(self.clients[idx]) / selected_total_size
            for idx in sampled_client_indices
        ]
        self.aggregate(sampled_client_indices, mixing_coefficients)

        # Aggregate prototypes after model aggregation
        self.aggregate_prototypes(sampled_client_indices)

    # ── Evaluation override ───────────────────────────────────────

    def evaluate_global_model(self, dataloader):
        """Evaluate using FFDModelWrapper in eval mode (returns logits only)."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            metadata = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)

                # FFDModelWrapper in eval mode returns logits directly
                prediction = self.model(data)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)

                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))

            metric = self.ds_bundle.dataset.eval(
                y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu")
            )
            print(metric)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]


class FCDServer(FedAvg):
    """Server for Federated Cyclic Disentanglement.

    Manages:
      - Model aggregation (FedAvg).
      - Global semantic prototypes (per-class z_inv means).
      - Global Environment GMM fitted from aggregated client statistics.
      - Warm-up scheduling for prototype alignment weight α(t).
    """

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.fcd_alpha = hparam.get("fcd_alpha", 1.0)
        self.fcd_warmup_rounds = hparam.get("fcd_warmup_rounds", 10)
        self.fcd_gmm_components = hparam.get("fcd_gmm_components", 8)
        self.fcd_gmm_pseudo_samples = hparam.get("fcd_gmm_pseudo_samples", 200)
        self.global_prototypes = {}
        self.global_gmm_params = None  # dict with 'weights', 'means', 'covariances'
        self.global_unimodal_params = None  # dict with 'mean', 'covariance'

    def setup_model(self, model_file=None, start_epoch=0):
        assert self._round == 0
        from .models import (
            Classifier,
            FCDFeaturizer,
            FCDModelWrapper,
            SpatialResNetBackbone,
            StyleEncoder,
        )

        arch = self.hparam.get("fcd_backbone", "resnet18")
        proj_dim = self.hparam.get("fcd_proj_dim", 256)

        backbone = SpatialResNetBackbone(arch=arch)
        self._featurizer = FCDFeaturizer(backbone, proj_dim=proj_dim)

        n_classes = self.ds_bundle.dataset.n_classes
        self._classifier = Classifier(self._featurizer.n_outputs, n_classes)
        self._style_encoder = StyleEncoder(z_dim=proj_dim, feat_dim=backbone.n_outputs)

        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(
            FCDModelWrapper(self._featurizer, self._classifier, self._style_encoder)
        )

        if model_file:
            self.model.load_state_dict(torch.load(model_file, weights_only=True))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        """Send FCD featurizer, classifier, and style encoder to each client."""
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in tqdm(self.clients):
            client.setup_model(
                copy.deepcopy(self._featurizer),
                copy.deepcopy(self._classifier),
                copy.deepcopy(self._style_encoder),
            )

    # ── Warm-up scheduling ────────────────────────────────────────

    def _compute_warmup_alpha(self):
        """Linear ramp: α(t) = min(1, t / warmup_rounds) * fcd_alpha.

        Note: this is called inside transmit_model(), which runs *before*
        update_clients() within a round.  Make sure self._round has been
        incremented by the parent class before this point, or the ramp
        will lag by one round.
        """
        if self.fcd_warmup_rounds <= 0:
            return self.fcd_alpha
        ramp = min(1.0, self._round / max(self.fcd_warmup_rounds, 1))
        return ramp * self.fcd_alpha

    # ── Model / prototype / GMM transmission ──────────────────────

    def transmit_model(self, sampled_client_indices=None):
        """Send model weights + prototypes + alpha + GMM to clients."""
        super().transmit_model(sampled_client_indices)

        current_alpha = self._compute_warmup_alpha()
        targets = (
            self.clients
            if sampled_client_indices is None
            else [self.clients[idx] for idx in sampled_client_indices]
        )
        for client in targets:
            client.set_prototypes(self.global_prototypes, current_alpha=current_alpha)
            client.set_gmm_params(self.global_gmm_params)
            client.set_global_unimodal_params(self.global_unimodal_params)

    # ── Client training ────────────────────────────────────────────

    def update_clients(self, sampled_client_indices):
        """Train sampled clients and aggregate per-component losses."""
        selected_total_size = 0
        loss_keys = [
            "total",
            "task",
            "cls",
            "stat",
            "align",
            "cov",
            "cov_internal",
            "cov_cross",
            "cf",
        ]
        agg_losses = {k: 0.0 for k in loss_keys}

        for idx in tqdm(sampled_client_indices, leave=False):
            client_losses = self.clients[idx].fit(self._round)
            client_size = len(self.clients[idx])
            selected_total_size += client_size
            if client_losses is not None:
                for k in loss_keys:
                    agg_losses[k] += client_losses.get(k, 0.0) * client_size

        # Log weighted-average losses
        if selected_total_size > 0 and self.hparam.get("wandb", False):
            wandb.log(
                {
                    f"server/loss_{k}": agg_losses[k] / selected_total_size
                    for k in loss_keys
                },
                step=self._round * self.hparam.get("local_epochs", 1),
            )

        return selected_total_size

    # ── Prototype aggregation ─────────────────────────────────────

    def aggregate_prototypes(self, sampled_client_indices):
        """Weighted average of client prototypes (by dataset size)."""
        proto_sums = {}
        proto_weights = {}
        for idx in sampled_client_indices:
            client = self.clients[idx]
            if not hasattr(client, "local_prototypes") or not client.local_prototypes:
                continue
            w = len(client)
            for c, proto in client.local_prototypes.items():
                if c not in proto_sums:
                    proto_sums[c] = proto.clone() * w
                    proto_weights[c] = w
                else:
                    proto_sums[c] += proto * w
                    proto_weights[c] += w

        self.global_prototypes = {
            c: proto_sums[c] / proto_weights[c] for c in proto_sums
        }

    # ── GMM aggregation ───────────────────────────────────────────

    def aggregate_gmm(self, sampled_client_indices):
        """Fit a global Environment GMM from client-level statistics.

        Each client transmits only (μ_env, Σ_env) — no raw data.

        Implementation note: the spec describes Federated EM, but this
        implementation uses a pseudo-sample approximation — each client's
        Gaussian is sampled to produce synthetic points, and a standard
        sklearn GMM is fitted on the pooled pseudo-samples.  This is
        equivalent when client distributions are well-approximated by
        single Gaussians and the number of pseudo-samples is sufficient.
        True federated EM (e.g. iterative sufficient-statistic exchange)
        could be substituted here without changing the client interface.
        """
        from sklearn.mixture import GaussianMixture

        all_pseudo_samples = []
        for idx in sampled_client_indices:
            client = self.clients[idx]
            if not hasattr(client, "local_env_stats") or client.local_env_stats is None:
                continue

            mu = client.local_env_stats["mean"]
            cov = client.local_env_stats["covariance"]

            # Generate pseudo-samples from this client's Gaussian
            rng = np.random.default_rng(seed=self._round * 1000 + idx)
            pseudo = rng.multivariate_normal(mu, cov, size=self.fcd_gmm_pseudo_samples)
            all_pseudo_samples.append(pseudo)

        if len(all_pseudo_samples) == 0:
            return  # No stats available yet

        X = np.concatenate(all_pseudo_samples, axis=0)

        n_components = min(self.fcd_gmm_components, len(all_pseudo_samples))
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            max_iter=100,
            random_state=self._round,
        )
        gmm.fit(X)

        # Store GMM parameters as numpy arrays for transmission.
        # sklearn's diag covariance is (M, d); expand to (M, d, d) for
        # compatibility with torch.distributions.MultivariateNormal in
        # the client's _sample_from_gmm.
        covariances = np.array(
            [np.diag(gmm.covariances_[m]) for m in range(n_components)]
        )

        self.global_gmm_params = {
            "weights": gmm.weights_,  # (M,)
            "means": gmm.means_,  # (M, d)
            "covariances": covariances,  # (M, d, d)
        }

    # ── Unimodal Gaussian aggregation (ablation condition 3) ──────

    def aggregate_unimodal_gaussian(self, sampled_client_indices):
        """Compute naive global unimodal Gaussian from client env stats.

        Weighted average of per-client (μ_env, Σ_env) by dataset size.
        Used as an ablation baseline to demonstrate that a single Gaussian
        over-smooths the multi-modal environment distribution.
        """
        mus, covs, weights = [], [], []
        for idx in sampled_client_indices:
            client = self.clients[idx]
            if not hasattr(client, "local_env_stats") or client.local_env_stats is None:
                continue
            w = len(client)
            mus.append(client.local_env_stats["mean"] * w)
            covs.append(client.local_env_stats["covariance"] * w)
            weights.append(w)

        if not weights:
            self.global_unimodal_params = None
            return

        total_w = sum(weights)
        mu_global = sum(mus) / total_w
        cov_global = sum(covs) / total_w
        self.global_unimodal_params = {"mean": mu_global, "covariance": cov_global}

    # ── Overridden training loop ──────────────────────────────────

    def train_federated_model(self):
        """FedAvg + prototype aggregation + GMM fitting after each round."""
        sampled_client_indices = self.sample_clients()
        self.transmit_model(sampled_client_indices)
        selected_total_size = self.update_clients(sampled_client_indices)

        mixing_coefficients = [
            len(self.clients[idx]) / selected_total_size
            for idx in sampled_client_indices
        ]
        self.aggregate(sampled_client_indices, mixing_coefficients)

        self.aggregate_prototypes(sampled_client_indices)
        self.aggregate_gmm(sampled_client_indices)
        self.aggregate_unimodal_gaussian(sampled_client_indices)

    # ── Evaluation override ───────────────────────────────────────

    def evaluate_global_model(self, dataloader):
        """Evaluate using FCDModelWrapper in eval mode (returns logits only)."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            metadata = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)

                # FCDModelWrapper in eval mode returns logits directly
                prediction = self.model(data)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)

                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))

            metric = self.ds_bundle.dataset.eval(
                y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu")
            )
            print(metric)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]


# ═══════════════════════════════════════════════════════════════════
# FCDv2 server (parallel to FCDServer; not a replacement)
# ═══════════════════════════════════════════════════════════════════


class FCDv2Server(FedAvg):
    """Server for FCDv2.

    Differences vs FCDServer:
      * No prototype aggregation / broadcast (L_align is dropped).
      * Style aggregator is pluggable via ``aggregator_type`` config:
        ``gaussian | gmm | vae | realnvp``.
      * Each round, per-client (mu_i, Sigma_i) -> deterministic
        pseudo-samples -> ``aggregator.fit`` -> aggregator broadcast.
    """

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        from .fcdv2_aggregators import build_aggregator

        self.fcd_gmm_pseudo_samples = hparam.get("fcd_gmm_pseudo_samples", 200)
        self.aggregator_type = hparam.get("aggregator_type", "gmm")

        proj_dim = int(hparam.get("fcd_proj_dim", 256))
        self.aggregator = build_aggregator(
            self.aggregator_type, dim=proj_dim, hparam=hparam
        )

    # ── Model setup ───────────────────────────────────────────────

    def setup_model(self, model_file=None, start_epoch=0):
        from .models import (
            Classifier,
            FCDv2Featurizer,
            FCDv2ModelWrapper,
            SpatialResNetBackbone,
            StyleEncoder,
        )

        assert self._round == 0
        arch = self.hparam.get("fcd_backbone", "resnet18")
        proj_dim = self.hparam.get("fcd_proj_dim", 256)

        backbone = SpatialResNetBackbone(arch=arch)
        self._featurizer = FCDv2Featurizer(backbone, proj_dim=proj_dim)

        n_classes = self.ds_bundle.dataset.n_classes
        self._classifier = Classifier(self._featurizer.n_outputs, n_classes)
        self._style_encoder = StyleEncoder(z_dim=proj_dim, feat_dim=backbone.n_outputs)

        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(
            FCDv2ModelWrapper(self._featurizer, self._classifier, self._style_encoder)
        )

        if model_file:
            self.model.load_state_dict(torch.load(model_file, weights_only=True))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in tqdm(self.clients):
            client.setup_model(
                copy.deepcopy(self._featurizer),
                copy.deepcopy(self._classifier),
                copy.deepcopy(self._style_encoder),
            )

    # ── Transmission ──────────────────────────────────────────────

    def transmit_model(self, sampled_client_indices=None):
        super().transmit_model(sampled_client_indices)
        targets = (
            self.clients
            if sampled_client_indices is None
            else [self.clients[idx] for idx in sampled_client_indices]
        )
        for client in targets:
            client.set_aggregator(copy.deepcopy(self.aggregator))

    # ── Aggregator fitting ────────────────────────────────────────

    def aggregate_style(self, sampled_client_indices):
        """Fit the style aggregator on per-client deterministic pseudo-samples.

        Mirrors FCDServer's pseudo-sample protocol: each client's local
        Gaussian (mu_i, Sigma_i) is sampled with seed ``round*1000 + idx``,
        and the pooled samples are fed to the aggregator's ``fit``.
        """
        all_samples = []
        all_indices = []
        for idx in sampled_client_indices:
            client = self.clients[idx]
            if not hasattr(client, "local_env_stats") or client.local_env_stats is None:
                continue
            mu = client.local_env_stats["mean"]
            cov = client.local_env_stats["covariance"]
            rng = np.random.default_rng(seed=self._round * 1000 + idx)
            pseudo = rng.multivariate_normal(mu, cov, size=self.fcd_gmm_pseudo_samples)
            all_samples.append(torch.tensor(pseudo, dtype=torch.float32))
            all_indices.append(torch.full((len(pseudo),), idx, dtype=torch.long))

        if not all_samples:
            return  # Round 0 -- no client stats yet.

        x = torch.cat(all_samples, dim=0)
        i = torch.cat(all_indices, dim=0)
        self.aggregator.fit(x, i)

    # ── Per-round training loop ───────────────────────────────────

    def update_clients(self, sampled_client_indices):
        loss_keys = ["total", "task", "inv", "stat", "cov_cross", "var", "cf"]
        agg_losses = {k: 0.0 for k in loss_keys}
        selected_total_size = 0

        for idx in tqdm(sampled_client_indices, leave=False):
            client_losses = self.clients[idx].fit(self._round)
            client_size = len(self.clients[idx])
            selected_total_size += client_size
            if client_losses is not None:
                for k in loss_keys:
                    agg_losses[k] += client_losses.get(k, 0.0) * client_size

        if selected_total_size > 0 and self.hparam.get("wandb", False):
            wandb.log(
                {
                    f"server/loss_{k}": agg_losses[k] / selected_total_size
                    for k in loss_keys
                },
                step=self._round * self.hparam.get("local_epochs", 1),
            )
        return selected_total_size

    def train_federated_model(self):
        sampled_client_indices = self.sample_clients()
        self.transmit_model(sampled_client_indices)
        selected_total_size = self.update_clients(sampled_client_indices)

        mixing_coefficients = [
            len(self.clients[idx]) / selected_total_size
            for idx in sampled_client_indices
        ]
        self.aggregate(sampled_client_indices, mixing_coefficients)
        self.aggregate_style(sampled_client_indices)

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate_global_model(self, dataloader):
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            metadata = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                prediction = self.model(data)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)

                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))

            metric = self.ds_bundle.dataset.eval(
                y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu")
            )
            print(metric)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]
