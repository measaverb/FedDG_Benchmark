import copy
import os
import random

import torch
import torch.autograd as autograd
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_train_loader
from wilds.common.metrics.loss import ElementwiseLoss
from wilds.common.utils import split_into_groups

import wandb
from src.models import Discriminator
from src.utils import *


class ERM(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """

    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        """Client object is initiated by the center server."""
        self.client_id = client_id
        self.device = device
        self.featurizer = None
        self.classifier = None
        self.model = None
        self.dataset = dataset
        self.ds_bundle = ds_bundle
        self.hparam = hparam
        self.n_groups_per_batch = hparam["n_groups_per_batch"]
        self.local_epochs = self.hparam["local_epochs"]
        self.batch_size = self.hparam["batch_size"]
        self.optimizer_name = self.hparam["optimizer"]
        self.optim_config = self.hparam["optimizer_config"]
        try:
            self.scheduler_name = self.hparam["scheduler"]
            self.scheduler_config = self.hparam["scheduler_config"]
        except KeyError:
            self.scheduler_name = "torch.optim.lr_scheduler.ConstantLR"
            self.scheduler_config = {"factor": 1, "total_iters": 1}
        self.dataloader = get_train_loader(
            self.loader_type,
            self.dataset,
            batch_size=self.batch_size,
            uniform_over_groups=None,
            grouper=self.ds_bundle.grouper,
            distinct_groups=False,
            n_groups_per_batch=self.n_groups_per_batch,
            num_workers=self.hparam.get("num_workers", 16),
            pin_memory=self.hparam.get("num_workers", 16) > 0,
            persistent_workers=self.hparam.get("num_workers", 16) > 0,
        )
        self.saved_optimizer = False
        self.opt_dict_path = "{}opt_dict/client_{}.pt".format(
            self.hparam["data_path"], self.client_id
        )
        self.sch_dict_path = "{}sch_dict/client_{}.pt".format(
            self.hparam["data_path"], self.client_id
        )
        if os.path.exists(self.opt_dict_path):
            os.remove(self.opt_dict_path)

    def setup_model(self, featurizer, classifier):
        self._featurizer = featurizer
        self._classifier = classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))

    @property
    def loader_type(self):
        return "standard"

    def update_model(self, model_dict):
        self.model.load_state_dict(model_dict)

    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = eval(self.optimizer_name)(
            self.model.parameters(), **self.optim_config
        )
        self.scheduler = eval(self.scheduler_name)(
            self.optimizer, **self.scheduler_config
        )
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
            self.scheduler.load_state_dict(torch.load(self.sch_dict_path))

    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.scheduler.state_dict(), self.sch_dict_path)
        del self.scheduler, self.optimizer
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def fit(self, server_round):
        """Update local model using local dataset."""
        self.init_train()
        training_loss = 0.0
        for e in range(self.local_epochs):
            for batch in tqdm(self.dataloader):
                results = self.process_batch(batch)
                training_loss += self.step(results)
            if self.hparam["wandb"]:
                wandb.log(
                    {
                        "loss/{}".format(self.client_id): training_loss
                        / len(self.dataset)
                    },
                    step=server_round * self.local_epochs + e,
                )
        self.end_train()

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        outputs = self.model(x)
        # print(outputs.shape)
        results = {
            "g": g,
            "y_true": y_true,
            "y_pred": outputs,
            "metadata": metadata,
        }
        return results

    def step(self, results):
        # print(results['y_true'])
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        loss = self.ds_bundle.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )
        objective = loss.mean()
        total_loss = loss.sum().item()
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.dataset)


class IRM(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.penalty_weight = hparam["hparam1"]
        self.penalty_anneal_iters = hparam["hparam2"]
        self.scale = torch.tensor(1.0).to(self.device).requires_grad_()
        self.update_count = 0

    @property
    def loader_type(self):
        return "group"

    def step(self, results):
        unique_groups, group_indices, _ = split_into_groups(results["g"])
        n_groups_per_batch = unique_groups.numel()
        avg_loss = 0.0
        penalty = 0.0
        # torch.save(results['y_pred'], "pred.pt")
        # torch.save(results['y_true'], "true.pt")
        for (
            i_group
        ) in group_indices:  # Each element of group_indices is a list of indices
            # print(i_group)
            group_losses, _ = self.ds_bundle.loss.compute_flattened(
                results["y_pred"][i_group] * self.scale,
                results["y_true"][i_group],
                return_dict=False,
            )
            if group_losses.numel() > 0:
                avg_loss += group_losses.mean()
            penalty += self.irm_penalty(group_losses)
        avg_loss /= n_groups_per_batch
        penalty /= n_groups_per_batch
        if self.update_count >= self.penalty_anneal_iters:
            penalty_weight = self.penalty_weight
        else:
            penalty_weight = self.update_count / self.penalty_anneal_iters
        penalty_weight = 0.0
        # print(self.update_count, penalty_weight)
        objective = avg_loss + penalty * penalty_weight
        # print(avg_loss, penalty, objective)
        # wprint(avg_loss, penalty)
        if self.update_count == self.penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = eval(self.optimizer_name)(params, **self.optim_config)
        if objective.grad_fn is None:
            pass
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_count += 1
        return (results["y_pred"].shape)[0] * objective.item()

    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        del grad_1, grad_2
        return result


class VREx(IRM):
    def irm_penalty(self, losses):
        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        return penalty


class Fish(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.meta_lr = hparam["hparam1"]

    @property
    def loader_type(self):
        return "group"

    def fit(self, server_round):
        self.init_train()
        training_loss = 0.0
        for e in range(self.local_epochs):
            for batch in self.dataloader:
                training_loss += self.step(batch)
            if self.hparam["wandb"]:
                wandb.log(
                    {
                        "loss/{}".format(self.client_id): training_loss
                        / len(self.dataset)
                    },
                    step=server_round * self.local_epochs + e,
                )
        self.end_train()

    def step(self, batch):
        param_dict = ParamDict(copy.deepcopy(self.model.state_dict()))
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        unique_groups, group_indices, _ = split_into_groups(g)
        for (
            i_group
        ) in group_indices:  # Each element of group_indices is a list of indices
            # print(i_group)
            group_loss = self.ds_bundle.loss.compute(
                self.model(x[i_group]), y_true[i_group], return_dict=False
            )
            if group_loss.grad_fn is None:
                # print('jump')
                pass
            else:
                group_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        param_dict = param_dict + self.meta_lr * (
            ParamDict(self.model.state_dict()) - param_dict
        )
        self.model.load_state_dict(copy.deepcopy(param_dict))
        return (y_true.shape)[0] * group_loss.item()


class MMD(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.penalty_weight = hparam["hparam1"]

    @property
    def loader_type(self):
        return "group"

    def penalty(self, x, y):
        def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
            if x.dim() > 2:
                # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
                # we flatten to Tensors of size (*, feature dimensionality)
                x = x.view(-1, x.size(-1))
                y = y.view(-1, y.size(-1))

            def my_cdist(x1, x2):
                x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
                x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
                res = torch.addmm(
                    x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
                ).add_(x1_norm)
                return res.clamp_min_(1e-30)

            D = my_cdist(x, y)
            K = torch.zeros_like(D)

            for g in gamma:
                K.add_(torch.exp(D.mul(-g)))
            return K

        Kxx = gaussian_kernel(x, x).mean()
        Kyy = gaussian_kernel(y, y).mean()
        Kxy = gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - features (Tensor): featurizer output for batch and unlabeled batch
                - y_pred (Tensor): full model output for batch and unlabeled batch
        """
        # forward pass
        x, y_true, metadata = batch
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        results = {
            "g": g,
            "y_true": y_true,
            "metadata": metadata,
        }
        x = x.to(self.device)
        features = self.featurizer(x)
        # print(features.shape)
        outputs = self.classifier(features)
        y_pred = outputs[: len(y_true)]
        results["features"] = features
        results["y_pred"] = y_pred
        return results

    def step(self, results):
        features = results.pop("features")
        unique_groups, group_indices, _ = split_into_groups(results["g"])
        n_groups_per_batch = unique_groups.numel()
        penalty = torch.zeros(1, device=self.device)

        for i_group in range(
            n_groups_per_batch
        ):  # Each element of group_indices is a list of indices
            for j_group in range(i_group + 1, n_groups_per_batch):
                penalty += self.penalty(
                    features[group_indices[i_group]], features[group_indices[j_group]]
                )
            if n_groups_per_batch > 1:
                penalty /= (
                    n_groups_per_batch * (n_groups_per_batch - 1) / 2
                )  # get the mean penalty
        else:
            penalty = 0.0
        avg_loss = self.ds_bundle.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        ).mean()
        # print({"loss/{}".format(self.client_id): avg_loss.item()})
        objective = avg_loss + penalty * self.penalty_weight
        if objective.grad_fn is None:
            pass
        else:
            objective.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return (results["y_pred"].shape)[0] * objective.item()


class Coral(MMD):
    def penalty(self, x, y):
        if x.dim() > 2:
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff


class GroupDRO(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.group_weights_step_size = hparam["hparam1"]
        self.group_weights = torch.zeros(self.ds_bundle.grouper.n_groups)
        train_g = self.ds_bundle.grouper.metadata_to_group(self.dataset.metadata_array)
        unique_groups, unique_counts = torch.unique(
            train_g, sorted=False, return_counts=True
        )
        counts = torch.zeros(self.ds_bundle.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights / self.group_weights.sum()

    def step(self, results):
        loss = torch.zeros_like(self.group_weights)
        unique_groups, group_indices, _ = split_into_groups(results["g"])
        for group_idx, i_group in zip(unique_groups, group_indices):
            group_losses = self.ds_bundle.loss.compute(
                results["y_pred"][i_group],
                results["y_true"][i_group],
                return_dict=False,
            ).mean()
            loss[group_idx] = group_losses
        self.group_weights = self.group_weights * torch.exp(
            self.group_weights_step_size * loss.data
        )
        self.group_weights = self.group_weights / (self.group_weights.sum())
        objective = self.group_weights @ loss
        if objective.grad_fn is None:
            # print('jump')
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return (results["y_pred"].shape)[0] * objective.item()

    def init_train(self):
        super().init_train()
        self.group_weights = self.group_weights.to(self.device)

    def end_train(self):
        super().end_train()
        self.group_weights = self.group_weights.to("cpu")


class Mixup(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.dataloader = get_train_loader(
            self.loader_type,
            self.dataset,
            batch_size=self.batch_size,
            uniform_over_groups=None,
            grouper=self.ds_bundle.grouper,
            distinct_groups=True,
            n_groups_per_batch=self.n_groups_per_batch,
            num_workers=self.hparam.get("num_workers", 16),
            pin_memory=self.hparam.get("num_workers", 16) > 0,
            persistent_workers=self.hparam.get("num_workers", 16) > 0,
        )
        self.alpha = hparam["hparam1"]

    @property
    def loader_type(self):
        return "group"

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        _, group_indices, _ = split_into_groups(g)
        lam = np.random.beta(self.alpha, self.alpha)

        outputs = self.model(
            lam * x[group_indices[0]] + (1 - lam) * x[group_indices[1]]
        )
        results = {
            "g": g,
            "y_true": y_true,
            "y_pred": outputs,
            "metadata": metadata,
            "lam": lam,
        }
        return results

    def step(self, results):
        _, group_indices, _ = split_into_groups(results["g"])
        objective = (
            results["lam"]
            * self.ds_bundle.loss.compute(
                results["y_pred"],
                results["y_true"][group_indices[0]],
                return_dict=False,
            ).mean()
            + (1 - results["lam"])
            * self.ds_bundle.loss.compute(
                results["y_pred"],
                results["y_true"][group_indices[1]],
                return_dict=False,
            ).mean()
        )
        # print({"loss/{}".format(self.client_id): objective.item()})
        if objective.grad_fn is None:
            # print('jump')
            pass
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return (results["y_pred"].shape)[0] * objective.item()


class FourierMixup(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.ratio_lower = self.hparam["hparam1"]
        self.ratio_upper = self.hparam["hparam2"]
        self.rng = np.random.default_rng()

    @property
    def ratio(self):
        return self.rng.uniform(self.ratio_lower, self.ratio_upper)

    def set_amploader(self, dataloader):
        self.amploader = dataloader
        self.iter_amploader = iter(dataloader)  # list of indices of dataset

    @property
    def loader_type(self):
        return "standard"

    def process_batch(self, batch):
        x, y_true, [metadata, amp, pha] = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        amp = amp.to(self.device)
        pha = pha.to(self.device)
        lmda = random.uniform(0, 1)
        try:
            _, _, [_, sampled_amp, _] = next(self.iter_amploader)
            if sampled_amp.shape[0] != amp.shape[0]:
                self.iter_amploader = iter(self.amploader)
                _, _, [_, sampled_amp, _] = next(self.iter_amploader)
        except StopIteration:
            self.iter_amploader = iter(self.amploader)
            _, _, [_, sampled_amp, _] = next(self.iter_amploader)
        sampled_amp = sampled_amp.to(self.device)
        new_amp = self._amp_spectrum_swap(
            amp, sampled_amp[0 : amp.shape[0]], L=lmda, ratio=self.ratio
        )
        fft_local_ = new_amp * torch.exp(1j * pha)
        new_x = torch.real(torch.fft.ifft2(fft_local_))

        _, group_indices, _ = split_into_groups(g)

        outputs = self.model(new_x)
        results = {"g": g, "y_true": y_true, "y_pred": outputs, "metadata": metadata}
        return results

    @staticmethod
    def _amp_spectrum_swap(amp_local, amp_target, L=0.1, ratio=0):
        a_local = torch.fft.fftshift(amp_local, dim=(-2, -1))
        a_trg = torch.fft.fftshift(amp_target, dim=(-2, -1))

        _, _, h, w = a_local.shape
        b = int(min(h, w) * L)
        c_h = int(h / 2)
        c_w = int(w / 2)

        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        try:
            a_local[:, :, h1:h2, w1:w2] = a_local[:, :, h1:h2, w1:w2] * ratio + a_trg[
                :, :, h1:h2, w1:w2
            ] * (1 - ratio)
        except RuntimeError:
            print(a_local.shape, a_trg.shape)
            exit()
        a_local = torch.fft.ifftshift(a_local, dim=(-2, -1))
        return a_local


class FedADGClient(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self._generator = None
        self._discriminator = None
        self.alpha = self.hparam["hparam1"]
        self.second_local_epochs = int(self.hparam["hparam2"])

    def setup_model(self, featurizer, classifier, generator):
        super().setup_model(featurizer, classifier)
        self._generator = generator
        self.generator = nn.DataParallel(self._generator)
        self._discriminator = Discriminator(
            self._featurizer.n_outputs, self.ds_bundle.n_classes
        )
        self.discriminator = nn.DataParallel(self._discriminator)

    def update_model(self, model_dict, generator_dict):
        super().update_model(model_dict)
        self._generator.load_state_dict(generator_dict)

    def init_train(self):
        super().init_train()
        self.generator.train()
        self.generator.to(self.device)
        self.discriminator.train()
        self.discriminator.to(self.device)
        self.gen_optimizer_lr = self.hparam["hparam3"]
        self.disc_optim_lr = self.hparam["hparam4"]
        self.criterion = ElementwiseLoss(
            loss_fn=nn.CrossEntropyLoss(
                reduction="none", ignore_index=-100, label_smoothing=0.2
            )
        )
        self.disc_optimizer = torch.optim.SGD(
            self.discriminator.parameters(),
            self.disc_optim_lr,
            momentum=0.9,
            weight_decay=1e-5,
        )
        self.gen_optimizer = torch.optim.SGD(
            self.generator.parameters(),
            self.gen_optimizer_lr,
            momentum=0.9,
            weight_decay=1e-5,
        )

    def end_train(self):
        self.generator.to("cpu")
        self.discriminator.to("cpu")
        super().end_train()

    def fit(self, server_round):
        """Update local model using local dataset."""
        self.init_train()
        for e in range(self.local_epochs):
            training_loss = 0.0
            for batch in self.dataloader:
                results = self.process_batch(batch)
                training_loss += self.step(results)

            if self.hparam["wandb"]:
                wandb.log(
                    {
                        "aln_loss/{}".format(self.client_id): training_loss
                        / len(self.dataset)
                    },
                    step=server_round * self.local_epochs + e,
                )

        for e in range(self.second_local_epochs):
            training_loss = np.zeros(3)
            for t, batch in enumerate(self.dataloader):
                training_loss += self.second_step(batch)
            if self.hparam["wandb"]:
                wandb.log(
                    {
                        "cla_loss/{}".format(self.client_id): training_loss[0]
                        / len(self.dataset)
                    },
                    step=server_round * self.local_epochs + e,
                )
                wandb.log(
                    {
                        "dist_loss/{}".format(self.client_id): training_loss[1]
                        / len(self.dataset)
                    },
                    step=server_round * self.local_epochs + e,
                )
                wandb.log(
                    {
                        "gen_loss/{}".format(self.client_id): training_loss[2]
                        / len(self.dataset)
                    },
                    step=server_round * self.local_epochs + e,
                )
        self.end_train()

    def second_step(self, batch):
        self.discriminator.eval()
        self.generator.eval()

        x, y_true = batch[0], batch[1]
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        y_onehot = torch.zeros(y_true.size(0), self.dataset.n_classes).to(self.device)
        y_onehot.scatter_(1, y_true.view(-1, 1), 0.6).to(self.device)
        randomn = torch.rand(y_true.size(0), self._generator.input_size).to(self.device)

        # training feature extractor and classifier
        self.optimizer.zero_grad()
        feature = self.featurizer(x)
        y_pred = self.classifier(feature)
        loss = self.criterion.compute(y_pred, y_true, return_dict=False).mean()
        loss_enc = torch.mean(torch.pow(1 - self.discriminator(y_onehot, feature), 2))
        loss_cla = self.alpha * loss + (1 - self.alpha) * loss_enc
        # wandb.log({"generator_loss/{}".format(self.client_id): loss.item(), "discriminator_loss/{}".format(self.client_id): loss_enc.item()})
        loss_cla.backward()
        self.optimizer.step()

        # training discriminator
        self.featurizer.eval()
        self.discriminator.train()
        self.disc_optimizer.zero_grad()
        feature = self.featurizer(x).detach()
        gen_feature = self.generator(y=y_onehot, x=randomn).detach()
        loss_discriminator = -torch.mean(
            torch.pow(self.discriminator(y_onehot, gen_feature), 2)
            + torch.pow(1 - self.discriminator(y_onehot, feature), 2)
        )
        loss_discriminator.backward()
        self.disc_optimizer.step()
        self.discriminator.eval()

        # training distribution generator
        self.generator.train()
        self.gen_optimizer.zero_grad()
        gen_feature = self.generator(y=y_onehot, x=randomn).detach()
        loss_gene = torch.mean(
            torch.pow(1 - self.discriminator(y_onehot, gen_feature), 2)
        )
        loss_gene.backward()
        self.gen_optimizer.step()
        self.generator.eval()

        return (
            np.array([loss_cla.item(), loss_discriminator.item(), loss_gene.item()])
            * y_true.shape[0]
        )

    @property
    def loader_type(self):
        return "standard"


class FedSR(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.l2_regularizer = hparam["hparam1"]
        self.cmi_regularizer = hparam["hparam2"]
        self.fp = "{}tmp/fedsr_ref_client_{}.pt".format(
            self.hparam["data_path"], self.client_id
        )

    def setup_model(self, featurizer, classifier):
        super().setup_model(featurizer, classifier)
        self.reference_params = nn.Parameter(
            torch.ones(
                self.ds_bundle.n_classes,
                2 * self._featurizer.n_outputs,
                device=self.device,
            )
        )
        torch.save(self.reference_params, self.fp)
        del self.reference_params

    def init_train(self):
        self.reference_params = torch.load(self.fp)
        self.model.train()
        self.model.to(self.device)
        self.optimizer = eval(self.optimizer_name)(
            list(self.model.parameters()) + [self.reference_params], **self.optim_config
        )
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))

    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.reference_params, self.fp)
        del self.reference_params, self.optimizer
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @property
    def loader_type(self):
        return "standard"

    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - features (Tensor): featurizer output for batch and unlabeled batch
                - y_pred (Tensor): full model output for batch and unlabeled batch
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        results = {
            "g": g,
            "y_true": y_true,
            "metadata": metadata,
        }
        features_params = self.featurizer(x)
        z_dim = int(features_params.shape[-1] / 2)
        if len(features_params.shape) == 2:
            z_mu = features_params[:, :z_dim]
            z_sigma = F.softplus(features_params[:, z_dim:])
            z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)
            features = z_dist.rsample()
        elif len(features_params.shape) == 3:
            flattened_features_params = features_params.view(
                -1, features_params.shape[-1]
            )
            z_mu = flattened_features_params[:, :z_dim]
            z_sigma = F.softplus(flattened_features_params[:, z_dim:])
            z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)
            features = z_dist.rsample()
            features = features.view(x.shape[0], -1, z_dim)
        y_pred = self.classifier(features)
        results["features"] = features
        results["z_mu"] = z_mu
        results["z_sigma"] = z_sigma
        results["feature_params"] = features_params
        results["y_pred"] = y_pred
        return results

    def l2_penalty(self, features):
        if self.ds_bundle.name == "py150":
            num_samples = features.shape[0] * features.shape[1]
        else:
            num_samples = features.shape[0]
        return torch.sum(features**2) / num_samples

    def cmi_penalty(self, y, z_mu, z_sigma):
        num_samples = y.shape[0]
        dimension = self.reference_params.shape[1] // 2
        if self.ds_bundle.name == "py150":
            is_labeled = ~torch.isnan(y)
            flattened_y = y[is_labeled]
            z_mu = z_mu[is_labeled.view(-1)]
            z_sigma = z_sigma[is_labeled.view(-1)]
            target_mu = self.reference_params[
                flattened_y.to(dtype=torch.long), :dimension
            ]
            target_sigma = F.softplus(
                self.reference_params[flattened_y.to(dtype=torch.long), dimension:]
            )
        else:
            target_mu = self.reference_params[y.to(dtype=torch.long), :dimension]
            target_sigma = F.softplus(
                self.reference_params[y.to(dtype=torch.long), dimension:]
            )
        cmi_loss = (
            torch.sum(
                (
                    torch.log(target_sigma)
                    - torch.log(z_sigma)
                    + (z_sigma**2 + (target_mu - z_mu) ** 2) / (2 * target_sigma**2)
                    - 0.5
                )
            )
            / num_samples
        )
        return cmi_loss

    def step(self, results):
        loss = self.ds_bundle.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        ).mean()
        l2_loss = self.l2_penalty(results["features"])
        cmi_loss = self.cmi_penalty(
            results["y_true"], results["z_mu"], results["z_sigma"]
        )

        self.optimizer.zero_grad()
        objective = (
            loss + self.l2_regularizer * l2_loss + self.cmi_regularizer * cmi_loss
        )
        objective.backward()

        self.optimizer.step()
        return (results["y_pred"].shape)[0] * objective.item()


class ScaffoldClient(ERM):
    def setup_model(self, featurizer, classifier):
        super().setup_model(featurizer, classifier)

        self.c_local = None
        self.c_global = None

    def fit(self, server_round):
        """Update local model using local dataset."""
        self.init_train()
        training_loss = 0.0
        global_model = ParamDict(self.model.state_dict())
        lr = self.optimizer.param_groups[0]["lr"]
        for e in range(self.local_epochs):
            for batch in self.dataloader:
                results = self.process_batch(batch)
                training_loss += self.step(results)
            if self.hparam["wandb"]:
                wandb.log(
                    {
                        "loss/{}".format(self.client_id): training_loss
                        / len(self.dataset)
                    },
                    step=server_round * self.local_epochs + e,
                )
        local_model = ParamDict(self.model.state_dict())
        if self.c_local is None:
            self.c_local = (global_model - local_model) / (self.local_epochs * lr)
        else:
            self.c_local = (
                self.c_local
                - self.c_global
                + (global_model - local_model) / (self.local_epochs * lr)
            )
        self.end_train()

    def init_train(self):
        super().init_train()

        if self.c_local is not None:
            self.c_local = self.c_local.to(self.device)

        if self.c_global is not None:
            self.c_global = self.c_global.to(self.device)

    def end_train(self):
        super().end_train()

        self.c_local = self.c_local.to("cpu")
        if self.c_global is not None:

            del self.c_global

    def step(self, results):

        objective = self.ds_bundle.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        ).mean()
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            pass
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            param_dict = ParamDict(self.model.state_dict())
            if self.c_local is not None:
                param_dict = param_dict - self.optimizer.param_groups[0]["lr"] * (
                    self.c_global - self.c_local
                )
            self.model.load_state_dict(copy.deepcopy(param_dict))
        return (results["y_pred"].shape)[0] * objective.item()


class FedProx(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.mu = self.hparam["hparam1"]

    def prox(self):
        proximal_term = 0.0
        for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
        return proximal_term

    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.optimizer = eval(self.optimizer_name)(
            self.model.parameters(), **self.optim_config
        )
        self.scheduler = eval(self.scheduler_name)(
            self.optimizer, **self.scheduler_config
        )
        if self.saved_optimizer:
            self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
            self.scheduler.load_state_dict(torch.load(self.sch_dict_path))

    def end_train(self):

        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.scheduler.state_dict(), self.sch_dict_path)
        del self.scheduler, self.optimizer, self.global_model
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def step(self, results):

        objective = (
            self.ds_bundle.loss.compute(
                results["y_pred"], results["y_true"], return_dict=False
            ).mean()
            + self.mu / 2 * self.prox()
        )
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            pass
        self.optimizer.step()
        self.optimizer.zero_grad()
        return (results["y_pred"].shape)[0] * objective.item()


class AFLClient(ERM):
    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)

    def update_vector(self, global_vector):
        self.group_weights = copy.deepcopy(global_vector)
        self.group_weights = self.group_weights.to(self.device).requires_grad_()

    def step(self, results):
        objective = 0.0
        unique_groups, group_indices, _ = split_into_groups(results["g"])
        for group_idx, i_group in zip(unique_groups, group_indices):
            group_losses = self.ds_bundle.loss.compute(
                results["y_pred"][i_group],
                results["y_true"][i_group],
                return_dict=False,
            )
            objective += group_losses * self.group_weights[group_idx]
        objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return objective.item()

    def gradient_lambda(self):
        self.init_train()
        loss_per_domain = torch.zeros_like(self.group_weights)
        samples_per_domain = torch.zeros_like(self.group_weights)
        self.model.eval()
        for batch in tqdm(self.dataloader):
            results = self.process_batch(batch)
            unique_groups, group_indices, _ = split_into_groups(results["g"])
            for group_idx, i_group in zip(unique_groups, group_indices):
                group_losses = self.ds_bundle.loss.compute(
                    results["y_pred"][i_group],
                    results["y_true"][i_group],
                    return_dict=False,
                ).sum()
                loss_per_domain[group_idx] += group_losses.item()
            samples_per_domain += torch.bincount(
                results["g"], minlength=len(samples_per_domain)
            )
        self.model.train()
        self.end_train()

        return loss_per_domain.to("cpu"), samples_per_domain.to("cpu")


class FFDClient(ERM):
    """Federated Feature Disentanglement client.

    Losses:
      - L_task: cross-entropy
      - L_align: α(t) * mean ||z_inv_i - P[y_i]||²  (prototype alignment)
      - L_var:  mean(max(0, 1 - sqrt(Var(z_inv) + ε)))  (variance regularisation)
      - L_cov:  off_diag(cov(z_inv))² + off_diag(cross_cov(z_inv, z_env))²
    """

    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        self.ffd_alpha = hparam.get("ffd_alpha", 1.0)
        self.ffd_lambda_var = hparam.get("ffd_lambda_var", 1.0)
        self.ffd_lambda_cov = hparam.get("ffd_lambda_cov", 0.04)
        self.ffd_eps = hparam.get("eps", 1e-8)

        # Current effective alpha (set by server each round via warm-up)
        self._current_alpha = 0.0

        self.local_prototypes = {}
        self.global_prototypes = {}

    def setup_model(self, featurizer, classifier):
        """FFD uses FFDModelWrapper instead of plain Sequential."""
        from .models import FFDModelWrapper

        self._featurizer = featurizer
        self._classifier = classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(
            FFDModelWrapper(self._featurizer, self._classifier)
        )

    def set_prototypes(self, global_prototypes, current_alpha=None):
        """Receive global prototypes and warm-up alpha from the server."""
        self.global_prototypes = copy.deepcopy(global_prototypes)
        if current_alpha is not None:
            self._current_alpha = current_alpha

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)

        logits, z_inv, z_env = self.model(x)

        results = {
            "g": g,
            "y_true": y_true,
            "y_pred": logits,
            "z_inv": z_inv,
            "z_env": z_env,
            "metadata": metadata,
        }
        return results

    # ── VICReg-style loss components ──────────────────────────────

    @staticmethod
    def _variance_loss(z, eps=1e-8):
        """Hinge loss on per-dimension std: max(0, 1 - std_j)."""
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    @staticmethod
    def _covariance_loss(z):
        """Sum of squared off-diagonal elements of the covariance matrix."""
        n = z.size(0)
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(n - 1, 1)
        # zero out diagonal
        off_diag = cov - torch.diag(cov.diag())
        return (off_diag**2).sum() / z.size(1)

    @staticmethod
    def _cross_covariance_loss(z1, z2):
        """Sum of squared elements of the cross-covariance matrix."""
        n = z1.size(0)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        cross_cov = (z1_centered.T @ z2_centered) / max(n - 1, 1)
        return (cross_cov**2).sum() / z1.size(1)

    # ── Training step ─────────────────────────────────────────────

    def step(self, results):
        """Compute FFD combined loss and return scalar total for ERM.fit."""
        # Task loss (cross-entropy)
        loss_task = self.ds_bundle.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        ).mean()

        z_inv = results["z_inv"]
        z_env = results["z_env"]
        y_true = results["y_true"]

        # Alignment loss: pull z_inv towards global class prototypes
        loss_align = torch.tensor(0.0, device=self.device)
        if self._current_alpha > 0 and len(self.global_prototypes) > 0:
            unique_classes = torch.unique(y_true)
            count = 0
            for c in unique_classes:
                c_item = c.item()
                if c_item in self.global_prototypes:
                    mask = y_true == c
                    z_c = z_inv[mask]
                    proto_c = self.global_prototypes[c_item].to(self.device)
                    loss_align = loss_align + F.mse_loss(z_c, proto_c.expand_as(z_c))
                    count += 1
            if count > 0:
                loss_align = loss_align / count

        # Variance loss: prevent feature collapse
        loss_var = self._variance_loss(z_inv, eps=self.ffd_eps)

        # Covariance loss: decorrelate z_inv dims + orthogonalise z_inv vs z_env
        loss_cov = self._covariance_loss(z_inv) + self._cross_covariance_loss(
            z_inv, z_env
        )

        # Weighted total
        objective = (
            loss_task
            # + self._current_alpha * loss_align
            + self.ffd_lambda_var * loss_var
            + self.ffd_lambda_cov * loss_cov
        )

        batch_size = z_inv.size(0)
        total_loss = objective.item() * batch_size

        # Store per-component losses (batch-scaled) for logging
        self._last_losses = {
            "task": loss_task.item() * batch_size,
            "align": loss_align.item() * batch_size,
            "var": loss_var.item() * batch_size,
            "cov": loss_cov.item() * batch_size,
        }

        if objective.grad_fn is not None:
            objective.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Return scalar so ERM.fit can do `training_loss += self.step(results)`
        return total_loss

    # ── Prototype computation ─────────────────────────────────────

    def compute_local_prototypes(self):
        """After training, compute per-class mean z_inv on local data."""
        self.model.to(self.device)
        self.model.eval()
        class_sums = {}
        class_counts = {}

        with torch.no_grad():
            for batch in self.dataloader:
                x, y_true, _ = batch
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                # Use model in eval mode (returns logits only via wrapper)
                # Access featurizer directly for z_inv
                features = self._featurizer.backbone(x)
                z_inv = self._featurizer.h_inv(features)

                for i in range(len(y_true)):
                    c = y_true[i].item()
                    if c not in class_sums:
                        class_sums[c] = z_inv[i].detach().cpu()
                        class_counts[c] = 1
                    else:
                        class_sums[c] += z_inv[i].detach().cpu()
                        class_counts[c] += 1

        self.local_prototypes = {c: class_sums[c] / class_counts[c] for c in class_sums}
        self.model.to("cpu")

    def fit(self, server_round):
        """Train locally then compute prototypes.

        Returns dict of average losses for server-level aggregation.
        """
        self.init_train()
        training_loss = 0.0
        loss_accum = {"task": 0.0, "align": 0.0, "var": 0.0, "cov": 0.0}

        for e in range(self.local_epochs):
            for batch in tqdm(self.dataloader):
                results = self.process_batch(batch)
                training_loss += self.step(results)
                for k in loss_accum:
                    loss_accum[k] += self._last_losses[k]

            if self.hparam["wandb"]:
                n = len(self.dataset)
                wandb.log(
                    {
                        f"loss/{self.client_id}": training_loss / n,
                        f"loss_task/{self.client_id}": loss_accum["task"] / n,
                        f"loss_align/{self.client_id}": loss_accum["align"] / n,
                        f"loss_var/{self.client_id}": loss_accum["var"] / n,
                        f"loss_cov/{self.client_id}": loss_accum["cov"] / n,
                    },
                    step=server_round * self.local_epochs + e,
                )

        self.end_train()
        self.compute_local_prototypes()

        n = len(self.dataset)
        return {
            "total": training_loss / n,
            "task": loss_accum["task"] / n,
            "align": loss_accum["align"] / n,
            "var": loss_accum["var"] / n,
            "cov": loss_accum["cov"] / n,
        }


class FCDClient(ERM):
    """Federated Cyclic Disentanglement client.

    Implements the four-phase local training pipeline:
      Phase A — Geometric Feature Decomposition (L_align + L_cov)
      Phase B — Statistical Feature Grounding   (L_stat)
      Phase C — Cyclic Invariance & Counterfactual Simulation (L_task)

    Combined objective:
      L_total = λ_task · L_task + λ_stat · L_stat + λ_reg · (L_align + L_cov)
    """

    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        # Loss weights
        self.lambda_task = hparam.get("fcd_lambda_task", 1.0)
        self.lambda_stat = hparam.get("fcd_lambda_stat", 1.0)
        self.lambda_reg = hparam.get("fcd_lambda_reg", 1.0)
        self.fcd_alpha = hparam.get("fcd_alpha", 1.0)
        self.eps = hparam.get("eps", 1e-8)
        self.fcd_cf_start_round = hparam.get("fcd_cf_start_round", 15)

        # Augmentation source for counterfactual pathway (ablation experiment)
        self.fcd_aug_source = hparam.get("fcd_aug_source", "gmm")

        # Warm-up alpha (set by server each round)
        self._current_alpha = 0.0

        # Prototypes and GMM (received from server)
        self.local_prototypes = {}
        self.global_prototypes = {}
        self.gmm_params = None  # dict with 'weights', 'means', 'covariances'
        self.global_unimodal_params = None  # dict with 'mean', 'covariance'

        # Local environment statistics (computed after training, sent to server)
        self.local_env_stats = None

    def setup_model(self, featurizer, classifier, style_encoder):
        """FCD uses FCDModelWrapper instead of plain Sequential."""
        from .models import FCDModelWrapper

        self._featurizer = featurizer
        self._classifier = classifier
        self._style_encoder = style_encoder
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(
            FCDModelWrapper(self._featurizer, self._classifier, self._style_encoder)
        )

    def set_prototypes(self, global_prototypes, current_alpha=None):
        """Receive global prototypes and warm-up alpha from the server."""
        self.global_prototypes = copy.deepcopy(global_prototypes)
        if current_alpha is not None:
            self._current_alpha = current_alpha

    def set_gmm_params(self, gmm_params):
        """Receive global GMM parameters from the server."""
        self.gmm_params = copy.deepcopy(gmm_params)

    def set_global_unimodal_params(self, global_unimodal_params):
        """Receive the naive unimodal global Gaussian params (ablation condition 3)."""
        self.global_unimodal_params = copy.deepcopy(global_unimodal_params)

    def update_model(self, model_dict):
        """Override: load model state dict."""
        self.model.load_state_dict(model_dict)

    # ── Forward ───────────────────────────────────────────────────

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)

        # Training mode: returns (logits, z_inv, z_env, H, gamma_hat, beta_hat)
        logits, z_inv, z_env, H, gamma_hat, beta_hat = self.model(x)

        results = {
            "g": g,
            "y_true": y_true,
            "y_pred": logits,
            "z_inv": z_inv,
            "z_env": z_env,
            "H": H,
            "gamma_hat": gamma_hat,
            "beta_hat": beta_hat,
            "metadata": metadata,
        }
        return results

    # ── Loss components ───────────────────────────────────────────

    def _alignment_loss(self, z_inv, y_true):
        """Phase A: Pull z_inv towards global semantic prototypes.

        L_align = α(t) * mean_c ||z_inv_c - p_c||²
        """
        if self._current_alpha <= 0 or len(self.global_prototypes) == 0:
            return torch.tensor(0.0, device=self.device)

        unique_classes = torch.unique(y_true)
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        for c in unique_classes:
            c_item = c.item()
            if c_item in self.global_prototypes:
                mask = y_true == c
                z_c = z_inv[mask]
                proto_c = self.global_prototypes[c_item].to(self.device)
                loss = loss + F.mse_loss(z_c, proto_c.expand_as(z_c))
                count += 1
        if count > 0:
            loss = self._current_alpha * loss / count
        return loss

    @staticmethod
    def _covariance_loss(z_inv, z_env):
        """Phase A: Internal decorrelation + subspace orthogonality.

        L_cov = Σ_{i≠j} C(z_inv)²_{ij}  +  Σ_{i,j} C(z_inv, z_env)²_{ij}
        """
        n = z_inv.size(0)
        d_inv = z_inv.size(1)
        denom = max(n - 1, 1)

        # Internal decorrelation of z_inv
        z_inv_c = z_inv - z_inv.mean(dim=0)
        cov_inv = (z_inv_c.T @ z_inv_c) / denom
        off_diag_inv = cov_inv - torch.diag(cov_inv.diag())
        loss_internal = (off_diag_inv**2).sum() / d_inv

        # Cross-covariance (subspace orthogonality)
        z_env_c = z_env - z_env.mean(dim=0)
        cross_cov = (z_inv_c.T @ z_env_c) / denom
        loss_cross = (cross_cov**2).sum() / d_inv

        return loss_internal, loss_cross

    @staticmethod
    def _statistical_grounding_loss(H, gamma_hat, beta_hat, eps=1e-8):
        """Phase B: Ground z_env to the empirical spatial statistics.

        L_stat = ||σ(H) - γ̂||² + ||μ(H) - β̂||²

        where μ(H) and σ(H) are the channel-wise mean and std of the
        spatial feature map H ∈ (B, C_feat, h, w).
        """
        # Channel-wise statistics of the spatial tensor
        mu_H = H.mean(dim=[2, 3])  # (B, C_feat)
        sigma_H = (H.var(dim=[2, 3]) + eps).sqrt()  # (B, C_feat)

        loss = F.mse_loss(sigma_H, gamma_hat) + F.mse_loss(mu_H, beta_hat)
        return loss

    def _sample_from_gmm(self, n_samples):
        """Sample latent environment vectors from the global GMM.

        Returns tensor of shape (n_samples, proj_dim) on self.device.
        """
        if self.gmm_params is None:
            return None

        weights = self.gmm_params["weights"]  # (M,)
        means = self.gmm_params["means"]  # (M, d)
        covs = self.gmm_params["covariances"]  # (M, d, d)

        M = len(weights)
        d = means.shape[1]

        # Choose components according to mixing weights
        component_indices = torch.multinomial(
            torch.tensor(weights, dtype=torch.float32),
            n_samples,
            replacement=True,
        )

        samples = torch.zeros(n_samples, d)
        for m in range(M):
            mask = component_indices == m
            count = mask.sum().item()
            if count > 0:
                mean_m = torch.tensor(means[m], dtype=torch.float32)
                cov_m = torch.tensor(covs[m], dtype=torch.float32)
                # Ensure covariance is positive definite
                cov_m = cov_m + 1e-6 * torch.eye(d)
                mvn = torch.distributions.MultivariateNormal(mean_m, cov_m)
                samples[mask] = mvn.sample((count,))

        return samples.to(self.device)

    def _sample_augmentation(self, n_samples):
        """Sample z_env^sim from the configured augmentation source.

        Supports four conditions for the augmentation source ablation:
          1. 'isotropic'       — N(0, I)
          2. 'local'           — N(μ_i, Σ_i)  (client's own empirical env stats)
          3. 'global_unimodal' — N(μ_global, Σ_global)  (naive weighted average)
          4. 'gmm'             — Multi-modal Global GMM (proposed method)
        """
        d = self.hparam.get("fcd_proj_dim", 256)

        if self.fcd_aug_source == "isotropic":
            # Condition 1: standard isotropic Gaussian noise
            return torch.randn(n_samples, d, device=self.device)

        elif self.fcd_aug_source == "local":
            # Condition 2: client's own local empirical Gaussian
            if self.local_env_stats is None:
                return torch.randn(n_samples, d, device=self.device)
            mu = torch.tensor(self.local_env_stats["mean"], dtype=torch.float32)
            cov = torch.tensor(self.local_env_stats["covariance"], dtype=torch.float32)
            cov = cov + 1e-6 * torch.eye(d)
            mvn = torch.distributions.MultivariateNormal(mu, cov)
            return mvn.sample((n_samples,)).to(self.device)

        elif self.fcd_aug_source == "global_unimodal":
            # Condition 3: naive unimodal global Gaussian
            if self.global_unimodal_params is None:
                return torch.randn(n_samples, d, device=self.device)
            mu = torch.tensor(self.global_unimodal_params["mean"], dtype=torch.float32)
            cov = torch.tensor(self.global_unimodal_params["covariance"], dtype=torch.float32)
            cov = cov + 1e-6 * torch.eye(d)
            mvn = torch.distributions.MultivariateNormal(mu, cov)
            return mvn.sample((n_samples,)).to(self.device)

        else:  # "gmm" — the proposed multi-modal GMM (condition 4, default)
            return self._sample_from_gmm(n_samples)

    def _counterfactual_task_loss(self, H, z_inv, logits_orig, y_true, eps=1e-8):
        B = H.size(0)
        ce_loss = nn.CrossEntropyLoss()
        loss_orig = ce_loss(logits_orig, y_true)

        # Counterfactual path — gated by source availability and round
        if self.fcd_aug_source == "gmm" and self.gmm_params is None:
            return loss_orig, 0.0
        if getattr(self, "current_server_round", 0) < self.fcd_cf_start_round:
            return loss_orig, 0.0

        z_env_sim = self._sample_augmentation(B)
        gamma_sim, beta_sim = self._style_encoder(z_env_sim)
        gamma_sim = gamma_sim.detach()
        beta_sim = beta_sim.detach()

        # Pool FIRST, then normalise — preserves content
        pooled = self._featurizer.gap(H).flatten(1)  # (B, C_feat)
        mu_p = pooled.mean(dim=1, keepdim=True)  # (B, 1)
        sigma_p = (pooled.var(dim=1, keepdim=True) + eps).sqrt()  # (B, 1)
        pooled_norm = (pooled - mu_p) / sigma_p  # (B, C_feat)

        # Apply foreign style in pooled space
        pooled_cf = gamma_sim * pooled_norm + beta_sim  # (B, C_feat)

        # Project through h_inv MLP (skip GAP — already pooled)
        z_inv_cf = self._featurizer.h_inv(pooled_cf)
        logits_cf = self._classifier(z_inv_cf)
        loss_cf = ce_loss(logits_cf, y_true)

        return loss_orig, loss_cf

    # def _counterfactual_task_loss(self, H, z_inv, logits_orig, y_true, eps=1e-8):
    #     """Phase C: Cyclic invariance via counterfactual simulation.

    #     1. Sample z_env^sim from the global GMM.
    #     2. Decode to (γ̂^sim, β̂^sim) via the style encoder.
    #     3. Stop-gradient on the simulated affine parameters.
    #     4. Instance-normalise H and apply the simulated style.
    #     5. Re-project H_cf through h_inv to get ẑ_inv^cf.
    #     6. Return CE(C(z_inv), y) + CE(C(ẑ_inv^cf), y).

    #     Args:
    #         H: Spatial feature map from the backbone.
    #         z_inv: Invariant features (used only if GMM unavailable).
    #         logits_orig: Logits already computed in process_batch —
    #             reused here to avoid a redundant classifier forward pass.
    #         y_true: Ground-truth labels.
    #         eps: Numerical stability constant for normalisation.
    #     """
    #     B = H.size(0)
    #     ce_loss = nn.CrossEntropyLoss()

    #     # Original task loss (reuse logits from process_batch)
    #     loss_orig = ce_loss(logits_orig, y_true)

    #     # Counterfactual path (disabled when GMM is unavailable or before start round)
    #     if (
    #         self.gmm_params is None
    #         or getattr(self, "current_server_round", 0) < self.fcd_cf_start_round
    #     ):
    #         # Return only the original CE; the effective weight of L_task
    #         # stays consistent across rounds rather than jumping when the
    #         # GMM first becomes available.
    #         return loss_orig, 0.0

    #     # Sample foreign style
    #     z_env_sim = self._sample_from_gmm(B)  # (B, proj_dim)
    #     gamma_sim, beta_sim = self._style_encoder(z_env_sim)  # (B, C_feat) each

    #     # Stop-gradient: prevent style encoder from accommodating classifier
    #     gamma_sim = gamma_sim.detach()
    #     beta_sim = beta_sim.detach()

    #     # Instance-normalise H and apply simulated affine parameters
    #     mu_H = H.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
    #     sigma_H = (H.var(dim=[2, 3], keepdim=True) + eps).sqrt()  # (B, C, 1, 1)
    #     H_norm = (H - mu_H) / sigma_H  # (B, C, h, w)

    #     # Reshape affine params for broadcasting: (B, C) → (B, C, 1, 1)
    #     gamma_sim = gamma_sim.unsqueeze(-1).unsqueeze(-1)
    #     beta_sim = beta_sim.unsqueeze(-1).unsqueeze(-1)

    #     H_cf = gamma_sim * H_norm + beta_sim  # (B, C, h, w)

    #     # Cyclic extraction: re-project through h_inv
    #     z_inv_cf = self._featurizer.forward_inv_from_spatial(H_cf)  # (B, proj_dim)
    #     logits_cf = self._classifier(z_inv_cf)
    #     loss_cf = ce_loss(logits_cf, y_true)

    #     return loss_orig, loss_cf

    # ── Training step ─────────────────────────────────────────────

    def step(self, results):
        """Compute the four-phase FCD loss and backpropagate."""
        z_inv = results["z_inv"]
        z_env = results["z_env"]
        H = results["H"]
        gamma_hat = results["gamma_hat"]
        beta_hat = results["beta_hat"]
        y_true = results["y_true"]
        logits_orig = results["y_pred"]

        # Phase A: Geometric feature decomposition
        loss_align = self._alignment_loss(z_inv, y_true)
        loss_cov_internal, loss_cov_cross = self._covariance_loss(z_inv, z_env)
        loss_cov = loss_cov_internal + loss_cov_cross

        # Phase B: Statistical feature grounding
        loss_stat = self._statistical_grounding_loss(
            H, gamma_hat, beta_hat, eps=self.eps
        )

        # Phase C: Cyclic invariance & counterfactual simulation
        loss_cls, loss_cf = self._counterfactual_task_loss(
            H, z_inv, logits_orig, y_true, eps=self.eps
        )
        loss_task = loss_cls + loss_cf

        # Combined objective (§4)
        objective = (
            self.lambda_task * loss_task
            + self.lambda_stat * loss_stat
            + self.lambda_reg * (loss_align + loss_cov)
        )

        batch_size = z_inv.size(0)
        total_loss = objective.item() * batch_size

        # Store per-component losses for logging
        if isinstance(loss_cf, float):
            self._last_losses = {
                "task": loss_task.item() * batch_size,
                "cls": loss_cls.item() * batch_size,
                "stat": loss_stat.item() * batch_size,
                "align": loss_align.item() * batch_size,
                "cov": loss_cov.item() * batch_size,
                "cov_internal": loss_cov_internal.item() * batch_size,
                "cov_cross": loss_cov_cross.item() * batch_size,
                "cf": 0.0 * batch_size,
            }
        else:
            self._last_losses = {
                "task": loss_task.item() * batch_size,
                "cls": loss_cls.item() * batch_size,
                "stat": loss_stat.item() * batch_size,
                "align": loss_align.item() * batch_size,
                "cov": loss_cov.item() * batch_size,
                "cov_internal": loss_cov_internal.item() * batch_size,
                "cov_cross": loss_cov_cross.item() * batch_size,
                "cf": loss_cf.item() * batch_size,
            }

        if objective.grad_fn is not None:
            objective.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss

    # ── Prototype and stats computation ───────────────────────────

    def compute_local_prototypes(self):
        """After training, compute per-class mean z_inv on local data."""
        self.model.to(self.device)
        self.model.eval()
        class_sums = {}
        class_counts = {}

        with torch.no_grad():
            for batch in self.dataloader:
                x, y_true, _ = batch
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                # In eval mode, featurizer returns z_inv directly
                z_inv = self._featurizer(x)

                for i in range(len(y_true)):
                    c = y_true[i].item()
                    if c not in class_sums:
                        class_sums[c] = z_inv[i].detach().cpu()
                        class_counts[c] = 1
                    else:
                        class_sums[c] += z_inv[i].detach().cpu()
                        class_counts[c] += 1

        self.local_prototypes = {c: class_sums[c] / class_counts[c] for c in class_sums}
        self.model.to("cpu")

    def compute_local_env_stats(self):
        """After training, compute mean and covariance of z_env on local data.

        These are sent to the server for GMM fitting (privacy-preserving:
        only aggregated statistics are transmitted, never raw data).

        Note: the featurizer's forward() does not return z_env in eval mode
        (it is discarded at inference per §5), so we call the env head
        directly via the dedicated helper.
        """
        self.model.to(self.device)
        self.model.eval()
        z_env_all = []

        with torch.no_grad():
            for batch in self.dataloader:
                x, _, _ = batch
                x = x.to(self.device)

                z_env = self._featurizer.extract_env(x)
                z_env_all.append(z_env.detach().cpu())

        z_env_cat = torch.cat(z_env_all, dim=0)  # (N_local, proj_dim)
        mu = z_env_cat.mean(dim=0).numpy()
        # Use diagonal covariance for efficiency (+ numerical stability)
        cov = torch.diag(z_env_cat.var(dim=0) + 1e-6).numpy()

        self.local_env_stats = {"mean": mu, "covariance": cov}
        self.model.to("cpu")

    # ── Training loop ─────────────────────────────────────────────

    def fit(self, server_round):
        """Train locally, then compute prototypes and environment stats.

        Returns dict of average losses for server-level aggregation.
        """
        self.current_server_round = server_round
        self.init_train()
        loss_keys = [
            "task",
            "cls",
            "stat",
            "align",
            "cov",
            "cov_internal",
            "cov_cross",
            "cf",
        ]

        total_training_loss = 0.0
        total_loss_accum = {k: 0.0 for k in loss_keys}

        for e in range(self.local_epochs):
            training_loss = 0.0
            loss_accum = {k: 0.0 for k in loss_keys}

            for batch in tqdm(self.dataloader):
                results = self.process_batch(batch)
                training_loss += self.step(results)
                for k in loss_keys:
                    loss_accum[k] += self._last_losses[k]

            if self.hparam["wandb"]:
                n = len(self.dataset)
                wandb.log(
                    {
                        f"loss/{self.client_id}": training_loss / n,
                        f"loss_task/{self.client_id}": loss_accum["task"] / n,
                        f"loss_cls/{self.client_id}": loss_accum["cls"] / n,
                        f"loss_stat/{self.client_id}": loss_accum["stat"] / n,
                        f"loss_align/{self.client_id}": loss_accum["align"] / n,
                        f"loss_cov/{self.client_id}": loss_accum["cov"] / n,
                        f"loss_cov_internal/{self.client_id}": loss_accum[
                            "cov_internal"
                        ]
                        / n,
                        f"loss_cov_cross/{self.client_id}": loss_accum["cov_cross"] / n,
                        f"loss_cf/{self.client_id}": loss_accum["cf"] / n,
                    },
                    step=server_round * self.local_epochs + e,
                )

            total_training_loss += training_loss
            for k in loss_keys:
                total_loss_accum[k] += loss_accum[k]

        self.end_train()
        self.compute_local_prototypes()
        self.compute_local_env_stats()

        n_total = len(self.dataset) * max(1, self.local_epochs)
        return {
            "total": total_training_loss / n_total,
            "task": total_loss_accum["task"] / n_total,
            "cls": total_loss_accum["cls"] / n_total,
            "stat": total_loss_accum["stat"] / n_total,
            "align": total_loss_accum["align"] / n_total,
            "cov": total_loss_accum["cov"] / n_total,
            "cov_internal": total_loss_accum["cov_internal"] / n_total,
            "cov_cross": total_loss_accum["cov_cross"] / n_total,
            "cf": total_loss_accum["cf"] / n_total,
        }


# ═══════════════════════════════════════════════════════════════════
# FCDv2 client
#
# Five-term objective:
#     L_total = lambda_task * L_task        (CE, original + cyclic, both views)
#             + lambda_inv  * L_inv         (BYOL-style ||z_inv_1 - z_inv_2||^2)
#             + lambda_stat * L_stat        (statistical grounding, both views)
#             + lambda_cov  * L_cov_cross   (cross-subspace orthogonality)
#             + lambda_var  * L_var         (VICReg variance preservation)
#
# Differences vs FCD:
#   * Two augmented views per sample via TwinViewAugmenter
#   * L_align (prototype alignment) and the internal-decorrelation half
#     of L_cov are dropped
#   * Cyclic counterfactual sampling uses a pluggable
#     FederatedStyleAggregator (gaussian / gmm / vae / realnvp) supplied
#     by the server, not a hardcoded GMM
# ═══════════════════════════════════════════════════════════════════


class FCDv2Client(ERM):
    """FCDv2 client: twin-view augmentation + five-term loss + cyclic CF.

    The aggregator object is set per round by the server via
    ``set_aggregator``. When the aggregator is missing or unfitted (e.g.
    round 0), the cyclic pathway is skipped and the task loss reduces to
    the cross-entropy on the two augmented original-view forwards.
    """

    def __init__(self, client_id, device, dataset, ds_bundle, hparam):
        super().__init__(client_id, device, dataset, ds_bundle, hparam)
        from .fcdv2_augmentations import TwinViewAugmenter

        self.lambda_task = hparam.get("fcdv2_lambda_task", 1.0)
        self.lambda_inv = hparam.get("fcdv2_lambda_inv", 1.0)
        self.lambda_stat = hparam.get("fcdv2_lambda_stat", 1.0)
        self.lambda_cov = hparam.get("fcdv2_lambda_cov", 1.0)
        self.lambda_var = hparam.get("fcdv2_lambda_var", 1.0)
        self.eps = hparam.get("eps", 1e-8)
        self.fcd_cf_start_round = hparam.get("fcd_cf_start_round", 1)

        self.augmenter = TwinViewAugmenter()
        self.aggregator = None  # set per round by FCDv2Server.transmit_model
        self.local_env_stats = None

    # ── Setup ─────────────────────────────────────────────────────

    def setup_model(self, featurizer, classifier, style_encoder):
        from .models import FCDv2ModelWrapper

        self._featurizer = featurizer
        self._classifier = classifier
        self._style_encoder = style_encoder
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(
            FCDv2ModelWrapper(self._featurizer, self._classifier, self._style_encoder)
        )

    def update_model(self, model_dict):
        self.model.load_state_dict(model_dict)

    def set_aggregator(self, aggregator):
        """Receive the latest fitted aggregator from the server."""
        self.aggregator = aggregator

    # ── Forward ───────────────────────────────────────────────────

    def process_batch(self, batch):
        """Run the model on two augmented views of the input.

        Returns a dict with per-view tensors keyed ``*_1`` / ``*_2`` plus
        the labels and metadata that are shared across views.
        """
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)

        x1, x2 = self.augmenter(x)

        logits_1, z_inv_1, z_env_1, H_1, gamma_1, beta_1 = self.model(x1)
        logits_2, z_inv_2, z_env_2, H_2, gamma_2, beta_2 = self.model(x2)

        return {
            "g": g,
            "y_true": y_true,
            "metadata": metadata,
            "logits_1": logits_1,
            "z_inv_1": z_inv_1,
            "z_env_1": z_env_1,
            "H_1": H_1,
            "gamma_1": gamma_1,
            "beta_1": beta_1,
            "logits_2": logits_2,
            "z_inv_2": z_inv_2,
            "z_env_2": z_env_2,
            "H_2": H_2,
            "gamma_2": gamma_2,
            "beta_2": beta_2,
        }

    # ── Loss components ───────────────────────────────────────────

    @staticmethod
    def _invariance_loss(z_inv_1, z_inv_2):
        """L_inv = ||z_inv_1 - z_inv_2||^2 / d (BYOL-style alignment)."""
        d = z_inv_1.shape[-1]
        return ((z_inv_1 - z_inv_2) ** 2).sum(dim=-1).mean() / d

    @staticmethod
    def _statistical_grounding_loss(H, gamma_hat, beta_hat, eps=1e-8):
        mu_H = H.mean(dim=[2, 3])
        sigma_H = (H.var(dim=[2, 3]) + eps).sqrt()
        return F.mse_loss(sigma_H, gamma_hat) + F.mse_loss(mu_H, beta_hat)

    @staticmethod
    def _cross_covariance_loss(z_inv, z_env):
        """Cross-subspace orthogonality only; no internal decorrelation."""
        n = z_inv.size(0)
        d_inv = z_inv.size(1)
        denom = max(n - 1, 1)
        z_inv_c = z_inv - z_inv.mean(dim=0)
        z_env_c = z_env - z_env.mean(dim=0)
        cross_cov = (z_inv_c.T @ z_env_c) / denom
        return (cross_cov ** 2).sum() / d_inv

    @staticmethod
    def _variance_loss(z, eps=1e-4):
        """VICReg-style variance preservation: encourage per-feature std >= 1."""
        std = (z.var(dim=0, unbiased=False) + eps).sqrt()
        return F.relu(1.0 - std).mean()

    def _sample_z_env_sim(self, n_samples):
        """Draw counterfactual style codes from the server's aggregator."""
        if self.aggregator is None or not getattr(self.aggregator, "fitted", False):
            return None
        with torch.no_grad():
            z = self.aggregator.sample(n_samples)
        return z.to(self.device)

    def _counterfactual_logits(self, H, eps=1e-8):
        """Apply foreign style via pooled-AdaIN, return reprojected logits.

        Returns ``None`` when the aggregator is unavailable or has not yet
        been fitted (round 0 or before ``fcd_cf_start_round``).
        """
        B = H.size(0)
        z_env_sim = self._sample_z_env_sim(B)
        if z_env_sim is None:
            return None
        gamma_sim, beta_sim = self._style_encoder(z_env_sim)
        gamma_sim = gamma_sim.detach()
        beta_sim = beta_sim.detach()

        pooled = self._featurizer.gap(H).flatten(1)
        mu_p = pooled.mean(dim=1, keepdim=True)
        sigma_p = (pooled.var(dim=1, keepdim=True) + eps).sqrt()
        pooled_norm = (pooled - mu_p) / sigma_p
        pooled_cf = gamma_sim * pooled_norm + beta_sim
        z_inv_cf = self._featurizer.h_inv(pooled_cf)
        return self._classifier(z_inv_cf)

    # ── Training step ─────────────────────────────────────────────

    def step(self, results):
        """Compose the five-term FCDv2 objective and backpropagate."""
        y_true = results["y_true"]
        z_inv_1, z_inv_2 = results["z_inv_1"], results["z_inv_2"]
        z_env_1, z_env_2 = results["z_env_1"], results["z_env_2"]
        H_1, H_2 = results["H_1"], results["H_2"]
        gamma_1, gamma_2 = results["gamma_1"], results["gamma_2"]
        beta_1, beta_2 = results["beta_1"], results["beta_2"]
        logits_1, logits_2 = results["logits_1"], results["logits_2"]

        ce = nn.CrossEntropyLoss()

        # L_inv
        loss_inv = self._invariance_loss(z_inv_1, z_inv_2)

        # L_stat (averaged across the two views)
        loss_stat = 0.5 * (
            self._statistical_grounding_loss(H_1, gamma_1, beta_1, eps=self.eps)
            + self._statistical_grounding_loss(H_2, gamma_2, beta_2, eps=self.eps)
        )

        # L_cov_cross (averaged across the two views)
        loss_cov = 0.5 * (
            self._cross_covariance_loss(z_inv_1, z_env_1)
            + self._cross_covariance_loss(z_inv_2, z_env_2)
        )

        # L_var (both subspaces, both views, summed)
        loss_var = (
            self._variance_loss(z_inv_1)
            + self._variance_loss(z_env_1)
            + self._variance_loss(z_inv_2)
            + self._variance_loss(z_env_2)
        )

        # L_task
        ce_orig = 0.5 * (ce(logits_1, y_true) + ce(logits_2, y_true))
        cf_active = (
            self.aggregator is not None
            and getattr(self.aggregator, "fitted", False)
            and getattr(self, "current_server_round", 0) >= self.fcd_cf_start_round
        )
        if cf_active:
            logits_cf_1 = self._counterfactual_logits(H_1, eps=self.eps)
            logits_cf_2 = self._counterfactual_logits(H_2, eps=self.eps)
            ce_cf = 0.5 * (ce(logits_cf_1, y_true) + ce(logits_cf_2, y_true))
            loss_task = 0.5 * (ce_orig + ce_cf)
        else:
            loss_task = ce_orig
            ce_cf = torch.tensor(0.0, device=self.device)

        objective = (
            self.lambda_task * loss_task
            + self.lambda_inv * loss_inv
            + self.lambda_stat * loss_stat
            + self.lambda_cov * loss_cov
            + self.lambda_var * loss_var
        )

        batch_size = z_inv_1.size(0)
        total_loss = objective.item() * batch_size

        self._last_losses = {
            "task": loss_task.item() * batch_size,
            "task_orig": ce_orig.item() * batch_size,
            "inv": loss_inv.item() * batch_size,
            "stat": loss_stat.item() * batch_size,
            "cov_cross": loss_cov.item() * batch_size,
            "var": loss_var.item() * batch_size,
            "cf": (ce_cf.item() if torch.is_tensor(ce_cf) else float(ce_cf)) * batch_size,
        }

        if objective.grad_fn is not None:
            objective.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss

    # ── Local statistics (unchanged structure) ────────────────────

    def compute_local_env_stats(self):
        """Empirical (mu, diag-Sigma) of z_env over local data on ORIGINAL images."""
        self.model.to(self.device)
        self.model.eval()
        z_env_all = []
        with torch.no_grad():
            for batch in self.dataloader:
                x, _, _ = batch
                x = x.to(self.device)
                z_env_all.append(self._featurizer.extract_env(x).detach().cpu())
        z_env_cat = torch.cat(z_env_all, dim=0)
        mu = z_env_cat.mean(dim=0).numpy()
        cov = torch.diag(z_env_cat.var(dim=0) + 1e-6).numpy()
        self.local_env_stats = {"mean": mu, "covariance": cov}
        self.model.to("cpu")

    # ── Training loop ─────────────────────────────────────────────

    def fit(self, server_round):
        self.current_server_round = server_round
        self.init_train()
        loss_keys = ["task", "task_orig", "inv", "stat", "cov_cross", "var", "cf"]
        total_training_loss = 0.0
        loss_accum = {k: 0.0 for k in loss_keys}

        for e in range(self.local_epochs):
            for batch in tqdm(self.dataloader):
                results = self.process_batch(batch)
                total_training_loss += self.step(results)
                for k in loss_keys:
                    loss_accum[k] += self._last_losses[k]

            if self.hparam.get("wandb", False):
                n = len(self.dataset)
                wandb.log(
                    {
                        f"loss/{self.client_id}": total_training_loss / n,
                        **{
                            f"loss_{k}/{self.client_id}": loss_accum[k] / n
                            for k in loss_keys
                        },
                    },
                    step=server_round * self.local_epochs + e,
                )

        self.end_train()
        self.compute_local_env_stats()

        n_total = len(self.dataset) * max(1, self.local_epochs)
        return {
            "total": total_training_loss / n_total,
            **{k: loss_accum[k] / n_total for k in loss_keys},
        }
