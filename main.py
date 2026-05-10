import argparse
import gc

import torch
import torch.cuda
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

import src.datasets as my_datasets
import wandb
from src.client import *
from src.dataset_bundle import *
from src.server import *
from src.splitter import *
from src.utils import *
from wandb_env import WANDB_ENTITY, WANDB_PROJECT

"""
The main file function:
1. Load the hyperparameter dict.
2. Initialize logger
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""


def main(args):
    hparam = vars(args)
    config_file = args.config_file
    with open(config_file) as fh:
        config = json.load(fh)
    hparam.update(config)
    wandb_project = WANDB_PROJECT + "_" + hparam["dataset"]
    # setup WanDB
    if not args.no_wandb:
        wandb.init(project=wandb_project, entity=WANDB_ENTITY, config=hparam, group=hparam.get("wandb_group", None))
        wandb.run.log_code()
        config["wandb"] = True
    else:
        config["wandb"] = False
    hparam.update(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    seed = hparam["seed"]
    set_seed(seed)
    data_path = hparam["data_path"]
    if not os.path.exists(data_path + "opt_dict/"):
        os.makedirs(data_path + "opt_dict/")
    if not os.path.exists(data_path + "models/"):
        os.makedirs(data_path + "models/")

    # optimizer preprocess
    if hparam["optimizer"] == "torch.optim.SGD":
        hparam["optimizer_config"] = {
            "lr": hparam["lr"],
            "momentum": hparam["momentum"],
            "weight_decay": hparam["weight_decay"],
        }
    elif (
        hparam["optimizer"] == "torch.optim.Adam"
        or hparam["optimizer"] == "torch.optim.AdamW"
    ):
        hparam["optimizer_config"] = {
            "lr": hparam["lr"],
            "eps": hparam["eps"],
            "weight_decay": hparam["weight_decay"],
        }

    # initialize data
    if hparam["dataset"].lower() == "pacs":
        dataset = my_datasets.PACS(
            version="1.0", root_dir=hparam["dataset_path"], download=True
        )
    elif hparam["dataset"].lower() == "officehome":
        dataset = my_datasets.OfficeHome(
            version="1.0",
            root_dir=hparam["dataset_path"],
            download=True,
            split_scheme=hparam["split_scheme"],
        )
    elif hparam["dataset"].lower() == "femnist":
        dataset = my_datasets.FEMNIST(
            version="1.0", root_dir=hparam["dataset_path"], download=True
        )
    elif hparam["dataset"].lower() == "celeba":
        dataset = get_dataset(
            dataset="celebA", root_dir=hparam["dataset_path"], download=True
        )
    else:
        dataset = get_dataset(
            dataset=hparam["dataset"].lower(),
            root_dir=hparam["dataset_path"],
            download=True,
        )

    if hparam["client_method"] == "FedSR":
        ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=True)
    else:
        if (
            hparam["dataset"].lower() == "py150"
            or hparam["dataset"].lower() == "civilcomments"
        ):
            ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=False)
        else:
            ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=False)
    if hparam["server_method"] == "FedDG":
        if hparam["dataset"].lower() == "iwildcam":
            dataset = my_datasets.FourierIwildCam(
                root_dir=hparam["dataset_path"], download=True
            )
            total_subset = dataset.get_subset(
                "train", transform=ds_bundle.test_transform
            )
        elif hparam["dataset"].lower() == "pacs":
            dataset = my_datasets.FourierPACS(
                root_dir=hparam["dataset_path"],
                download=True,
                split_scheme=hparam["split_scheme"],
            )
            total_subset = dataset.get_subset(
                "train", transform=ds_bundle.test_transform
            )
        elif hparam["dataset"].lower() == "celeba":
            dataset = my_datasets.FourierCelebA(
                root_dir=hparam["dataset_path"],
                download=True,
                split_scheme=hparam["split_scheme"],
            )
            total_subset = dataset.get_subset(
                "train", transform=ds_bundle.test_transform
            )
        elif hparam["dataset"].lower() == "camelyon17":
            dataset = my_datasets.FourierCamelyon17(
                root_dir=hparam["dataset_path"],
                download=True,
                split_scheme=hparam["split_scheme"],
            )
            total_subset = dataset.get_subset(
                "train", transform=ds_bundle.test_transform
            )
        elif hparam["dataset"].lower() == "femnist":
            dataset = my_datasets.FourierFEMNIST(
                root_dir=hparam["dataset_path"],
                download=True,
                split_scheme=hparam["split_scheme"],
            )
            total_subset = dataset.get_subset(
                "train", transform=ds_bundle.test_transform
            )
        else:
            raise NotImplementedError
    else:
        total_subset = dataset.get_subset("train", transform=ds_bundle.train_transform)

    testloader = {}
    for split in dataset.split_names:
        if split != "train":
            ds = dataset.get_subset(split, transform=ds_bundle.test_transform)
            dl = get_eval_loader(
                loader="standard", dataset=ds, batch_size=hparam["batch_size"]
            )
            testloader[split] = dl

    sampler = RandomSampler(total_subset, replacement=True)
    global_dataloader = DataLoader(
        total_subset, batch_size=hparam["batch_size"], sampler=sampler
    )

    num_shards = hparam["num_clients"]
    if num_shards == 1:
        training_datasets = [total_subset]
    elif num_shards > 1:
        training_datasets = NonIIDSplitter(
            num_shards=num_shards, iid=hparam["iid"], seed=seed
        ).split(
            dataset.get_subset("train"),
            ds_bundle.groupby_fields,
            transform=ds_bundle.train_transform,
        )
    else:
        raise ValueError(
            "num_shards should be greater or equal to 1, we got {}".format(num_shards)
        )

    # initialize client
    clients = []
    for k in tqdm(range(hparam["num_clients"]), leave=False):
        client = eval(hparam["client_method"])(
            k, device, training_datasets[k], ds_bundle, hparam
        )
        clients.append(client)
    message = f"successfully initialize all clients!"
    logging.info(message)

    gc.collect()

    # initialize server (model should be initialized in the server. )
    central_server = eval(hparam["server_method"])(device, ds_bundle, hparam)
    if hparam["server_method"] == "FedDG":
        central_server.set_amploader(global_dataloader)
    if hparam["start_epoch"] == 0:
        central_server.setup_model(None, 0)
    else:
        central_server.setup_model(hparam["resume_file"], hparam["start_epoch"])
    central_server.register_clients(clients)
    central_server.register_testloader(testloader)
    # do federated learning
    central_server.fit()

    # bye!
    message = "...done all learning process!\n...exit program!"
    logging.info(message)
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedDG Benchmark")
    parser.add_argument("--config_file", help="config file", default="config.json")
    parser.add_argument("--no_wandb", default=False, action="store_true")
    parser.add_argument("--seed", default=1001, type=int)
    parser.add_argument("--num_clients", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--iid", default=1, type=float)
    parser.add_argument("--server_method", default="FedAvg")
    parser.add_argument("--fraction", default=1, type=float)
    parser.add_argument("--f", default=10, type=int)
    parser.add_argument("--num_rounds", default=20, type=int)
    parser.add_argument("--dataset", default="camelyon17")
    parser.add_argument("--split_scheme", default="official")
    parser.add_argument("--client_method", default="ERM")
    parser.add_argument("--local_epochs", default=1, type=int)
    parser.add_argument("--n_groups_per_batch", default=2, type=int)
    parser.add_argument("--optimizer", default="torch.optim.Adam")
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--momentum", default=0, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument(
        "--hparam1",
        default=1,
        type=float,
        help="irm: lambda; rex: lambda; fish: meta_lr; mixup: alpha; mmd: lambda; coral: lambda; groupdro: groupdro_eta; fedprox: mu; feddg: ratio; fedadg: alpha; fedgma: mask_threshold; fedsr: l2_regularizer;",
    )
    parser.add_argument(
        "--hparam2",
        default=1,
        type=float,
        help="fedsr: cmi_regularizer; irm: penalty_anneal_iters; fedadg: second_local_epochs",
    )
    parser.add_argument("--hparam3", default=0, type=float)
    parser.add_argument("--hparam4", default=0, type=float)
    parser.add_argument("--hparam5", default=0, type=float)
    parser.add_argument(
        "--ffd_alpha", default=0.1, type=float, help="Alpha for FFD prototype momentum"
    )
    parser.add_argument("--ffd_lambda_cov", default=0.05, type=float)
    parser.add_argument("--ffd_lambda_var", default=0.5, type=float)
    parser.add_argument("--ffd_warmup_rounds", default=5, type=int)

    # FCD (Federated Cyclic Disentanglement) arguments
    parser.add_argument(
        "--fcd_backbone",
        default="resnet18",
        type=str,
        choices=["resnet18", "resnet50"],
        help="Backbone architecture for FCD (resnet18 or resnet50)",
    )
    parser.add_argument(
        "--fcd_proj_dim",
        default=256,
        type=int,
        help="FCD projection dimension (z_inv, z_env)",
    )
    parser.add_argument(
        "--fcd_alpha", default=1.0, type=float, help="FCD prototype alignment weight"
    )
    parser.add_argument(
        "--fcd_warmup_rounds",
        default=10,
        type=int,
        help="FCD warm-up rounds for alpha ramp",
    )
    parser.add_argument(
        "--fcd_lambda_task", default=1.0, type=float, help="FCD task loss weight"
    )
    parser.add_argument(
        "--fcd_lambda_stat",
        default=1.0,
        type=float,
        help="FCD statistical grounding loss weight",
    )
    parser.add_argument(
        "--fcd_lambda_reg",
        default=1.0,
        type=float,
        help="FCD regularisation loss weight (align + cov)",
    )
    parser.add_argument(
        "--fcd_gmm_components",
        default=8,
        type=int,
        help="Number of GMM components for environment model",
    )
    parser.add_argument(
        "--fcd_gmm_pseudo_samples",
        default=200,
        type=int,
        help="Pseudo-samples per client for GMM fitting",
    )
    parser.add_argument(
        "--fcd_cf_start_round",
        default=15,
        type=int,
        help="Round at which counterfactual augmentation begins",
    )
    parser.add_argument(
        "--fcd_aug_source",
        default="gmm",
        type=str,
        choices=["isotropic", "local", "global_unimodal", "gmm"],
        help="Augmentation source for FCD counterfactual pathway ablation",
    )

    args = parser.parse_args()
    main(args)
