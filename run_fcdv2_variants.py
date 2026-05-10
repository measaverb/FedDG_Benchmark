"""Driver: run FCDv2 across aggregator/backbone/LODO/seed variants.

Each combination:

* aggregator_type   {gaussian, gmm, vae, realnvp}
* fcd_backbone      {resnet18, resnet50}
* split_scheme      {pcs-a, pas-c, pac-s, acs-p}   (PACS LODO)
* seed              user-specified list

is materialised into a temporary JSON config (cloned from a template),
then ``main.py`` is invoked as a subprocess. Each run logs to its own
wandb run; ``wandb_group`` is set to the matrix coordinates so runs are
grouped sensibly in the UI.

The actual training-time and end-of-training metrics requested in the
task spec are emitted by ``FCDv2Server`` itself (see ``src/server.py``)
and ``src/fcdv2_eval.py``; this driver is just the matrix runner.

Examples
--------
Smoke-test a single cell::

    python run_fcdv2_variants.py \
        --aggregators gaussian --backbones resnet18 \
        --split_schemes pcs-a --seeds 1001 \
        --num_rounds 2 --batch_size 8 --num_clients 4

Full sweep (default)::

    python run_fcdv2_variants.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs" / "fcdv2"
TMP_DIR = REPO_ROOT / "configs" / "fcdv2" / "_generated"

DEFAULT_AGGREGATORS = ["gaussian", "gmm", "vae", "realnvp"]
DEFAULT_BACKBONES = ["resnet18", "resnet50"]
DEFAULT_SPLIT_SCHEMES = ["pcs-a", "pas-c", "pac-s", "acs-p"]  # PACS LODO
DEFAULT_SEEDS = [1001]


def _template_for(backbone: str) -> Path:
    """Pick an existing config file as the JSON template.

    The aggregator-specific fields (lambda_*, etc.) carry over unchanged;
    we override only the dimensions of the sweep.
    """
    if backbone == "resnet18":
        return CONFIG_DIR / "fcdv2_pacs_gaussian.json"
    if backbone == "resnet50":
        return CONFIG_DIR / "fcdv2_pacs_r50_gaussian.json"
    raise ValueError(f"unsupported backbone: {backbone}")


def _materialise_config(template: Path, overrides: dict, run_id: str) -> Path:
    with open(template) as fh:
        cfg = json.load(fh)
    cfg = deepcopy(cfg)
    cfg.update(overrides)
    cfg["id"] = run_id
    cfg["wandb_group"] = run_id

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = TMP_DIR / f"{run_id}.json"
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return out


def _build_overrides(args, aggregator: str, backbone: str, split_scheme: str,
                     seed: int) -> dict:
    """Compute the JSON overrides for one matrix cell."""
    overrides: dict = {
        "aggregator_type": aggregator,
        "fcd_backbone": backbone,
        "split_scheme": split_scheme,
        "seed": seed,
    }
    # Optional CLI overrides for quick smoke-tests.
    if args.num_rounds is not None:
        overrides["num_rounds"] = args.num_rounds
    if args.num_clients is not None:
        overrides["num_clients"] = args.num_clients
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.local_epochs is not None:
        overrides["local_epochs"] = args.local_epochs
    return overrides


def _run_one(config_path: Path, args, run_id: str) -> int:
    cmd = [sys.executable, "main.py", "--config_file", str(config_path)]
    if args.no_wandb:
        cmd.append("--no_wandb")
    print(f"\n[run_fcdv2_variants] launching {run_id}")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregators", nargs="+", default=DEFAULT_AGGREGATORS,
                        choices=["gaussian", "gmm", "vae", "realnvp"])
    parser.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES,
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--split_schemes", nargs="+", default=DEFAULT_SPLIT_SCHEMES,
                        help="PACS leave-one-domain-out CSV stems")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)

    parser.add_argument("--num_rounds", type=int, default=None,
                        help="override num_rounds in the generated config")
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)

    parser.add_argument("--no_wandb", action="store_true",
                        help="skip wandb logging (passed to main.py)")
    parser.add_argument("--dry_run", action="store_true",
                        help="print commands but don't execute")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="keep going if one cell fails")
    args = parser.parse_args()

    matrix = []
    for agg in args.aggregators:
        for bb in args.backbones:
            for sch in args.split_schemes:
                for sd in args.seeds:
                    matrix.append((agg, bb, sch, sd))

    print(f"[run_fcdv2_variants] sweep size: {len(matrix)} runs")
    failures = []

    t_start = time.time()
    for idx, (agg, bb, sch, sd) in enumerate(matrix, start=1):
        run_id = f"fcdv2_{agg}_{bb}_{sch}_seed{sd}"
        template = _template_for(bb)
        overrides = _build_overrides(args, agg, bb, sch, sd)
        cfg_path = _materialise_config(template, overrides, run_id)

        rc = _run_one(cfg_path, args, run_id)
        elapsed = time.time() - t_start
        print(
            f"[run_fcdv2_variants] [{idx}/{len(matrix)}] {run_id} -> "
            f"rc={rc} elapsed={elapsed:.0f}s"
        )
        if rc != 0:
            failures.append((run_id, rc))
            if not args.continue_on_error:
                break

    if failures:
        print("\n[run_fcdv2_variants] FAILED RUNS:")
        for rid, rc in failures:
            print(f"  {rid}  rc={rc}")
        sys.exit(1)

    print("[run_fcdv2_variants] ALL RUNS COMPLETED")


if __name__ == "__main__":
    main()
