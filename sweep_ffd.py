#!/usr/bin/env python3
"""FFD Hyperparameter Tuning Script.

Supports three modes:

  1. bayesian (default) — Optuna TPE sampler for sample-efficient Bayesian search
  2. wandb             — W&B Sweeps with Bayesian optimisation
  3. grid              — Exhaustive grid search via subprocesses

Usage:
  # Bayesian tuning with Optuna (default, 50 trials):
  python sweep_ffd.py

  # Bayesian with custom trial budget:
  python sweep_ffd.py --n_trials 30

  # Bayesian dry-run (prints search space, no execution):
  python sweep_ffd.py --dry_run

  # W&B Bayesian sweep:
  python sweep_ffd.py --mode wandb

  # Exhaustive grid search:
  python sweep_ffd.py --mode grid --dry_run

  # Override search bounds (bayesian mode):
  python sweep_ffd.py --ffd_alpha_min 0.1 --ffd_alpha_max 2.0

Notes:
  - Bayesian mode requires `optuna` (pip install optuna).
  - The objective is val.acc_avg logged by the training run.
  - Results are saved to an Optuna SQLite DB and a summary CSV.
"""

import argparse
import copy
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── Default search space ──────────────────────────────────────────────
# For Bayesian: continuous/discrete ranges.  For grid: explicit lists.

BAYESIAN_SPACE = {
    "ffd_alpha": {"low": 0.05, "high": 3.0, "log": True},
    "ffd_warmup_rounds": {"low": 3, "high": 30, "type": "int"},
    "ffd_lambda_var": {"low": 0.1, "high": 5.0, "log": True},
    "ffd_lambda_cov": {"low": 0.005, "high": 0.5, "log": True},
}

GRID_SPACE = {
    "ffd_alpha": [0.1, 0.5, 1.0, 2.0],
    "ffd_warmup_rounds": [5, 10, 20],
    "ffd_lambda_var": [0.5, 1.0, 2.0],
    "ffd_lambda_cov": [0.01, 0.04, 0.1],
}

BASE_CONFIG = "configs/ffd/ffd_pacs.json"


# ── Helpers ───────────────────────────────────────────────────────────


def _make_temp_config(base: dict, overrides: dict) -> str:
    """Write a temporary JSON config with overrides applied."""
    merged = copy.deepcopy(base)
    merged.update(overrides)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ffd_sweep_")
    with os.fdopen(fd, "w") as f:
        json.dump(merged, f, indent=2)
    return path


def _combo_str(combo: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in sorted(combo.items()))


def _run_experiment(combo: dict, base_config: dict, args) -> float | None:
    """Run a single training experiment and return val.acc_avg (or None on failure).

    Parses the metric from stdout/wandb logs. Falls back to scanning the
    subprocess output for the metric pattern.
    """
    cfg_path = _make_temp_config(base_config, combo)

    cmd = [sys.executable, "main.py", "--config_file", cfg_path]
    if args.no_wandb:
        cmd.append("--no_wandb")

    print(f"\n{'─'*60}")
    print(f"  Trial params: {_combo_str(combo)}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            timeout=args.timeout,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        rc = result.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {args.timeout}s")
        _cleanup(cfg_path, args)
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        _cleanup(cfg_path, args)
        return None

    _cleanup(cfg_path, args)

    if rc != 0:
        print(f"  Run failed (rc={rc})")
        return None

    # Parse val.acc_avg from output.
    # The framework typically logs lines like: "val.acc_avg: 0.8234"
    metric = _parse_metric(output)
    if metric is not None:
        print(f"  ✓ val.acc_avg = {metric:.4f}")
    else:
        print("  ⚠ Could not parse val.acc_avg from output")
    return metric


def _parse_metric(output: str) -> float | None:
    """Extract the last val.acc_avg value from training output."""
    # Try common log patterns
    patterns = [
        r"val\.acc_avg[:\s]+([0-9]+\.[0-9]+)",
        r"val_acc_avg[:\s]+([0-9]+\.[0-9]+)",
        r"'val\.acc_avg'[:\s]+([0-9]+\.[0-9]+)",
        r"\"val\.acc_avg\"[:\s]+([0-9]+\.[0-9]+)",
        r"acc_avg[:\s]+([0-9]+\.[0-9]+)",
    ]
    last_match = None
    for pat in patterns:
        matches = re.findall(pat, output)
        if matches:
            last_match = float(matches[-1])
    return last_match


def _cleanup(cfg_path: str, args):
    if not args.keep_configs:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


# ── Bayesian mode (Optuna) ────────────────────────────────────────────


def run_bayesian(args, space: dict):
    """Bayesian hyperparameter optimisation using Optuna TPE sampler."""
    try:
        import optuna
    except ImportError:
        print(
            "ERROR: optuna is not installed.\n"
            "Install with:  pip install optuna\n"
            "Optional viz:  pip install optuna-dashboard",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.base_config) as f:
        base_config = json.load(f)

    print(f"FFD Bayesian Hyperparameter Tuning (Optuna TPE)")
    print(f"  Trials:      {args.n_trials}")
    print(f"  Base config:  {args.base_config}")
    print(f"  Search space:")
    for k, v in sorted(space.items()):
        print(f"    {k}: {v}")
    print()

    if args.dry_run:
        print("=== DRY RUN — no trials will be executed ===")
        return

    # Results directory + CSV
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"ffd_bayesian_{timestamp}.csv"
    db_path = results_dir / f"ffd_bayesian_{timestamp}.db"

    # Create Optuna study
    study = optuna.create_study(
        study_name=f"ffd_bayesian_{timestamp}",
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    # Write CSV header
    param_keys = sorted(space.keys())
    with open(csv_path, "w") as f:
        f.write(",".join(["trial"] + param_keys + ["val_acc_avg", "status"]) + "\n")

    def objective(trial: optuna.Trial) -> float:
        combo = {}
        for k, spec in sorted(space.items()):
            if spec.get("type") == "int":
                combo[k] = trial.suggest_int(k, spec["low"], spec["high"])
            elif spec.get("log", False):
                combo[k] = trial.suggest_float(k, spec["low"], spec["high"], log=True)
            else:
                combo[k] = trial.suggest_float(k, spec["low"], spec["high"])

        metric = _run_experiment(combo, base_config, args)

        # Log to CSV
        row = [str(trial.number)] + [str(combo[k]) for k in param_keys]
        if metric is not None:
            row += [f"{metric:.6f}", "COMPLETE"]
        else:
            row += ["", "FAIL"]
        with open(csv_path, "a") as f:
            f.write(",".join(row) + "\n")

        if metric is None:
            raise optuna.TrialPruned("Run failed or metric not found")
        return metric

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Bayesian Sweep Complete")
    print(f"{'='*60}")
    print(f"  Best val.acc_avg: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}: {v}")
    print(f"\n  Results CSV:    {csv_path}")
    print(f"  Optuna DB:      {db_path}")
    print(f"  View dashboard: optuna-dashboard sqlite:///{db_path}")
    print(f"{'='*60}\n")

    # Also save best params as a JSON config for easy reuse
    best_config_path = results_dir / f"ffd_best_{timestamp}.json"
    best_cfg = copy.deepcopy(base_config)
    best_cfg.update(study.best_params)
    with open(best_config_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"  Best config saved to: {best_config_path}")


# ── W&B Bayesian sweep mode ──────────────────────────────────────────


def run_wandb_sweep(args):
    """Create and run a W&B Bayesian sweep."""
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is not installed. pip install wandb", file=sys.stderr)
        sys.exit(1)

    sweep_config_path = (
        Path(__file__).parent / "sweep" / "hparam_search" / "pacs" / "ffd_hs.json"
    )
    if not sweep_config_path.exists():
        print(f"ERROR: sweep config not found at {sweep_config_path}", file=sys.stderr)
        sys.exit(1)

    with open(sweep_config_path) as f:
        sweep_config = json.load(f)

    sys.path.insert(0, str(Path(__file__).parent))
    from wandb_env import WANDB_ENTITY, WANDB_PROJECT

    dataset = sweep_config["parameters"]["dataset"]["values"][0]
    project = f"{WANDB_PROJECT}_{dataset}"

    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=WANDB_ENTITY)
    print(f"Sweep ID: {sweep_id}")
    print(f"Dashboard: https://wandb.ai/{WANDB_ENTITY}/{project}/sweeps/{sweep_id}")

    # Workaround for wandb 0.25.0 bug: is_flapping() references
    # wandb.START_TIME which doesn't exist in this version.
    import time as _time

    if not hasattr(wandb, "START_TIME"):
        wandb.START_TIME = _time.time()

    wandb.agent(sweep_id, count=args.max_runs if args.max_runs > 0 else None)


# ── Grid search mode ─────────────────────────────────────────────────


def run_grid(args, grid: dict):
    """Exhaustive grid search via subprocesses."""
    with open(args.base_config) as f:
        base_config = json.load(f)

    keys = sorted(grid.keys())
    combos = [
        dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))
    ]
    total = len(combos)

    print(f"FFD Grid Search — {total} combinations")
    print(f"Grid: {json.dumps(grid, indent=2)}")
    print(f"Base config: {args.base_config}\n")

    if args.dry_run:
        print("=== DRY RUN — commands that would be executed ===\n")
        for i, combo in enumerate(combos, 1):
            print(f"[{i:3d}/{total}] {_combo_str(combo)}")
        print(f"\nTotal: {total} experiments")
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"ffd_grid_{timestamp}.csv"

    param_keys = sorted(grid.keys())
    with open(csv_path, "w") as f:
        f.write(",".join(param_keys + ["val_acc_avg", "return_code"]) + "\n")

    for i, combo in enumerate(combos, 1):
        print(f"\n[{i}/{total}] {_combo_str(combo)}")
        metric = _run_experiment(combo, base_config, args)

        row_vals = [str(combo[k]) for k in param_keys]
        row_vals += [f"{metric:.6f}" if metric else "", "0" if metric else "-1"]
        with open(csv_path, "a") as f:
            f.write(",".join(row_vals) + "\n")

    print(f"\nGrid sweep complete. Results CSV: {csv_path}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="FFD Hyperparameter Tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["bayesian", "wandb", "grid"],
        default="bayesian",
        help="Tuning strategy (default: bayesian via Optuna TPE)",
    )
    parser.add_argument(
        "--base_config",
        default=BASE_CONFIG,
        help="Path to base FFD config JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb in local runs"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Preview without executing"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Bayesian trials (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Optuna sampler seed (default: 42)"
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=0,
        help="Max runs for wandb agent (0 = unlimited)",
    )
    parser.add_argument(
        "--timeout", type=int, default=7200, help="Per-run timeout in seconds"
    )
    parser.add_argument(
        "--results_dir", default="results/ffd_sweep", help="Output directory"
    )
    parser.add_argument(
        "--keep_configs", action="store_true", help="Keep temp config files"
    )

    # Bayesian search space overrides
    bayes_group = parser.add_argument_group("Bayesian search space overrides")
    bayes_group.add_argument("--ffd_alpha_min", type=float, default=None)
    bayes_group.add_argument("--ffd_alpha_max", type=float, default=None)
    bayes_group.add_argument("--ffd_warmup_rounds_min", type=int, default=None)
    bayes_group.add_argument("--ffd_warmup_rounds_max", type=int, default=None)
    bayes_group.add_argument("--ffd_lambda_var_min", type=float, default=None)
    bayes_group.add_argument("--ffd_lambda_var_max", type=float, default=None)
    bayes_group.add_argument("--ffd_lambda_cov_min", type=float, default=None)
    bayes_group.add_argument("--ffd_lambda_cov_max", type=float, default=None)

    # Grid overrides
    grid_group = parser.add_argument_group(
        "Grid search overrides (space-separated values)"
    )
    grid_group.add_argument("--ffd_alpha", nargs="+", type=float, default=None)
    grid_group.add_argument("--ffd_warmup_rounds", nargs="+", type=int, default=None)
    grid_group.add_argument("--ffd_lambda_var", nargs="+", type=float, default=None)
    grid_group.add_argument("--ffd_lambda_cov", nargs="+", type=float, default=None)
    grid_group.add_argument("--ffd_proj_dim", nargs="+", type=int, default=None)

    args = parser.parse_args()

    if args.mode == "bayesian":
        space = copy.deepcopy(BAYESIAN_SPACE)
        # Apply CLI overrides to search bounds
        for param in [
            "ffd_alpha",
            "ffd_warmup_rounds",
            "ffd_lambda_var",
            "ffd_lambda_cov",
        ]:
            lo = getattr(args, f"{param}_min", None)
            hi = getattr(args, f"{param}_max", None)
            if lo is not None and param in space:
                space[param]["low"] = lo
            if hi is not None and param in space:
                space[param]["high"] = hi
        run_bayesian(args, space)

    elif args.mode == "wandb":
        run_wandb_sweep(args)

    else:  # grid
        grid = copy.deepcopy(GRID_SPACE)
        for key in [
            "ffd_alpha",
            "ffd_warmup_rounds",
            "ffd_lambda_var",
            "ffd_lambda_cov",
            "ffd_proj_dim",
        ]:
            cli_val = getattr(args, key, None)
            if cli_val is not None:
                grid[key] = cli_val
        run_grid(args, grid)


if __name__ == "__main__":
    main()
