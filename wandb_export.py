#!/usr/bin/env python3
"""
wandb_export.py — Export Weights & Biases runs to CSV and/or JSON.

Designed for analysis workflows where you want a tidy table of run configs,
summary metrics, and optionally the full step-level history — without piping
the entire project into your working environment.

Typical usage:
    # All runs in a project, summary + config only, both CSV and JSON
    python wandb_export.py --entity my-team --project fcd

    # Restrict to a tag (e.g. the linear-probing experiment cluster)
    python wandb_export.py --entity my-team --project fcd --tags phase_a_probing

    # Restrict by run name (regex)
    python wandb_export.py --entity my-team --project fcd --name-regex "fcd_lambda_.*"

    # Include step-level history, downsampled to 500 points per run,
    # only for selected metric keys
    python wandb_export.py --entity my-team --project fcd \
        --history --history-samples 500 \
        --history-keys train/loss val/acc val/probe_acc_inv val/probe_acc_env

Authentication:
    Run `wandb login` once on this machine, or set the WANDB_API_KEY
    environment variable. This script never asks for or stores credentials.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import wandb
except ImportError:
    sys.exit("wandb is not installed. Install with: pip install wandb pandas")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wandb_export")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary so it serialises cleanly to CSV columns."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep).items())
        elif isinstance(v, (list, tuple)):
            # Encode lists as JSON strings so a CSV cell can hold them
            items.append((new_key, json.dumps(v, default=str)))
        else:
            items.append((new_key, v))
    return dict(items)


def matches_name(run_name: str, regex: str | None) -> bool:
    if regex is None:
        return True
    return re.search(regex, run_name) is not None


def build_filters(
    states: list[str] | None,
    tags: list[str] | None,
) -> dict[str, Any] | None:
    """Build a MongoDB-style filter dict for the W&B Public API."""
    f: dict[str, Any] = {}
    if states:
        f["state"] = {"$in": states}
    if tags:
        # Runs must contain ALL specified tags
        f["tags"] = {"$all": tags}
    return f or None


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------

def export_runs(
    entity: str,
    project: str,
    output_dir: Path,
    *,
    tags: list[str] | None = None,
    states: list[str] | None = None,
    name_regex: str | None = None,
    include_history: bool = False,
    history_samples: int = 500,
    history_keys: list[str] | None = None,
    formats: Iterable[str] = ("csv", "json"),
    output_stem: str = "runs",
) -> None:
    api = wandb.Api(timeout=60)
    path = f"{entity}/{project}"
    log.info("Querying %s ...", path)

    filters = build_filters(states, tags)
    runs = api.runs(path=path, filters=filters)

    rows: list[dict[str, Any]] = []      # flat rows for CSV
    nested: list[dict[str, Any]] = []    # nested records for JSON
    history_frames: list[pd.DataFrame] = []

    n_total = 0
    for i, run in enumerate(runs, start=1):
        n_total = i
        if not matches_name(run.name, name_regex):
            continue

        # Drop W&B internals (keys beginning with "_")
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        summary = {k: v for k, v in dict(run.summary).items() if not k.startswith("_")}

        meta = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "runtime_sec": run.summary.get("_runtime"),
            "tags": ",".join(run.tags) if run.tags else "",
            "url": run.url,
            "group": run.group,
            "job_type": run.job_type,
        }

        # Flat row — every config/summary field becomes its own column
        flat_row: dict[str, Any] = dict(meta)
        flat_row.update({f"config.{k}": v for k, v in flatten(config).items()})
        flat_row.update({f"summary.{k}": v for k, v in flatten(summary).items()})
        rows.append(flat_row)

        # Nested record — preserves original structure for JSON consumers
        nested.append({
            "metadata": meta,
            "config": config,
            "summary": summary,
        })

        if include_history:
            try:
                hist = run.history(
                    samples=history_samples,
                    keys=history_keys,
                    pandas=True,
                )
                if hist is not None and not hist.empty:
                    hist["run_id"] = run.id
                    hist["run_name"] = run.name
                    history_frames.append(hist)
            except Exception as e:
                log.warning("Could not fetch history for %s: %s", run.name, e)

        if i % 25 == 0:
            log.info("  processed %d runs", i)

    log.info("Iterated %d runs, kept %d after filtering.", n_total, len(rows))
    output_dir.mkdir(parents=True, exist_ok=True)

    if "csv" in formats:
        csv_path = output_dir / f"{output_stem}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info("Wrote %s", csv_path)

    if "json" in formats:
        json_path = output_dir / f"{output_stem}.json"
        json_path.write_text(json.dumps(nested, indent=2, default=str))
        log.info("Wrote %s", json_path)

    if include_history and history_frames:
        hist_df = pd.concat(history_frames, ignore_index=True)
        if "csv" in formats:
            hist_csv = output_dir / f"{output_stem}_history.csv"
            hist_df.to_csv(hist_csv, index=False)
            log.info("Wrote %s (%d rows)", hist_csv, len(hist_df))
        if "json" in formats:
            hist_json = output_dir / f"{output_stem}_history.json"
            hist_df.to_json(hist_json, orient="records", indent=2)
            log.info("Wrote %s", hist_json)
    elif include_history:
        log.info("History requested but no history rows were collected.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export W&B runs (config + summary + optional history) to CSV/JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--entity", required=True,
                   help="W&B entity (your username or team name).")
    p.add_argument("--project", required=True,
                   help="W&B project name.")
    p.add_argument("--output-dir", default="./wandb_export", type=Path,
                   help="Directory where exported files will be written.")
    p.add_argument("--output-stem", default="runs",
                   help="Filename stem (default: 'runs' -> runs.csv, runs.json).")
    p.add_argument("--tags", nargs="+", default=None,
                   help="Only include runs that have ALL of these tags.")
    p.add_argument("--states", nargs="+", default=None,
                   choices=["finished", "running", "crashed", "failed", "killed"],
                   help="Only include runs in these states.")
    p.add_argument("--name-regex", default=None,
                   help="Only include runs whose name matches this regular expression.")
    p.add_argument("--history", action="store_true",
                   help="Also export step-level history (downsampled).")
    p.add_argument("--history-samples", type=int, default=500,
                   help="Approximate number of samples per run when --history is set.")
    p.add_argument("--history-keys", nargs="+", default=None,
                   help="Restrict history to these metric keys (saves bandwidth).")
    p.add_argument("--format", choices=["csv", "json", "both"], default="both",
                   help="Which output format(s) to produce.")
    args = p.parse_args()

    formats = ("csv", "json") if args.format == "both" else (args.format,)

    export_runs(
        entity=args.entity,
        project=args.project,
        output_dir=args.output_dir,
        tags=args.tags,
        states=args.states,
        name_regex=args.name_regex,
        include_history=args.history,
        history_samples=args.history_samples,
        history_keys=args.history_keys,
        formats=formats,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()