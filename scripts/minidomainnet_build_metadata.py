"""Build miniDomainNet metadata.csv (and debug subset) from splits_mini txt files.

Uses the existing full-DomainNet image data under resources/domainnet_v1.0/.
miniDomainNet (Zhou et al. 2020) restricts to 126 classes across 4 domains:
clipart, painting, real, sketch. Image paths in splits_mini/ point at the same
files extracted from the full-DomainNet zips, so no new image data is needed.

Default L2DO: val=painting, test=real (analog of full-DomainNet default).
Train pool becomes {clipart, sketch} — natural 2-client FedDG topology.

Output schema mirrors PACS/OfficeHome/DomainNet metadata.csv:
    ,split,domain_remapped,domain,category,y,path

Output paths:
    resources/domainnet_v1.0/minidomainnet.csv
    resources/domainnet_v1.0/minidomainnet_debug.csv (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["clipart", "painting", "real", "sketch"]
N_CLASSES = 126


def parse_nf2_txt(path: Path, domain: str) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            p, y = ln.split()
            rows.append((p, int(y)))
    df = pd.DataFrame(rows, columns=["path", "y"])
    df["domain"] = domain
    df["category"] = df["path"].str.split("/").str[1]
    return df


def load_domain(domain: str, splits_dir: Path) -> pd.DataFrame:
    train = parse_nf2_txt(splits_dir / f"{domain}_train.txt", domain)
    train["src_split"] = "train"
    test = parse_nf2_txt(splits_dir / f"{domain}_test.txt", domain)
    test["src_split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def assign_splits(
    df: pd.DataFrame,
    val_domain: str,
    test_domain: str,
    seed: int,
    id_val_frac: float = 0.05,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    df["split"] = ""
    df.loc[df["domain"] == val_domain, "split"] = "val"
    df.loc[df["domain"] == test_domain, "split"] = "test"

    pool_mask = df["split"] == ""
    pool_train_mask = pool_mask & (df["src_split"] == "train")
    pool_test_mask = pool_mask & (df["src_split"] == "test")
    df.loc[pool_test_mask, "split"] = "id_test"

    train_idx = np.where(pool_train_mask)[0]
    rng.shuffle(train_idx)
    n_id_val = int(round(len(train_idx) * id_val_frac))
    df.iloc[train_idx[:n_id_val], df.columns.get_loc("split")] = "id_val"
    df.iloc[train_idx[n_id_val:], df.columns.get_loc("split")] = "train"
    return df


def assign_codes(
    df: pd.DataFrame, val_domain: str, test_domain: str
) -> pd.DataFrame:
    train_pool = [d for d in DOMAINS if d not in (val_domain, test_domain)]
    remap = {test_domain: 0, val_domain: 1}
    for i, d in enumerate(train_pool):
        remap[d] = 2 + i
    df = df.copy()
    df["domain_remapped"] = df["domain"].map(remap)
    return df


def write_metadata(df: pd.DataFrame, out_path: Path) -> None:
    df = df[["split", "domain_remapped", "domain", "category", "y", "path"]]
    df = df.reset_index(drop=True)
    df.to_csv(out_path)


def make_debug_subset(df: pd.DataFrame, seed: int, frac: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    parts = []
    for (split, domain), grp in df.groupby(["split", "domain"]):
        n = max(1, int(round(len(grp) * frac)))
        pick = rng.choice(grp.index.to_numpy(), size=min(n, len(grp)), replace=False)
        parts.append(df.loc[pick])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("resources/domainnet_v1.0"),
    )
    parser.add_argument(
        "--splits-dir-name",
        default="splits_mini",
        help="subdirectory under data-root containing the 8 splits_mini txt files",
    )
    parser.add_argument("--val-domain", default="painting")
    parser.add_argument("--test-domain", default="real")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--debug-frac", type=float, default=0.05)
    parser.add_argument("--skip-debug", action="store_true")
    args = parser.parse_args()

    if args.val_domain not in DOMAINS or args.test_domain not in DOMAINS:
        raise SystemExit(f"--val-domain/--test-domain must be in {DOMAINS}")
    if args.val_domain == args.test_domain:
        raise SystemExit("--val-domain and --test-domain must differ")

    data_root = args.data_root.resolve()
    splits_dir = data_root / args.splits_dir_name

    print(f"[1/4] loading splits_mini txt files from {splits_dir}")
    dfs = {d: load_domain(d, splits_dir) for d in DOMAINS}
    for d, df in dfs.items():
        n_tr = (df["src_split"] == "train").sum()
        n_te = (df["src_split"] == "test").sum()
        print(f"      {d}: {len(df)} rows ({n_tr} train / {n_te} test)")

    full = pd.concat(dfs.values(), ignore_index=True)
    n_classes = int(full["y"].max()) + 1
    if n_classes != N_CLASSES:
        print(f"WARN: expected {N_CLASSES} classes, observed max(y)+1={n_classes}")

    print(f"[2/4] assigning splits (val={args.val_domain}, test={args.test_domain}, seed={args.seed})")
    full = assign_codes(full, args.val_domain, args.test_domain)
    full = assign_splits(full, args.val_domain, args.test_domain, args.seed)

    out_path = data_root / "minidomainnet.csv"
    write_metadata(full, out_path)
    print(f"[3/4] wrote {out_path} ({len(full)} rows)")
    print(full.groupby(["split", "domain"])["path"].count())

    if not args.skip_debug:
        debug = make_debug_subset(full, args.seed, args.debug_frac)
        debug_path = data_root / "minidomainnet_debug.csv"
        write_metadata(debug, debug_path)
        print(f"      wrote {debug_path} ({len(debug)} rows, ~{args.debug_frac:.0%})")
        print(debug.groupby(["split", "domain"])["path"].count())

    print(f"[4/4] miniDomainNet ready. Use split_scheme='minidomainnet' in configs.")


if __name__ == "__main__":
    main()
