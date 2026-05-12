"""Build VLCS metadata.csv (and debug subset).

Walks resources/vlcs_v1.0/VLCS/<domain>/<src_split>/<class>/<file>.jpg
where domain ∈ {CALTECH, LABELME, PASCAL, SUN}, src_split ∈ {train, crossval, test},
and class ∈ {0..4} (bird, car, chair, dog, person — labels intrinsic to dir name).

The dataset ships its own train/crossval/test partitions, so no synthetic carving:
    train pool train/    -> split=train
    train pool crossval/ -> split=id_val
    train pool test/     -> split=id_test
    val_domain (all)     -> split=val
    test_domain (all)    -> split=test

Default L2DO: val=SUN, test=PASCAL.

Output schema mirrors PACS/OfficeHome metadata.csv:
    ,split,domain_remapped,domain,category,y,path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["CALTECH", "LABELME", "PASCAL", "SUN"]
SRC_SPLITS = ["train", "crossval", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
N_CLASSES = 5


def scan_domain(domain: str, data_root: Path) -> pd.DataFrame:
    domain_dir = data_root / "VLCS" / domain
    if not domain_dir.is_dir():
        raise FileNotFoundError(domain_dir)
    rows = []
    for src in SRC_SPLITS:
        src_dir = domain_dir / src
        if not src_dir.is_dir():
            continue
        for cls_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
            try:
                y = int(cls_dir.name)
            except ValueError:
                continue
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                rows.append(
                    {
                        "domain": domain,
                        "category": cls_dir.name,
                        "y": y,
                        "src_split": src,
                        "path": str(img.relative_to(data_root)),
                    }
                )
    return pd.DataFrame(rows)


def assign_splits(
    df: pd.DataFrame, val_domain: str, test_domain: str
) -> pd.DataFrame:
    df = df.copy()
    df["split"] = ""
    df.loc[df["domain"] == val_domain, "split"] = "val"
    df.loc[df["domain"] == test_domain, "split"] = "test"

    pool_mask = df["split"] == ""
    df.loc[pool_mask & (df["src_split"] == "train"), "split"] = "train"
    df.loc[pool_mask & (df["src_split"] == "crossval"), "split"] = "id_val"
    df.loc[pool_mask & (df["src_split"] == "test"), "split"] = "id_test"
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
        default=Path("resources/vlcs_v1.0"),
    )
    parser.add_argument("--val-domain", default="SUN")
    parser.add_argument("--test-domain", default="PASCAL")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--debug-frac", type=float, default=0.1)
    parser.add_argument("--skip-debug", action="store_true")
    args = parser.parse_args()

    if args.val_domain not in DOMAINS or args.test_domain not in DOMAINS:
        raise SystemExit(f"--val-domain/--test-domain must be in {DOMAINS}")
    if args.val_domain == args.test_domain:
        raise SystemExit("--val-domain and --test-domain must differ")

    data_root = args.data_root.resolve()

    print(f"[1/4] scanning VLCS at {data_root}")
    dfs = []
    for d in DOMAINS:
        sub = scan_domain(d, data_root)
        dfs.append(sub)
        per_src = sub.groupby("src_split")["path"].count().to_dict()
        print(f"      {d}: {len(sub)} rows ({per_src})")
    full = pd.concat(dfs, ignore_index=True)

    n_classes = int(full["y"].max()) + 1
    if n_classes != N_CLASSES:
        print(f"WARN: expected {N_CLASSES} classes, observed max(y)+1={n_classes}")

    print(f"[2/4] assigning splits (val={args.val_domain}, test={args.test_domain})")
    full = assign_codes(full, args.val_domain, args.test_domain)
    full = assign_splits(full, args.val_domain, args.test_domain)

    out_path = data_root / "metadata.csv"
    write_metadata(full, out_path)
    print(f"[3/4] wrote {out_path} ({len(full)} rows)")
    print(full.groupby(["split", "domain"])["path"].count())

    if not args.skip_debug:
        debug = make_debug_subset(full, args.seed, args.debug_frac)
        debug_path = data_root / "metadata_debug.csv"
        write_metadata(debug, debug_path)
        print(f"      wrote {debug_path} ({len(debug)} rows, ~{args.debug_frac:.0%})")
        print(debug.groupby(["split", "domain"])["path"].count())

    release = data_root / "RELEASE_v1.0.txt"
    release.write_text(
        f"VLCS metadata generated locally.\n"
        f"val_domain={args.val_domain} test_domain={args.test_domain}\n"
    )
    print(f"[4/4] wrote {release}")


if __name__ == "__main__":
    main()
