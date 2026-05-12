"""Build DomainNet metadata.csv (and optional metadata_debug.csv) for FedDG_Benchmark.

Layout assumption (post-extraction under --data-root):
    <domain>/<class>/<file>.jpg                  for all 6 domains

Source-of-truth per domain for train/test split + class label:
    infograph/quickdraw/real/sketch  -> use BU's NF=2 txt files (path, label)
    clipart                          -> scan dir + seeded stratified 70/30 split
                                        (BU's cleaned clipart txt uses hash-flat paths
                                         that do not exist in any extant zip)
    painting                         -> entire domain goes to one bucket
                                        (val under default L2DO), so no split needed

Default L2DO = (test=real, val=painting). Other 4 domains form the train pool;
each contributes 95% txt-train -> train, 5% txt-train -> id_val, all txt-test -> id_test.

Output: same column schema as PACS/OfficeHome metadata.csv:
    ,split,domain_remapped,domain,category,y,path
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
NF2_DOMAINS = ["infograph", "quickdraw", "real", "sketch"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
CLIPART_TRAIN_FRAC = 0.7  # match BU's ~70/30 ratio on other domains


def parse_nf2_txt(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            p, y = ln.split()
            rows.append((p, int(y)))
    return pd.DataFrame(rows, columns=["path", "y_official"])


def load_nf2_domain(domain: str, data_root: Path) -> pd.DataFrame:
    train = parse_nf2_txt(data_root / f"{domain}_train.txt")
    train["src_split"] = "train"
    test = parse_nf2_txt(data_root / f"{domain}_test.txt")
    test["src_split"] = "test"
    df = pd.concat([train, test], ignore_index=True)
    df["domain"] = domain
    df["category"] = df["path"].str.split("/").str[1]
    return df


def scan_clipart(data_root: Path, seed: int) -> pd.DataFrame:
    domain_dir = data_root / "clipart"
    if not domain_dir.is_dir():
        raise FileNotFoundError(domain_dir)
    rows = []
    for cls_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            rows.append(
                {
                    "domain": "clipart",
                    "category": cls_dir.name,
                    "path": str(img.relative_to(data_root)),
                }
            )
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    df["src_split"] = ""
    for cls, grp in df.groupby("category"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n_train = int(round(len(idx) * CLIPART_TRAIN_FRAC))
        df.loc[idx[:n_train], "src_split"] = "train"
        df.loc[idx[n_train:], "src_split"] = "test"
    return df


def scan_painting(data_root: Path) -> pd.DataFrame:
    """Painting goes entirely to one bucket under default L2DO; no split needed."""
    domain_dir = data_root / "painting"
    if not domain_dir.is_dir():
        raise FileNotFoundError(domain_dir)
    rows = []
    for cls_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            rows.append(
                {
                    "domain": "painting",
                    "category": cls_dir.name,
                    "path": str(img.relative_to(data_root)),
                    "src_split": "train",
                }
            )
    return pd.DataFrame(rows)


def build_class_map(nf2_dfs: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Derive class_name -> y from NF=2 files; verify cross-file consistency."""
    name_to_y = {}
    for d, df in nf2_dfs.items():
        for category, y in df[["category", "y_official"]].drop_duplicates().values:
            y = int(y)
            if category in name_to_y and name_to_y[category] != y:
                raise ValueError(
                    f"class {category} has conflicting labels across NF=2 files: "
                    f"{name_to_y[category]} vs {y} (in {d})"
                )
            name_to_y[category] = y
    if len(name_to_y) != 345:
        raise ValueError(f"expected 345 classes, got {len(name_to_y)}")
    if sorted(name_to_y.values()) != list(range(345)):
        raise ValueError("NF=2 label range is not 0..344 contiguous")
    return name_to_y


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
    df: pd.DataFrame,
    val_domain: str,
    test_domain: str,
    class_map: dict[str, int],
) -> pd.DataFrame:
    """domain_remapped: test=0, val=1, train_pool=2.. (PACS convention)."""
    train_pool = [d for d in DOMAINS if d not in (val_domain, test_domain)]
    remap = {test_domain: 0, val_domain: 1}
    for i, d in enumerate(train_pool):
        remap[d] = 2 + i
    df = df.copy()
    df["domain_remapped"] = df["domain"].map(remap)
    df["y"] = df["category"].map(class_map)
    if df["y"].isna().any():
        missing = df.loc[df["y"].isna(), "category"].unique()
        raise ValueError(f"unmapped classes: {missing}")
    df["y"] = df["y"].astype(int)
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
    parser.add_argument("--val-domain", default="painting")
    parser.add_argument("--test-domain", default="real")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument(
        "--debug-frac",
        type=float,
        default=0.05,
        help="fraction kept in metadata_debug.csv (default 5%)",
    )
    parser.add_argument(
        "--skip-debug",
        action="store_true",
        help="do not emit metadata_debug.csv",
    )
    args = parser.parse_args()

    if args.val_domain not in DOMAINS or args.test_domain not in DOMAINS:
        raise SystemExit(f"--val-domain/--test-domain must be in {DOMAINS}")
    if args.val_domain == args.test_domain:
        raise SystemExit("--val-domain and --test-domain must differ")

    data_root = args.data_root.resolve()

    print(f"[1/5] loading NF=2 txt files for {NF2_DOMAINS}")
    nf2_dfs = {d: load_nf2_domain(d, data_root) for d in NF2_DOMAINS}
    for d, df in nf2_dfs.items():
        print(f"      {d}: {len(df)} rows ({(df['src_split']=='train').sum()} train / {(df['src_split']=='test').sum()} test)")

    print("[2/5] building class_name -> y from NF=2 label columns")
    class_map = build_class_map(nf2_dfs)
    print(f"      {len(class_map)} classes, y in [0, 344]")
    alpha = sorted(class_map.keys())
    n_match = sum(1 for c, i in zip(alpha, range(345)) if class_map[c] == i)
    print(f"      alphabetical-vs-canonical-label match: {n_match}/345 classes")

    print("[3/5] scanning clipart (synthetic stratified 70/30 split)")
    clipart_df = scan_clipart(data_root, args.seed)
    print(f"      clipart: {len(clipart_df)} rows ({(clipart_df['src_split']=='train').sum()} train / {(clipart_df['src_split']=='test').sum()} test)")
    print("[3/5] scanning painting (all -> val under default L2DO)")
    painting_df = scan_painting(data_root)
    print(f"      painting: {len(painting_df)} rows")

    nf2_combined = pd.concat([df.drop(columns=["y_official"]) for df in nf2_dfs.values()], ignore_index=True)
    full = pd.concat([clipart_df, painting_df, nf2_combined], ignore_index=True)

    print(f"[4/5] assigning splits (val={args.val_domain}, test={args.test_domain}, seed={args.seed})")
    full = assign_codes(full, args.val_domain, args.test_domain, class_map)
    full = assign_splits(full, args.val_domain, args.test_domain, args.seed)

    out_path = data_root / "metadata.csv"
    write_metadata(full, out_path)
    print(f"      wrote {out_path} ({len(full)} rows)")
    print(full.groupby(["split", "domain"])["path"].count())

    if not args.skip_debug:
        debug = make_debug_subset(full, args.seed, args.debug_frac)
        debug_path = data_root / "metadata_debug.csv"
        write_metadata(debug, debug_path)
        print(f"      wrote {debug_path} ({len(debug)} rows, ~{args.debug_frac:.0%} of full)")
        print(debug.groupby(["split", "domain"])["path"].count())

    release = data_root / "RELEASE_v1.0.txt"
    release.write_text(
        f"DomainNet metadata generated locally.\n"
        f"val_domain={args.val_domain} test_domain={args.test_domain} "
        f"seed={args.seed} id_val_frac=0.05 clipart_train_frac={CLIPART_TRAIN_FRAC}\n"
    )
    print(f"[5/5] wrote {release}")


if __name__ == "__main__":
    main()
