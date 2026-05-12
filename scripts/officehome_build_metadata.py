"""Build OfficeHome metadata.csv mirroring the PACS layout.

Scans resources/office_home_v1.0/office_home_dg/<domain>/<train|val>/<class>/*
and emits a CSV with columns: split, domain_remapped, domain, category, y, path.

One domain is held out entirely as `val`, one as `test`. Remaining two domains
are pooled and split 90/5/5 into train/id_val/id_test (matching PACS).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DOMAINS = ["art", "clipart", "product", "real_world"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan(root: Path):
    rows = []
    for domain in DOMAINS:
        for src_split in ("train", "val"):
            split_dir = root / "office_home_dg" / domain / src_split
            if not split_dir.is_dir():
                raise FileNotFoundError(split_dir)
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for img in sorted(cls_dir.iterdir()):
                    if img.suffix.lower() not in IMG_EXTS:
                        continue
                    rows.append(
                        {
                            "domain": domain,
                            "category": cls_dir.name,
                            "src_split": src_split,
                            "path": str(img.relative_to(root)),
                        }
                    )
    return pd.DataFrame(rows)


def verify_images(df: pd.DataFrame, root: Path) -> None:
    bad = []
    for p in df["path"]:
        try:
            with Image.open(root / p) as im:
                im.verify()
        except Exception as e:
            bad.append((p, repr(e)))
    if bad:
        raise RuntimeError(f"{len(bad)} unreadable images, first: {bad[0]}")


def assign_splits(
    df: pd.DataFrame,
    val_domain: str,
    test_domain: str,
    seed: int,
    train_frac: float = 0.9,
    id_val_frac: float = 0.05,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""

    df.loc[df["domain"] == val_domain, "split"] = "val"
    df.loc[df["domain"] == test_domain, "split"] = "test"

    train_pool = df["split"] == ""
    pool_idx = np.where(train_pool)[0]
    perm = rng.permutation(pool_idx)
    n = len(perm)
    n_train = int(round(n * train_frac))
    n_id_val = int(round(n * id_val_frac))
    df.iloc[perm[:n_train], df.columns.get_loc("split")] = "train"
    df.iloc[perm[n_train : n_train + n_id_val], df.columns.get_loc("split")] = "id_val"
    df.iloc[perm[n_train + n_id_val :], df.columns.get_loc("split")] = "id_test"
    return df


def assign_codes(
    df: pd.DataFrame, val_domain: str, test_domain: str
) -> pd.DataFrame:
    others = [d for d in DOMAINS if d not in (val_domain, test_domain)]
    domain_remap = {test_domain: 0, val_domain: 1, others[0]: 2, others[1]: 3}
    classes = sorted(df["category"].unique())
    y_map = {c: i for i, c in enumerate(classes)}
    df = df.copy()
    df["domain_remapped"] = df["domain"].map(domain_remap)
    df["y"] = df["category"].map(y_map)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("resources/office_home_v1.0"),
    )
    parser.add_argument("--val-domain", default="art")
    parser.add_argument("--test-domain", default="real_world")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    if args.val_domain not in DOMAINS or args.test_domain not in DOMAINS:
        raise SystemExit(f"--val-domain/--test-domain must be in {DOMAINS}")
    if args.val_domain == args.test_domain:
        raise SystemExit("--val-domain and --test-domain must differ")

    root = args.root.resolve()
    df = scan(root)
    if args.verify_images:
        verify_images(df, root)
    df = assign_codes(df, args.val_domain, args.test_domain)
    df = assign_splits(df, args.val_domain, args.test_domain, args.seed)

    df = df[["split", "domain_remapped", "domain", "category", "y", "path"]]
    df = df.reset_index(drop=True)

    out = root / "metadata.csv"
    df.to_csv(out)
    print(f"wrote {out} ({len(df)} rows)")
    print(df.groupby(["split", "domain"])["path"].count())

    release = root / "RELEASE_v1.0.txt"
    release.write_text(
        f"OfficeHome metadata generated locally.\n"
        f"val_domain={args.val_domain} test_domain={args.test_domain} seed={args.seed}\n"
    )
    print(f"wrote {release}")


if __name__ == "__main__":
    main()
