"""
Split data/processed/products_inr.parquet into train, validation and test sets.

    uv run python scripts/make_splits.py [--data data/processed] [--val-fraction 0.1] [--seed 42]

Test is the whole amazon_2026 dataset (scraped June 2026) and is never used for training,
tuning or picking weights. Amazon 2025 and 2026 share products, so every product whose
ASIN or normalised title appears in the test set is removed from train and validation
first. Validation is 10% of the rest, stratified by category. Writes train.parquet,
val.parquet, test.parquet and splits.json (counts and the leakage report).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_DATASET = "amazon_2026"


def ids_of(df: pd.DataFrame) -> set:
    ids = set(df["product_key"])
    for alt in df["alt_ids"].fillna(""):
        ids.update(i for i in alt.split(";") if i)
    return {i for i in ids if not i.startswith("title:")}


def leaks(pool: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Flags per pool row: shares a product id with test, shares a normalised title with test."""
    test_ids, test_titles = ids_of(test), set(test["norm_title"])
    by_id = [
        bool({key, *[i for i in alt.split(";") if i]} & test_ids)
        for key, alt in zip(pool["product_key"], pool["alt_ids"].fillna(""))
    ]
    return pd.DataFrame({"by_id": by_id, "by_title": pool["norm_title"].isin(test_titles).to_list()}, index=pool.index)


def split(df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42):
    test = df[df["source_dataset"] == TEST_DATASET].reset_index(drop=True)
    pool = df[df["source_dataset"] != TEST_DATASET]
    flags = leaks(pool, test)
    leaked = flags["by_id"] | flags["by_title"]
    report = {
        "removed_by_id": int(flags["by_id"].sum()),
        "removed_by_title_only": int((flags["by_title"] & ~flags["by_id"]).sum()),
        "removed_total": int(leaked.sum()),
        "removed_by_dataset": pool[leaked]["source_dataset"].value_counts().to_dict(),
    }
    pool = pool[~leaked]
    # Categories with a single row cannot be stratified; they go to train.
    counts = pool["category"].value_counts()
    rare = pool["category"].map(counts) < 2
    train, val = train_test_split(
        pool[~rare], test_size=val_fraction, random_state=seed, stratify=pool.loc[~rare, "category"]
    )
    train = pd.concat([train, pool[rare]]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val.reset_index(drop=True), test, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = Path(args.data)
    df = pd.read_parquet(data / "products_inr.parquet")
    train, val, test, report = split(df, args.val_fraction, args.seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_parquet(data / f"{name}.parquet", index=False)
    report["rows"] = {"train": len(train), "val": len(val), "test": len(test)}
    report["by_dataset"] = {
        name: part["source_dataset"].value_counts().to_dict() for name, part in (("train", train), ("val", val), ("test", test))
    }
    report["by_category"] = {
        name: part["category"].value_counts().to_dict() for name, part in (("train", train), ("val", val), ("test", test))
    }
    (data / "splits.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Leakage: removed {report['removed_total']} train/val rows that are test products "
          f"({report['removed_by_id']} by ASIN, {report['removed_by_title_only']} more by normalised title)")
    print(f"Rows: {report['rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
