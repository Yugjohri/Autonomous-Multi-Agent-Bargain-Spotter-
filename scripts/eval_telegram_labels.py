"""
A separate check on products seen in Telegram deal posts (data/labels.csv from
scripts/build_labels.py). These labels are DEAL prices, below the normal selling price the
models estimate, so this is not a test score: a good model should predict ABOVE them, and
the median predicted/label ratio shows by how much. It also shows how the models do on the
cheap, non-electronics items that live deals are mostly about.

    uv run python scripts/eval_telegram_labels.py [--labels data/labels.csv] [--label median_price|p75_price]

Only rows with a real Amazon or Flipkart product id (agents.normalize.is_product_id) are
used, and products that are in the train split are skipped. Free models only (no API calls):
category median, TF-IDF + Ridge, embedding kNN and the INR neural network.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from inr_common import DATA, ROOT, embeddings, load_splits, seed_everything

from agents.evaluator_inr import breakdown, metrics, with_bands
from agents.normalize import is_product_id
from agents.taxonomy import taxonomy_for
from agents.titles import normalise_title


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(ROOT / "data" / "labels.csv"))
    parser.add_argument("--label", default="median_price", choices=["median_price", "p75_price"])
    args = parser.parse_args()
    seed_everything(42)

    from run_baselines import category_median, knn, tfidf_ridge

    from agents.inr_features import embed

    labels = pd.read_csv(args.labels)
    labels = labels[labels["canonical_id"].map(lambda c: is_product_id(str(c)))].copy()
    splits = load_splits()
    train = splits["train"]
    train_ids = set(train["product_key"]) | {i for a in train["alt_ids"].fillna("") for i in a.split(";") if i}
    labels = labels[~labels["canonical_id"].isin(train_ids)]
    labels = labels[labels["title"].fillna("").str.len() > 0]
    labels = labels[~labels["title"].map(normalise_title).isin(set(train["norm_title"]))]
    df = pd.DataFrame({
        "product_key": labels["canonical_id"], "text": labels["title"].fillna(""),
        "price_inr": labels[args.label].astype(float), "observations": labels["observations"],
        "category": [taxonomy_for(t, c) for t, c in zip(labels["title"].fillna(""), labels["category"])],
    })
    df = df[df["price_inr"] > 0].reset_index(drop=True)
    print(f"{len(df)} Telegram products with a real product id, not in the train split")

    frames = {"val": splits["val"], "telegram": df}
    preds = {"category_median": category_median(train, {"telegram": df})["telegram"]}
    ridge, _ = tfidf_ridge(train, frames)
    preds["tfidf_ridge"] = ridge["telegram"]
    vectors = embed(df["text"].tolist())
    preds["knn"] = knn(embeddings("train", train), train["price_inr"].to_numpy(), vectors)
    model_path = ROOT / "models" / "nn_inr.pth"
    if model_path.exists():
        from agents.deep_neural_network import InrPriceModel

        preds["nn"] = InrPriceModel(str(model_path)).predict(df["text"].tolist(), df["category"].tolist())

    report = {"label": args.label, "n": len(df), "models": {}, "by_band": {}, "by_category": {}}
    banded = with_bands(df)
    for name, pred in preds.items():
        m = metrics(df["price_inr"], pred)
        report["models"][name] = m
        banded["pred"] = pred
        report["by_band"][name] = breakdown(banded, "price_band").to_dict("records")
        report["by_category"][name] = breakdown(banded, "category").to_dict("records")
        print(f"{name:16s} MdAPE {100 * m['mdape']:5.1f}%  within20 {100 * m['within_20']:5.1f}%  "
              f"median predicted/deal price {m['median_ratio']:.2f}")
    report["share_under_1k"] = float((df["price_inr"] < 1000).mean())
    report["categories"] = df["category"].value_counts().to_dict()
    (DATA / "telegram_check.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
