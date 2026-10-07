"""
Simple baselines for INR price estimation, on the validation and test splits.

    uv run python scripts/run_baselines.py [--k 5]

    category_median  median train price of the item's category
    tfidf_ridge      TF-IDF (words and character n-grams) + Ridge on log price; alpha picked on val
    knn              median price of the k nearest train items by all-MiniLM-L6-v2 cosine similarity

Predictions go to data/processed/predictions/<model>__<split>.parquet for the report and
the ensemble fit. The Frontier baseline costs money and lives in scripts/eval_frontier.py.
"""

import argparse
import json
import sys

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from inr_common import DATA, embeddings, load_splits, save_predictions, seed_everything

from agents.evaluator_inr import metrics


def category_median(train, frames):
    medians = train.groupby("category")["price_inr"].median()
    overall = float(train["price_inr"].median())
    return {name: df["category"].map(medians).fillna(overall).to_numpy() for name, df in frames.items()}


def tfidf_ridge(train, frames, alphas=(0.03, 0.1, 0.3, 1.0, 3.0)):
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    x_train = hstack([words.fit_transform(train["text"]), chars.fit_transform(train["text"])]).tocsr()
    xs = {name: hstack([words.transform(df["text"]), chars.transform(df["text"])]).tocsr() for name, df in frames.items()}
    y = np.log(train["price_inr"].to_numpy())
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(x_train, y)
        score = metrics(frames["val"]["price_inr"], np.exp(model.predict(xs["val"])))["mean_abs_log"]
        print(f"  tfidf_ridge alpha={alpha}: val mean |log error| {score:.4f}")
        if best is None or score < best[0]:
            best = (score, alpha, model)
    _, alpha, model = best
    return {name: np.exp(model.predict(x)) for name, x in xs.items()}, {"alpha": alpha}


def knn(train_vectors, train_prices, vectors, k=5, batch=2048):
    out = np.empty(len(vectors))
    for start in range(0, len(vectors), batch):
        sims = vectors[start : start + batch] @ train_vectors.T
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        out[start : start + batch] = np.median(train_prices[top], axis=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="neighbours for the kNN baseline")
    args = parser.parse_args()
    seed_everything(42)

    splits = load_splits()
    train = splits["train"]
    frames = {"val": splits["val"], "test": splits["test"]}
    results, params = {}, {}

    preds = category_median(train, frames)
    results["category_median"] = preds

    print("TF-IDF + Ridge")
    preds, params["tfidf_ridge"] = tfidf_ridge(train, frames)
    results["tfidf_ridge"] = preds

    print("Embedding kNN")
    vectors = {name: embeddings(name, df) for name, df in splits.items()}
    prices = train["price_inr"].to_numpy()
    results["knn"] = {name: knn(vectors["train"], prices, vectors[name], args.k) for name in frames}
    params["knn"] = {"k": args.k}

    summary = {}
    for model, by_split in results.items():
        for split, pred in by_split.items():
            save_predictions(model, split, frames[split], pred)
            m = metrics(frames[split]["price_inr"], pred)
            summary[f"{model}/{split}"] = m
            print(f"{model:16s} {split:4s} MdAPE {100 * m['mdape']:5.1f}%  within20 {100 * m['within_20']:5.1f}%  "
                  f"R2log {m['r2_log']:.3f}  ratio {m['median_ratio']:.2f}")
    (DATA / "baselines.json").write_text(json.dumps({"params": params, "metrics": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
