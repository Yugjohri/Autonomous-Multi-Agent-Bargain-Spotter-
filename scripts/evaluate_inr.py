"""
Score every saved model on the validation and test splits and draw the report plots.

    uv run python scripts/evaluate_inr.py [--plots docs/images]

Reads data/processed/predictions/ (written by run_baselines.py, eval_frontier.py,
train_nn_inr.py and fit_ensemble.py). Free models are scored on the whole test set; the
Frontier and the ensemble only have answers for a fixed sample, so every model is also
scored on that same sample. Writes markdown tables to data/processed/report_tables.md and
truth-vs-prediction scatter plots (log scale) to --plots.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from inr_common import DATA, ROOT, load_predictions, load_splits

from agents.evaluator_inr import BAND_ORDER, METRICS_HEADER, breakdown, format_metrics_row, metrics, scatter_png, with_bands

ORDER = ["category_median", "tfidf_ridge", "knn", "market", "nn", "frontier", "frontier_mrp", "ensemble"]
NAMES = {
    "category_median": "Category median", "tfidf_ridge": "TF-IDF + Ridge", "knn": "Embedding kNN (k=5)",
    "market": "Market signal (store neighbours)", "nn": "INR neural network", "frontier": "Frontier (GPT-5.1 + RAG)",
    "frontier_mrp": "Frontier, MRP in prompt", "ensemble": "Fitted ensemble",
}


def table(df, models) -> str:
    rows = [METRICS_HEADER]
    for model in models:
        if model in df.columns:
            sub = df[df[model].notna()]
            rows.append(format_metrics_row(NAMES.get(model, model), metrics(sub["price_inr"], sub[model])))
    return "\n".join(rows)


def grouped(df, models, column, order=None) -> str:
    """MdAPE (and n) per group, one column per model."""
    keys = order or list(df[column].value_counts().index)
    head = f"| {column} | n | " + " | ".join(NAMES.get(m, m) for m in models) + " |"
    rows = [head, "|---" * (len(models) + 2) + "|"]
    for key in keys:
        part = df[df[column] == key]
        if part.empty:
            continue
        cells = []
        for model in models:
            sub = part[part[model].notna()]
            m = metrics(sub["price_inr"], sub[model])
            cells.append(f"{100 * m['mdape']:.0f}%" if m.get("n") else "-")
        rows.append(f"| {key} | {len(part)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plots", default=str(ROOT / "docs" / "images"))
    args = parser.parse_args()
    plots = Path(args.plots)
    plots.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    out = []
    for split in ("val", "test"):
        df = with_bands(splits[split]).set_index("product_key").join(load_predictions(split))
        models = [m for m in ORDER if m in df.columns]
        free = [m for m in models if m not in ("frontier", "frontier_mrp", "ensemble")]
        sample = df[df["frontier"].notna()] if "frontier" in df.columns else df.iloc[:0]
        out.append(f"## {split}\n\n### All {len(df)} items (free models)\n\n{table(df, free)}\n")
        out.append(f"### Same {len(sample)} sampled items (all models)\n\n{table(sample, models)}\n")
        if split == "test":
            out.append("### Median APE by category, whole test set\n\n" + grouped(df, free, "category") + "\n")
            out.append("### Median APE by category, sampled items\n\n" + grouped(sample, models, "category") + "\n")
            out.append("### Median APE by price band, whole test set\n\n"
                       + grouped(df, free, "price_band", BAND_ORDER) + "\n")
            out.append("### Median APE by price band, sampled items\n\n"
                       + grouped(sample, models, "price_band", BAND_ORDER) + "\n")
            for model in models:
                part = df if model in free else sample
                part = part[part[model].notna()]
                scatter_png(part["price_inr"], part[model], plots / f"test_{model}.png",
                            f"{NAMES.get(model, model)}, test (n={len(part)})")
    (DATA / "report_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
