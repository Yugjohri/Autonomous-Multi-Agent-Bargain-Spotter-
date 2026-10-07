"""
Fit the INR ensemble weights on the validation split and score them on the test split.

    uv run python scripts/fit_ensemble.py [--step 0.05] [--tolerance 0.002]

Signals are the ones the app blends in agents.price_signal.value_inr: frontier (GPT with
RAG), market (median of close products_inr neighbours, missing when there are too few),
neural_network (scripts/train_nn_inr.py) and mrp (MRP x mrp_factor, only when an MRP exists).
The blend is simulated with value_inr itself, so missing signals are renormalised exactly
as in the app. Weights are non-negative, sum to 1 and are searched on a grid over the
validation items that have a Frontier answer (scripts/eval_frontier.py). A signal is
dropped when removing it does not raise the validation error by more than --tolerance
(mean |log error|). The result goes to data/processed/ensemble_fit.json; copy the weights
into settings.yaml (ensemble.inr).
"""

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

from inr_common import DATA, load_predictions, load_splits

from agents.config import get, load_settings
from agents.evaluator_inr import metrics
from agents.price_signal import value_inr

SIGNALS = ("frontier", "market", "neural_network", "specialist", "mrp")
COLUMNS = {"frontier": "frontier", "market": "market", "neural_network": "nn"}


FRONTIER_COLUMN = "frontier"
SUFFIX = ""


def frame(split: str) -> pd.DataFrame:
    df = load_splits()[split].set_index("product_key").join(load_predictions(split))
    # The Frontier answers of the model being fitted (e.g. "frontier@gpt-6-luna").
    df["frontier"] = df[FRONTIER_COLUMN]
    if "specialist" not in df.columns:
        df["specialist"] = np.nan
    return df[df["frontier"].notna()]


def blend(df: pd.DataFrame, weights, mrp_factor: float):
    """Estimates and confidence labels from value_inr, item by item."""
    estimates, confidence = [], []
    for row in df.itertuples():
        mrp = row.mrp_inr if row.mrp_inr == row.mrp_inr else None
        has_market = row.market == row.market
        # Valued at its own price: the estimate does not depend on the price, but the MRP
        # signal only counts when the MRP is above it, as for a live deal.
        v = value_inr(
            price=row.price_inr, mrp=mrp, llm_estimate=row.frontier, weights={s: float(weights.get(s, 0.0)) for s in SIGNALS},
            mrp_factor=mrp_factor, nn_estimate=row.nn, specialist_estimate=row.specialist, similars=_fake_market(row.market) if has_market else None,
        )
        estimates.append(v.estimate if v.signals else np.nan)
        confidence.append(v.confidence)
    return np.array(estimates, dtype=float), confidence


def _fake_market(median: float):
    """Three identical close neighbours make value_inr see exactly this market median."""
    from agents.inr_store import Similar

    return [Similar(document="", price=median, distance=0.0) for _ in range(3)]


def grid(names, step: float):
    units = int(round(1 / step))
    for combo in itertools.product(range(units + 1), repeat=len(names)):
        if sum(combo) == units:
            yield {n: c / units for n, c in zip(names, combo)}


def error(df, weights, mrp_factor) -> float:
    estimates, _ = blend(df, weights, mrp_factor)
    return metrics(df["price_inr"], estimates)["mean_abs_log"]


def best_weights(df, names, step, mrp_factor):
    best = None
    for weights in grid(names, step):
        score = error(df, weights, mrp_factor)
        if best is None or score < best[0] - 1e-9:
            best = (score, weights)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument("--frontier", default="frontier",
                        help="prediction column for the Frontier signal, e.g. frontier@gpt-6-luna")
    args = parser.parse_args()
    global FRONTIER_COLUMN
    FRONTIER_COLUMN = args.frontier
    global SUFFIX
    SUFFIX = "" if args.frontier == "frontier" else "@" + args.frontier.split("@")[-1]

    settings = load_settings()
    mrp_factor = float(get(settings, "ensemble.inr.mrp_factor", 0.75))
    val, test = frame("val"), frame("test")
    print(f"Fitting on {len(val)} validation items, scoring on {len(test)} test items")

    names = list(SIGNALS)
    score, weights = best_weights(val, names, args.step, mrp_factor)
    print(f"All signals: val |log err| {score:.4f} with {weights}")
    subsets = {}
    changed = True
    while changed and len(names) > 1:
        changed = False
        for name in [n for n in names if n != "frontier"]:
            reduced = [n for n in names if n != name]
            s, w = best_weights(val, reduced, args.step, mrp_factor)
            subsets[f"without {name}"] = s
            print(f"  without {name:15s}: {s:.4f}")
            if s <= score + args.tolerance:
                print(f"  -> drop {name}: it does not improve validation error")
                names, score, weights = reduced, s, w
                changed = True
                break
    weights = {n: round(weights.get(n, 0.0), 2) for n in SIGNALS}
    print(f"Chosen: {weights} (val |log err| {score:.4f})")

    results = {"weights": weights, "val_mean_abs_log": score, "subsets": subsets, "val_n": len(val),
               "test_n": len(test), "mrp_factor": mrp_factor, "test": {}, "val": {}}
    current = {k: v for k, v in (get(settings, "ensemble.inr", {}) or {}).items() if k in SIGNALS}
    candidates = {
        "frontier only": {"frontier": 1.0},
        "current settings": current,
        "fitted ensemble": weights,
    }
    for split, df in (("val", val), ("test", test)):
        for label, w in candidates.items():
            estimates, confidence = blend(df, w, mrp_factor)
            m = metrics(df["price_inr"], estimates)
            by_conf = {}
            for level in ("high", "medium", "low"):
                mask = np.array(confidence) == level
                if mask.any():
                    by_conf[level] = {"share": float(mask.mean()), **metrics(df["price_inr"][mask], estimates[mask])}
            results[split][label] = {"weights": w, "metrics": m, "by_confidence": by_conf}
            print(f"{split:4s} {label:17s} MdAPE {100 * m['mdape']:5.1f}%  within20 {100 * m['within_20']:5.1f}%  "
                  f"|log err| {m['mean_abs_log']:.3f}  "
                  + "  ".join(f"{k} {100 * v['share']:.0f}% (MdAPE {100 * v['mdape']:.0f}%)" for k, v in by_conf.items()))
            if label == "fitted ensemble":
                key = df.index.to_series()
                pd.DataFrame({"product_key": key.values, "pred": estimates, "confidence": confidence}).to_parquet(
                    DATA / "predictions" / f"ensemble{SUFFIX}__{split}.parquet", index=False)
    (DATA / f"ensemble_fit{SUFFIX}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
