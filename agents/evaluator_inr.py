"""
Evaluation harness for INR price estimates. agents/evaluator.py stays as the USD notebook
harness; this one runs anywhere (tqdm.auto, matplotlib PNGs) and judges errors in percent,
since a Rs 500 miss is fine on a laptop and terrible on a phone case.

    metrics(truth, pred)        MAE in rupees, median APE, share within 10% and 20%, R² on log price,
                                and the median ratio pred/truth (above 1 means overestimates)
    breakdown(df, column)       the same metrics per category or per price band
    evaluate(predictor, items)  runs a predictor over rows (in threads), with coloured progress
    scatter_png(...)            truth vs prediction on log axes
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"green": GREEN, "orange": YELLOW, "red": RED}

PRICE_BANDS = [(0, 1_000, "under 1k"), (1_000, 10_000, "1k to 10k"), (10_000, 50_000, "10k to 50k"),
               (50_000, math.inf, "over 50k")]
BAND_ORDER = [name for _, _, name in PRICE_BANDS]


def price_band(price: float) -> str:
    for low, high, name in PRICE_BANDS:
        if low <= price < high:
            return name
    return BAND_ORDER[-1]


def color_for(truth: float, pred: float) -> str:
    """Green within 20%, orange within 40%, red beyond."""
    error = abs(pred - truth) / truth
    if error <= 0.2:
        return "green"
    if error <= 0.4:
        return "orange"
    return "red"


def metrics(truth: Sequence[float], pred: Sequence[float]) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(pred) & (pred > 0)
    truth, pred = truth[ok], pred[ok]
    if len(truth) == 0:
        return {"n": 0}
    ape = np.abs(pred - truth) / truth
    log_t, log_p = np.log(truth), np.log(pred)
    ss_res = float(np.sum((log_t - log_p) ** 2))
    ss_tot = float(np.sum((log_t - log_t.mean()) ** 2))
    return {
        "n": int(len(truth)),
        "mae_inr": float(np.mean(np.abs(pred - truth))),
        "mdape": float(np.median(ape)),
        "within_10": float(np.mean(ape <= 0.10)),
        "within_20": float(np.mean(ape <= 0.20)),
        "r2_log": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mean_abs_log": float(np.mean(np.abs(log_t - log_p))),
        "median_ratio": float(np.median(pred / truth)),
    }


def breakdown(df: pd.DataFrame, column: str, truth: str = "price_inr", pred: str = "pred",
              order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(column, observed=True):
        rows.append({column: key, **metrics(group[truth], group[pred])})
    out = pd.DataFrame(rows)
    if order is not None and len(out):
        out[column] = pd.Categorical(out[column], categories=list(order), ordered=True)
        out = out.sort_values(column)
    elif len(out):
        out = out.sort_values("n", ascending=False)
    return out.reset_index(drop=True)


def with_bands(df: pd.DataFrame, truth: str = "price_inr") -> pd.DataFrame:
    df = df.copy()
    df["price_band"] = df[truth].map(price_band)
    return df


def evaluate(predictor: Callable[[dict], Optional[float]], items: pd.DataFrame, workers: int = 1,
             verbose: bool = True) -> pd.Series:
    """Run the predictor on each row (as a dict) and return predictions aligned with items."""
    records = items.to_dict("records")

    def run(i: int):
        try:
            return predictor(records[i])
        except Exception:  # noqa: BLE001 - one failed item should not stop the run
            return None

    preds = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, value in tqdm(enumerate(pool.map(run, range(len(records)))), total=len(records), leave=False):
            preds[i] = value
            if verbose and value:
                truth = records[i]["price_inr"]
                print(f"{COLOR_MAP[color_for(truth, value)]}{100 * abs(value - truth) / truth:.0f}%{RESET} ", end="")
    if verbose:
        print()
    return pd.Series([np.nan if p is None else float(p) for p in preds], index=items.index, dtype=float)


def scatter_png(truth: Sequence[float], pred: Sequence[float], path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(pred) & (pred > 0)
    truth, pred = truth[ok], pred[ok]
    colors = {"green": "#2e9d5b", "orange": "#e69b22", "red": "#d1453b"}
    point_colors = [colors[color_for(t, p)] for t, p in zip(truth, pred)]
    low = max(10.0, float(min(truth.min(), pred.min())) * 0.8)
    high = float(max(truth.max(), pred.max())) * 1.25
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=120)
    ax.scatter(truth, pred, s=9, c=point_colors, alpha=0.7, linewidths=0)
    ax.plot([low, high], [low, high], color="#5b8dd6", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Actual price (Rs, log scale)")
    ax.set_ylabel("Predicted price (Rs, log scale)")
    m = metrics(truth, pred)
    ax.set_title(f"{title}\nMdAPE {100 * m['mdape']:.0f}%, within 20%: {100 * m['within_20']:.0f}%, "
                 f"R² log {m['r2_log']:.2f}", fontsize=10)
    ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_metrics_row(name: str, m: Dict[str, float]) -> str:
    """One markdown table row: model | n | MAE | MdAPE | within 10% | within 20% | R² log | median ratio."""
    if not m.get("n"):
        return f"| {name} | 0 | - | - | - | - | - | - |"
    return (
        f"| {name} | {m['n']} | {m['mae_inr']:,.0f} | {100 * m['mdape']:.1f}% | {100 * m['within_10']:.1f}% | "
        f"{100 * m['within_20']:.1f}% | {m['r2_log']:.3f} | {m['median_ratio']:.2f} |"
    )


METRICS_HEADER = (
    "| model | n | MAE (Rs) | median APE | within 10% | within 20% | R² (log) | median pred/actual |\n"
    "|---|---|---|---|---|---|---|---|"
)
