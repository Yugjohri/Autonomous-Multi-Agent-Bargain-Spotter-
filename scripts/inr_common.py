"""
Shared helpers for the INR training scripts: loading the splits, cached sentence
embeddings, and saving or loading model predictions under data/processed/predictions/.
"""

import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
PREDICTIONS = DATA / "predictions"
SPLITS = ("train", "val", "test")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_splits(data: Path = DATA) -> Dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(data / f"{name}.parquet") for name in SPLITS}


def embeddings(split: str, df: pd.DataFrame, data: Path = DATA) -> np.ndarray:
    """MiniLM embeddings of the text column, cached next to the split and checked by row keys."""
    from agents.inr_features import embed

    path, keys_path = data / f"emb_{split}.npy", data / f"emb_{split}.keys.txt"
    keys = "\n".join(df["product_key"])
    if path.exists() and keys_path.exists() and keys_path.read_text(encoding="utf-8") == keys:
        return np.load(path)
    import torch

    vectors = embed(df["text"].tolist(), device="cuda" if torch.cuda.is_available() else None)
    np.save(path, vectors)
    keys_path.write_text(keys, encoding="utf-8")
    return vectors


def save_predictions(model: str, split: str, df: pd.DataFrame, pred) -> Path:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"product_key": df["product_key"].to_numpy(), "pred": np.asarray(pred, dtype=float)})
    path = PREDICTIONS / f"{model}__{split}.parquet"
    out.to_parquet(path, index=False)
    return path


def load_predictions(split: str) -> pd.DataFrame:
    """All saved predictions for a split, one column per model, indexed by product_key."""
    frames = {}
    for path in sorted(PREDICTIONS.glob(f"*__{split}.parquet")):
        model = path.stem.split("__")[0]
        frames[model] = pd.read_parquet(path).drop_duplicates("product_key").set_index("product_key")["pred"]
    return pd.DataFrame(frames)
