"""
Input features for the INR pricing models, shared by training (scripts/) and inference
(agents/deep_neural_network.py), so both build exactly the same vectors.

    hashing     5000 binary word features (the same HashingVectorizer as the USD model)
    embeddings  384-dim all-MiniLM-L6-v2 sentence embedding, normalised
    category    one-hot over the training taxonomy (agents/taxonomy.py)
"""

from typing import Dict, Sequence

import numpy as np

from agents.taxonomy import TAXONOMY

HASH_FEATURES = 5000
EMBED_DIM = 384


def hashing_vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(n_features=HASH_FEATURES, stop_words="english", binary=True)


def embed(texts: Sequence[str], batch_size: int = 256, device=None) -> np.ndarray:
    from agents.inr_store import get_encoder

    encoder = get_encoder()
    if device is not None:
        encoder = encoder.to(device)
    return encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                          show_progress_bar=len(texts) > 2000).astype(np.float32)


def category_onehot(categories: Sequence[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(TAXONOMY)}
    out = np.zeros((len(categories), len(TAXONOMY)), dtype=np.float32)
    for row, category in enumerate(categories):
        out[row, index.get(category, index["other"])] = 1.0
    return out


def input_size(spec: Dict[str, bool]) -> int:
    return (HASH_FEATURES if spec.get("hashing", True) else 0) + (EMBED_DIM if spec.get("embeddings") else 0) + (
        len(TAXONOMY) if spec.get("category") else 0
    )


def build_features(texts: Sequence[str], categories: Sequence[str], spec: Dict[str, bool],
                   embeddings: np.ndarray = None) -> np.ndarray:
    """Concatenate the feature groups named in spec. Precomputed embeddings may be passed in."""
    parts = []
    if spec.get("hashing", True):
        parts.append(hashing_vectorizer().transform(list(texts)).toarray().astype(np.float32))
    if spec.get("embeddings"):
        parts.append(embeddings if embeddings is not None else embed(texts))
    if spec.get("category"):
        parts.append(category_onehot(categories))
    return np.hstack(parts)
