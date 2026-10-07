"""
products_inr: a Chroma collection of Indian products with rupee prices, used as RAG
context for INR valuation. It bootstraps itself: every priced deal the scanner sees is
added with its price, MRP, store and time, so price context grows with every scan.
populate_vectorstore.py --currency INR seeds it from memory, observations or local CSVs.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

COLLECTION = "products_inr"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class InrItem:
    document: str
    price: float
    canonical_id: str = ""
    mrp: Optional[float] = None
    store: str = "other"
    category: str = "Other"
    seen_at: Optional[str] = None
    source: str = ""

    @property
    def id(self) -> str:
        """One row per product and price, so repeated sightings at the same price are upserts."""
        key = self.canonical_id or hashlib.sha1(self.document.encode("utf-8")).hexdigest()[:16]
        return f"{key}@{round(self.price)}"

    def metadata(self) -> dict:
        meta = {
            "price": float(self.price),
            "store": self.store,
            "category": self.category or "Other",
            "canonical_id": self.canonical_id,
            "seen_at": self.seen_at or datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "currency": "INR",
        }
        if self.mrp:
            meta["mrp"] = float(self.mrp)
        return meta


@dataclass
class Similar:
    document: str
    price: float
    distance: float
    canonical_id: str = ""
    mrp: Optional[float] = None
    store: str = "other"
    seen_at: Optional[str] = None


_encoder = None


def get_encoder():
    """Load the sentence encoder once per process (from the local HF cache when offline)."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder


class InrProductStore:
    def __init__(self, client=None, path: str = "products_vectorstore", encoder=None):
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=path)
        self.collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = get_encoder()
        return self._encoder

    def count(self) -> int:
        return self.collection.count()

    def add(self, items: Iterable[InrItem], batch: int = 500) -> int:
        items = [i for i in items if i.document and i.price and i.price > 0]
        unique = {i.id: i for i in items}
        items = list(unique.values())
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            vectors = self.encoder.encode([i.document for i in chunk], show_progress_bar=False)
            self.collection.upsert(
                ids=[i.id for i in chunk],
                documents=[i.document for i in chunk],
                embeddings=vectors.astype(float).tolist(),
                metadatas=[i.metadata() for i in chunk],
            )
        return len(items)

    def similar(self, text: str, n: int = 8, exclude_canonical: Optional[str] = None) -> List[Similar]:
        total = self.count()
        if not total or not text:
            return []
        vector = self.encoder.encode([text])
        result = self.collection.query(
            query_embeddings=vector.astype(float).tolist(), n_results=min(total, n + 5)
        )
        found = []
        for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            if exclude_canonical and meta.get("canonical_id") == exclude_canonical:
                continue
            found.append(
                Similar(
                    document=doc,
                    price=float(meta["price"]),
                    distance=float(distance),
                    canonical_id=meta.get("canonical_id", ""),
                    mrp=meta.get("mrp"),
                    store=meta.get("store", "other"),
                    seen_at=meta.get("seen_at"),
                )
            )
        return found[:n]


def offline_hf() -> None:
    """Make sentence-transformers use the local cache only (fixture mode, no network)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
