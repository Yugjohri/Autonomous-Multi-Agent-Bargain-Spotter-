"""
Append-only log of every parsed post, for future price model training.

Each row is one post (identified by obs_id), so re-reading the same Telegram message
or feed item on a later scan, or rerunning a backfill, never adds duplicate rows.
The file is never rewritten, only appended to.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Set

from agents.categorize import guess_brand, guess_category
from agents.normalize import NormalizedLink
from agents.sources.base import RawDeal

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/observations.jsonl")


def obs_id_for(raw: RawDeal) -> str:
    if raw.external_id:
        return raw.external_id
    digest = hashlib.sha1(f"{raw.url}|{raw.price_hint}|{raw.title}".encode("utf-8")).hexdigest()[:16]
    return f"{raw.source}:{digest}"


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def observation_row(raw: RawDeal, link: Optional[NormalizedLink] = None, seen_at: Optional[datetime] = None) -> dict:
    text_for_guess = f"{raw.title} {raw.text}"
    store = link.store if link and link.store != "other" else raw.store
    return {
        "obs_id": obs_id_for(raw),
        "canonical_id": link.canonical_id if link else None,
        "title": raw.title,
        "description": raw.text[:1000],
        "price_inr": raw.price_hint if raw.currency == "INR" else None,
        "mrp_inr": raw.mrp_hint if raw.currency == "INR" else None,
        "highest_price_inr": raw.highest_price_hint,
        "store": store,
        "source": raw.source,
        "category": guess_category(text_for_guess),
        "brand": guess_brand(text_for_guess),
        "url": link.url if link else raw.url,
        "posted_at": _iso(raw.posted_at),
        "seen_at": _iso(seen_at or datetime.now(timezone.utc)),
        "is_deal": bool(raw.priceable),
        "drop_reason": raw.drop_reason,
    }


class ObservationLog:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._ids: Optional[Set[str]] = None

    def _load_ids(self) -> Set[str]:
        ids: Set[str] = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as file:
                for line in file:
                    try:
                        ids.add(json.loads(line)["obs_id"])
                    except (ValueError, KeyError):
                        continue
        return ids

    def __contains__(self, obs_id: str) -> bool:
        with self._lock:
            if self._ids is None:
                self._ids = self._load_ids()
            return obs_id in self._ids

    def append(self, rows: Iterable[dict]) -> int:
        """Append rows whose obs_id is new. Returns how many were written."""
        with self._lock:
            if self._ids is None:
                self._ids = self._load_ids()
            fresh = []
            for row in rows:
                if row["obs_id"] in self._ids:
                    continue
                self._ids.add(row["obs_id"])
                fresh.append(row)
            if fresh:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as file:
                    for row in fresh:
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
            return len(fresh)
