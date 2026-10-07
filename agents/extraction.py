"""
Cheap-first extraction: turn raw posts into priced, normalized, deduplicated candidates
using regex and link normalization only. The LLM later sees at most a few dozen of
these, and only has to resolve what regex could not (payable price vs. conditional
offers, a clean product description, category and brand).
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents.categorize import guess_brand, guess_category
from agents.normalize import (
    RECOGNISED_STORES,
    NormalizedLink,
    RedirectResolver,
    SeenIndex,
    dedupe_by_canonical,
    is_fresh,
    normalize_link,
    title_from_url,
)
from agents.sources.base import RawDeal

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
EMOJI_RE = re.compile(r"[^\w\s&+.,'()/\-:%₹@|!?]", re.UNICODE)


@dataclass
class Candidate:
    raw: RawDeal
    link: NormalizedLink
    price: float
    mrp: Optional[float]
    title: str
    category: str
    brand: Optional[str]
    seen_in: List[str] = field(default_factory=list)
    trust: float = 1.0

    @property
    def canonical_id(self) -> str:
        return self.link.canonical_id

    @property
    def store(self) -> str:
        return self.link.store

    @property
    def posted_at(self) -> Optional[datetime]:
        return self.raw.posted_at

    @property
    def mrp_discount_pct(self) -> Optional[float]:
        if self.mrp and self.mrp > self.price:
            return round(100 * (self.mrp - self.price) / self.mrp, 1)
        return None

    def age_minutes(self, now: Optional[datetime] = None) -> Optional[int]:
        if self.posted_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        posted = self.posted_at if self.posted_at.tzinfo else self.posted_at.replace(tzinfo=timezone.utc)
        return max(0, int((now - posted).total_seconds() // 60))

    def clean_text(self, limit: int = 400) -> str:
        text = URL_RE.sub("", self.raw.text)
        text = EMOJI_RE.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()[:limit]

    def to_inr_item(self):
        """This candidate as a products_inr row (self-bootstrapping INR price context)."""
        from agents.inr_store import InrItem

        return InrItem(
            document=f"{self.title}\n{self.clean_text(300)}".strip(),
            price=self.price,
            canonical_id=self.canonical_id,
            mrp=self.mrp,
            store=self.store,
            category=self.category,
            seen_at=self.posted_at.isoformat() if self.posted_at else None,
            source=self.seen_in[0] if self.seen_in else "",
        )


STOPWORDS = {
    "the", "and", "for", "with", "pack", "set", "combo", "deal", "loot", "offer", "new", "buy", "online",
    "india", "black", "white", "blue", "red", "grey", "green", "men", "women", "kids", "free", "off",
}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if w not in STOPWORDS}


def link_matches_title(title: str, resolved_url: str) -> bool:
    """
    False when the post title and the product slug of the resolved link share no word,
    e.g. a "Titan Talk Smartwatch" post whose short link leads to a wall charger.
    Links without a readable slug (bare /dp/ASIN) cannot be checked and pass.
    """
    slug = title_from_url(resolved_url)
    if not slug or not title:
        return True
    # Only titles that name something checkable (a known brand or a model number) are compared;
    # marketing titles like "Unmissable deal on your new travel companion" have nothing to match.
    if not (guess_brand(title) or re.search(r"\b[a-z]*\d+[a-z0-9]*\b", title.lower())):
        return True
    return bool(_words(title) & _words(slug))


def to_candidate(raw: RawDeal, resolver: Optional[RedirectResolver]) -> Optional[Candidate]:
    """Normalize one priceable post. Returns None if it has no usable link or price."""
    if not raw.priceable or raw.price_hint is None or not raw.url:
        return None
    link = normalize_link(raw.url, resolver)
    if not link_matches_title(raw.title, link.resolved):
        logger.info(f"Dropping '{raw.title}': its link leads to a different product ({title_from_url(link.resolved)})")
        raw.priceable, raw.drop_reason = False, "link_mismatch"
        return None
    if link.store == "other" and raw.store != "other":
        # Unresolvable link (e.g. robots.txt forbids it): trust the store named in the post.
        link.store = raw.store
    title = raw.title or title_from_url(link.resolved) or ""
    text = f"{title} {raw.text}"
    return Candidate(
        raw=raw,
        link=link,
        price=float(raw.price_hint),
        mrp=raw.mrp_hint,
        title=title,
        category=guess_category(text),
        brand=guess_brand(text),
        seen_in=[raw.source],
        trust=raw.trust,
    )


@dataclass
class FilterReport:
    raw: int = 0
    unpriceable: int = 0
    stale: int = 0
    unknown_store: int = 0
    duplicates: int = 0
    seen: int = 0
    kept: int = 0

    def summary(self) -> str:
        return (
            f"{self.raw} posts -> {self.kept} candidates "
            f"(unpriceable {self.unpriceable}, stale {self.stale}, unknown store {self.unknown_store}, "
            f"duplicates {self.duplicates}, already seen {self.seen})"
        )


def _merge(kept: Candidate, dup: Candidate) -> None:
    for source in dup.seen_in:
        if source not in kept.seen_in:
            kept.seen_in.append(source)
    kept.trust = max(kept.trust, dup.trust)
    if kept.mrp is None and dup.mrp:
        kept.mrp = dup.mrp
    if len(dup.title) > len(kept.title):
        kept.title = dup.title


def build_candidates(
    raws: List[RawDeal],
    resolver: Optional[RedirectResolver],
    seen: SeenIndex,
    freshness_hours: float,
    currency: str = "INR",
    now: Optional[datetime] = None,
) -> Tuple[List[Candidate], FilterReport]:
    """Pre-filter: has a price, has a recognised store, is fresh, and was not seen before."""
    report = FilterReport(raw=len(raws))
    candidates: List[Candidate] = []
    for raw in raws:
        if raw.currency != currency:
            continue
        if not is_fresh(raw.posted_at, freshness_hours, now):
            report.stale += 1
            continue
        candidate = to_candidate(raw, resolver)
        if candidate is None:
            report.unpriceable += 1
            continue
        if candidate.store not in RECOGNISED_STORES:
            report.unknown_store += 1
            continue
        candidates.append(candidate)

    before = len(candidates)
    candidates = dedupe_by_canonical(candidates, key=lambda c: c.canonical_id, merge=_merge)
    report.duplicates = before - len(candidates)

    fresh = [c for c in candidates if seen.is_new(c.canonical_id, c.price)]
    report.seen = len(candidates) - len(fresh)
    report.kept = len(fresh)
    return fresh, report


def rank_candidates(candidates: List[Candidate], limit: int) -> List[Candidate]:
    """Order by evidence of a good deal: several channels posting it, trust, MRP discount, recency."""

    def score(c: Candidate) -> float:
        age = c.age_minutes()
        recency = 1.0 if age is None else max(0.0, 1 - age / 360)
        return len(c.seen_in) * 2 + c.trust + (c.mrp_discount_pct or 0) / 25 + recency

    return sorted(candidates, key=score, reverse=True)[:limit]


class SeenStore:
    """
    SeenIndex persisted to data/state/seen.json, so candidates already shown to the LLM
    are not sent again every five minutes. Entries expire after `ttl_days`.
    """

    def __init__(self, path: Path = Path("data/state/seen.json"), ttl_days: float = 7):
        self.path = Path(path)
        self.ttl = ttl_days * 86400
        self.entries: Dict[str, Tuple[float, float]] = {}
        if self.path.exists():
            try:
                self.entries = {k: tuple(v) for k, v in json.loads(self.path.read_text(encoding="utf-8")).items()}
            except (OSError, ValueError):
                self.entries = {}
        cutoff = time.time() - self.ttl
        self.entries = {k: v for k, v in self.entries.items() if v[1] >= cutoff}

    def index(self, extra: Optional[SeenIndex] = None) -> SeenIndex:
        index = SeenIndex({k: v[0] for k, v in self.entries.items()})
        for key, price in (extra.prices if extra else {}).items():
            index.add(key, price)
        return index

    def mark(self, candidates: List[Candidate]) -> None:
        now = time.time()
        for c in candidates:
            old = self.entries.get(c.canonical_id)
            self.entries[c.canonical_id] = (min(c.price, old[0]) if old else c.price, now)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries), encoding="utf-8")
        tmp.replace(self.path)
