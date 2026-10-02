"""
The source abstraction. Every deal source returns RawDeal objects through
DealSource.fetch(), which isolates failures: a broken source logs a warning and
returns an empty list, so one dead source never kills a scan.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from agents.http import HttpClient

logger = logging.getLogger(__name__)


@dataclass
class RawDeal:
    """A deal as posted by a source, before normalization and valuation."""

    title: str
    text: str
    url: str
    store: str = "other"
    price_hint: Optional[float] = None
    mrp_hint: Optional[float] = None
    posted_at: Optional[datetime] = None
    source: str = ""
    links: List[str] = field(default_factory=list)
    # Stable id of the post within its source, e.g. "telegram:OMGDeals/102047".
    external_id: str = ""
    highest_price_hint: Optional[float] = None
    coupon_hint: Optional[str] = None
    # False for posts that cannot be priced (percentage-only, expired, no price).
    priceable: bool = True
    drop_reason: Optional[str] = None
    trust: float = 1.0
    currency: str = "INR"


class DealSource(ABC):
    """Base class for deal sources. Subclasses implement _fetch()."""

    name: str = "source"

    def __init__(self, name: str, http: Optional[HttpClient] = None, enabled: bool = True):
        self.name = name
        self.http = http
        self.enabled = enabled
        self.last_error: Optional[str] = None
        self.last_count = 0
        self.last_duration = 0.0

    @abstractmethod
    def _fetch(self, limit: int) -> List[RawDeal]:
        """Return up to `limit` recent deals. May raise; fetch() isolates errors."""

    def fetch(self, limit: int = 50) -> List[RawDeal]:
        start = time.monotonic()
        self.last_error = None
        try:
            deals = self._fetch(limit)
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Source {self.name} failed and was skipped: {self.last_error}")
            deals = []
        self.last_duration = time.monotonic() - start
        self.last_count = len(deals)
        logger.info(f"Source {self.name} returned {len(deals)} posts in {self.last_duration:.1f}s")
        return deals

    def close(self) -> None:
        """Release connections or background tasks, if any."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
