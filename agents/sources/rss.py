"""
RSSSource reads any RSS or Atom feed through the shared HTTP client (robots.txt,
User-Agent, timeouts, retries, per-host rate limit). Feeds are fetched with
conditional GETs: the last ETag / Last-Modified and body are kept in a disk cache,
so an unchanged feed costs one 304 response.

DealsMagnetSource adapts the DealsMagnet feed, whose items carry the offer price,
MRP and store in the description and only a "Last Updated on <date>" stamp.
"""

import calendar
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

import feedparser

from agents.cache import DiskCache, MemoryCache
from agents.http import HttpClient
from agents.sources.base import DealSource, RawDeal
from agents.sources.telegram_parser import extract_title, parse_prices, store_hint

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _entry_time(entry) -> Optional[datetime]:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
    return None


class RSSSource(DealSource):
    def __init__(
        self,
        name: str,
        urls: Union[str, List[str]],
        http: HttpClient,
        currency: str = "INR",
        store: str = "other",
        cache: Optional[DiskCache] = None,
    ):
        super().__init__(f"rss:{name}", http)
        self.feed_name = name
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.currency = currency
        self.store = store
        self.cache = cache if cache is not None else MemoryCache()

    def download(self, url: str) -> str:
        cached = self.cache.get(url) or {}
        headers = {}
        if cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        if cached.get("last_modified"):
            headers["If-Modified-Since"] = cached["last_modified"]
        response = self.http.get(url, headers=headers)
        if response.status_code == 304 and cached.get("body"):
            return cached["body"]
        response.raise_for_status()
        self.cache.set(
            url,
            {
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "body": response.text,
            },
        )
        return response.text

    def entry_to_raw(self, entry) -> Optional[RawDeal]:
        title = entry.get("title", "")
        summary = re.sub(r"<[^>]+>", " ", entry.get("summary", ""))
        text = f"{title}\n{summary}".strip()
        price, mrp, highest = parse_prices(text) if self.currency == "INR" else (None, None, None)
        link = entry.get("link", "")
        return RawDeal(
            title=title.strip()[:120] or extract_title(text),
            text=text,
            url=link,
            store=self.store if self.store != "other" else store_hint(text),
            price_hint=price,
            mrp_hint=mrp,
            posted_at=_entry_time(entry),
            source=self.name,
            links=[link] if link else [],
            external_id=f"{self.name}/{entry.get('id') or link}",
            highest_price_hint=highest,
            priceable=price is not None and bool(link),
            drop_reason=None if price is not None else "no_price",
            currency=self.currency,
        )

    def _fetch(self, limit: int) -> List[RawDeal]:
        deals: List[RawDeal] = []
        for url in self.urls:
            feed = feedparser.parse(self.download(url))
            for entry in feed.entries[:limit]:
                try:
                    raw = self.entry_to_raw(entry)
                except Exception as exc:  # noqa: BLE001 - one bad item must not drop the feed
                    logger.warning(f"{self.name}: skipped a malformed item ({exc})")
                    continue
                if raw:
                    deals.append(raw)
        return deals


LAST_UPDATED_RE = re.compile(r"Last Updated on (\d{1,2})(?:st|nd|rd|th)? (\w+),? (\d{4})", re.IGNORECASE)
OFFER_STORE_RE = re.compile(r"Offer Store:\s*([A-Za-z][\w .&-]{1,30}?)(?:\s+Last Updated|[.,]|$)", re.IGNORECASE)


class DealsMagnetSource(RSSSource):
    """
    https://www.dealsmagnet.com/feed. Item links point to the DealsMagnet deal page; the
    store link sits behind /buy, which robots.txt disallows, so the deal page is kept.
    """

    def __init__(self, http: HttpClient, url: str = "https://www.dealsmagnet.com/feed", cache: Optional[DiskCache] = None):
        super().__init__("dealsmagnet", url, http, currency="INR", cache=cache)

    def entry_to_raw(self, entry) -> Optional[RawDeal]:
        raw = super().entry_to_raw(entry)
        if raw is None:
            return None
        store = OFFER_STORE_RE.search(raw.text)
        tags = " ".join(t.get("term", "") for t in entry.get("tags", []) or [])
        raw.store = store_hint(store.group(1) if store else tags) if (store or tags) else raw.store
        if raw.posted_at is None:
            match = LAST_UPDATED_RE.search(raw.text)
            if match:
                day, month, year = match.groups()
                try:
                    date = datetime.strptime(f"{day} {month[:3]} {year}", "%d %b %Y").replace(tzinfo=IST)
                    # Day precision only: treat the item as updated at the end of that day.
                    raw.posted_at = min(date + timedelta(hours=23, minutes=59), datetime.now(IST))
                except ValueError:
                    pass
        return raw
