"""
AggregatorSource reads an HTML listing page of an Indian deal community, using CSS
selectors from sources.yaml, so new sites can be added without code.

Before a site is enabled, check its robots.txt and terms of use and record what you
found in docs/sources.md. The source refuses to run unless the config says
`terms_checked: true`, and every request still goes through the robots.txt check.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from agents.http import HttpClient
from agents.money import parse_amount
from agents.sources.base import DealSource, RawDeal
from agents.sources.telegram_parser import parse_prices, store_hint

logger = logging.getLogger(__name__)


class TermsNotChecked(Exception):
    pass


class AggregatorSource(DealSource):
    """
    selectors: item (required), title, price, mrp, link, store, time (datetime attribute
    or ISO text), votes. Missing selectors are simply skipped.
    """

    def __init__(self, name: str, url: str, http: HttpClient, selectors: Dict[str, str], terms_checked: bool = False):
        super().__init__(f"site:{name}", http)
        self.url = url
        self.selectors = selectors or {}
        self.terms_checked = terms_checked

    def _text(self, node, key: str) -> Optional[str]:
        selector = self.selectors.get(key)
        if not selector:
            return None
        found = node.select_one(selector)
        return found.get_text(" ", strip=True) if found else None

    def parse(self, html: str) -> List[RawDeal]:
        soup = BeautifulSoup(html, "html.parser")
        deals = []
        for node in soup.select(self.selectors["item"]):
            try:
                title = self._text(node, "title") or ""
                link_node = node.select_one(self.selectors.get("link", "a[href]"))
                link = urljoin(self.url, link_node["href"]) if link_node and link_node.get("href") else ""
                text = node.get_text(" ", strip=True)
                price = parse_amount(self._text(node, "price")) if self.selectors.get("price") else None
                mrp = parse_amount(self._text(node, "mrp")) if self.selectors.get("mrp") else None
                if price is None:
                    price, mrp_guess, _ = parse_prices(text)
                    mrp = mrp or mrp_guess
                posted_at = None
                time_node = node.select_one(self.selectors["time"]) if self.selectors.get("time") else None
                if time_node is not None:
                    stamp = time_node.get("datetime") or time_node.get_text(strip=True)
                    try:
                        posted_at = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    except ValueError:
                        posted_at = None
                votes = self._text(node, "votes")
                deals.append(
                    RawDeal(
                        title=title[:120],
                        text=text[:1000],
                        url=link,
                        store=store_hint(self._text(node, "store") or text),
                        price_hint=price,
                        mrp_hint=mrp if mrp and price and mrp > price else None,
                        posted_at=posted_at,
                        source=self.name,
                        links=[link] if link else [],
                        external_id=f"{self.name}/{link or title}",
                        priceable=price is not None and bool(link),
                        drop_reason=None if price is not None else "no_price",
                        # Community votes, when shown, nudge trust up a little.
                        trust=1.0 + min(int(parse_amount(votes) or 0), 100) / 200 if votes else 1.0,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - per item isolation
                logger.warning(f"{self.name}: skipped an item ({exc})")
        return deals

    def _fetch(self, limit: int) -> List[RawDeal]:
        if not self.terms_checked:
            raise TermsNotChecked(
                f"{self.name} is enabled but terms_checked is not true in sources.yaml; "
                "check the site's terms of use and docs/sources.md first"
            )
        response = self.http.get(self.url)
        response.raise_for_status()
        return self.parse(response.text)[:limit]
