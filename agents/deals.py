from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from typing import Any, List, Dict, Literal, Optional, Self
from agents.categorize import CATEGORIES
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import logging
import requests
import time

logger = logging.getLogger(__name__)

# Defaults for the legacy US pipeline; sources.yaml (rss: legacy_us) overrides them.
feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]

# You could also add: "https://www.dealnews.com/c238/Automotive/?rss=1"
# "https://www.dealnews.com/c196/Home-Garden/?rss=1"


def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    """
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """

    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    TIMEOUT = 15

    def __init__(self, entry: Dict[str, str], session: Optional[requests.Session] = None):
        """
        Populate this instance based on the provided dict.
        The deal page is optional context: if it cannot be fetched or has no
        content section, the RSS summary is used instead of failing.
        """
        self.title = entry["title"]
        self.summary = extract(entry.get("summary", ""))
        self.url = entry["links"][0]["href"]
        self.details, self.features = self.summary, ""
        try:
            response = (session or requests).get(self.url, timeout=self.TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            section = soup.find("div", class_="content-section")
            if section is not None:
                content = section.get_text().replace("\nmore", "").replace("\n", " ")
                if "Features" in content:
                    self.details, self.features = content.split("Features", 1)
                else:
                    self.details = content
            else:
                logger.info(f"No content section on {self.url}; using the RSS summary")
        except requests.RequestException as exc:
            logger.warning(f"Could not fetch deal page {self.url} ({type(exc).__name__}); using the RSS summary")
        self.truncate()

    def truncate(self):
        """
        Limit the fields to a sensible length to avoid sending too much info to the model
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress: bool = False, urls: Optional[List[str]] = None, per_feed: int = 10) -> List[Self]:
        """
        Retrieve all deals from the legacy US RSS feeds (sources.yaml, rss: legacy_us).
        A failing feed or item is logged and skipped.
        """
        urls = urls or legacy_feed_urls()
        session = requests.Session()
        session.headers["User-Agent"] = "BargainSpotter/0.2"
        deals = []
        feed_iter = tqdm(urls) if show_progress else urls
        for feed_url in feed_iter:
            try:
                response = session.get(feed_url, timeout=cls.TIMEOUT)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
            except requests.RequestException as exc:
                logger.warning(f"Legacy feed {feed_url} failed ({type(exc).__name__}); skipping it")
                continue
            for entry in feed.entries[:per_feed]:
                try:
                    deals.append(cls(entry, session))
                except (KeyError, IndexError) as exc:
                    logger.warning(f"Skipping a malformed item in {feed_url} ({exc})")
                time.sleep(0.05)
        return deals


def legacy_feed_urls() -> List[str]:
    """The legacy_us feed list from sources.yaml, or the built-in DealNews defaults."""
    try:
        from agents.config import load_sources_config

        for feed in load_sources_config().get("rss", []) or []:
            if feed.get("name") == "legacy_us" and feed.get("url"):
                url = feed["url"]
                return [url] if isinstance(url, str) else list(url)
    except Exception as exc:  # noqa: BLE001 - fall back to the defaults
        logger.warning(f"Could not read legacy feeds from sources.yaml ({exc})")
    return list(feeds)


class Deal(BaseModel):
    """
    A deal with a summary description and the metadata needed to dedupe, value and display it.
    The price is what the buyer actually pays, in the deal's currency.
    """

    product_description: str
    price: float
    url: str
    currency: str = "INR"
    mrp: Optional[float] = None
    store: str = "other"
    source: str = ""
    posted_at: Optional[datetime] = None
    canonical_id: str = ""
    coupon_note: Optional[str] = None
    title: str = ""
    category: str = "Other"
    brand: Optional[str] = None
    # Every source that posted this deal; several channels posting it is a quality signal.
    seen_in: List[str] = Field(default_factory=list)
    # Highest past price, when a source publishes price history (e.g. "Highest Price: ₹X").
    highest_price: Optional[float] = None


class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    """

    deals: List[Deal]


CategoryName = Literal[tuple(CATEGORIES)]


class InrPick(BaseModel):
    """Structured output the INR scanner asks the LLM to fill for each chosen candidate."""

    candidate_id: int = Field(description="The [id] number of the chosen candidate, exactly as given")
    title: str = Field(description="Short product name with brand, model and variant, under 80 characters")
    product_description: str = Field(
        description="2 to 4 sentences about the product itself: what it is, key specs, model and variant. "
        "Do not describe the deal, discounts or coupons here."
    )
    price: float = Field(
        description="The amount in Indian rupees (INR) that any buyer actually pays for one unit. "
        "If the post says '₹X off' or 'save ₹X', that is a discount, not the price. "
        "Include coupons that anyone can apply, but do not subtract conditional offers such as "
        "specific bank cards, exchange bonuses or cashback."
    )
    coupon_note: Optional[str] = Field(
        description="Conditions and extra savings in a few words: coupon to apply, bank card offer, "
        "exchange or cashback, effective price after them if stated. Null if there are none."
    )
    category: CategoryName = Field(description="The best matching category")
    brand: Optional[str] = Field(description="Brand name, or null if unknown")


class InrSelection(BaseModel):
    picks: List[InrPick] = Field(
        description="The best deals among the candidates: clear single products with a clear payable price"
    )


class LegacyDealPick(BaseModel):
    """Structured output schema the legacy US scanner asks the LLM to fill."""

    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")

    def to_deal(self) -> Deal:
        return Deal(
            product_description=self.product_description,
            price=self.price,
            url=self.url,
            currency="USD",
            source="rss:dealnews",
        )


class LegacyDealSelection(BaseModel):
    deals: List[LegacyDealPick] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    """

    deal: Deal
    estimate: float
    discount: float
    currency: Optional[str] = None
    discount_pct: Optional[float] = None
    confidence: Optional[str] = None
    confidence_reason: Optional[str] = None
    found_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _currency_from_deal(self) -> "Opportunity":
        if self.currency is None:
            self.currency = self.deal.currency
        return self


LEGACY_CURRENCY = "USD"


def migrate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upgrade a memory.json record written before deals carried a currency.
    Records without a currency come from the US pipeline, so they default to USD.
    """
    record = dict(record)
    deal = dict(record.get("deal", {}))
    deal.setdefault("currency", record.get("currency") or LEGACY_CURRENCY)
    if "dealnews.com" in deal.get("url", ""):
        deal.setdefault("source", "rss:dealnews")
        deal.setdefault("store", "other")
    record["deal"] = deal
    record.setdefault("currency", deal["currency"])
    return record


def load_opportunities(records: List[Dict[str, Any]]) -> List[Opportunity]:
    """Build Opportunities from raw memory records, migrating old formats on the way."""
    return [Opportunity(**migrate_record(record)) for record in records]
