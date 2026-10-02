from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from typing import Any, List, Dict, Optional, Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time

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

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        """
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.url = entry["links"][0]["href"]
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, "html.parser")
        content = soup.find("div", class_="content-section").get_text()
        content = content.replace("\nmore", "").replace("\n", " ")
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
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
    def fetch(cls, show_progress: bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                deals.append(cls(entry))
                time.sleep(0.05)
        return deals


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
