from pathlib import Path

import pytest

from agents.cache import MemoryCache
from agents.sources import build_sources
from agents.sources.aggregator import AggregatorSource, TermsNotChecked
from agents.sources.rss import DealsMagnetSource, RSSSource
from agents.sources.store_api import AmazonCreatorsApiSource, StoreApiNotConfigured
from tests.fakes import FakeHttp, FakeResponse

FIXTURES = Path(__file__).parent / "fixtures"
FEED = (FIXTURES / "rss" / "dealsmagnet_like.xml").read_text(encoding="utf-8")
LISTING = (FIXTURES / "aggregator" / "listing.html").read_text(encoding="utf-8")
FEED_URL = "https://www.dealsmagnet.com/feed"


def test_dealsmagnet_items_parse_price_mrp_store_and_date():
    source = DealsMagnetSource(FakeHttp(pages={FEED_URL: FEED}))
    lamp, shampoo, sale = source.fetch(10)
    assert (lamp.price_hint, lamp.mrp_hint, lamp.store) == (699, 1999, "myntra")
    assert lamp.title == "Example Ceramic Table Lamp with Fabric Shade"
    assert lamp.url.startswith("https://www.dealsmagnet.com/deal/")
    assert lamp.posted_at.date().isoformat() == "2026-10-02"
    assert lamp.source == "rss:dealsmagnet" and lamp.external_id.startswith("rss:dealsmagnet/")
    assert (shampoo.price_hint, shampoo.mrp_hint, shampoo.store) == (129, None, "amazon_in")
    assert not sale.priceable and sale.drop_reason == "no_price"


class ConditionalHttp(FakeHttp):
    def __init__(self):
        super().__init__()
        self.headers_seen = []

    def get(self, url, headers=None, **kwargs):
        self.headers_seen.append(dict(headers or {}))
        if headers and headers.get("If-None-Match") == '"v1"':
            return FakeResponse(304)
        response = FakeResponse(200, text=FEED)
        response.headers["ETag"] = '"v1"'
        return response


def test_conditional_get_reuses_cached_feed():
    http = ConditionalHttp()
    source = RSSSource("dealsmagnet", FEED_URL, http, cache=MemoryCache())
    first = source.fetch(10)
    second = source.fetch(10)
    assert http.headers_seen[1] == {"If-None-Match": '"v1"'}
    assert [d.title for d in first] == [d.title for d in second]


SELECTORS = {"item": "div.deal-card", "title": "a.title", "link": "a.title", "price": ".price", "mrp": ".mrp",
             "store": ".store", "time": "time", "votes": ".votes"}


def test_aggregator_selectors():
    source = AggregatorSource("example", "https://deals.example.in/new", FakeHttp(), SELECTORS, terms_checked=True)
    earbuds, mixer, sale = source.parse(LISTING)
    assert (earbuds.price_hint, earbuds.mrp_hint, earbuds.store) == (1299, 4999, "flipkart")
    assert earbuds.url == "https://deals.example.in/deals/example-earbuds-123"
    assert earbuds.posted_at.isoformat() == "2026-10-02T09:15:00+05:30"
    assert earbuds.trust > 1.0
    assert (mixer.price_hint, mixer.store) == (2149, "amazon_in")
    assert not sale.priceable


def test_aggregator_refuses_without_terms_check():
    http = FakeHttp(pages={"https://deals.example.in/new": LISTING})
    source = AggregatorSource("example", "https://deals.example.in/new", http, SELECTORS, terms_checked=False)
    assert source.fetch(10) == []
    assert "TermsNotChecked" in source.last_error
    assert http.requests == [], "no request before the terms are confirmed"
    with pytest.raises(TermsNotChecked):
        source._fetch(10)


def test_store_api_stub():
    source = AmazonCreatorsApiSource()
    assert source.fetch(5) == []
    with pytest.raises(StoreApiNotConfigured):
        source.lookup("amazon_in:B0CHX1W1XY")


def test_registry_respects_enabled_flags():
    from agents.config import load_sources_config

    config = load_sources_config(Path("sources.yaml"), Path("does-not-exist.yaml"))
    sources = build_sources(config, FakeHttp())
    # The committed sources.yaml enables only DealLooto: no Telegram placeholder, no site whose
    # terms forbid bots (DealsMagnet, DesiDime) and no store API.
    assert [s.name for s in sources] == ["rss:deallooto"]
    next(f for f in config["rss"] if f["name"] == "dealsmagnet")["enabled"] = True
    config["aggregators"]["example"] = {"enabled": True, "url": "https://deals.example.in/new", "selectors": SELECTORS}
    names = [s.name for s in build_sources(config, FakeHttp())]
    assert names == ["rss:deallooto", "rss:dealsmagnet", "site:example"]


LOOT_FEED = (FIXTURES / "rss" / "deallooto_like.xml").read_text(encoding="utf-8")
LOOT_URL = "https://www.deallooto.com/rss.xml"


def test_deallooto_header_gives_price_mrp_and_store():
    from agents.sources.rss import DealLootoSource

    sneakers, polish, shirt, cashback, sale, kurta = DealLootoSource(FakeHttp(pages={LOOT_URL: LOOT_FEED})).fetch(10)
    assert sneakers.title == "Example Runner V2 Sneakers For Men"
    assert (sneakers.price_hint, sneakers.mrp_hint, sneakers.store) == (1281, 4999, "flipkart")
    assert sneakers.priceable and sneakers.url.startswith("https://www.deallooto.com/2026/10/")
    assert sneakers.posted_at.isoformat() == "2026-10-06T12:19:57+00:00"
    assert sneakers.source == "rss:deallooto" and sneakers.external_id.startswith("rss:deallooto/")
    assert (polish.price_hint, polish.mrp_hint, polish.store) == (174, 507, "amazon_in")
    # No MRP in the header: price only.
    assert (shirt.price_hint, shirt.mrp_hint, shirt.store, shirt.priceable) == (200, None, "myntra", True)
    assert kurta.title == "EXAMPLE COLLECTION Women Casual Kurta"


def test_deallooto_offers_and_sales_are_not_priceable():
    from agents.sources.rss import DealLootoSource

    deals = DealLootoSource(FakeHttp(pages={LOOT_URL: LOOT_FEED})).fetch(10)
    cashback, sale = deals[3], deals[4]
    assert not cashback.priceable and cashback.drop_reason == "offer_not_product"
    assert not sale.priceable and sale.drop_reason == "offer_not_product"


def test_deallooto_is_built_from_sources_yaml():
    from agents.sources.rss import DealLootoSource

    config = {"telegram": {"enabled": False}, "rss": [{"name": "deallooto", "url": LOOT_URL, "enabled": True}]}
    sources = build_sources(config, FakeHttp(pages={LOOT_URL: LOOT_FEED}))
    assert [type(s) for s in sources] == [DealLootoSource]
