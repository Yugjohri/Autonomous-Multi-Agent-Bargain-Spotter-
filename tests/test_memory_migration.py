import json
from datetime import datetime, timezone

from agents.deals import Deal, Opportunity, load_opportunities, migrate_record

# Exact shape of records written by the original US pipeline (no currency, no metadata).
LEGACY_RECORD = {
    "deal": {
        "product_description": "The Samsung Galaxy Watch Ultra is a premium 47mm LTE Titanium smartwatch.",
        "price": 350.0,
        "url": "https://www.dealnews.com/Samsung-Galaxy-Watch-Ultra/21663266.html?iref=rss-c142",
    },
    "estimate": 773.81,
    "discount": 423.81,
}


def test_legacy_record_defaults_to_usd():
    [opp] = load_opportunities([LEGACY_RECORD])
    assert opp.currency == "USD"
    assert opp.deal.currency == "USD"
    assert opp.deal.source == "rss:dealnews"
    assert opp.deal.price == 350.0
    assert opp.discount_pct is None


def test_migrate_record_does_not_mutate_input():
    original = json.loads(json.dumps(LEGACY_RECORD))
    migrate_record(LEGACY_RECORD)
    assert LEGACY_RECORD == original


def test_new_deal_defaults_to_inr_and_opportunity_inherits_currency():
    deal = Deal(product_description="boAt Airdopes 141", price=1099, url="https://www.amazon.in/dp/B09N3ZNHTY")
    opp = Opportunity(deal=deal, estimate=1799, discount=700)
    assert deal.currency == "INR"
    assert opp.currency == "INR"


def test_inr_record_round_trips_through_json():
    deal = Deal(
        product_description="boAt Airdopes 141",
        price=1099,
        url="https://www.amazon.in/dp/B09N3ZNHTY",
        mrp=4490,
        store="amazon_in",
        source="telegram:DCLOOTS",
        posted_at=datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc),
        canonical_id="amazon_in:B09N3ZNHTY",
        seen_in=["telegram:DCLOOTS", "telegram:TrickXpert"],
    )
    opp = Opportunity(deal=deal, estimate=1799, discount=700, discount_pct=38.9, confidence="medium")
    stored = json.loads(json.dumps([opp.model_dump(mode="json")]))
    [loaded] = load_opportunities(stored)
    assert loaded == opp
    assert loaded.currency == "INR"


def test_mixed_memory_loads():
    deal = Deal(product_description="x", price=10, url="https://www.amazon.in/dp/B000000001")
    new = Opportunity(deal=deal, estimate=20, discount=10).model_dump(mode="json")
    loaded = load_opportunities([LEGACY_RECORD, new])
    assert [o.currency for o in loaded] == ["USD", "INR"]
