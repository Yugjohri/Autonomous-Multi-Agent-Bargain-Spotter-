import json
from datetime import datetime
from pathlib import Path

import pytest

from agents.sources.telegram_channels import ChannelRef, parse_channels
from agents.sources.telegram_parser import extract_links, parse_post, parse_prices, pick_primary_link
from agents.sources.telegram_public import parse_preview_html

FIXTURES = Path(__file__).parent / "fixtures" / "telegram"
POSTS = json.loads((FIXTURES / "posts.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("post", POSTS, ids=[f"{p['channel']}-{p['message_id']}" for p in POSTS])
def test_fixture_posts(post):
    deal = parse_post(
        post["text"], post["channel"], post["message_id"], datetime.fromisoformat(post["posted_at"]), post["links"]
    )
    expected = post["expected"]
    assert deal.price_hint == expected["price"]
    assert deal.mrp_hint == expected["mrp"]
    assert deal.highest_price_hint == expected["highest_price"]
    assert deal.priceable == expected["priceable"]
    assert deal.drop_reason == expected["drop_reason"]
    assert deal.title == expected["title"]
    assert deal.url == expected["url"]
    assert deal.source == f"telegram:{post['channel']}"
    assert deal.external_id == f"telegram:{post['channel']}/{post['message_id']}"


@pytest.mark.parametrize(
    "text, price, mrp",
    [
        ("Titan Talk Smartwatch Now only ₹4,784 (MRP ₹14,995)", 4784, 14995),
        ("Loot 199 https://fkrt.cc/x", 199, None),
        ("3399\nhttps://amzn.to/x", 3399, None),
        ("Perfume Combo(Pack of 2) @199.", 199, None),
        ("Lenovo laptop at 10,999 after coupon", 10999, None),
        ("Philips trimmer ₹1435 / 65% off", 1435, 4100),
        ("Air fryer ₹8,988 / 63% off", 8988, 24292),
        ("LG TV ₹700 dropped! Current Price: ₹20,999 Highest Price: ₹25,299", 20999, None),
        ("Sony WH-1000XM5 ₹26,990 + ₹2,000 bank discount", 26990, None),
        ("Get ₹500 OFF on your first booking", None, None),
        ("Buy ₹100 Gift Card @ Just ₹1 [For ICICI Bank Credit Card Only]", 1, 100),
        ("Upto 85% off on Puma shoes", None, None),
    ],
)
def test_parse_prices(text, price, mrp):
    p, m, _ = parse_prices(text)
    assert p == price
    assert m == mrp


def test_highest_price_is_kept_as_history_signal():
    _, _, highest = parse_prices("Current Price: ₹2,520\nHighest Price: ₹3,769")
    assert highest == 3769


def test_non_deal_posts_are_skipped():
    assert parse_post("", "c", 1) is None
    assert parse_post("   ", "c", 1) is None
    assert parse_post("Who wants a giveaway?", "c", 1, is_poll=True) is None
    assert parse_post("Good morning everyone, have a nice day", "c", 1, forwarded=True) is None


def test_percentage_only_and_category_posts_are_not_priceable():
    assert parse_post("Shein Loot: Best Trousers & Jeans Upto 70% Off\nhttps://bitli.in/x", "c", 1).drop_reason == "percentage_only"
    assert parse_post("AJIO Loot : Shein Clothing Starts @60\nhttps://ajiio.in/x", "c", 2).drop_reason == "starting_price"


def test_links_from_entities_and_junk_removed():
    links = extract_links(
        "Deal @499 https://amzn.to/abc join https://t.me/somechannel",
        extra=["https://hcti.io/v1/image/x.jpeg", "https://fkrt.cc/def", "https://amzn.to/abc"],
    )
    assert links == ["https://amzn.to/abc", "https://fkrt.cc/def"]


def test_buy_now_link_preferred_over_read_more():
    text = "Read More: https://ddime.in/a\nBuy Now: https://ddime.in/b"
    assert pick_primary_link(text, ["https://ddime.in/a", "https://ddime.in/b"]) == "https://ddime.in/b"


# ------------------------------------------------------------ preview snapshots


def test_preview_snapshot_grabon():
    html = (FIXTURES / "preview_GrabOnIndiaOfficial.html").read_text(encoding="utf-8")
    deals = parse_preview_html(html, ChannelRef(username="GrabOnIndiaOfficial"))
    assert len(deals) == 20
    titan = next(d for d in deals if d.external_id.endswith("/13841"))
    assert (titan.title, titan.price_hint, titan.mrp_hint) == ("Titan Talk Smartwatch", 4784, 14995)
    assert titan.url == "https://grbn.in/7LfC12"
    assert titan.posted_at.isoformat() == "2026-10-02T11:38:07+00:00"
    assert all(d.source == "telegram:GrabOnIndiaOfficial" for d in deals)
    # Category round-ups like "Epic Electronics & Fashion Loot" have no price.
    assert sum(not d.priceable for d in deals) >= 5


def test_preview_snapshot_desidime_hot():
    html = (FIXTURES / "preview_desidimeHot.html").read_text(encoding="utf-8")
    deals = parse_preview_html(html, ChannelRef(username="desidimeHot", trust=0.8))
    fryer = next(d for d in deals if d.external_id.endswith("/17637"))
    assert (fryer.price_hint, fryer.mrp_hint, fryer.store) == (8988, 24292, "flipkart")
    assert fryer.url == "https://ddime.in/hZtI"
    assert fryer.trust == 0.8
    assert all("hcti.io" not in link for d in deals for link in d.links)


def test_channel_refs():
    channels = parse_channels(
        [
            {"username": "@OMGDeals"},
            {"id": -1001439586091, "name": "DCLootsOffers"},
            {"invite": "https://t.me/+AbCdEf123", "enabled": False},
        ]
    )
    assert [c.key for c in channels] == ["OMGDeals", "DCLootsOffers", "invite-AbCdEf"]
    assert [c.public_ok for c in channels] == [True, False, False]
    assert channels[2].invite_hash == "AbCdEf123"
    assert "AbCdEf" not in channels[2].describe()
