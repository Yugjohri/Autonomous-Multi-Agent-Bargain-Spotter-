from datetime import datetime, timedelta, timezone

import pytest

from agents.normalize import (
    RedirectResolver,
    SeenIndex,
    canonical_id,
    clean_url,
    dedupe_by_canonical,
    detect_store,
    is_fresh,
    normalize_link,
    title_from_url,
    unwrap,
)
from tests.fakes import FakeHttp

# Redirect hops recorded from real shortened links on 2 Oct 2026 (store hops never requested).
REDIRECTS = {
    "https://amzn.to/4gsVd7F": "https://amazon.in/dp/B09T73T1PC?th=1&tag=supcootri-21",
    "https://link.amazon/B0heDcPOP": "https://amzlinks.in/B0heDcPOP",
    "https://amzlinks.in/B0heDcPOP": (
        "https://www.amazon.in/dp/B0CR19CJZS/ref=cm_sw_r_as_gl_api_gl_i_G9JHNA3N7MPZDKC3198V"
        "?linkCode=ml1&tag=omtr-21&linkId=4ebb47b46a3120d9fefe1f06f81af385&ascsubtag=srctok-5f46efe6e74bb907"
    ),
    "https://fkrt.cc/hGau0CX": (
        "https://dl.flipkart.com/dl/oscar-forever-luxury-aqua-oud-premium-perfume-combo/p/itm71f2b7f22cce7?pid=PERGQUT"
    ),
    "https://grbn.in/7LfC12": (
        "https://www.flipkart.com/v7-12-w-2-1-wall-charger-mobile/p/itme1101acaa86cb"
        "?pid=SMWGG5GFTQFBFZC2&affid=bh7162&affExtParam1=1005&affExtParam2=gb"
    ),
    "https://links.bigtricks.in/fzlta": (
        "https://go.bigtricks.in/?o=https%3A%2F%2Fwww.amazon.in%2Fdp%2FB07BH3RKL4%3Fpsc%3D1%26th%3D1%26tag%3Dbigin-21"
    ),
    "https://bitli.in/rXMB4UB": (
        "https://trackingv3.linkredirect.in/visitretailer/3191?id=1445208&shareid=rXMB4UB"
        "&dl=https%3A%2F%2Fwww.sheinindia.in%2Fs%2Fstreetwear-bottoms-210700%3Fsource_cal%3Dtg%26utm_source%3Dbitli"
    ),
    "https://bit.ly/45f9GhY": "https://www.croma.com/apple-iphone-15-128gb-black-/p/300652?utm_source=affiliate&affid=xyz",
    "https://cutt.ly/aB3dE": "https://www.amazon.in/gp/product/B0D7QF3ZXY?tag=abc-21&smid=A14CZOWI0VEHLG&psc=1",
}

CASES = [
    ("https://amzn.to/4gsVd7F", "amazon_in", "amazon_in:B09T73T1PC", "https://www.amazon.in/dp/B09T73T1PC"),
    ("https://link.amazon/B0heDcPOP", "amazon_in", "amazon_in:B0CR19CJZS", "https://www.amazon.in/dp/B0CR19CJZS"),
    (
        "https://fkrt.cc/hGau0CX",
        "flipkart",
        "flipkart:ITM71F2B7F22CCE7",
        "https://www.flipkart.com/oscar-forever-luxury-aqua-oud-premium-perfume-combo/p/itm71f2b7f22cce7?pid=PERGQUT",
    ),
    (
        "https://grbn.in/7LfC12",
        "flipkart",
        "flipkart:ITME1101ACAA86CB",
        "https://www.flipkart.com/v7-12-w-2-1-wall-charger-mobile/p/itme1101acaa86cb?pid=SMWGG5GFTQFBFZC2",
    ),
    ("https://links.bigtricks.in/fzlta", "amazon_in", "amazon_in:B07BH3RKL4", "https://www.amazon.in/dp/B07BH3RKL4"),
    ("https://bitli.in/rXMB4UB", "shein_in", None, "https://www.sheinindia.in/s/streetwear-bottoms-210700"),
    ("https://bit.ly/45f9GhY", "croma", None, "https://www.croma.com/apple-iphone-15-128gb-black-/p/300652"),
    ("https://cutt.ly/aB3dE", "amazon_in", "amazon_in:B0D7QF3ZXY", "https://www.amazon.in/dp/B0D7QF3ZXY"),
    (
        "https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX1W1XY/ref=sr_1_1?dib=eyJ2&dib_tag=se&qid=1759735404&s=electronics",
        "amazon_in",
        "amazon_in:B0CHX1W1XY",
        "https://www.amazon.in/dp/B0CHX1W1XY",
    ),
    (
        "https://www.flipkart.com/apple-iphone-15/p/itm6ac6485515ae4?pid=MOBGTAGPTB3VS24W&lid=LSTMOB&marketplace=FLIPKART&otracker=search&affid=xyz",
        "flipkart",
        "flipkart:ITM6AC6485515AE4",
        "https://www.flipkart.com/apple-iphone-15/p/itm6ac6485515ae4?pid=MOBGTAGPTB3VS24W",
    ),
    (
        "https://www.myntra.com/sports-shoes/puma/puma-men-running-shoes/2412341/buy?utm_source=telegram&utm_medium=deals",
        "myntra",
        None,
        "https://www.myntra.com/sports-shoes/puma/puma-men-running-shoes/2412341/buy",
    ),
]


@pytest.mark.parametrize("url, store, canonical, cleaned", CASES)
def test_canonicalization(url, store, canonical, cleaned):
    http = FakeHttp(REDIRECTS)
    link = normalize_link(url, RedirectResolver(http))
    assert link.store == store
    assert link.url == cleaned
    if canonical:
        assert link.canonical_id == canonical
    else:
        assert link.canonical_id.startswith(f"{store}:") and len(link.canonical_id.split(":")[1]) == 16


def test_affiliate_params_removed_and_stores_never_requested():
    http = FakeHttp(REDIRECTS)
    resolver = RedirectResolver(http)
    for url, *_ in CASES:
        link = normalize_link(url, resolver)
        for marker in ("tag=", "affid", "affExtParam", "utm_", "linkCode", "ascsubtag", "otracker"):
            assert marker not in link.url
    # FakeHttp asserts on any store request; check the wrapper was unwrapped without a request.
    assert not any("go.bigtricks.in" in u or "linkredirect.in" in u for _, u in http.requests)


def test_robots_disallowed_host_is_not_followed():
    http = FakeHttp({"https://ddime.in/hZtI": "https://visit.desidime.com/visit/x"}, disallowed_hosts={"ddime.in"})
    link = normalize_link("https://ddime.in/hZtI", RedirectResolver(http))
    assert link.resolved == "https://ddime.in/hZtI"
    assert http.requests == []


def test_head_not_supported_falls_back_to_get():
    url = "https://link.amazon/X1"
    http = FakeHttp({url: "https://www.amazon.in/dp/B0CR19CJZS"}, head_unsupported={url})
    link = normalize_link(url, RedirectResolver(http))
    assert link.canonical_id == "amazon_in:B0CR19CJZS"
    assert [method for method, _ in http.requests] == ["HEAD", "GET"]


def test_resolutions_are_cached():
    http = FakeHttp(REDIRECTS)
    resolver = RedirectResolver(http)
    resolver.resolve("https://amzn.to/4gsVd7F")
    resolver.resolve("https://amzn.to/4gsVd7F")
    assert len(http.requests) == 1


def test_offline_resolution_uses_shortener_store():
    link = normalize_link("https://amzn.to/4gsVd7F", RedirectResolver(None))
    assert link.store == "amazon_in"
    assert link.canonical_id.startswith("amazon_in:")


def test_max_hops():
    chain = {f"https://bit.ly/{i}": f"https://bit.ly/{i + 1}" for i in range(10)}
    http = FakeHttp(chain)
    final = RedirectResolver(http, max_hops=5).resolve("https://bit.ly/0")
    assert final == "https://bit.ly/5"


def test_unwrap_nested_param():
    url = "https://links.ddime.in/?cid=1&url=https%3A%2F%2Fwww.flipkart.com%2Fx%2Fp%2Fitm59297445bce52%3Faffid%3Dsalescueli"
    assert unwrap(url) == "https://www.flipkart.com/x/p/itm59297445bce52?affid=salescueli"


def test_detect_store_and_title():
    assert detect_store("https://dl.flipkart.com/dl/x/p/itm1") == "flipkart"
    assert detect_store("https://www.amazon.com/dp/B000000000") == "other"
    assert title_from_url("https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX1W1XY") == "Apple Iphone 15 128 Gb"
    assert canonical_id("https://www.amazon.in/dp/b0chx1w1xy") == "amazon_in:B0CHX1W1XY"
    assert clean_url("https://Example.com/a/?utm_source=x&id=7#frag") == "https://example.com/a?id=7"


# ---------------------------------------------------------------- dedupe logic


class Item:
    def __init__(self, cid, source, price=100):
        self.cid, self.seen_in, self.price = cid, [source], price


def test_dedupe_merges_sources():
    items = [Item("amazon_in:A", "telegram:x"), Item("flipkart:B", "telegram:x"), Item("amazon_in:A", "telegram:y")]
    result = dedupe_by_canonical(items, key=lambda i: i.cid, merge=lambda kept, dup: kept.seen_in.extend(dup.seen_in))
    assert [i.cid for i in result] == ["amazon_in:A", "flipkart:B"]
    assert result[0].seen_in == ["telegram:x", "telegram:y"]


def test_seen_index_counts_price_drops_as_new():
    seen = SeenIndex()
    assert seen.is_new("amazon_in:A", 999)
    seen.add("amazon_in:A", 999)
    assert not seen.is_new("amazon_in:A", 999)
    assert not seen.is_new("amazon_in:A", 1099)
    assert seen.is_new("amazon_in:A", 899)
    seen.add("amazon_in:A", 899)
    assert not seen.is_new("amazon_in:A", 950)


def test_freshness_window():
    now = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    assert is_fresh(now - timedelta(hours=5), 6, now)
    assert not is_fresh(now - timedelta(hours=7), 6, now)
    assert is_fresh(None, 6, now)
    assert is_fresh(datetime(2026, 10, 2, 11, 0), 6, now)
