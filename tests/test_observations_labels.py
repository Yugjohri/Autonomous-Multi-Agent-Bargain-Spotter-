import json
import sys
from pathlib import Path

from agents.extraction import SeenStore
from agents.normalize import RedirectResolver
from agents.observations import ObservationLog
from agents.scanner_agent import ScannerAgent
from agents.sources.fixture import FixtureSource

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_labels import build_labels  # noqa: E402

SETTINGS = {"pricer_mode": "inr", "scan": {"freshness_hours": 6, "max_llm_candidates": 30, "picks_per_scan": 5}}


def test_every_parsed_post_is_logged_once(tmp_path):
    log = ObservationLog(tmp_path / "obs.jsonl")
    agent = ScannerAgent(settings=SETTINGS, sources=[FixtureSource()], resolver=RedirectResolver(None),
                         seen_store=SeenStore(tmp_path / "seen.json"), offline=True, sources_config={}, observations=log)
    agent.scan(memory=[])
    rows = [json.loads(line) for line in (tmp_path / "obs.jsonl").read_text(encoding="utf-8").splitlines()]
    # All 12 fixture posts, including expired and percentage-only ones, before filtering.
    assert len(rows) == 12
    assert {r["is_deal"] for r in rows} == {True, False}
    required = {"canonical_id", "title", "description", "price_inr", "mrp_inr", "store", "source", "category",
                "brand", "posted_at", "seen_at", "is_deal"}
    assert required <= set(rows[0])
    agent.scan(memory=[])
    assert len((tmp_path / "obs.jsonl").read_text(encoding="utf-8").splitlines()) == 12, "rescans add no duplicates"


def row(cid, price, when, source="telegram:a", deal=True, mrp=None, high=None):
    return {"canonical_id": cid, "title": "Mivi Play", "price_inr": price, "mrp_inr": mrp, "highest_price_inr": high,
            "store": "amazon_in", "category": "Audio", "brand": "Mivi", "source": source,
            "posted_at": when, "seen_at": when, "is_deal": deal}


def test_build_labels_groups_by_product():
    rows = [
        row("amazon_in:A", 999, "2026-09-01T10:00:00+00:00", mrp=1999),
        row("amazon_in:A", 899, "2026-09-10T10:00:00+00:00", source="telegram:b"),
        row("amazon_in:A", 1099, "2026-09-05T10:00:00+00:00", high=1499),
        row("amazon_in:A", 799, "2026-09-12T10:00:00+00:00", deal=False),  # not a priced deal post
        row("flipkart:ITM1", 5000, "2026-09-02T10:00:00+00:00"),
        row(None, 100, "2026-09-02T10:00:00+00:00"),
    ]
    labels = {l["canonical_id"]: l for l in build_labels(rows)}
    a = labels["amazon_in:A"]
    assert (a["median_price"], a["p75_price"], a["min_price"], a["observations"], a["sources"]) == (999, 1049, 899, 3, 2)
    assert a["first_seen"].startswith("2026-09-01") and a["last_seen"].startswith("2026-09-10")
    assert (a["mrp"], a["highest_price"]) == (1999, 1499)
    assert labels["flipkart:ITM1"]["p75_price"] == 5000
    assert len(labels) == 2
    assert [l["canonical_id"] for l in build_labels(rows, min_count=2)] == ["amazon_in:A"]


def test_labels_use_resolved_short_links():
    from build_labels import recanonicalize

    rows = [
        {**row("amazon_in:aaaa1111aaaa1111", 999, "2026-09-01T10:00:00+00:00"), "url": "https://amzn.to/x1"},
        {**row("amazon_in:bbbb2222bbbb2222", 949, "2026-09-03T10:00:00+00:00", source="telegram:b"), "url": "https://amzn.to/x2"},
    ]
    resolved = {"https://amzn.to/x1": "https://www.amazon.in/dp/B0CHX1W1XY?tag=a-21",
                "https://amzn.to/x2": "https://amazon.in/dp/B0CHX1W1XY"}
    labels = build_labels(recanonicalize(rows, resolved))
    assert [(l["canonical_id"], l["observations"], l["sources"]) for l in labels] == [("amazon_in:B0CHX1W1XY", 2, 2)]
    assert build_labels(rows)[0]["observations"] == 1, "without resolution the two links stay separate"


def test_resolve_script_picks_unresolved_priced_links(tmp_path):
    from resolve_links import urls_to_resolve

    from agents.cache import MemoryCache

    obs = tmp_path / "obs.jsonl"
    lines = [
        {"url": "https://amzn.to/a", "is_deal": True},
        {"url": "https://amzn.to/b", "is_deal": True},
        {"url": "https://amzn.to/c", "is_deal": False},
        {"url": "https://www.amazon.in/dp/B0CHX1W1XY", "is_deal": True},
        {"url": "https://example.com/x", "is_deal": True},
    ]
    obs.write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")
    cache = MemoryCache()
    cache.set("https://amzn.to/b", "https://www.amazon.in/dp/B0CHX1W1XY")
    pending = urls_to_resolve(obs, RedirectResolver(None, cache))
    assert dict(pending) == {"amzn.to": {"https://amzn.to/a"}}


def test_inconsistent_and_token_prices_are_dropped():
    rows = [
        row("amazon_in:B0D842QBMB", 200, "2026-09-01T10:00:00+00:00"),  # booking amount
        row("amazon_in:B0D842QBMB", 169798, "2026-09-02T10:00:00+00:00"),
        row("amazon_in:B0D842QBMB", 169798, "2026-09-03T10:00:00+00:00"),
        row("flipkart:ITM8EB4D0780889A", 12, "2026-09-01T10:00:00+00:00"),  # token price
        row("amazon_in:B08FXNP7CH", 595, "2026-09-01T10:00:00+00:00"),
        row("amazon_in:B08FXNP7CH", 649, "2026-09-05T10:00:00+00:00"),
    ]
    report = {}
    labels = build_labels(rows, report=report)
    assert [l["canonical_id"] for l in labels] == ["amazon_in:B08FXNP7CH"]
    assert report == {"below_min_price": 1, "inconsistent_groups": 1}
    assert len(build_labels(rows, max_spread=0)) == 2, "--max-spread 0 keeps everything priced"
