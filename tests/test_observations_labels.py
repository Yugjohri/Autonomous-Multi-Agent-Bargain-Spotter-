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
