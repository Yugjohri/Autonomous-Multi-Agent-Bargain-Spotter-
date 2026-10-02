import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.deals import InrPick, InrSelection
from agents.extraction import SeenStore, build_candidates
from agents.normalize import RedirectResolver, SeenIndex
from agents.scanner_agent import ScannerAgent
from agents.sources.base import DealSource
from agents.sources.telegram_parser import parse_post

POSTS = json.loads((Path(__file__).parent / "fixtures" / "telegram" / "posts.json").read_text(encoding="utf-8"))
SETTINGS = {
    "pricer_mode": "inr",
    "scan": {"freshness_hours": 6, "max_llm_candidates": 30, "picks_per_scan": 5},
    "models": {"scanner": "gpt-5-mini", "prices_per_million": {"gpt-5-mini": [0.25, 2.0]}},
}


def fresh_posts(minutes_ago=10):
    """Fixture posts re-dated to just now, so the freshness window keeps them."""
    now = datetime.now(timezone.utc)
    deals = []
    for i, post in enumerate(POSTS):
        deal = parse_post(post["text"], post["channel"], post["message_id"], now - timedelta(minutes=minutes_ago + i), post["links"])
        deals.append(deal)
    return deals


class ListSource(DealSource):
    def __init__(self, deals):
        super().__init__("fixture")
        self.deals = deals

    def _fetch(self, limit):
        return list(self.deals)


class FakeOpenAI:
    """Mimics client.chat.completions.parse and records the prompt."""

    def __init__(self, picks):
        self.picks = picks
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=InrSelection(picks=self.picks))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=300))


def scanner(tmp_path, picks=(), deals=None, offline=False):
    return ScannerAgent(
        settings=SETTINGS,
        sources=[ListSource(deals if deals is not None else fresh_posts())],
        client=None if offline else FakeOpenAI(list(picks)),
        resolver=RedirectResolver(None),
        seen_store=SeenStore(tmp_path / "seen.json"),
        offline=offline,
        sources_config={},
    )


def test_prefilter_keeps_only_priced_fresh_known_store_posts(tmp_path):
    agent = scanner(tmp_path)
    candidates = agent.candidates(memory=[])
    report = agent.last_report
    # Expired, percentage-only and pricebefore/grabon posts (no recognised store) are gone.
    titles = {c.title for c in candidates}
    assert "Parachute Shampoo 1.2L" in titles
    assert "Mivi Play Bluetooth Speaker with 12 Hours Playtime" in titles
    assert "LG 81.28 cm" not in titles
    assert report.unpriceable >= 2 and report.unknown_store >= 1
    assert all(c.store != "other" for c in candidates)


def test_stale_posts_are_dropped(tmp_path):
    agent = scanner(tmp_path, deals=fresh_posts(minutes_ago=7 * 60))
    assert agent.candidates(memory=[]) == []
    assert agent.last_report.stale == len(POSTS)


def test_llm_prompt_is_inr_only_and_picks_map_to_candidates(tmp_path):
    agent = scanner(tmp_path)
    candidates = agent.candidates(memory=[])
    idx = next(i for i, c in enumerate(candidates) if c.title.startswith("Mivi"))
    agent.openai = FakeOpenAI([
        InrPick(candidate_id=idx, title="Mivi Play speaker", product_description="Portable Bluetooth 5.0 speaker.",
                price=264, coupon_note=None, category="Audio", brand="Mivi"),
        InrPick(candidate_id=999, title="ghost", product_description="x", price=1, coupon_note=None, category="Other", brand=None),
    ])
    agent.seen_store = SeenStore(tmp_path / "seen2.json")
    selection = agent.scan(memory=[])
    prompt = agent.openai.calls[0]["messages"][1]["content"] + agent.openai.calls[0]["messages"][0]["content"]
    assert "$" not in prompt and "₹" in prompt
    assert len(selection.deals) == 1
    deal = selection.deals[0]
    assert deal.currency == "INR"
    assert deal.url == "https://www.amazon.in/dp/B08FTB3CCK"
    assert deal.canonical_id == "amazon_in:B08FTB3CCK"
    assert deal.store == "amazon_in"
    assert deal.source == "telegram:DCLootsOffers"


def test_wild_llm_price_falls_back_to_post_price(tmp_path):
    agent = scanner(tmp_path)
    candidates = agent.candidates(memory=[])
    c = candidates[0]
    deal = ScannerAgent.to_deal(
        InrPick(candidate_id=0, title="t", product_description="d", price=c.price * 50, coupon_note=None, category="Other", brand=None), c
    )
    assert deal.price == c.price


def test_candidates_are_not_sent_twice(tmp_path):
    agent = scanner(tmp_path, offline=True)
    first = agent.scan(memory=[])
    assert first and first.deals
    assert agent.scan(memory=[]) is None


def test_offline_pick_needs_no_client(tmp_path):
    agent = scanner(tmp_path, offline=True)
    selection = agent.scan(memory=[])
    assert 0 < len(selection.deals) <= 5
    assert all(d.currency == "INR" and d.price > 0 for d in selection.deals)


def test_cross_channel_duplicates_count_once_with_all_sources():
    now = datetime.now(timezone.utc)
    a = parse_post("Mivi Play speaker @264\nhttps://www.amazon.in/dp/B08FTB3CCK?tag=a-21", "DCLootsOffers", 1, now)
    b = parse_post("Mivi Play Bluetooth speaker @264\nhttps://amazon.in/dp/B08FTB3CCK?tag=b-21", "OMGDeals", 2, now)
    candidates, report = build_candidates([a, b], None, SeenIndex(), 6)
    assert len(candidates) == 1 and report.duplicates == 1
    assert candidates[0].seen_in == ["telegram:DCLootsOffers", "telegram:OMGDeals"]


@pytest.mark.parametrize("price, new", [(264, False), (300, False), (199, True)])
def test_memory_dedupe_uses_canonical_id_and_price(price, new):
    now = datetime.now(timezone.utc)
    seen = SeenIndex({"amazon_in:B08FTB3CCK": 264})
    post = parse_post(f"Mivi speaker @{price}\nhttps://www.amazon.in/dp/B08FTB3CCK", "x", 1, now)
    candidates, _ = build_candidates([post], None, seen, 6)
    assert bool(candidates) is new
