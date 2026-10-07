from datetime import datetime, timezone

import pytest

from agents.extraction import SeenStore
from agents.normalize import RedirectResolver
from agents.price_signal import Valuation, cap_for_multipack, is_multipack
from agents.scanner_agent import ScannerAgent
from agents.sources.base import DealSource
from agents.sources.telegram_parser import parse_post


@pytest.mark.parametrize(
    "text, multi",
    [
        ("Perfume Combo (Pack of 2) @199", True),
        ("Mamaearth Vitamin C Soap Pack Of 4 @177", True),
        ("Bella vita Perfume Combo Gift Set @449", True),
        ("Duracell AA batteries 8 pcs", True),
        ("Coca-Cola 2 x 750ml", True),
        ("Chemist at Play B1G1 Free", True),
        ("Inalsa Aero Crisp16 12-in-1 Air Fryer", False),
        ("Bajaj 750W Mixer Grinder with 3 jars", False),
        ("Redmi 13C 5G (4GB RAM, 128GB)", False),
        ("Parachute Shampoo 1.2L", False),
        ("Pack of 1 Sandisk 64GB pendrive", False),
        ("Nivea Body Lotion 2x400ml", True),
        ("LG 24U631A, 24-inch, IPS, QHD 2560x1440, 100Hz", False),
        ("LG 126 cm (50 inches) UA82 4K Ultra HD (3840 x 2160) Smart webOS LED TV", False),
        ("XPPen Deco 640 Drawing Pen Tablet 6x4 inches", False),
    ],
)
def test_multipack_detection(text, multi):
    assert is_multipack(text) is multi


def test_multipack_is_capped_at_low_confidence():
    v = cap_for_multipack(Valuation(800, 400, 50.0, "high", "GPT and market agree"), "Perfume Combo (Pack of 2)")
    assert v.confidence == "low" and v.reason.startswith("combo or multipack")
    single = cap_for_multipack(Valuation(800, 400, 50.0, "high", "ok"), "Philips trimmer BT3221")
    assert single.confidence == "high"


class CountingSource(DealSource):
    def __init__(self):
        super().__init__("all-channels")
        self.calls = 0

    def _fetch(self, limit):
        self.calls += 1
        return []


def test_live_run_does_not_reread_sources(tmp_path):
    source = CountingSource()
    agent = ScannerAgent(settings={"pricer_mode": "inr"}, sources=[source], resolver=RedirectResolver(None),
                         seen_store=SeenStore(tmp_path / "s.json"), offline=True, sources_config={})
    live_post = parse_post("Mivi Play speaker @1,099\nhttps://www.amazon.in/dp/B08FTB3CCK", "OMGDeals", 9,
                           datetime.now(timezone.utc))
    selection = agent.scan(memory=[], extra=[live_post], fetch=False)
    assert source.calls == 0, "a live run must only process the new posts"
    assert [d.canonical_id for d in selection.deals] == ["amazon_in:B08FTB3CCK"]
    agent.scan(memory=[])
    assert source.calls == 1, "the timed scan still reads every source"
