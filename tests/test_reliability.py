import time

import requests

from agents.deals import ScrapedDeal
from agents.extraction import SeenStore
from agents.normalize import RedirectResolver
from agents.scanner_agent import ScannerAgent
from agents.sources.base import DealSource, RawDeal

ENTRY = {"title": "Example 4K TV", "summary": "<div class='snippet summary'>A 55 inch TV</div>",
         "links": [{"href": "https://www.dealnews.com/x.html"}]}


class Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class Session:
    def __init__(self, content=b"", exc=None):
        self.content, self.exc, self.timeouts = content, exc, []

    def get(self, url, timeout=None):
        self.timeouts.append(timeout)
        if self.exc:
            raise self.exc
        return Resp(self.content)


def test_scraped_deal_without_content_section_uses_summary():
    session = Session(b"<html><body>No section here</body></html>")
    deal = ScrapedDeal(ENTRY, session)
    assert deal.details == "A 55 inch TV" and deal.features == ""
    assert session.timeouts == [ScrapedDeal.TIMEOUT]


def test_scraped_deal_survives_network_errors():
    deal = ScrapedDeal(ENTRY, Session(exc=requests.ConnectionError("down")))
    assert deal.details == "A 55 inch TV"


def test_scraped_deal_reads_content_section():
    html = b"<div class='content-section'>Great TV\nmore Features HDR10 and Dolby Vision</div>"
    deal = ScrapedDeal(ENTRY, Session(html))
    assert deal.details.strip() == "Great TV" and "Dolby Vision" in deal.features


class Slow(DealSource):
    def __init__(self, name, delay, deals=()):
        super().__init__(name)
        self.delay, self.deals, self.calls = delay, list(deals), 0

    def _fetch(self, limit):
        self.calls += 1
        time.sleep(self.delay)
        return list(self.deals)


def scanner(tmp_path, sources, budget):
    settings = {"pricer_mode": "inr", "scan": {"time_budget_seconds": budget, "source_workers": 4}}
    return ScannerAgent(settings=settings, sources=sources, resolver=RedirectResolver(None),
                        seen_store=SeenStore(tmp_path / "s.json"), offline=True, sources_config={})


def test_sources_run_in_parallel_within_budget(tmp_path):
    deal = RawDeal(title="t", text="t", url="https://www.amazon.in/dp/B0ABCDEFGH", price_hint=999)
    fast = [Slow(f"fast{i}", 0.3, [deal]) for i in range(3)]
    hung = Slow("hung", 2.0)
    agent = scanner(tmp_path, fast + [hung], budget=1.0)
    start = time.monotonic()
    raws = agent.fetch_raw()
    elapsed = time.monotonic() - start
    assert len(raws) == 3, "results of the fast sources are kept"
    assert elapsed < 2.0, "three 0.3s sources in parallel plus a hung one finish inside the 1s budget"
    # The hung source is not started again while its first fetch is still running.
    agent.fetch_raw()
    assert hung.calls == 1
    agent._running["hung"].result()  # let it finish before the test ends


def test_failing_source_does_not_stop_scan(tmp_path):
    class Boom(DealSource):
        def _fetch(self, limit):
            raise RuntimeError("source exploded")

    deal = RawDeal(title="t", text="t", url="https://www.amazon.in/dp/B0ABCDEFGH", price_hint=999)
    agent = scanner(tmp_path, [Boom("boom"), Slow("ok", 0, [deal])], budget=5)
    assert len(agent.fetch_raw()) == 1
