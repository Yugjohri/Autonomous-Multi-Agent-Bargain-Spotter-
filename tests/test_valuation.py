from types import SimpleNamespace

import numpy as np
import pytest

from agents.deals import Deal
from agents.ensemble_agent import EnsembleAgent
from agents.frontier_agent import FrontierAgent
from agents.inr_store import InrItem, InrProductStore, Similar
from agents.price_signal import market_evidence, value_inr

SETTINGS = {
    "pricer_mode": "inr",
    "ensemble": {"inr": {"frontier": 0.6, "market": 0.3, "mrp": 0.1, "mrp_factor": 0.75,
                         "market_max_distance": 0.45, "market_min_items": 3}},
}


def similars(prices, distance=0.2):
    return [Similar(document=f"item {p}", price=p, distance=distance) for p in prices]


def test_all_signals_agree_high_confidence():
    v = value_inr(price=1099, mrp=4490, llm_estimate=1799, similars=similars([1650, 1700, 1800, 1750, 1999]))
    assert v.confidence == "high"
    # 0.6*1799 + 0.3*1750 + 0.1*(4490*0.75) = 1079.4 + 525 + 336.75 = 1941.15, kept in whole rupees
    assert v.estimate == 1941
    assert v.discount == 842
    assert v.discount_pct == pytest.approx(43.4)
    assert "GPT and market agree" in v.reason


def test_disagreeing_signals_lower_confidence():
    v = value_inr(price=1099, llm_estimate=4000, similars=similars([1500, 1600, 1700]))
    assert v.confidence == "low"
    assert "differ" in v.reason


def test_mrp_only_is_low_confidence():
    v = value_inr(price=4784, mrp=14995)
    assert v.confidence == "low"
    assert v.estimate == round(14995 * 0.75)
    assert "only one signal (mrp)" in v.reason


def test_no_signal_means_no_discount():
    v = value_inr(price=999)
    assert (v.estimate, v.discount, v.confidence) == (999, 0.0, "low")


def test_far_neighbours_are_not_market_evidence():
    assert market_evidence(similars([100, 200, 300], distance=0.8)) is None
    assert market_evidence(similars([100, 200])) is None
    assert market_evidence(similars([100, 200, 300])).median == 200


def test_weights_from_config_and_missing_signals_renormalize():
    v = value_inr(price=900, llm_estimate=1000, similars=similars([1200, 1200, 1200]), weights={"frontier": 1.0, "market": 1.0, "mrp": 0})
    assert v.estimate == pytest.approx(1100)


# ----------------------------------------------------------------- ensemble


class FakeEncoder:
    """Deterministic bag-of-words vectors, so tests need no model download."""

    def encode(self, texts, show_progress_bar=False):
        vocab = ["speaker", "bluetooth", "mivi", "phone", "samsung", "shampoo", "earbuds", "boat"]
        rows = []
        for text in texts:
            words = text.lower()
            row = [float(words.count(w)) for w in vocab] + [0.01]
            rows.append(row)
        return np.array(rows)


@pytest.fixture
def store(tmp_path):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return InrProductStore(client=client, encoder=FakeEncoder())


class FakeFrontier:
    def __init__(self, estimate):
        self.estimate, self.calls = estimate, []

    def estimate_inr(self, text, similars):
        self.calls.append((text, similars))
        return self.estimate


def test_inr_store_upserts_and_excludes_own_product(store):
    items = [InrItem(document="Mivi Play bluetooth speaker", price=p, canonical_id=f"amazon_in:B0{i}") for i, p in enumerate([899, 999, 1099, 1199])]
    items.append(InrItem(document="Mivi Play bluetooth speaker", price=899, canonical_id="amazon_in:B00"))
    assert store.add(items) == 4
    found = store.similar("mivi bluetooth speaker", exclude_canonical="amazon_in:B00")
    assert "amazon_in:B00" not in {s.canonical_id for s in found}
    assert len(found) == 3


def test_ensemble_inr_uses_frontier_and_market_only(store):
    store.add([InrItem(document="Mivi bluetooth speaker", price=p, canonical_id=f"amazon_in:X{p}") for p in (1400, 1500, 1600)])
    frontier = FakeFrontier(1500)
    ensemble = EnsembleAgent(settings=SETTINGS, inr_store=store, frontier=frontier)
    assert not hasattr(ensemble, "specialist") and not hasattr(ensemble, "neural_network")
    deal = Deal(product_description="Portable bluetooth speaker", title="Mivi Play speaker", price=999,
                url="https://www.amazon.in/dp/B08FTB3CCK", canonical_id="amazon_in:B08FTB3CCK")
    v = ensemble.value(deal)
    assert v.confidence == "high"
    assert v.estimate == pytest.approx(1500)
    assert len(frontier.calls) == 1


def test_ensemble_survives_frontier_failure(store):
    class Broken:
        def estimate_inr(self, *a):
            raise TimeoutError("api down")

    ensemble = EnsembleAgent(settings=SETTINGS, inr_store=store, frontier=Broken())
    deal = Deal(product_description="x", title="Phone", price=9999, url="https://www.amazon.in/dp/B0ABCDEFGH", mrp=15999)
    v = ensemble.value(deal)
    assert v.confidence == "low" and v.estimate == round(15999 * 0.75)


def test_price_in_legacy_api_refuses_inr_mode(store):
    ensemble = EnsembleAgent(settings=SETTINGS, inr_store=store, frontier=FakeFrontier(1))
    with pytest.raises(RuntimeError):
        ensemble.price("anything")


def test_frontier_inr_prompt_and_parsing():
    class Client:
        def __init__(self):
            self.messages = None
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.messages = kwargs["messages"]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="₹1,49,999"))])

    client = Client()
    frontier = FrontierAgent(client=client, encoder=object())
    result = frontier.estimate_inr("Samsung Galaxy S24 Ultra 256GB", [Similar("Galaxy S24", 129999, 0.1, mrp=134999)])
    prompt = client.messages[0]["content"]
    assert result == 149999
    assert "India" in prompt and "INR" in prompt and "$" not in prompt
    assert "₹1,29,999" in prompt and "MRP ₹1,34,999" in prompt


def test_populate_reads_indian_csv_formats(tmp_path):
    from populate_vectorstore import items_from_csv

    a = tmp_path / "flipkart.csv"
    a.write_text(
        'category_1,title,selling_price,mrp\n"Sports",Cricket Net (Green),"₹1,615","₹4,000"\n"x",Bad Row,,\n',
        encoding="utf-8",
    )
    b = tmp_path / "amazon.csv"
    b.write_text(
        "﻿asin,product_title,price_inr,original_price_inr,currency,scraped_at\n"
        "B0FN7QTRPY,Samsung Galaxy M07 Mobile 4GB 64GB,9999,10999,INR,2026-01-05\n"
        "B000000001,US item,10,20,USD,2026-01-05\n",
        encoding="utf-8",
    )
    rows = list(items_from_csv(a)) + list(items_from_csv(b))
    assert [(r.document[:12], r.price, r.mrp) for r in rows] == [("Cricket Net ", 1615, 4000), ("Samsung Gala", 9999, 10999)]
    assert rows[1].canonical_id == "amazon_in:B0FN7QTRPY" and rows[1].category == "Mobiles"


FRONTIER_ONLY = {"frontier": 1.0, "market": 0.0, "mrp": 0.0, "neural_network": 0.0}


def test_zero_weight_signals_still_check_confidence():
    v = value_inr(price=1099, mrp=4490, llm_estimate=1799, similars=similars([1650, 1700, 1800]), weights=FRONTIER_ONLY)
    assert v.estimate == 1799
    assert v.confidence == "high" and "GPT and market agree" in v.reason
    assert set(v.signals) == {"frontier", "market", "mrp"}


def test_checks_become_the_estimate_when_gpt_fails():
    v = value_inr(price=999, similars=similars([1400, 1500, 1600]), weights=FRONTIER_ONLY)
    assert v.estimate == 1500 and v.confidence == "low"
    assert "estimate from checks only" in v.reason


def test_search_page_deal_is_capped_at_low_confidence():
    from agents.price_signal import Valuation, cap_for_listing

    v = cap_for_listing(Valuation(1500, 691, 46.1, "medium", "GPT ₹1,500"), "https://www.flipkart.com/search?q=watches")
    assert v.confidence == "low" and v.reason.startswith("links to a search or sale page")
    product = cap_for_listing(Valuation(1500, 691, 46.1, "medium", "ok"), "https://www.amazon.in/dp/B0BHSWVGYB")
    assert product.confidence == "medium"
