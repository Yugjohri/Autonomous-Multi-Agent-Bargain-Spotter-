import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from agents.deals import Deal
from agents.deep_neural_network import DeepNeuralNetwork, InrPriceModel
from agents.evaluator_inr import breakdown, color_for, evaluate, metrics, price_band, scatter_png, with_bands
from agents.inr_features import build_features, category_onehot, input_size
from agents.price_signal import value_inr
from agents.taxonomy import TAXONOMY

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_splits import split  # noqa: E402


# ---------------------------------------------------------------- evaluator


def test_metrics_in_percent_and_log_space():
    m = metrics([100, 1000, 10000], [110, 800, 10000])
    assert m["n"] == 3
    assert m["mdape"] == pytest.approx(0.1)
    assert m["within_10"] == pytest.approx(2 / 3)
    assert m["within_20"] == pytest.approx(1.0)
    assert m["mae_inr"] == pytest.approx((10 + 200 + 0) / 3)
    assert m["median_ratio"] == pytest.approx(1.0)
    assert 0.9 < m["r2_log"] <= 1


def test_metrics_skip_missing_predictions():
    assert metrics([100, 200], [np.nan, 200])["n"] == 1
    assert metrics([100], [np.nan]) == {"n": 0}


def test_bands_and_colours():
    assert [price_band(p) for p in (999, 1000, 49999, 50000)] == ["under 1k", "1k to 10k", "10k to 50k", "over 50k"]
    assert color_for(1000, 1150) == "green" and color_for(1000, 1300) == "orange" and color_for(1000, 2000) == "red"


def test_breakdown_by_band_and_scatter(tmp_path):
    df = with_bands(pd.DataFrame({"price_inr": [500, 600, 5000, 60000], "pred": [550, 500, 5000, 30000]}))
    out = breakdown(df, "price_band", order=["under 1k", "1k to 10k", "10k to 50k", "over 50k"])
    assert list(out["price_band"]) == ["under 1k", "1k to 10k", "over 50k"]
    assert list(out["n"]) == [2, 1, 1]
    scatter_png(df["price_inr"], df["pred"], tmp_path / "s.png", "test")
    assert (tmp_path / "s.png").stat().st_size > 1000


def test_evaluate_runs_predictor_and_survives_errors():
    items = pd.DataFrame({"price_inr": [100.0, 200.0, 300.0], "title": ["a", "boom", "c"]})

    def predictor(row):
        if row["title"] == "boom":
            raise RuntimeError("api down")
        return row["price_inr"] * 1.1

    preds = evaluate(predictor, items, workers=2, verbose=False)
    assert preds.iloc[0] == pytest.approx(110) and np.isnan(preds.iloc[1])


# ------------------------------------------------------------------- splits


def _products(rows):
    base = {"text": "", "brand": "", "mrp_inr": np.nan, "store": "amazon_in", "scraped_at": "", "n_variants": 1,
            "category_source": "rules"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_split_removes_test_products_from_train_by_id_and_title():
    rows = [
        {"product_key": "amazon_in:T1", "alt_ids": "amazon_in:T1", "title": "Phone A", "norm_title": "phone a",
         "category": "smartphone", "price_inr": 10000, "source_dataset": "amazon_2026"},
        {"product_key": "amazon_in:T2", "alt_ids": "amazon_in:T2", "title": "Phone B", "norm_title": "phone b",
         "category": "smartphone", "price_inr": 12000, "source_dataset": "amazon_2026"},
        # Same ASIN as a test product, listed among the colour variants.
        {"product_key": "amazon_in:X1", "alt_ids": "amazon_in:X1;amazon_in:T1", "title": "Phone A (Blue)",
         "norm_title": "phone a blue", "category": "smartphone", "price_inr": 9000, "source_dataset": "amazon_2025"},
        # Different id, same normalised title.
        {"product_key": "flipkart:P1", "alt_ids": "flipkart:P1", "title": "Phone B", "norm_title": "phone b",
         "category": "smartphone", "price_inr": 11000, "source_dataset": "flipkart_2025"},
    ]
    rows += [
        {"product_key": f"amazon_in:C{i}", "alt_ids": f"amazon_in:C{i}", "title": f"Case {i}",
         "norm_title": f"case {i}", "category": "mobile_accessory" if i % 2 else "smartphone",
         "price_inr": 200 + i, "source_dataset": "amazon_2025"}
        for i in range(40)
    ]
    train, val, test, report = split(_products(rows), val_fraction=0.1, seed=1)
    assert report["removed_by_id"] == 1 and report["removed_by_title_only"] == 1
    assert len(test) == 2 and len(train) + len(val) == 40
    assert not set(train["product_key"]) & {"amazon_in:X1", "flipkart:P1"}
    assert set(val["category"]) == {"smartphone", "mobile_accessory"}


# ---------------------------------------------------------------- features


def test_feature_sizes_match_spec():
    spec = {"hashing": True, "embeddings": True, "category": True}
    fake_embeddings = np.zeros((2, 384), dtype=np.float32)
    x = build_features(["boAt earbuds", "Samsung phone"], ["earbuds_headphones", "smartphone"], spec, fake_embeddings)
    assert x.shape == (2, input_size(spec)) == (2, 5000 + 384 + len(TAXONOMY))
    onehot = category_onehot(["not a category"])
    assert onehot[0, TAXONOMY.index("other")] == 1.0


# ------------------------------------------------------------ INR NN model


@pytest.fixture
def tiny_model(tmp_path):
    spec = {"hashing": True, "embeddings": False, "category": True}
    torch.manual_seed(0)
    model = DeepNeuralNetwork(input_size(spec), num_layers=3, hidden_size=16)
    torch.save(model.state_dict(), tmp_path / "nn_inr.pth")
    meta = {"features": spec, "hidden_size": 16, "num_layers": 3, "dropout": 0.2, "y_mean": 7.0, "y_std": 1.5}
    (tmp_path / "nn_inr_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path / "nn_inr.pth"


def test_inr_model_reads_sizes_and_target_stats_from_meta(tiny_model):
    model = InrPriceModel(str(tiny_model), device="cpu")
    assert model.y_mean == 7.0 and model.y_std == 1.5
    price = model.price("boAt Airdopes 141 TWS earbuds", "Audio")
    assert price > 0
    # The output is expm1(standardised * std + mean), so a zero output gives expm1(7).
    with torch.no_grad():
        for p in model.model.output_layer.parameters():
            p.zero_()
    assert model.price("anything") == pytest.approx(np.expm1(7.0), rel=1e-5)


def test_nn_agent_loads_inr_model(tiny_model):
    from agents.neural_network_agent import NeuralNetworkAgent

    agent = NeuralNetworkAgent("inr", weights=str(tiny_model))
    assert agent.price("Samsung Galaxy A17 5G (6GB RAM, 128GB Storage)", "Mobiles") > 0


# ------------------------------------------------------- valuation signal


def test_neural_network_signal_is_off_without_weight():
    v = value_inr(price=999, llm_estimate=1500, nn_estimate=1400)
    assert "neural_network" not in v.signals and v.confidence == "low"


def test_frontier_and_model_agreeing_give_medium_confidence():
    weights = {"frontier": 0.6, "market": 0.3, "neural_network": 0.2, "mrp": 0.0}
    v = value_inr(price=999, llm_estimate=1500, nn_estimate=1400, weights=weights)
    assert v.confidence == "medium" and "GPT and model agree" in v.reason
    assert v.estimate == round((0.6 * 1500 + 0.2 * 1400) / 0.8)
    far = value_inr(price=999, llm_estimate=3000, nn_estimate=1400, weights=weights)
    assert far.confidence == "low" and "differ" in far.reason


def test_high_confidence_still_needs_market_listings():
    from agents.inr_store import Similar

    weights = {"frontier": 0.5, "market": 0.3, "neural_network": 0.2, "mrp": 0.0}
    close = [Similar(document="x", price=p, distance=0.1) for p in (1450, 1500, 1550)]
    v = value_inr(price=999, llm_estimate=1500, nn_estimate=1450, similars=close, weights=weights)
    assert v.confidence == "high"


class FakeModel:
    def __init__(self, estimate):
        self.estimate = estimate
        self.calls = []

    def price(self, text, category=None):
        self.calls.append((text, category))
        return self.estimate


def test_ensemble_passes_nn_estimate_and_category(tmp_path):
    import chromadb

    from agents.ensemble_agent import EnsembleAgent
    from agents.inr_store import InrProductStore
    from tests.test_valuation import FakeEncoder, FakeFrontier

    store = InrProductStore(client=chromadb.PersistentClient(path=str(tmp_path / "c")), encoder=FakeEncoder())
    settings = {"pricer_mode": "inr", "ensemble": {"inr": {"frontier": 0.6, "market": 0.3, "mrp": 0.1,
                                                           "neural_network": 0.2}}}
    model = FakeModel(1450)
    ensemble = EnsembleAgent(settings=settings, inr_store=store, frontier=FakeFrontier(1500), inr_model=model)
    deal = Deal(product_description="Running shoes", title="Nike Revolution 7", price=999,
                url="https://www.amazon.in/dp/B0ABCDEFGH", category="Footwear")
    v = ensemble.value(deal)
    assert model.calls and model.calls[0][1] == "Footwear"
    assert v.signals["neural_network"] == 1450 and v.confidence == "medium"


def test_ensemble_without_model_file_skips_signal(tmp_path, monkeypatch):
    import chromadb

    from agents.ensemble_agent import EnsembleAgent
    from agents.inr_store import InrProductStore
    from tests.test_valuation import FakeEncoder, FakeFrontier

    monkeypatch.chdir(tmp_path)
    store = InrProductStore(client=chromadb.PersistentClient(path=str(tmp_path / "c")), encoder=FakeEncoder())
    settings = {"pricer_mode": "inr", "ensemble": {"inr": {"frontier": 0.8, "neural_network": 0.2}}}
    ensemble = EnsembleAgent(settings=settings, inr_store=store, frontier=FakeFrontier(1500))
    assert ensemble.inr_model is None
    deal = Deal(product_description="x", title="Phone", price=999, url="https://www.amazon.in/dp/B0ABCDEFGH")
    assert ensemble.value(deal).estimate == 1500


# ------------------------------------------------------------ store rebuild


def test_rebuild_keeps_telegram_rows_and_excludes_holdout(tmp_path):
    import chromadb

    from agents.inr_store import InrItem, InrProductStore
    from populate_vectorstore import holdout_keys, items_from_split, kept_rows
    from tests.test_valuation import FakeEncoder

    store = InrProductStore(client=chromadb.PersistentClient(path=str(tmp_path / "c")), encoder=FakeEncoder())
    store.add([
        InrItem(document="boAt earbuds deal", price=999, canonical_id="amazon_in:KEEP", source="telegram:loot"),
        InrItem(document="Samsung phone deal", price=9999, canonical_id="amazon_in:TEST1", source="telegram:loot"),
        InrItem(document="Old raw row", price=500, canonical_id="amazon_in:OLD", source="dataset:amazon_2025/x.csv"),
    ])
    test = _products([{"product_key": "amazon_in:TEST1", "alt_ids": "amazon_in:TEST1", "title": "Phone",
                       "norm_title": "mivi speaker", "category": "smartphone", "price_inr": 1,
                       "source_dataset": "amazon_2026"}])
    test.to_parquet(tmp_path / "test.parquet")
    train = _products([{"product_key": "amazon_in:TR1", "alt_ids": "", "title": "Mivi speaker",
                        "norm_title": "mivi speaker", "text": "Mivi speaker", "category": "audio_other",
                        "price_inr": 1499, "source_dataset": "amazon_2025"},
                       {"product_key": "title:abc", "alt_ids": "", "title": "Kurta", "norm_title": "kurta",
                        "text": "Kurta | cotton", "category": "fashion", "price_inr": 499,
                        "source_dataset": "flipkart_khanna"}])
    train.to_parquet(tmp_path / "train.parquet")

    holdout = holdout_keys([tmp_path / "test.parquet"])
    kept = kept_rows(store, ["telegram:"], holdout)
    assert [row[2]["canonical_id"] for row in kept] == ["amazon_in:KEEP"]
    items = list(items_from_split(tmp_path / "train.parquet"))
    assert items[1].canonical_id == "" and items[1].source == "split:train/flipkart_khanna"
    assert items[0].price == 1499 and items[0].category == "audio_other"
