import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.taxonomy import TAXONOMY, classify, from_app_category, rule_category, taxonomy_for
from agents.titles import normalise_title, specs_key

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_data import (  # noqa: E402
    GIFT_OR_WARRANTY,
    REFURBISHED,
    LlmCategorizer,
    clean_title,
    features_text,
    group_variants,
    iqr_outliers,
    is_range_price,
    khanna_category,
    prepare,
    trim_text,
)


@pytest.mark.parametrize(
    "title, category",
    [
        ("Samsung Galaxy S25 5G Smartphone (Silver Shadow, 12GB RAM, 256GB Storage)", "smartphone"),
        ("POCO C71 (Desert Gold, 128 GB)", "smartphone"),
        ("Spigen Ultra Hybrid MagFit Back Cover Case Compatible with iPhone 17 Pro", "mobile_accessory"),
        ("Sounce Earphone Case Cover for OnePlus Nord Buds 3", "mobile_accessory"),
        ("HP Wired On Ear Headphones with Mic, Compatible with Laptop", "earbuds_headphones"),
        ("Lenovo LOQ, AMD Ryzen 7 7435HS, NVIDIA RTX 3050A 4GB, 24GB RAM, 512GB SSD", "laptop"),
        ("EVM 8GB DDR4 Laptop RAM 2666MHz", "computer_accessory"),
        ("Elfora Heavy Duty Metal Adjustable Tablet Stand Holder for iPad", "computer_accessory"),
        ("Xiaomi 108 cm (43 inch) A Full HD Smart Google LED TV", "tv"),
        ("Redmi 80 cm (32 inches) F Series HD Ready Smart LED Fire TV", "tv"),
        ("Haier 325 L 2 Star Frost Free Double Door Bottom Mount Refrigerator", "large_appliance"),
        ("Samsung 1.5 Ton 3 Star Split AC", "large_appliance"),
        ("Larah by Borosil Dinner Set | Microwave & Dishwasher Safe", "small_appliance"),
        ("Samsung Galaxy Watch8 Classic (46mm Bluetooth, Black)", "smartwatch"),
        ("JBL Cinema SB560 Soundbar for Smart TV", "audio_other"),
        ("LEROKAS J36 Ultra Handheld Gaming Console", "other"),
        ("realme 10000mAh Power Bank", "mobile_accessory"),
    ],
)
def test_rule_categories(title, category):
    assert rule_category(title) == category


def test_accessory_after_device_word_is_unsure():
    assert classify("Spigen Optik Armor for Samsung Galaxy S24 Ultra Case") == ("smartphone", False)


def test_hint_used_when_rules_find_nothing():
    assert classify("Mystery gadget 3000", hint="camera") == ("camera", True)
    assert classify("Mystery gadget 3000") == ("other", False)


def test_app_category_mapping_is_inside_taxonomy():
    assert from_app_category("Mobiles") == "smartphone"
    assert from_app_category("Gift Cards") == "other"
    assert taxonomy_for("Mystery gadget", "Footwear") == "footwear"
    assert taxonomy_for("Nike Revolution 7 running shoes", "Other") == "footwear"
    assert from_app_category("Wearables") in TAXONOMY


def test_normalise_title_merges_colour_variants():
    a = normalise_title("Samsung Galaxy M56 5G (Black, 8 GB RAM, 128 GB Storage)| Slimmest Phone")
    b = normalise_title("Samsung Galaxy M56 5G (Light Green, 8 GB RAM, 128 GB Storage)| Slimmest Phone")
    assert a == b == "samsung galaxy m56 5g 8 gb ram 128 gb storage slimmest phone"


def test_specs_key_keeps_storage_variants_apart():
    assert specs_key("iQOO Z10 5G (Glacier Silver, 12GB RAM, 256GB Storage)") == "ram12/256gb/"
    assert specs_key("iQOO Z10 5G (Glacier Silver, 8GB RAM, 128GB Storage)") != specs_key(
        "iQOO Z10 5G (Glacier Silver, 12GB RAM, 256GB Storage)"
    )
    assert specs_key('LG 27 inch IPS Monitor').endswith("27inch")


def test_cleaning_helpers():
    assert clean_title("Boult Astra Neo 70Hrs Playtime, 4 Mic ENC...") == "Boult Astra Neo 70Hrs Playtime, 4 Mic ENC"
    assert clean_title("Redmi Watch 5 Lite, 1.96&quot; Amoled") == 'Redmi Watch 5 Lite, 1.96" Amoled'
    assert features_text("['8 GB RAM', '512 GB SSD']") == "8 GB RAM; 512 GB SSD"
    assert len(trim_text("word " * 200, 50)) <= 50
    assert is_range_price("₹299 - ₹599") and not is_range_price("₹1,299")
    assert REFURBISHED.search("Apple iPhone 13 (Renewed)")
    assert GIFT_OR_WARRANTY.search("Onsitego 2 Year Extended Warranty for Mobiles")
    assert not GIFT_OR_WARRANTY.search("Samsung Galaxy S25 with 1 year warranty")


def test_khanna_category_tree():
    assert khanna_category("Men's wear", "Foot Wear") == "footwear"
    assert khanna_category("Women's wear", "Ethnic Wear") == "fashion"
    assert khanna_category("Men's wear", "Men's Grooming") is None
    assert khanna_category("Home and Furniture", "Kitchen Storage") == "home_kitchen"
    assert khanna_category("Electronics", "Mobile Accessories") is None
    assert khanna_category("Bady and Kids", "Toys") is None


def _rows(**columns):
    n = len(columns["title"])
    base = {"product_id": [""] * n, "store": ["amazon_in"] * n, "scraped_at": ["2025-10-06"] * n,
            "category_source": ["rules"] * n, "mrp_inr": [np.nan] * n, "source_dataset": ["amazon_2025"] * n}
    base.update(columns)
    df = pd.DataFrame(base)
    df["text"] = df["title"]
    df["norm_title"] = df["title"].map(normalise_title)
    df["specs"] = df["title"].map(specs_key)
    return df


def test_group_variants_takes_median_and_keeps_ids():
    df = _rows(
        title=["Phone X (Black, 8GB RAM, 128GB Storage)", "Phone X (Blue, 8GB RAM, 128GB Storage)",
               "Phone X (Red, 8GB RAM, 128GB Storage)", "Phone X (Black, 12GB RAM, 256GB Storage)"],
        price_inr=[10000.0, 11000.0, 12000.0, 15000.0], brand=["x"] * 4, category=["smartphone"] * 4,
        product_id=["amazon_in:B1", "amazon_in:B2", "", "amazon_in:B4"],
    )
    out = group_variants(df).sort_values("price_inr")
    assert list(out["price_inr"]) == [11000.0, 15000.0]
    assert out.iloc[0]["alt_ids"] == "amazon_in:B1;amazon_in:B2"
    assert out.iloc[0]["n_variants"] == 3


def test_group_variants_merges_same_product_id_across_titles():
    df = _rows(title=["Boult Astra", "Boult Astra Neo"], price_inr=[999.0, 1099.0], brand=["boult"] * 2,
               category=["earbuds_headphones"] * 2, product_id=["flipkart:ABC", "flipkart:ABC"])
    assert len(group_variants(df)) == 1


def test_iqr_outliers_per_category():
    prices = [1000.0 + 10 * i for i in range(30)] + [500000.0]
    df = pd.DataFrame({"price_inr": prices, "category": ["earbuds_headphones"] * 31})
    assert iqr_outliers(df).tolist() == [False] * 30 + [True]


class FakeChat:
    """Answers with categories keyed by number, as the prompt asks."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0
        self.completions = self

    def create(self, **kwargs):
        import json
        from types import SimpleNamespace

        self.calls += 1
        content = json.dumps(self.answer)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def test_llm_categorizer_uses_keys_and_caches(tmp_path):
    chat = FakeChat({"1": "laptop", "0": "smartphone"})
    llm = LlmCategorizer(tmp_path / "cache.json", client=type("C", (), {"chat": chat})())
    found = llm.classify(["b title", "a title"])
    assert found == {"a title": "smartphone", "b title": "laptop"}
    assert llm.cost > 0
    again = LlmCategorizer(tmp_path / "cache.json", client=type("C", (), {"chat": FakeChat({})})())
    assert again.classify(["a title"]) == {"a title": "smartphone"} and again.calls == 0


def test_prepare_end_to_end_on_tiny_raw_folder(tmp_path):
    raw = tmp_path / "raw"
    (raw / "amazon_2026").mkdir(parents=True)
    pd.DataFrame({
        "asin": ["B0AAAAAAA1", "B0AAAAAAA2", "B0AAAAAAA3", "B0AAAAAAA4"],
        "product_title": ["Samsung Galaxy A17 5G (Blue, 6GB RAM, 128GB Storage)",
                          "Samsung Galaxy A17 5G (Black, 6GB RAM, 128GB Storage)",
                          "Perfume Combo (Pack of 2)", "LG 7 Kg 5 Star Front Load Washing Machine"],
        "category": ["smartphone", "smartphone", "smartphone", "washing machine"],
        "price_inr": [16999, 17499, 499, None], "original_price_inr": [19999, None, 999, 30000],
        "scraped_at": ["05-06-2026"] * 4,
    }).to_csv(raw / "amazon_2026" / "amazon_india_products_cleaned.csv", index=False)
    df, funnel, llm, _ = prepare(raw, use_llm=False)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["price_inr"] == 17249 and row["category"] == "smartphone"
    assert row["product_key"] == "amazon_in:B0AAAAAAA1" and row["scraped_at"] == "2026-06-05"
    assert funnel.steps["raw rows"]["amazon_2026"] == 4
    assert funnel.steps["price present"]["amazon_2026"] == 3
