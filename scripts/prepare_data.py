"""
Clean the raw Indian product datasets into one table for training the INR pricing models.

    uv run python scripts/prepare_data.py [--raw data/raw] [--out data/processed] [--no-llm]

Writes data/processed/products_inr.parquet with one row per product:
    product_key, title, text, brand, category, price_inr, mrp_inr, store, source_dataset,
    scraped_at, plus norm_title, alt_ids, n_variants and category_source.
and data/processed/funnel.json with the row counts dropped at each step.

Steps: parse prices (agents.money.parse_amount), drop missing, zero and range prices,
refurbished items, combos and multipacks (agents.price_signal.is_multipack), gift cards and
warranties; assign a category from the fixed taxonomy (agents/taxonomy.py rules, then
gpt-5-mini for rows the rules are unsure about, cached in data/cache/taxonomy_llm.json);
merge colour variants (same brand, normalised title and RAM/storage/size) into one row at
the median price; drop log-price outliers beyond 3x IQR within each category.
flipkart_khanna is about four years old, so only its fashion, footwear and home rows are kept.
"""

import argparse
import ast
import hashlib
import html
import json
import logging
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.categorize import guess_brand  # noqa: E402
from agents.money import parse_amount  # noqa: E402
from agents.normalize import canonical_id  # noqa: E402
from agents.price_signal import is_multipack  # noqa: E402
from agents.taxonomy import HINTS, TAXONOMY, classify  # noqa: E402
from agents.titles import normalise_title, specs_key, title_hash  # noqa: E402

COLUMNS = ["product_key", "title", "text", "brand", "category", "price_inr", "mrp_inr", "store",
           "source_dataset", "scraped_at"]
EXTRA_COLUMNS = ["norm_title", "alt_ids", "n_variants", "category_source"]

TEXT_CHARS = 400
REFURBISHED = re.compile(r"\b(refurbished|renewed|pre-?owned|open[- ]box)\b", re.IGNORECASE)
GIFT_OR_WARRANTY = re.compile(
    r"\b(gift ?cards?|e-?gift|gift vouchers?|vouchers?|extended warranty|warranty (plan|extension)|"
    r"protection plan|damage protection|care ?pack|applecare|onsitego|oneassist)\b",
    re.IGNORECASE,
)
# Datasets whose own category is more reliable than title rules: khanna has a full category
# tree, and each flipkart_2025 file is one search (phones, laptops, earphones) with titles
# cut at about 60 characters.
HINT_FIRST = ("flipkart_khanna", "flipkart_2025")
RANGE_PRICE = re.compile(r"\d\s*(-|–|to)\s*(₹|rs\.?)?\s*\d", re.IGNORECASE)


# ------------------------------------------------------------------ helpers


def clean_title(title) -> str:
    text = html.unescape(str(title or "")).replace("\xa0", " ")
    text = re.sub(r"\s*(\.\.\.|…)\s*$", "", text)  # Flipkart listing titles are cut with "..."
    return " ".join(text.split())


def trim_text(text: str, limit: int = TEXT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(" ")] if " " in cut else cut


def features_text(raw) -> str:
    """Flipkart 'Key Features' are stored as the text of a Python list."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        items = ast.literal_eval(raw)
        if isinstance(items, (list, tuple)):
            return "; ".join(str(i) for i in items)
    except (ValueError, SyntaxError):
        pass
    return raw


def is_range_price(raw) -> bool:
    return isinstance(raw, str) and bool(RANGE_PRICE.search(raw))


def brand_of(title: str) -> str:
    known = guess_brand(title)
    if known:
        return known.lower()
    first = re.sub(r"[^\w&'-]", "", (title.split() or [""])[0])
    return first.lower()


def khanna_category(c1: str, c2: str) -> Optional[str]:
    """Map flipkart_khanna's own category tree onto the taxonomy; None means out of scope."""
    c1, c2 = (c1 or "").strip().lower(), (c2 or "").strip().lower()
    if "foot" in c2:
        return "footwear"
    if c1 in ("men's wear", "women's wear"):
        if any(w in c2 for w in ("grooming", "beauty", "personal care")):
            return None
        return "fashion"
    if c1 == "bady and kids":
        if any(w in c2 for w in ("clothing", "winter wear", "sunglasses", "watches")):
            return "fashion"
        return None
    if c1 == "home and furniture":
        return None if "smart home" in c2 else "home_kitchen"
    return None


# ------------------------------------------------------------------ loaders


def load_amazon_2025(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(folder / "amazon_all_electronics_data.csv", encoding="utf-8-sig")
    # The search result URL carries the query time (qid, unix seconds): early October 2025.
    qid = pd.to_numeric(df["Product_URL"].str.extract(r"qid=(\d+)")[0], errors="coerce")
    when = pd.to_datetime(qid, unit="s", utc=True).dt.strftime("%Y-%m-%d").fillna("2025-10-06")
    return pd.DataFrame({
        "title": df["Product_Name"], "extra": "", "price_raw": df["Price"], "mrp_raw": None,
        "product_id": "amazon_in:" + df["ASIN"].astype(str).str.upper(), "store": "amazon_in",
        "scraped_at": when, "hint": None,
    })


def load_flipkart_2025(folder: Path) -> pd.DataFrame:
    hints = {"flipkart_earphones.csv": "earphones", "flipkart_laptops.csv": "laptop",
             "flipkart_mobile_data.csv": "mobile"}
    frames = []
    for path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        link = "Product URL" if "Product URL" in df.columns else "Product Link"
        urls = df[link].fillna("").str.replace("https://www.flipkart.comhttps://", "https://", regex=False)
        extra = df["Key Features"].map(features_text) if "Key Features" in df.columns else ""
        frames.append(pd.DataFrame({
            "title": df["Title"], "extra": extra, "price_raw": df["Price"], "mrp_raw": df["Original Price"],
            "product_id": urls.map(lambda u: canonical_id(u) if u else ""), "store": "flipkart",
            "scraped_at": pd.to_datetime(df["Timestamp"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "hint": hints.get(path.name),
        }))
    return pd.concat(frames, ignore_index=True)


def load_amazon_2026(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(folder / "amazon_india_products_cleaned.csv", encoding="utf-8-sig")
    return pd.DataFrame({
        "title": df["product_title"], "extra": "", "price_raw": df["price_inr"],
        "mrp_raw": df["original_price_inr"], "product_id": "amazon_in:" + df["asin"].astype(str).str.upper(),
        "store": "amazon_in",
        "scraped_at": pd.to_datetime(df["scraped_at"], format="%d-%m-%Y", errors="coerce").dt.strftime("%Y-%m-%d"),
        "hint": df["category"],
    })


def load_flipkart_khanna(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(folder / "dataset.csv", encoding="utf-8-sig")
    extra = (df["description"].fillna("") + " " + df["highlights"].fillna("")).str.strip()
    return pd.DataFrame({
        "title": df["title"], "extra": extra, "price_raw": df["selling_price"], "mrp_raw": df["mrp"],
        "product_id": "", "store": "flipkart", "scraped_at": None,
        "hint": [khanna_category(a, b) for a, b in zip(df["category_1"].fillna(""), df["category_2"].fillna(""))],
    })


LOADERS = OrderedDict([
    ("amazon_2025", load_amazon_2025),
    ("flipkart_2025", load_flipkart_2025),
    ("flipkart_khanna", load_flipkart_khanna),
    ("amazon_2026", load_amazon_2026),
])


# ---------------------------------------------------------- LLM categories


class LlmCategorizer:
    """gpt-5-mini for titles the rules are unsure about. Results are cached on disk by title."""

    BATCH = 25

    def __init__(self, cache_path: Path, model: str = "gpt-5-mini", prices=(0.25, 2.00), client=None):
        self.cache_path = cache_path
        self.cache: Dict[str, str] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.model = model
        self.prices = prices
        self.client = client
        self.cost = 0.0
        self.calls = 0

    @staticmethod
    def key(title: str) -> str:
        return hashlib.sha1(title.encode("utf-8")).hexdigest()

    def prompt(self, titles: List[str]) -> str:
        lines = "\n".join(f"{i}. {t[:200]}" for i, t in enumerate(titles))
        return (
            "Classify each Indian e-commerce product title into exactly one category from this list:\n"
            f"{', '.join(TAXONOMY)}.\n"
            "Accessories for a device (cases, covers, chargers, cables, straps, stands) are mobile_accessory, "
            "or computer_accessory when they are for laptops, PCs, tablets or printers. Storage, keyboards, mice, "
            "routers and printers are computer_accessory. Speakers, soundbars and microphones are audio_other. "
            "Use other when nothing fits (stationery, batteries, toys, consoles, tools).\n"
            'Reply with a JSON object keyed by the title number: {"0": "<category>", "1": "<category>", ...}\n\n'
            + lines
        )

    def classify(self, titles: List[str]) -> Dict[str, str]:
        todo = sorted({t for t in titles if self.key(t) not in self.cache})
        if todo and self.client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv(ROOT / ".env", override=True)
            self.client = OpenAI()
        for start in range(0, len(todo), self.BATCH):
            chunk = todo[start : start + self.BATCH]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": self.prompt(chunk)}],
                response_format={"type": "json_object"},
                reasoning_effort="minimal",
            )
            self.calls += 1
            usage = response.usage
            if usage is not None:
                self.cost += (usage.prompt_tokens * self.prices[0] + usage.completion_tokens * self.prices[1]) / 1e6
            # Keyed by number rather than a list, so a skipped title cannot shift the others.
            try:
                answer = json.loads(response.choices[0].message.content)
            except ValueError:
                answer = {}
            for i, title in enumerate(chunk):
                category = answer.get(str(i)) if isinstance(answer, dict) else None
                if category in TAXONOMY:
                    self.cache[self.key(title)] = category
            self.save()
            logging.info(f"LLM categories: {min(start + self.BATCH, len(todo))}/{len(todo)}, cost so far ${self.cost:.4f}")
        return {t: self.cache.get(self.key(t)) for t in titles}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=0, sort_keys=True), encoding="utf-8")


# ------------------------------------------------------------------ pipeline


class Funnel:
    """Row counts per dataset after each step, for the data report."""

    def __init__(self):
        self.steps: "OrderedDict[str, Dict[str, int]]" = OrderedDict()

    def record(self, step: str, df: pd.DataFrame) -> None:
        counts = df["source_dataset"].value_counts().to_dict()
        self.steps[step] = {k: int(counts.get(k, 0)) for k in LOADERS}

    def table(self) -> str:
        names = list(LOADERS)
        lines = ["| step | " + " | ".join(names) + " | total |", "|---" * (len(names) + 2) + "|"]
        for step, counts in self.steps.items():
            values = [counts[n] for n in names]
            lines.append(f"| {step} | " + " | ".join(f"{v:,}" for v in values) + f" | {sum(values):,} |")
        return "\n".join(lines)


def drop(df: pd.DataFrame, mask: pd.Series, step: str, funnel: Funnel) -> pd.DataFrame:
    df = df[~mask].copy()
    funnel.record(step, df)
    return df


def group_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per product within each dataset. Rows are the same product when they share a
    product id, or the same brand, normalised title and RAM/storage/size (colour variants).
    """
    df = df.reset_index(drop=True)
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in (
        df["source_dataset"] + "|" + df["product_id"].fillna(""),
        df["source_dataset"] + "|" + df["brand"] + "|" + df["norm_title"] + "|" + df["specs"],
    ):
        first: Dict[str, int] = {}
        for i, key in enumerate(keys):
            if key.endswith("|"):  # no product id
                continue
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    df["group"] = [find(i) for i in range(len(df))]

    rows = []
    for _, g in df.groupby("group", sort=False):
        price = float(g["price_inr"].median())
        # Keep the listing whose price is closest to the median as the representative.
        rep = g.iloc[int(np.argmin(np.abs(g["price_inr"].to_numpy() - price)))]
        ids = sorted({i for i in g["product_id"] if i})
        mrp = g["mrp_inr"].dropna()
        rows.append({
            "product_key": ids[0] if ids else f"title:{title_hash(rep['title'])}",
            "title": rep["title"], "text": rep["text"], "brand": rep["brand"], "category": rep["category"],
            "price_inr": round(price, 2), "mrp_inr": float(mrp.median()) if len(mrp) else np.nan,
            "store": rep["store"], "source_dataset": rep["source_dataset"], "scraped_at": rep["scraped_at"],
            "norm_title": rep["norm_title"], "alt_ids": ";".join(ids), "n_variants": len(g),
            "category_source": rep["category_source"],
        })
    return pd.DataFrame(rows)


def iqr_outliers(df: pd.DataFrame, k: float = 3.0, min_rows: int = 20) -> pd.Series:
    """True for rows whose log price is more than k IQRs outside the quartiles of their category."""
    log_price = np.log(df["price_inr"])
    mask = pd.Series(False, index=df.index)
    for _, idx in df.groupby("category").groups.items():
        values = log_price.loc[idx]
        if len(values) < min_rows:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        spread = q3 - q1
        mask.loc[idx] = (values < q1 - k * spread) | (values > q3 + k * spread)
    return mask


def prepare(raw: Path, use_llm: bool = True, cache: Optional[Path] = None, categorizer=None):
    funnel = Funnel()
    frames = []
    for name, loader in LOADERS.items():
        folder = raw / name
        if not folder.exists():
            logging.warning(f"{folder} is missing, skipped")
            continue
        part = loader(folder)
        part["source_dataset"] = name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["title"] = df["title"].map(clean_title)
    df["product_id"] = df["product_id"].fillna("").astype(str)
    df.loc[~df["product_id"].str.contains(":"), "product_id"] = ""
    funnel.record("raw rows", df)

    df = drop(df, df["title"].str.len() == 0, "has a title", funnel)
    ranged = df["price_raw"].map(is_range_price)
    df["price_inr"] = df["price_raw"].map(lambda v: parse_amount(v) if pd.notna(v) else None).astype(float)
    df["mrp_inr"] = df["mrp_raw"].map(lambda v: parse_amount(v) if pd.notna(v) else None).astype(float)
    df = drop(df, df["price_inr"].isna(), "price present", funnel)
    df = drop(df, df["price_inr"] <= 0, "price above zero", funnel)
    df = drop(df, ranged.loc[df.index], "single price (no ranges)", funnel)
    df.loc[df["mrp_inr"] < df["price_inr"], "mrp_inr"] = np.nan
    df = drop(df, df["title"].map(lambda t: bool(REFURBISHED.search(t))), "not refurbished or renewed", funnel)
    df = drop(df, df["title"].map(is_multipack), "not a combo or multipack", funnel)
    df = drop(df, df["title"].map(lambda t: bool(GIFT_OR_WARRANTY.search(t))), "not a gift card or warranty", funnel)
    khanna = df["source_dataset"] == "flipkart_khanna"
    df = drop(df, khanna & df["hint"].isna(), "flipkart_khanna: fashion, footwear, home only", funnel)

    # Categories: the khanna tree for khanna rows, rules elsewhere, the LLM for unsure rows.
    labels = [
        (HINTS[hint], True, "dataset") if dataset in HINT_FIRST and hint in HINTS
        else (*classify(title, hint), "rules")
        for title, hint, dataset in zip(df["title"], df["hint"], df["source_dataset"])
    ]
    df["category"] = [c for c, _, _ in labels]
    df["category_source"] = [s for _, _, s in labels]
    unsure = ~pd.Series([sure for _, sure, _ in labels], index=df.index)
    llm_stats = {"unsure_rows": int(unsure.sum()), "calls": 0, "cost_usd": 0.0}
    if use_llm and unsure.any():
        categorizer = categorizer or LlmCategorizer(cache or ROOT / "data" / "cache" / "taxonomy_llm.json")
        found = categorizer.classify(df.loc[unsure, "title"].tolist())
        for idx in df.index[unsure]:
            category = found.get(df.at[idx, "title"])
            if category:
                df.at[idx, "category"] = category
                df.at[idx, "category_source"] = "llm"
        llm_stats.update(calls=categorizer.calls, cost_usd=round(categorizer.cost, 4))

    df["text"] = [trim_text(f"{t} | {e}" if isinstance(e, str) and e.strip() else t) for t, e in zip(df["title"], df["extra"])]
    df["brand"] = df["title"].map(brand_of)
    df["norm_title"] = df["title"].map(normalise_title)
    df["specs"] = df["title"].map(specs_key)
    df = group_variants(df)
    funnel.record("one row per product (colour variants merged)", df)

    outliers = iqr_outliers(df)
    dropped_outliers = df[outliers]
    df = drop(df, outliers, "log price within 3x IQR of its category", funnel)
    return df[COLUMNS + EXTRA_COLUMNS].reset_index(drop=True), funnel, llm_stats, dropped_outliers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw", help="folder with one subfolder per dataset")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--cache", default="data/cache/taxonomy_llm.json", help="LLM category cache")
    parser.add_argument("--no-llm", action="store_true", help="rules only; unsure rows keep the rule category")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    df, funnel, llm_stats, outliers = prepare(Path(args.raw), use_llm=not args.no_llm, cache=Path(args.cache))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "products_inr.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "funnel": funnel.steps,
        "llm": llm_stats,
        "categories": {d: g["category"].value_counts().to_dict() for d, g in df.groupby("source_dataset")},
        "category_source": df["category_source"].value_counts().to_dict(),
        "outliers": outliers[["source_dataset", "category", "price_inr", "title"]].to_dict("records"),
    }
    (out / "funnel.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(funnel.table())
    print(f"\nLLM categories: {llm_stats}")
    print(f"Category source: {dict(Counter(df['category_source']))}")
    print(f"Wrote {len(df):,} products to {out / 'products_inr.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
