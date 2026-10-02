"""
Populate the Chroma vector store used for RAG and for the 3D plot in price_is_right.py.

USD (legacy) "products" collection, from a HuggingFace dataset of Items:
    python populate_vectorstore.py --dataset <hf-user>/<dataset-name> [--limit 5000] [--reset]

INR "products_inr" collection (PRICER_MODE=inr), from data we collected ourselves:
    python populate_vectorstore.py --currency INR --from-memory
    python populate_vectorstore.py --currency INR --from-observations
    python populate_vectorstore.py --currency INR --from-raw data/raw

--from-raw reads local CSV files of Indian product listings (Amazon.in, Flipkart, ...).
Columns are detected by name (title / price / MRP / URL / ASIN / timestamp), prices may
be written with the rupee sign and commas or as plain numbers, and rows marked with a currency other than INR are
skipped. products_vectorstore/ is not tracked in git, so this builds it locally.
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from deal_agent_framework import CATEGORIES, DealAgentFramework

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 1000

TITLE_COLUMNS = ("product_title", "product_name", "title", "name")
PRICE_COLUMNS = ("price_inr", "selling_price", "product_price", "price", "discounted_price")
MRP_COLUMNS = ("original_price_inr", "product_original_price", "original price", "mrp", "actual_price")
URL_COLUMNS = ("product_url", "product url", "product link", "url", "link")
TIME_COLUMNS = ("scraped_at", "timestamp", "date")
CATEGORY_COLUMNS = ("category", "category_2", "category_1", "main_category")


def description_for(item) -> str:
    """The text stored as the Chroma document, matching what FrontierAgent embeds."""
    text = item.full or item.summary or item.title
    # Items built by make_prompt carry the price in the tail; strip it so the
    # document is the product description only.
    return text.split("Price is $")[0].strip()


def populate_usd(args) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    logging.info(f"Loading {args.dataset} split={args.split}")
    rows = load_dataset(args.dataset, split=args.split)
    if args.limit:
        rows = rows.select(range(min(args.limit, len(rows))))
    logging.info(f"Loaded {len(rows)} items")

    client = chromadb.PersistentClient(path=DealAgentFramework.DB)
    if args.reset:
        try:
            client.delete_collection("products")
            logging.info("Deleted existing collection")
        except Exception:
            pass
    collection = client.get_or_create_collection("products")

    model = SentenceTransformer(ENCODER)

    skipped = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        keys = list(chunk.keys())
        records = [dict(zip(keys, vals)) for vals in zip(*chunk.values())]

        docs, metas, ids = [], [], []
        for offset, rec in enumerate(records):
            category = rec.get("category")
            price = rec.get("price")
            # get_plot_data indexes CATEGORIES to pick a color, so anything
            # outside that list would crash the plot.
            if category not in CATEGORIES or price is None:
                skipped += 1
                continue
            text = (rec.get("full") or rec.get("summary") or rec.get("title") or "")
            text = text.split("Price is $")[0].strip()
            if not text:
                skipped += 1
                continue
            docs.append(text)
            metas.append({"category": category, "price": float(price)})
            ids.append(f"doc_{start + offset}")

        if not docs:
            continue
        vectors = model.encode(docs, show_progress_bar=False).astype(float).tolist()
        collection.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
        logging.info(f"Added {start + len(docs)}/{len(rows)} (skipped {skipped})")

    logging.info(f"Done. Collection now holds {collection.count()} documents.")


# ----------------------------------------------------------------------- INR


def _pick(row: dict, names: Iterable[str]) -> Optional[str]:
    lowered = {k.strip().lower().lstrip("﻿"): v for k, v in row.items() if k}
    for name in names:
        value = lowered.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def items_from_csv(path: Path):
    from agents.categorize import guess_category
    from agents.inr_store import InrItem
    from agents.money import parse_amount
    from agents.normalize import canonical_id, detect_store

    fallback_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as file:
        for row in csv.DictReader(file):
            currency = _pick(row, ("currency",))
            if currency and currency.upper() != "INR":
                continue
            title = _pick(row, TITLE_COLUMNS)
            price = parse_amount(_pick(row, PRICE_COLUMNS))
            if not title or not price or price <= 0:
                continue
            url = _pick(row, URL_COLUMNS) or ""
            url = url.replace("https://www.flipkart.comhttps://", "https://")
            asin = _pick(row, ("asin",))
            if asin and not url:
                url = f"https://www.amazon.in/dp/{asin}"
            mrp = parse_amount(_pick(row, MRP_COLUMNS))
            yield InrItem(
                document=title.strip()[:300],
                price=price,
                canonical_id=canonical_id(url) if url else "",
                mrp=mrp if mrp and mrp > price else None,
                store=detect_store(url) if url else "other",
                category=guess_category(f"{title} {_pick(row, CATEGORY_COLUMNS) or ''}"),
                seen_at=_pick(row, TIME_COLUMNS) or fallback_time,
                source=f"dataset:{path.parent.name}/{path.name}",
            )


def items_from_memory():
    from agents.inr_store import InrItem

    for opp in DealAgentFramework.load_memory_file():
        deal = opp.deal
        if deal.currency != "INR":
            continue
        yield InrItem(
            document=f"{deal.title}\n{deal.product_description}".strip(),
            price=deal.price,
            canonical_id=deal.canonical_id,
            mrp=deal.mrp,
            store=deal.store,
            category=deal.category,
            seen_at=deal.posted_at.isoformat() if deal.posted_at else None,
            source=deal.source,
        )


def items_from_observations(path: Path):
    from agents.inr_store import InrItem

    if not path.exists():
        return
    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if not row.get("is_deal") or not row.get("price_inr") or row.get("store") in (None, "other"):
                continue
            yield InrItem(
                document=f"{row.get('title') or ''}\n{(row.get('description') or '')[:300]}".strip(),
                price=row["price_inr"],
                canonical_id=row.get("canonical_id") or "",
                mrp=row.get("mrp_inr"),
                store=row.get("store") or "other",
                category=row.get("category") or "Other",
                seen_at=row.get("posted_at") or row.get("seen_at"),
                source=row.get("source") or "",
            )


def populate_inr(args) -> None:
    from agents.inr_store import COLLECTION, InrProductStore

    store = InrProductStore(path=DealAgentFramework.DB)
    if args.reset:
        store.collection._client.delete_collection(COLLECTION)
        store = InrProductStore(path=DealAgentFramework.DB)
        logging.info(f"Deleted existing {COLLECTION} collection")

    batches: List = []
    if args.from_memory:
        batches.append(("memory.json", list(items_from_memory())))
    if args.from_observations:
        batches.append(("observations", list(items_from_observations(Path(args.observations)))))
    if args.from_raw:
        excluded = set(args.exclude or [])
        for path in sorted(Path(args.from_raw).rglob("*.csv")):
            if excluded & set(path.relative_to(args.from_raw).parts):
                logging.info(f"{path}: skipped (--exclude)")
                continue
            batches.append((str(path), list(items_from_csv(path))))

    for name, items in batches:
        if args.limit:
            items = items[: args.limit]
        added = store.add(items)
        logging.info(f"{name}: added {added} INR items")
    logging.info(f"Done. {COLLECTION} now holds {store.count()} items.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--currency", choices=["USD", "INR"], default="USD")
    parser.add_argument("--dataset", help="HuggingFace dataset name (USD)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of items (per input for INR)")
    parser.add_argument("--reset", action="store_true", help="Drop the collection first")
    parser.add_argument("--from-memory", action="store_true", help="INR: deals saved in memory.json")
    parser.add_argument("--from-observations", action="store_true", help="INR: data/observations.jsonl")
    parser.add_argument("--observations", default="data/observations.jsonl")
    parser.add_argument("--from-raw", metavar="DIR", help="INR: folder of CSV product listings")
    parser.add_argument(
        "--exclude", nargs="*", metavar="FOLDER",
        help="INR --from-raw: subfolders to skip, e.g. a held-out test set (amazon_2026)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.currency == "USD":
        if not args.dataset:
            parser.error("--dataset is required for --currency USD")
        populate_usd(args)
    else:
        if not (args.from_memory or args.from_observations or args.from_raw):
            parser.error("--currency INR needs --from-memory, --from-observations and/or --from-raw")
        populate_inr(args)


if __name__ == "__main__":
    sys.exit(main())
