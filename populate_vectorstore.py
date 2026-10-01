"""
Populate the Chroma vector store that the Frontier Agent's RAG search
and the 3D plot in price_is_right.py both read from.

products_vectorstore/ is not tracked in git, so this builds the
"products" collection from a HuggingFace dataset of Items.

Usage:
    python populate_vectorstore.py --dataset <hf-user>/<dataset-name>
    python populate_vectorstore.py --dataset <name> --limit 5000
"""

import argparse
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from deal_agent_framework import CATEGORIES, DealAgentFramework

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 1000


def description_for(item) -> str:
    """The text stored as the Chroma document, matching what FrontierAgent embeds."""
    text = item.full or item.summary or item.title
    # Items built by make_prompt carry the price in the tail; strip it so the
    # document is the product description only.
    return text.split("Price is $")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of items")
    parser.add_argument("--reset", action="store_true", help="Drop the collection first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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


if __name__ == "__main__":
    main()
