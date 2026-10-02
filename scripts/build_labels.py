"""
Turn data/observations.jsonl into one labelled row per product, for model training.

    uv run python scripts/build_labels.py [--in data/observations.jsonl] [--out data/labels.csv] [--min-count 1]

Rows are grouped by canonical_id (amazon_in:<ASIN>, flipkart:<ITM id>, ...). Only priced
deal posts count. For each product the output has the median and 75th percentile price,
the number of observations, first and last seen dates (post date when known), plus the
lowest price, latest MRP, highest past price, and how many distinct sources posted it.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIELDS = [
    "canonical_id", "title", "store", "category", "brand", "median_price", "p75_price", "min_price",
    "observations", "sources", "first_seen", "last_seen", "mrp", "highest_price",
]


def p75(values: List[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=4, method="inclusive")[2]


def read_rows(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def recanonicalize(rows: Iterable[dict], resolved: Dict[str, str]) -> Iterable[dict]:
    """
    Swap in the real product id for rows whose short link was resolved later
    (scripts/resolve_links.py). The observation log itself is never rewritten.
    """
    from agents.normalize import canonical_id, detect_store

    for row in rows:
        target = resolved.get(row.get("url") or "")
        if target:
            row = dict(row)
            row["canonical_id"] = canonical_id(target)
            store = detect_store(target)
            if store != "other":
                row["store"] = store
        yield row


def load_resolved() -> Dict[str, str]:
    """Short link -> resolved url, from the redirect cache in .cache/."""
    from agents.cache import DiskCache

    cache = DiskCache("redirects", ttl_seconds=365 * 24 * 3600)
    return {short: target for short, target in cache.items() if target and target != short}


MIN_PRICE = 30.0  # below this a "price" is a booking amount, token deal or misparse


def build_labels(rows: Iterable[dict], min_count: int = 1, max_spread: float = 3.0, report: dict = None) -> List[Dict]:
    """
    One row per product. Groups whose 75th percentile price is more than `max_spread` times
    their lowest price are dropped: they mix a product with a booking amount, a category page
    or a misread number (e.g. "Bajaj Pulsar" at Rs 200). Counts go into `report` if given.
    """
    report = report if report is not None else {}
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("is_deal") or not row.get("canonical_id") or not row.get("price_inr"):
            continue
        if float(row["price_inr"]) < MIN_PRICE:
            report["below_min_price"] = report.get("below_min_price", 0) + 1
            continue
        groups[row["canonical_id"]].append(row)

    labels = []
    for canonical, items in groups.items():
        if len(items) < min_count:
            continue
        items.sort(key=lambda r: r.get("posted_at") or r.get("seen_at") or "")
        prices = sorted(float(r["price_inr"]) for r in items)
        if max_spread and p75(prices) > max_spread * prices[0]:
            report["inconsistent_groups"] = report.get("inconsistent_groups", 0) + 1
            continue
        dates = [r.get("posted_at") or r.get("seen_at") for r in items if r.get("posted_at") or r.get("seen_at")]
        titles = Counter(r.get("title") for r in items if r.get("title"))
        mrps = [r["mrp_inr"] for r in items if r.get("mrp_inr")]
        highs = [r["highest_price_inr"] for r in items if r.get("highest_price_inr")]
        latest = items[-1]
        labels.append(
            {
                "canonical_id": canonical,
                "title": titles.most_common(1)[0][0] if titles else "",
                "store": latest.get("store"),
                "category": latest.get("category"),
                "brand": latest.get("brand"),
                "median_price": round(statistics.median(prices), 2),
                "p75_price": round(p75(prices), 2),
                "min_price": prices[0],
                "observations": len(items),
                "sources": len({r.get("source") for r in items}),
                "first_seen": min(dates) if dates else "",
                "last_seen": max(dates) if dates else "",
                "mrp": mrps[-1] if mrps else "",
                "highest_price": max(highs) if highs else "",
            }
        )
    labels.sort(key=lambda r: (-r["observations"], r["canonical_id"]))
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="source", default="data/observations.jsonl")
    parser.add_argument("--out", default="data/labels.csv")
    parser.add_argument("--min-count", type=int, default=1, help="Skip products seen fewer times")
    parser.add_argument("--max-spread", type=float, default=3.0,
                        help="Drop products whose p75 price exceeds this multiple of their lowest price (0 = keep all)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"{source} does not exist yet. Run the app or scripts/backfill_telegram.py first.")
        return 1
    resolved = load_resolved()
    report: dict = {}
    labels = build_labels(recanonicalize(read_rows(source), resolved), args.min_count, args.max_spread, report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(labels)
    real = [l for l in labels if l["canonical_id"].startswith(("amazon_in:B", "flipkart:ITM"))]
    print(f"Wrote {len(labels)} products to {out} (using {len(resolved)} resolved short links)")
    print(f"  with a real Amazon/Flipkart id: {len(real)}")
    print(f"  seen 2+ times: {sum(l['observations'] >= 2 for l in labels)} "
          f"({sum(l['observations'] >= 2 for l in real)} with a real id)")
    print(f"  dropped: {report.get('below_min_price', 0)} posts under Rs {MIN_PRICE:.0f}, "
          f"{report.get('inconsistent_groups', 0)} products with inconsistent prices (--max-spread {args.max_spread:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
