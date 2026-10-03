"""
Resolve the short links stored in data/observations.jsonl, so build_labels.py can group
posts by their real product id (amazon_in:<ASIN>, flipkart:<ITM id>).

    uv run python scripts/resolve_links.py                 # resolve everything not yet resolved
    uv run python scripts/resolve_links.py --limit 500     # a quick partial run

Useful after `backfill_telegram.py --no-resolve`. Hosts are resolved in parallel, but each
host still gets at most one request per --min-interval seconds, robots.txt is respected,
and no store page is ever requested. Results go to the redirect cache in .cache/, so the
run can be stopped with Ctrl+C and resumed, and the observation log is never rewritten.
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.cache import DiskCache  # noqa: E402
from agents.config import load_sources_config  # noqa: E402
from agents.normalize import RedirectResolver, canonical_id, host_of, is_product_id, is_store_url  # noqa: E402
from agents.sources import build_http  # noqa: E402

log = logging.getLogger("resolve")

CACHE_TTL = 365 * 24 * 3600


def urls_to_resolve(path: Path, resolver: RedirectResolver) -> dict:
    """Unique short links of priced posts that do not have a real product id yet, by host."""
    by_host = defaultdict(set)
    with open(path, encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            url = row.get("url") or ""
            if not row.get("is_deal") or not url or is_store_url(url):
                continue
            if resolver.should_resolve(url) and resolver.cache.get(url) is None:
                by_host[host_of(url)].add(url)
    return by_host


def resolve_host(resolver: RedirectResolver, host: str, urls, limit: int) -> tuple:
    done = real = 0
    for url in sorted(urls)[: limit or None]:
        final = resolver.resolve(url)
        done += 1
        if is_product_id(canonical_id(final)):
            real += 1
        if done % 200 == 0:
            resolver.cache.flush()
            log.info(f"{host}: {done}/{len(urls)} resolved, {real} to a real product id")
    resolver.cache.flush()
    return host, done, real


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="source", default="data/observations.jsonl")
    parser.add_argument("--min-interval", type=float, default=1.0, help="Seconds between requests to one host")
    parser.add_argument("--limit", type=int, default=0, help="Resolve at most this many links per host")
    parser.add_argument("--workers", type=int, default=8, help="Hosts resolved in parallel")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    for noisy in ("urllib3", "agents.normalize"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    source = Path(args.source)
    if not source.exists():
        print(f"{source} does not exist yet.")
        return 1

    http = build_http(load_sources_config())
    http.min_interval = args.min_interval
    resolver = RedirectResolver(http, DiskCache("redirects", ttl_seconds=CACHE_TTL))
    by_host = urls_to_resolve(source, resolver)
    total = sum(len(v) for v in by_host.values())
    if not total:
        print("Nothing left to resolve.")
        return 0
    slowest = max(len(v) for v in by_host.values())
    log.info(f"{total} links on {len(by_host)} hosts; roughly {slowest * args.min_interval / 60:.0f}+ minutes "
             "(the busiest host sets the pace)")
    for host, urls in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
        log.info(f"  {host}: {len(urls)}")

    start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(resolve_host, resolver, h, u, args.limit) for h, u in by_host.items()]
            for future in as_completed(futures):
                host, done, real = future.result()
                log.info(f"{host} finished: {done} resolved, {real} to a real product id")
    except KeyboardInterrupt:
        log.info("Stopped; progress is saved in the cache, rerun to continue")
    finally:
        resolver.cache.flush()
    log.info(f"Done in {(time.monotonic() - start) / 60:.0f} min. Now run: uv run python scripts/build_labels.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
