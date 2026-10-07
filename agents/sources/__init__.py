"""
Builds the configured deal sources from sources.yaml (+ sources.local.yaml).
"""

import logging
from typing import Any, Dict, List, Optional

from agents.cache import DiskCache
from agents.config import get
from agents.http import HttpClient
from agents.sources.base import DealSource, RawDeal
from agents.sources.telegram_channels import ChannelRef, parse_channels

logger = logging.getLogger(__name__)

__all__ = ["DealSource", "RawDeal", "build_http", "build_sources", "build_telegram_source", "telegram_channels"]


def build_http(config: Dict[str, Any], offline: bool = False) -> HttpClient:
    return HttpClient(
        user_agent=get(config, "http.user_agent") or HttpClient().user_agent,
        timeout=float(get(config, "http.timeout_seconds", 15)),
        retries=int(get(config, "http.retries", 2)),
        backoff=float(get(config, "http.backoff_seconds", 1.5)),
        min_interval=float(get(config, "http.min_interval_per_host_seconds", 1.0)),
        respect_robots=bool(get(config, "http.respect_robots_txt", True)),
        offline=offline,
        robots_cache=None if offline else DiskCache("robots", ttl_seconds=24 * 3600),
    )


def telegram_channels(config: Dict[str, Any]) -> List[ChannelRef]:
    return parse_channels(get(config, "telegram.channels", []))


def build_telegram_source(config: Dict[str, Any], http: HttpClient) -> Optional[DealSource]:
    from agents.sources.telegram_mtproto import TelegramMTProtoSource
    from agents.sources.telegram_public import TelegramPublicSource

    if not get(config, "telegram.enabled", True):
        return None
    channels = telegram_channels(config)
    if not any(c.enabled for c in channels):
        logger.info("No enabled Telegram channels; add them to sources.local.yaml")
        return None
    delay = float(get(config, "telegram.channel_delay_seconds", 1.5))
    public = TelegramPublicSource(channels, http, channel_delay=delay)
    if get(config, "telegram.public_only", False):
        return public
    return TelegramMTProtoSource(
        channels,
        fallback=public,
        session_path=get(config, "telegram.session_path", "secrets/bargain_spotter.session"),
        channel_delay=delay,
        max_flood_wait=int(get(config, "telegram.max_flood_wait_seconds", 300)),
    )


def build_rss_sources(config: Dict[str, Any], http: HttpClient) -> List[DealSource]:
    from agents.sources.rss import DealsMagnetSource, RSSSource

    sources: List[DealSource] = []
    cache = DiskCache("feeds", ttl_seconds=24 * 3600) if not http.offline else None
    for feed in get(config, "rss", []) or []:
        if not feed.get("enabled", False):
            continue
        if feed.get("name") == "legacy_us":
            # The US DealNews feeds are read by the legacy pipeline (PRICER_MODE=usd_legacy).
            continue
        if feed.get("kind") == "dealsmagnet" or feed.get("name") == "dealsmagnet":
            sources.append(DealsMagnetSource(http, url=feed.get("url", "https://www.dealsmagnet.com/feed"), cache=cache))
        else:
            sources.append(
                RSSSource(feed["name"], feed["url"], http, currency=feed.get("currency", "INR"),
                          store=feed.get("store", "other"), cache=cache)
            )
    return sources


def build_aggregator_sources(config: Dict[str, Any], http: HttpClient) -> List[DealSource]:
    from agents.sources.aggregator import AggregatorSource

    sources: List[DealSource] = []
    for name, site in (get(config, "aggregators", {}) or {}).items():
        if not site.get("enabled", False):
            continue
        if not site.get("selectors"):
            logger.warning(f"Aggregator {name} is enabled but has no selectors; skipping")
            continue
        sources.append(
            AggregatorSource(name, site["url"], http, site["selectors"], terms_checked=bool(site.get("terms_checked")))
        )
    return sources


def build_store_api_sources(config: Dict[str, Any], http: HttpClient) -> List[DealSource]:
    from agents.sources.store_api import AmazonCreatorsApiSource, FlipkartAffiliateApiSource

    classes = {"amazon_creators": AmazonCreatorsApiSource, "flipkart_affiliate": FlipkartAffiliateApiSource}
    return [cls() for key, cls in classes.items() if get(config, f"store_apis.{key}.enabled", False)]


def build_sources(config: Dict[str, Any], http: HttpClient) -> List[DealSource]:
    """Every enabled source. A source that fails to build is logged and left out."""
    sources: List[DealSource] = []
    builders = [build_telegram_source, build_rss_sources, build_aggregator_sources, build_store_api_sources]
    for builder in builders:
        try:
            built = builder(config, http)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not set up source via {builder.__name__}: {exc}")
            continue
        if built is None:
            continue
        sources.extend(built if isinstance(built, list) else [built])
    return sources
