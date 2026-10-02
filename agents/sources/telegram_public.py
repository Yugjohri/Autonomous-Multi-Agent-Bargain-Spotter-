"""
TelegramPublicSource reads the public web preview at https://t.me/s/<channel>.
No login and no personal account are involved, which makes it the right source
for a hosted public demo and the automatic fallback when MTProto is unavailable.
Only channels with a username have a web preview.
"""

import logging
import re
import time
from datetime import datetime
from typing import List, Optional

from bs4 import BeautifulSoup

from agents.http import HttpClient
from agents.sources.base import DealSource, RawDeal
from agents.sources.telegram_channels import ChannelRef
from agents.sources.telegram_parser import parse_post

logger = logging.getLogger(__name__)

PREVIEW_URL = "https://t.me/s/{username}"


def _message_text(node) -> str:
    """Text of a message bubble with <br> turned into newlines and emoji kept inline."""
    if node is None:
        return ""
    for br in node.find_all("br"):
        br.replace_with("\n")
    return node.get_text("")


def parse_preview_html(html: str, channel: ChannelRef) -> List[RawDeal]:
    soup = BeautifulSoup(html, "html.parser")
    deals = []
    for message in soup.select("div.tgme_widget_message[data-post]"):
        post = message.get("data-post", "")
        match = re.search(r"/(\d+)$", post)
        message_id = int(match.group(1)) if match else None
        text_node = message.select_one(".tgme_widget_message_text")
        links = [a.get("href") for a in text_node.select("a[href]")] if text_node else []
        preview = message.select_one("a.tgme_widget_message_link_preview[href]")
        if preview:
            links.append(preview.get("href"))
        time_node = message.select_one(".tgme_widget_message_date time[datetime]")
        posted_at = datetime.fromisoformat(time_node["datetime"]) if time_node else None
        deal = parse_post(
            _message_text(text_node),
            channel=channel.key,
            message_id=message_id,
            posted_at=posted_at,
            extra_links=links,
            forwarded=message.select_one(".tgme_widget_message_forwarded_from") is not None,
            is_poll=message.select_one(".tgme_widget_message_poll") is not None,
            trust=channel.trust,
        )
        if deal:
            deals.append(deal)
    return deals


class TelegramPublicSource(DealSource):
    def __init__(self, channels: List[ChannelRef], http: HttpClient, channel_delay: float = 1.0):
        super().__init__("telegram_public", http)
        self.channels = [c for c in channels if c.enabled and c.public_ok]
        self.skipped = [c for c in channels if c.enabled and not c.public_ok]
        self.channel_delay = channel_delay
        for channel in self.skipped:
            logger.info(f"Telegram channel {channel.describe()} has no username; it is MTProto-only")

    def fetch_channel(self, channel: ChannelRef, limit: int = 20) -> List[RawDeal]:
        """Latest posts of one channel, following ?before= pages until `limit` posts."""
        deals: List[RawDeal] = []
        url: Optional[str] = PREVIEW_URL.format(username=channel.username)
        while url and len(deals) < limit:
            response = self.http.get(url)
            response.raise_for_status()
            page = parse_preview_html(response.text, channel)
            if not page:
                break
            deals = page + deals
            oldest = min(int(d.external_id.rsplit("/", 1)[1]) for d in page if d.external_id)
            url = f"{PREVIEW_URL.format(username=channel.username)}?before={oldest}" if len(deals) < limit else None
        deals.sort(key=lambda d: d.posted_at or datetime.min, reverse=True)
        return deals[:limit]

    def _fetch(self, limit: int) -> List[RawDeal]:
        deals: List[RawDeal] = []
        for i, channel in enumerate(self.channels):
            if i:
                time.sleep(self.channel_delay)
            try:
                deals.extend(self.fetch_channel(channel, limit))
            except Exception as exc:  # noqa: BLE001 - one channel must not stop the others
                logger.warning(f"Telegram preview for @{channel.username} failed: {exc}")
        return deals
