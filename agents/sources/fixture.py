"""
FixtureSource replays saved Telegram posts from tests/fixtures, re-dated to "a few
minutes ago" so the freshness window keeps them. Used by --fixture runs and tests:
the whole pipeline works offline with it.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from agents.sources.base import DealSource, RawDeal
from agents.sources.telegram_parser import parse_post

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
POSTS_FILE = FIXTURE_DIR / "telegram" / "posts.json"
REDIRECTS_FILE = FIXTURE_DIR / "redirects.json"


def load_fixture_redirects() -> dict:
    """Short link -> resolved url, recorded from real redirect chains."""
    if REDIRECTS_FILE.exists():
        return json.loads(REDIRECTS_FILE.read_text(encoding="utf-8"))
    return {}


class FixtureSource(DealSource):
    def __init__(self, path: Path = POSTS_FILE, spacing_minutes: int = 6):
        super().__init__("fixture")
        self.path = Path(path)
        self.spacing = spacing_minutes

    def _fetch(self, limit: int) -> List[RawDeal]:
        posts = json.loads(self.path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc)
        deals = []
        for i, post in enumerate(posts[:limit]):
            deal = parse_post(
                post["text"],
                post["channel"],
                post["message_id"],
                now - timedelta(minutes=3 + i * self.spacing),
                post.get("links", []),
            )
            if deal:
                deals.append(deal)
        return deals
