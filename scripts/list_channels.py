"""
Check that every configured Telegram channel resolves with your session.

    uv run python scripts/list_channels.py

Prints, for each channel in sources.yaml + sources.local.yaml: the resolved title, the
numeric id, whether the latest message could be read, and whether the public t.me/s/
fallback is available for it. Read-only: nothing is joined or sent.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from agents.config import get, load_sources_config  # noqa: E402
from agents.sources import telegram_channels  # noqa: E402
from agents.sources.telegram_mtproto import (  # noqa: E402
    TelegramUnavailable,
    connect_client,
    credentials_from_env,
    resolve_channel,
)


async def main() -> int:
    from telethon import utils

    load_dotenv(override=True)
    config = load_sources_config()
    channels = telegram_channels(config)
    try:
        client = await connect_client(
            get(config, "telegram.session_path", "secrets/bargain_spotter.session"), credentials_from_env()
        )
    except TelegramUnavailable as exc:
        print(f"MTProto unavailable: {exc}")
        return 1

    rows = []
    for channel in channels:
        status, title, peer_id, latest = "skipped (disabled)", "", "", ""
        if channel.enabled:
            try:
                entity = await resolve_channel(client, channel)
                title = getattr(entity, "title", "") or ""
                peer_id = str(utils.get_peer_id(entity))
                messages = await client.get_messages(entity, limit=1)
                latest = messages[0].date.strftime("%Y-%m-%d %H:%M") if messages else "no messages"
                status = "reachable"
            except Exception as exc:  # noqa: BLE001
                status = f"NOT reachable: {exc}"
            await asyncio.sleep(0.5)
        public = "yes" if channel.public_ok else "no (MTProto only)"
        rows.append((channel.key, channel.describe(), title, peer_id, latest, public, status))
    await client.disconnect()

    headers = ("key", "config", "title", "id", "latest post", "public fallback", "status")
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    for row in [headers] + rows:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))
    enabled = [r for r, c in zip(rows, channels) if c.enabled]
    bad = [r for r in enabled if r[-1] != "reachable"]
    print(f"\n{len(enabled) - len(bad)}/{len(enabled)} enabled channels reachable.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
