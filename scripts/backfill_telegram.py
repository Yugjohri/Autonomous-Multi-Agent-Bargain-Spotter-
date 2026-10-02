"""
Backfill Telegram history into data/observations.jsonl.

    uv run python scripts/backfill_telegram.py --days 90 --channels all
    uv run python scripts/backfill_telegram.py --days 30 --channels OMGDeals,DCLootsOffers --no-resolve

Every message goes through the same parser and normalizer as live scans and is
appended with its original post date. The run is:
  * resumable: per-channel progress lives in data/state/backfill_state.json, saved
    after every batch, so an interrupted run continues where it stopped;
  * idempotent: each message is one row keyed by its obs_id (telegram:<channel>/<id>),
    so reruns never add duplicate rows;
  * throttled: Telethon waits between history requests, channels are read one at a
    time, and FloodWait is honoured.
Short links are resolved (cached on disk, robots.txt respected) unless --no-resolve.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from agents.cache import DiskCache  # noqa: E402
from agents.config import get, load_sources_config  # noqa: E402
from agents.normalize import RedirectResolver, normalize_link  # noqa: E402
from agents.observations import ObservationLog, observation_row  # noqa: E402
from agents.sources import build_http, telegram_channels  # noqa: E402
from agents.sources.telegram_mtproto import (  # noqa: E402
    TelegramUnavailable,
    connect_client,
    credentials_from_env,
    parse_telethon_message,
    resolve_channel,
)

STATE_PATH = Path("data/state/backfill_state.json")
BATCH = 200

log = logging.getLogger("backfill")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


async def rows_for(messages, channel, resolver, seen_at):
    rows = []
    for message in messages:
        raw = parse_telethon_message(message, channel)
        if raw is None:
            continue
        link = None
        if raw.url:
            if resolver is not None and raw.priceable:
                link = await asyncio.to_thread(normalize_link, raw.url, resolver)
            else:
                link = normalize_link(raw.url)
        rows.append(observation_row(raw, link, seen_at))
    return rows


async def backfill_channel(client, channel, entity, days, log_file, resolver, state, wait_time):
    """
    Two passes per channel, both newest-to-oldest:
      1. messages newer than the newest id from earlier runs;
      2. older history from the oldest id reached so far, down to the cutoff date.
    Progress is committed only after the rows of a batch are written, and iterators are
    recreated from the committed position after a FloodWait, so nothing is skipped.
    """
    from telethon.errors import FloodWaitError

    key = channel.key
    progress = state.setdefault(key, {"newest_id": 0, "oldest_id": 0, "complete_since": None})
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    written = 0

    async def run_pass(make_iterator, keep, commit):
        """Iterate, writing batches; keep(message) False ends the pass; commit(last) saves progress."""
        nonlocal written
        while True:
            batch = []
            try:
                async for message in make_iterator():
                    if not keep(message):
                        break
                    batch.append(message)
                    if len(batch) >= BATCH:
                        written += log_file.append(await rows_for(batch, channel, resolver, datetime.now(timezone.utc)))
                        commit(batch)
                        save_state(state)
                        log.info(f"{key}: {written} new rows so far")
                        batch = []
                if batch:
                    written += log_file.append(await rows_for(batch, channel, resolver, datetime.now(timezone.utc)))
                    commit(batch)
                save_state(state)
                return
            except FloodWaitError as exc:
                if batch:
                    written += log_file.append(await rows_for(batch, channel, resolver, datetime.now(timezone.utc)))
                    commit(batch)
                save_state(state)
                if exc.seconds > 900:
                    raise
                log.warning(f"{key}: FloodWait {exc.seconds}s, sleeping, then resuming")
                await asyncio.sleep(exc.seconds + 1)

    # Pass 1: posts newer than the previous run.
    if progress["newest_id"]:
        floor = progress["newest_id"]
        cursor = {"offset": 0, "max": floor}

        def commit_new(batch):
            cursor["offset"] = batch[-1].id
            cursor["max"] = max(cursor["max"], max(m.id for m in batch))

        await run_pass(
            lambda: client.iter_messages(entity, min_id=floor, offset_id=cursor["offset"], wait_time=wait_time),
            lambda message: True,
            commit_new,
        )
        progress["newest_id"] = cursor["max"]
        save_state(state)

    # Pass 2: older history down to the cutoff.
    done_since = progress.get("complete_since")
    if done_since and datetime.fromisoformat(done_since) <= cutoff:
        return written

    def keep_old(message):
        return message.date >= cutoff

    def commit_old(batch):
        progress["oldest_id"] = min(m.id for m in batch)
        progress["newest_id"] = max(progress["newest_id"], max(m.id for m in batch))

    await run_pass(
        lambda: client.iter_messages(entity, offset_id=progress["oldest_id"], wait_time=wait_time),
        keep_old,
        commit_old,
    )
    progress["complete_since"] = cutoff.isoformat()
    save_state(state)
    return written


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--channels", default="all", help="'all' or comma-separated channel keys")
    parser.add_argument("--no-resolve", action="store_true", help="Do not resolve short links (faster)")
    parser.add_argument("--wait", type=float, default=1.0, help="Seconds between history requests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv(override=True)
    config = load_sources_config()
    channels = [c for c in telegram_channels(config) if c.enabled]
    if args.channels != "all":
        wanted = {name.strip().lower() for name in args.channels.split(",")}
        channels = [c for c in channels if c.key.lower() in wanted]
    if not channels:
        print("No matching enabled channels in sources.yaml / sources.local.yaml.")
        return 1

    try:
        client = await connect_client(
            get(config, "telegram.session_path", "secrets/bargain_spotter.session"), credentials_from_env()
        )
    except TelegramUnavailable as exc:
        print(f"MTProto unavailable: {exc}")
        return 1

    resolver = None
    if not args.no_resolve:
        resolver = RedirectResolver(build_http(config), DiskCache("redirects", ttl_seconds=90 * 24 * 3600))
    log_file = ObservationLog()
    state = load_state()
    total = 0
    try:
        for channel in channels:
            try:
                entity = await resolve_channel(client, channel)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Skipping {channel.key}: {exc}")
                continue
            log.info(f"Backfilling {channel.key} ({args.days} days)")
            total += await backfill_channel(client, channel, entity, args.days, log_file, resolver, state, args.wait)
            await asyncio.sleep(2)
    finally:
        save_state(state)
        if resolver:
            resolver.cache.flush()
        await client.disconnect()
    log.info(f"Done. {total} new observations written to {log_file.path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
