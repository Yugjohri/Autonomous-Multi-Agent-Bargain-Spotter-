"""
TelegramMTProtoSource reads channels through Telethon, logged in as your own account.

Read-only by design: this module only connects, resolves channels, reads messages and
listens for new ones. It never sends, forwards, reacts, or joins anything. Invite links
are checked with messages.checkChatInvite, which does not join.

Telethon is asyncio based, so the client lives on a dedicated event-loop thread.
Polling (fetch) and live updates (events.NewMessage) both run on that loop.

Without credentials or a session file this source falls back to TelegramPublicSource
(the t.me/s/ web preview). The session is created once, interactively, by
scripts/telegram_login.py; this module never prompts and never creates a session.
"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from agents.sources.base import DealSource, RawDeal
from agents.sources.telegram_channels import ChannelRef
from agents.sources.telegram_parser import parse_post
from agents.sources.telegram_public import TelegramPublicSource

logger = logging.getLogger(__name__)

DEFAULT_SESSION = "secrets/bargain_spotter.session"


class TelegramUnavailable(Exception):
    """MTProto cannot be used (missing keys, missing session, not authorized)."""


class ChannelNotJoined(Exception):
    pass


def session_base(path: str) -> str:
    """Telethon wants the session path without the .session suffix."""
    return path[: -len(".session")] if path.endswith(".session") else path


def credentials_from_env() -> Optional[tuple]:
    api_id, api_hash = os.getenv("TG_API_ID", "").strip(), os.getenv("TG_API_HASH", "").strip()
    if not api_id or not api_hash or not api_id.isdigit():
        return None
    return int(api_id), api_hash


def links_of_message(message) -> List[str]:
    """Hidden links (text entities) and URL buttons of a Telethon message."""
    links = []
    for entity in getattr(message, "entities", None) or []:
        url = getattr(entity, "url", None)
        if url:
            links.append(url)
    markup = getattr(message, "reply_markup", None)
    for row in getattr(markup, "rows", None) or []:
        for button in getattr(row, "buttons", None) or []:
            url = getattr(button, "url", None)
            if url:
                links.append(url)
    return links


def parse_telethon_message(message, channel: ChannelRef) -> Optional[RawDeal]:
    return parse_post(
        getattr(message, "message", None) or "",
        channel=channel.key,
        message_id=getattr(message, "id", None),
        posted_at=getattr(message, "date", None),
        extra_links=links_of_message(message),
        forwarded=getattr(message, "fwd_from", None) is not None,
        is_poll=getattr(message, "poll", None) is not None,
        trust=channel.trust,
    )


async def connect_client(session_path: str, credentials: Optional[tuple]):
    """
    Connect with an existing, logged-in session. Never prompts and never creates a
    session: run scripts/telegram_login.py once for that.
    """
    from telethon import TelegramClient

    if credentials is None:
        raise TelegramUnavailable("TG_API_ID / TG_API_HASH are not set")
    if not Path(session_path).exists():
        raise TelegramUnavailable(
            f"No Telegram session at {session_path}. Run: uv run python scripts/telegram_login.py"
        )
    api_id, api_hash = credentials
    client = TelegramClient(session_base(session_path), api_id, api_hash, flood_sleep_threshold=60)
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        raise TelegramUnavailable("Telegram session is not logged in. Run scripts/telegram_login.py")
    # Loading dialogs once caches access hashes, so numeric ids and private chats resolve.
    await client.get_dialogs()
    return client


async def resolve_channel(client, channel: ChannelRef):
    """Resolve a channel reference to a Telethon entity without joining anything."""
    from telethon.tl.functions.messages import CheckChatInviteRequest

    if channel.username:
        return await client.get_entity(channel.username)
    if channel.id is not None:
        try:
            return await client.get_entity(channel.id)
        except ValueError as exc:
            raise ChannelNotJoined(
                f"Channel id {channel.id} is not in your dialogs. Join it in the Telegram app first."
            ) from exc
    if channel.invite_hash:
        result = await client(CheckChatInviteRequest(channel.invite_hash))
        if type(result).__name__ == "ChatInviteAlready":
            return result.chat
        raise ChannelNotJoined(
            f"Invite channel {channel.key} is not joined. Join it in the Telegram app; "
            "this tool never joins channels by itself."
        )
    raise ValueError("channel needs a username, id or invite")


class TelegramMTProtoSource(DealSource):
    def __init__(
        self,
        channels: List[ChannelRef],
        fallback: Optional[TelegramPublicSource] = None,
        session_path: str = DEFAULT_SESSION,
        credentials: Optional[tuple] = None,
        channel_delay: float = 1.5,
        max_flood_wait: int = 300,
        request_timeout: float = 90,
    ):
        super().__init__("telegram_mtproto")
        self.channels = [c for c in channels if c.enabled]
        self.fallback = fallback
        self.session_path = session_path
        self.credentials = credentials if credentials is not None else credentials_from_env()
        self.channel_delay = channel_delay
        self.max_flood_wait = max_flood_wait
        self.request_timeout = request_timeout
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._entities: Dict[str, object] = {}
        self._by_peer_id: Dict[int, ChannelRef] = {}
        self.unresolved: Dict[str, str] = {}
        self.using_fallback = False

    # -------------------------------------------------------------- lifecycle

    def available(self) -> bool:
        return self.credentials is not None and Path(self.session_path).exists()

    def _run(self, coro, timeout: Optional[float] = None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout or self.request_timeout)

    def ensure_started(self) -> None:
        with self._lock:
            if self._client is not None:
                return
            if self.credentials is None:
                raise TelegramUnavailable("TG_API_ID / TG_API_HASH are not set")
            if not Path(self.session_path).exists():
                raise TelegramUnavailable(
                    f"No Telegram session at {self.session_path}. Run: uv run python scripts/telegram_login.py"
                )
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="telethon-loop", daemon=True)
            self._thread.start()
            try:
                self._run(self._connect(), timeout=180)
            except Exception:
                self._shutdown_loop()
                raise

    async def _connect(self) -> None:
        from telethon import utils

        self._client = await connect_client(self.session_path, self.credentials)
        for channel in self.channels:
            try:
                entity = await resolve_channel(self._client, channel)
                self._entities[channel.key] = entity
                self._by_peer_id[utils.get_peer_id(entity)] = channel
            except Exception as exc:  # noqa: BLE001
                self.unresolved[channel.key] = f"{type(exc).__name__}: {exc}"
                logger.warning(f"Skipping Telegram channel {channel.key}: {exc}")
            await asyncio.sleep(0.3)

    def _shutdown_loop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._client = None

    def close(self) -> None:
        if self._client is not None and self._loop is not None:
            try:
                self._run(self._client.disconnect(), timeout=10)
            except Exception:  # noqa: BLE001
                pass
        self._shutdown_loop()

    # ---------------------------------------------------------------- reading

    async def _get_messages(self, entity, limit: int):
        from telethon.errors import FloodWaitError

        for attempt in range(2):
            try:
                return await self._client.get_messages(entity, limit=limit)
            except FloodWaitError as exc:
                if exc.seconds > self.max_flood_wait or attempt:
                    raise
                logger.warning(f"Telegram asked us to wait {exc.seconds}s (FloodWait); sleeping")
                await asyncio.sleep(exc.seconds + 1)
        return []

    def _fallback_channel(self, channel: ChannelRef, limit: int) -> List[RawDeal]:
        if not (self.fallback and channel.public_ok):
            return []
        try:
            return self.fallback.fetch_channel(channel, limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Public preview fallback for {channel.describe()} failed: {exc}")
            return []

    def _fetch(self, limit: int) -> List[RawDeal]:
        try:
            self.ensure_started()
            self.using_fallback = False
        except TelegramUnavailable as exc:
            if not self.fallback:
                raise
            if not self.using_fallback:
                logger.info(f"MTProto unavailable ({exc}); using the public t.me/s/ preview instead")
            self.using_fallback = True
            return self.fallback.fetch(limit)

        deals: List[RawDeal] = []
        for i, channel in enumerate(self.channels):
            if i:
                time.sleep(self.channel_delay)
            entity = self._entities.get(channel.key)
            if entity is None:
                deals.extend(self._fallback_channel(channel, limit))
                continue
            try:
                messages = self._run(self._get_messages(entity, limit))
                for message in messages or []:
                    deal = parse_telethon_message(message, channel)
                    if deal:
                        deals.append(deal)
            except Exception as exc:  # noqa: BLE001 - per channel isolation
                logger.warning(f"MTProto read of {channel.describe()} failed ({exc}); trying public preview")
                deals.extend(self._fallback_channel(channel, limit))
        return deals

    # ------------------------------------------------------------------- live

    def start_live(self, on_deal: Callable[[RawDeal], None]) -> bool:
        """
        Subscribe to new posts in every resolved channel. on_deal is called from the
        Telethon thread for each parsed post, so it should only enqueue work.
        Returns False when MTProto is unavailable (polling still works via fallback).
        """
        try:
            self.ensure_started()
        except TelegramUnavailable as exc:
            logger.info(f"Live Telegram updates disabled: {exc}")
            return False
        if not self._entities:
            return False
        from telethon import events

        async def handler(event):
            channel = self._by_peer_id.get(event.chat_id)
            if channel is None:
                return
            deal = parse_telethon_message(event.message, channel)
            if deal:
                on_deal(deal)

        self._client.add_event_handler(handler, events.NewMessage(chats=list(self._entities.values())))
        logger.info(f"Listening live to {len(self._entities)} Telegram channels")
        return True
