import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from telethon.errors import FloodWaitError

from agents.sources.telegram_channels import ChannelRef
from agents.sources.telegram_mtproto import (
    TelegramMTProtoSource,
    links_of_message,
    parse_telethon_message,
)

CHANNEL = ChannelRef(username="OMGDeals", trust=1.2)


def message(text, id=1, **kwargs):
    return SimpleNamespace(
        message=text,
        id=id,
        date=datetime(2026, 10, 2, 9, 0, tzinfo=timezone.utc),
        entities=kwargs.get("entities"),
        reply_markup=kwargs.get("reply_markup"),
        fwd_from=kwargs.get("fwd_from"),
        poll=kwargs.get("poll"),
    )


def test_links_from_entities_and_buttons():
    msg = message(
        "Mivi speaker @264 Buy here",
        entities=[SimpleNamespace(url="https://amzn.to/hidden"), SimpleNamespace(offset=0, length=4)],
        reply_markup=SimpleNamespace(rows=[SimpleNamespace(buttons=[SimpleNamespace(url="https://fkrt.cc/btn"), SimpleNamespace(text="x")])]),
    )
    assert links_of_message(msg) == ["https://amzn.to/hidden", "https://fkrt.cc/btn"]
    deal = parse_telethon_message(msg, CHANNEL)
    assert deal.price_hint == 264
    assert deal.url == "https://amzn.to/hidden"
    assert deal.external_id == "telegram:OMGDeals/1"
    assert deal.trust == 1.2


def test_polls_media_only_and_forwarded_chatter_skipped():
    assert parse_telethon_message(message("Which phone?", poll=object()), CHANNEL) is None
    assert parse_telethon_message(message(""), CHANNEL) is None
    assert parse_telethon_message(message("hello all", fwd_from=object()), CHANNEL) is None


class FakePublic:
    def __init__(self):
        self.calls = []

    def fetch(self, limit):
        self.calls.append(("all", limit))
        return ["public-deal"]

    def fetch_channel(self, channel, limit):
        self.calls.append((channel.key, limit))
        return [f"public-{channel.key}"]


def test_falls_back_to_public_preview_without_session(tmp_path):
    public = FakePublic()
    source = TelegramMTProtoSource([CHANNEL], fallback=public, session_path=str(tmp_path / "missing.session"), credentials=(1, "x"))
    assert source.fetch(5) == ["public-deal"]
    assert source.using_fallback
    assert not (tmp_path / "missing.session").exists(), "must never create a session file"


def test_falls_back_without_credentials(tmp_path):
    public = FakePublic()
    source = TelegramMTProtoSource([CHANNEL], fallback=public, session_path=str(tmp_path / "s.session"), credentials=None)
    source.credentials = None
    assert source.fetch(5) == ["public-deal"]


def test_live_disabled_without_session(tmp_path):
    source = TelegramMTProtoSource([CHANNEL], session_path=str(tmp_path / "missing.session"), credentials=(1, "x"))
    assert source.start_live(lambda deal: None) is False


class FakeClient:
    def __init__(self, flood_first=True):
        self.calls = 0
        self.flood_first = flood_first

    async def get_messages(self, entity, limit):
        self.calls += 1
        if self.flood_first and self.calls == 1:
            raise FloodWaitError(request=None, capture=0)
        return [message("Parachute Shampoo 1.2L @374.\nhttps://link.amazon/B09YLQjd2", id=7)]


def _started_source(client, channels=(CHANNEL,), fallback=None):
    source = TelegramMTProtoSource(list(channels), fallback=fallback, credentials=(1, "x"), channel_delay=0)
    source._loop = asyncio.new_event_loop()
    import threading

    source._thread = threading.Thread(target=source._loop.run_forever, daemon=True)
    source._thread.start()
    source._client = client
    return source


def test_flood_wait_is_honoured_then_retried():
    client = FakeClient(flood_first=True)
    source = _started_source(client)
    source._entities = {"OMGDeals": object()}
    try:
        deals = source.fetch(5)
    finally:
        source._shutdown_loop()
    assert client.calls == 2
    assert [d.price_hint for d in deals] == [374]


def test_unresolved_channel_uses_public_fallback_per_channel():
    public = FakePublic()
    private = ChannelRef(id=-1001439586091, name="DCLootsOffers")
    source = _started_source(FakeClient(flood_first=False), channels=(CHANNEL, private), fallback=public)
    source._entities = {"DCLootsOffers": object()}
    try:
        deals = source.fetch(3)
    finally:
        source._shutdown_loop()
    # OMGDeals did not resolve and has a username, so it is read from t.me/s/;
    # the id-only channel was read via MTProto.
    assert public.calls == [("OMGDeals", 3)]
    assert "public-OMGDeals" in deals and any(getattr(d, "price_hint", None) == 374 for d in deals)


def test_source_is_read_only():
    """The Telegram account must only ever read: no send, forward, join or react calls."""
    code = Path("agents/sources/telegram_mtproto.py").read_text(encoding="utf-8")
    code += Path("scripts/backfill_telegram.py").read_text(encoding="utf-8")
    code += Path("scripts/list_channels.py").read_text(encoding="utf-8")
    forbidden = r"send_message|send_file|forward_messages|ImportChatInvite|JoinChannel|SendReaction|edit_message|delete_messages"
    assert not re.search(forbidden, code)
