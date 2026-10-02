"""
One-time interactive Telegram login for this project.

    uv run python scripts/telegram_login.py

Reads TG_API_ID and TG_API_HASH from .env, asks for your phone number, the login code
Telegram sends you, and your 2FA password if you have one, then saves a new session at
secrets/bargain_spotter.session. That file grants access to your account: it is
gitignored and must never be shared or copied to a server.
"""

import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from agents.config import load_sources_config, get  # noqa: E402
from agents.sources.telegram_mtproto import credentials_from_env, session_base  # noqa: E402


async def main() -> int:
    from telethon import TelegramClient

    load_dotenv(override=True)
    credentials = credentials_from_env()
    if credentials is None:
        print("TG_API_ID and TG_API_HASH must be set in .env (see .env.example).")
        return 1
    session_path = get(load_sources_config(), "telegram.session_path", "secrets/bargain_spotter.session")
    Path(session_path).parent.mkdir(parents=True, exist_ok=True)

    api_id, api_hash = credentials
    client = TelegramClient(session_base(session_path), api_id, api_hash)
    await client.start(
        phone=lambda: input("Phone number with country code (e.g. +91...): ").strip(),
        code_callback=lambda: input("Login code from Telegram: ").strip(),
        password=lambda: getpass.getpass("2FA password (input hidden): "),
    )
    me = await client.get_me()
    handle = f" (@{me.username})" if me.username else ""
    print(f"Logged in as {me.first_name or ''}{handle}. Session saved to {session_path}.")
    print("Next: uv run python scripts/list_channels.py")
    await client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
