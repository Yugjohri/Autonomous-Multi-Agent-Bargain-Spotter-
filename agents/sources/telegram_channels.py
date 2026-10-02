"""
Channel references from sources.yaml. A channel can be identified three ways:

    username: GrabOnIndiaOfficial          public; also readable via t.me/s/ without login
    id: -1001439586091                     numeric id from Telegram Web URLs; MTProto only
    invite: https://t.me/+Ur52sNQaMl1c...  private invite link; MTProto only, must already be joined
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChannelRef:
    username: Optional[str] = None
    id: Optional[int] = None
    invite: Optional[str] = None
    name: Optional[str] = None
    enabled: bool = True
    trust: float = 1.0
    note: Optional[str] = None

    @property
    def key(self) -> str:
        """Stable short name used in source ids, e.g. telegram:<key>."""
        if self.name:
            return self.name
        if self.username:
            return self.username
        if self.id is not None:
            return str(self.id)
        return f"invite-{self.invite_hash[:6]}" if self.invite_hash else "unknown"

    @property
    def invite_hash(self) -> Optional[str]:
        if not self.invite:
            return None
        match = re.search(r"(?:t\.me/(?:\+|joinchat/)|^\+)([\w-]+)", self.invite)
        return match.group(1) if match else None

    @property
    def public_ok(self) -> bool:
        """Only channels with a username have a public t.me/s/ web preview."""
        return bool(self.username)

    def describe(self) -> str:
        if self.username:
            return f"@{self.username}"
        if self.id is not None:
            return f"id {self.id}"
        return "invite link (hidden)"


def parse_channels(items: Optional[List[Dict[str, Any]]]) -> List[ChannelRef]:
    channels = []
    for item in items or []:
        username = item.get("username")
        if username:
            username = str(username).lstrip("@").strip()
        raw_id = item.get("id")
        channels.append(
            ChannelRef(
                username=username or None,
                id=int(raw_id) if raw_id not in (None, "") else None,
                invite=item.get("invite"),
                name=item.get("name"),
                enabled=bool(item.get("enabled", True)),
                trust=float(item.get("trust", 1.0)),
                note=item.get("note"),
            )
        )
    return channels
