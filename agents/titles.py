"""
Product title normalisation, used to group colour variants and to detect the same product
across datasets (the leakage check between training data and the test set).

"Samsung Galaxy M56 5G (Black, 8 GB RAM, 128 GB Storage)" and the "(Light Green, ...)"
variant both normalise to "samsung galaxy m56 5g 8 gb ram 128 gb storage".
"""

import hashlib
import html
import re

COLOURS = [
    "black", "white", "blue", "red", "green", "grey", "gray", "silver", "gold", "golden", "pink", "purple",
    "violet", "yellow", "orange", "brown", "beige", "navy", "maroon", "cream", "teal", "turquoise", "mint",
    "lavender", "lilac", "peach", "coral", "olive", "khaki", "charcoal", "graphite", "midnight", "starlight",
    "titanium", "bronze", "copper", "rose", "aqua", "cyan", "magenta", "ivory", "multicolor", "multicolour",
    "multi", "transparent", "clear", "space", "jet", "matte", "glossy", "dark", "light", "sky", "ocean",
    "forest", "sage", "emerald", "ruby", "sapphire", "onyx", "obsidian", "pearl", "champagne", "mustard",
    "wine", "burgundy", "rust", "sand", "stone", "slate", "ash", "smoke", "frost", "ice", "snow", "phantom",
    "cosmic", "nebula", "glacier", "aurora", "mystic", "mystique", "stellar", "lunar", "astral", "shadow",
]
_COLOUR_WORD = re.compile(r"\b(" + "|".join(COLOURS) + r")\b", re.IGNORECASE)
_BRACKETS = re.compile(r"[(\[]([^()\[\]]*)[)\]]")
_NON_WORD = re.compile(r"[^a-z0-9.+]+")

_RAM = re.compile(r"(\d+)\s*gb\s*ram", re.IGNORECASE)
_STORAGE = re.compile(r"(\d+)\s*(gb|tb)\s*(storage|rom|ssd|hdd|emmc|ufs)?\b(?!\s*ram)", re.IGNORECASE)
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(inch|inches|\"|cm)\b", re.IGNORECASE)


def _drop_colour_parts(match: re.Match) -> str:
    """Inside brackets, drop comma-separated parts that are only a colour name ("Glacier Silver")."""
    kept = []
    for part in match.group(1).split(","):
        if _COLOUR_WORD.search(part) and not re.search(r"\d", part):
            continue
        kept.append(part)
    return f"({','.join(kept)})" if kept else " "


def normalise_title(title: str) -> str:
    text = html.unescape(title or "").lower().replace("...", " ")
    text = _BRACKETS.sub(_drop_colour_parts, text)
    text = _COLOUR_WORD.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def specs_key(title: str) -> str:
    """RAM, storage and screen size, so variants that differ only in those stay apart."""
    text = html.unescape(title or "")
    ram = _RAM.search(text)
    storage = [f"{n}{u.lower()}" for n, u, _ in _STORAGE.findall(text)]
    size = _SIZE.search(text)
    parts = [
        f"ram{ram.group(1)}" if ram else "",
        "+".join(sorted(set(storage))),
        f"{size.group(1)}{size.group(2).lower()}" if size else "",
    ]
    return "/".join(parts)


def title_hash(title: str) -> str:
    return hashlib.sha1(normalise_title(title).encode("utf-8")).hexdigest()[:16]
