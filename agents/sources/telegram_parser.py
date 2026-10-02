"""
One parser for Telegram deal posts, shared by the MTProto source, the public web
preview source and the backfill script.

Typical formats seen in Indian deal channels:
    "Titan Talk Smartwatch 68% OFF Now only ₹4,784 (MRP ₹14,995)"
    "Perfume Combo (Pack of 2) @199."          "Loot 199"          "3399"
    "₹8,988 / 63% off"                          "... - Rs. 490"
    "₹700 dropped! Current Price: ₹20,999  Highest Price: ₹25,299"
    "Upto 85% off on Puma shoes"   (percentage only: not priceable)
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional

from agents.money import Amount, find_inr_amounts
from agents.normalize import RESOLVE_HOSTS, host_of, is_store_url
from agents.sources.base import RawDeal

URL_RE = re.compile(r"https?://[^\s<>()\"'\]]+", re.IGNORECASE)

JUNK_HOSTS = (
    "t.me", "telegram.me", "telegram.dog", "whatsapp.com", "wa.me", "chat.whatsapp.com", "instagram.com",
    "youtube.com", "youtu.be", "facebook.com", "x.com", "twitter.com", "hcti.io", "telegra.ph",
)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")

EXPIRED_RE = re.compile(
    r"\b(over now|deal over|expired|out of stock|sold out|oos|deal is dead|price increased)\b", re.IGNORECASE
)
PERCENT_RE = re.compile(r"\d{1,2}\s*%\s*off", re.IGNORECASE)
# "Clothing starts @60" is a category listing, not a single product price.
STARTING_RE = re.compile(r"\b(starts?|starting)\s*(@|at|from|@\s*just)?\s*(₹|rs\.?)?\s*\d", re.IGNORECASE)
LOOT_RE = re.compile(r"\bloot\s*[:\-]?\s*₹?\s*(\d{2,3}(?:,\d{2,3})+|\d{2,7})\b(?!\s*(%|gb|tb|ltr|kg|ml|mah))", re.IGNORECASE)
BARE_PRICE_LINE_RE = re.compile(r"^\s*₹?\s*(\d{1,3}(?:,\d{2,3})+|\d{2,7})\s*(/-)?\s*(https?://\S+)?\s*$")

STORE_WORDS = {
    "amazon": "amazon_in", "flipkart": "flipkart", "myntra": "myntra", "croma": "croma",
    "reliance digital": "reliance_digital", "ajio": "ajio", "tata cliq": "tatacliq", "tatacliq": "tatacliq",
    "nykaa": "nykaa", "jiomart": "jiomart", "meesho": "meesho", "shein": "shein_in", "vijay sales": "vijay_sales",
}
STORE_WORD_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in STORE_WORDS) + r")\b", re.IGNORECASE)

COUPON_LINE_RE = re.compile(
    r"(coupon|use code|promo code|code\s*:|bank offer|credit card|debit card|\bicici\b|\bhdfc\b|\bsbi\b|"
    r"\baxis\b|\bkotak\b|cashback|\bcb\b|super ?coins?|exchange)",
    re.IGNORECASE,
)

BOILERPLATE_LINE_RE = re.compile(
    r"(grab this|before it'?s gone|^loot\s*(fast)?\s*:?$|^buy now|^shop now|^read more|^grab the deal|"
    r"share\s*•|join\s+@|^edit\b|^deal\s*:?$|^hurry|limited time|^valid for|^\W*$)",
    re.IGNORECASE,
)
TITLE_PREFIX_RE = re.compile(r"^(loot\s*fast|loot|grab|deal|hot deal|steal deal|price drop)\s*[:\-–|]\s*", re.IGNORECASE)
NON_TEXT_RE = re.compile(r"[^\w\s&+.,'()/\-:%₹@|]", re.UNICODE)

# Words that directly precede or follow an amount and tell us what it is.
MRP_BEFORE = re.compile(r"(mrp|m\.r\.p\.?|was|original price|worth|regular price|list price|retail price)\W*$", re.I)
FACE_VALUE_BEFORE = re.compile(r"(gift card|voucher|recharge)( of| worth)?\W*$", re.I)
FACE_VALUE_AFTER = re.compile(r"^\W*(gift card|voucher|amazon pay|e-?gift)", re.I)
HIGHEST_BEFORE = re.compile(r"(highest price|high price|all time high|ath)\W*$", re.I)
LOWEST_BEFORE = re.compile(r"(lowest price|all time low|atl)\W*$", re.I)
CURRENT_BEFORE = re.compile(r"(current price|deal price|offer price|price now|now only|now at|effective price)\W*$", re.I)
OFF_AFTER = re.compile(
    r"^\W*(off\b|discount|dropped|drop\b|cashback|cb\b|instant|savings?|bank|coupon|cash ?back|extra off|less)", re.I
)
OFF_BEFORE = re.compile(r"(save|saving|savings|discount of|cashback of|coupon of|extra|upto|up to|min|minimum|above)\W*$", re.I)
PERCENT_AFTER = re.compile(r"^\s*/?\s*(\d{1,2})\s*%\s*off", re.I)

PRIORITY_MARKERS = {"@", "at", "only", "just", "for", "price", "deal", "loot", "line"}


@dataclass
class ClassifiedAmount:
    amount: Amount
    kind: str  # price | mrp | highest | lowest | off
    priority: int = 0


def extract_links(text: str, extra: Iterable[str] = ()) -> List[str]:
    """All http(s) links in the text plus extra ones (entities, buttons), deduped, junk removed."""
    seen, result = set(), []
    for link in list(URL_RE.findall(text or "")) + list(extra or []):
        link = link.rstrip(".,;:!)]}'\"")
        host = host_of(link)
        if not host or any(host == j or host.endswith("." + j) for j in JUNK_HOSTS):
            continue
        if link.lower().split("?")[0].endswith(IMAGE_EXT):
            continue
        if link not in seen:
            seen.add(link)
            result.append(link)
    return result


def pick_primary_link(text: str, links: List[str]) -> str:
    """Prefer the link that follows "Buy Now", then store or shortener links, then the first link."""
    if not links:
        return ""
    lowered = (text or "").lower()
    buy_pos = lowered.find("buy now")
    if buy_pos >= 0:
        after = [link for link in links if lowered.find(link.lower(), buy_pos) >= 0]
        if after:
            return after[0]
    for link in links:
        if is_store_url(link) or host_of(link) in RESOLVE_HOSTS:
            return link
    return links[0]


def _classify(text: str, amounts: List[Amount]) -> List[ClassifiedAmount]:
    result = []
    for a in amounts:
        before = text[max(0, a.start - 30) : a.start]
        after = text[a.end : a.end + 25]
        if HIGHEST_BEFORE.search(before):
            kind, priority = "highest", 0
        elif LOWEST_BEFORE.search(before):
            kind, priority = "lowest", 0
        elif MRP_BEFORE.search(before) or FACE_VALUE_BEFORE.search(before) or FACE_VALUE_AFTER.search(after):
            kind, priority = "mrp", 0
        elif OFF_AFTER.search(after) or OFF_BEFORE.search(before):
            kind, priority = "off", 0
        elif CURRENT_BEFORE.search(before):
            kind, priority = "price", 3
        elif a.marker in PRIORITY_MARKERS:
            kind, priority = "price", 2
        else:
            kind, priority = "price", 1
        result.append(ClassifiedAmount(a, kind, priority))
    return result


def _extra_amounts(text: str) -> List[Amount]:
    """Telegram-only price forms: "Loot 199" and a line holding only a number ("3399")."""
    found = []
    for m in LOOT_RE.finditer(text):
        found.append(Amount(float(m.group(1).replace(",", "")), m.start(), m.end(), "loot"))
    offset = 0
    for line in text.splitlines(keepends=True):
        m = BARE_PRICE_LINE_RE.match(line)
        if m:
            value = float(m.group(1).replace(",", ""))
            if value >= 10:
                found.append(Amount(value, offset + m.start(1), offset + m.end(1), "line"))
        offset += len(line)
    return found


def parse_prices(text: str):
    """Return (price, mrp, highest_price) found in a post. Any of them may be None."""
    amounts = find_inr_amounts(text)
    taken = [range(a.start, a.end) for a in amounts]
    for extra in _extra_amounts(text):
        if not any(extra.start < r.stop and extra.end > r.start for r in taken):
            amounts.append(extra)
    amounts.sort(key=lambda a: a.start)
    classified = _classify(text, amounts)

    prices = [c for c in classified if c.kind == "price"]
    price_item = max(prices, key=lambda c: (c.priority, -c.amount.start)) if prices else None
    price = price_item.amount.value if price_item else None
    mrps = [c.amount.value for c in classified if c.kind == "mrp"]
    highs = [c.amount.value for c in classified if c.kind == "highest"]
    mrp = mrps[0] if mrps else None

    if price is not None and mrp is None:
        # "₹8,988 / 63% off": derive the implied MRP from the advertised percentage.
        pct = PERCENT_AFTER.match(text[price_item.amount.end : price_item.amount.end + 15])
        if pct and 0 < int(pct.group(1)) < 95:
            mrp = round(price / (1 - int(pct.group(1)) / 100))
    if mrp is not None and price is not None and mrp <= price:
        mrp = None
    return price, mrp, (highs[0] if highs else None)


def _clean_line(line: str) -> str:
    line = NON_TEXT_RE.sub(" ", line)
    return re.sub(r"\s+", " ", line).strip(" -:|•.")


def extract_title(text: str) -> str:
    for raw in (text or "").splitlines():
        line = _clean_line(raw)
        if not line or BOILERPLATE_LINE_RE.search(line) or URL_RE.search(raw):
            continue
        line = TITLE_PREFIX_RE.sub("", line)
        # Cut the title where the price or discount part starts.
        line = re.split(r"\s(?:@|at\s+₹?\d|now only|now at|just\s+₹|only\s+₹|for\s+₹|₹|rs\.?\s*\d|\d{1,2}\s*%\s*off)",
                        " " + line, maxsplit=1, flags=re.IGNORECASE)[0].strip(" -:|•.")
        words = [w for w in line.split() if sum(ch.isalpha() for ch in w) >= 2]
        if len(words) >= 2:
            return line[:120]
    return ""


def store_hint(text: str) -> str:
    match = STORE_WORD_RE.search(text or "")
    return STORE_WORDS[match.group(1).lower()] if match else "other"


def coupon_hint(text: str) -> Optional[str]:
    lines = [_clean_line(line) for line in (text or "").splitlines() if COUPON_LINE_RE.search(line)]
    lines = [line for line in lines if line and not URL_RE.search(line)]
    return " | ".join(lines)[:160] or None


def parse_post(
    text: str,
    channel: str,
    message_id: Optional[int] = None,
    posted_at: Optional[datetime] = None,
    extra_links: Iterable[str] = (),
    forwarded: bool = False,
    is_poll: bool = False,
    trust: float = 1.0,
) -> Optional[RawDeal]:
    """
    Parse one Telegram message. Returns None for posts that are not deal posts at all
    (media-only, polls, forwarded chatter). Posts that look like deals but cannot be
    priced are returned with priceable=False so they can still be logged.
    """
    if is_poll or not text or not text.strip():
        return None
    links = extract_links(text, extra_links)
    price, mrp, highest = parse_prices(text)
    if forwarded and price is None and not links:
        return None

    drop_reason = None
    if EXPIRED_RE.search(text):
        drop_reason = "expired"
    elif price is None:
        drop_reason = "percentage_only" if PERCENT_RE.search(text) else "no_price"
    elif STARTING_RE.search(text):
        drop_reason = "starting_price"
    elif not links:
        drop_reason = "no_link"

    return RawDeal(
        title=extract_title(text),
        text=text.strip(),
        url=pick_primary_link(text, links),
        store=store_hint(text),
        price_hint=price,
        mrp_hint=mrp,
        posted_at=posted_at,
        source=f"telegram:{channel}",
        links=links,
        external_id=f"telegram:{channel}/{message_id}" if message_id is not None else "",
        highest_price_hint=highest,
        coupon_hint=coupon_hint(text),
        priceable=drop_reason is None,
        drop_reason=drop_reason,
        trust=trust,
    )
