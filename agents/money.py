"""
Parsing and formatting of money amounts, with first-class support for Indian rupees.

Indian deal posts write prices in many ways: "₹1,29,999", "Rs. 499", "INR 999",
"₹49,999/-", "@264", "at 10,999". find_inr_amounts() locates every amount that is
marked as money (by a currency sign, "/-" suffix, or a price word such as "@" or "at")
and keeps its position, so callers can classify it as a price, MRP, discount, etc.
Bare numbers such as "128GB", "65%" or "5000mAh" are never treated as money.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

# A number with optional Indian (1,29,999) or Western (129,999) grouping and decimals.
_NUMBER = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"

# Strong markers: an explicit currency.
_STRONG_PREFIX = r"₹|\bRs\.?|\bINR|\bRupees?"

# Weak markers: words that usually introduce a price in deal posts.
_WEAK_PREFIX = r"@|\bat\b|\bfor\b|\bonly\b|\bjust\b|\bprice\b:?|\bdeal\b:?"

# Units that mean the number is a spec or a percentage, not money.
_UNIT_AFTER = re.compile(
    r"\s*(%|percent|gb\b|tb\b|mb\b|mah\b|w\b|watt|hz\b|inch|\"|mp\b|x\b|pcs?\b|pack\b|"
    r"months?\b|years?\b|yrs?\b|days?\b|hrs?\b|hours?\b|mins?\b|am\b|pm\b|kg\b|g\b|gm\b|ml\b|"
    r"l\b|ltr|litre|m\b|cm\b|mm\b|ft\b|k\b|th\b|st\b|nd\b|rd\b|cores?\b|ram\b|ssd\b)",
    re.IGNORECASE,
)

_STRONG = re.compile(rf"(?P<marker>{_STRONG_PREFIX})\s*(?P<num>{_NUMBER})(?P<suffix>\s*/-)?", re.IGNORECASE)
_SUFFIX_ONLY = re.compile(rf"(?<![\w.,])(?P<num>{_NUMBER})\s*/-")
_WEAK = re.compile(
    rf"(?P<marker>{_WEAK_PREFIX})\s*(?:₹|Rs\.?|INR)?\s*(?P<num>{_NUMBER})",
    re.IGNORECASE,
)

# Weak markers produce many false positives on small numbers ("for 2 people").
MIN_WEAK_AMOUNT = 10.0


@dataclass(frozen=True)
class Amount:
    """A money amount found in text, with its location and the marker that introduced it."""

    value: float
    start: int
    end: int
    marker: str


def _to_float(number: str) -> float:
    return float(number.replace(",", ""))


def find_inr_amounts(text: str) -> List[Amount]:
    """Return every rupee amount in the text, in order of appearance."""
    if not text:
        return []
    found: List[Amount] = []
    taken: List[range] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < r.stop and end > r.start for r in taken)

    def add(match: re.Match, marker: str, weak: bool = False) -> None:
        start, end = match.start(), match.end()
        if overlaps(start, end):
            return
        # A unit right after the number ("for 6 months", "@ 128GB") means it is not money.
        if weak and _UNIT_AFTER.match(text, match.end("num")):
            return
        value = _to_float(match.group("num"))
        if weak and value < MIN_WEAK_AMOUNT:
            return
        found.append(Amount(value=value, start=start, end=end, marker=marker))
        taken.append(range(start, end))

    for m in _STRONG.finditer(text):
        add(m, m.group("marker").strip().rstrip(".").lower())
    for m in _SUFFIX_ONLY.finditer(text):
        add(m, "/-")
    for m in _WEAK.finditer(text):
        add(m, m.group("marker").strip().rstrip(":").lower(), weak=True)

    found.sort(key=lambda a: a.start)
    return found


def parse_inr(text: str) -> List[float]:
    """Return the rupee amounts in the text, in order of appearance."""
    return [a.value for a in find_inr_amounts(text)]


def parse_amount(text: Optional[str]) -> Optional[float]:
    """Parse a single amount such as "₹1,299", "1299.00" or "$45" into a float."""
    if text is None:
        return None
    match = re.search(_NUMBER, str(text).replace(" ", ""))
    return _to_float(match.group()) if match else None


def _indian_grouping(integer: int) -> str:
    digits = str(integer)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$"}


def format_money(amount: Optional[float], currency: str = "INR") -> str:
    """
    Format an amount for display. INR uses Indian digit grouping (₹1,29,999) and
    drops the paise when the amount is whole; other currencies use Western grouping.
    """
    if amount is None:
        return "-"
    currency = (currency or "INR").upper()
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    if currency == "INR":
        rounded = round(amount, 2)
        whole = int(rounded)
        paise = round((rounded - whole) * 100)
        body = _indian_grouping(whole)
        if paise:
            body += f".{paise:02d}"
        return f"{sign}{symbol}{body}"
    return f"{sign}{symbol}{amount:,.2f}"
