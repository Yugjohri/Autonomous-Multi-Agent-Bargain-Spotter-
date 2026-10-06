"""
Link normalization: resolve shorteners, strip affiliate and tracking parameters,
detect the store, and derive a canonical product id used for dedupe.

Store product pages (amazon.in, flipkart.com, ...) are never requested. Redirect
chains are followed one hop at a time by reading Location headers, and resolution
stops as soon as the next hop is a store URL. Wrapper links that carry their
destination in a query parameter (?url=, ?o=, ?dl=) are unwrapped without any request.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit

from agents.cache import DiskCache, MemoryCache
from agents.http import HttpClient, HttpError

logger = logging.getLogger(__name__)

# Host suffix -> store key. Longest suffix wins.
STORE_DOMAINS: Dict[str, str] = {
    "amazon.in": "amazon_in",
    "flipkart.com": "flipkart",
    "myntra.com": "myntra",
    "croma.com": "croma",
    "reliancedigital.in": "reliance_digital",
    "ajio.com": "ajio",
    "tatacliq.com": "tatacliq",
    "nykaa.com": "nykaa",
    "nykaafashion.com": "nykaa",
    "jiomart.com": "jiomart",
    "meesho.com": "meesho",
    "sheinindia.in": "shein_in",
    "vijaysales.com": "vijay_sales",
    "firstcry.com": "firstcry",
    "pepperfry.com": "pepperfry",
    "bigbasket.com": "bigbasket",
    "boat-lifestyle.com": "boat",
    "samsung.com": "samsung_in",
    "mi.com": "mi_in",
}

# Short-link hosts that always point at one store. Used when a link cannot be resolved
# (offline, or robots.txt forbids it) so the store is still known.
SHORTENER_STORES: Dict[str, str] = {
    "amzn.to": "amazon_in", "amzn.in": "amazon_in", "a.co": "amazon_in", "link.amazon": "amazon_in",
    "amzlinks.in": "amazon_in", "fkrt.it": "flipkart", "fkrt.cc": "flipkart", "fkrt.co": "flipkart",
    "fktr.in": "flipkart", "myntr.it": "myntra", "ajiio.in": "ajio",
}

# Stores we treat as recognised Indian retailers for the pre-filter.
RECOGNISED_STORES = set(STORE_DOMAINS.values())

STORE_NAMES = {
    "amazon_in": "Amazon",
    "flipkart": "Flipkart",
    "myntra": "Myntra",
    "croma": "Croma",
    "reliance_digital": "Reliance Digital",
    "ajio": "AJIO",
    "tatacliq": "Tata CLiQ",
    "nykaa": "Nykaa",
    "jiomart": "JioMart",
    "meesho": "Meesho",
    "shein_in": "SHEIN",
    "vijay_sales": "Vijay Sales",
    "firstcry": "FirstCry",
    "pepperfry": "Pepperfry",
    "bigbasket": "BigBasket",
    "boat": "boAt",
    "samsung_in": "Samsung",
    "mi_in": "Xiaomi",
    "other": "Other",
}

# Hosts we are willing to resolve with a request. Anything else is kept as-is,
# so we never crawl arbitrary merchant or blog pages.
RESOLVE_HOSTS = {
    "amzn.to", "amzn.in", "amzn.eu", "a.co", "link.amazon", "amzlinks.in",
    "fkrt.it", "fkrt.cc", "fkrt.co", "fktr.in", "dl.flipkart.com",
    "bit.ly", "cutt.ly", "tinyurl.com", "rb.gy", "is.gd", "t.ly", "shorturl.at", "ow.ly",
    "grbn.in", "bitli.in", "bitli.io", "clnk.in", "ekaro.in", "extp.in", "myntr.it",
    "links.bigtricks.in", "go.bigtricks.in", "wishlink.com", "t2m.io", "inrdeals.com", "ezlnk.in",
}

# Query parameters that carry the real destination inside a wrapper link.
WRAPPER_PARAMS = ("url", "u", "o", "d", "dl", "redirect", "redirect_url", "target", "link", "goto", "r")

TRACKING_PARAMS = {
    "tag", "ref", "ref_", "linkcode", "linkid", "camp", "creative", "creativeasin", "ascsubtag",
    "btn_type", "btn_ref", "th", "psc", "smid", "qid", "sr", "keywords", "dib", "dib_tag",
    "sprefix", "crid", "content-id", "social_share", "affid", "lid", "marketplace", "store",
    "srno", "fm", "iid", "ppt", "ppn", "ssid", "qh", "cmpid", "gclid", "fbclid", "source",
    "src", "cid", "clickid", "igshid", "si", "spm", "trk", "_encoding", "share", "shareid",
    "_appid", "_refid", "requestid", "pageid", "q", "sid",
}
TRACKING_PREFIXES = ("utm_", "affextparam", "otracker", "pf_rd_", "pd_rd_", "aff_", "subid", "mc_", "trk_", "source_")

AMAZON_ASIN = re.compile(r"/(?:dp|gp/product|gp/aw/d|product|exec/obidos/asin|d)/([A-Z0-9]{10})(?:[/?]|$)", re.I)
FLIPKART_ITEM = re.compile(r"/p/(itm[0-9a-z]{6,})", re.I)
FLIPKART_PID = re.compile(r"[A-Z0-9]{16}")

# Canonical ids that name one real store product (used to judge label quality).
PRODUCT_ID_RE = re.compile(r"^(amazon_in:[A-Z0-9]{10}|flipkart:(ITM[0-9A-Z]{6,}|[A-Z0-9]{16}))$")


def is_product_id(canonical: str) -> bool:
    return bool(canonical) and bool(PRODUCT_ID_RE.match(canonical))

# Paths a source's robots.txt forbids, which we must never request even via a redirect.
NEVER_REQUEST = [
    re.compile(r"^https?://(www\.)?dealsmagnet\.com/(buy|redirectto|rd|rdt)(/|$)", re.I),
    re.compile(r"^https?://(www\.)?desidime\.com/(goto|links|redirector)(/|$)", re.I),
]


def host_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def detect_store(url: str) -> str:
    host = host_of(url)
    best, best_len = "other", 0
    for suffix, store in STORE_DOMAINS.items():
        if (host == suffix or host.endswith("." + suffix)) and len(suffix) > best_len:
            best, best_len = store, len(suffix)
    return best


def is_store_url(url: str) -> bool:
    return detect_store(url) != "other"


def unwrap(url: str) -> str:
    """If the url wraps another http(s) url in a query parameter, return the inner one."""
    for _ in range(5):
        params = parse_qsl(urlsplit(url).query)
        inner = next(
            (v for k, v in params if k.lower() in WRAPPER_PARAMS and v.lower().startswith(("http://", "https://"))),
            None,
        )
        if not inner:
            return url
        url = inner
    return url


def _is_tracking(key: str) -> bool:
    key = key.lower()
    return key in TRACKING_PARAMS or key.startswith(TRACKING_PREFIXES)


def asin_of(url: str) -> Optional[str]:
    if detect_store(url) != "amazon_in" and "amazon." not in host_of(url):
        return None
    match = AMAZON_ASIN.search(urlsplit(url).path + "/")
    if match:
        return match.group(1).upper()
    # Promotion and offer pages name their product in the query (?redirectAsin=B0..., ?asin=B0...).
    query = {k.lower(): v for k, v in parse_qsl(urlsplit(url).query)}
    asin = query.get("asin") or query.get("redirectasin")
    return asin.upper() if asin and re.fullmatch(r"[A-Za-z0-9]{10}", asin) else None


# Search parameters that look like tracking ("q", "sid", "keywords") but are the search itself.
SEARCH_PARAMS = {"q", "k", "sid", "rh", "i", "keywords", "p[]", "rawquery", "node"}


def is_search_or_listing(url: str) -> bool:
    """
    True for a store link that is a search, category or sale page rather than one product
    (amazon.in/s?k=..., flipkart.com/search?q=..., myntra.com/men-watches). Short links
    that were not resolved (dl.flipkart.com/s/..., amzn.to/...) are not judged.
    """
    store = detect_store(url)
    parts = urlsplit(url)
    path = parts.path
    if store == "amazon_in":
        return asin_of(url) is None
    if store == "flipkart":
        if parts.netloc.lower().startswith("dl.") and path.startswith("/s/"):
            return False
        return not (FLIPKART_ITEM.search(path) or flipkart_pid(url))
    if store == "myntra":
        return not re.search(r"/\d{5,}(/buy)?/?$", path)
    return False


def clean_url(url: str) -> str:
    """Return the url without affiliate or tracking parameters, in a stable canonical form."""
    url = unwrap(url.strip())
    parts = urlsplit(url)
    store = detect_store(url)
    if store == "amazon_in":
        asin = asin_of(url)
        if asin:
            return f"https://www.amazon.in/dp/{asin}"
    if store == "flipkart":
        match = FLIPKART_ITEM.search(parts.path)
        if match:
            path = parts.path
            if path.startswith("/dl/"):
                path = path[3:]
            path = path[: match.end()]
            pid = dict(parse_qsl(parts.query)).get("pid")
            query = urlencode({"pid": pid}) if pid else ""
            return urlunsplit(("https", "www.flipkart.com", path, query, ""))
    search = is_search_or_listing(url)
    kept = sorted(
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not _is_tracking(k) or (search and k.lower() in SEARCH_PARAMS)
    )
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", netloc, path, urlencode(kept), ""))


def flipkart_pid(url: str) -> Optional[str]:
    pid = dict(parse_qsl(urlsplit(url).query)).get("pid", "")
    return pid.upper() if FLIPKART_PID.fullmatch(pid.upper()) else None


def canonical_id(url: str) -> str:
    """
    amazon_in:<ASIN>; flipkart:<PID> (the 16 character product id, present on almost every
    Flipkart link, including /product/p/itme?pid=... links with no item id), else
    flipkart:<ITM id>; otherwise <store>:<hash of the cleaned url>.
    """
    cleaned = clean_url(url)
    store = detect_store(cleaned)
    if store == "amazon_in":
        asin = asin_of(cleaned)
        if asin:
            return f"amazon_in:{asin}"
    if store == "flipkart":
        pid = flipkart_pid(cleaned)
        if pid:
            return f"flipkart:{pid}"
        match = FLIPKART_ITEM.search(urlsplit(cleaned).path)
        if match:
            return f"flipkart:{match.group(1).upper()}"
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16]
    return f"{store}:{digest}"


def title_from_url(url: str) -> str:
    """Product slug from a store url, e.g. /oscar-forever-perfume-combo/p/itm.. -> 'Oscar Forever Perfume Combo'."""
    path = urlsplit(unwrap(url)).path
    store = detect_store(url)
    slug = ""
    if store == "flipkart":
        match = FLIPKART_ITEM.search(path)
        if match:
            slug = path[: match.start()].strip("/").split("/")[-1]
    elif store == "amazon_in":
        match = AMAZON_ASIN.search(path + "/")
        if match:
            slug = path[: match.start()].strip("/").split("/")[-1]
    if not slug or len(slug) < 6:
        return ""
    return re.sub(r"[-_]+", " ", unquote(slug)).strip().title()[:120]


class RedirectResolver:
    """Follows shortener redirects one hop at a time, with a disk cache and robots checks."""

    def __init__(
        self,
        http: Optional[HttpClient] = None,
        cache: Optional[DiskCache] = None,
        max_hops: int = 5,
        extra_hosts: Iterable[str] = (),
        offline: bool = False,
    ):
        self.http = http
        self.cache = cache if cache is not None else MemoryCache()
        self.max_hops = max_hops
        self.hosts = set(RESOLVE_HOSTS) | set(extra_hosts)
        self.offline = offline or http is None or http.offline

    def should_resolve(self, url: str) -> bool:
        if is_store_url(url):
            # Store hosts are never requested, except Flipkart's short share links
            # (dl.flipkart.com/s/...), which only answer with a redirect.
            return host_of(url) == "dl.flipkart.com" and not FLIPKART_ITEM.search(urlsplit(url).path)
        return host_of(url) in self.hosts

    def resolve(self, url: str) -> str:
        url = unwrap(url)
        if not self.should_resolve(url):
            return url
        cached = self.cache.get(url)
        if cached:
            return cached
        if self.offline:
            return url
        final = self._follow(url)
        self.cache.set(url, final)
        return final

    def _follow(self, url: str) -> str:
        current = url
        for _ in range(self.max_hops):
            current = unwrap(current)
            if not self.should_resolve(current):
                return current
            if any(p.search(current) for p in NEVER_REQUEST):
                return current
            nxt = self._next_hop(current)
            if not nxt or nxt == current:
                return current
            current = nxt
        return current

    def _next_hop(self, url: str) -> Optional[str]:
        try:
            if not self.http.allowed(url):
                logger.info(f"robots.txt disallows resolving {host_of(url)}; keeping short link")
                return None
            for method in ("HEAD", "GET"):
                response = self.http.request(method, url, allow_redirects=False, stream=True, check_robots=False)
                location = response.headers.get("Location")
                status = response.status_code
                response.close()
                if 300 <= status < 400 and location:
                    return urljoin(url, location)
                if method == "HEAD" and status in (200, 400, 403, 404, 405):
                    continue
                return None
        except HttpError as exc:
            logger.info(f"Could not resolve {url}: {exc}")
        return None


@dataclass
class NormalizedLink:
    original: str
    resolved: str
    url: str
    store: str
    canonical_id: str


def normalize_link(url: str, resolver: Optional[RedirectResolver] = None) -> NormalizedLink:
    resolved = resolver.resolve(url) if resolver else unwrap(url)
    cleaned = clean_url(resolved)
    store = detect_store(cleaned)
    canonical = canonical_id(cleaned)
    if store == "other":
        store = SHORTENER_STORES.get(host_of(cleaned), "other")
        canonical = f"{store}:{canonical.split(':', 1)[1]}"
    return NormalizedLink(
        original=url,
        resolved=resolved,
        url=cleaned,
        store=store,
        canonical_id=canonical,
    )


# ---------------------------------------------------------------------- dedupe


def is_fresh(posted_at: Optional[datetime], window_hours: float, now: Optional[datetime] = None) -> bool:
    """Posts with no timestamp are kept; everything else must be inside the freshness window."""
    if posted_at is None:
        return True
    now = now or datetime.now(timezone.utc)
    if posted_at.tzinfo is None:
        posted_at = posted_at.replace(tzinfo=timezone.utc)
    return now - posted_at <= timedelta(hours=window_hours)


def dedupe_by_canonical(items: List, key: Callable[[object], str], merge: Callable[[object, object], None]) -> List:
    """
    Collapse items sharing a canonical id into the first one seen, calling merge(kept, dup)
    so the caller can record every source that posted it.
    """
    kept: Dict[str, object] = {}
    order: List[str] = []
    for item in items:
        k = key(item)
        if k in kept:
            merge(kept[k], item)
        else:
            kept[k] = item
            order.append(k)
    return [kept[k] for k in order]


class SeenIndex:
    """
    Remembers the lowest price each canonical id has been seen at. A product counts as
    new if we never saw it, or if it is now cheaper than before (a price drop).
    """

    def __init__(self, entries: Optional[Dict[str, float]] = None, tolerance: float = 1.0):
        self.prices: Dict[str, float] = dict(entries or {})
        self.tolerance = tolerance

    def is_new(self, canonical: str, price: Optional[float]) -> bool:
        if canonical not in self.prices:
            return True
        if price is None:
            return False
        return price < self.prices[canonical] - self.tolerance

    def add(self, canonical: str, price: Optional[float]) -> None:
        if price is None:
            self.prices.setdefault(canonical, float("inf"))
            return
        self.prices[canonical] = min(price, self.prices.get(canonical, float("inf")))

    @classmethod
    def from_opportunities(cls, opportunities) -> "SeenIndex":
        index = cls()
        for opp in opportunities:
            deal = opp.deal
            index.add(deal.canonical_id or canonical_id(deal.url), deal.price)
        return index
