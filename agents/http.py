"""
Polite HTTP access shared by every source: one User-Agent, timeouts, retries with
exponential backoff, a minimum interval per host, and robots.txt checks.

robots.txt follows RFC 9309: a 2xx file is parsed, a 4xx (including 403) means no
rules apply, and a 5xx or network failure means the host is treated as disallowed.
In offline mode every request raises OfflineError, so fixture runs provably make
no network calls.
"""

import logging
import threading
import time
import urllib.robotparser
from typing import Dict, Optional
from urllib.parse import urlsplit

import requests

from agents.cache import DiskCache, MemoryCache

DEFAULT_USER_AGENT = (
    "BargainSpotter/0.2 (+https://github.com/Yugjohri/Autonomous-Multi-Agent-Bargain-Spotter-)"
)

RETRY_STATUSES = {429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)


class HttpError(Exception):
    pass


class OfflineError(HttpError):
    pass


class DisallowedByRobots(HttpError):
    pass


class HttpClient:
    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 15,
        retries: int = 2,
        backoff: float = 1.5,
        min_interval: float = 1.0,
        respect_robots: bool = True,
        offline: bool = False,
        robots_cache: Optional[DiskCache] = None,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.min_interval = min_interval
        self.respect_robots = respect_robots
        self.offline = offline
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "en-IN,en;q=0.8"})
        self._robots_cache = robots_cache if robots_cache is not None else MemoryCache(24 * 3600)
        self._robots_parsers: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self._host_locks: Dict[str, threading.Lock] = {}
        self._last_request: Dict[str, float] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ robots

    def allowed(self, url: str) -> bool:
        """True if robots.txt for the url's host permits our User-Agent to fetch it."""
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        parser = self._robots_for(origin)
        if parser is None:
            return False
        return parser.can_fetch(self.user_agent, url)

    def _robots_for(self, origin: str) -> Optional[urllib.robotparser.RobotFileParser]:
        with self._guard:
            if origin in self._robots_parsers:
                return self._robots_parsers[origin]
        cached = self._robots_cache.get(origin)
        if cached is None:
            cached = self._download_robots(origin)
            self._robots_cache.set(origin, cached)
        parser: Optional[urllib.robotparser.RobotFileParser]
        if cached["status"] == "disallow_all":
            parser = None
        else:
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(cached.get("body", "").splitlines())
        with self._guard:
            self._robots_parsers[origin] = parser
        return parser

    def _download_robots(self, origin: str) -> dict:
        if self.offline:
            raise OfflineError(f"offline: not fetching {origin}/robots.txt")
        try:
            self._wait_for_host(urlsplit(origin).netloc)
            response = self.session.get(f"{origin}/robots.txt", timeout=self.timeout, allow_redirects=True)
        except requests.RequestException as exc:
            logger.warning(f"robots.txt unreachable for {origin} ({exc}); treating host as disallowed")
            return {"status": "disallow_all"}
        if 200 <= response.status_code < 300:
            return {"status": "ok", "body": response.text}
        if 400 <= response.status_code < 500:
            return {"status": "ok", "body": ""}
        return {"status": "disallow_all"}

    # ---------------------------------------------------------------- requests

    def _wait_for_host(self, host: str) -> None:
        with self._guard:
            lock = self._host_locks.setdefault(host, threading.Lock())
        with lock:
            last = self._last_request.get(host, 0.0)
            delay = self.min_interval - (time.monotonic() - last)
            if delay > 0:
                time.sleep(delay)
            self._last_request[host] = time.monotonic()

    def request(self, method: str, url: str, check_robots: bool = True, **kwargs) -> requests.Response:
        if self.offline:
            raise OfflineError(f"offline: not requesting {url}")
        if check_robots and not self.allowed(url):
            raise DisallowedByRobots(url)
        kwargs.setdefault("timeout", self.timeout)
        host = urlsplit(url).netloc
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self._wait_for_host(host)
            try:
                response = self.session.request(method, url, **kwargs)
                if response.status_code in RETRY_STATUSES and attempt < self.retries:
                    retry_after = response.headers.get("Retry-After", "")
                    wait = float(retry_after) if retry_after.isdigit() else self.backoff * (2**attempt)
                    response.close()
                    time.sleep(min(wait, 60))
                    continue
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff * (2**attempt))
        raise HttpError(f"{method} {url} failed after {self.retries + 1} attempts: {last_error}")

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def head(self, url: str, **kwargs) -> requests.Response:
        return self.request("HEAD", url, **kwargs)
