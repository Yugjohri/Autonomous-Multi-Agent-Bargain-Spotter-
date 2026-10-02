"""
A small JSON-file cache with per-entry expiry, safe to share between threads.
Used for redirect resolutions, robots.txt rules and fetched feeds.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path(".cache")


class DiskCache:
    def __init__(self, name: str, ttl_seconds: float = 7 * 24 * 3600, directory: Path = CACHE_DIR):
        self.path = Path(directory) / f"{name}.json"
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict = {}
        self._dirty = 0
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            if time.time() - entry["t"] > self.ttl:
                del self._data[key]
                return None
            return entry["v"]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = {"t": time.time(), "v": value}
            self._dirty += 1
            if self._dirty >= 20:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = 0

    def __len__(self) -> int:
        return len(self._data)


class MemoryCache(DiskCache):
    """Same interface as DiskCache, but never touches the disk (used in tests)."""

    def __init__(self, ttl_seconds: float = 3600):
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._data = {}
        self._dirty = 0

    def _flush_locked(self) -> None:
        self._dirty = 0
