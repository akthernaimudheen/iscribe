"""Small in-memory sliding-window rate limiter.

Protects the authentication endpoint (credential stuffing) and the expensive
processing endpoints (STT is usage-based; an unauthenticated or compromised
session must not be able to burn unlimited credits). Single-process by
design, like the rest of the service — this is a trial deployment guard, not
a distributed rate limiter.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allow at most `limit` events per `window_seconds` per key."""

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record one event for `key`; True if it is within the limit."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            # Opportunistic cleanup so abandoned keys do not grow forever.
            if len(self._hits) > 10_000:
                for k in [k for k, d in self._hits.items() if not d or
                          (d and d[-1] < cutoff)]:
                    del self._hits[k]
            return True

    def retry_after(self, key: str) -> float:
        """Seconds until the next event for `key` would be allowed."""
        with self._lock:
            hits = self._hits.get(key)
            if not hits or len(hits) < self.limit:
                return 0.0
            return max(0.0, hits[0] + self.window - time.monotonic())
