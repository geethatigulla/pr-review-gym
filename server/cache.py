"""Simple in-memory cache with TTL, used to expire idle HTTP environment sessions."""

from __future__ import annotations

from time import monotonic
from typing import Any, Callable, Hashable, Iterator, MutableMapping, Optional

ExpireCallback = Callable[[Hashable, Any], None]


class TTLCache:
    """Minimal dict-backed cache with per-key expiry.

    Args:
        default_ttl_sec: Lifetime applied by :meth:`set` when no explicit TTL is given.
        store: Optional backing mapping (a fresh ``dict`` per instance by default).
        on_expire: Optional callback invoked as ``on_expire(key, value)`` whenever an
            entry is dropped because it expired or was evicted to respect ``max_size``.
        max_size: Optional upper bound on live entries; the entry closest to expiry
            is evicted first when the bound is exceeded.
        clock: Monotonic time source (injectable for tests).
    """

    def __init__(
        self,
        default_ttl_sec: float = 300.0,
        store: Optional[MutableMapping[Hashable, dict[str, Any]]] = None,
        on_expire: Optional[ExpireCallback] = None,
        max_size: Optional[int] = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if default_ttl_sec <= 0:
            raise ValueError("default_ttl_sec must be positive")
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be >= 1 when set")
        self.default_ttl_sec = default_ttl_sec
        self.store: MutableMapping[Hashable, dict[str, Any]] = (
            {} if store is None else store
        )
        self.on_expire = on_expire
        self.max_size = max_size
        self._clock = clock

    def set(self, key: Hashable, value: Any, ttl_sec: Optional[float] = None) -> None:
        """Store ``value`` under ``key`` until ``ttl_sec`` (or default TTL) elapses."""
        ttl = self.default_ttl_sec if ttl_sec is None else ttl_sec
        self.store[key] = {"value": value, "expire_at": self._clock() + ttl, "ttl": ttl}
        self._enforce_max_size()

    def get(self, key: Hashable, touch: bool = False) -> Any | None:
        """Return the value for ``key`` or ``None`` if missing or expired.

        With ``touch=True`` a live entry's expiry is pushed forward by its TTL
        (sliding expiration), which keeps active sessions alive.
        """
        entry = self.store.get(key)
        if entry is None:
            return None
        now = self._clock()
        if entry["expire_at"] <= now:
            self._drop(key, notify=True)
            return None
        if touch:
            entry["expire_at"] = now + entry.get("ttl", self.default_ttl_sec)
        return entry["value"]

    def pop(self, key: Hashable) -> Any | None:
        """Remove ``key`` and return its value (``None`` if missing). No callback fires."""
        entry = self.store.pop(key, None)
        return None if entry is None else entry["value"]

    def cleanup(self) -> int:
        """Remove all expired entries; return how many were dropped."""
        now = self._clock()
        expired = [k for k, e in list(self.store.items()) if e["expire_at"] <= now]
        for key in expired:
            self._drop(key, notify=True)
        return len(expired)

    def __contains__(self, key: object) -> bool:
        entry = self.store.get(key)  # type: ignore[arg-type]
        return entry is not None and entry["expire_at"] > self._clock()

    def __len__(self) -> int:
        return len(self.store)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(list(self.store))

    def _drop(self, key: Hashable, notify: bool) -> None:
        entry = self.store.pop(key, None)
        if entry is not None and notify and self.on_expire is not None:
            self.on_expire(key, entry["value"])

    def _enforce_max_size(self) -> None:
        if self.max_size is None:
            return
        self.cleanup()
        while len(self.store) > self.max_size:
            oldest = min(self.store, key=lambda k: self.store[k]["expire_at"])
            self._drop(oldest, notify=True)
