"""Tests for the TTL cache used to expire idle HTTP sessions."""
from __future__ import annotations

import pytest

from server.cache import TTLCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_set_get_and_expiry() -> None:
    clock = FakeClock()
    cache = TTLCache(default_ttl_sec=10, clock=clock)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert "a" in cache
    clock.now = 10
    assert cache.get("a") is None
    assert "a" not in cache
    assert len(cache) == 0


def test_instances_do_not_share_store() -> None:
    a, b = TTLCache(), TTLCache()
    a.set("k", 1)
    assert b.get("k") is None


def test_touch_extends_lifetime() -> None:
    clock = FakeClock()
    cache = TTLCache(default_ttl_sec=10, clock=clock)
    cache.set("a", 1)
    clock.now = 8
    assert cache.get("a", touch=True) == 1
    clock.now = 15
    assert cache.get("a") == 1
    clock.now = 18
    assert cache.get("a") is None


def test_on_expire_called_for_expired_and_evicted() -> None:
    clock = FakeClock()
    expired: list[tuple[str, int]] = []
    cache = TTLCache(
        default_ttl_sec=10,
        clock=clock,
        max_size=2,
        on_expire=lambda k, v: expired.append((k, v)),
    )
    cache.set("a", 1)
    clock.now = 1
    cache.set("b", 2)
    clock.now = 2
    cache.set("c", 3)  # evicts "a", the entry closest to expiry
    assert expired == [("a", 1)]
    assert len(cache) == 2

    clock.now = 100
    assert cache.cleanup() == 2
    assert sorted(expired) == [("a", 1), ("b", 2), ("c", 3)]


def test_pop_does_not_fire_callback() -> None:
    fired: list = []
    cache = TTLCache(on_expire=lambda k, v: fired.append(k))
    cache.set("a", 1)
    assert cache.pop("a") == 1
    assert cache.pop("a") is None
    assert fired == []


def test_cleanup_while_iterating_is_safe() -> None:
    clock = FakeClock()
    cache = TTLCache(default_ttl_sec=1, clock=clock)
    for i in range(5):
        cache.set(i, i)
    clock.now = 5
    assert cache.cleanup() == 5


@pytest.mark.parametrize("kwargs", [{"default_ttl_sec": 0}, {"max_size": 0}])
def test_invalid_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TTLCache(**kwargs)
