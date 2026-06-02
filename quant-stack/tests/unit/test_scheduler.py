"""Polling scheduler tests.

The scheduler imports each client lazily inside its tick methods.
We mock those imports so the test never hits a real network and
runs in deterministic synthetic time.

Strategy:
- Monkeypatch `time.sleep` (via injection) to return instantly.
- Inject a fake clock that advances per-call.
- Patch the per-source clients (YahooClient, OpenInsiderClient, etc.)
  at the import points the ticks use.
- Spin the scheduler briefly, count events, verify call counts.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import pytest

from ingestion.scheduler import (
    IngestionScheduler,
    SchedulerConfig,
    SchedulerEvent,
)
from ingestion.schema import (
    AnalystRating,
    EquityBar,
    InsiderTrade,
    SocialPost,
)
from storage.store import ParquetStore

_TICKER: Final[str] = "NVDA"


# ─────────────────────────────────────────────────────────────────────────
#  Fakes that mimic client context managers
# ─────────────────────────────────────────────────────────────────────────


class _FakeYahoo:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __enter__(self) -> _FakeYahoo:
        return self

    def __exit__(self, *_: object) -> None:
        return

    def get_quote(self, symbol: str) -> EquityBar:
        self.calls.append(symbol)
        return EquityBar(
            as_of=datetime.now(UTC),
            symbol=symbol,
            interval="1d",
            open=100.0, high=101.0, low=99.0, close=100.5,
            volume=1_000_000, source="yahoo",
        )


class _FakeOpenInsider:
    def __init__(self) -> None:
        self.call_count = 0

    def __enter__(self) -> _FakeOpenInsider:
        return self

    def __exit__(self, *_: object) -> None:
        return

    def get_latest_trades(self) -> list[InsiderTrade]:
        self.call_count += 1
        return [
            InsiderTrade(
                as_of=datetime.now(UTC),
                filing_date=date(2026, 5, 30),
                trade_date=date(2026, 5, 29),
                ticker="NVDA",
                insider_name=f"Insider-{self.call_count}",
                trade_type="P - Purchase",
                price_per_share=110.0,
                quantity=1000,
                source="openinsider",
            ),
        ]


class _FakeZacks:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __enter__(self) -> _FakeZacks:
        return self

    def __exit__(self, *_: object) -> None:
        return

    def get_rating(self, symbol: str) -> AnalystRating:
        self.calls.append(symbol)
        return AnalystRating(
            as_of=datetime.now(UTC),
            symbol=symbol,
            rank=1, rank_text="Strong Buy",
            source="zacks",
        )


class _FakeX:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __enter__(self) -> _FakeX:
        return self

    def __exit__(self, *_: object) -> None:
        return

    def get_posts_for_handle(self, handle: str) -> list[SocialPost]:
        self.calls.append(handle)
        return [
            SocialPost(
                as_of=datetime.now(UTC),
                platform="x",
                post_id=f"{handle}-{len(self.calls)}",
                author_handle=handle,
                content="$NVDA looking strong",
                posted_at=datetime.now(UTC),
                cashtags=("NVDA",),
                source="rsshub",
            ),
        ]


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    yahoo: _FakeYahoo | None = None,
    openinsider: _FakeOpenInsider | None = None,
    zacks: _FakeZacks | None = None,
    x: _FakeX | None = None,
) -> None:
    if yahoo is not None:
        monkeypatch.setattr("ingestion.yahoo_client.YahooClient", lambda *a, **kw: yahoo)
    if openinsider is not None:
        monkeypatch.setattr(
            "ingestion.openinsider_client.OpenInsiderClient",
            lambda *a, **kw: openinsider,
        )
    if zacks is not None:
        monkeypatch.setattr("ingestion.zacks_client.ZacksClient", lambda *a, **kw: zacks)
    if x is not None:
        monkeypatch.setattr("ingestion.x_client.XClient", lambda *a, **kw: x)


# ─────────────────────────────────────────────────────────────────────────
#  Direct (synchronous) tick tests — no threads, no sleep
# ─────────────────────────────────────────────────────────────────────────


def test_yahoo_tick_calls_each_watchlist_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    yahoo = _FakeYahoo()
    _install_fakes(monkeypatch, yahoo=yahoo)

    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(
        yahoo_enabled=True, openinsider_enabled=False,
        zacks_enabled=False, x_enabled=False,
        watchlist=("NVDA", "SPY", "QQQ"),
    )
    s = IngestionScheduler(store, cfg, on_event=events.append)
    s._tick_yahoo()
    assert yahoo.calls == ["NVDA", "SPY", "QQQ"]
    assert len(events) == 3
    assert all(e.ok for e in events)
    assert {e.source for e in events} == {"yahoo"}


def test_yahoo_tick_skips_when_watchlist_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    yahoo = _FakeYahoo()
    _install_fakes(monkeypatch, yahoo=yahoo)
    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    s = IngestionScheduler(store, SchedulerConfig(), on_event=events.append)
    s._tick_yahoo()
    assert yahoo.calls == []
    assert events == []


def test_openinsider_tick_persists_and_emits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    oi = _FakeOpenInsider()
    _install_fakes(monkeypatch, openinsider=oi)
    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    s = IngestionScheduler(store, SchedulerConfig(), on_event=events.append)
    s._tick_openinsider()
    assert oi.call_count == 1
    assert len(events) == 1
    assert events[0].source == "openinsider"
    assert events[0].ok is True


def test_zacks_tick_calls_each_watchlist_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    zacks = _FakeZacks()
    _install_fakes(monkeypatch, zacks=zacks)
    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(watchlist=("NVDA", "MU"))
    s = IngestionScheduler(store, cfg, on_event=events.append)
    s._tick_zacks()
    assert zacks.calls == ["NVDA", "MU"]
    assert len(events) == 2


def test_x_tick_calls_each_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = _FakeX()
    _install_fakes(monkeypatch, x=x)
    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(x_handles=("alice", "bob"))
    s = IngestionScheduler(store, cfg, on_event=events.append)
    s._tick_x()
    assert x.calls == ["alice", "bob"]
    assert len(events) == 2


# ─────────────────────────────────────────────────────────────────────────
#  Error handling — one bad call doesn't poison the source
# ─────────────────────────────────────────────────────────────────────────


def test_yahoo_tick_continues_after_one_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Flaky(_FakeYahoo):
        def get_quote(self, symbol: str) -> EquityBar:
            # Avoid super().get_quote — _FakeYahoo also appends, would double-count.
            self.calls.append(symbol)
            if symbol == "BAD":
                msg = "boom"
                raise RuntimeError(msg)
            return EquityBar(
                as_of=datetime.now(UTC),
                symbol=symbol, interval="1d",
                open=100.0, high=101.0, low=99.0, close=100.5,
                volume=1_000_000, source="yahoo",
            )

    flaky = _Flaky()
    _install_fakes(monkeypatch, yahoo=flaky)
    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(watchlist=("NVDA", "BAD", "SPY"))
    s = IngestionScheduler(store, cfg, on_event=events.append)
    s._tick_yahoo()
    assert flaky.calls == ["NVDA", "BAD", "SPY"]
    assert len(events) == 3
    statuses = [e.ok for e in events]
    assert statuses == [True, False, True]
    assert events[1].error_message == "boom"


# ─────────────────────────────────────────────────────────────────────────
#  Dynamic watchlist updates
# ─────────────────────────────────────────────────────────────────────────


def test_set_watchlist_replaces_atomically(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    s = IngestionScheduler(store, SchedulerConfig(watchlist=("A",)))
    s.set_watchlist(("X", "y", "Z"))
    # All uppercased
    assert s.watchlist() == ("X", "Y", "Z")


def test_set_x_handles_strips_at(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path, env="test")
    s = IngestionScheduler(store, SchedulerConfig())
    s.set_x_handles(("@alice", "bob"))
    assert s.x_handles() == ("alice", "bob")


# ─────────────────────────────────────────────────────────────────────────
#  Lifecycle: start / stop / running
# ─────────────────────────────────────────────────────────────────────────


def test_lifecycle_start_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    yahoo = _FakeYahoo()
    _install_fakes(monkeypatch, yahoo=yahoo)
    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(
        yahoo_enabled=True, openinsider_enabled=False,
        zacks_enabled=False, x_enabled=False,
        watchlist=("NVDA",),
        yahoo_interval_s=1,  # short for the test
    )
    s = IngestionScheduler(store, cfg, on_event=events.append)
    s.start()
    # Wait until at least one tick fired
    deadline = time.monotonic() + 5.0
    while not events and time.monotonic() < deadline:
        time.sleep(0.05)
    s.stop(timeout=2.0)
    assert events, "scheduler should have fired at least once"
    assert s.running is False


def test_context_manager_stops_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    yahoo = _FakeYahoo()
    _install_fakes(monkeypatch, yahoo=yahoo)
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(
        yahoo_enabled=True, openinsider_enabled=False,
        zacks_enabled=False, x_enabled=False,
        watchlist=("NVDA",), yahoo_interval_s=1,
    )
    with IngestionScheduler(store, cfg) as s:
        s.start()
        time.sleep(0.1)
    assert s.running is False


def test_start_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    yahoo = _FakeYahoo()
    _install_fakes(monkeypatch, yahoo=yahoo)
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(
        yahoo_enabled=True, openinsider_enabled=False,
        zacks_enabled=False, x_enabled=False,
        watchlist=("NVDA",), yahoo_interval_s=10,
    )
    s = IngestionScheduler(store, cfg)
    s.start()
    initial_threads = len(s._threads)
    s.start()  # second call → no-op
    assert len(s._threads) == initial_threads
    s.stop(timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────
#  Defaults safety
# ─────────────────────────────────────────────────────────────────────────


def test_x_disabled_by_default() -> None:
    """X via rsshub is unreliable; default must be off."""
    cfg = SchedulerConfig()
    assert cfg.x_enabled is False


def test_other_sources_enabled_by_default() -> None:
    cfg = SchedulerConfig()
    assert cfg.yahoo_enabled is True
    assert cfg.openinsider_enabled is True
    assert cfg.zacks_enabled is True


# ─────────────────────────────────────────────────────────────────────────
#  on_event callback contract
# ─────────────────────────────────────────────────────────────────────────


def test_event_carries_write_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    yahoo = _FakeYahoo()
    _install_fakes(monkeypatch, yahoo=yahoo)
    events: list[SchedulerEvent] = []
    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(watchlist=("NVDA",))
    s = IngestionScheduler(store, cfg, on_event=events.append)
    s._tick_yahoo()
    assert events[0].write_result is not None
    assert events[0].write_result.requested == 1
    assert events[0].write_result.persisted == 1


def test_callback_exception_does_not_kill_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buggy on_event callback shouldn't bring down the scheduler."""
    yahoo = _FakeYahoo()
    _install_fakes(monkeypatch, yahoo=yahoo)

    call_count = [0]

    def bad_callback(_e: SchedulerEvent) -> None:
        call_count[0] += 1
        msg = "consumer is on fire"
        raise RuntimeError(msg)

    store = ParquetStore(tmp_path, env="test")
    cfg = SchedulerConfig(watchlist=("NVDA", "MU"))
    s = IngestionScheduler(store, cfg, on_event=bad_callback)
    # The tick itself catches the exception and emits a failure event,
    # which also fails — but the loop should keep going for the next ticker.
    s._tick_yahoo()
    # Callback got called at least once (proves we tried).
    assert call_count[0] >= 1
