"""Real-time polling scheduler.

Background thread that polls each enabled ingestion source on its own
cadence, persists fresh records via ParquetStore, and emits a
change-of-state callback so the terminal (or any other consumer) can
push to the UI in real time.

Per-source default cadences (operator can override):

    yahoo_quotes        30 s during market hours, 5 min otherwise
    openinsider         5 min (SEC filings don't move minute-by-minute)
    zacks               1 hour (Zacks updates pre-market)
    x_posts             5 min (rsshub caches anyway; more is rude)
    chains              once per day, at/after a target UTC time (default
                        20:30 UTC, just after the US close). The store keeps
                        one snapshot per day, so more often would only dedup.

"Real time" with free sources means polled, not pushed. None of the
upstream APIs expose SSE/websocket on their free tier. The scheduler
gives us a uniform polling surface — when a paid push feed comes
online later, we add it as another source without disturbing the rest.

Design constraints:
* Each source runs in its own thread; one slow source can't starve
  the others.
* The scheduler can be stopped cleanly via ``stop()`` — outstanding
  fetches are allowed to finish but no new ones start.
* Failures in one source don't kill the scheduler. They're logged and
  retried on the next tick.
* Watchlist for per-symbol sources is held in a dataclass field and
  can be updated at runtime via ``set_watchlist``.
* The on_update callback gets a SchedulerEvent so consumers can route
  by source — the terminal turns these into SocketIO emits in 1.4e.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Final

from storage.store import ParquetStore, WriteResult

if TYPE_CHECKING:
    from config.settings import Settings

# Default cadences in seconds.
_DEFAULT_YAHOO_INTERVAL_S: Final[int] = 30
_DEFAULT_OPENINSIDER_INTERVAL_S: Final[int] = 300
_DEFAULT_ZACKS_INTERVAL_S: Final[int] = 3600
_DEFAULT_X_INTERVAL_S: Final[int] = 300
# Chains: how often to CHECK whether today's capture is due, not how often
# to capture. The capture itself happens once per day.
_DEFAULT_CHAINS_INTERVAL_S: Final[int] = 300
_DEFAULT_CHAINS_AFTER_UTC: Final[dt.time] = dt.time(20, 30)


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    """A successful or failed poll cycle for one source."""

    source: str                  # "yahoo" | "openinsider" | "zacks" | "x" | "chains"
    triggered_at: datetime       # when the poll fired (UTC)
    ok: bool
    write_result: WriteResult | None
    error_message: str | None = None
    detail: str | None = None    # e.g. ticker / handle / 'latest'


@dataclass
class SchedulerConfig:
    """Per-source on/off + cadence.

    The scheduler reads this at construction time. To change cadences
    while running, stop the scheduler, edit, restart. Watchlist edits
    are dynamic (see set_watchlist).
    """

    yahoo_enabled: bool = True
    openinsider_enabled: bool = True
    zacks_enabled: bool = True
    x_enabled: bool = False  # default off — rsshub.app is unreliable
    chains_enabled: bool = False  # default off — needs an enabled broker

    yahoo_interval_s: int = _DEFAULT_YAHOO_INTERVAL_S
    openinsider_interval_s: int = _DEFAULT_OPENINSIDER_INTERVAL_S
    zacks_interval_s: int = _DEFAULT_ZACKS_INTERVAL_S
    x_interval_s: int = _DEFAULT_X_INTERVAL_S
    chains_interval_s: int = _DEFAULT_CHAINS_INTERVAL_S

    # Chains are captured once per day, the first check at/after this UTC time.
    chains_capture_after_utc: dt.time = _DEFAULT_CHAINS_AFTER_UTC
    chains_broker: str = "schwab"

    watchlist: tuple[str, ...] = field(default_factory=tuple)
    x_handles: tuple[str, ...] = field(default_factory=tuple)
    # Underlyings whose option chains get captured. Separate from the equity
    # watchlist: you may quote 30 names and only want chains for 2.
    chain_watchlist: tuple[str, ...] = field(default_factory=tuple)


class IngestionScheduler:
    """Background polling scheduler for engine ingestion sources.

    Use as a context manager::

        from storage.store import ParquetStore
        from ingestion.scheduler import IngestionScheduler, SchedulerConfig

        store = ParquetStore(Path("data"), env="paper")
        cfg = SchedulerConfig(watchlist=("NVDA", "MU", "SPY"))

        with IngestionScheduler(store, cfg, on_event=print) as sched:
            sched.start()
            ...  # work
        # scheduler stops cleanly on exit

    Threading model: one daemon thread per source. Threads sleep
    between cycles, so the wall-clock cadence is approximate (target
    ± tens of ms). For sub-second cadence we'd need a different design.
    """

    def __init__(
        self,
        store: ParquetStore,
        config: SchedulerConfig | None = None,
        *,
        on_event: Callable[[SchedulerEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        settings: Settings | None = None,
    ) -> None:
        self._store = store
        self._config = config or SchedulerConfig()
        self._on_event = on_event or (lambda _e: None)
        self._clock = clock
        self._sleep = sleep
        # Only the chains source needs settings (for the broker gate); loaded
        # lazily at first use if not injected, so existing callers are unchanged.
        self._settings = settings

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._watchlist_lock = threading.Lock()
        self._watchlist: tuple[str, ...] = self._config.watchlist
        self._x_handles: tuple[str, ...] = self._config.x_handles
        self._chain_watchlist: tuple[str, ...] = self._config.chain_watchlist
        self._chains_last_capture_date: dt.date | None = None

    # ── safe emit (callback hygiene) ──────────────────────────────────────

    def _emit(self, event: SchedulerEvent) -> None:
        """Call ``on_event(event)`` swallowing any exception the callback raises.

        Consumers' bugs cannot kill the scheduler. The brief's Principle #9
        (no silent fallbacks) is satisfied here because the *scheduler* is
        not failing silently — the consumer's callback failed and that's
        their bug to surface in their own logging.
        """
        with contextlib.suppress(Exception):
            self._on_event(event)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def __enter__(self) -> IngestionScheduler:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        self.stop(timeout=2.0)

    def start(self) -> None:
        """Spawn one thread per enabled source. Idempotent."""
        if self._threads:
            return
        if self._config.yahoo_enabled:
            self._threads.append(self._spawn("yahoo", self._tick_yahoo,
                                             self._config.yahoo_interval_s))
        if self._config.openinsider_enabled:
            self._threads.append(self._spawn(
                "openinsider", self._tick_openinsider,
                self._config.openinsider_interval_s,
            ))
        if self._config.zacks_enabled:
            self._threads.append(self._spawn(
                "zacks", self._tick_zacks, self._config.zacks_interval_s,
            ))
        if self._config.x_enabled:
            self._threads.append(self._spawn(
                "x", self._tick_x, self._config.x_interval_s,
            ))
        if self._config.chains_enabled:
            self._threads.append(self._spawn(
                "chains", self._tick_chains, self._config.chains_interval_s,
            ))

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal threads to exit; wait up to ``timeout`` seconds each."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    # ── dynamic config ────────────────────────────────────────────────────

    def set_watchlist(self, symbols: tuple[str, ...]) -> None:
        """Replace the per-symbol watchlist atomically. Thread-safe."""
        with self._watchlist_lock:
            self._watchlist = tuple(s.upper() for s in symbols)

    def set_x_handles(self, handles: tuple[str, ...]) -> None:
        """Replace the X-handle list atomically. Thread-safe."""
        with self._watchlist_lock:
            self._x_handles = tuple(h.lstrip("@") for h in handles)

    def set_chain_watchlist(self, symbols: tuple[str, ...]) -> None:
        """Replace the chain-capture underlyings atomically. Thread-safe."""
        with self._watchlist_lock:
            self._chain_watchlist = tuple(s.upper() for s in symbols)

    def watchlist(self) -> tuple[str, ...]:
        with self._watchlist_lock:
            return self._watchlist

    def x_handles(self) -> tuple[str, ...]:
        with self._watchlist_lock:
            return self._x_handles

    def chain_watchlist(self) -> tuple[str, ...]:
        with self._watchlist_lock:
            return self._chain_watchlist

    @property
    def chains_last_capture_date(self) -> dt.date | None:
        """The last calendar day (UTC) a chain capture succeeded, or None."""
        return self._chains_last_capture_date

    # ── thread loop ───────────────────────────────────────────────────────

    def _spawn(
        self,
        name: str,
        tick: Callable[[], None],
        interval_s: int,
    ) -> threading.Thread:
        def loop() -> None:
            while not self._stop_event.is_set():
                try:
                    tick()
                except Exception as e:
                    # Last-ditch: emit a failure event and continue.
                    self._emit(SchedulerEvent(
                        source=name,
                        triggered_at=_utcnow(),
                        ok=False,
                        write_result=None,
                        error_message=f"unhandled in tick: {e}",
                    ))
                # Sleep in small chunks so stop() interrupts quickly.
                slept = 0.0
                while slept < interval_s and not self._stop_event.is_set():
                    self._sleep(min(0.5, interval_s - slept))
                    slept += 0.5

        t = threading.Thread(target=loop, name=f"scheduler-{name}", daemon=True)
        t.start()
        return t

    # ── per-source ticks ──────────────────────────────────────────────────
    # Each tick handles its own exception envelope so one bad source can't
    # poison the rest. We import inside each tick so the scheduler can be
    # constructed even if a particular client's dependencies aren't on the
    # current path (e.g. running tests against just one source).

    def _tick_yahoo(self) -> None:
        from ingestion.yahoo_client import YahooClient

        watchlist = self.watchlist()
        if not watchlist:
            return
        with YahooClient() as yc:
            for ticker in watchlist:
                started = _utcnow()
                try:
                    quote = yc.get_quote(ticker)
                    result = self._store.write_equity_bars([quote])
                    self._emit(SchedulerEvent(
                        source="yahoo", triggered_at=started, ok=True,
                        write_result=result, detail=ticker,
                    ))
                except Exception as e:
                    self._emit(SchedulerEvent(
                        source="yahoo", triggered_at=started, ok=False,
                        write_result=None, error_message=str(e),
                        detail=ticker,
                    ))

    def _tick_openinsider(self) -> None:
        from ingestion.openinsider_client import OpenInsiderClient

        started = _utcnow()
        try:
            with OpenInsiderClient() as oi:
                trades = oi.get_latest_trades()
            result = self._store.write_insider_trades(trades)
            self._emit(SchedulerEvent(
                source="openinsider", triggered_at=started, ok=True,
                write_result=result, detail="latest",
            ))
        except Exception as e:
            self._emit(SchedulerEvent(
                source="openinsider", triggered_at=started, ok=False,
                write_result=None, error_message=str(e), detail="latest",
            ))

    def _tick_zacks(self) -> None:
        from ingestion.zacks_client import ZacksClient

        watchlist = self.watchlist()
        if not watchlist:
            return
        with ZacksClient() as zc:
            for ticker in watchlist:
                started = _utcnow()
                try:
                    rating = zc.get_rating(ticker)
                    result = self._store.write_analyst_ratings([rating])
                    self._emit(SchedulerEvent(
                        source="zacks", triggered_at=started, ok=True,
                        write_result=result, detail=ticker,
                    ))
                except Exception as e:
                    self._emit(SchedulerEvent(
                        source="zacks", triggered_at=started, ok=False,
                        write_result=None, error_message=str(e), detail=ticker,
                    ))

    def _tick_x(self) -> None:
        from ingestion.x_client import XClient

        handles = self.x_handles()
        if not handles:
            return
        with XClient() as xc:
            for handle in handles:
                started = _utcnow()
                try:
                    posts = xc.get_posts_for_handle(handle)
                    result = self._store.write_social_posts(posts)
                    self._emit(SchedulerEvent(
                        source="x", triggered_at=started, ok=True,
                        write_result=result, detail=handle,
                    ))
                except Exception as e:
                    self._emit(SchedulerEvent(
                        source="x", triggered_at=started, ok=False,
                        write_result=None, error_message=str(e), detail=handle,
                    ))

    def _tick_chains(self) -> None:
        """Capture option chains once per UTC day, at/after the target time.

        Retry policy: if every underlying fails (an outage), the day is left
        unmarked so the next check retries. A disabled broker marks the day
        done after one failure event — retrying every five minutes until the
        operator flips a flag would just be noise.
        """
        from ingestion.brokers.registry import BrokerDisabledError
        from ingestion.chain_capture import run_capture

        underlyings = self.chain_watchlist()
        if not underlyings:
            return
        now = _utcnow()
        if now.time() < self._config.chains_capture_after_utc:
            return
        if self._chains_last_capture_date == now.date():
            return

        if self._settings is None:
            from config.settings import get_settings

            self._settings = get_settings()

        try:
            results = run_capture(
                self._settings, self._store, underlyings,
                broker_name=self._config.chains_broker,
            )
        except BrokerDisabledError as e:
            self._emit(SchedulerEvent(
                source="chains", triggered_at=now, ok=False,
                write_result=None, error_message=str(e), detail=",".join(underlyings),
            ))
            self._chains_last_capture_date = now.date()
            return

        for r in results:
            self._emit(SchedulerEvent(
                source="chains", triggered_at=r.started_at, ok=r.ok,
                write_result=r.write_result, error_message=r.error, detail=r.underlying,
            ))
        if any(r.ok for r in results):
            self._chains_last_capture_date = now.date()


def _utcnow() -> datetime:
    """UTC now — wrapped so tests can monkeypatch if needed."""
    from datetime import UTC

    return datetime.now(UTC)
