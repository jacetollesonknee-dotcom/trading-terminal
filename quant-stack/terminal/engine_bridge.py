"""engine_bridge.py — thin glue between the terminal and the quant-stack engine.

ADR-004 makes the terminal the all-in-one app and the engine its backend,
consumed "via in-process import." This module is that import boundary.

It does three jobs:

1. Puts the quant-stack root on ``sys.path`` so ``storage`` / ``ingestion`` /
   ``config`` import even when the terminal is launched from ``terminal/``.
2. Runs the engine's :class:`IngestionScheduler` in the background and turns
   every :class:`SchedulerEvent` into a SocketIO ``ingestion_update`` emit via
   a caller-supplied ``emit`` callable (so this module never imports Flask).
3. Reads persisted records back out of the engine's point-in-time store for
   the terminal's REST routes (sentiment, insider).

Everything degrades gracefully: if the engine's dependencies aren't importable
or the feature is disabled, :attr:`EngineBridge.available` is ``False`` and the
terminal keeps running on ``data_feeds.py`` alone. The terminal is a
deliberately un-gated prototype (see pyproject ``exclude``); the tested,
reusable pieces live in the engine (``ingestion/serialize.py``).
"""

from __future__ import annotations

import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

# ── Make the engine importable when launched from terminal/ ────────────────
_ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

EmitFn = Callable[[str, dict[str, Any]], None]


class EngineBridge:
    """Owns the engine scheduler + store on behalf of the terminal.

    Construct once at process start. Call :meth:`start` with a SocketIO-style
    ``emit(event_name, payload)`` callback to begin streaming ingestion events.
    All public methods are safe to call even when the engine is unavailable —
    they no-op or return empty/degraded results rather than raising.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        watchlist: tuple[str, ...] = (),
    ) -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._scheduler: Any = None
        self._store: Any = None
        self._emit: EmitFn | None = None
        self._import_error: str | None = None
        self._initial_watchlist = tuple(s.upper() for s in watchlist)

        # Lazily imported engine handles (kept as attributes so query helpers
        # can reuse them without re-importing on every request).
        self._PointInTimeQuery: Any = None
        self._serialize: Any = None

        if enabled:
            self._try_import()

    # ── availability ──────────────────────────────────────────────────────

    def _try_import(self) -> None:
        try:
            from config.settings import get_settings
            from ingestion import serialize
            from ingestion.scheduler import IngestionScheduler, SchedulerConfig
            from storage.query import PointInTimeQuery
            from storage.store import ParquetStore
        except Exception as e:  # noqa: BLE001 — degrade, don't crash the terminal
            self._import_error = f"{type(e).__name__}: {e}"
            return

        try:
            settings = get_settings()
            self._store = ParquetStore(settings.data_dir, settings.app_env.value)
            self._env = settings.app_env.value
            cfg = SchedulerConfig(watchlist=self._initial_watchlist)
            self._scheduler = IngestionScheduler(
                self._store, cfg, on_event=self._on_event
            )
            self._PointInTimeQuery = PointInTimeQuery
            self._serialize = serialize
        except Exception as e:  # noqa: BLE001
            self._import_error = f"{type(e).__name__}: {e}"
            self._scheduler = None
            self._store = None

    @property
    def available(self) -> bool:
        """True when the engine scheduler + store are wired and ready."""
        return self._scheduler is not None and self._store is not None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self, emit: EmitFn) -> bool:
        """Begin streaming ingestion events through ``emit``. Idempotent.

        Returns True if the scheduler actually started, False if the engine
        is unavailable (the terminal should carry on with its own feeds).
        """
        if not self.available:
            return False
        with self._lock:
            self._emit = emit
            self._scheduler.start()
        return True

    def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.stop(timeout=2.0)

    def set_watchlist(self, symbols: list[str] | tuple[str, ...]) -> None:
        """Push the terminal's watchlist into the scheduler (thread-safe)."""
        if self._scheduler is not None:
            self._scheduler.set_watchlist(tuple(symbols))

    # ── event bridge ──────────────────────────────────────────────────────

    def _on_event(self, event: Any) -> None:
        """Scheduler callback: serialize and forward to the browser.

        Runs on a scheduler worker thread. Any exception here is swallowed by
        the scheduler's ``_emit`` guard, but we keep this defensive anyway so a
        serialization bug can't silently drop the whole stream.
        """
        emit = self._emit
        if emit is None or self._serialize is None:
            return
        payload = self._serialize.scheduler_event_to_dict(event)
        emit("ingestion_update", payload)

    # ── query helpers (point-in-time reads for REST routes) ───────────────

    def _query(self) -> Any:
        return self._PointInTimeQuery(self._store, as_of=datetime.now(UTC))

    def sentiment(self, symbol: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """Recent X/social posts mentioning ``$symbol`` from the engine store.

        Returns ``[]`` when the engine is unavailable or has no posts yet —
        the terminal falls back to its own feeds in that case.
        """
        if not self.available:
            return []
        try:
            posts = self._query().social_posts_mentioning(symbol.upper(), limit=limit)
        except Exception:  # noqa: BLE001 — empty on any read error
            return []
        return [p.model_dump(mode="json") for p in posts]

    def insider(self, symbol: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """Recent insider trades for ``symbol``, shaped for the terminal table.

        The terminal's insider table (and ``data_feeds.openinsider_latest``)
        use ``title`` / ``price`` / ``value``; the engine model uses
        ``insider_title`` / ``price_per_share`` / ``dollar_value``. We map to
        the terminal's contract here so the engine-first route renders
        identically to the live-scrape fallback.
        """
        if not self.available:
            return []
        try:
            trades = self._query().insider_trades(symbol.upper())
        except Exception:  # noqa: BLE001
            return []
        return [_insider_to_row(t) for t in trades[:limit]]

    def zacks(self, symbol: str) -> dict[str, Any] | None:
        """Latest Zacks rating for ``symbol``, shaped for the terminal card.

        Returns ``None`` when the engine is unavailable or has no rating yet,
        so the route falls back to the live ``data_feeds.zacks_rating`` scrape.
        The output mirrors that scrape's keys (``zacks_rank`` / ``rank_text`` /
        ``price_target`` / ``stats``) so the front-end renders identically.
        """
        if not self.available:
            return None
        try:
            rating = self._query().latest_analyst_rating(symbol.upper())
        except Exception:  # noqa: BLE001
            return None
        return _rating_to_zacks(rating) if rating is not None else None

    # ── status (for /api/engine/status) ───────────────────────────────────

    def status(self) -> dict[str, Any]:
        running = bool(self._scheduler is not None and self._scheduler.running)
        return {
            "enabled": self._enabled,
            "available": self.available,
            "running": running,
            "env": getattr(self, "_env", None),
            "watchlist": list(self._scheduler.watchlist()) if self._scheduler else [],
            "import_error": self._import_error,
        }


def _rating_to_zacks(rating: Any) -> dict[str, Any]:
    """Map an engine ``AnalystRating`` to the terminal's Zacks-card shape."""
    stats: dict[str, str] = {}
    if rating.style_score_value:
        stats["Value"] = rating.style_score_value
    if rating.style_score_growth:
        stats["Growth"] = rating.style_score_growth
    if rating.style_score_momentum:
        stats["Momentum"] = rating.style_score_momentum
    if rating.style_score_vgm:
        stats["VGM"] = rating.style_score_vgm
    if rating.industry_rank_text:
        stats["Industry Rank"] = rating.industry_rank_text
    return {
        "symbol": rating.symbol,
        "zacks_rank": str(rating.rank) if rating.rank is not None else "N/A",
        "rank_text": rating.rank_text or "",
        "price_target": (
            f"${rating.price_target:,.2f}" if rating.price_target is not None else None
        ),
        "stats": stats,
        "source": "engine",
    }


def _insider_to_row(trade: Any) -> dict[str, Any]:
    """Map an engine ``InsiderTrade`` to the terminal table's key names."""
    return {
        "filing_date": trade.filing_date.isoformat() if trade.filing_date else "",
        "trade_date": trade.trade_date.isoformat() if trade.trade_date else "",
        "ticker": trade.ticker,
        "company": trade.company or "",
        "insider_name": trade.insider_name,
        "title": trade.insider_title or "",
        "trade_type": trade.trade_type,
        "price": trade.price_per_share if trade.price_per_share is not None else "",
        "qty": trade.quantity,
        "owned": trade.shares_owned_after if trade.shares_owned_after is not None else "",
        "value": trade.dollar_value if trade.dollar_value is not None else "",
        "source": "engine",
    }
