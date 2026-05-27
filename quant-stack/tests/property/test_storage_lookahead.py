"""Look-ahead property test — Principle #1 made concrete.

For ANY set of equity bars with arbitrary ``as_of`` times, and ANY decision
time, a ``PointInTimeQuery(as_of=decision_time)`` MUST NEVER return a bar
with ``as_of > decision_time``.

Also verifies the dual: every bar that SHOULD be visible (as_of <= decision_time)
is in the result. No silent drops.

Uses ``hypothesis`` to randomize the input.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ingestion.schema import EquityBar
from storage.query import PointInTimeQuery
from storage.store import ParquetStore

_SYMBOLS = ("AAA", "BBB", "CCC")
_INTERVAL = "1d"


@st.composite
def _equity_bars(draw: Any, *, max_size: int = 25) -> list[EquityBar]:
    n = draw(st.integers(min_value=1, max_value=max_size))
    bars: list[EquityBar] = []
    seen: set[tuple[str, datetime]] = set()
    for _ in range(n):
        sym = draw(st.sampled_from(_SYMBOLS))
        ts = draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2025, 12, 31),
                timezones=st.just(UTC),
            )
        )
        # Microsecond precision in the timestamp → enough uniqueness for the
        # primary key. If hypothesis collides we let the store dedup.
        if (sym, ts) in seen:
            continue
        seen.add((sym, ts))
        open_v = draw(
            st.floats(min_value=1.0, max_value=1000.0,
                      allow_nan=False, allow_infinity=False)
        )
        delta = draw(
            st.floats(min_value=0.0, max_value=20.0,
                      allow_nan=False, allow_infinity=False)
        )
        # Construct OHLC consistent with the schema's high>=low semantics.
        high = open_v + delta
        low_delta = draw(
            st.floats(min_value=0.0,
                      max_value=min(20.0, max(open_v - 0.01, 0.01)),
                      allow_nan=False, allow_infinity=False)
        )
        low = max(0.01, open_v - low_delta)
        close = draw(
            st.floats(min_value=low, max_value=high,
                      allow_nan=False, allow_infinity=False)
        )
        volume = draw(st.integers(min_value=0, max_value=10_000_000))
        bars.append(
            EquityBar(
                as_of=ts,
                symbol=sym,
                interval=_INTERVAL,
                open=round(open_v, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=volume,
                source="schwab",
            )
        )
    return bars


@given(
    bars=_equity_bars(),
    decision_time=st.datetimes(
        min_value=datetime(2019, 1, 1),
        max_value=datetime(2027, 1, 1),
        timezones=st.just(UTC),
    ),
)
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)
def test_no_lookahead_leakage(bars: list[EquityBar], decision_time: datetime) -> None:
    """For any bars + any decision_time, no returned row has as_of > decision_time.

    Also checks the dual: every (symbol, as_of) with as_of <= decision_time
    appears in the result.
    """
    # Fresh tmp dir per hypothesis iteration so examples don't leak into
    # each other. ``tmp_path`` is function-scoped and would accumulate.
    with tempfile.TemporaryDirectory() as td:
        store = ParquetStore(Path(td), env="prop")
        store.write_equity_bars(bars)

        q = PointInTimeQuery(store, as_of=decision_time)

        expected = {(b.symbol, b.as_of) for b in bars if b.as_of <= decision_time}
        actual: set[tuple[str, datetime]] = set()
        for sym in set(_SYMBOLS):
            for row in q.equity_bars(sym):
                # Primary invariant: NEVER return a future row.
                assert row.as_of <= decision_time, (
                    f"LEAK: bar at {row.as_of} returned for decision_time={decision_time}"
                )
                actual.add((row.symbol, row.as_of))

        # Dual: nothing legitimate is silently dropped.
        missing = expected - actual
        extra = actual - expected
        assert not missing, f"missing visible rows: {missing}"
        assert not extra, f"extra rows that shouldn't exist: {extra}"


def test_decision_time_far_future_returns_all() -> None:
    """Sanity check: decision_time in the far future → every bar is visible."""
    with tempfile.TemporaryDirectory() as td:
        store = ParquetStore(Path(td), env="prop")
        bars = [
            EquityBar(
                as_of=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
                symbol="X", interval="1d",
                open=10.0, high=11.0, low=9.0, close=10.5,
                volume=1000, source="schwab",
            ),
            EquityBar(
                as_of=datetime(2024, 1, 3, 21, 0, tzinfo=UTC),
                symbol="X", interval="1d",
                open=10.0, high=12.0, low=9.0, close=11.0,
                volume=2000, source="schwab",
            ),
        ]
        store.write_equity_bars(bars)
        q = PointInTimeQuery(store, as_of=datetime(2030, 1, 1, tzinfo=UTC))
        assert len(q.equity_bars("X")) == 2


def test_decision_time_far_past_returns_nothing() -> None:
    with tempfile.TemporaryDirectory() as td:
        store = ParquetStore(Path(td), env="prop")
        store.write_equity_bars(
            [
                EquityBar(
                    as_of=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
                    symbol="X", interval="1d",
                    open=10.0, high=11.0, low=9.0, close=10.5,
                    volume=1000, source="schwab",
                )
            ]
        )
        q = PointInTimeQuery(store, as_of=datetime(2010, 1, 1, tzinfo=UTC))
        assert q.equity_bars("X") == []
