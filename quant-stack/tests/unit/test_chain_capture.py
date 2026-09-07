"""Chain capture tests, against a fake broker that satisfies the protocol."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from config.settings import Settings
from ingestion.brokers.registry import BrokerDisabledError
from ingestion.chain_capture import capture_chains, run_capture
from ingestion.schema import OptionsChainSnapshot
from options import synthetic_chain
from storage.store import ParquetStore

_AS_OF = datetime(2026, 3, 2, 21, 0, tzinfo=UTC)


class FakeBroker:
    """Minimal BrokerClient: returns a synthetic chain, or raises for a symbol."""

    name = "fake"

    def __init__(self, *, fail_for: str | None = None) -> None:
        self.fail_for = fail_for
        self.entered = 0
        self.exited = 0
        self.requested: list[str] = []

    async def __aenter__(self) -> FakeBroker:
        self.entered += 1
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.exited += 1

    async def get_options_chain(
        self,
        underlying: str,
        *,
        expiration: date | None = None,
        strike_range: tuple[float, float] | None = None,
    ) -> OptionsChainSnapshot:
        self.requested.append(underlying)
        if underlying == self.fail_for:
            msg = f"upstream rejected {underlying}"
            raise RuntimeError(msg)
        return synthetic_chain(
            _AS_OF, underlying, 100.0, [date(2026, 3, 20)], [95.0, 100.0, 105.0]
        )


def _store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path, env="test")


@pytest.mark.asyncio
async def test_captures_and_persists_each_underlying(tmp_path: Path) -> None:
    broker = FakeBroker()
    results = await capture_chains(broker, _store(tmp_path), ["qqq", " spy "])
    assert [r.underlying for r in results] == ["QQQ", "SPY"]
    assert all(r.ok for r in results)
    assert all(r.write_result is not None and r.write_result.persisted == 6 for r in results)
    assert broker.requested == ["QQQ", "SPY"]
    # One broker session for the whole batch.
    assert broker.entered == 1
    assert broker.exited == 1


@pytest.mark.asyncio
async def test_second_capture_same_day_is_deduplicated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = await capture_chains(FakeBroker(), store, ["QQQ"])
    second = await capture_chains(FakeBroker(), store, ["QQQ"])
    assert first[0].write_result is not None and first[0].write_result.persisted == 6
    assert second[0].write_result is not None
    assert second[0].write_result.persisted == 0
    assert second[0].write_result.deduplicated == 6


@pytest.mark.asyncio
async def test_one_bad_symbol_does_not_stop_the_batch(tmp_path: Path) -> None:
    broker = FakeBroker(fail_for="BAD")
    results = await capture_chains(broker, _store(tmp_path), ["QQQ", "BAD", "SPY"])
    by = {r.underlying: r for r in results}
    assert by["QQQ"].ok and by["SPY"].ok
    assert not by["BAD"].ok
    assert by["BAD"].error is not None and "upstream rejected BAD" in by["BAD"].error
    assert by["BAD"].write_result is None


@pytest.mark.asyncio
async def test_rejects_empty_underlyings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        await capture_chains(FakeBroker(), _store(tmp_path), ["", "  "])


def test_run_capture_with_injected_broker(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    results = run_capture(settings, _store(tmp_path), ["QQQ"], broker=FakeBroker())
    assert results[0].ok


def test_run_capture_refuses_disabled_broker(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    assert not settings.brokers_enabled.get("schwab", False)
    with pytest.raises(BrokerDisabledError, match="not enabled"):
        run_capture(settings, _store(tmp_path), ["QQQ"], broker_name="schwab")
