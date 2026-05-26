"""Schema validation tests.

The schemas are the contract every other module relies on; if these break,
nothing downstream is trustworthy.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from ingestion.schema import (
    EquityBar,
    Fill,
    OptionContract,
    OptionRight,
    OptionsChainSnapshot,
    Order,
    OrderSide,
    OrderStatus,
)


def _now() -> datetime:
    return datetime.now(UTC)


# ─────────────────────────────────────────────────────────────────────────────
#  as_of mandatory + UTC
# ─────────────────────────────────────────────────────────────────────────────


def test_equity_bar_rejects_naive_datetime() -> None:
    naive = datetime.now()  # noqa: DTZ005 — intentionally naive
    with pytest.raises(ValidationError) as ei:
        EquityBar(
            as_of=naive,
            symbol="NVDA",
            interval="1d",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1_000_000,
            source="schwab",
        )
    assert "timezone-aware" in str(ei.value)


def test_equity_bar_ok_with_utc() -> None:
    bar = EquityBar(
        as_of=_now(),
        symbol="NVDA",
        interval="1d",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000_000,
        source="schwab",
    )
    assert bar.symbol == "NVDA"


def test_equity_bar_high_below_low_rejected() -> None:
    with pytest.raises(ValidationError):
        EquityBar(
            as_of=_now(),
            symbol="NVDA",
            interval="1d",
            open=100.0,
            high=98.0,  # < low
            low=99.0,
            close=100.5,
            volume=1_000_000,
            source="schwab",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Immutability
# ─────────────────────────────────────────────────────────────────────────────


def test_equity_bar_frozen() -> None:
    bar = EquityBar(
        as_of=_now(),
        symbol="NVDA",
        interval="1d",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000_000,
        source="schwab",
    )
    with pytest.raises(ValidationError):
        bar.close = 999.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
#  Options
# ─────────────────────────────────────────────────────────────────────────────


def test_option_contract_ask_below_bid_rejected() -> None:
    with pytest.raises(ValidationError):
        OptionContract(
            as_of=_now(),
            underlying="NVDA",
            expiration=date.today() + timedelta(days=30),
            strike=180.0,
            right=OptionRight.call,
            bid=2.50,
            ask=2.40,  # < bid
            source="schwab",
        )


def test_options_chain_snapshot_round_trip() -> None:
    now = _now()
    chain = OptionsChainSnapshot(
        as_of=now,
        underlying="NVDA",
        underlying_price=180.0,
        contracts=(
            OptionContract(
                as_of=now,
                underlying="NVDA",
                expiration=date.today() + timedelta(days=30),
                strike=180.0,
                right=OptionRight.call,
                bid=2.50,
                ask=2.55,
                source="schwab",
            ),
        ),
        source="schwab",
    )
    assert len(chain.contracts) == 1
    assert chain.contracts[0].right is OptionRight.call


# ─────────────────────────────────────────────────────────────────────────────
#  Orders — naked-only schema gate
# ─────────────────────────────────────────────────────────────────────────────


def test_order_accepts_naked_structures() -> None:
    for structure in ("long_call", "long_put", "cash_secured_put", "covered_call"):
        order = Order(
            as_of=_now(),
            order_id=f"test-{structure}",
            account_id="X",
            structure=structure,  # type: ignore[arg-type]
            underlying="NVDA",
            side=OrderSide.buy,
            quantity=1,
            order_type="market",
            status=OrderStatus.pending,
            submitted_at=_now(),
            source="paper",
        )
        assert order.structure == structure  # noqa: PLR2004 — covered by parametrize style


@pytest.mark.parametrize(
    "structure",
    ["vertical", "iron_condor", "naked_short_call", "short_strangle", "calendar"],
)
def test_order_rejects_disallowed_structures(structure: str) -> None:
    """Schema-level guardrail: anything outside the naked-only allowlist fails."""
    with pytest.raises(ValidationError):
        Order(
            as_of=_now(),
            order_id="x",
            account_id="X",
            structure=structure,  # type: ignore[arg-type]
            underlying="NVDA",
            side=OrderSide.sell,
            quantity=1,
            order_type="market",
            status=OrderStatus.pending,
            submitted_at=_now(),
            source="paper",
        )


def test_fill_quantity_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Fill(
            as_of=_now(),
            fill_id="f1",
            order_id="o1",
            quantity=0,
            price=10.0,
            commission=0.65,
            fees=0.0,
            source="paper",
        )
