"""Naked-only options simulator tests.

The two things that matter most: the naked-only gate cannot be bypassed,
and P&L reconciles to bid/ask fills, mid marks, and intrinsic settlement.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from backtest import compute_metrics
from ingestion.schema import OptionContract, OptionRight, OptionsChainSnapshot
from options import (
    Book,
    ContractKey,
    MarkError,
    NakedOnlyError,
    Portfolio,
    SimConfig,
    simulate,
    synthetic_chain,
)

_T0 = datetime(2026, 3, 2, 21, 0, tzinfo=UTC)
_EXP = date(2026, 3, 20)
_STRIKES = [90.0, 95.0, 100.0, 105.0, 110.0]

_CALL_100 = ContractKey(_EXP, 100.0, OptionRight.call)
_PUT_95 = ContractKey(_EXP, 95.0, OptionRight.put)


def _chains(spots: list[float], start: datetime = _T0) -> list[OptionsChainSnapshot]:
    return [
        synthetic_chain(start + timedelta(days=i), "QQQ", s, [_EXP], _STRIKES, iv=0.25)
        for i, s in enumerate(spots)
    ]


def _quote(chain: OptionsChainSnapshot, key: ContractKey) -> OptionContract:
    return next(c for c in chain.contracts if ContractKey.of(c) == key)


def _buy_call_once(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
    return pf.book if pf.book.options else pf.book.with_option(_CALL_100, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  P&L reconciliation
# ─────────────────────────────────────────────────────────────────────────────


def test_long_call_fills_at_ask_marks_at_mid_and_gains_on_rally() -> None:
    chains = _chains([100.0, 105.0])
    cfg = SimConfig(initial_capital=10_000.0, commission_per_contract=0.65)
    sim = simulate(chains, _buy_call_once, cfg)

    q0 = _quote(chains[0], _CALL_100)
    q1 = _quote(chains[1], _CALL_100)
    mid0 = (q0.bid + q0.ask) / 2
    mid1 = (q1.bid + q1.ask) / 2
    # Day 0: paid the ask + commission, marked at mid immediately (spread loss).
    expected_day0 = 10_000.0 - q0.ask * 100 - 0.65 + mid0 * 100
    assert sim.result.ledger["equity"].iloc[0] == pytest.approx(expected_day0)
    # Day 1: cash unchanged, marked at the new mid.
    expected_day1 = 10_000.0 - q0.ask * 100 - 0.65 + mid1 * 100
    assert sim.result.ledger["equity"].iloc[1] == pytest.approx(expected_day1)
    assert expected_day1 > expected_day0
    assert len(sim.fills) == 1
    assert sim.fills[0].quantity == 1
    assert sim.fills[0].price == q0.ask


def test_commissions_are_charged_and_reported() -> None:
    sim = simulate(_chains([100.0, 100.0]), _buy_call_once, SimConfig(commission_per_contract=2.0))
    assert sim.fills[0].commission == 2.0
    assert sim.result.ledger["costs"].iloc[0] == pytest.approx(2.0 / 10_000.0)


def test_expiry_settles_at_intrinsic_and_removes_contract() -> None:
    # Hold the 100 call through expiry; spot ends at 108 -> $800 intrinsic.
    start = datetime(2026, 3, 18, 21, 0, tzinfo=UTC)
    chains = [
        synthetic_chain(start, "QQQ", 100.0, [_EXP], _STRIKES),
        synthetic_chain(start + timedelta(days=2), "QQQ", 108.0, [_EXP], _STRIKES),  # expiry day
        synthetic_chain(start + timedelta(days=3), "QQQ", 120.0, [date(2026, 4, 17)], _STRIKES),
    ]

    def buy_on_first_day_only(chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        return pf.book.with_option(_CALL_100, 1) if chain is chains[0] else pf.book

    sim = simulate(chains, buy_on_first_day_only, SimConfig(commission_per_contract=0.0))
    assert len(sim.settlements) == 1
    s = sim.settlements[0]
    assert s.key == _CALL_100
    assert s.quantity == 1
    # Settled at the last spot on/before expiry (108), not the post-expiry 120.
    assert s.settle_spot == 108.0
    assert s.intrinsic == pytest.approx(8.0)
    assert sim.final_book.is_flat
    paid = _quote(chains[0], _CALL_100).ask * 100
    assert sim.result.ledger["equity"].iloc[-1] == pytest.approx(10_000.0 - paid + 800.0)


def test_short_put_receives_bid_and_is_cash_secured() -> None:
    chains = _chains([100.0, 100.0])

    def csp(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        return pf.book if pf.book.options else pf.book.with_option(_PUT_95, -1)

    sim = simulate(chains, csp, SimConfig(initial_capital=20_000.0, commission_per_contract=0.0))
    q0 = _quote(chains[0], _PUT_95)
    assert sim.fills[0].price == q0.bid
    assert sim.fills[0].quantity == -1


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio view
# ─────────────────────────────────────────────────────────────────────────────


def test_strategy_sees_cash_and_equity() -> None:
    seen: list[Portfolio] = []

    def spy(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        seen.append(pf)
        return pf.book if pf.book.options else pf.book.with_option(_CALL_100, 1)

    chains = _chains([100.0, 105.0])
    simulate(chains, spy, SimConfig(initial_capital=10_000.0, commission_per_contract=0.0))
    assert seen[0].cash == 10_000.0
    assert seen[0].equity == 10_000.0
    assert seen[0].book.is_flat
    # Day 1: cash is down the premium paid; the call is held.
    ask0 = _quote(chains[0], _CALL_100).ask
    assert seen[1].cash == pytest.approx(10_000.0 - ask0 * 100)
    assert seen[1].book.options[_CALL_100] == 1
    assert seen[1].equity > seen[1].cash


def test_can_secure_puts_helper() -> None:
    pf = Portfolio(book=Book(), cash=9_500.0, equity=9_500.0)
    assert pf.can_secure_puts(95.0)
    assert not pf.can_secure_puts(96.0)
    assert not pf.can_secure_puts(95.0, contracts=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Naked-only gate
# ─────────────────────────────────────────────────────────────────────────────


def test_naked_short_call_is_refused() -> None:
    def naked(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        return pf.book.with_option(_CALL_100, -1)

    with pytest.raises(NakedOnlyError, match="short call"):
        simulate(_chains([100.0, 100.0]), naked, SimConfig(initial_capital=100_000.0))


def test_covered_call_is_allowed_with_shares() -> None:
    def covered(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        if pf.book.options:
            return pf.book
        return pf.book.with_shares(100).with_option(_CALL_100, -1)

    sim = simulate(_chains([100.0, 100.0]), covered, SimConfig(initial_capital=20_000.0))
    assert sim.final_book.shares == 100
    assert sim.final_book.options[_CALL_100] == -1


def test_short_put_without_cash_is_refused() -> None:
    def csp(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        return pf.book.with_option(_PUT_95, -1)  # needs $9,500 cash

    with pytest.raises(NakedOnlyError, match="cash-secured"):
        simulate(_chains([100.0, 100.0]), csp, SimConfig(initial_capital=5_000.0))


def test_cannot_spend_more_cash_than_held() -> None:
    def overspend(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        return pf.book.with_option(_CALL_100, 500)  # ~$200k of premium

    with pytest.raises(ValueError, match="negative"):
        simulate(_chains([100.0, 100.0]), overspend, SimConfig(initial_capital=10_000.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Marks and validation
# ─────────────────────────────────────────────────────────────────────────────


def test_unknown_contract_raises_mark_error() -> None:
    def bad(_chain: OptionsChainSnapshot, pf: Portfolio) -> Book:
        return pf.book.with_option(ContractKey(_EXP, 123.0, OptionRight.call), 1)

    with pytest.raises(MarkError, match="not in chain"):
        simulate(_chains([100.0, 100.0]), bad)


def test_held_contract_vanishing_from_chain_is_loud() -> None:
    # Buy the Mar 100 call, then the next chain only lists Apr: the held
    # contract has no mark, and that is an error, not a silent zero.
    chains = [
        synthetic_chain(_T0, "QQQ", 100.0, [_EXP], _STRIKES),
        synthetic_chain(_T0 + timedelta(days=1), "QQQ", 100.0, [date(2026, 4, 17)], _STRIKES),
    ]
    with pytest.raises(MarkError, match="missing from chain"):
        simulate(chains, _buy_call_once)


def test_rejects_unsorted_or_mixed_snapshots() -> None:
    a, b = _chains([100.0, 100.0])
    with pytest.raises(ValueError, match="ascending"):
        simulate([b, a], _buy_call_once)
    other = synthetic_chain(b.as_of + timedelta(days=1), "SPY", 100.0, [_EXP], _STRIKES)
    with pytest.raises(ValueError, match="mixed"):
        simulate([a, b, other], _buy_call_once)


def test_ledger_is_downstream_compatible() -> None:
    sim = simulate(_chains([100.0, 101.0, 99.0, 103.0]), _buy_call_once)
    ledger = sim.result.ledger
    for col in ("returns", "position", "turnover", "gross", "carry", "costs", "net", "equity"):
        assert col in ledger.columns
    m = compute_metrics(ledger, sim.result.config)
    assert m.n_periods == 4
    assert (ledger["position"] == 1.0).all()  # a call is held from day 0


def test_flat_strategy_has_flat_equity() -> None:
    sim = simulate(_chains([100.0, 110.0, 90.0]), lambda _c, pf: pf.book)
    assert (sim.result.ledger["equity"] == 10_000.0).all()
    assert sim.fills == ()
