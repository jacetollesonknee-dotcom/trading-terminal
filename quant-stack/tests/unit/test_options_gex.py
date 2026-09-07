"""GEX profile tests on synthetic chains."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from ingestion.schema import OptionContract, OptionRight, OptionsChainSnapshot
from options import gex_profile, synthetic_chain

_AS_OF = datetime(2026, 3, 2, 21, 0, tzinfo=UTC)
_EXP = date(2026, 3, 20)
_STRIKES = [90.0, 95.0, 100.0, 105.0, 110.0]


def test_walls_follow_open_interest() -> None:
    def oi(k: float, r: OptionRight) -> int:
        if r is OptionRight.call:
            return 5000 if k == 110.0 else 100
        return 4000 if k == 90.0 else 100

    chain = synthetic_chain(_AS_OF, "QQQ", 100.0, [_EXP], _STRIKES, open_interest=oi)
    prof = gex_profile(chain)
    assert prof.call_wall == 110.0
    assert prof.put_wall == 90.0
    assert prof.n_used == len(chain.contracts)
    assert prof.n_skipped == 0


def test_sign_convention_calls_positive_puts_negative() -> None:
    only_calls = synthetic_chain(
        _AS_OF, "QQQ", 100.0, [_EXP], _STRIKES,
        open_interest=lambda _k, r: 100 if r is OptionRight.call else 0,
    )
    only_puts = synthetic_chain(
        _AS_OF, "QQQ", 100.0, [_EXP], _STRIKES,
        open_interest=lambda _k, r: 100 if r is OptionRight.put else 0,
    )
    assert gex_profile(only_calls).net_gex > 0.0
    assert gex_profile(only_calls).dealer_gamma == "long"
    assert gex_profile(only_puts).net_gex < 0.0
    assert gex_profile(only_puts).dealer_gamma == "short"


def test_gamma_flip_between_put_heavy_low_and_call_heavy_high_strikes() -> None:
    # Puts dominate below spot, calls above -> cumulative net GEX starts
    # negative and turns positive somewhere in between.
    def oi(k: float, r: OptionRight) -> int:
        if r is OptionRight.put:
            return 1000 if k < 100.0 else 0
        return 1000 if k > 100.0 else 0

    prof = gex_profile(synthetic_chain(_AS_OF, "QQQ", 100.0, [_EXP], _STRIKES, open_interest=oi))
    assert prof.gamma_flip is not None
    assert 90.0 < prof.gamma_flip < 110.0


def test_no_flip_when_one_sided() -> None:
    prof = gex_profile(
        synthetic_chain(_AS_OF, "QQQ", 100.0, [_EXP], _STRIKES,
                        open_interest=lambda _k, r: 100 if r is OptionRight.call else 0)
    )
    assert prof.gamma_flip is None


def test_skips_contracts_without_gamma_and_raises_when_nothing_usable() -> None:
    bare = OptionsChainSnapshot(
        as_of=_AS_OF,
        underlying="QQQ",
        underlying_price=100.0,
        contracts=(
            OptionContract(as_of=_AS_OF, underlying="QQQ", expiration=_EXP, strike=100.0,
                           right=OptionRight.call, bid=1.0, ask=1.1, open_interest=100,
                           source="schwab"),  # no gamma
        ),
        source="schwab",
    )
    with pytest.raises(ValueError, match="no contract has both"):
        gex_profile(bare)


def test_by_strike_table_shape() -> None:
    prof = gex_profile(synthetic_chain(_AS_OF, "QQQ", 100.0, [_EXP], _STRIKES))
    assert list(prof.by_strike.columns) == ["call_gex", "put_gex", "net_gex", "call_oi", "put_oi"]
    assert list(prof.by_strike.index) == _STRIKES
