"""Event-driven, naked-only options simulator over chain snapshots.

Feed it an ascending sequence of :class:`~ingestion.schema.OptionsChainSnapshot`
and a strategy that, at each snapshot, returns the *book it wants to hold*.
The simulator diffs that against the current book and trades the difference:

- **Fills at the quoted side** — buy at ask, sell at bid. The spread is the
  slippage; there is no separate slippage parameter to set to zero.
- **Marks at mid** each snapshot (``last`` if there is no two-sided quote;
  an error if there is neither — a held contract with no price is not a
  position, it's a guess).
- **Settles at expiry** at intrinsic value, cash-equivalent. A short put
  assigned or a covered call called away has the same P&L as cash settlement
  at intrinsic; the share leg is not carried past expiry.
- **Refuses any book that is not naked-only.** A short call needs 100 shares
  per contract in the same book (covered call); a short put needs its strike
  notional held in cash (cash-secured put). Anything else raises
  :class:`NakedOnlyError` and the simulation stops. This is the code-level
  twin of the schema's ``AllowedStructure`` gate.

No look-ahead: the strategy sees the chain at ``as_of`` and trades at that
snapshot's quotes. It never sees the next snapshot.

The ledger has the same columns as :mod:`backtest.engine`, so
``compute_metrics``, ``deflated_sharpe``, ``regime_report`` and
``health_check`` apply unchanged. ``position`` is 1.0 on bars with any open
position (exposure), not a leverage fraction; ``turnover`` and ``costs`` are
premium traded and commissions as fractions of prior equity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import pairwise

import pandas as pd

from backtest.config import BacktestConfig
from backtest.result import BacktestResult
from ingestion.schema import OptionContract, OptionRight, OptionsChainSnapshot

# One option contract controls this many shares.
MULTIPLIER = 100

_MIN_SNAPSHOTS = 2


class NakedOnlyError(ValueError):
    """The target book contains a structure outside the naked-only allowlist."""


class MarkError(ValueError):
    """A held or traded contract has no usable price in the current chain."""


@dataclass(frozen=True, order=True)
class ContractKey:
    """Identifies one contract across snapshots."""

    expiration: date
    strike: float
    right: OptionRight

    @classmethod
    def of(cls, c: OptionContract) -> ContractKey:
        return cls(c.expiration, c.strike, c.right)


@dataclass(frozen=True)
class Book:
    """A target or current holding: signed contract counts plus shares.

    ``options[key] > 0`` is long that many contracts; ``< 0`` is short.
    """

    options: Mapping[ContractKey, int] = field(default_factory=dict)
    shares: int = 0

    def with_option(self, key: ContractKey, quantity: int) -> Book:
        """A copy with ``key`` set to ``quantity`` (0 removes it)."""
        opts = {k: q for k, q in self.options.items() if k != key}
        if quantity != 0:
            opts[key] = quantity
        return Book(options=opts, shares=self.shares)

    def with_shares(self, shares: int) -> Book:
        return Book(options=dict(self.options), shares=shares)

    @property
    def is_flat(self) -> bool:
        return not self.options and self.shares == 0


@dataclass(frozen=True)
class Portfolio:
    """Read-only view a strategy gets each snapshot: what it holds and can afford.

    Attributes:
        book: Current holdings.
        cash: Cash on hand before this snapshot's trades.
        equity: Cash plus holdings marked at this snapshot's quotes.
    """

    book: Book
    cash: float
    equity: float

    def can_secure_puts(self, strike: float, contracts: int = 1) -> bool:
        """True if ``cash`` covers the strike notional for a cash-secured put."""
        return self.cash >= strike * MULTIPLIER * contracts


StrategyFn = Callable[[OptionsChainSnapshot, Portfolio], Book]


@dataclass(frozen=True)
class SimConfig:
    """Economics of the simulation.

    Attributes:
        initial_capital: Starting cash.
        commission_per_contract: Charged on every contract traded, each way.
        periods_per_year: Snapshots per year, for annualization.
    """

    initial_capital: float = 10_000.0
    commission_per_contract: float = 0.65
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.initial_capital <= 0.0:
            msg = f"initial_capital must be > 0, got {self.initial_capital}"
            raise ValueError(msg)
        if self.commission_per_contract < 0.0:
            msg = f"commission_per_contract must be >= 0, got {self.commission_per_contract}"
            raise ValueError(msg)
        if self.periods_per_year <= 0:
            msg = f"periods_per_year must be > 0, got {self.periods_per_year}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Fill:
    """One execution. ``key`` is ``None`` for a share trade."""

    as_of: datetime
    key: ContractKey | None
    quantity: int
    price: float
    commission: float


@dataclass(frozen=True)
class Settlement:
    """One contract settled at expiry."""

    as_of: datetime
    key: ContractKey
    quantity: int
    intrinsic: float
    settle_spot: float


@dataclass(frozen=True)
class SimulationResult:
    """What a run produced."""

    result: BacktestResult
    fills: tuple[Fill, ...]
    settlements: tuple[Settlement, ...]
    final_book: Book


@dataclass
class _State:
    """Mutable running state; internal to :func:`simulate`."""

    cash: float
    book: Book
    fills: list[Fill] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)


def _mark(c: OptionContract) -> float:
    if c.bid > 0.0 or c.ask > 0.0:
        return (c.bid + c.ask) / 2.0
    if c.last is not None and c.last > 0.0:
        return c.last
    msg = f"no usable price for {ContractKey.of(c)} at {c.as_of.isoformat()}"
    raise MarkError(msg)


def _intrinsic(key: ContractKey, spot: float) -> float:
    if key.right is OptionRight.call:
        return max(spot - key.strike, 0.0)
    return max(key.strike - spot, 0.0)


def _enforce_naked_only(book: Book, cash: float) -> None:
    """Raise unless ``book`` is naked-only and affordable. Order matters: an
    unaffordable book is a cash problem, not a structure problem."""
    if cash < 0.0:
        msg = f"cash would be negative (${cash:,.0f}); no margin lending"
        raise ValueError(msg)
    short_calls = sum(
        -q for k, q in book.options.items() if q < 0 and k.right is OptionRight.call
    )
    if short_calls * MULTIPLIER > book.shares:
        msg = (
            f"{short_calls} short call(s) need {short_calls * MULTIPLIER} shares to be "
            f"covered; book holds {book.shares}. Naked short calls are not allowed."
        )
        raise NakedOnlyError(msg)
    reserve = sum(
        -q * k.strike * MULTIPLIER
        for k, q in book.options.items()
        if q < 0 and k.right is OptionRight.put
    )
    if cash < reserve:
        msg = (
            f"short puts need ${reserve:,.0f} held in cash to be cash-secured; "
            f"cash after trades is ${cash:,.0f}. Naked short puts are not allowed."
        )
        raise NakedOnlyError(msg)


def _validate_snapshots(snapshots: Sequence[OptionsChainSnapshot]) -> None:
    if len(snapshots) < _MIN_SNAPSHOTS:
        msg = f"need >= {_MIN_SNAPSHOTS} snapshots, got {len(snapshots)}"
        raise ValueError(msg)
    underlying = snapshots[0].underlying
    for prev, cur in pairwise(snapshots):
        if cur.as_of <= prev.as_of:
            msg = f"snapshots must be strictly ascending: {prev.as_of} then {cur.as_of}"
            raise ValueError(msg)
        if cur.underlying != underlying:
            msg = f"mixed underlyings: {underlying} and {cur.underlying}"
            raise ValueError(msg)


def _settle_expired(st: _State, as_of: datetime, settle_spot: float) -> None:
    """Cash-settle every contract whose expiration is before ``as_of``'s date."""
    today = as_of.date()
    for key, qty in list(st.book.options.items()):
        if key.expiration < today:
            intrinsic = _intrinsic(key, settle_spot)
            st.cash += qty * intrinsic * MULTIPLIER
            st.settlements.append(Settlement(as_of, key, qty, intrinsic, settle_spot))
            st.book = st.book.with_option(key, 0)


def _trade_to_target(
    st: _State,
    target: Book,
    quotes: Mapping[ContractKey, OptionContract],
    spot: float,
    as_of: datetime,
    cfg: SimConfig,
) -> tuple[float, float]:
    """Trade from the current book to ``target``. Returns (premium traded, commissions)."""
    premium_traded = 0.0
    commissions = 0.0
    for key in sorted(set(st.book.options) | set(target.options)):
        delta = target.options.get(key, 0) - st.book.options.get(key, 0)
        if delta == 0:
            continue
        c = quotes.get(key)
        if c is None:
            msg = f"{key} not in chain at {as_of.isoformat()}"
            raise MarkError(msg)
        price = c.ask if delta > 0 else c.bid
        if price <= 0.0:
            msg = f"no {'ask' if delta > 0 else 'bid'} for {key} at {as_of.isoformat()}"
            raise MarkError(msg)
        notional = delta * price * MULTIPLIER
        commission = abs(delta) * cfg.commission_per_contract
        st.cash -= notional + commission
        premium_traded += abs(notional)
        commissions += commission
        st.fills.append(Fill(as_of, key, delta, price, commission))
    d_shares = target.shares - st.book.shares
    if d_shares != 0:
        st.cash -= d_shares * spot
        premium_traded += abs(d_shares) * spot
        st.fills.append(Fill(as_of, None, d_shares, spot, 0.0))
    st.book = Book(options=dict(target.options), shares=target.shares)
    return premium_traded, commissions


def _options_value(
    book: Book, quotes: Mapping[ContractKey, OptionContract], as_of: datetime
) -> float:
    value = 0.0
    for key, qty in book.options.items():
        c = quotes.get(key)
        if c is None:
            msg = f"held contract {key} missing from chain at {as_of.isoformat()}"
            raise MarkError(msg)
        value += qty * _mark(c) * MULTIPLIER
    return value


def simulate(
    snapshots: Sequence[OptionsChainSnapshot],
    strategy: StrategyFn,
    cfg: SimConfig | None = None,
) -> SimulationResult:
    """Run ``strategy`` over ``snapshots``.

    Args:
        snapshots: Chain snapshots, strictly ascending ``as_of``, one underlying.
        strategy: ``strategy(chain, portfolio) -> target_book``. The
            :class:`Portfolio` shows current holdings, cash and marked equity
            so the strategy can size within what it can actually afford.
        cfg: Economics. Defaults to :class:`SimConfig`.

    Raises:
        NakedOnlyError: The strategy asked for a disallowed structure.
        MarkError: A contract to trade or hold has no usable price.
        ValueError: Bad snapshots, or cash would go negative.
    """
    if cfg is None:
        cfg = SimConfig()
    _validate_snapshots(snapshots)

    st = _State(cash=cfg.initial_capital, book=Book())
    rows: list[dict[str, float]] = []
    prev_equity = cfg.initial_capital
    prev_spot: float | None = None

    for chain in snapshots:
        spot = chain.underlying_price
        quotes = {ContractKey.of(c): c for c in chain.contracts}

        # Expiries settle at the last spot seen on or before expiry — the
        # previous snapshot's — falling back to this one if there was none.
        _settle_expired(st, chain.as_of, prev_spot if prev_spot is not None else spot)
        pre_trade_equity = (
            st.cash + _options_value(st.book, quotes, chain.as_of) + st.book.shares * spot
        )
        target = strategy(chain, Portfolio(book=st.book, cash=st.cash, equity=pre_trade_equity))
        premium_traded, commissions = _trade_to_target(st, target, quotes, spot, chain.as_of, cfg)
        _enforce_naked_only(st.book, st.cash)

        equity = st.cash + _options_value(st.book, quotes, chain.as_of) + st.book.shares * spot
        if equity <= 0.0:
            msg = f"equity went to ${equity:,.0f} at {chain.as_of.isoformat()}"
            raise ValueError(msg)
        net = math.log(equity / prev_equity)
        costs = commissions / prev_equity
        rows.append(
            {
                "returns": 0.0 if prev_spot is None else math.log(spot / prev_spot),
                "position": 0.0 if st.book.is_flat else 1.0,
                "turnover": premium_traded / prev_equity,
                "gross": net + costs,
                "carry": 0.0,
                "costs": costs,
                "net": net,
                "equity": equity,
            }
        )
        prev_equity = equity
        prev_spot = spot

    ledger = pd.DataFrame(rows, index=pd.DatetimeIndex([s.as_of for s in snapshots]))
    bt_cfg = BacktestConfig(
        initial_capital=cfg.initial_capital,
        fee_bps=0.0,
        slippage_bps=0.0,
        periods_per_year=cfg.periods_per_year,
    )
    return SimulationResult(
        result=BacktestResult.from_ledger(ledger, bt_cfg),
        fills=tuple(st.fills),
        settlements=tuple(st.settlements),
        final_book=st.book,
    )
