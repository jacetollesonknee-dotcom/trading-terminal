"""Pydantic schemas for every record this engine ingests, stores, or routes.

Two rules govern this module:

1. **Every record carries ``as_of: datetime`` (UTC).** This is the point-in-time
   stamp the backtester uses to enforce no look-ahead bias. A record with no
   ``as_of`` is unstorable.

2. **Records are immutable.** ``model_config["frozen"] = True`` everywhere.
   To "update" a record, write a new one with a later ``as_of``.

UTC at storage; presentation-layer code in the terminal renders MST. Never
convert at the storage boundary.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─────────────────────────────────────────────────────────────────────────────
#  Common
# ─────────────────────────────────────────────────────────────────────────────


class _ImmutableModel(BaseModel):
    """Base for all ingested records: immutable + strict + UTC-aware timestamps."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _ensure_utc(v: datetime) -> datetime:
    """Reject naive datetimes; everything stored is UTC."""
    if v.tzinfo is None:
        msg = "datetime must be timezone-aware (UTC). got naive."
        raise ValueError(msg)
    return v


_AsOf = Annotated[
    datetime,
    Field(description="Point-in-time the record was true. UTC. Mandatory."),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Reference types
# ─────────────────────────────────────────────────────────────────────────────


class OptionRight(StrEnum):
    call = "CALL"
    put = "PUT"


class OrderSide(StrEnum):
    buy = "BUY"
    sell = "SELL"


class OrderStatus(StrEnum):
    pending = "PENDING"
    submitted = "SUBMITTED"
    filled = "FILLED"
    partial = "PARTIAL"
    cancelled = "CANCELLED"
    rejected = "REJECTED"


class AssetClass(StrEnum):
    equity = "EQUITY"
    option = "OPTION"


# Permitted option structures for v1 (naked-only). Phase 8 widens this set;
# until then, schema validation refuses anything else.
AllowedStructure = Literal[
    "long_call",
    "long_put",
    "cash_secured_put",
    "covered_call",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Market data
# ─────────────────────────────────────────────────────────────────────────────


class EquityBar(_ImmutableModel):
    """One OHLCV bar of a single equity at a single resolution."""

    as_of: _AsOf
    symbol: str = Field(min_length=1, max_length=12)
    interval: Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]
    open: float = Field(ge=0)
    high: float = Field(ge=0)
    low: float = Field(ge=0)
    close: float = Field(ge=0)
    volume: int = Field(ge=0)
    source: Literal["schwab", "yahoo"]

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @model_validator(mode="after")
    def _validate_high_low(self) -> EquityBar:
        # Field validators run in declaration order, so a field_validator on
        # `high` never sees `low` in info.data. Use a model validator instead.
        if self.high < self.low:
            msg = f"high={self.high} < low={self.low}"
            raise ValueError(msg)
        return self


class OptionContract(_ImmutableModel):
    """A single option contract, identified by its OCC-style fields."""

    as_of: _AsOf
    underlying: str = Field(min_length=1, max_length=12)
    expiration: date
    strike: float = Field(gt=0)
    right: OptionRight
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    last: float | None = Field(default=None, ge=0)
    volume: int = Field(default=0, ge=0)
    open_interest: int = Field(default=0, ge=0)
    iv: float | None = Field(default=None, ge=0)
    delta: float | None = Field(default=None, ge=-1, le=1)
    gamma: float | None = Field(default=None, ge=0)
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    source: Literal["schwab", "optionsdx", "yahoo"]

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    @field_validator("ask")
    @classmethod
    def _ask_ge_bid(cls, v: float, info) -> float:  # type: ignore[no-untyped-def]
        bid = info.data.get("bid")
        if bid is not None and v < bid:
            msg = f"ask={v} < bid={bid}"
            raise ValueError(msg)
        return v


class OptionsChainSnapshot(_ImmutableModel):
    """A full chain for one underlying at one ``as_of``.

    Stored once per nightly capture-forward pass and once per OptionsDX backfill day.
    """

    as_of: _AsOf
    underlying: str = Field(min_length=1, max_length=12)
    underlying_price: float = Field(gt=0)
    contracts: tuple[OptionContract, ...]
    source: Literal["schwab", "optionsdx", "yahoo"]

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


# ─────────────────────────────────────────────────────────────────────────────
#  Reference data
# ─────────────────────────────────────────────────────────────────────────────


class CorporateAction(_ImmutableModel):
    """Split, dividend, spin-off — anything that adjusts historical price/qty."""

    as_of: _AsOf
    symbol: str = Field(min_length=1, max_length=12)
    ex_date: date
    type: Literal["split", "cash_dividend", "stock_dividend", "spinoff", "merger"]
    ratio: float | None = Field(default=None, gt=0, description="e.g. 2.0 for 2-for-1")
    cash_amount: float | None = Field(default=None, ge=0)
    source: str

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


class EarningsEvent(_ImmutableModel):
    """One scheduled or reported earnings event."""

    as_of: _AsOf
    symbol: str = Field(min_length=1, max_length=12)
    report_date: date
    when: Literal["bmo", "amc", "during"]  # before/after market, or intraday
    eps_estimate: float | None = None
    eps_actual: float | None = None
    eps_surprise_pct: float | None = None
    source: str

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


# ─────────────────────────────────────────────────────────────────────────────
#  Sentiment / activity feeds
# ─────────────────────────────────────────────────────────────────────────────


class InsiderTrade(_ImmutableModel):
    """A single Form-4 insider transaction.

    Source v1: OpenInsider's HTML tables (Phase 1.4b). Primary key for
    dedup at the storage layer:
        (ticker, filing_date, trade_date, insider_name, trade_type,
         price_per_share, quantity).

    Fields can be None when OpenInsider leaves a cell blank — common for
    option-exercise rows where there's no per-share price, for instance.
    """

    as_of: _AsOf
    filing_date: date
    trade_date: date
    ticker: str = Field(min_length=1, max_length=12)
    company: str | None = None
    insider_name: str = Field(min_length=1)
    insider_title: str | None = None
    trade_type: str = Field(min_length=1)        # e.g. "P - Purchase", "S - Sale"
    price_per_share: float | None = Field(default=None, ge=0)
    quantity: int = Field(ge=0)
    shares_owned_after: int | None = Field(default=None, ge=0)
    dollar_value: float | None = None
    source: Literal["openinsider"]

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


# ─────────────────────────────────────────────────────────────────────────────
#  Account / portfolio state
# ─────────────────────────────────────────────────────────────────────────────


class AccountSnapshot(_ImmutableModel):
    """End-of-day account state — cash, equity, margin, buying power."""

    as_of: _AsOf
    account_id: str
    cash: float
    equity_market_value: float
    options_market_value: float
    total_value: float
    margin_balance: float = Field(ge=0)
    buying_power: float
    maintenance_requirement: float = Field(ge=0)
    pdt_flagged: bool
    source: Literal["schwab"]

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


class Position(_ImmutableModel):
    """One open position — equity or option lot — at a point in time."""

    as_of: _AsOf
    account_id: str
    symbol: str  # underlying for options
    asset_class: AssetClass
    quantity: float  # signed: long positive, short negative
    avg_cost: float = Field(gt=0)
    market_price: float = Field(ge=0)
    market_value: float
    unrealized_pnl: float

    # Options-only fields
    option_expiration: date | None = None
    option_strike: float | None = Field(default=None, gt=0)
    option_right: OptionRight | None = None

    source: Literal["schwab"]

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


# ─────────────────────────────────────────────────────────────────────────────
#  Orders + fills
# ─────────────────────────────────────────────────────────────────────────────


class Order(_ImmutableModel):
    """A submitted order. Immutable — state transitions become new Fill records."""

    as_of: _AsOf
    order_id: str
    account_id: str
    structure: AllowedStructure  # SCHEMA-LEVEL naked-only gate
    underlying: str
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: Literal["market", "limit"]
    limit_price: float | None = Field(default=None, gt=0)
    status: OrderStatus
    submitted_at: datetime
    source: Literal["paper", "live"]

    @field_validator("as_of", "submitted_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


class Fill(_ImmutableModel):
    """A fill event against an order. Multiple fills allowed per order (partials)."""

    as_of: _AsOf
    fill_id: str
    order_id: str
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    commission: float = Field(ge=0)
    fees: float = Field(ge=0)
    source: Literal["paper", "live"]

    @field_validator("as_of")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)
