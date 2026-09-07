"""Fixed-fractional position sizing with an honest stop.

Size a trade so that, if the stop is hit, you lose a fixed fraction of
capital — then cap the notional so a tight stop can't turn into an outsized
position. The output is a signed *fraction of capital*, the same unit the
engine's ``position`` uses, so it plugs straight into a signal series.

Three things the textbook version assumes that crypto does not grant:

1. **Stops fill at the stop price.** They don't — wicks, gaps and thin books
   fill you worse. The loss budget here includes ``stop_slippage_bps`` on the
   exit and fees on both legs, so ``loss_if_stopped`` is what you'd actually
   lose, and the size is reduced to keep it inside ``risk_fraction``.
2. **Any stop is a valid stop.** A long with its stop *above* entry, or a
   short with its stop *below*, is a fat-finger, not a trade. ``side`` makes
   the intent explicit and the wrong-side stop is rejected.
3. **Quantities are continuous.** Exchanges enforce a lot step and a minimum
   order size. ``lot_step`` floors the quantity (never rounds up — that
   would exceed the risk budget); ``min_notional`` zeroes a size the venue
   would reject rather than returning an untradeable number.

The slippage term also regularizes the degenerate case: with a stop a
float-noise distance from entry, effective risk per unit is still at least
the slippage, so the size stays bounded instead of exploding into the cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

# Basis-point divisor.
_BPS = 1e4


class Side(StrEnum):
    """Direction of the trade being sized."""

    long = "long"
    short = "short"


@dataclass(frozen=True)
class SizingConfig:
    """Risk budget and venue constraints.

    Attributes:
        risk_fraction: Fraction of capital lost if the stop fills (costs
            included). ``0.01`` = risk 1% per trade.
        max_position_fraction: Cap on ``notional / capital``. Keep this at or
            below the engine's ``max_leverage`` — anything above it would be
            clipped there anyway.
        stop_slippage_bps: How much worse than the stop price the exit is
            assumed to fill, in bps of the stop price.
        fee_bps: Exchange fee per side, in bps of notional. Charged on entry
            and on the stop exit.
        lot_step: Venue quantity increment (e.g. ``0.001`` BTC). Quantity is
            floored to a multiple of it. ``None`` = continuous.
        min_notional: Venue minimum order value in quote currency. A size
            below it is zeroed. ``None`` = no minimum.
    """

    risk_fraction: float = 0.01
    max_position_fraction: float = 0.20
    stop_slippage_bps: float = 3.0
    fee_bps: float = 5.0
    lot_step: float | None = None
    min_notional: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.risk_fraction <= 1.0:
            msg = f"risk_fraction must be in (0, 1], got {self.risk_fraction}"
            raise ValueError(msg)
        if self.max_position_fraction <= 0.0:
            msg = f"max_position_fraction must be > 0, got {self.max_position_fraction}"
            raise ValueError(msg)
        if self.stop_slippage_bps < 0.0:
            msg = f"stop_slippage_bps must be >= 0, got {self.stop_slippage_bps}"
            raise ValueError(msg)
        if self.fee_bps < 0.0:
            msg = f"fee_bps must be >= 0, got {self.fee_bps}"
            raise ValueError(msg)
        if self.lot_step is not None and self.lot_step <= 0.0:
            msg = f"lot_step must be > 0, got {self.lot_step}"
            raise ValueError(msg)
        if self.min_notional is not None and self.min_notional < 0.0:
            msg = f"min_notional must be >= 0, got {self.min_notional}"
            raise ValueError(msg)


@dataclass(frozen=True)
class PositionSize:
    """A sized trade, with every adjustment that was applied made visible.

    Attributes:
        side: Direction.
        units: Unsigned quantity to trade.
        notional: ``units * entry`` in quote currency.
        position_fraction: SIGNED ``notional / capital`` — negative for a
            short. This is the value to place in an engine signal series.
        loss_if_stopped: Quote-currency loss if the stop fills, including
            slippage and both fees.
        risk_fraction_realized: ``loss_if_stopped / capital``. Equals the
            configured ``risk_fraction`` only when nothing below bound.
        capped: ``max_position_fraction`` reduced the size.
        floored: ``lot_step`` reduced the size.
        below_min_notional: The size was zeroed by ``min_notional``.
    """

    side: Side
    units: float
    notional: float
    position_fraction: float
    loss_if_stopped: float
    risk_fraction_realized: float
    capped: bool
    floored: bool
    below_min_notional: bool


def _validate_trade(capital: float, entry: float, stop: float, side: Side) -> None:
    if not (math.isfinite(capital) and capital > 0.0):
        msg = f"capital must be a positive finite number, got {capital}"
        raise ValueError(msg)
    if not (math.isfinite(entry) and entry > 0.0):
        msg = f"entry must be a positive finite price, got {entry}"
        raise ValueError(msg)
    if not (math.isfinite(stop) and stop > 0.0):
        msg = f"stop must be a positive finite price, got {stop}"
        raise ValueError(msg)
    if stop == entry:
        msg = "stop cannot equal entry"
        raise ValueError(msg)
    if side is Side.long and stop > entry:
        msg = f"long stop must be below entry: stop={stop} > entry={entry}"
        raise ValueError(msg)
    if side is Side.short and stop < entry:
        msg = f"short stop must be above entry: stop={stop} < entry={entry}"
        raise ValueError(msg)


def position_size(
    capital: float,
    entry: float,
    stop: float,
    side: Side,
    cfg: SizingConfig | None = None,
) -> PositionSize:
    """Size a trade to risk ``cfg.risk_fraction`` of capital if the stop fills.

    Args:
        capital: Account equity in quote currency.
        entry: Intended entry price.
        stop: Stop price. Must be below ``entry`` for a long, above for a short.
        side: :class:`Side`.
        cfg: Risk budget and venue constraints. Defaults to :class:`SizingConfig`.

    Returns:
        A :class:`PositionSize`. Check ``capped`` / ``floored`` /
        ``below_min_notional`` — when any is set, ``risk_fraction_realized``
        is below the target and that is deliberate.
    """
    if cfg is None:
        cfg = SizingConfig()
    _validate_trade(capital, entry, stop, side)

    # Effective loss per unit if stopped: the stop distance, plus the exit
    # filling worse than the stop, plus a fee on each leg.
    distance = abs(entry - stop)
    slippage_per_unit = stop * cfg.stop_slippage_bps / _BPS
    fee_per_unit = (entry + stop) * cfg.fee_bps / _BPS
    risk_per_unit = distance + slippage_per_unit + fee_per_unit

    units = capital * cfg.risk_fraction / risk_per_unit

    cap_notional = capital * cfg.max_position_fraction
    capped = units * entry > cap_notional
    if capped:
        units = cap_notional / entry

    floored = False
    if cfg.lot_step is not None:
        stepped = math.floor(units / cfg.lot_step) * cfg.lot_step
        floored = stepped < units
        units = stepped

    below_min = cfg.min_notional is not None and units * entry < cfg.min_notional
    if below_min:
        units = 0.0

    notional = units * entry
    loss = units * risk_per_unit
    sign = 1.0 if side is Side.long else -1.0

    return PositionSize(
        side=side,
        units=units,
        notional=notional,
        position_fraction=sign * notional / capital,
        loss_if_stopped=loss,
        risk_fraction_realized=loss / capital,
        capped=capped,
        floored=floored,
        below_min_notional=below_min,
    )
