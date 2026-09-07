"""Options: pricing, dealer-gamma (GEX) analytics, and a naked-only simulator.

- :mod:`options.pricing` — Black-Scholes price and greeks (European; used to
  mark synthetic chains and as a sanity reference, never as a substitute for
  a real quote).
- :mod:`options.gex` — dealer gamma exposure by strike from a chain
  snapshot: net GEX, call/put walls, gamma flip.
- :mod:`options.simulator` — event-driven simulator over
  :class:`~ingestion.schema.OptionsChainSnapshot` sequences. Fills at
  bid/ask, marks at mid, settles at expiry, and refuses any book that is
  not naked-only (long call, long put, cash-secured put, covered call).
- :mod:`options.synthetic` — Black-Scholes chain generator for tests and
  offline demos. Synthetic data, always labeled as such.

The simulator emits the same ledger as :mod:`backtest.engine`, so the whole
validation stack — metrics, deflated Sharpe, regime attribution, health
monitor — applies unchanged.
"""

from __future__ import annotations

from options.gex import GexProfile, gex_by_strike, gex_profile
from options.pricing import Greeks, bs_greeks, bs_price
from options.simulator import (
    Book,
    CollateralError,
    ContractKey,
    Fill,
    MarkError,
    NakedOnlyError,
    Portfolio,
    SimConfig,
    SimulationResult,
    StrategyFn,
    simulate,
)
from options.synthetic import synthetic_chain

__all__ = [
    "Book",
    "CollateralError",
    "ContractKey",
    "Fill",
    "GexProfile",
    "Greeks",
    "MarkError",
    "NakedOnlyError",
    "Portfolio",
    "SimConfig",
    "SimulationResult",
    "StrategyFn",
    "bs_greeks",
    "bs_price",
    "gex_by_strike",
    "gex_profile",
    "simulate",
    "synthetic_chain",
]
