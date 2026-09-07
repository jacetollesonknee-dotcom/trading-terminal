"""Deflated Sharpe Ratio — the selection-bias correction for multiple trials.

If you tested N strategy variations and report the best one, its Sharpe is
inflated by selection: the maximum of N noisy draws is high even when every
draw is pure noise. The Deflated Sharpe Ratio (DSR) asks: what is the
probability the observed Sharpe exceeds the Sharpe you'd expect from the
*best of N* noise strategies?

Reference: Bailey, D. H. & López de Prado, M. (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
Journal of Portfolio Management 40(5).

Two rules that this module enforces and that the textbook one-liner usually
gets wrong:

1. **Everything is per-period, never annualized.** The test statistic scales
   with ``sqrt(n_obs - 1)`` where ``n_obs`` is the number of *periods*, so the
   Sharpe, the trial Sharpes, skew and kurtosis must all be measured at that
   same frequency. Feeding an annualized Sharpe overstates significance by a
   factor of ``sqrt(periods_per_year)``. :func:`deflated_sharpe` therefore
   derives the Sharpe and moments directly from the per-period return series,
   so the trap cannot occur; :func:`deannualize_sharpe` converts trial Sharpes
   that were recorded annualized.

2. **The expected max is scaled by the dispersion of the trials.** ``SR*``
   is ``sqrt(V[{SR_n}]) * E[max of N standard normals]``. Omitting the
   ``sqrt(V)`` factor (a common copy-paste error) yields a number in no units
   that cannot be compared to a real Sharpe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _raw_kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

# Euler-Mascheroni constant, gamma in the paper's E[max] approximation.
_EULER_MASCHERONI = 0.5772156649015329

# Kurtosis of the normal distribution in the *raw* (non-excess) convention the
# paper uses. scipy's default is *excess* (normal = 0); we always request raw.
_NORMAL_KURTOSIS = 3.0

# Minimum observations for a defined sample kurtosis.
_MIN_OBS = 4

# Minimum observations for the PSR test statistic (needs sqrt(n_obs - 1) > 0).
_MIN_PSR_OBS = 2

# Default confidence at which a strategy "passes" the deflated test.
_DEFAULT_THRESHOLD = 0.95


def deannualize_sharpe(sharpe_annualized: float, periods_per_year: int) -> float:
    """Convert an annualized Sharpe back to per-period units.

    Use this on trial Sharpes recorded from :class:`PerformanceMetrics`
    (which are annualized) before passing them to :func:`deflated_sharpe`.
    """
    if periods_per_year <= 0:
        msg = f"periods_per_year must be > 0, got {periods_per_year}"
        raise ValueError(msg)
    return sharpe_annualized / math.sqrt(periods_per_year)


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """``SR*``: the Sharpe you'd expect from the best of ``n_trials`` noise strategies.

    ``SR* = sqrt(V) * [(1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e))]`` where
    ``g`` is Euler-Mascheroni and ``sqrt(V)`` is the standard deviation of
    the trial Sharpes (per-period units).

    With a single trial there is no selection effect, and with zero dispersion
    the "best" is no better than any other — both return 0.0 rather than the
    ``-inf`` the raw formula produces for ``N = 1``.
    """
    if n_trials < 1:
        msg = f"n_trials must be >= 1, got {n_trials}"
        raise ValueError(msg)
    if trial_sharpe_std < 0:
        msg = f"trial_sharpe_std must be >= 0, got {trial_sharpe_std}"
        raise ValueError(msg)
    if n_trials == 1 or trial_sharpe_std == 0.0:
        return 0.0
    z_top = float(norm.ppf(1.0 - 1.0 / n_trials))
    z_inner = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    g = _EULER_MASCHERONI
    return trial_sharpe_std * ((1.0 - g) * z_top + g * z_inner)


def probabilistic_sharpe(
    sharpe: float,
    n_obs: int,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = _NORMAL_KURTOSIS,
) -> float:
    """``PSR(benchmark)``: probability the true Sharpe exceeds ``benchmark``.

    All of ``sharpe``, ``benchmark`` and the moments are per-period.
    ``kurtosis`` is *raw* (normal = 3), not excess.

    The non-normality adjustment in the denominator can go non-positive for
    pathological skew/Sharpe combinations; that is raised, not returned as NaN.
    """
    if n_obs < _MIN_PSR_OBS:
        msg = f"n_obs must be >= {_MIN_PSR_OBS}, got {n_obs}"
        raise ValueError(msg)
    var_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if var_term <= 0.0:
        msg = (
            f"non-normality adjustment is non-positive ({var_term:.4g}) for "
            f"sharpe={sharpe}, skew={skew}, kurtosis={kurtosis}"
        )
        raise ValueError(msg)
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    return float(norm.cdf(z))


def minimum_sharpe_to_pass(
    n_obs: int,
    n_trials: int,
    trial_sharpe_std: float | None = None,
    threshold: float = _DEFAULT_THRESHOLD,
) -> float:
    """The per-period Sharpe a best-of-N result must show to clear the DSR threshold.

    Answers "how good does my best variant have to look before it is
    distinguishable from noise?" *before* running anything. Assumes normal
    returns (skew 0, kurtosis 3), so it is a floor — fat tails raise it.

    Args:
        n_obs: Periods the selected strategy will be evaluated over.
        n_trials: Variations that will be tried.
        trial_sharpe_std: Dispersion of the trial Sharpes. Defaults to
            ``1/sqrt(n_obs)`` — the sampling noise of a per-period Sharpe,
            i.e. what N pure-noise trials would show.
        threshold: DSR confidence to clear.
    """
    if n_obs < 2:  # noqa: PLR2004  — same bound as probabilistic_sharpe
        msg = f"n_obs must be >= 2, got {n_obs}"
        raise ValueError(msg)
    if not 0.0 < threshold < 1.0:
        msg = f"threshold must be in (0, 1), got {threshold}"
        raise ValueError(msg)
    std = trial_sharpe_std if trial_sharpe_std is not None else 1.0 / math.sqrt(n_obs)
    sr_star = expected_max_sharpe(n_trials, std)
    # PSR is monotone increasing in the Sharpe; bisect for the crossing.
    lo, hi = sr_star, sr_star + 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if probabilistic_sharpe(mid, n_obs, benchmark=sr_star) >= threshold:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Outcome of a deflated-Sharpe test. All Sharpes are per-period."""

    deflated_sharpe: float
    sharpe_per_period: float
    expected_max_sharpe: float
    skew: float
    kurtosis: float
    n_obs: int
    n_trials: int
    trial_sharpe_std: float
    threshold: float

    @property
    def passes(self) -> bool:
        """True if the deflated Sharpe clears the confidence threshold."""
        return self.deflated_sharpe >= self.threshold


def deflated_sharpe(
    returns: pd.Series,
    trial_sharpes: Sequence[float],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> DeflatedSharpeResult:
    """Deflated Sharpe of a strategy, given the full set of trials it was picked from.

    Args:
        returns: Per-period net returns of the *selected* strategy (e.g.
            ``result.ledger["net"]``). Sharpe, skew and kurtosis are derived
            from this series, so they are guaranteed to be in per-period units.
        trial_sharpes: The per-period Sharpe of EVERY variation you tested,
            including the selected one. Be honest — the count and dispersion
            of this list is what deflates the result. Annualized values must
            be converted with :func:`deannualize_sharpe` first.
        threshold: Confidence at which :attr:`DeflatedSharpeResult.passes`
            flips to True. Default 0.95.

    Returns:
        A :class:`DeflatedSharpeResult`.
    """
    if not 0.0 < threshold < 1.0:
        msg = f"threshold must be in (0, 1), got {threshold}"
        raise ValueError(msg)
    n_trials = len(trial_sharpes)
    if n_trials < 1:
        msg = "trial_sharpes must contain at least the selected strategy"
        raise ValueError(msg)
    trials = np.asarray(trial_sharpes, dtype=float)
    if not np.isfinite(trials).all():
        msg = "trial_sharpes contains NaN or inf"
        raise ValueError(msg)

    r = returns.dropna()
    n_obs = len(r)
    if n_obs < _MIN_OBS:
        msg = f"need >= {_MIN_OBS} observations for a defined kurtosis, got {n_obs}"
        raise ValueError(msg)
    std = float(r.std(ddof=1))
    if not std > 0.0:
        msg = "returns have zero variance; Sharpe is undefined"
        raise ValueError(msg)

    sharpe = float(r.mean()) / std
    skew = float(_skew(r.to_numpy()))
    kurtosis = float(_raw_kurtosis(r.to_numpy(), fisher=False))
    if not (math.isfinite(skew) and math.isfinite(kurtosis)):
        msg = f"non-finite moments: skew={skew}, kurtosis={kurtosis}"
        raise ValueError(msg)

    trial_std = float(np.std(trials, ddof=1)) if n_trials > 1 else 0.0
    sr_star = expected_max_sharpe(n_trials, trial_std)
    dsr = probabilistic_sharpe(sharpe, n_obs, benchmark=sr_star, skew=skew, kurtosis=kurtosis)

    return DeflatedSharpeResult(
        deflated_sharpe=dsr,
        sharpe_per_period=sharpe,
        expected_max_sharpe=sr_star,
        skew=skew,
        kurtosis=kurtosis,
        n_obs=n_obs,
        n_trials=n_trials,
        trial_sharpe_std=trial_std,
        threshold=threshold,
    )
