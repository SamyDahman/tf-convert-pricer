"""Closed forms used by the JTD engine and its validation tests."""

from math import exp, sqrt

import numpy as np
from scipy.stats import norm


def black_scholes_call(
    spot: np.ndarray | float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> np.ndarray | float:
    """European call with continuous yield; vectorized over ``spot``.

    Degenerates cleanly: ``maturity = 0`` gives intrinsic, ``spot = 0``
    gives zero.
    """
    s = np.asarray(spot, dtype=float)
    if maturity <= 0.0:
        out = np.maximum(s - strike, 0.0)
        return out if out.ndim else float(out)
    vol_t = volatility * sqrt(maturity)
    with np.errstate(divide="ignore"):
        d1 = (
            np.log(np.maximum(s, 1e-300) / strike)
            + (rate - dividend_yield + 0.5 * volatility**2) * maturity
        ) / vol_t
    d2 = d1 - vol_t
    out = s * exp(-dividend_yield * maturity) * norm.cdf(d1) - strike * exp(
        -rate * maturity
    ) * norm.cdf(d2)
    out = np.where(s > 0.0, out, 0.0)
    return out if out.ndim else float(out)


def risky_zero_coupon(
    face: float, maturity: float, rate: float, intensity: float, recovery: float
) -> float:
    """Zero-coupon bond under constant ``λ`` with recovery of par paid at default (spec T2)."""
    k = rate + intensity
    return face * exp(-k * maturity) + recovery * face * intensity / k * (
        1.0 - exp(-k * maturity)
    )


__all__ = ["black_scholes_call", "risky_zero_coupon"]
