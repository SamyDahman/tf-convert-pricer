"""Path simulation and per-frame greek computation for the demo app.

Kept separate from ``app.py`` so the numerical logic is testable
without needing to spin up Streamlit.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from tf_convert_pricer.greeks import (
    credit_spread_sensitivity,
    delta,
    gamma,
    theta,
    vega,
)
from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.market import MarketData
from tf_convert_pricer.pricer import price


def simulate_path(
    market: MarketData,
    n_days: int,
    n_frames: int,
    seed: int,
) -> np.ndarray:
    """Generate a GBM spot path with ``n_frames + 1`` samples over ``n_days``.

    Uses the market's own ``r``, ``q``, ``σ`` as drift and diffusion
    parameters, so the path is consistent with the pricer's Q-measure
    world. Returns an array of length ``n_frames + 1`` including the
    starting spot at index 0.
    """
    total_years = n_days / 252.0
    dt = total_years / n_frames
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_frames)
    drift = (market.risk_free_rate - market.dividend_yield - 0.5 * market.volatility**2) * dt
    diffusion = market.volatility * np.sqrt(dt) * z
    log_increments = drift + diffusion
    log_path = np.concatenate([[0.0], np.cumsum(log_increments)])
    return market.spot * np.exp(log_path)


GREEK_COLUMNS = (
    "delta",
    "gamma",
    "vega",
    "credit_spread_sensitivity",
    "theta",
)


def compute_series(
    bond: ConvertibleBond,
    market: MarketData,
    spot_path: np.ndarray,
    tree_steps: int,
    n_days: int,
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Price + greeks at each frame along ``spot_path``.

    Only spot changes across frames — the bond's remaining maturity is
    held constant. This isolates the "same bond, spot moves" story
    that the animation is telling, and avoids two sources of numerical
    noise:

      - coupon dates crossing period boundaries as maturity shrinks
        (which caused discrete jumps in bond value and huge theta
        spikes in the finite-difference numerator)
      - the tree grid re-anchoring at a slightly different ``dt`` per
        frame, compounding the delta/gamma bump-and-reprice noise

    After the raw greeks are computed pointwise, each greek series is
    passed through a centered rolling-median filter of width
    ``smoothing_window`` (set to ``0`` or ``1`` to disable). Median
    smoothing removes single-frame spikes from bump-and-reprice noise
    while preserving the underlying trend. Fair value / spot / parity
    are left un-smoothed — those come straight from the pricer with no
    finite-difference amplification.

    ``n_days`` is retained purely to label the x-axis; it doesn't
    advance the bond.
    """
    total_years = n_days / 252.0
    times = np.linspace(0.0, total_years, len(spot_path))
    c = bond.conversion_ratio
    records = []
    for frame_idx, (t_elapsed, spot) in enumerate(zip(times, spot_path)):
        moved_market = replace(market, spot=float(spot))
        # Wider bumps for delta/gamma so the finite-difference signal
        # comfortably clears tree discretization noise.
        spot_bump = 0.02 * float(spot)
        result = price(bond, moved_market, tree_steps)
        # Rescale all greeks to option-style trader conventions:
        #   delta / c        → [0, 1] hedge ratio per share
        #   gamma / c        → per-share gamma (1/$)
        #   vega × 0.01      → $ per 1 vol point
        #   cs_sens × 0.0001 → $ per 1 basis point of credit spread
        #   theta / 365      → $ per calendar day
        records.append(
            {
                "frame": frame_idx,
                "day": t_elapsed * 252.0,
                "spot": float(spot),
                "remaining_maturity": bond.maturity,
                "fair_value": result.price,
                "parity": result.parity,
                "equity_component": result.equity_component,
                "cash_only_component": result.cash_only_component,
                "delta": delta(bond, moved_market, tree_steps, bump=spot_bump) / c,
                "gamma": gamma(bond, moved_market, tree_steps, bump=spot_bump) / c,
                "vega": vega(bond, moved_market, tree_steps) * 0.01,
                "credit_spread_sensitivity": (
                    credit_spread_sensitivity(bond, moved_market, tree_steps) * 0.0001
                ),
                "theta": theta(bond, moved_market, tree_steps) / 365.0,
            }
        )
    df = pd.DataFrame(records)
    if smoothing_window and smoothing_window > 1:
        for col in GREEK_COLUMNS:
            df[col] = (
                df[col]
                .rolling(window=smoothing_window, center=True, min_periods=1)
                .median()
            )
    return df
