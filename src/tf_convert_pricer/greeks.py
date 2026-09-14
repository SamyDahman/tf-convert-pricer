"""Risk sensitivities via bump-and-reprice on top of the TF pricer.

Each greek is a finite difference of ``price()`` output over a small
perturbation of one input. ``dataclasses.replace`` builds the bumped
``MarketData`` or ``ConvertibleBond`` (both frozen), and the pricer is
called two or three times — no shared-tree optimization.

Sign conventions:
  - ``delta``:                       ``dV/dS``   (positive)
  - ``gamma``:                       ``d²V/dS²`` (positive for convex payoffs)
  - ``vega``:                        ``dV/dσ``   (positive)
  - ``credit_spread_sensitivity``:   ``dV/ds``   (negative)
  - ``theta``:                       ``dV/dt``   (usually POSITIVE for a
    discount bond pulling to par — see ``theta`` docstring)

Bumps default to sizes big enough to overcome tree discretization
noise, small enough to keep truncation error low. Override via the
``bump`` argument if a specific unit convention is wanted (e.g., pass
``bump=0.0001`` to ``credit_spread_sensitivity`` and multiply by
0.0001 to get a DV01).
"""

from dataclasses import replace

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.market import MarketData
from tf_convert_pricer.pricer import price


def delta(
    bond: ConvertibleBond,
    market: MarketData,
    steps: int,
    bump: float | None = None,
) -> float:
    """``dV/dS`` via central difference. Default bump = 1% of spot."""
    if bump is None:
        bump = 0.01 * market.spot
    up = price(bond, replace(market, spot=market.spot + bump), steps).price
    dn = price(bond, replace(market, spot=market.spot - bump), steps).price
    return (up - dn) / (2.0 * bump)


def gamma(
    bond: ConvertibleBond,
    market: MarketData,
    steps: int,
    bump: float | None = None,
) -> float:
    """``d²V/dS²`` via central second difference. Default bump = 1% of spot."""
    if bump is None:
        bump = 0.01 * market.spot
    up = price(bond, replace(market, spot=market.spot + bump), steps).price
    md = price(bond, market, steps).price
    dn = price(bond, replace(market, spot=market.spot - bump), steps).price
    return (up - 2.0 * md + dn) / (bump * bump)


def vega(
    bond: ConvertibleBond,
    market: MarketData,
    steps: int,
    bump: float = 0.01,
) -> float:
    """``dV/dσ`` via central difference. Default bump = 1 vol point (0.01)."""
    up = price(bond, replace(market, volatility=market.volatility + bump), steps).price
    dn = price(bond, replace(market, volatility=market.volatility - bump), steps).price
    return (up - dn) / (2.0 * bump)


def credit_spread_sensitivity(
    bond: ConvertibleBond,
    market: MarketData,
    steps: int,
    bump: float = 0.0001,
) -> float:
    """``dV/ds`` via central difference. Default bump = 1 bp (0.0001).

    Result is dollars per unit spread. Multiply by 0.0001 to get the
    dollar change per basis point (DV01).
    """
    up = price(bond, replace(market, credit_spread=market.credit_spread + bump), steps).price
    dn = price(bond, replace(market, credit_spread=market.credit_spread - bump), steps).price
    return (up - dn) / (2.0 * bump)


def theta(
    bond: ConvertibleBond,
    market: MarketData,
    steps: int,
    bump: float = 1.0 / 365.0,
) -> float:
    """``dV/dt`` via one-sided forward difference. Default bump = 1 day.

    Bumps ``bond.maturity`` down by ``bump``; since coupon dates are
    generated backward from maturity in fixed increments, shortening
    maturity by ``bump`` shifts all coupon dates earlier by the same
    amount — the right behaviour for "``bump`` years of calendar time
    passing," as long as no coupon falls in ``[0, bump]``.

    Sign convention: theta is value change per year of calendar time
    at unchanged market. Unlike an option (which loses time value as
    it decays), a bond can have positive theta — cash flows arriving
    sooner have higher PV. Whether a convertible's theta is positive
    or negative depends on regime:

      - **discount / busted**: bond-leg pull-to-par dominates → theta > 0
      - **par-ish / hybrid**:  mix of pull-to-par and option decay
      - **equity-like**:       near-parity; option leg is largely gone
        (early conversion is nearly optimal) so theta is near zero
    """
    older = replace(bond, maturity=bond.maturity - bump)
    return (price(older, market, steps).price - price(bond, market, steps).price) / bump
