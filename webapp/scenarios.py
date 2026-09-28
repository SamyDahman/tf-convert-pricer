"""Scenarios for the demo: one bond, priced by both models.

The three TF regimes (equity-like / hybrid / busted) carry a flat credit
spread ``s`` and a volatility ``σ_BS``. To compare like with like, the JTD
market is calibrated to the same two market quotes:

  - **credit**: a 5y CDS on the issuer prices at ``s`` (``bootstrap_hazard``);
  - **vol**: a 1y at-the-money option prices at its Black–Scholes value
    with ``σ_BS`` (spec §05, "listed options" mode).

``σ_BS`` is *not* reused as the JTD diffusion vol: a BS implied vol already
prices the stock's crash/default risk, and JTD adds default through λ, so
reusing it double-counts (spec §07 traps). The diffusion vol comes out
lower, most so where the hazard is high. With ``p > 0`` each calibration
depends on the other's output, so they are alternated to a fixed point.

The fourth scenario is the JTD spec's reference convertible, which has a
soft call (callable only when ``S ≥ 130% × conversion price``). The TF tree
has no soft-call trigger, so that scenario is JTD-only.
"""

from dataclasses import dataclass, replace

from tf_convert_pricer import examples
from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.jtd.analytic import black_scholes_call
from tf_convert_pricer.jtd.calibration import (
    bootstrap_hazard,
    model_par_spread,
    option_implied_volatility,
)
from tf_convert_pricer.jtd.examples import reference_convertible
from tf_convert_pricer.jtd.grid import GridSpec
from tf_convert_pricer.jtd.market import CDSCurve, HazardCurve, JTDMarketData
from tf_convert_pricer.jtd.pricer import JTDTerms
from tf_convert_pricer.market import MarketData

CALIBRATION_GRID = GridSpec(space_nodes=200, steps_per_year=100)
CDS_TENOR = 5.0
VOL_TENOR = 1.0
VOL_FLOOR = 0.05


@dataclass(frozen=True)
class Scenario:
    name: str
    bond: ConvertibleBond
    jtd_market: JTDMarketData
    jtd_terms: JTDTerms
    tf_market: MarketData | None  # None → TF can't price this bond faithfully
    quoted_volatility: float | None = None  # BS implied vol the JTD σ was calibrated to
    vol_floored: bool = False  # True → no σ reprices the option; σ set to VOL_FLOOR

    @property
    def cds_spread(self) -> float:
        """Model 5y CDS par spread implied by the JTD hazard curve."""
        return model_par_spread(self.jtd_market, CDS_TENOR, CALIBRATION_GRID)


TF_REGIMES = {
    "hybrid": examples.hybrid,
    "equity-like": examples.equity_like,
    "busted": examples.busted,
}
SOFT_CALL = "soft-call (spec reference)"
SCENARIOS = (*TF_REGIMES, SOFT_CALL)


def atm_option_target(market: MarketData, tenor: float = VOL_TENOR) -> float:
    """Black–Scholes price of the ATM call that stands in for the listed-option quote."""
    return black_scholes_call(
        market.spot, market.spot, tenor, market.risk_free_rate, market.volatility, market.dividend_yield
    )


def jtd_market_from_tf(
    market: MarketData, hazard_elasticity: float, recovery: float, tol: float = 1e-5, max_iter: int = 10
) -> tuple[JTDMarketData, bool]:
    """JTD market matching TF's 5y credit spread *and* its ATM option price.

    Alternates: bootstrap λ0 at the current σ, then solve for the σ that
    reprices the ATM option at that λ0. Converges in two or three rounds
    (with ``p = 0`` the CDS doesn't depend on σ, so one round is exact).

    Returns ``(market, floored)``. ``floored`` means the two quotes are
    inconsistent in this model: the hazard alone makes the option worth
    more than its market price even with no diffusion (survivors drift at
    ``r + ηλ``), so no σ ≥ ``VOL_FLOOR`` fits and σ is pinned at the floor.
    This happens for wide spreads with a steep ``p`` (busted, p = 2).
    """
    jtd = JTDMarketData(
        spot=market.spot,
        volatility=market.volatility,
        risk_free_rate=market.risk_free_rate,
        hazard=HazardCurve.flat(0.0, market.spot),
        hazard_elasticity=hazard_elasticity,
        dividend_yield=market.dividend_yield,
        recovery=recovery,
    )
    cds = CDSCurve((CDS_TENOR,), (market.credit_spread,))
    target = atm_option_target(market)
    floored = False
    for _ in range(max_iter):
        jtd = replace(jtd, hazard=bootstrap_hazard(jtd, cds, CALIBRATION_GRID, initial=jtd.hazard))
        try:
            vol = option_implied_volatility(
                jtd, market.spot, VOL_TENOR, True, target, CALIBRATION_GRID, bounds=(VOL_FLOOR, 2.0)
            )
            floored = False
        except ValueError:
            vol, floored = VOL_FLOOR, True
        converged = abs(vol - jtd.volatility) < tol
        jtd = replace(jtd, volatility=vol)
        if converged:
            break
    return replace(jtd, hazard=bootstrap_hazard(jtd, cds, CALIBRATION_GRID, initial=jtd.hazard)), floored


def build_scenario(name: str, hazard_elasticity: float, recovery: float) -> Scenario:
    if name == SOFT_CALL:
        bond, terms, market = reference_convertible()
        # Keep the spec's λ0 = 3% and σ = 25% (already a diffusion vol);
        # only p and R follow the sidebar.
        market = replace(market, hazard_elasticity=hazard_elasticity, recovery=recovery)
        return Scenario(name, bond, market, terms, tf_market=None)
    bond, tf_market = TF_REGIMES[name]()
    jtd_market, floored = jtd_market_from_tf(tf_market, hazard_elasticity, recovery)
    return Scenario(
        name, bond, jtd_market, JTDTerms(), tf_market, quoted_volatility=tf_market.volatility, vol_floored=floored
    )
