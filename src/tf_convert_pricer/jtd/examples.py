"""The spec's §08 validation setup as reusable fixtures.

Common parameters: ``N = 100, κ = 1, S0 = 100, T = 5, r = 4%, q = b = 0,
σ = 25%, R = 40%, η = 1, λmax = 2``. The reference convertible adds a
flat ``λ0 = 3%``, ``p = 1``, a 2% semi-annual coupon and a soft call
from year 3 at 100 + accrued when ``S ≥ 130``, no notice, no put.
"""

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.jtd.grid import GridSpec
from tf_convert_pricer.jtd.market import HazardCurve, JTDMarketData
from tf_convert_pricer.jtd.pricer import JTDTerms

# The prototype that produced the spec's reference numbers: uniform
# ΔS = 0.25, Δt = 1/400, Smax = 800.
PROTOTYPE_GRID = GridSpec(
    space_nodes=3200, steps_per_year=400, concentration=None, smax=800.0
)


def common_market(
    intensity: float = 0.0, hazard_elasticity: float = 0.0
) -> JTDMarketData:
    return JTDMarketData(
        spot=100.0,
        volatility=0.25,
        risk_free_rate=0.04,
        hazard=HazardCurve.flat(intensity, reference_spot=100.0),
        hazard_elasticity=hazard_elasticity,
        recovery=0.4,
        jump_size=1.0,
        lambda_max=2.0,
    )


def reference_convertible() -> tuple[ConvertibleBond, JTDTerms, JTDMarketData]:
    """The spec's reference convertible.

    The call period starts on a coupon date (year 3), so the result
    depends on ``JTDTerms.exercise_before_coupon``; the spec's numbers
    are reproduced with the default (``True``).
    """
    bond = ConvertibleBond(
        face_value=100.0,
        coupon_rate=0.02,
        coupon_frequency=2,
        maturity=5.0,
        conversion_ratio=1.0,
        call_schedule={3.0: 100.0},
    )
    terms = JTDTerms(call_trigger=1.3, call_notice=0.0, prices_are_clean=True)
    return bond, terms, common_market(intensity=0.03, hazard_elasticity=1.0)
