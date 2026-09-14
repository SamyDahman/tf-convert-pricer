"""Market inputs to the pricer.

Everything the tree needs that isn't a term of the bond itself: spot,
volatility, rates, credit spread, and dividend yield — all encoded per
the v1 assumptions in docs/MODEL.md.

All rates and yields are continuously compounded and annualized;
volatility is annualized (σ, not σ²).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketData:
    """Market inputs for a single pricing run.

    The credit spread is added to the risk-free rate to discount the
    cash-only component (COCB); the equity component is discounted at
    the risk-free rate alone.
    """

    spot: float
    volatility: float
    risk_free_rate: float
    credit_spread: float
    dividend_yield: float
