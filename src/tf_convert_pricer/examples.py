"""Made-up example instruments spanning the three convert regimes.

Three ``(ConvertibleBond, MarketData)`` pairs sharing identical bond
terms — only the market state varies. This isolates the regime as a
property of the market (spot, credit spread, vol, dividend yield)
rather than of the instrument itself, which matches how a convert
desk actually thinks: same bond, different market states through time.

Each example is picked so the regime is obvious from the pricer's
output — the equity-like example has ``equity_component / price → 1``,
the busted example has ``cash_only_component / price → 1``, and the
hybrid example has both components materially positive.
"""

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.market import MarketData


_BOND_TERMS = ConvertibleBond(
    face_value=1000.0,
    coupon_rate=0.05,
    coupon_frequency=2,
    maturity=5.0,
    conversion_ratio=10.0,
)


def equity_like() -> tuple[ConvertibleBond, MarketData]:
    """Deep-ITM convertible: stock has appreciated past the conversion price.

    Parity today is ``2 × face``, so terminal nodes overwhelmingly
    favour conversion. Credit spread is tight because the issuer's
    stock has done well. Dividend yield is set above the coupon-vs-
    dividend break-even (``q > coupon_rate × face / parity = 2.5%``),
    which makes early conversion optimal — the holder captures parity
    now rather than bleeding dividends they don't receive. All the
    value ends up in the equity component.
    """
    market = MarketData(
        spot=200.0,
        volatility=0.30,
        risk_free_rate=0.04,
        credit_spread=0.01,
        dividend_yield=0.04,
    )
    return _BOND_TERMS, market


def hybrid() -> tuple[ConvertibleBond, MarketData]:
    """At-the-money convertible: spot right at the conversion price.

    Both the bond floor and the conversion option are meaningfully in
    play, so the price is sensitive to both stock and credit. Credit
    spread is moderate — the issuer isn't stressed but isn't stellar
    either. This is the regime where TF's dual-discount structure
    matters most, because both components carry real value.
    """
    market = MarketData(
        spot=100.0,
        volatility=0.30,
        risk_free_rate=0.04,
        credit_spread=0.03,
        dividend_yield=0.02,
    )
    return _BOND_TERMS, market


def busted() -> tuple[ConvertibleBond, MarketData]:
    """Busted convertible: stock has cratered well below the conversion price.

    Parity today is well under face, so terminal nodes overwhelmingly
    favour redemption. Credit spread is wide (distressed issuer),
    volatility is elevated (stress regime), and the dividend has been
    cut (``q = 0``). The bond floor — even discounted heavily for
    credit — supports the price above parity, so the convert trades
    at a premium to its equity value and is driven by credit-spread
    duration rather than delta.
    """
    market = MarketData(
        spot=40.0,
        volatility=0.50,
        risk_free_rate=0.04,
        credit_spread=0.08,
        dividend_yield=0.0,
    )
    return _BOND_TERMS, market
