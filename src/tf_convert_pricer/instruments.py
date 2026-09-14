"""Convertible bond instrument definitions.

Terms of a single convertible bond issue. All times are year fractions
from the valuation date (ACT/365) — any absolute-date-to-year-fraction
conversion is the caller's job and should use ACT/365 to stay
consistent with the tree.

See docs/MODEL.md for the v1 assumptions this schema encodes, notably:
hard-call schedule only (no provisional/contingent calls), coupons as
discrete lump sums on their scheduled dates, and no call notice periods.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvertibleBond:
    """Terms of a single convertible bond issue.

    Call/put schedules use step-function semantics: each entry
    ``{t: price}`` means "callable (or puttable) at ``price`` for all
    times ``>= t``, until the next entry's time or maturity." A schedule
    of ``None`` means no callability / no puttability.

    ``redemption_value`` of ``None`` means redeem at ``face_value``,
    which is the common case; specify explicitly only for premium
    redemption.
    """

    face_value: float
    coupon_rate: float
    coupon_frequency: int
    maturity: float
    conversion_ratio: float
    redemption_value: float | None = None
    call_schedule: dict[float, float] | None = None
    put_schedule: dict[float, float] | None = None
