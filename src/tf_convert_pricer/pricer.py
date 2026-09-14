"""Tsiveriotis-Fernandes binomial tree pricer.

Built in stages, each independently testable:

  1. Stock price tree (``StockTree``): CRR up/down factors, risk-neutral
     probability, level-at-step access. No bond logic.
  2. Terminal payoffs at maturity: ``max(redemption, parity)``.
  3. Backward induction with a single discount rate — the
     ``credit_spread → 0`` limit of step 4.
  4. Equity/cash-only terminal split and two-rate backward induction:
     the TF core.
  5a. Discrete coupons injected into the cash-only component.
  5b. Interior early-exercise: holder put, voluntary conversion,
      issuer call. Applied in that order at each interior step.

Each stage keeps the previous stage's tests green.
"""

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.market import MarketData


@dataclass(frozen=True)
class StockTree:
    """CRR binomial tree on the underlying stock price.

    ``u = exp(σ √Δt)``, ``d = 1 / u``, and the risk-neutral probability
    of an up move is ``p = (exp((r - q) Δt) - d) / (u - d)``. The tree
    is recombining, so at time step ``i`` there are ``i + 1`` distinct
    prices, indexed by up-move count ``j ∈ [0, i]``:

        S(i, j) = spot · u^(2j - i)

    which follows from ``d = 1 / u``.
    """

    spot: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float
    maturity: float
    steps: int

    @property
    def dt(self) -> float:
        return self.maturity / self.steps

    @property
    def u(self) -> float:
        return exp(self.volatility * sqrt(self.dt))

    @property
    def d(self) -> float:
        return 1.0 / self.u

    @property
    def p(self) -> float:
        return (exp((self.risk_free_rate - self.dividend_yield) * self.dt) - self.d) / (self.u - self.d)

    def prices_at(self, step: int) -> np.ndarray:
        """Stock prices at time step ``step``, shape ``(step + 1,)``.

        Index ``j`` corresponds to ``j`` up-moves out of ``step`` total
        moves along that path.
        """
        if not 0 <= step <= self.steps:
            raise ValueError(f"step must be in [0, {self.steps}], got {step}")
        j = np.arange(step + 1)
        return self.spot * self.u ** (2 * j - step)


def _resolve_redemption(bond: ConvertibleBond) -> float:
    """Redemption value, defaulting to face when not explicitly set."""
    return bond.redemption_value if bond.redemption_value is not None else bond.face_value


def _active_prices_by_step(
    schedule: dict[float, float] | None,
    dt: float,
    steps: int,
) -> dict[int, float]:
    """Materialize a step-function call/put schedule as ``{step: price}``.

    The input schedule uses ``{t: price}`` step-function semantics:
    at time ``t`` the price steps to that value and holds until the
    next entry. This function walks each interior step ``0..steps-1``
    and records the active price if any. Missing entries mean "not
    exercisable at this step."
    """
    if schedule is None:
        return {}
    sorted_events = sorted(schedule.items())
    result: dict[int, float] = {}
    for k in range(steps):
        t = k * dt
        active: float | None = None
        for event_t, event_price in sorted_events:
            if event_t <= t + 1e-9:
                active = event_price
            else:
                break
        if active is not None:
            result[k] = active
    return result


def coupon_schedule(bond: ConvertibleBond) -> list[tuple[float, float]]:
    """Coupon events for a bond as ``(time, amount)`` pairs, ascending.

    Times are year fractions from the valuation date; the schedule is
    derived by stepping backward from ``bond.maturity`` in
    ``1 / coupon_frequency`` increments. The amount at each date is
    ``face_value × coupon_rate / coupon_frequency``.

    Bonds with ``coupon_rate = 0`` still return the date grid with
    zero-amount coupons — the pricer adds zero, which is a no-op.
    """
    period = 1.0 / bond.coupon_frequency
    amount = bond.face_value * bond.coupon_rate / bond.coupon_frequency
    times: list[float] = []
    t = bond.maturity
    while t > 1e-9:
        times.append(t)
        t -= period
    times.reverse()
    return [(t, amount) for t in times]


def terminal_payoff_split(
    bond: ConvertibleBond,
    terminal_prices: np.ndarray,
    terminal_coupon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """TF terminal split at each terminal node: ``(equity, cash_only)``.

    At maturity the holder chooses: convert (parity ≥ redemption cash
    flow) or redeem (parity < redemption cash flow). The TF split
    assigns the full node value to whichever branch wins, so exactly
    one of the two returned arrays is nonzero at each node:

      - convert node → ``E = parity``, ``B = 0`` (no more credit risk)
      - redeem node  → ``E = 0``, ``B = redemption + terminal_coupon``

    ``terminal_coupon`` is the amount of any coupon paid at maturity;
    it belongs to the redemption branch only, because converting means
    surrendering the bond (and its final coupon) in exchange for shares.
    The nonzero coupon shifts the convert/redeem boundary upward at
    terminal.

    The split is what enables the two-rate backward induction that is
    the point of TF — ``E`` is discounted at ``r``, ``B`` at ``r + s``.
    """
    redemption = _resolve_redemption(bond)
    redeem_cf = redemption + terminal_coupon
    parity = bond.conversion_ratio * terminal_prices
    converts = parity >= redeem_cf
    equity = np.where(converts, parity, 0.0)
    cash_only = np.where(converts, 0.0, redeem_cf)
    return equity, cash_only


def terminal_payoff(bond: ConvertibleBond, terminal_prices: np.ndarray) -> np.ndarray:
    """Bond payoff at each terminal node: ``max(redemption, parity)``.

    Equivalent to summing the TF split from ``terminal_payoff_split``,
    exposed as a convenience for callers that don't need the components.
    """
    equity, cash_only = terminal_payoff_split(bond, terminal_prices)
    return equity + cash_only


@dataclass
class PricingResult:
    price: float
    parity: float
    equity_component: float
    cash_only_component: float


def price(bond: ConvertibleBond, market: MarketData, steps: int) -> PricingResult:
    """Price a convertible bond under the TF model.

    Step 5b (current): TF split with two discount rates, discrete
    coupons in the cash-only leg, and interior early-exercise checks
    (put → voluntary conversion → issuer call) applied at each step
    after discounting and coupon injection.

    Exercise mechanics at each interior step:
      - Holder put at price ``P``: if ``P > V_cont``, holder puts.
        Node becomes ``E = 0, B = P``.
      - Voluntary conversion: if ``parity > V_cont`` (after put),
        holder converts. Node becomes ``E = parity, B = 0``.
      - Issuer call at price ``K``: if ``max(K, parity) < V_cont``
        (after conversion), issuer calls. Holder responds by taking
        ``parity`` if ``parity ≥ K``, else ``K``.

    Put/call schedules at exactly ``t = maturity`` are not consulted;
    maturity behavior is fully covered by ``terminal_payoff_split``.
    Route maturity puts through ``redemption_value`` if needed.

    Sanity relationships:
      - ``conversion_ratio = 0`` → plain credit-risky coupon bond by
        closed form: ``Σ c_i exp(-(r+s) t_i) + F exp(-(r+s) T)``
      - no put/call schedules and ``coupon_rate = 0`` → step-4 pricer
      - no put/call schedules, ``coupon_rate = 0``, ``s = 0`` → step-3
      - put at ``P = 0`` or call at ``K = ∞`` are no-ops
      - adding a put weakly increases price; adding a call weakly
        decreases it
    """
    tree = StockTree(
        spot=market.spot,
        volatility=market.volatility,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
        maturity=bond.maturity,
        steps=steps,
    )
    interior_coupons: dict[int, float] = {}
    terminal_coupon = 0.0
    for t_c, amount in coupon_schedule(bond):
        step_idx = round(t_c / tree.dt)
        if step_idx >= steps:
            terminal_coupon += amount
        else:
            interior_coupons[step_idx] = interior_coupons.get(step_idx, 0.0) + amount
    puts_by_step = _active_prices_by_step(bond.put_schedule, tree.dt, steps)
    calls_by_step = _active_prices_by_step(bond.call_schedule, tree.dt, steps)
    equity, cash_only = terminal_payoff_split(
        bond, tree.prices_at(steps), terminal_coupon=terminal_coupon
    )
    discount_r = exp(-market.risk_free_rate * tree.dt)
    discount_rs = exp(-(market.risk_free_rate + market.credit_spread) * tree.dt)
    p = tree.p
    for i in range(steps):
        equity = discount_r * (p * equity[1:] + (1.0 - p) * equity[:-1])
        cash_only = discount_rs * (p * cash_only[1:] + (1.0 - p) * cash_only[:-1])
        cur_step = steps - i - 1
        if cur_step in interior_coupons:
            cash_only = cash_only + interior_coupons[cur_step]
        parity_at_step = bond.conversion_ratio * tree.prices_at(cur_step)
        total = equity + cash_only
        if cur_step in puts_by_step:
            put_price = puts_by_step[cur_step]
            put_wins = put_price > total
            equity = np.where(put_wins, 0.0, equity)
            cash_only = np.where(put_wins, put_price, cash_only)
            total = equity + cash_only
        convert_wins = parity_at_step > total
        equity = np.where(convert_wins, parity_at_step, equity)
        cash_only = np.where(convert_wins, 0.0, cash_only)
        total = equity + cash_only
        if cur_step in calls_by_step:
            call_price = calls_by_step[cur_step]
            call_response = np.maximum(call_price, parity_at_step)
            call_wins = call_response < total
            take_parity = parity_at_step >= call_price
            equity = np.where(
                call_wins,
                np.where(take_parity, parity_at_step, 0.0),
                equity,
            )
            cash_only = np.where(
                call_wins,
                np.where(take_parity, 0.0, call_price),
                cash_only,
            )
    equity_root = float(equity[0])
    cash_root = float(cash_only[0])
    return PricingResult(
        price=equity_root + cash_root,
        parity=bond.conversion_ratio * market.spot,
        equity_component=equity_root,
        cash_only_component=cash_root,
    )
