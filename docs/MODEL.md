# Model: Tsiveriotis-Fernandes Convertible Pricer

## Why TF

Standard, well-documented approach for pricing convertibles with credit
risk, without needing a full stochastic credit-spread model. Splits the
bond into two components at every tree node:

- **Equity component** — value attributable to the conversion option,
  discounted at the risk-free rate (conversion removes credit exposure)
- **Cash-only component (COCB)** — value attributable to guaranteed cash
  flows, discounted at risk-free + credit spread

## Tree mechanics

- Underlying: CRR binomial tree on the stock price
  (`u = exp(σ√Δt)`, `d = 1/u`, risk-neutral probability
  `p = (exp((r − q)Δt) − d) / (u − d)` where `q` is the continuous
  dividend yield)
- Backward induction from maturity: at each node, apply conversion / call
  / put logic before discounting the two components back one step
- Terminal condition at maturity: `max(redemption value, parity)`
- Coupons enter as lump-sum additions to the cash-only component at the
  tree node nearest each coupon date

## Assumptions log (v1)

Record each decision here as it's made, with a one-line rationale —
this is the "why" record for the model, not just the "what."

| Assumption | Decision | Rationale |
|---|---|---|
| Credit spread term structure | Flat, constant across the tree | TF is framed around a single credit-risky discount rate; term structure adds curve fitting with little marginal insight for a demo |
| Volatility | Constant | 1-factor CRR tree can't naturally consume a surface without going to local-vol; constant is the textbook TF setup |
| Dividend treatment | Continuous yield | Discrete ex-date drops break tree recombination; continuous yield keeps the tree clean |
| Day count / compounding | ACT/365, continuous compounding | Makes CRR up/down factors clean; single convention throughout for internal consistency |
| Tree steps / convergence | Default 500, with a convergence test at 100/200/500/1000 | 500 is safely inside CRR's O(1/N) converged regime; the test doubles as a regression check |
| Call/put notice periods | Ignored (calls treated as immediate) | Notice periods require a "soft-call" region layered on the base tree; TF literature typically starts without them |
| Up/down factors | CRR | Recombines symmetrically around spot; cleaner for early-exercise checks than Jarrow-Rudd |
| Coupons | Discrete lump-sum at coupon dates, added to COCB | Matches how a convert desk reasons about a real bond and preserves the credit-risky-vs-conversion split that motivates TF |
| Call schedule type | Hard-call only (price-by-date schedule) | Provisional/contingent calls introduce path dependence that breaks a plain binomial tree — v2 problem |

## Known simplifications

Every choice above is a simplification of something a real convert desk
would model more carefully. Calling them out explicitly so the pricer's
scope is honest:

- **Flat credit spread** — no credit-curve steepness. Spread duration
  reported for a busted long-dated convert is directionally right but
  loses curvature detail.
- **Constant volatility** — no smile. Matters most for busted
  convertibles where the embedded equity option is deep out-of-the-money;
  vega should be read as "vega at the chosen vol level," not "vega vs.
  the market surface."
- **Continuous dividend yield** — loses fidelity around ex-dates for
  equity-like converts where optimal conversion could time around dividends.
- **No call notice periods** — materially overstates the issuer's ability
  to call for bonds with tight call schedules; the model may show a
  premium being capped that in reality would still have upside during
  the notice window.
- **No provisional/contingent calls** — this is the largest gap between
  the pricer and a real-world convert. Most modern US converts are
  callable only after a contingent trigger has been met; this pricer
  will treat them as unconditionally callable on the schedule dates.

Each of these is a candidate for a v2 extension; none is a bug.

## Example instruments (see src/tf_convert_pricer/examples.py)

- **Equity-like**: deep in-the-money, trades near parity
- **Hybrid**: mixed sensitivity to stock and credit
- **Busted**: stock has cratered, trades near bond floor, credit-spread
  driven

## Validation approach

- Sanity check: credit spread → 0 should converge toward a standard
  convertible bond with no credit differentiation
- Sanity check: conversion ratio → 0 should converge toward a plain
  credit-risky bond (no optionality)
- Compare against a known worked example from the literature, if one
  can be found with matching assumptions
