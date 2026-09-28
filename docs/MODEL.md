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

---

# Model 2: Jump-to-Default PDE (`src/tf_convert_pricer/jtd/`)

Implements `docs/cb_jtd_spec.pdf`. It sits alongside TF and doesn't
replace it: both price the same `ConvertibleBond`, and the test suite
cross-checks them where they must agree.

## Why a second model

TF handles credit with a bookkeeping split: cash-like value is
discounted at `r + s` and equity-like value at `r`. It has no default
*event*. The JTD model makes default explicit. At a stock-dependent
intensity `λ(S,t) = min(λ0(t)(S0/S)^p, λmax)` the stock jumps down by a
fraction `η` and the bond pays recovery:

    V_t + ½σ²S²V_SS + (r − q − b + ηλ)S V_S − (r + λ)V + λ V_D = 0
    V_D = max(κ(1−η)S, R·N)

That gives three things TF can't:
- **Credit delta.** Spreads widen as the stock falls, so part of the
  hedge ratio comes from the credit channel. The report splits it out
  (`equity_delta` / `credit_delta`).
- **Jump-to-default P&L.** Both naked and delta-hedged.
- **One calibrated credit model.** `λ0(t)` is bootstrapped from a CDS
  curve with the *same* PDE, so bond and CDS are priced consistently.

## Module map

| Module | Role |
|---|---|
| `market.py` | `JTDMarketData`, `HazardCurve` (piecewise-flat λ0, anchored reference spot), `CDSCurve`, `RecoveryConvention` |
| `grid.py` | `GridSpec`; sinh-stretched S grid with S0, conversion price and soft-call trigger pinned to nodes; time grid with every event on a node |
| `pde.py` | Generic θ-scheme solver: non-uniform central differences, upwinding where needed, Crank–Nicolson + Rannacher restarts, tridiagonal solve. Also used for CDS legs and options. |
| `pricer.py` | `JTDTerms` (soft call, notice, clean/dirty, accrued, coupon-date ordering); `solve_grid`; `price` → dirty/clean, parity, premiums, bond floor, grid Δ/Γ |
| `calibration.py` | CDS bootstrap, convert-implied σ, European options in the model and their implied σ |
| `greeks.py` | `risk_report` (everything in spec §07), `spot_ladder` |
| `examples.py` | The spec's §08 reference convertible and prototype grid |

## Decisions log

| Question | Decision | Rationale |
|---|---|---|
| Where the new term-sheet fields live | New `JTDTerms`, not new fields on `ConvertibleBond` | TF can't model soft calls or notice periods; fields it silently ignored would be a trap |
| Put schedule semantics | **Bermudan** in JTD (put only on the listed dates); TF stays step-function | Spec §03; real converts have discrete put dates |
| Rates, q, b | Flat | Matches the TF side; the spec allows curves but it adds nothing to the model insight |
| Hazard | Piecewise-constant λ0 at the CDS pillars, flat extrapolation | Spec §05 |
| Early exercise | **Projection** after each step (first order in time) | Spec says it's acceptable for v1, and it's what the reference prototype used; penalty method deferred |
| Coupon-date ordering | **Exercise checks happen cum-coupon** (`exercise_before_coupon=True`) | Reproduces the spec's reference run exactly, matches TF, and matches issuers timing calls so forced conversion forfeits the coupon. See "Findings". |
| Call/put prices | Clean + accrued by default; flag `prices_are_clean` | Spec §04 |
| Conversion and accrued | Forfeits accrued by default; flag | Spec §04 |
| Recovery | Par by default; par+accrued and market value selectable | Spec §02. Market-value recovery is linear (folded into discounting), so it ignores post-default conversion. Exact for η = 1. |
| Default grid | 400 sinh nodes, 400 steps/yr | Smallest grid tested that meets the spec's convergence target (halving moves price < 0.02 pts and Δ < 0.001) at S = 40/100/160 |
| Bumped Greeks | Same space and time grid as the base solve | Spec §07. Re-gridding adds a few 1e-3 of delta noise (see `test_close_to_full_reprice_at_bumped_spot`) |
| Re-bootstrap on bumps | Rates, recovery and CDS bumps re-bootstrap when a CDS curve is present; σ, p, q, b bumps don't | Spec §07 wording. Without a CDS curve: CS01 bumps λ0 by 1bp/(1−R), and the recovery bump rescales λ0 to hold λ(1−R) fixed |
| Theta | 1 calendar day (ACT/365) roll, coupons paid in the window added back; `theta_days` for weekends | Spec says "1 business day"; the calendar is out of scope |
| Carry | coupon accrual − Δ·dividends + Δ·S·(r − b) rebate | Spec §07; excludes financing the bond |

## Validation status (spec §08 acceptance criteria)

| Check | Result |
|---|---|
| T1 no credit | 112.1857 (exact 112.1857) ✓ |
| T2 risky zero | 75.5313 ✓ |
| T3 call under JTD | 36.9563 ✓ |
| T4 limits | deep ITM → κS, Δ → κ; κ = 0 → closed-form risky coupon bond ✓ |
| T5 monotonicity | V ≥ max(κS, floor) ✓. ∂V/∂λ0 ≤ 0 and vega ≥ 0 hold except in two regions where the model *correctly* violates them (see Findings) |
| Reference run | 116.45 / Δ 0.780 / floor 84.03, full spot ladder, Δ_eq ladder, vega, CS01, rho, Γ, JTD: all within tolerance ✓ |
| Convergence | halving ΔS and Δt: ≤ 0.004 pts, ≤ 1e-4 Δ at S = 40/100/160 ✓ |
| TF cross-check | no-credit limit agrees with TF to ≤ 0.15 on ~1000–2000 prices, with and without calls; κ = 0, R = 0 floor agrees with TF at s = λ to 0.01 ✓ |

## Findings, including where the spec's text doesn't hold

1. **Coupon-date ordering is worth a full coupon.** Applying call and
   conversion *after* paying the coupon put the reference run 0.3 pts
   high. The call period starts on a coupon date, and at the start of a
   call window there is no earlier callable step to absorb the ordering.
   Ordering cum-coupon reproduces the spec's table and TF.
2. **∂V/∂λ0 > 0 when equity-like** (spec T5 says ≤ 0). Surviving
   stock drifts at `r + ηλ`, so the conversion option gains from λ. T3
   is the pure case: a call under JTD is Black–Scholes with `r → r + λ`.
   Deep in the money this beats the bond leg's loss.
3. **Vega < 0 deep busted when p > 0** (spec T5 says ≥ 0). λ ∝ S^-p
   is convex, so more vol raises E[λ].
4. **The λ0 = s/(1−R) shortcut has no fixed sign of error** (spec §05
   says it overstates). The bootstrapped 5y λ0 is above the shortcut for
   p = 1 and below it for p = 2: Jensen convexity pulls one way,
   survivor drift the other.
5. **European puts: V_D = K·e^{−r(T−t)}, not K** (spec §05). The strike
   of a European put is paid at T. For η < 1 the post-default stock
   keeps diffusing, so V_D is a Black–Scholes price on (1−η)S.

## Deliberately left out (v1)

Beyond spec §09 (exact path-dependent soft call, make-whole/takeover,
resets, contingent conversion, stochastic rates/credit, dilution,
cross-currency):

- penalty-method early exercise (projection instead)
- time-varying conversion ratio κ(t) and restricted conversion windows
- rate / dividend-yield / borrow *curves* (flat only)
- puts exactly at maturity (use `redemption_value`)
- post-default conversion under market-value recovery
- default during the call-notice window (as in the spec)
- listed-option calibration uses European options only
