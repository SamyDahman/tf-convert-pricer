# TF Convert Pricer — Project Context

## Purpose

A from-scratch implementation of a Tsiveriotis-Fernandes (TF) convertible
bond pricer, plus a risk/greeks layer built on top of it. This is a
portfolio project: the goal is to demonstrate quant modeling ability and a
disciplined, transparent workflow using Claude Code — not to ship a
production trading system. Built by Samy (quant ETF trader), project #2
after `majra` (a Rust prediction-markets order book library).

Samy wants to *understand* every design decision here, not just receive
working code. Treat this as a paired-learning project, not a
code-generation task.

## Domain glossary

- **Parity** — conversion ratio × current stock price. What the bond would
  be worth if converted right now.
- **Conversion ratio / conversion price** — number of shares per bond /
  the implied stock price at which conversion breaks even.
- **Investment value (bond floor)** — the value of the bond's cash flows
  alone, ignoring the conversion option — a credit-risky, rate-sensitive
  number.
- **Busted / hybrid / equity-like convertible** — regimes describing how
  close the bond trades to its bond floor (busted, stock has cratered),
  a mix of both (hybrid), or its conversion value (equity-like, deep ITM).
- **Credit spread** — the extra yield over risk-free demanded for the
  issuer's credit risk; drives the bond-floor discounting.
- **Delta / gamma** — sensitivity of the convert's price to the stock
  price (first/second order) — the basis for convert-arb delta hedging.
- **Vega** — sensitivity to volatility.
- **Credit-spread duration** — sensitivity to changes in the credit
  spread (the convert-arb analogue of rho).
- **Theta** — time decay of the convert's value, holding other inputs fixed.

## Model: Tsiveriotis-Fernandes

Binomial tree on the underlying stock price. At each node, the bond's
value is split into two components with different discounting, since
conversion removes credit risk but redemption doesn't:

- **Equity component (COCB-complement)** — the part of value attributable
  to the conversion option — discounted at the **risk-free rate**.
- **Cash-only component (COCB)** — the part of value attributable to
  guaranteed cash flows (coupons, redemption) — discounted at
  **risk-free + credit spread**.
- **Total node value** = equity component + cash-only component.
- At each node, apply conversion, call, and put logic before discounting
  back (standard backward induction with early-exercise checks).

This is the standard academic approach (Tsiveriotis & Fernandes, 1998)
and is a defensible, well-documented choice for a project like this —
simpler than a full stochastic-credit-spread model, more realistic than
ignoring credit risk entirely.

## Open modeling assumptions — decide and document in docs/MODEL.md

All decided. See the assumptions log in `docs/MODEL.md` (TF) and its
"Decisions log" (JTD). New ones go there too; don't pick silently.

- [x] Constant credit spread across the tree, or term structure?
- [x] Constant volatility, or surface/smile?
- [x] Discrete dividends (drop at ex-date) or continuous yield?
- [x] Day count / compounding convention
- [x] Number of tree steps and a convergence check
- [x] Call/put schedule representation (date → price map; notice periods ignored for v1?)

## Resuming work

Start with `docs/STATUS.md`: current state, decisions made in
conversation, loose ends, next steps, and how to run and check the app.

## Architecture

```
src/tf_convert_pricer/
    instruments.py   # ConvertibleBond definition (terms, schedules)
    market.py         # MarketData (spot, vol, rate, credit spread, dividends)
    pricer.py         # TF binomial tree pricing engine
    greeks.py         # Greeks via bump-and-reprice on top of pricer
    examples.py       # Made-up instruments spanning equity-like / hybrid / busted
    jtd/              # Second model: jump-to-default PDE (docs/cb_jtd_spec.pdf)
        market.py, grid.py, pde.py, pricer.py, calibration.py, greeks.py, examples.py
tests/                # One test module per source module (test_jtd_* for jtd/)
docs/MODEL.md         # Model writeup + assumptions log (TF and JTD)
docs/STATUS.md        # Handoff: where things stand, loose ends, next steps
docs/cb_jtd_spec.pdf  # The JTD implementation spec
webapp/               # Streamlit demo (app.py, scenarios.py, jtd_simulation.py, simulation.py)
notebooks/            # Exploration / plots (not authoritative — code is)
```

## Conventions

- Python 3.11+, type hints on all public functions
- `dataclasses` for `ConvertibleBond` and `MarketData` — no bare dicts
  for structured domain objects
- One pytest module per source module; tests live alongside features,
  not bolted on at the end
- Formatting: black / ruff (adjust if Samy prefers otherwise)
- No implementation logic belongs in this file — it's context only

## Working style (for Claude Code)

- Explain the *why* behind non-trivial design decisions, not just the *what*
- Explicitly flag anywhere a financial assumption is being simplified
- Keep functions small and independently testable — this is a learning
  project first, a deliverable second
- Push back if a design choice is technically wrong, even if Samy proposed
  it — same working relationship as on majra
- Ask before large refactors or restructuring existing modules
- If uncertain about a piece of finance math, say so plainly rather than
  guessing confidently
