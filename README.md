# TF Convert Pricer

A from-scratch convertible bond pricer implementing the Tsiveriotis-Fernandes
(TF) model on a CRR binomial tree, plus a risk/greeks layer and an
interactive Streamlit demo.

## Live demo

**➤ [tf-convert-pricer-zndtdpcb3sohjdexoyjkql.streamlit.app](https://tf-convert-pricer-zndtdpcb3sohjdexoyjkql.streamlit.app)**

Pick a model (**jump-to-default PDE** or the **TF tree**) and a scenario
(equity-like / hybrid / busted, or the JTD spec's soft-callable reference
bond), then scrub the slider or hit Play. A simulated stock path drives
the convertible, and every Greek updates along it. A value-vs-stock
panel shows the dot riding the model's curve. Under JTD the path can
actually default: the stock goes to zero and the bond pays recovery. The
page also shows risk tiles at t = 0 and a TF-vs-JTD comparison across
the stock price, which highlights the credit delta TF can't see.

Streamlit Community Cloud's free tier sleeps idle apps, so the first
visit after a lull may take ~30 seconds to wake.

## What's in the repo

- **`src/tf_convert_pricer/`** — the pricer library
  - `pricer.py` — CRR stock tree, TF terminal split, two-rate backward
    induction, coupon injection, interior early-exercise (put / convert
    / call)
  - `greeks.py` — delta, gamma, vega, credit-spread sensitivity, theta,
    via bump-and-reprice
  - `examples.py` — three regime-tagged `(bond, market)` pairs
  - `jtd/` — a second model: a jump-to-default PDE with a
    stock-dependent hazard rate, CDS bootstrap, soft call with notice,
    Bermudan puts, discrete dividends, and a full risk report (credit-delta
    split, CS01 buckets, jump-to-default). Implements
    [`docs/cb_jtd_spec.pdf`](docs/cb_jtd_spec.pdf) and reproduces its
    reference run
- **`webapp/`** — Streamlit demo (`app.py`); `scenarios.py` (one bond,
  both models, JTD hazard calibrated to TF's credit spread),
  `jtd_simulation.py` (default-capable path, curve-based JTD series),
  `simulation.py` (TF path and series)
- **`docs/MODEL.md`** — model writeup and assumptions log

Sanity checks baked into the test suite:

- `conversion_ratio → 0` reproduces a plain credit-risky coupon bond
  by closed form (matched to floating-point precision)
- `credit_spread → 0` collapses the TF split to a single-rate pricer
  and every single-rate test still passes unchanged
- Deep-ITM and deep-OTM regimes match parity and the bond floor
  respectively

181 tests, all passing (JTD adds the spec's T1–T5, reference run,
convergence check, and cross-checks against the TF pricer).

## Local setup

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,web]'
pytest                          # runs the whole suite in ~20s
streamlit run webapp/app.py     # opens on http://localhost:8501
```

## Model

See [`docs/MODEL.md`](docs/MODEL.md) for the writeup, the v1 assumptions
log, and the deliberate simplifications called out honestly (flat credit
spread, constant vol, continuous dividend yield, no call notice periods,
hard-call schedule only).

## Roadmap

- [x] Core TF pricer
- [x] Example instruments (equity-like / hybrid / busted)
- [x] Risk / greeks module
- [x] Interactive demo webapp, deployed publicly
- [x] Jump-to-default PDE model + risk report
- [x] JTD model in the webapp
- [ ] Write-up

## Built with Claude Code

Built as a paired-learning project using Claude Code — design decisions
were surfaced and discussed at each step rather than silently made.
`docs/MODEL.md` records the "why" behind every modeling choice, and
`CLAUDE.md` captures the working conventions and constraints that shaped
the collaboration.
