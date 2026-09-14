# TF Convert Pricer

A from-scratch convertible bond pricer implementing the Tsiveriotis-Fernandes
(TF) model on a CRR binomial tree, plus a risk/greeks layer and an
interactive Streamlit demo.

## Live demo

**➤ [tf-convert-pricer-zndtdpcb3sohjdexoyjkql.streamlit.app](https://tf-convert-pricer-zndtdpcb3sohjdexoyjkql.streamlit.app)**

Pick a regime (equity-like / hybrid / busted), scrub the slider or hit
Play, and watch how the bond's fair value and greeks respond as a random
GBM spot path evolves. Delta, gamma, vega, credit-spread DV01, and theta
are all shown in trader-standard option notation. Full explanation of
the pricer and the display conventions is in the expander at the top of
the app.

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
- **`webapp/`** — Streamlit demo (`app.py`) plus the path-simulation
  helper (`simulation.py`) it depends on
- **`docs/MODEL.md`** — model writeup and assumptions log

Sanity checks baked into the test suite:

- `conversion_ratio → 0` reproduces a plain credit-risky coupon bond
  by closed form (matched to floating-point precision)
- `credit_spread → 0` collapses the TF split to a single-rate pricer
  and every single-rate test still passes unchanged
- Deep-ITM and deep-OTM regimes match parity and the bond floor
  respectively

69 tests, all passing.

## Local setup

Requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,web]'
pytest                          # runs the whole suite in ~1s
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
- [ ] Write-up

## Built with Claude Code

Built as a paired-learning project using Claude Code — design decisions
were surfaced and discussed at each step rather than silently made.
`docs/MODEL.md` records the "why" behind every modeling choice, and
`CLAUDE.md` captures the working conventions and constraints that shaped
the collaboration.
