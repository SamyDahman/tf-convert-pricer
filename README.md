# TF Convert Pricer

🚧 In progress.

A convertible bond pricer built on the Tsiveriotis-Fernandes model, plus
a risk/greeks layer (delta, gamma, vega, credit-spread duration, theta).
Built as a hands-on learning project in convertible arbitrage, and as a
demonstration of a disciplined workflow using Claude Code.

## What this is

- A binomial-tree convertible bond pricer that splits bond value into a
  credit-risky cash component and a credit-risk-free equity component
- A set of example instruments spanning equity-like, hybrid, and
  "busted" convertible regimes (see `src/tf_convert_pricer/examples.py`)
- A risk module computing the greeks a convert-arb desk would actually
  hedge against

See [`docs/MODEL.md`](docs/MODEL.md) for the model writeup and the
assumptions log.

## Setup

```bash
pip install -e .
pytest
```

## Roadmap

- [x] Core TF pricer
- [x] Example instruments (equity-like / hybrid / busted)
- [x] Risk / greeks module
- [ ] Write-up + portfolio site

## Built with Claude Code

This project is built using Claude Code as a development tool. This
section will describe, once the build is further along: what was
directed vs. generated, and how correctness was validated (e.g.
sanity checks against known pricing examples and limiting cases).
