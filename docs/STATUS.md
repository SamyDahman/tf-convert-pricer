# Project status & handoff

Read this first when resuming. `CLAUDE.md` has the working conventions,
`docs/MODEL.md` has the modeling decisions for both models.

_Last updated: 2026-09-27_

## Where things stand

Two convertible pricers share one `ConvertibleBond`:

| | TF tree (`pricer.py`, `greeks.py`) | Jump-to-default PDE (`jtd/`) |
|---|---|---|
| Credit | flat spread `s`, cash leg at `r + s` | default intensity `λ(S) = λ0 (S0/S)^p`, recovery on default |
| Numerics | CRR tree, 500 steps | Crank–Nicolson + Rannacher, sinh grid, projection for early exercise |
| Greeks | bump-and-reprice (noisy gamma) | grid Δ/Γ + same-grid bumps; full spec §07 report |
| Spec | — | `docs/cb_jtd_spec.pdf` (reproduced: T1–T3, reference run, convergence) |

The Streamlit app (`webapp/`) defaults to JTD, keeps TF as a toggle, and
is deployed from `main`: <https://tf-convert-pricer-zndtdpcb3sohjdexoyjkql.streamlit.app>.

185 tests pass (`pytest`, ~35s; the JTD scenario calibration is the slow part).

## Decisions made in conversation (not obvious from code)

- **JTD lives in its own subpackage; TF is untouched.** New term-sheet
  fields (soft-call trigger, notice, clean/dirty, accrued, coupon-date
  ordering) are in `JTDTerms`, not `ConvertibleBond`, so TF never silently
  ignores fields it can't model.
- **Puts are Bermudan in JTD, step-function in TF.** Same `put_schedule`
  field, different semantics. Documented in both docstrings.
- **`exercise_before_coupon=True` is the default.** It reproduces the spec's
  reference run exactly and matches TF; the other ordering was 0.3 pts off
  (see MODEL.md "Findings").
- **Spec claims we deliberately don't follow**, each with a test: ∂V/∂λ0 > 0
  when equity-like, vega < 0 deep busted with p > 0, the s/(1−R) shortcut's
  error sign, and European-put V_D = K·e^{−r(T−t)}.
- **Webapp scenarios calibrate JTD jointly** to TF's 5y credit spread *and* a
  1y ATM option at the TF (Black–Scholes) vol. Reusing the BS vol as the
  diffusion vol double-counts default: it made busted deltas look flat and
  too high. Busted with p = 2 has no consistent σ, so it's floored at 5% and
  the app warns.
- **Scope cuts, per Samy** ("exclude narrow, arduous things"): penalty
  method, κ(t), conversion windows, rate/div curves, exact soft call,
  make-whole, etc. The full list is in MODEL.md "Deliberately left out".

## Loose ends

- `docs/` has two uncommitted screenshot changes: the Sep 13 screenshot is
  deleted on disk (it was tracked), and a Sep 27 screenshot (seed 44
  busted, used to debug the flat-delta question) is untracked. Nothing
  references either. Samy hasn't decided whether to keep them.

## Candidate next steps (none started)

- **PDE speed**: the time loop is plain Python and rebuilds the tridiagonal
  matrix every step. Caching per (step size, hazard bucket) or numba
  should give 3–5×. The default solve is ~85 ms and the risk report ~1.5 s.
  Measured: TF is ~20× faster per price (3.9 ms at N = 500) but ~20× less
  accurate, and oscillates in N.
- Penalty method (Forsyth–Vetzal) for second-order early exercise.
- CDS-curve input in the app (bucketed CS01 is already implemented in
  `jtd.greeks`, just not surfaced; with a CDS curve the report takes
  several seconds).
- Write-up (the README roadmap item).

## How to run

```bash
.venv/bin/python -m pytest -q
# --server.headless true skips Streamlit's first-run email prompt,
# which otherwise blocks when launched from Claude Code's `!` shell.
.venv/bin/streamlit run webapp/app.py --server.headless true   # → http://localhost:8501
```

`st.cache_data` doesn't notice edits to imported modules
(`webapp/scenarios.py`, `jtd/*`): restart the server after changing them.

Checking the app without a browser: `streamlit.testing.v1.AppTest`
(absolute path to `webapp/app.py`) catches exceptions. For visuals, write a
figure with `fig.write_html(..., auto_play=False)` and screenshot it with
headless Chrome. Screenshotting the live Streamlit page only captures the
loading skeleton.
