"""Streamlit demo: time-lapse of a TF convertible's fair value and greeks
as a random GBM spot path evolves.

The heavy lifting (path + greeks precomputation) lives in
``webapp.simulation``. This module is the UI wiring: parameter display,
slider, and the Plotly grid.
"""

import sys
from pathlib import Path

# Make ``src/`` and the project root importable without depending on the
# editable install, which has been flaky under hatchling's PEP-660 mode.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tf_convert_pricer import examples
from tf_convert_pricer.instruments import ConvertibleBond
from tf_convert_pricer.market import MarketData
from webapp.simulation import compute_series, simulate_path


REGIMES = {
    "hybrid": examples.hybrid,
    "equity-like": examples.equity_like,
    "busted": examples.busted,
}

CHART_LABELS = {
    "fair_value": "Fair Value ($)",
    "spot": "Spot ($)",
    "delta": "Delta  [0, 1]",
    "gamma": "Gamma  (1/$)",
    "vega": "Vega  ($ / 1% vol)",
    "credit_spread_sensitivity": "CS DV01  ($ / bp)",
    "theta": "Theta  ($ / day)",
}

VALUE_FMT = {
    "fair_value": "{:.2f}",
    "spot": "{:.2f}",
    "delta": "{:.4f}",
    "gamma": "{:.5f}",
    "vega": "{:.2f}",
    "credit_spread_sensitivity": "{:.4f}",
    "theta": "{:.4f}",
}

N_DAYS = 252
N_FRAMES = 100
TREE_STEPS = 500

# Anthropic-inspired palette (light-blue variant; echoes .streamlit/config.toml)
ACCENT = "#6BA5D5"
INK = "#1A1A1A"
MUTED = "#8B8680"
BG = "#FBF9F5"
CARD = "#F0EDE7"
GRID = "#E5E1DC"


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1A1A1A;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: #1A1A1A;
}

h1 { font-size: 2.4rem; margin-bottom: 0.3rem; }
h2 { font-size: 1.5rem; margin-top: 1.5rem; }
h3 { font-size: 1.15rem; margin-top: 1rem; }

.regime-tag {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    background: #6BA5D5;
    color: #FFFFFF;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

hr { border: none; border-top: 1px solid #E5E1DC; margin: 1.5rem 0; }
</style>
"""


def render_params(bond: ConvertibleBond, market: MarketData, regime_name: str) -> None:
    conversion_price = bond.face_value / bond.conversion_ratio
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Bond terms")
        st.markdown(
            f"""
- Face value: **{bond.face_value:,.0f}**
- Coupon rate: **{bond.coupon_rate:.1%}** ({bond.coupon_frequency}× / year)
- Maturity: **{bond.maturity:.1f}** years
- Conversion ratio: **{bond.conversion_ratio:.1f}** shares / bond
- Conversion price: **{conversion_price:.2f}**
"""
        )
    with col2:
        st.markdown(f"### Market state at t=0  <span class='regime-tag'>{regime_name}</span>", unsafe_allow_html=True)
        st.markdown(
            f"""
- Spot: **{market.spot:.2f}**
- Volatility: **{market.volatility:.1%}**
- Risk-free rate: **{market.risk_free_rate:.2%}**
- Credit spread: **{market.credit_spread:.2%}**
- Dividend yield: **{market.dividend_yield:.2%}**
"""
        )


def render_animated_figure(df: pd.DataFrame) -> go.Figure:
    """Plotly-native animated figure with a growing-line reveal.

    Each frame ``k`` shows the full trace so far (data ``[0:k+1]``),
    plus a marker at the current point. By the last frame the whole
    path is visible. X and Y axes are fixed to the full-data ranges
    so the trace reveals into a stable coordinate system without
    rescaling. The animation slider and play/pause buttons are
    embedded in the Plotly figure — Streamlit doesn't re-run per
    frame, so there's no flashing.
    """
    days = df["day"].tolist()
    max_day = float(df["day"].max())

    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"colspan": 3}, None, None],
            [{}, {}, {}],
            [{}, {}, {}],
        ],
        subplot_titles=[CHART_LABELS[k] for k in (
            "fair_value", "spot", "delta", "gamma",
            "vega", "credit_spread_sensitivity", "theta",
        )],
        row_heights=[0.40, 0.29, 0.29],
        horizontal_spacing=0.08,
        vertical_spacing=0.20,
    )

    charts = [
        ("fair_value", 1, 1),
        ("spot", 2, 1),
        ("delta", 2, 2),
        ("gamma", 2, 3),
        ("vega", 3, 1),
        ("credit_spread_sensitivity", 3, 2),
        ("theta", 3, 3),
    ]

    # Order of subplot_titles annotations matches ``charts`` (row-major).
    chart_order = [c[0] for c in charts]
    base_annotations = list(fig.layout.annotations)

    def title_with_value(key: str, value: float) -> str:
        val = VALUE_FMT[key].format(value)
        return (
            f"{CHART_LABELS[key]}   "
            f"<span style='color:{ACCENT};font-weight:600'>{val}</span>"
        )

    def annotations_for_frame(k: int) -> list[dict]:
        row = df.iloc[k]
        out = []
        for i, ann in enumerate(base_annotations):
            key = chart_order[i]
            ann_dict = ann.to_plotly_json()
            ann_dict["text"] = title_with_value(key, float(row[key]))
            ann_dict["yshift"] = 14
            ann_dict["font"] = dict(family="Fraunces, serif", size=14, color=INK)
            out.append(ann_dict)
        return out

    # Fixed y-ranges per chart from full data range (with small buffer),
    # so the tail moves through a stable coordinate system.
    y_ranges: dict[str, tuple[float, float]] = {}
    for key, _, _ in charts:
        lo, hi = float(df[key].min()), float(df[key].max())
        if hi == lo:
            span = max(abs(hi), 1.0)
            y_ranges[key] = (lo - 0.1 * span, hi + 0.1 * span)
        else:
            pad = 0.10 * (hi - lo)
            y_ranges[key] = (lo - pad, hi + pad)

    x_range = (0.0, max_day * 1.02)

    # Initial state: last frame — full path visible, marker at end.
    initial_k = len(df) - 1
    for key, row, col in charts:
        fig.add_trace(
            go.Scatter(
                x=days[: initial_k + 1],
                y=df[key].iloc[: initial_k + 1].tolist(),
                mode="lines",
                line=dict(color=ACCENT, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[days[initial_k]],
                y=[float(df[key].iloc[initial_k])],
                mode="markers",
                marker=dict(color=ACCENT, size=11, line=dict(color=INK, width=1.2)),
                showlegend=False,
                hovertemplate=f"Day %{{x:.0f}}<br>{key}: %{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            range=x_range, row=row, col=col,
            showgrid=True, gridcolor=GRID, zeroline=False,
            title_text="Day" if row == 3 else None,
        )
        fig.update_yaxes(
            range=y_ranges[key], row=row, col=col,
            showgrid=True, gridcolor=GRID, zeroline=False,
        )

    # Build one frame per timestep. Each frame swaps in an accumulating
    # slice ``[0:k+1]`` (14 traces = 7 line + 7 marker, in add_trace order)
    # plus a fresh annotations list so subplot titles show the current
    # value of each metric.
    frames = []
    for k in range(len(df)):
        frame_data = []
        for key, _, _ in charts:
            frame_data.append(
                go.Scatter(
                    x=days[: k + 1],
                    y=df[key].iloc[: k + 1].tolist(),
                    mode="lines",
                    line=dict(color=ACCENT, width=2.5),
                )
            )
            frame_data.append(
                go.Scatter(
                    x=[days[k]],
                    y=[float(df[key].iloc[k])],
                    mode="markers",
                    marker=dict(color=ACCENT, size=11, line=dict(color=INK, width=1.2)),
                )
            )
        frames.append(
            go.Frame(
                data=frame_data,
                name=str(k),
                layout=go.Layout(annotations=annotations_for_frame(k)),
            )
        )
    fig.frames = frames

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(k)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=f"{int(df['day'].iloc[k])}",
        )
        for k in range(len(df))
    ]
    sliders = [
        dict(
            active=initial_k,
            currentvalue=dict(
                prefix="Day: ",
                font=dict(family="Inter, sans-serif", size=13, color=INK),
            ),
            steps=slider_steps,
            x=0.09,
            y=-0.02,
            len=0.88,
            pad=dict(t=45),
            transition=dict(duration=0),
            bgcolor=CARD,
            activebgcolor=ACCENT,
            font=dict(family="Inter, sans-serif", size=10, color=MUTED),
        )
    ]

    play_button = dict(
        type="buttons",
        direction="right",
        showactive=False,
        xanchor="right",
        yanchor="bottom",
        x=1.0,
        y=1.02,
        pad=dict(t=4, r=4, b=4, l=4),
        bgcolor=ACCENT,
        bordercolor=INK,
        borderwidth=1,
        font=dict(family="Inter, sans-serif", size=14, color="#FFFFFF"),
        buttons=[
            dict(
                label="  ▶  Play  ",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=80, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    ),
                ],
            ),
            dict(
                label="  ❚❚  Pause  ",
                method="animate",
                args=[
                    [None],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0),
                    ),
                ],
            ),
        ],
    )

    fig.update_layout(
        sliders=sliders,
        updatemenus=[play_button],
        annotations=annotations_for_frame(initial_k),
        height=840,
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=50, r=20, t=80, b=120),
    )
    return fig


@st.cache_data(show_spinner=False)
def cached_series(regime_name: str, seed: int) -> pd.DataFrame:
    bond, market = REGIMES[regime_name]()
    path = simulate_path(market, N_DAYS, N_FRAMES, seed)
    return compute_series(bond, market, path, TREE_STEPS, N_DAYS)


def main() -> None:
    st.set_page_config(
        page_title="TF Convert Pricer — Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("TF Convertible Bond — Greek Time-Lapse")
    st.markdown(
        "A random GBM spot path drives a Tsiveriotis-Fernandes convertible bond pricer. "
        "Scrub the slider to see how the bond's fair value and greeks respond as the underlying stock moves."
    )

    with st.expander("About the pricer, and how to read the charts", expanded=True):
        st.markdown(
            """
**The pricer.** Convertibles are valued with the Tsiveriotis-Fernandes (TF) model on a
Cox-Ross-Rubinstein binomial tree of the underlying stock. At each node the bond's value
is split into an **equity component** (discounted at the risk-free rate `r`, because a
converted holder no longer bears the issuer's credit risk) and a **cash-only component**
(discounted at `r + s`, where `s` is the credit spread). Discrete coupons land in the
cash-only leg, and at every interior node the pricer applies put → voluntary conversion →
issuer call decisions before rolling back one step. The tree here runs at
`N = 500` steps, which converges the bond price to a few basis points.

**Greeks are bump-and-reprice.** Perturb one market input (spot, vol, spread, maturity),
reprice, take a finite difference. All values below are shown in the trader-standard
**option notation**:

- **Fair Value** ($) — the model price of the convertible
- **Spot** ($) — the underlying stock, sampled from a GBM path with the regime's own
  drift and vol
- **Delta** — dimensionless, in `[0, 1]`. Interpreted like an option delta: 0.65 means
  the bond moves 65¢ for every $1 of stock, per share of equivalent exposure. Multiply
  by conversion ratio to get shares-to-short hedge size
- **Gamma** — `1/$`, per share equivalent. How fast delta changes as spot moves
- **Vega** — `$` per 1 percentage point of implied vol
- **Credit-spread DV01** — `$` per 1 basis point of credit-spread widening. Negative
  because wider spreads cheapen the bond floor
- **Theta** — `$` per calendar day. Positive when the bond's yield accretion outweighs
  the option's time decay — the "pull to par" you see in credit-sensitive bonds

**A note on gamma noise.** For our examples there's no call schedule, so the payoff is
convex in spot and true gamma is ≥ 0 everywhere. But gamma is a second derivative
computed as `(V₊ − 2V + V₋) / bump²`, which amplifies whatever discretization noise the
tree pricer has. Even at N=500 the noise can flip the sign near zero. We already run a
rolling-median smoother across the greeks to strip the worst spikes; a fully clean gamma
would need N=1000+ or an analytic Malliavin-style adjoint.
            """
        )

    with st.sidebar:
        st.markdown("### Configuration")
        regime_name = st.selectbox("Regime", options=list(REGIMES.keys()), index=0)
        seed = st.number_input("Path seed", value=42, min_value=0, max_value=10_000, step=1)
        st.caption(f"{N_FRAMES + 1} frames over {N_DAYS} trading days, {TREE_STEPS}-step tree per pricing.")

    bond, market = REGIMES[regime_name]()
    render_params(bond, market, regime_name)

    st.markdown("<hr/>", unsafe_allow_html=True)

    with st.spinner("Computing path and greeks…"):
        df = cached_series(regime_name, int(seed))

    if df.empty:
        st.error("Path produced no valid frames — check that horizon < bond maturity.")
        return

    st.plotly_chart(render_animated_figure(df), use_container_width=True)


if __name__ == "__main__":
    main()
