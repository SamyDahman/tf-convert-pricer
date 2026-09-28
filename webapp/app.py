"""Streamlit demo: a convertible's fair value and greeks along a simulated
stock path, under either the jump-to-default PDE or the TF tree.

Page layout, top to bottom:
  1. Terms and market for the chosen scenario.
  2. Snapshot risk at t = 0 (metric tiles).
  3. The animated time-lapse. Its top-right panel is the static
     value-vs-stock curve with a dot that rides it frame by frame, so
     every time series below reads as "where on this curve are we".
  4. TF vs JTD across the stock price — where the two models disagree.

Numerics live in ``webapp.simulation`` (TF), ``webapp.jtd_simulation``
and ``webapp.scenarios``; this module is UI wiring only.
"""

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

# Make ``src/`` and the project root importable without depending on the
# editable install, which has been flaky under hatchling's PEP-660 mode.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tf_convert_pricer.greeks import (
    credit_spread_sensitivity,
    delta,
    gamma,
    theta,
    vega,
)
from tf_convert_pricer.jtd.greeks import risk_report
from tf_convert_pricer.pricer import price as tf_price
from webapp.jtd_simulation import (
    compute_jtd_series,
    jtd_profile,
    simulate_jtd_path,
    solve_curves,
    tf_profile,
)
from webapp.scenarios import SCENARIOS, Scenario, build_scenario
from webapp.simulation import compute_series, simulate_path

JTD = "Jump-to-default PDE"
TF = "Tsiveriotis-Fernandes tree"

N_DAYS = 252
N_FRAMES = 100
TREE_STEPS = 500

# Anthropic-inspired palette (light-blue variant; echoes .streamlit/config.toml)
ACCENT = "#6BA5D5"
PARITY = "#D08C4A"
FLOOR = "#6E9C82"
TF_COLOR = "#8B6BB5"
ALERT = "#C0504D"
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

[data-testid="stMetricValue"] { font-family: 'Fraunces', serif; font-size: 1.6rem; }

hr { border: none; border-top: 1px solid #E5E1DC; margin: 1.5rem 0; }
</style>
"""


# ── Panels ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Overlay:
    column: str
    name: str
    color: str
    dash: str = "dash"


@dataclass(frozen=True)
class Panel:
    key: str
    title: str
    fmt: str
    overlays: tuple[Overlay, ...] = field(default_factory=tuple)


VALUE_OVERLAYS = (Overlay("parity", "parity κS", PARITY), Overlay("bond_floor", "bond floor", FLOOR, "dot"))

COMMON_PANELS = {
    "fair_value": Panel("fair_value", "Fair Value ($)", "{:.2f}", VALUE_OVERLAYS),
    "spot": Panel("spot", "Spot ($)", "{:.2f}"),
    "gamma": Panel("gamma", "Gamma  (1/$)", "{:.5f}"),
    "vega": Panel("vega", "Vega  ($ / 1% vol)", "{:.2f}"),
    "credit_spread_sensitivity": Panel("credit_spread_sensitivity", "CS01  ($ / bp)", "{:.4f}"),
    "theta": Panel("theta", "Theta  ($ / day)", "{:.4f}"),
}

JTD_GRID = [
    ["fair_value", "fair_value", "profile"],
    ["spot", "delta", "gamma"],
    ["vega", "credit_spread_sensitivity", "theta"],
    ["hazard_pct", "jtd_naked", "jtd_hedged"],
]
JTD_PANELS = {
    **COMMON_PANELS,
    "delta": Panel("delta", "Delta  [per share]", "{:.4f}", (Overlay("equity_delta", "Δ, hazard frozen", MUTED),)),
    "hazard_pct": Panel("hazard_pct", "Hazard λ(S)  (%/yr)", "{:.2f}"),
    "jtd_naked": Panel("jtd_naked", "Jump-to-default, naked ($)", "{:.1f}"),
    "jtd_hedged": Panel("jtd_hedged", "Jump-to-default, Δ-hedged ($)", "{:.1f}"),
}

TF_GRID = [
    ["fair_value", "fair_value", "profile"],
    ["spot", "delta", "gamma"],
    ["vega", "credit_spread_sensitivity", "theta"],
]
TF_PANELS = {**COMMON_PANELS, "delta": Panel("delta", "Delta  [per share]", "{:.4f}")}


# ── Rendering ─────────────────────────────────────────────────────────


def render_params(scenario: Scenario, model: str) -> None:
    bond, market = scenario.bond, scenario.jtd_market
    conversion_price = bond.face_value / bond.conversion_ratio
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Bond terms")
        call = "none"
        if bond.call_schedule:
            (start, level), *_ = sorted(bond.call_schedule.items())
            trigger = scenario.jtd_terms.call_trigger
            call = f"from yr {start:g} at {level:g} + accrued" + (
                f", if S ≥ {trigger * conversion_price:.0f}" if trigger else ""
            )
        st.markdown(
            f"""
- Face **{bond.face_value:,.0f}**, maturity **{bond.maturity:.1f}**y
- Coupon **{bond.coupon_rate:.1%}** ({bond.coupon_frequency}× / year)
- Conversion ratio **{bond.conversion_ratio:g}** → price **{conversion_price:.2f}**
- Call: **{call}**
"""
        )
    with col2:
        st.markdown(
            f"### Market at t=0  <span class='regime-tag'>{scenario.name}</span>",
            unsafe_allow_html=True,
        )
        if scenario.quoted_volatility is not None:
            vol_line = f"1y ATM implied vol (Black–Scholes) **{scenario.quoted_volatility:.1%}**"
        else:
            vol_line = f"diffusion vol **{market.volatility:.1%}** (spec input)"
        st.markdown(
            f"""
- Spot **{market.spot:.2f}**, {vol_line}
- Risk-free **{market.risk_free_rate:.2%}**, dividend yield **{market.dividend_yield:.2%}**
- 5y credit spread **{scenario.cds_spread:.2%}**
"""
        )
    with col3:
        st.markdown("### Credit model")
        if model == JTD:
            st.markdown(
                f"""
- Hazard at spot **λ0 = {market.hazard.intensities[0]:.2%}**/yr, diffusion vol **σ = {market.volatility:.1%}**
- Elasticity **p = {market.hazard_elasticity:g}** (λ ∝ S<sup>−p</sup>), cap **{market.lambda_max:g}**
- Recovery **R = {market.recovery:.0%}** of par, stock loss **η = {market.jump_size:.0%}**
""",
                unsafe_allow_html=True,
            )
            if scenario.quoted_volatility is not None:
                st.caption(
                    "λ0 and σ are fitted jointly: a 5y CDS reprices at the credit spread and a 1y ATM option "
                    "at its Black–Scholes value. σ sits below the BS vol because the BS vol already prices "
                    "the default risk that JTD models through λ."
                )
        else:
            st.markdown(
                f"""
- Cash-only leg discounted at **r + s = {scenario.tf_market.risk_free_rate + scenario.tf_market.credit_spread:.2%}**
- Equity leg discounted at **r = {scenario.tf_market.risk_free_rate:.2%}**
- No default event, no recovery; the tree uses the BS vol **{scenario.tf_market.volatility:.1%}** directly
"""
            )


def render_snapshot(tiles: list[tuple[str, str, str | None]]) -> None:
    for start in range(0, len(tiles), 5):
        cols = st.columns(5)
        for col, (label, value, help_text) in zip(cols, tiles[start : start + 5], strict=False):
            col.metric(label, value, help=help_text)


def _padded_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    if hi == lo:
        span = max(abs(hi), 1.0)
        return (lo - 0.1 * span, hi + 0.1 * span)
    pad = 0.10 * (hi - lo)
    return (lo - pad, hi + pad)


def render_animated_figure(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    layout: list[list[str]],
    panels: dict[str, Panel],
    default_day: float | None,
) -> go.Figure:
    """Animated grid: growing-line reveal on every time series panel,
    plus a dot riding the static value-vs-stock curve.

    Frames update only the animated traces (``go.Frame(traces=...)``);
    the profile curves are drawn once. Slider and play/pause are native
    Plotly controls, so Streamlit doesn't re-run per frame.
    """
    n_rows = len(layout)
    specs: list[list[dict | None]] = []
    cells: list[tuple[str, int, int]] = []
    for r, row in enumerate(layout, start=1):
        spec_row: list[dict | None] = []
        c = 1
        while c <= len(row):
            key = row[c - 1]
            span = 1
            while c - 1 + span < len(row) and row[c - 1 + span] == key:
                span += 1
            spec_row.append({"colspan": span} if span > 1 else {})
            spec_row.extend([None] * (span - 1))
            cells.append((key, r, c))
            c += span
        specs.append(spec_row)

    def static_title(key: str) -> str:
        return "Value vs. stock price  (dot = today)" if key == "profile" else panels[key].title

    fig = make_subplots(
        rows=n_rows,
        cols=3,
        specs=specs,
        subplot_titles=[static_title(k) for k, _, _ in cells],
        row_heights=[0.34] + [0.66 / (n_rows - 1)] * (n_rows - 1),
        horizontal_spacing=0.07,
        vertical_spacing=0.33 / n_rows,
    )
    base_annotations = list(fig.layout.annotations)
    days = df["day"].to_numpy()
    x_range = (0.0, float(days.max()) * 1.02)
    last = len(df) - 1
    legend_seen: set[str] = set()

    def legend_kwargs(name: str) -> dict:
        show = name not in legend_seen
        legend_seen.add(name)
        return dict(name=name, legendgroup=name, showlegend=show)

    animated: list[tuple[int, str, str, dict]] = []  # (trace index, x col, y col, style)

    def add(trace: go.Scatter, row: int, col: int) -> int:
        fig.add_trace(trace, row=row, col=col)
        return len(fig.data) - 1

    for key, row, col in cells:
        if key == "profile":
            fig.add_trace(
                go.Scatter(x=profile["spot"], y=profile["parity"], mode="lines",
                           line=dict(color=PARITY, width=1.8, dash="dash"), hoverinfo="skip", **legend_kwargs("parity κS")),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(x=profile["spot"], y=profile["floor"], mode="lines",
                           line=dict(color=FLOOR, width=1.8, dash="dot"), hoverinfo="skip", **legend_kwargs("bond floor")),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(x=profile["spot"], y=profile["value"], mode="lines",
                           line=dict(color=ACCENT, width=2.5), hoverinfo="skip", **legend_kwargs("convertible")),
                row=row, col=col,
            )
            idx = add(
                go.Scatter(x=[df["spot"].iloc[last]], y=[df["fair_value"].iloc[last]], mode="markers",
                           marker=dict(color=ACCENT, size=12, line=dict(color=INK, width=1.2)), showlegend=False,
                           hovertemplate="S %{x:.2f}<br>V %{y:.2f}<extra></extra>"),
                row, col,
            )
            animated.append((idx, "spot", "fair_value", {"marker": True}))
            fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, row=row, col=col)
            y_hi = float(np.nanmax(profile["value"])) * 1.05
            fig.update_yaxes(range=(0.0, y_hi), showgrid=True, gridcolor=GRID, zeroline=False, row=row, col=col)
            continue

        panel = panels[key]
        columns = [panel.key, *(o.column for o in panel.overlays)]
        for overlay in panel.overlays:
            idx = add(
                go.Scatter(x=days, y=df[overlay.column], mode="lines",
                           line=dict(color=overlay.color, width=1.6, dash=overlay.dash), hoverinfo="skip",
                           **legend_kwargs(overlay.name)),
                row, col,
            )
            animated.append((idx, "day", overlay.column, {}))
        idx = add(
            go.Scatter(x=days, y=df[panel.key], mode="lines", line=dict(color=ACCENT, width=2.5),
                       hoverinfo="skip", showlegend=False),
            row, col,
        )
        animated.append((idx, "day", panel.key, {}))
        idx = add(
            go.Scatter(x=[days[last]], y=[df[panel.key].iloc[last]], mode="markers",
                       marker=dict(color=ACCENT, size=10, line=dict(color=INK, width=1.2)), showlegend=False,
                       hovertemplate=f"Day %{{x:.0f}}<br>{panel.key}: %{{y:.4f}}<extra></extra>"),
            row, col,
        )
        animated.append((idx, "day", panel.key, {"marker": True}))
        fig.update_xaxes(range=x_range, showgrid=True, gridcolor=GRID, zeroline=False, row=row, col=col)
        fig.update_yaxes(range=_padded_range(df[columns].to_numpy(dtype=float).ravel()),
                         showgrid=True, gridcolor=GRID, zeroline=False, row=row, col=col)
        if default_day is not None:
            fig.add_vline(x=default_day, line=dict(color=ALERT, width=1.5, dash="dash"), row=row, col=col)

    def annotations_for_frame(k: int) -> list[dict]:
        out = []
        for ann, (key, _, _) in zip(base_annotations, cells, strict=True):
            ann_dict = ann.to_plotly_json()
            if key != "profile":
                value = df[panels[key].key].iloc[k]
                shown = "—" if not np.isfinite(value) else panels[key].fmt.format(value)
                ann_dict["text"] = f"{panels[key].title}   <span style='color:{ACCENT};font-weight:600'>{shown}</span>"
            ann_dict["yshift"] = 12
            ann_dict["font"] = dict(family="Fraunces, serif", size=14, color=INK)
            out.append(ann_dict)
        return out

    def frame_traces(k: int) -> list[go.Scatter]:
        traces = []
        for _, x_col, y_col, style in animated:
            if style.get("marker"):
                traces.append(go.Scatter(x=[df[x_col].iloc[k]], y=[df[y_col].iloc[k]]))
            else:
                traces.append(go.Scatter(x=df[x_col].iloc[: k + 1], y=df[y_col].iloc[: k + 1]))
        return traces

    trace_ids = [idx for idx, *_ in animated]
    fig.frames = [
        go.Frame(data=frame_traces(k), traces=trace_ids, name=str(k), layout=go.Layout(annotations=annotations_for_frame(k)))
        for k in range(len(df))
    ]

    animate = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
    slider = dict(
        active=last,
        currentvalue=dict(prefix="Day: ", font=dict(family="Inter, sans-serif", size=13, color=INK)),
        steps=[dict(method="animate", args=[[str(k)], animate], label=f"{int(days[k])}") for k in range(len(df))],
        x=0.15, y=-0.02, len=0.85, pad=dict(t=45),
        transition=dict(duration=0), bgcolor=CARD, activebgcolor=ACCENT,
        font=dict(family="Inter, sans-serif", size=10, color=MUTED),
    )
    play_button = dict(
        type="buttons", direction="right", showactive=False,
        xanchor="left", yanchor="top", x=0.0, y=-0.035,
        pad=dict(t=4, r=4, b=4, l=4), bgcolor=ACCENT, bordercolor=INK, borderwidth=1,
        font=dict(family="Inter, sans-serif", size=14, color="#FFFFFF"),
        buttons=[
            dict(label="  ▶  Play  ", method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="  ❚❚  Pause  ", method="animate", args=[[None], animate]),
        ],
    )
    fig.update_layout(
        sliders=[slider],
        updatemenus=[play_button],
        annotations=annotations_for_frame(last),
        height=280 * n_rows,
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", x=0.0, y=1.06, xanchor="left", yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=110, b=130),
    )
    return fig


def render_comparison(jtd: pd.DataFrame, tf: pd.DataFrame | None, spot: float) -> go.Figure:
    """Static TF-vs-JTD profiles: value (with floors) and delta (with split)."""
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.08,
        subplot_titles=("Value vs. stock", "Delta per share vs. stock"),
    )

    def line(df, col, name, color, dash="solid", width=2.2, c=1):
        fig.add_trace(go.Scatter(x=df["spot"], y=df[col], mode="lines", name=name,
                                 line=dict(color=color, dash=dash, width=width)), row=1, col=c)

    line(jtd, "parity", "parity κS", PARITY, "dash", 1.6)
    line(jtd, "floor", "JTD bond floor", FLOOR, "dot")
    line(jtd, "value", "JTD convertible", ACCENT, width=2.8)
    line(jtd, "delta", "JTD Δ (total)", ACCENT, width=2.8, c=2)
    line(jtd, "equity_delta", "JTD Δ, hazard frozen", MUTED, "dash", c=2)
    if tf is not None:
        line(tf, "floor", "TF bond floor", TF_COLOR, "dot", 1.6)
        line(tf, "value", "TF convertible", TF_COLOR, "dash")
        line(tf, "delta", "TF Δ", TF_COLOR, "dash", c=2)
    fig.add_trace(go.Scatter(x=jtd["spot"], y=jtd["delta"] - jtd["equity_delta"], fill="tozeroy", mode="none",
                             fillcolor="rgba(208,140,74,0.18)", name="credit delta"), row=1, col=2)
    for c in (1, 2):
        fig.add_vline(x=spot, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=c)
        fig.update_xaxes(title_text="Stock price", showgrid=True, gridcolor=GRID, row=1, col=c)
        fig.update_yaxes(showgrid=True, gridcolor=GRID, row=1, col=c)
    fig.update_layout(
        height=430, paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", y=-0.25), margin=dict(l=50, r=20, t=50, b=40),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(family="Fraunces, serif", size=14, color=INK)
    return fig


# ── Cached computation ────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def cached_scenario(name: str, p: float, recovery: float) -> Scenario:
    return build_scenario(name, p, recovery)


@st.cache_data(show_spinner=False)
def cached_jtd(name: str, p: float, recovery: float, seed: int, allow_default: bool):
    sc = cached_scenario(name, p, recovery)
    bond, market = sc.bond, sc.jtd_market
    curves = solve_curves(bond, market, sc.jtd_terms)
    path = simulate_jtd_path(market, N_DAYS, N_FRAMES, seed, allow_default)
    df = compute_jtd_series(bond, market, curves, path, N_DAYS)
    df["hazard_pct"] = 100.0 * df["hazard"]
    upper = max(2.5 * bond.face_value / bond.conversion_ratio, 1.1 * float(path.spots.max()))
    profile = jtd_profile(bond, market, curves, np.linspace(0.0, upper, 160))
    default_day = None if path.default_frame is None else float(df["day"].iloc[path.default_frame])
    return df, profile, default_day


@st.cache_data(show_spinner=False)
def cached_jtd_tiles(name: str, p: float, recovery: float) -> list[tuple[str, str, str | None]]:
    sc = cached_scenario(name, p, recovery)
    rep = risk_report(sc.bond, sc.jtd_market, sc.jtd_terms)
    to_cash = sc.bond.face_value / 100.0
    pr = rep.pricing
    return [
        ("Fair value", f"{pr.dirty_price:,.2f}", f"{pr.clean_points:.2f} points of par (clean)"),
        ("Conversion premium", f"{pr.conversion_premium:.1%}", f"Parity {pr.parity:,.2f}"),
        ("Investment premium", f"{pr.investment_premium:.1%}", f"Bond floor {pr.bond_floor:,.2f} (same model, κ = 0)"),
        ("Delta / share", f"{rep.delta_pct_parity:.3f}",
         (f"{rep.delta:.2f} shares per bond. Hazard-frozen Δ {rep.equity_delta / sc.bond.conversion_ratio:.3f}; "
          f"the rest ({rep.credit_delta / sc.bond.conversion_ratio:+.3f}) is credit delta.")),
        ("Gamma · S · 1%", f"{rep.gamma_per_1pct:.3f}", "Change in hedge shares per bond for a 1% stock move"),
        ("Vega", f"{rep.vega * to_cash:.2f}", "$ per bond per 1 vol point"),
        ("CS01", f"{rep.cs01 * to_cash:.4f}", "$ per bond per 1bp of spread (λ0 bumped by 1bp/(1−R))"),
        ("Rho", f"{rep.rho * to_cash:.4f}", "$ per bond per 1bp of rates"),
        ("Theta", f"{rep.theta * to_cash:.4f}", "$ per bond per calendar day, coupons included"),
        ("Jump-to-default", f"{rep.jtd_naked * to_cash:,.1f}",
         f"Naked loss on default. Delta-hedged: {rep.jtd_hedged * to_cash:+,.1f} (short Δ shares gain ηS each)"),
    ]


@st.cache_data(show_spinner=False)
def cached_tf(name: str, seed: int):
    sc = cached_scenario(name, 1.0, 0.4)
    bond, market = sc.bond, sc.tf_market
    path = simulate_path(market, N_DAYS, N_FRAMES, seed)
    df = compute_series(bond, market, path, TREE_STEPS, N_DAYS)
    upper = max(2.5 * bond.face_value / bond.conversion_ratio, 1.1 * float(path.max()))
    profile = tf_profile(bond, market, np.linspace(upper / 60, upper, 60))
    profile["parity"] = bond.conversion_ratio * profile["spot"]
    df["bond_floor"] = np.interp(df["spot"], profile["spot"], profile["floor"])
    return df, profile


@st.cache_data(show_spinner=False)
def cached_tf_profile(name: str, spots: tuple[float, ...]) -> pd.DataFrame:
    sc = cached_scenario(name, 1.0, 0.4)
    return tf_profile(sc.bond, sc.tf_market, np.asarray(spots))


@st.cache_data(show_spinner=False)
def cached_tf_tiles(name: str) -> list[tuple[str, str, str | None]]:
    sc = cached_scenario(name, 1.0, 0.4)
    bond, market = sc.bond, sc.tf_market
    kappa = bond.conversion_ratio
    result = tf_price(bond, market, TREE_STEPS)
    floor = tf_price(replace(bond, conversion_ratio=0.0), market, TREE_STEPS).price
    return [
        ("Fair value", f"{result.price:,.2f}", f"Equity leg {result.equity_component:,.2f}, cash leg {result.cash_only_component:,.2f}"),
        ("Conversion premium", f"{result.price / result.parity - 1:.1%}", f"Parity {result.parity:,.2f}"),
        ("Investment premium", f"{result.price / floor - 1:.1%}", f"Bond floor {floor:,.2f}"),
        ("Delta / share", f"{delta(bond, market, TREE_STEPS) / kappa:.3f}", "No credit channel in TF: spreads don't move with the stock"),
        ("Gamma · S · 1%", f"{gamma(bond, market, TREE_STEPS) * market.spot * 0.01:.3f}", "Change in hedge shares per bond for a 1% stock move"),
        ("Vega", f"{vega(bond, market, TREE_STEPS) * 0.01:.2f}", "$ per bond per 1 vol point"),
        ("CS01", f"{credit_spread_sensitivity(bond, market, TREE_STEPS) * 1e-4:.4f}", "$ per bond per 1bp of credit spread"),
        ("Theta", f"{theta(bond, market, TREE_STEPS) / 365:.4f}", "$ per bond per calendar day"),
    ]


# ── Page ──────────────────────────────────────────────────────────────

ABOUT = """
**Two models, one bond.** Both price the same convertible and differ in how they treat
credit.

- **Jump-to-default PDE (default).** The stock diffuses, but the issuer can default at
  any moment with intensity `λ(S) = λ0·(S0/S)^p`, which rises as the stock falls. On
  default the stock drops by `η` (here 100%) and the bond pays recovery `R` of par.
  Surviving stock drifts up by `ηλ` to pay for that risk. The price solves a PDE on a
  stock grid (Crank–Nicolson), so **one solve gives the whole value-vs-stock curve**,
  and delta and gamma are read straight off it. `λ0` and the diffusion vol `σ` are fitted
  jointly so that a 5y CDS prices at the scenario's credit spread and a 1y at-the-money
  option at its Black–Scholes value. `σ` comes out below the BS vol, because a BS vol
  already prices the default risk JTD adds through `λ`; reusing it would count that risk
  twice.
- **Tsiveriotis–Fernandes tree.** Value splits into an equity leg discounted at `r` and a
  cash leg discounted at `r + s`, on a 500-step binomial tree. The spread is a fixed
  number, so there's no default event and no credit delta.

**How to read the time-lapse.** The stock follows one simulated path. The bond's remaining
maturity is held fixed, so the top-right dot slides along a curve that doesn't move, and
every other panel is a Greek read at that point. Under JTD the path can **default**
(red dashed line): the stock goes to zero, the bond is worth its recovery, and all
sensitivities stop.

**Units.** Money is $ per bond. Delta and gamma are per share of conversion (divide by κ),
so delta lives in `[0, 1]`, except deep in the busted zone under JTD, where the credit
channel can push it higher. Vega is per vol point, CS01 and rho per basis point, theta per
calendar day.

**What JTD shows that TF can't.**
- **Credit delta.** In the delta panel, the grey dashed line freezes the hazard at its
  current level. The gap to the blue line is the part of the hedge that comes from spreads
  moving with the stock. It's large when busted and vanishes when equity-like.
- **The bond floor rises with the stock**, because a higher stock means lower hazard.
  TF's floor is flat.
- **Jump-to-default.** The naked loss if the issuer defaults today, and what's left after
  the delta hedge (short Δ shares, which fall by ηS on default).
- **Gamma can go negative** in the busted zone, where delta falls as the stock rises
  because credit improves.
"""


def main() -> None:
    st.set_page_config(page_title="Convertible Pricer — JTD & TF", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Configuration")
        model = st.radio("Model", (JTD, TF), index=0)
        scenario_name = st.selectbox("Scenario", options=SCENARIOS, index=0)
        seed = int(st.number_input("Path seed", value=42, min_value=0, max_value=10_000, step=1))
        st.markdown("### Credit model (JTD)")
        p = st.slider("Hazard elasticity p", 0.0, 2.0, 1.0, 0.25,
                      help="λ ∝ S^-p: how fast credit deteriorates as the stock falls. p = 0 is a constant hazard.")
        recovery = st.slider("Recovery R (% of par)", 0.0, 0.8, 0.4, 0.05)
        allow_default = st.checkbox("Allow default on the path", value=True)
        st.caption(f"{N_FRAMES + 1} frames over {N_DAYS} trading days.")

    scenario = cached_scenario(scenario_name, p, recovery)
    if model == JTD and scenario.vol_floored:
        st.sidebar.warning(
            f"No diffusion vol reprices the {scenario.quoted_volatility:.0%} ATM option at this p: the hazard "
            "alone already makes the option worth more than its market price. σ is floored at 5%. Lower p "
            "to get a consistent fit."
        )
    if model == TF and scenario.tf_market is None:
        st.sidebar.warning("The TF tree has no soft-call trigger, so this scenario is JTD-only.")
        model = JTD

    st.title("Convertible Bond — Greek Time-Lapse")
    st.markdown(
        f"Pricing with the **{model}**. A simulated stock path drives the convertible; "
        "scrub the slider or hit Play to watch fair value and every Greek respond."
    )
    with st.expander("About the models, and how to read the charts", expanded=False):
        st.markdown(ABOUT)

    render_params(scenario, model)
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("## Risk at t = 0")
    with st.spinner("Pricing…"):
        tiles = cached_jtd_tiles(scenario_name, p, recovery) if model == JTD else cached_tf_tiles(scenario_name)
    render_snapshot(tiles)

    st.markdown("## Along a simulated path")
    with st.spinner("Solving and simulating…"):
        if model == JTD:
            df, profile, default_day = cached_jtd(scenario_name, p, recovery, seed, allow_default)
            layout, panels = JTD_GRID, JTD_PANELS
        else:
            df, profile = cached_tf(scenario_name, seed)
            default_day, layout, panels = None, TF_GRID, TF_PANELS
    if default_day is not None:
        st.error(
            f"The issuer defaulted on day {default_day:.0f} of this path: the stock went to "
            f"{(1 - scenario.jtd_market.jump_size):.0%} of its value and the bond now pays "
            f"recovery ({scenario.jtd_market.recovery:.0%} of par). Try another seed for a surviving path."
        )
    st.plotly_chart(render_animated_figure(df, profile, layout, panels, default_day), width="stretch")

    st.markdown("## TF vs. jump-to-default across the stock price")
    _, jtd_prof, _ = cached_jtd(scenario_name, p, recovery, seed, allow_default)
    # From 10% of the conversion price (the spec's ladder): below that, λ(S)
    # hits its cap and delta spikes, which would flatten the rest of the chart.
    conversion_price = scenario.bond.face_value / scenario.bond.conversion_ratio
    jtd_prof = jtd_prof[jtd_prof["spot"] >= 0.1 * conversion_price]
    spots = jtd_prof["spot"].to_numpy()[::4]
    tf_prof = None if scenario.tf_market is None else cached_tf_profile(scenario_name, tuple(float(s) for s in spots))
    st.plotly_chart(render_comparison(jtd_prof, tf_prof, scenario.jtd_market.spot), width="stretch")
    st.caption(
        "Both models see the same 5y credit spread. TF discounts a fixed cash leg at r + s, so its floor is flat "
        "and its delta is pure equity. JTD's hazard falls as the stock rises: the floor climbs, and the shaded "
        "credit delta adds to the hedge when the bond is busted."
        + (" TF can't model this bond's soft call, so only JTD is shown." if tf_prof is None else "")
    )


if __name__ == "__main__":
    main()
