from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from heatscope.google_map import render_google_heat_map

try:
    import geopandas as gpd
except Exception:
    gpd = None


# ============================================================
# HEATSCOPE DEMO
# Save as: /home/ashwin/conference/HeatScope/heatscope/demo.py
#
# Run:
#   cd /home/ashwin/conference/HeatScope
#   source .venv/bin/activate
#   python -m streamlit run heatscope/demo.py
# ============================================================

st.set_page_config(
    page_title="HeatScope | Research Demo",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
MODELS = ROOT / "models"

# These are the validated HeatScope results already produced by the project.
# They are displayed as study-specific results, never as universal claims.
R = {
    "base_mae": 1.2241,
    "base_rmse": 1.6060,
    "base_r2": 0.6286,
    "ctx_mae": 1.1738,
    "ctx_rmse": 1.5478,
    "ctx_r2": 0.6548,
    "ctx_rmse_gain": 3.62,
    "q5_mae": 1.6305,
    "q5_bias": 1.3792,
    "unc_coverage": 0.9500,
    "unc_width": 5.8623,
    "q5_coverage": 0.8863,
    "greedy_benefit": 1616.436,
    "milp_benefit": 2074.491,
    "milp_gain": 28.34,
    "grid_cells": 21845,
    "study_area": 1348.25,
    "population": 7458314,
}

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(255,107,69,.12), transparent 25%),
        radial-gradient(circle at 92% 10%, rgba(74,144,226,.10), transparent 25%),
        linear-gradient(180deg,#070b13 0%,#0a111d 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080d16,#0c1421);
    border-right: 1px solid rgba(255,255,255,.08);
}
.block-container {max-width:1500px;padding-top:1.2rem;}
[data-testid="stMetric"] {
    border:1px solid rgba(255,255,255,.09);
    border-radius:16px;
    background:rgba(255,255,255,.035);
    padding:12px 14px;
}
.hs-card {
    border:1px solid rgba(255,255,255,.09);
    border-radius:18px;
    padding:18px;
    background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.018));
    margin-bottom:14px;
}
.kicker {
    color:#ffd166;
    font-size:.72rem;
    font-weight:800;
    letter-spacing:.13em;
    text-transform:uppercase;
}
.hero {
    border:1px solid rgba(255,255,255,.10);
    border-radius:24px;
    padding:28px 30px;
    background:
      linear-gradient(120deg,rgba(255,107,69,.10),rgba(90,160,230,.05)),
      rgba(255,255,255,.025);
}
.hero-title {font-size:2.6rem;font-weight:850;letter-spacing:-.035em;}
.hero-sub {color:#aeb9ca;font-size:1.05rem;}
.muted {color:#aeb9ca;}
.callout {
    border-left:4px solid #ff6b45;
    padding:12px 15px;
    background:rgba(255,107,69,.07);
    border-radius:10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def find_first(*patterns: str) -> Path | None:
    for root in (RESULTS, DATA, MODELS):
        if not root.exists():
            continue
        for pattern in patterns:
            hits = sorted(root.rglob(pattern))
            if hits:
                return hits[0]
    return None


def ncol(df: pd.DataFrame, names: list[str]) -> str | None:
    norm = {''.join(ch.lower() for ch in str(c) if ch.isalnum()): c for c in df.columns}
    for n in names:
        k = ''.join(ch.lower() for ch in n if ch.isalnum())
        if k in norm:
            return norm[k]
    for c in df.columns:
        cc = ''.join(ch.lower() for ch in str(c) if ch.isalnum())
        for n in names:
            nn = ''.join(ch.lower() for ch in n if ch.isalnum())
            if nn and (nn in cc or cc in nn):
                return c
    return None


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_geo(path: str):
    if gpd is None:
        return None
    return gpd.read_file(path)


def load_pareto() -> pd.DataFrame | None:
    p = find_first("pareto_frontier.csv")
    if p:
        return read_csv(str(p))
    return None


def load_master() -> tuple[pd.DataFrame | None, Path | None]:
    p = find_first("heatscope_master_v2.csv", "*master*v2*.csv", "*heatscope*master*.csv")
    if p:
        return read_csv(str(p)), p

    gp = find_first("heatscope_master_v2.gpkg", "*master*v2*.gpkg", "*heatscope*master*.gpkg")
    if gp and gpd is not None:
        gdf = read_geo(str(gp))
        if gdf is not None:
            return gdf, gp
    return None, None


PARETO = load_pareto()
MASTER, MASTER_PATH = load_master()

# Google Maps polygon grid.
# The GPKG contains the validated 21,845-cell HeatScope spatial grid.
GRID_GDF = None
GRID_PATH = ROOT / "results" / "heatscope_master_v2.gpkg"

if GRID_PATH.exists() and gpd is not None:
    try:
        GRID_GDF = read_geo(str(GRID_PATH))
    except Exception:
        GRID_GDF = None

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
<div class="hero">
  <div class="kicker">Live research demonstration</div>
  <div class="hero-title">HEATSCOPE</div>
  <div class="hero-sub">
    Spatial heat intelligence for Bengaluru — prediction, explanation, uncertainty,
    prioritization, and equity-aware mitigation planning.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## HeatScope")
page = st.sidebar.radio(
    "Demo section",
    [
        "1 • Executive View",
        "2 • Heat Intelligence",
        "3 • Model Performance",
        "4 • Explainability",
        "5 • Uncertainty",
        "6 • Intervention Planner",
        "7 • Pareto / Equity",
        "8 • Viva Defense",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Connected project")
st.sidebar.code(str(ROOT), language="text")
st.sidebar.success("Project mode: LIVE")
if MASTER_PATH:
    st.sidebar.caption(f"Spatial data: {MASTER_PATH.name}")
else:
    st.sidebar.caption("Spatial master file not detected; planning results remain available.")


# ============================================================
# 1. Executive View
# ============================================================
if page == "1 • Executive View":
    cols = st.columns(4)
    cols[0].metric("Spatial R²", f"{R['ctx_r2']:.3f}")
    cols[1].metric("MAE", f"{R['ctx_mae']:.2f} °C")
    cols[2].metric("Empirical coverage", f"{100*R['unc_coverage']:.1f}%")
    cols[3].metric("MILP vs Greedy", f"+{R['milp_gain']:.2f}%")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            """
<div class="hs-card">
<div class="kicker">What HeatScope does</div>
<h3>From satellite observation to an auditable mitigation decision</h3>
<p class="muted">
Real Landsat thermal/reflectance data are aggregated to 250 m cells and combined
with WorldPop population and urban-form indicators. A spatially validated ML model
predicts high-temperature exposure, SHAP explains model behavior, adaptive
uncertainty enters operational priority, and an MILP allocates interventions under
budget and equity constraints.
</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="hs-card">
<div class="kicker">Study footprint</div>
<h2>21,845</h2>
250 m cells<br><br>
<h2>1,348.25 km²</h2>
Bengaluru study area<br><br>
<h2>7.46M</h2>
WorldPop population represented
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### Research evidence chain")
    flow = pd.DataFrame(
        {
            "Stage": [
                "Observe",
                "Preprocess",
                "Predict",
                "Explain",
                "Quantify uncertainty",
                "Prioritize",
                "Optimize",
            ],
            "Output": [
                "Landsat + population + urban form",
                "QA mask + spectral indices",
                "Spatial-context HGB",
                "SHAP feature attribution",
                "Adaptive spatial intervals",
                "Exposure + structural vulnerability",
                "MILP + budgets + equity",
            ],
        }
    )
    st.dataframe(flow, use_container_width=True, hide_index=True)

    st.info(
        "Scientific boundary: HeatScope predicts satellite-derived LST P95 (thermal exposure), "
        "not clinical heat-health outcomes. Intervention benefit is a normalized scenario score."
    )


# ============================================================
# 2. Heat Intelligence
# ============================================================
elif page == "2 • Heat Intelligence":
    st.markdown("### Heat intelligence map")

    if MASTER is None:
        st.warning(
            "The live spatial master file was not found in this environment. "
            "The application is still connected to the saved planning frontier. "
            "Run the HeatScope pipeline locally and this map will automatically populate."
        )
        st.stop()

    df = MASTER.copy()

    lat = ncol(df, ["centroid_lat", "latitude", "lat"])
    lon = ncol(df, ["centroid_lon", "longitude", "lon"])

    layers = {
        "Observed LST P95": ["lst_celsius_p95", "lst_p95", "observed_lst_p95"],
        "Predicted LST P95": ["predicted_lst_p95", "prediction_lst_p95", "lst_p95_pred"],
        "Operational Priority": ["operational_priority", "priority_score", "priority"],
        "Structural Vulnerability": ["structural_vulnerability", "structural_vuln"],
        "Population Density": ["population_density", "pop_density"],
        "NDVI": ["ndvi_mean", "ndvi"],
        "NDMI": ["ndmi_mean", "ndmi"],
        "Building Coverage": ["building_coverage_pct", "building_coverage"],
    }

    available = {}
    for label, aliases in layers.items():
        c = ncol(df, aliases)
        if c:
            available[label] = c

    if not lat or not lon:
        st.error("Centroid coordinates are not present in the detected master file.")
        st.dataframe(df.head(20), use_container_width=True)
        st.stop()

    layer = st.selectbox("Select spatial layer", list(available))
    c = available[layer]

    max_points = st.slider("Map points", 1000, min(len(df), 21845), min(len(df), 9000), 500)

    mapdf = df[[lat, lon, c]].copy()
    mapdf[c] = pd.to_numeric(mapdf[c], errors="coerce")
    mapdf = mapdf.dropna()
    if len(mapdf) > max_points:
        mapdf = mapdf.sample(max_points, random_state=42)

    # ========================================================
    # LOCAL SPATIAL MAP
    # No OpenStreetMap / MapLibre / internet tiles required.
    # This makes the conference demo work offline.
    # ========================================================

    fig = px.scatter(
        mapdf,
        x=lon,
        y=lat,
        color=c,
        color_continuous_scale="Turbo",
        height=650,
        hover_data={
            lon: ":.5f",
            lat: ":.5f",
            c: ":.3f",
        },
        labels={
            lon: "Longitude",
            lat: "Latitude",
            c: layer,
        },
    )

    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.78,
        )
    )

    fig.update_xaxes(
        title="Longitude",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
    )

    fig.update_yaxes(
        title="Latitude",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        title=f"HeatScope • {layer}",
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,28,0.75)",
        font_color="white",
        coloraxis_colorbar=dict(
            title=layer,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    st.markdown(
        '<div class="callout"><b>Demo line:</b> “The map is not just a hotspot picture; '
        'the same spatial cells feed the prediction, priority, and optimization stages.”</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 3. Model Performance
# ============================================================
elif page == "3 • Model Performance":
    st.markdown("### Model capability benchmark")

    a, b = st.columns(2)
    with a:
        st.markdown("#### Base 5-feature HGB")
        st.metric("MAE", f"{R['base_mae']:.4f} °C")
        st.metric("RMSE", f"{R['base_rmse']:.4f} °C")
        st.metric("R²", f"{R['base_r2']:.4f}")

    with b:
        st.markdown("#### Context_9 HGB")
        st.metric("MAE", f"{R['ctx_mae']:.4f} °C")
        st.metric("RMSE", f"{R['ctx_rmse']:.4f} °C")
        st.metric("R²", f"{R['ctx_r2']:.4f}")

    bench = pd.DataFrame(
        {
            "Model": ["Base 5", "Context_9"],
            "MAE (°C)": [R["base_mae"], R["ctx_mae"]],
            "RMSE (°C)": [R["base_rmse"], R["ctx_rmse"]],
            "R²": [R["base_r2"], R["ctx_r2"]],
        }
    )
    st.dataframe(bench, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base 5 R²", x=["Base 5"], y=[R["base_r2"]]))
    fig.add_trace(go.Bar(name="Context_9 R²", x=["Context_9"], y=[R["ctx_r2"]]))
    fig.update_layout(
        title="Spatial-context improvement",
        yaxis_title="R²",
        yaxis_range=[0, 0.75],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        barmode="group",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"Adding four queen-neighbor mean features improved RMSE by "
        f"{R['ctx_rmse_gain']:.2f}% and increased R² from {R['base_r2']:.4f} to {R['ctx_r2']:.4f}."
    )

    st.caption(
        "Validation used 5-fold spatial GroupKFold. The target is Landsat LST P95, "
        "so these metrics describe thermal-exposure prediction, not health-risk prediction."
    )


# ============================================================
# 4. Explainability
# ============================================================
elif page == "4 • Explainability":
    st.markdown("### Why did the model make this prediction?")

    shap = pd.DataFrame(
        {
            "Feature": [
                "NDMI (focal)",
                "Population density (neighbor)",
                "NDMI (neighbor)",
                "NDVI (focal)",
                "Building coverage (neighbor)",
                "NDVI (neighbor)",
                "Population density (focal)",
                "Building coverage (focal)",
                "Building count",
            ],
            "Mean absolute SHAP (%)": [34.04, 16.04, 15.34, 10.03, 7.67, 6.38, 4.51, 3.03, 2.96],
        }
    ).sort_values("Mean absolute SHAP (%)")

    fig = px.bar(
        shap,
        x="Mean absolute SHAP (%)",
        y="Feature",
        orientation="h",
        title="Context_9 SHAP profile",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
<div class="hs-card">
<div class="kicker">What we can say</div>
<b>Focal + neighborhood NDMI ≈ 49.4%</b> of mean absolute SHAP importance.
<br><br>
Neighbor variables collectively contribute ≈ <b>45.4%</b> of mean absolute SHAP importance.
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div class="hs-card">
<div class="kicker">What we cannot say</div>
SHAP explains the trained model's behavior. It does <b>not</b> prove that changing one
feature alone will cause the same temperature response in the real city.
</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# 5. Uncertainty
# ============================================================
elif page == "5 • Uncertainty":
    st.markdown("### Adaptive spatial prediction uncertainty")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall coverage", f"{100*R['unc_coverage']:.1f}%")
    c2.metric("Mean interval width", f"{R['unc_width']:.2f} °C")
    c3.metric("Hot-tail coverage", f"{100*R['q5_coverage']:.2f}%")
    c4.metric("Hot-tail MAE", f"{R['q5_mae']:.2f} °C")

    q = pd.DataFrame(
        {
            "Heat quantile": ["Q1", "Q2", "Q3", "Q4", "Q5 (hottest)"],
            "Coverage (%)": [93.68, 97.52, 98.14, 97.02, 88.63],
        }
    )
    fig = px.bar(
        q,
        x="Heat quantile",
        y="Coverage (%)",
        text="Coverage (%)",
        range_y=[80, 100],
        title="Empirical coverage by heat quantile",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "The hottest 20% has lower coverage (88.63%). This is an explicit reliability limitation. "
        "The intervals are empirical spatial cross-validation intervals, not a formal conformal guarantee."
    )

    st.markdown(
        """
<div class="hs-card">
<div class="kicker">Decision integration</div>
Primary operational setting:
<b>risk exposure = predicted LST P95 + 0.5 × uncertainty</b>.
<br><br>
This lets the priority score reflect both the predicted heat and how uncertain that prediction is.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# 6. Intervention Planner
# ============================================================
elif page == "6 • Intervention Planner":
    st.markdown("### Constrained intervention planning")

    budget = st.select_slider(
        "Budget",
        options=[25, 50, 75, 100],
        value=75,
        format_func=lambda x: f"{x}% budget",
    )
    equity = st.select_slider(
        "Minimum equity target",
        options=[50, 60, 70, 80],
        value=80,
        format_func=lambda x: f"{x}% equity",
    )

    if PARETO is None:
        st.error("results/pareto_frontier.csv was not found.")
        st.stop()

    pf = PARETO.copy()
    bc = ncol(pf, ["budget_fraction", "budget"])
    ec = ncol(pf, ["equity_target"])
    ben = ncol(pf, ["benefit"])
    es = ncol(pf, ["equity_share"])
    bl = ncol(pf, ["benefit_loss_fraction"])
    cells = ncol(pf, ["selected_cells", "n_selected"])

    pf["_budget_pct"] = pd.to_numeric(pf[bc], errors="coerce") * 100
    pf["_equity_target_pct"] = pd.to_numeric(pf[ec], errors="coerce") * 100 if ec else np.nan

    sub = pf[np.isclose(pf["_budget_pct"], budget, atol=.01)]
    exact = sub[np.isclose(sub["_equity_target_pct"], equity, atol=.01)] if len(sub) else sub

    # The Pareto file contains NaN for unconstrained and explicit equity targets.
    if len(exact) == 0:
        st.warning("No exact scenario row found for this combination.")
    else:
        row = exact.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Budget", f"{budget}%")
        c2.metric("Equity achieved", f"{100*float(row[es]):.2f}%")
        c3.metric("Modeled benefit", f"{float(row[ben]):,.2f}")
        if cells and pd.notna(row[cells]):
            c4.metric("Selected cells", f"{int(float(row[cells])):,}")
        else:
            c4.metric("Benefit loss", f"{100*float(row[bl]):.2f}%")

        if float(row[bl]) > 0:
            st.warning(
                f"This equity target creates a {100*float(row[bl]):.2f}% modeled benefit loss "
                f"relative to the unconstrained solution at the same budget."
            )
        else:
            st.success(
                "This equity target is non-binding at this budget under the current scenario definition."
            )

    st.markdown("#### Four experimental intervention classes")
    cards = st.columns(4)
    items = [
        ("🌳 Trees", "Low NDVI + high priority / structural need", "Cost = 1.0"),
        ("⛱️ Shade", "Dense locations + high priority", "Cost = 1.4"),
        ("🏠 Cool roofs", "High built coverage + high priority", "Cost = 2.0"),
        ("💧 Cooling", "Low NDMI + dense + high priority", "Cost = 2.5"),
    ]
    for c, (t, d, cost) in zip(cards, items):
        with c:
            st.markdown(
                f"""
<div class="hs-card">
<div class="kicker">{cost}</div>
<b>{t}</b>
<p class="muted">{d}</p>
</div>
""",
                unsafe_allow_html=True,
            )

    st.caption(
        "Costs, effects and capacities are experimental planning parameters. "
        "Benefit is a normalized scenario score, not a guaranteed cooling amount."
    )


# ============================================================
# 7. Pareto / Equity
# ============================================================
elif page == "7 • Pareto / Equity":
    st.markdown("### Benefit–equity Pareto view")

    if PARETO is None:
        st.error("results/pareto_frontier.csv was not found.")
        st.stop()

    pf = PARETO.copy()
    bc = ncol(pf, ["budget_fraction", "budget"])
    ec = ncol(pf, ["equity_target"])
    ben = ncol(pf, ["benefit"])
    es = ncol(pf, ["equity_share"])
    bl = ncol(pf, ["benefit_loss_fraction"])

    pf["_budget_pct"] = pd.to_numeric(pf[bc], errors="coerce") * 100
    pf["_equity_pct"] = pd.to_numeric(pf[ec], errors="coerce") * 100 if ec else np.nan

    fig = go.Figure()
    series = [
        (math.nan, "Unconstrained"),
        (0.70, "70% equity"),
        (0.80, "80% equity"),
    ]
    for target, label in series:
        if math.isnan(target):
            sub = pf[pf[ec].isna()].sort_values("_budget_pct")
        else:
            sub = pf[np.isclose(pd.to_numeric(pf[ec], errors="coerce"), target, atol=.001)].sort_values("_budget_pct")
        if len(sub):
            fig.add_trace(
                go.Scatter(
                    x=sub["_budget_pct"],
                    y=sub[ben],
                    mode="lines+markers",
                    name=label,
                    hovertemplate="Budget=%{x:.0f}%<br>Benefit=%{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Modeled benefit vs available budget",
        xaxis_title="Budget (%)",
        yaxis_title="Modeled intervention benefit",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
<div class="hs-card">
<div class="kicker">Key findings from the saved frontier</div>
<b>75% budget + 80% equity:</b> 80.005% equity, benefit 1750.75,
benefit loss 6.58%.<br>
<b>100% budget + 70% equity:</b> 70.000% equity, benefit 1996.90,
benefit loss 3.74%.<br>
<b>100% budget + 80% equity:</b> 80.001% equity, benefit 1750.82,
benefit loss 15.60%.
</div>
""",
        unsafe_allow_html=True,
    )

    st.dataframe(PARETO, use_container_width=True, hide_index=True)


# ============================================================
# 8. Viva Defense
# ============================================================
else:
    st.markdown("### Viva defense — answers built into the demo")

    answers = [
        ("What exactly are you predicting?",
         "Landsat-derived LST P95 for each 250 m cell. It represents thermal exposure, not a clinical health-risk label."),
        ("How accurate is it?",
         "The final spatial-context HGB achieved MAE 1.174°C, RMSE 1.548°C and R² 0.655 under 5-fold spatial block cross-validation."),
        ("Why spatial validation?",
         "Nearby cells are spatially correlated. GroupKFold by spatial blocks reduces leakage and tests performance on geographically separated cells."),
        ("What did spatial context add?",
         "Neighbor means improved RMSE by 3.62% and increased R² from 0.629 to 0.655."),
        ("Why SHAP?",
         "To explain which features drive the trained model's predictions and expose nonlinear behavior. SHAP is not causal evidence."),
        ("What does the optimizer optimize?",
         "A normalized modeled intervention benefit under budget, action-capacity and minimum-equity constraints."),
        ("Why MILP?",
         "The intervention decisions are discrete and the defined budget/equity constraints are linear, making MILP suitable for exact optimization of the stated scenario."),
        ("How much better is MILP than greedy?",
         "Under capacity-matched conditions, MILP produced 28.34% higher modeled benefit than the greedy baseline."),
        ("Is the 95% uncertainty coverage guaranteed?",
         "No. It is empirical spatial-CV coverage. Overall coverage is 95.0%, while the hottest quintile is 88.63%."),
        ("Can you claim HeatScope is universally better than current systems?",
         "No. We claim an integrated, experimentally validated framework for this Bengaluru study and a measured advantage over the matched greedy optimization baseline."),
    ]

    for q, a in answers:
        with st.expander(q):
            st.write(a)

    st.markdown(
        """
<div class="callout">
<b>Recommended 30-second closing:</b><br>
“HeatScope converts real Earth-observation data into a spatially validated heat-exposure
model, explains its predictions with SHAP, quantifies uncertainty, identifies priority
areas, and then allocates multiple mitigation actions under explicit budget and equity
constraints. The interface makes every stage inspectable during the demonstration.”
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "HeatScope • Bengaluru • Study-specific research prototype. "
    "No universal superiority, health-risk, or guaranteed intervention-effect claim is made."
)
