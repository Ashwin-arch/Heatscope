from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from heatscope.google_map import render_google_heat_map


# ============================================================
# HeatScope — Google Maps Spatial Demo
#
# Separate from heatscope/demo.py
#
# Run:
#   cd /home/ashwin/conference/HeatScope
#   source .venv/bin/activate
#   python -m streamlit run heatscope/google_maps_demo.py
# ============================================================

st.set_page_config(
    page_title="HeatScope | Google Maps",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = ROOT / "demo_data"

GRID_PATH = DEMO_DATA / "heatscope_master_v2.gpkg"
MASTER_PATH = DEMO_DATA / "heatscope_master_v2.csv"
OPERATIONAL_PATH = DEMO_DATA / "heatscope_operational_priority.csv"


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
}

.subtitle {
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(
        135deg,
        rgba(31,41,55,.95),
        rgba(17,24,39,.95)
    );

    border: 1px solid rgba(255,255,255,.08);

    border-radius: 12px;

    padding: 15px;

    text-align: center;

    min-height: 90px;
}

.metric-value {
    font-size: 1.45rem;
    font-weight: 750;
}

.metric-label {
    color: #9ca3af;
    font-size: .78rem;
    margin-top: 4px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_master_csv() -> pd.DataFrame:
    return pd.read_csv(MASTER_PATH)


@st.cache_data(show_spinner=False)
def load_operational() -> pd.DataFrame:
    return pd.read_csv(OPERATIONAL_PATH)


@st.cache_data(show_spinner=False)
def load_grid() -> gpd.GeoDataFrame:
    return gpd.read_file(GRID_PATH)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🌡️ HeatScope — Google Maps Spatial Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Interactive 250 m spatial heat-exposure and vulnerability exploration'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Validate files
# ============================================================

missing = []

for path in [
    GRID_PATH,
    MASTER_PATH,
    OPERATIONAL_PATH,
]:
    if not path.exists():
        missing.append(str(path))

if missing:
    st.error("Required HeatScope files are missing:")

    for path in missing:
        st.code(path)

    st.stop()


# ============================================================
# Load data
# ============================================================

try:
    grid = load_grid()
    master = load_master_csv()
    operational = load_operational()
except Exception as exc:
    st.error(f"Could not load HeatScope spatial data: {exc}")
    st.stop()


# ============================================================
# Basic validation
# ============================================================

if "cell_id" not in grid.columns:
    st.error("Grid GPKG does not contain cell_id.")
    st.stop()

if "cell_id" not in master.columns:
    st.error("Master CSV does not contain cell_id.")
    st.stop()

if "cell_id" not in operational.columns:
    st.error("Operational priority CSV does not contain cell_id.")
    st.stop()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown("## 🗺️ Map Controls")

layer_options = {
    "Observed LST P95": (
        master,
        "lst_celsius_p95",
        "Observed LST P95 (°C)",
    ),

    "Predicted LST P95": (
        operational,
        "predicted_lst_p95",
        "Predicted LST P95 (°C)",
    ),

    "Uncertainty": (
        operational,
        "uncertainty_half_width",
        "Adaptive uncertainty half-width (°C)",
    ),

    "Risk Exposure": (
        operational,
        "risk_exposure_lambda05",
        "Risk exposure (λ = 0.5)",
    ),

    "Structural Vulnerability": (
        operational,
        "structural_vulnerability",
        "Structural vulnerability",
    ),

    "Operational Priority": (
        operational,
        "operational_priority_lambda05",
        "Operational priority",
    ),

    "Heat Vulnerability Index": (
        operational,
        "hvi",
        "Heat Vulnerability Index",
    ),

    "Population Density": (
        master,
        "population_density",
        "Population density",
    ),

    "NDVI": (
        master,
        "ndvi_mean",
        "NDVI",
    ),

    "NDMI": (
        master,
        "ndmi_mean",
        "NDMI",
    ),

    "Building Coverage": (
        master,
        "building_coverage_pct",
        "Building coverage (%)",
    ),
}


layer = st.sidebar.selectbox(
    "Spatial layer",
    list(layer_options.keys()),
)


data_source, value_column, layer_title = layer_options[layer]


# ============================================================
# Additional controls
# ============================================================

st.sidebar.markdown("---")

map_style = st.sidebar.radio(
    "Google Maps base",
    [
        "Roadmap",
        "Satellite",
    ],
    index=0,
)

show_equity = st.sidebar.checkbox(
    "Equity-priority context",
    value=False,
    help="Retains the selected HeatScope layer while highlighting cells flagged for the equity-priority group in the click information.",
)


st.sidebar.markdown("---")

st.sidebar.markdown(
    """
**HeatScope spatial grid**

- 250 m nominal cells
- 21,845 analysis cells
- Bengaluru study area
- Satellite-derived thermal exposure
- Population-aware prioritization
- Spatial ML
- Uncertainty-aware planning
"""
)


# ============================================================
# Prepare map data
# ============================================================

map_data = data_source.copy()

map_data = map_data.drop_duplicates(
    subset=["cell_id"]
)

required = [
    "cell_id",
    value_column,
]

missing_columns = [
    c for c in required
    if c not in map_data.columns
]

if missing_columns:
    st.error(
        f"Required columns missing for {layer}: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# Map
# ============================================================

st.markdown(
    f"### {layer_title}"
)

if map_style == "Satellite":
    st.info(
        "Google Maps satellite mode is selected. "
        "The spatial HeatScope cells remain the same."
    )


# The renderer currently uses Google's default roadmap.
# Satellite switching is handled below through the dedicated
# map component patch after the first successful render.

try:
    render_google_heat_map(
        grid_gdf=grid,
        data_df=map_data,
        value_column=value_column,
        layer_title=layer_title,
        height=720,
    )
except Exception as exc:
    st.error(
        "Google Maps rendering failed."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# Study summary
# ============================================================

st.markdown("### Study summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(grid):,}</div>
            <div class="metric-label">Spatial cells</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    if "area_m2" in master.columns:
        area_km2 = master["area_m2"].sum() / 1_000_000
        area_text = f"{area_km2:,.2f}"
    else:
        area_text = "—"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{area_text}</div>
            <div class="metric-label">Study area (km²)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    if "population" in master.columns:
        population = master["population"].sum()
        population_text = f"{population:,.0f}"
    else:
        population_text = "—"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{population_text}</div>
            <div class="metric-label">Population represented</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    if value_column in map_data.columns:
        values = pd.to_numeric(
            map_data[value_column],
            errors="coerce",
        ).dropna()

        valid_text = f"{len(values):,}"
    else:
        valid_text = "—"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{valid_text}</div>
            <div class="metric-label">Cells with layer data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Selected layer statistics
# ============================================================

st.markdown("### Layer statistics")

values = pd.to_numeric(
    map_data[value_column],
    errors="coerce",
).dropna()

if not values.empty:

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "Minimum",
            f"{values.min():.3f}",
        )

    with s2:
        st.metric(
            "Median",
            f"{values.median():.3f}",
        )

    with s3:
        st.metric(
            "P95",
            f"{values.quantile(.95):.3f}",
        )

    with s4:
        st.metric(
            "Maximum",
            f"{values.max():.3f}",
        )


# ============================================================
# Research interpretation
# ============================================================

st.markdown("### Interpretation")

interpretation = {
    "Observed LST P95":
        "Observed satellite-derived thermal exposure. This is not a clinical heat-health outcome.",

    "Predicted LST P95":
        "Spatial ML estimate of LST P95 using the validated HeatScope spatial model.",

    "Uncertainty":
        "Adaptive spatial cross-validated prediction-interval half-width. It represents predictive uncertainty rather than a formal conformal guarantee.",

    "Risk Exposure":
        "Heat exposure combined with adaptive uncertainty using the HeatScope λ=0.5 formulation.",

    "Structural Vulnerability":
        "Population sensitivity and limited adaptive capacity, independent of the observed LST target.",

    "Operational Priority":
        "HeatScope's population-aware operational prioritization combining robust heat exposure and structural vulnerability.",

    "Heat Vulnerability Index":
        "Analytical vulnerability score constructed from exposure, sensitivity and capacity. It is not ground truth.",

    "Population Density":
        "Population density associated with each HeatScope spatial cell.",

    "NDVI":
        "Normalized Difference Vegetation Index derived from Landsat reflectance.",

    "NDMI":
        "Normalized Difference Moisture Index derived from Landsat reflectance.",

    "Building Coverage":
        "Percentage of each spatial cell covered by mapped building footprints.",
}

st.info(interpretation.get(layer, ""))


# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption(
    "HeatScope research demo • Bengaluru AOI • "
    "250 m spatial analysis grid • "
    "Google Maps visualization • "
    "Study-specific results"
)
