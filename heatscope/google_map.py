from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit.components.v1 as components


def load_google_maps_key() -> str:
    """
    Load Google Maps API key.

    Priority:
    1. Streamlit Secrets (deployed app)
    2. .env.local (local development)
    3. Environment variable
    """

    # Streamlit Cloud / Streamlit Secrets
    try:
        import streamlit as st

        key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")

        if key:
            return str(key).strip()
    except Exception:
        pass

    # Local .env.local
    p = Path(__file__).resolve().parents[1] / ".env.local"

    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[7:].strip()

            if line.startswith("GOOGLE_MAPS_API_KEY="):
                value = line.split("=", 1)[1].strip()
                return value.strip("\"'")

    # Environment variable fallback
    import os

    return os.getenv("GOOGLE_MAPS_API_KEY", "").strip()


def _color(value, vmin, vmax):
    """
    Return a Google Maps-compatible RGB color.

    Blue -> cyan -> yellow -> orange -> red.
    """

    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)

    t = max(0.0, min(1.0, float(t)))

    stops = [
        (33, 102, 172),
        (103, 169, 207),
        (247, 247, 247),
        (239, 138, 98),
        (178, 24, 43),
    ]

    x = t * (len(stops) - 1)

    i = min(int(x), len(stops) - 2)
    f = x - i

    a = stops[i]
    b = stops[i + 1]

    r = round(a[0] + f * (b[0] - a[0]))
    g = round(a[1] + f * (b[1] - a[1]))
    bval = round(a[2] + f * (b[2] - a[2]))

    return f"rgb({r},{g},{bval})"


def render_google_heat_map(
    grid_gdf: gpd.GeoDataFrame,
    data_df: pd.DataFrame,
    value_column: str,
    layer_title: str,
    height: int = 700,
):
    """
    Render HeatScope polygons on Google Maps.

    grid_gdf:
        Actual HeatScope 250m polygon grid.

    data_df:
        Operational HeatScope values keyed by cell_id.

    value_column:
        Numeric column used to color cells.
    """

    if not isinstance(grid_gdf, gpd.GeoDataFrame):
        raise TypeError("grid_gdf must be a GeoDataFrame.")

    if "cell_id" not in grid_gdf.columns:
        raise ValueError("Grid GeoDataFrame must contain cell_id.")

    if "cell_id" not in data_df.columns:
        raise ValueError("Data dataframe must contain cell_id.")

    if value_column not in data_df.columns:
        raise ValueError(
            f"Map variable '{value_column}' was not found."
        )

    api_key = load_google_maps_key()

    if not api_key:
        st_html = """
        <div style="
            padding:24px;
            border-radius:12px;
            background:#3b0d0d;
            color:#fecaca;
            font-family:Arial;
        ">
            <b>Google Maps API key not configured.</b><br><br>
            Add GOOGLE_MAPS_API_KEY to .env.local.
        </div>
        """

        components.html(st_html, height=150)
        return

    # Convert grid to WGS84 for Google Maps.
    grid = grid_gdf.to_crs(4326).copy()

    # Merge decision variables onto the real polygon grid.
    # The GPKG already contains several base variables such as
    # lst_celsius_p95, NDVI, NDMI, population, etc. Remove
    # overlapping non-key columns from the grid before merging
    # so pandas does not create _x/_y suffixes.
    keep_cols = [
        "cell_id",
        value_column,
    ]

    # Useful information shown when clicking a cell.
    optional = [
        "population",
        "population_density",
        "ndvi_mean",
        "ndmi_mean",
        "building_coverage_pct",
        "lst_celsius_p95",
        "predicted_lst_p95",
        "uncertainty_half_width",
        "risk_exposure_lambda05",
        "structural_vulnerability",
        "operational_priority_lambda05",
        "equity_priority",
        "hvi",
        "hvi_category",
    ]

    for c in optional:
        if c in data_df.columns and c not in keep_cols:
            keep_cols.append(c)

    values = data_df[keep_cols].copy()

    # Ensure one record per cell.
    values = values.drop_duplicates("cell_id")

    # Avoid pandas merge suffixes when the polygon grid and the
    # decision-data table contain the same analytical columns.
    # The decision-data table is the authoritative source for
    # the selected layer and operational variables.
    overlap = [
        c for c in values.columns
        if c != "cell_id" and c in grid.columns
    ]

    if overlap:
        grid = grid.drop(columns=overlap)

    merged = grid.merge(
        values,
        on="cell_id",
        how="left",
    )

    merged[value_column] = pd.to_numeric(
        merged[value_column],
        errors="coerce",
    )

    merged = merged.dropna(
        subset=[value_column]
    ).copy()

    if merged.empty:
        raise ValueError(
            f"No valid spatial values found for '{value_column}'."
        )

    # Convert polygons to GeoJSON.
    # Build the column list explicitly and remove duplicates.
    # GeoPandas refuses to serialize a GeoDataFrame containing
    # duplicate column names.
    geo_columns = ["cell_id"]

    for col in [value_column] + optional:
        if col in merged.columns and col not in geo_columns:
            geo_columns.append(col)

    geo_columns.append("geometry")

    merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    geojson = json.loads(
        merged[geo_columns].to_json()
    )

    values_only = merged[value_column].astype(float)

    vmin = float(values_only.min())
    vmax = float(values_only.max())

    bounds = merged.total_bounds

    west, south, east, north = [
        float(x) for x in bounds
    ]

    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    geojson_text = json.dumps(
        geojson,
        separators=(",", ":"),
    )

    # HTML/JS-safe title.
    safe_title = (
        str(layer_title)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>

<meta charset="utf-8">

<style>

html, body {{
    width:100%;
    height:100%;
    margin:0;
    padding:0;
    overflow:hidden;
    font-family:Arial, sans-serif;
}}

#map {{
    width:100%;
    height:100%;
}}

#title {{
    position:absolute;
    z-index:10;
    top:14px;
    left:14px;

    padding:10px 14px;

    background:rgba(17,24,39,.95);
    color:white;

    border-radius:9px;

    font-size:14px;
    font-weight:bold;

    box-shadow:
        0 3px 12px rgba(0,0,0,.30);
}}

#legend {{
    position:absolute;
    z-index:10;

    left:14px;
    bottom:25px;

    width:230px;

    padding:13px 15px;

    background:rgba(17,24,39,.95);
    color:white;

    border-radius:10px;

    box-shadow:
        0 3px 12px rgba(0,0,0,.35);
}}

.legend-title {{
    font-weight:bold;
    font-size:13px;
    margin-bottom:8px;
}}

.gradient {{
    height:12px;

    border-radius:6px;

    background:
        linear-gradient(
            90deg,
            #2166ac,
            #67a9cf,
            #f7f7f7,
            #ef8a62,
            #b2182b
        );
}}

.legend-labels {{
    display:flex;
    justify-content:space-between;

    margin-top:5px;

    font-size:11px;
    color:#d1d5db;
}}

</style>

</head>

<body>

<div id="title">
    HeatScope • {safe_title}
</div>

<div id="map"></div>

<div id="legend">

    <div class="legend-title">
        {safe_title}
    </div>

    <div class="gradient"></div>

    <div class="legend-labels">
        <span>{vmin:.2f}</span>
        <span>{vmax:.2f}</span>
    </div>

</div>

<script>

const HEATSCOPE_GEOJSON = {geojson_text};

const MIN_VALUE = {vmin};
const MAX_VALUE = {vmax};

function clamp(x, min, max) {{
    return Math.max(min, Math.min(max, x));
}}

function heatColor(value) {{

    const t = clamp(
        (Number(value) - MIN_VALUE) /
        Math.max(MAX_VALUE - MIN_VALUE, 0.000001),
        0,
        1
    );

    const stops = [
        [33,102,172],
        [103,169,207],
        [247,247,247],
        [239,138,98],
        [178,24,43]
    ];

    const scaled = t * 4;

    const i = Math.min(
        Math.floor(scaled),
        3
    );

    const f = scaled - i;

    const a = stops[i];
    const b = stops[i + 1];

    const r = Math.round(
        a[0] + f * (b[0] - a[0])
    );

    const g = Math.round(
        a[1] + f * (b[1] - a[1])
    );

    const bl = Math.round(
        a[2] + f * (b[2] - a[2])
    );

    return `rgb(${{r}},${{g}},${{bl}})`;
}}

function formatValue(value) {{

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {{
        return "—";
    }}

    if (typeof value === "number") {{
        return value.toFixed(3);
    }}

    return String(value);
}}

function initMap() {{

    const map = new google.maps.Map(
        document.getElementById("map"),
        {{

            center: {{
                lat: {center_lat},
                lng: {center_lon}
            }},

            zoom: 11,

            mapTypeId: "roadmap",

            fullscreenControl: true,

            streetViewControl: false,

            mapTypeControl: true,

            zoomControl: true,

            gestureHandling: "greedy"

        }}
    );

    const layer = new google.maps.Data();

    layer.addGeoJson(
        HEATSCOPE_GEOJSON
    );

    layer.setStyle(
        function(feature) {{

            const value =
                feature.getProperty(
                    "{value_column}"
                );

            return {{

                fillColor:
                    heatColor(value),

                fillOpacity:0.67,

                strokeColor:"#ffffff",

                strokeOpacity:0.22,

                strokeWeight:0.35

            }};

        }}
    );

    layer.addListener(
        "click",
        function(event) {{

            let html = `
                <div style="
                    min-width:280px;
                    color:#111827;
                    font-family:Arial;
                ">

                <h3 style="
                    margin:0 0 10px 0;
                ">
                    HeatScope Cell
                </h3>
            `;

            const fields = [
                ["Cell ID","cell_id"],
                ["Observed LST P95","lst_celsius_p95"],
                ["Predicted LST P95","predicted_lst_p95"],
                ["Uncertainty","uncertainty_half_width"],
                ["Population","population"],
                ["Population Density","population_density"],
                ["NDVI","ndvi_mean"],
                ["NDMI","ndmi_mean"],
                ["Building Coverage","building_coverage_pct"],
                ["Structural Vulnerability","structural_vulnerability"],
                ["Risk Exposure","risk_exposure_lambda05"],
                ["Operational Priority","operational_priority_lambda05"],
                ["Equity Priority","equity_priority"],
                ["HVI","hvi"],
                ["HVI Category","hvi_category"]
            ];

            fields.forEach(
                function(pair) {{

                    const label = pair[0];
                    const key = pair[1];

                    const value =
                        event.feature.getProperty(
                            key
                        );

                    if (
                        value !== undefined &&
                        value !== null
                    ) {{

                        html += `
                            <div style="
                                display:flex;
                                justify-content:space-between;
                                gap:15px;
                                padding:5px 0;
                                border-bottom:
                                    1px solid #e5e7eb;
                            ">

                                <span>
                                    <b>${{label}}</b>
                                </span>

                                <span>
                                    ${{formatValue(value)}}
                                </span>

                            </div>
                        `;
                    }}

                }}
            );

            html += "</div>";

            const info =
                new google.maps.InfoWindow({{
                    content:html,
                    position:event.latLng
                }});

            info.open({{
                map:map
            }});

        }}
    );

    layer.setMap(map);

    const bounds =
        new google.maps.LatLngBounds(
            {{
                lat:{south},
                lng:{west}
            }},
            {{
                lat:{north},
                lng:{east}
            }}
        );

    map.fitBounds(bounds);
}}

window.initMap = initMap;

</script>

<script
    src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap&v=weekly"
    async
    defer>
</script>

</body>
</html>
"""

    components.html(
        html,
        height=height,
        scrolling=False,
    )
