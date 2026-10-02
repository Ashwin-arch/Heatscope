import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = (
    RESULTS / "heatscope_operational_priority.gpkg"
)

OUTPUT_GPKG = (
    RESULTS / "heatscope_operational_interventions.gpkg"
)

OUTPUT_CSV = (
    RESULTS / "heatscope_operational_interventions.csv"
)


INTERVENTIONS = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


# Same scenario multipliers as previous optimizer.
# These remain scenario assumptions, NOT measured effects.
COST = {
    "tree": 1.0,
    "shade": 1.4,
    "cool_roof": 2.0,
    "cooling": 2.5,
}

EFFECT = {
    "tree": 1.0,
    "shade": 1.2,
    "cool_roof": 1.1,
    "cooling": 1.3,
}


print("=" * 70)
print("HeatScope Operational Intervention Benefits")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="operational_priority",
)

gdf = gdf[
    gdf["predicted_lst_p95"].notna()
].copy()

gdf = gdf.reset_index(
    drop=True
)


print(
    "\nCells:",
    f"{len(gdf):,}"
)


# =========================================================
# NORMALIZATION
# =========================================================

def robust_norm(series):

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    lo = x.quantile(0.01)
    hi = x.quantile(0.99)

    if hi <= lo:

        return pd.Series(
            0.5,
            index=series.index,
        )

    x = x.clip(
        lo,
        hi,
    )

    return (
        (x - lo)
        / (hi - lo)
    )


# =========================================================
# AVAILABLE-INFERENCE VARIABLES
# =========================================================

risk = robust_norm(
    gdf[
        "risk_exposure_lambda05"
    ]
)

structural = gdf[
    "structural_vulnerability"
].clip(
    0,
    1,
)

population = robust_norm(
    np.log1p(
        gdf["population"]
    )
)

population_density = robust_norm(
    np.log1p(
        gdf["population_density"]
    )
)

ndvi = robust_norm(
    gdf["ndvi_mean"]
)

ndmi = robust_norm(
    gdf["ndmi_mean"]
)

building = robust_norm(
    gdf["building_coverage_pct"]
)

neighbor_ndvi = robust_norm(
    gdf["ndvi_mean_nbr_mean"]
)

neighbor_ndmi = robust_norm(
    gdf["ndmi_mean_nbr_mean"]
)

neighbor_population = robust_norm(
    np.log1p(
        gdf[
            "population_density_nbr_mean"
        ]
    )
)

neighbor_building = robust_norm(
    gdf[
        "building_coverage_pct_nbr_mean"
    ]
)


# =========================================================
# INTERVENTION SUITABILITY
# =========================================================
#
# IMPORTANT:
# No observed LST-P95 or observed HVI is used here.
#
# Only variables available before deployment are used.
# =========================================================


# ---------------------------------------------------------
# TREE / GREENING
# ---------------------------------------------------------

tree_suitability = (
    0.30 * (1 - ndvi)
    +
    0.15 * (1 - ndmi)
    +
    0.20 * (1 - neighbor_ndvi)
    +
    0.10 * (1 - neighbor_ndmi)
    +
    0.15 * risk
    +
    0.10 * structural
)


# ---------------------------------------------------------
# SHADE
# ---------------------------------------------------------

shade_suitability = (
    0.25 * population_density
    +
    0.15 * neighbor_population
    +
    0.15 * (1 - ndvi)
    +
    0.10 * (1 - neighbor_ndvi)
    +
    0.20 * risk
    +
    0.15 * structural
)


# ---------------------------------------------------------
# COOL ROOF
# ---------------------------------------------------------

cool_roof_suitability = (
    0.30 * building
    +
    0.20 * neighbor_building
    +
    0.20 * risk
    +
    0.15 * structural
    +
    0.15 * population_density
)


# ---------------------------------------------------------
# COOLING / WATER
# ---------------------------------------------------------

cooling_suitability = (
    0.25 * (1 - ndmi)
    +
    0.15 * (1 - neighbor_ndmi)
    +
    0.20 * population
    +
    0.15 * neighbor_population
    +
    0.15 * risk
    +
    0.10 * structural
)


suitability = {
    "tree": tree_suitability.clip(0, 1),
    "shade": shade_suitability.clip(0, 1),
    "cool_roof": cool_roof_suitability.clip(0, 1),
    "cooling": cooling_suitability.clip(0, 1),
}


# =========================================================
# OPERATIONAL BENEFIT
# =========================================================
#
# Population-weighted relative scenario benefit.
#
# Again: this is NOT an empirical degree-C reduction.
# =========================================================

for intervention in INTERVENTIONS:

    gdf[
        f"{intervention}_operational_suitability"
    ] = suitability[
        intervention
    ]

    gdf[
        f"{intervention}_operational_benefit"
    ] = (
        suitability[
            intervention
        ]
        *
        population
        *
        structural
        *
        EFFECT[
            intervention
        ]
    )


# =========================================================
# BEST OPTION
# =========================================================

benefit_columns = [
    f"{i}_operational_benefit"
    for i in INTERVENTIONS
]

benefit_matrix = gdf[
    benefit_columns
]


gdf[
    "best_operational_intervention"
] = (
    benefit_matrix
    .idxmax(axis=1)
    .str.replace(
        "_operational_benefit",
        "",
        regex=False,
    )
)


gdf[
    "best_operational_benefit"
] = (
    benefit_matrix.max(axis=1)
)


gdf[
    "operational_benefit_percentile"
] = (
    gdf[
        "best_operational_benefit"
    ].rank(
        pct=True,
        method="average",
    )
)


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "OPERATIONAL SUITABILITY"
)

print(
    "=" * 70
)


for intervention in INTERVENTIONS:

    col = (
        f"{intervention}_operational_suitability"
    )

    print(
        f"{intervention:12s}"
        f" median={gdf[col].median():.4f}"
        f" P90={gdf[col].quantile(.90):.4f}"
        f" max={gdf[col].max():.4f}"
    )


print(
    "\n" + "=" * 70
)

print(
    "BEST OPERATIONAL INTERVENTION"
)

print(
    "=" * 70
)

print(
    gdf[
        "best_operational_intervention"
    ].value_counts()
)


# =========================================================
# HIGH OPERATIONAL PRIORITY
# =========================================================

top10 = gdf[
    gdf[
        "operational_benefit_percentile"
    ] >= 0.90
]


print(
    "\nTop 10% operational-benefit cells:",
    f"{len(top10):,}"
)

print(
    "Population:",
    f"{top10['population'].sum():,.0f}"
)


print(
    "\nIntervention mix in top 10%:"
)

print(
    top10[
        "best_operational_intervention"
    ].value_counts()
)


# =========================================================
# TOP CELLS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 10 OPERATIONAL INTERVENTION CELLS"
)

print(
    "=" * 70
)


top = gdf.nlargest(
    10,
    "best_operational_benefit",
)


print(
    top[
        [
            "cell_id",
            "longitude",
            "latitude",
            "predicted_lst_p95",
            "uncertainty_half_width",
            "structural_vulnerability",
            "best_operational_intervention",
            "best_operational_benefit",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}",
    )
)


# =========================================================
# SAVE
# =========================================================

csv = gdf.drop(
    columns=["geometry"],
    errors="ignore",
)


csv.to_csv(
    OUTPUT_CSV,
    index=False,
)


gdf.to_file(
    OUTPUT_GPKG,
    layer="operational_interventions",
    driver="GPKG",
)


print(
    "\nSaved:"
)

print(
    OUTPUT_GPKG
)

print(
    OUTPUT_CSV
)


print(
    "\n" + "=" * 70
)

print(
    "OPERATIONAL INTERVENTION BENEFITS COMPLETE"
)

print(
    "=" * 70
)
