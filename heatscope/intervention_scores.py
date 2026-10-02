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


INPUT_FILE = RESULTS / "heatscope_hvi.gpkg"

OUTPUT_GPKG = (
    RESULTS / "heatscope_intervention_scores.gpkg"
)

OUTPUT_CSV = (
    RESULTS / "heatscope_intervention_scores.csv"
)


print("=" * 70)
print("HeatScope Intervention Suitability & Benefit Scoring")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="hvi",
)


valid = gdf["hvi"].notna()

gdf = gdf.loc[
    valid
].copy()


print(
    "\nValid cells:",
    f"{len(gdf):,}"
)


# =========================================================
# ROBUST NORMALIZATION
# =========================================================

def robust_norm(series):

    x = pd.to_numeric(
        series,
        errors="coerce"
    ).astype(float)

    lo = x.quantile(0.01)
    hi = x.quantile(0.99)

    if hi <= lo:
        return pd.Series(
            0.5,
            index=series.index
        )

    x = x.clip(
        lo,
        hi
    )

    return (
        (x - lo)
        /
        (hi - lo)
    )


# =========================================================
# NORMALIZED BASE VARIABLES
# =========================================================

gdf["n_hvi"] = robust_norm(
    gdf["hvi"]
)

gdf["n_population"] = robust_norm(
    np.log1p(
        gdf["population"]
    )
)

gdf["n_population_density"] = robust_norm(
    np.log1p(
        gdf["population_density"]
    )
)

gdf["n_ndvi"] = robust_norm(
    gdf["ndvi_mean"]
)

gdf["n_ndmi"] = robust_norm(
    gdf["ndmi_mean"]
)

gdf["n_building"] = robust_norm(
    gdf["building_coverage_pct"]
)

gdf["n_lst"] = robust_norm(
    gdf["lst_celsius_p95"]
)


# =========================================================
# ELIGIBILITY
# =========================================================
#
# We use a soft suitability score rather than a hard
# threshold. This avoids arbitrary "eligible/not eligible"
# decisions at this stage.
# =========================================================


# ---------------------------------------------------------
# 1. TREE / GREENING
# ---------------------------------------------------------
#
# High opportunity where:
#   low NDVI
#   low NDMI
#   high heat
#   high vulnerability
#
# Population is included because the intervention ultimately
# serves people.
# ---------------------------------------------------------

tree_suitability = (
    0.35 * (1 - gdf["n_ndvi"])
    +
    0.20 * (1 - gdf["n_ndmi"])
    +
    0.25 * gdf["n_lst"]
    +
    0.20 * gdf["n_hvi"]
)

gdf[
    "tree_suitability"
] = tree_suitability.clip(
    0,
    1
)


# ---------------------------------------------------------
# 2. COOL ROOF
# ---------------------------------------------------------
#
# High opportunity where:
#   built coverage is high
#   heat is high
#   vulnerability is high
#
# ---------------------------------------------------------

cool_roof_suitability = (
    0.40 * gdf["n_building"]
    +
    0.30 * gdf["n_lst"]
    +
    0.30 * gdf["n_hvi"]
)

gdf[
    "cool_roof_suitability"
] = cool_roof_suitability.clip(
    0,
    1
)


# ---------------------------------------------------------
# 3. SHADE
# ---------------------------------------------------------
#
# High opportunity where:
#   people are concentrated
#   vegetation is limited
#   heat is high
#   vulnerability is high
#
# ---------------------------------------------------------

shade_suitability = (
    0.30 * gdf["n_population_density"]
    +
    0.20 * (1 - gdf["n_ndvi"])
    +
    0.25 * gdf["n_lst"]
    +
    0.25 * gdf["n_hvi"]
)

gdf[
    "shade_suitability"
] = shade_suitability.clip(
    0,
    1
)


# ---------------------------------------------------------
# 4. COOLING / WATER
# ---------------------------------------------------------
#
# High opportunity where:
#   vulnerability is high
#   heat is high
#   moisture is relatively low
#   population is affected
#
# ---------------------------------------------------------

cooling_suitability = (
    0.30 * gdf["n_hvi"]
    +
    0.25 * gdf["n_lst"]
    +
    0.20 * (1 - gdf["n_ndmi"])
    +
    0.25 * gdf["n_population"]
)

gdf[
    "cooling_suitability"
] = cooling_suitability.clip(
    0,
    1
)


# =========================================================
# BENEFIT POTENTIAL
# =========================================================
#
# This is NOT °C reduction.
#
# It is an intervention-priority quantity representing:
#
#     vulnerability × affected population × suitability
#
# The population multiplier is normalized so that the score
# does not simply become population count.
# =========================================================

for intervention in [
    "tree",
    "cool_roof",
    "shade",
    "cooling",
]:

    suitability_col = (
        f"{intervention}_suitability"
    )

    gdf[
        f"{intervention}_benefit_potential"
    ] = (
        gdf["n_hvi"]
        * gdf["n_population"]
        * gdf[suitability_col]
    )


# =========================================================
# POPULATION-WEIGHTED BENEFIT
# =========================================================
#
# Useful for equity and intervention optimization.
# =========================================================

for intervention in [
    "tree",
    "cool_roof",
    "shade",
    "cooling",
]:

    potential_col = (
        f"{intervention}_benefit_potential"
    )

    gdf[
        f"{intervention}_population_benefit"
    ] = (
        gdf[potential_col]
        * gdf["population"]
    )


# =========================================================
# PRIMARY INTERVENTION
# =========================================================

benefit_columns = [
    "tree_benefit_potential",
    "cool_roof_benefit_potential",
    "shade_benefit_potential",
    "cooling_benefit_potential",
]


gdf[
    "best_intervention"
] = (
    gdf[
        benefit_columns
    ]
    .idxmax(axis=1)
    .str.replace(
        "_benefit_potential",
        "",
        regex=False
    )
)


# =========================================================
# PRIORITY
# =========================================================

max_benefit = gdf[
    benefit_columns
].max(axis=1)


gdf[
    "max_intervention_benefit"
] = max_benefit


gdf[
    "intervention_priority_percentile"
] = (
    max_benefit.rank(
        pct=True,
        method="average"
    )
)


gdf[
    "priority_top_10pct"
] = (
    gdf[
        "intervention_priority_percentile"
    ] >= 0.90
)


gdf[
    "priority_top_20pct"
] = (
    gdf[
        "intervention_priority_percentile"
    ] >= 0.80
)


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "INTERVENTION SUITABILITY SUMMARY"
)

print(
    "=" * 70
)


for intervention in [
    "tree",
    "cool_roof",
    "shade",
    "cooling",
]:

    col = (
        f"{intervention}_suitability"
    )

    print(
        f"\n{intervention}"
    )

    print(
        f"  median suitability: "
        f"{gdf[col].median():.4f}"
    )

    print(
        f"  P90 suitability: "
        f"{gdf[col].quantile(.90):.4f}"
    )

    print(
        f"  max suitability: "
        f"{gdf[col].max():.4f}"
    )


# =========================================================
# BEST INTERVENTION COUNTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "BEST INTERVENTION BY CELL"
)

print(
    "=" * 70
)

print(
    gdf[
        "best_intervention"
    ].value_counts()
)


# =========================================================
# HIGH-VULNERABILITY AREA
# =========================================================

high_hvi = gdf[
    gdf["hvi_category"].isin(
        [
            "High",
            "Very High",
        ]
    )
]


print(
    "\n" + "=" * 70
)

print(
    "INTERVENTION PRIORITY IN HIGH-VULNERABILITY CELLS"
)

print(
    "=" * 70
)

print(
    "Cells:",
    f"{len(high_hvi):,}"
)

print(
    "Population:",
    f"{high_hvi['population'].sum():,.0f}"
)


print(
    "\nBest intervention:"
)

print(
    high_hvi[
        "best_intervention"
    ].value_counts()
)


# =========================================================
# TOP 100 CELLS
# =========================================================

top = gdf.sort_values(
    "max_intervention_benefit",
    ascending=False,
).head(100)


print(
    "\n" + "=" * 70
)

print(
    "TOP 10 INTERVENTION PRIORITY CELLS"
)

print(
    "=" * 70
)


print(
    top[
        [
            "cell_id",
            "longitude",
            "latitude",
            "hvi",
            "population",
            "lst_celsius_p95",
            "best_intervention",
            "max_intervention_benefit",
        ]
    ].head(10).to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# SAVE
# =========================================================

drop_columns = [
    "n_hvi",
    "n_population",
    "n_population_density",
    "n_ndvi",
    "n_ndmi",
    "n_building",
    "n_lst",
]


csv_data = gdf.drop(
    columns=[
        "geometry"
    ],
    errors="ignore",
).copy()


csv_data.to_csv(
    OUTPUT_CSV,
    index=False,
)


gdf.to_file(
    OUTPUT_GPKG,
    layer="intervention_scores",
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
    "INTERVENTION SCORING COMPLETE"
)

print(
    "=" * 70
)
