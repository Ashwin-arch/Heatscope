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
    RESULTS
    / "heatscope_operational_interventions_v2.gpkg"
)

OUTPUT_GPKG = (
    RESULTS
    / "heatscope_intervention_eligibility.gpkg"
)

OUTPUT_CSV = (
    RESULTS
    / "heatscope_intervention_eligibility.csv"
)


print("=" * 70)
print("HeatScope Intervention Eligibility Analysis")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="operational_interventions_v2",
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
# THRESHOLDS
# =========================================================

ndvi_p50 = gdf[
    "ndvi_mean"
].quantile(0.50)

ndvi_p25 = gdf[
    "ndvi_mean"
].quantile(0.25)

ndmi_p50 = gdf[
    "ndmi_mean"
].quantile(0.50)

building_p50 = gdf[
    "building_coverage_pct"
].quantile(0.50)

building_p75 = gdf[
    "building_coverage_pct"
].quantile(0.75)

population_p50 = gdf[
    "population_density"
].quantile(0.50)

population_p75 = gdf[
    "population_density"
].quantile(0.75)

risk_p50 = gdf[
    "risk_exposure_lambda05"
].quantile(0.50)

risk_p75 = gdf[
    "risk_exposure_lambda05"
].quantile(0.75)

structural_p50 = gdf[
    "structural_vulnerability"
].quantile(0.50)


print(
    "\n" + "=" * 70
)

print(
    "ELIGIBILITY THRESHOLDS"
)

print(
    "=" * 70
)

print(
    f"NDVI P25: {ndvi_p25:.4f}"
)

print(
    f"NDVI P50: {ndvi_p50:.4f}"
)

print(
    f"NDMI P50: {ndmi_p50:.4f}"
)

print(
    f"Building coverage P50: "
    f"{building_p50:.4f}%"
)

print(
    f"Building coverage P75: "
    f"{building_p75:.4f}%"
)

print(
    f"Population density P50: "
    f"{population_p50:.2f}"
)

print(
    f"Population density P75: "
    f"{population_p75:.2f}"
)

print(
    f"Risk P50: {risk_p50:.4f}"
)

print(
    f"Risk P75: {risk_p75:.4f}"
)


# =========================================================
# ELIGIBILITY RULES
# =========================================================
#
# These are transparent scenario rules.
# They are not real-world regulatory requirements.
# =========================================================


# ---------------------------------------------------------
# TREE
# ---------------------------------------------------------
#
# Low vegetation AND sufficient heat / vulnerability.
# ---------------------------------------------------------

gdf[
    "tree_eligible"
] = (
    (gdf["ndvi_mean"] <= ndvi_p50)
    &
    (gdf["risk_exposure_lambda05"] >= risk_p50)
    &
    (
        gdf["structural_vulnerability"]
        >= structural_p50
    )
)


# ---------------------------------------------------------
# SHADE
# ---------------------------------------------------------
#
# Population concentration + heat.
# ---------------------------------------------------------

gdf[
    "shade_eligible"
] = (
    (gdf["population_density"] >= population_p50)
    &
    (gdf["risk_exposure_lambda05"] >= risk_p50)
)


# ---------------------------------------------------------
# COOL ROOF
# ---------------------------------------------------------
#
# Substantial built coverage + heat.
# ---------------------------------------------------------

gdf[
    "cool_roof_eligible"
] = (
    (gdf["building_coverage_pct"] >= building_p50)
    &
    (gdf["risk_exposure_lambda05"] >= risk_p50)
)


# ---------------------------------------------------------
# COOLING / WATER
# ---------------------------------------------------------
#
# High heat + low moisture + population/vulnerability.
# ---------------------------------------------------------

gdf[
    "cooling_eligible"
] = (
    (gdf["ndmi_mean"] <= ndmi_p50)
    &
    (gdf["population_density"] >= population_p50)
    &
    (gdf["risk_exposure_lambda05"] >= risk_p50)
)


# =========================================================
# REQUIREMENT STRENGTH
# =========================================================

eligibility_columns = [
    "tree_eligible",
    "shade_eligible",
    "cool_roof_eligible",
    "cooling_eligible",
]


gdf[
    "number_of_eligible_interventions"
] = gdf[
    eligibility_columns
].sum(
    axis=1
)


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "ELIGIBLE CELLS"
)

print(
    "=" * 70
)


for col in eligibility_columns:

    n = int(
        gdf[col].sum()
    )

    pct = (
        n
        /
        len(gdf)
        * 100
    )

    print(
        f"{col:22s}: "
        f"{n:,} "
        f"({pct:.2f}%)"
    )


# =========================================================
# OVERLAP
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "NUMBER OF AVAILABLE INTERVENTIONS PER CELL"
)

print(
    "=" * 70
)


print(
    gdf[
        "number_of_eligible_interventions"
    ].value_counts()
    .sort_index()
)


# =========================================================
# HIGH PRIORITY CELLS
# =========================================================

top10 = gdf[
    gdf[
        "operational_benefit_percentile_v2"
    ] >= 0.90
]


print(
    "\n" + "=" * 70
)

print(
    "ELIGIBILITY WITHIN TOP 10% PRIORITY"
)

print(
    "=" * 70
)


print(
    "Cells:",
    f"{len(top10):,}"
)


for col in eligibility_columns:

    print(
        f"{col:22s}: "
        f"{int(top10[col].sum()):,}"
        f" ({top10[col].mean() * 100:.2f}%)"
    )


# =========================================================
# SAVE
# =========================================================

csv = gdf.drop(
    columns=["geometry"],
    errors="ignore"
)


csv.to_csv(
    OUTPUT_CSV,
    index=False
)


gdf.to_file(
    OUTPUT_GPKG,
    layer="eligibility",
    driver="GPKG"
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
    "INTERVENTION ELIGIBILITY COMPLETE"
)

print(
    "=" * 70
)
