import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = RESULTS / "heatscope_features.gpkg"

OUTPUT_GPKG = RESULTS / "heatscope_hvi.gpkg"
OUTPUT_CSV = RESULTS / "heatscope_hvi.csv"


print("=" * 70)
print("HeatScope Vulnerability Index Construction")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading feature dataset...")

gdf = gpd.read_file(
    INPUT_FILE,
    layer="features",
)

print(
    "Cells:",
    f"{len(gdf):,}"
)


# =========================================================
# REQUIRED VARIABLES
# =========================================================

required = [
    "cell_id",
    "area_m2",
    "population",
    "building_coverage_pct",
    "lst_celsius_mean",
    "lst_celsius_p95",
    "ndvi_mean",
    "ndmi_mean",
    "ndbi_mean",
]


missing_columns = [
    c for c in required
    if c not in gdf.columns
]


if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# =========================================================
# VALID LANDSAT MASK
# =========================================================

valid = (
    gdf["lst_celsius_mean"].notna()
    & gdf["lst_celsius_p95"].notna()
    & gdf["ndvi_mean"].notna()
    & gdf["ndmi_mean"].notna()
    & gdf["ndbi_mean"].notna()
)


print(
    "\nValid cells:",
    f"{valid.sum():,}"
)

print(
    "Excluded cells:",
    f"{(~valid).sum():,}"
)


# =========================================================
# POPULATION DENSITY
# =========================================================

gdf["population_density"] = (
    gdf["population"]
    / (gdf["area_m2"] / 1_000_000)
)


# =========================================================
# ROBUST MIN-MAX NORMALIZATION
# =========================================================

def robust_minmax(series):

    x = series.copy()

    lower = x.quantile(0.01)
    upper = x.quantile(0.99)

    clipped = x.clip(
        lower=lower,
        upper=upper,
    )

    denominator = upper - lower

    if denominator == 0:

        return pd.Series(
            0.5,
            index=x.index,
        )

    return (
        (clipped - lower)
        / denominator
    )


# =========================================================
# NORMALIZED COMPONENTS
# =========================================================

print(
    "\nConstructing normalized vulnerability components..."
)


# ---------------------------------------------------------
# EXPOSURE
# ---------------------------------------------------------

gdf["exposure_lst"] = (
    robust_minmax(
        gdf["lst_celsius_mean"]
    )
)

gdf["exposure_lst_p95"] = (
    robust_minmax(
        gdf["lst_celsius_p95"]
    )
)

gdf["exposure_ndbi"] = (
    robust_minmax(
        gdf["ndbi_mean"]
    )
)


# ---------------------------------------------------------
# SENSITIVITY
# ---------------------------------------------------------

# Log transformation reduces the influence of extreme
# population-density cells.

gdf["population_density_log"] = np.log1p(
    gdf["population_density"]
)

gdf["sensitivity_population"] = (
    robust_minmax(
        gdf["population_density_log"]
    )
)


# ---------------------------------------------------------
# ADAPTIVE / ENVIRONMENTAL CAPACITY
# ---------------------------------------------------------

# Higher vegetation and moisture imply greater
# environmental cooling capacity.

gdf["capacity_ndvi"] = (
    robust_minmax(
        gdf["ndvi_mean"]
    )
)

gdf["capacity_ndmi"] = (
    robust_minmax(
        gdf["ndmi_mean"]
    )
)

# Dense built form is treated as reduced adaptive capacity
# in the environmental component.

gdf["capacity_built_form"] = (
    1
    - robust_minmax(
        gdf["building_coverage_pct"]
    )
)


# =========================================================
# COMPONENT SCORES
# =========================================================

gdf["exposure_score"] = (
    0.50 * gdf["exposure_lst"]
    + 0.30 * gdf["exposure_lst_p95"]
    + 0.20 * gdf["exposure_ndbi"]
)


gdf["sensitivity_score"] = (
    gdf["sensitivity_population"]
)


gdf["capacity_score"] = (
    0.45 * gdf["capacity_ndvi"]
    + 0.35 * gdf["capacity_ndmi"]
    + 0.20 * gdf["capacity_built_form"]
)


# =========================================================
# VULNERABILITY SCORE
# =========================================================

# Vulnerability increases with exposure and sensitivity,
# while increasing adaptive capacity reduces vulnerability.

gdf["hvi_raw"] = (
    gdf["exposure_score"]
    * gdf["sensitivity_score"]
    * (
        1
        - gdf["capacity_score"]
    )
)


# =========================================================
# NORMALIZED HVI
# =========================================================

gdf["hvi"] = np.nan

gdf.loc[valid, "hvi"] = robust_minmax(
    gdf.loc[valid, "hvi_raw"]
)


# =========================================================
# HVI PERCENTILE
# =========================================================

gdf["hvi_percentile"] = np.nan

gdf.loc[valid, "hvi_percentile"] = (
    gdf.loc[valid, "hvi"]
    .rank(
        pct=True,
        method="average",
    )
)


# =========================================================
# HVI CATEGORIES
# =========================================================

gdf["hvi_category"] = "No Data"

valid_hvi = gdf.loc[
    valid,
    "hvi"
]

gdf.loc[
    valid,
    "hvi_category"
] = pd.cut(
    valid_hvi,
    bins=[
        -np.inf,
        0.20,
        0.40,
        0.60,
        0.80,
        np.inf,
    ],
    labels=[
        "Very Low",
        "Low",
        "Moderate",
        "High",
        "Very High",
    ],
)


# =========================================================
# STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("COMPONENT STATISTICS")
print("=" * 70)

for col in [
    "exposure_score",
    "sensitivity_score",
    "capacity_score",
    "hvi",
]:

    values = gdf.loc[
        valid,
        col
    ].dropna()

    print(
        f"{col:25s} "
        f"min={values.min():.4f} "
        f"median={values.median():.4f} "
        f"mean={values.mean():.4f} "
        f"max={values.max():.4f}"
    )


# =========================================================
# CATEGORY COUNTS
# =========================================================

print("\n" + "=" * 70)
print("HVI CATEGORIES")
print("=" * 70)

print(
    gdf["hvi_category"]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# =========================================================
# POPULATION IN HVI CATEGORIES
# =========================================================

print("\n" + "=" * 70)
print("POPULATION BY HVI CATEGORY")
print("=" * 70)

population_by_category = (
    gdf[
        valid
    ]
    .groupby(
        "hvi_category",
        observed=True,
    )["population"]
    .sum()
)

print(
    population_by_category
    .to_string()
)


# =========================================================
# SAVE
# =========================================================

print("\nSaving GeoPackage...")

gdf.to_file(
    OUTPUT_GPKG,
    layer="hvi",
    driver="GPKG",
)

print(
    "Saved:",
    OUTPUT_GPKG
)


print("\nSaving CSV...")

gdf.drop(
    columns="geometry"
).to_csv(
    OUTPUT_CSV,
    index=False,
)

print(
    "Saved:",
    OUTPUT_CSV
)


print("\n" + "=" * 70)
print("HVI CONSTRUCTION COMPLETE")
print("=" * 70)
