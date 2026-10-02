import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


GRID_FILE = RESULTS / "heatscope_grid.gpkg"
POP_FILE = RESULTS / "heatscope_population.gpkg"
BUILDING_FILE = RESULTS / "heatscope_buildings.gpkg"
LANDSAT_FILE = RESULTS / "heatscope_landsat_features.gpkg"

OUTPUT_FILE = RESULTS / "heatscope_master_v2.gpkg"
CSV_FILE = RESULTS / "heatscope_master_v2.csv"


print("=" * 70)
print("HeatScope Master Feature Integration")
print("=" * 70)


# =========================================================
# LOAD DATASETS
# =========================================================

print("\nLoading datasets...")

grid = gpd.read_file(
    GRID_FILE,
    layer="grid",
)

population = gpd.read_file(
    POP_FILE,
    layer="population",
)

buildings = gpd.read_file(
    BUILDING_FILE,
    layer="buildings",
)

landsat = gpd.read_file(
    LANDSAT_FILE,
    layer="landsat_features",
)

print(f"Grid:       {len(grid):,}")
print(f"Population: {len(population):,}")
print(f"Buildings:  {len(buildings):,}")
print(f"Landsat:    {len(landsat):,}")


# =========================================================
# VALIDATE CELL IDS
# =========================================================

datasets = {
    "population": population,
    "buildings": buildings,
    "landsat": landsat,
}

for name, df in datasets.items():

    if "cell_id" not in df.columns:
        raise RuntimeError(
            f"{name} does not contain cell_id"
        )

    if df["cell_id"].duplicated().any():

        duplicates = df["cell_id"].duplicated().sum()

        raise RuntimeError(
            f"{name} contains {duplicates} duplicate cell_id values"
        )


# =========================================================
# REQUIRED COLUMNS
# =========================================================

pop_cols = [
    "cell_id",
    "population",
]

building_cols = [
    "cell_id",
    "building_count",
    "building_area_m2",
    "building_coverage_pct",
]

landsat_cols = [
    "cell_id",
    "lst_celsius_mean",
    "lst_celsius_median",
    "lst_celsius_p90",
    "lst_celsius_p95",
    "ndvi_mean",
    "ndvi_median",
    "ndbi_mean",
    "ndbi_median",
    "ndmi_mean",
    "ndmi_median",
]


for col in pop_cols:

    if col not in population.columns:
        raise RuntimeError(
            f"Missing population column: {col}"
        )


for col in building_cols:

    if col not in buildings.columns:
        raise RuntimeError(
            f"Missing building column: {col}"
        )


for col in landsat_cols:

    if col not in landsat.columns:
        raise RuntimeError(
            f"Missing Landsat column: {col}"
        )


# =========================================================
# PREPARE TABLES
# =========================================================

pop_df = population[pop_cols].copy()

building_df = buildings[building_cols].copy()

landsat_df = landsat[landsat_cols].copy()


# =========================================================
# MERGE
# =========================================================

print("\nMerging by cell_id...")

master = grid.copy()

master = master.merge(
    pop_df,
    on="cell_id",
    how="left",
    validate="one_to_one",
)

master = master.merge(
    building_df,
    on="cell_id",
    how="left",
    validate="one_to_one",
)

master = master.merge(
    landsat_df,
    on="cell_id",
    how="left",
    validate="one_to_one",
)


# =========================================================
# ROW COUNT CHECK
# =========================================================

if len(master) != len(grid):

    raise RuntimeError(
        "Row count changed during merge!"
    )


# =========================================================
# STRUCTURAL ZERO VALUES
# =========================================================

for col in [
    "population",
    "building_count",
    "building_area_m2",
    "building_coverage_pct",
]:

    master[col] = master[col].fillna(0)


# =========================================================
# MASTER SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("MASTER DATASET")
print("=" * 70)

print(
    "Cells:",
    f"{len(master):,}",
)

print(
    "Columns:",
    len(master.columns),
)


print("\nFeature columns:")

for col in master.columns:

    if col != "geometry":
        print(" ", col)


# =========================================================
# MISSING VALUES
# =========================================================

print("\n" + "=" * 70)
print("Missing Values")
print("=" * 70)

missing = master.isna().sum()

total_missing = 0

for col, count in missing.items():

    if count > 0:

        print(
            f"{col:25s}: {count:,}"
        )

        total_missing += count


if total_missing == 0:

    print("No missing values.")


# =========================================================
# KEY STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("Key Feature Statistics")
print("=" * 70)

key_features = [
    "population",
    "building_coverage_pct",
    "lst_celsius_mean",
    "lst_celsius_p95",
    "ndvi_mean",
    "ndbi_mean",
    "ndmi_mean",
]

for col in key_features:

    values = master[col].dropna()

    print(
        f"{col:25s} "
        f"min={values.min():8.3f} "
        f"median={values.median():8.3f} "
        f"mean={values.mean():8.3f} "
        f"max={values.max():8.3f}"
    )


# =========================================================
# SPEARMAN CORRELATION
# =========================================================

print("\n" + "=" * 70)
print("Correlation With LST")
print("=" * 70)

numeric = master[
    [
        "population",
        "building_coverage_pct",
        "lst_celsius_mean",
        "lst_celsius_p95",
        "ndvi_mean",
        "ndbi_mean",
        "ndmi_mean",
    ]
].copy()

corr = numeric.corr(
    method="spearman"
)

print(
    corr[
        "lst_celsius_mean"
    ].sort_values(
        ascending=False
    ).to_string()
)


# =========================================================
# SAVE GEOPACKAGE
# =========================================================

print("\nSaving GeoPackage...")

master.to_file(
    OUTPUT_FILE,
    layer="master",
    driver="GPKG",
)

print(
    "Saved:",
    OUTPUT_FILE,
)


# =========================================================
# SAVE CSV
# =========================================================

print("\nSaving CSV...")

master.drop(
    columns="geometry"
).to_csv(
    CSV_FILE,
    index=False,
)

print(
    "Saved:",
    CSV_FILE,
)


print("\n" + "=" * 70)
print("MASTER INTEGRATION COMPLETE")
print("=" * 70)
