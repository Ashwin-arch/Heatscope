import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


GRID = RESULTS / "heatscope_grid.gpkg"
POP = RESULTS / "heatscope_population.gpkg"
BUILDINGS = RESULTS / "heatscope_buildings.gpkg"

OUTPUT = RESULTS / "heatscope_master.gpkg"
CSV_OUTPUT = RESULTS / "heatscope_master.csv"


print("=" * 70)
print("HeatScope Master Feature Construction")
print("=" * 70)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("\nLoading grid...")
grid = gpd.read_file(GRID, layer="grid")

print("Loading population...")
pop = gpd.read_file(POP, layer="population")

print("Loading buildings...")
buildings = gpd.read_file(BUILDINGS, layer="buildings")


print("\nDataset sizes:")
print("Grid      :", len(grid))
print("Population:", len(pop))
print("Buildings :", len(buildings))


# ---------------------------------------------------------
# Check common key
# ---------------------------------------------------------

if "cell_id" not in grid.columns:
    raise RuntimeError("Grid does not contain cell_id")

if "cell_id" not in pop.columns:
    raise RuntimeError("Population does not contain cell_id")

if "cell_id" not in buildings.columns:
    raise RuntimeError("Buildings does not contain cell_id")


# ---------------------------------------------------------
# Keep only useful population columns
# ---------------------------------------------------------

pop_columns = [
    "cell_id",
]

for col in [
    "population",
    "population_density",
]:
    if col in pop.columns:
        pop_columns.append(col)

pop_small = pop[pop_columns].copy()


# ---------------------------------------------------------
# Keep useful building columns
# ---------------------------------------------------------

building_columns = [
    "cell_id",
]

for col in [
    "building_count",
    "building_area_m2",
    "building_coverage_pct",
    "building_density",
]:
    if col in buildings.columns:
        building_columns.append(col)

building_small = buildings[building_columns].copy()


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

print("\nMerging population...")

master = grid.merge(
    pop_small,
    on="cell_id",
    how="left",
    validate="one_to_one",
)

print("Merging building features...")

master = master.merge(
    building_small,
    on="cell_id",
    how="left",
    validate="one_to_one",
)


# ---------------------------------------------------------
# Check merge
# ---------------------------------------------------------

print("\nFinal cells:", len(master))

if len(master) != len(grid):
    raise RuntimeError(
        "Cell count changed during merge!"
    )


# ---------------------------------------------------------
# Fill zero for spatial absence
# ---------------------------------------------------------

for col in [
    "population",
    "population_density",
    "building_count",
    "building_area_m2",
    "building_coverage_pct",
    "building_density",
]:

    if col in master.columns:
        master[col] = master[col].fillna(0)


# ---------------------------------------------------------
# Numeric summary
# ---------------------------------------------------------

print("\nFeature summary:")

numeric = master.select_dtypes(
    include=["number"]
)

print(
    numeric.describe()
    .T[
        [
            "min",
            "mean",
            "50%",
            "max",
        ]
    ]
    .to_string()
)


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\nMissing values:")

missing = master.isna().sum()

print(
    missing[
        missing > 0
    ].sort_values(ascending=False).to_string()
    if (missing > 0).any()
    else "No missing values."
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\nSaving master GeoPackage...")

master.to_file(
    OUTPUT,
    layer="master",
    driver="GPKG",
)

print("Saved:", OUTPUT)


print("\nSaving CSV...")

master.drop(
    columns="geometry"
).to_csv(
    CSV_OUTPUT,
    index=False,
)

print("Saved:", CSV_OUTPUT)

print("\nDONE.")
