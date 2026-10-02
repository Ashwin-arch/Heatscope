import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = RESULTS / "heatscope_hvi.gpkg"
OUTPUT_FILE = RESULTS / "heatscope_spatial_blocks.gpkg"


print("=" * 70)
print("HeatScope Spatial Block Construction")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="hvi",
)

print(
    "\nCells:",
    f"{len(gdf):,}"
)

print(
    "CRS:",
    gdf.crs
)


# =========================================================
# CREATE BLOCKS
# =========================================================

# 250 m modelling cells are grouped into approximately
# 2.5 km spatial blocks.
#
# This prevents nearby cells from appearing in both
# training and validation partitions.

BLOCK_SIZE = 2500.0


gdf["block_x"] = np.floor(
    gdf["centroid_x"] / BLOCK_SIZE
).astype(int)


gdf["block_y"] = np.floor(
    gdf["centroid_y"] / BLOCK_SIZE
).astype(int)


# Create deterministic block identifier.

gdf["spatial_block"] = (
    gdf["block_x"].astype(str)
    + "_"
    + gdf["block_y"].astype(str)
)


# =========================================================
# BLOCK STATISTICS
# =========================================================

n_blocks = (
    gdf["spatial_block"]
    .nunique()
)


print(
    "\nBlock size:",
    f"{BLOCK_SIZE / 1000:.1f} km"
)

print(
    "Spatial blocks:",
    f"{n_blocks:,}"
)


block_sizes = (
    gdf
    .groupby(
        "spatial_block"
    )
    .size()
)


print(
    "\nCells per block:"
)

print(
    f"min    : {block_sizes.min():,}"
)

print(
    f"median : {block_sizes.median():,.0f}"
)

print(
    f"mean   : {block_sizes.mean():,.1f}"
)

print(
    f"max    : {block_sizes.max():,}"
)


# =========================================================
# VALID DATA
# =========================================================

valid = (
    gdf["lst_celsius_p95"]
    .notna()
)


print(
    "\nValid target cells:",
    f"{valid.sum():,}"
)


# =========================================================
# SAVE
# =========================================================

gdf.to_file(
    OUTPUT_FILE,
    layer="spatial_blocks",
    driver="GPKG",
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


print(
    "\n" + "=" * 70
)

print(
    "SPATIAL BLOCK CONSTRUCTION COMPLETE"
)

print(
    "=" * 70
)
