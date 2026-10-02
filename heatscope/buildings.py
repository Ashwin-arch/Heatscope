import sys
import gzip
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, box
from shapely.strtree import STRtree
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BBOX, RESULTS, DATA_RAW


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BUILDING_FILE = DATA_RAW / "buildings" / "india_123303312.csv.gz"

GRID_FILE = RESULTS / "heatscope_grid.gpkg"
OUTPUT_FILE = RESULTS / "heatscope_buildings.gpkg"

AOI = box(
    BBOX[0],  # min longitude
    BBOX[1],  # min latitude
    BBOX[2],  # max longitude
    BBOX[3],  # max latitude
)

# Building footprints are provided in WGS84.
# HeatScope grid is in UTM Zone 43N.
WGS84 = "EPSG:4326"
GRID_CRS = "EPSG:32643"

transformer = Transformer.from_crs(
    WGS84,
    GRID_CRS,
    always_xy=True,
)


# ---------------------------------------------------------
# Load HeatScope grid
# ---------------------------------------------------------

print("=" * 70)
print("HeatScope Building Footprint Aggregation")
print("=" * 70)

print("\nLoading HeatScope grid...")

grid = gpd.read_file(
    GRID_FILE,
    layer="grid",
)

grid = grid.to_crs(GRID_CRS)

print("Grid cells:", f"{len(grid):,}")
print("Grid CRS:", grid.crs)


# ---------------------------------------------------------
# Prepare spatial index
# ---------------------------------------------------------

print("\nPreparing grid spatial index...")

grid_geometries = grid.geometry.tolist()
grid_tree = STRtree(grid_geometries)

print("Spatial index ready.")


# ---------------------------------------------------------
# Accumulators
# ---------------------------------------------------------

building_count = [0] * len(grid)
building_area = [0.0] * len(grid)

total_records = 0
valid_buildings = 0
aoi_buildings = 0


# ---------------------------------------------------------
# Stream building footprints
# ---------------------------------------------------------

print("\nStreaming building footprints...")
print("Source:", BUILDING_FILE)
print()

with gzip.open(BUILDING_FILE, "rt", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        total_records += 1

        try:
            obj = json.loads(line)
            geom_data = obj.get("geometry")

            if not geom_data:
                continue

            geom_wgs84 = shape(geom_data)

            if geom_wgs84.is_empty:
                continue

            if not geom_wgs84.is_valid:
                geom_wgs84 = geom_wgs84.buffer(0)

            if geom_wgs84.is_empty:
                continue

            valid_buildings += 1

            # -------------------------------------------------
            # Fast AOI filtering in geographic coordinates
            # -------------------------------------------------

            if not geom_wgs84.intersects(AOI):
                continue

            aoi_buildings += 1

            # -------------------------------------------------
            # Project building to UTM 43N
            # -------------------------------------------------

            geom_utm = gpd.GeoSeries(
                [geom_wgs84],
                crs=WGS84,
            ).to_crs(GRID_CRS).iloc[0]

            if geom_utm.is_empty:
                continue

            # -------------------------------------------------
            # Find candidate grid cells
            # -------------------------------------------------

            candidates = grid_tree.query(geom_utm)

            for candidate in candidates:

                # Shapely STRtree normally returns integer indices.
                # Handle object references defensively.
                if hasattr(candidate, "item"):
                    idx = int(candidate.item())
                else:
                    idx = int(candidate)

                cell = grid_geometries[idx]

                if not geom_utm.intersects(cell):
                    continue

                intersection = geom_utm.intersection(cell)

                if intersection.is_empty:
                    continue

                area = intersection.area

                if area <= 0:
                    continue

                building_count[idx] += 1
                building_area[idx] += area

        except Exception as exc:

            print(
                f"\nWarning: skipped record {total_records}: {exc}",
                file=sys.stderr,
            )

        # Progress every 100,000 records
        if total_records % 100000 == 0:

            print(
                f"\rProcessed: {total_records:,} | "
                f"Valid: {valid_buildings:,} | "
                f"AOI: {aoi_buildings:,}",
                end="",
                flush=True,
            )


print()
print("\nStreaming complete.")

print("Total records :", f"{total_records:,}")
print("Valid         :", f"{valid_buildings:,}")
print("Inside AOI    :", f"{aoi_buildings:,}")


# ---------------------------------------------------------
# Add derived variables
# ---------------------------------------------------------

print("\nCalculating building metrics...")

grid["building_count"] = building_count
grid["building_area_m2"] = building_area

grid["building_coverage_pct"] = (
    grid["building_area_m2"]
    / grid.geometry.area
    * 100.0
)

grid["building_density"] = (
    grid["building_area_m2"]
    / (grid.geometry.area / 1_000_000.0)
)


# ---------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------

grid["building_count"] = grid["building_count"].astype("int32")
grid["building_area_m2"] = grid["building_area_m2"].astype("float32")
grid["building_coverage_pct"] = (
    grid["building_coverage_pct"].astype("float32")
)
grid["building_density"] = (
    grid["building_density"].astype("float32")
)

print("\nSanity checks:")

print(
    "Building count:",
    int(grid["building_count"].sum()),
)

print(
    "Total building area:",
    f"{grid['building_area_m2'].sum() / 1e6:.2f} km²",
)

print(
    "Coverage min:",
    f"{grid['building_coverage_pct'].min():.3f}%",
)

print(
    "Coverage median:",
    f"{grid['building_coverage_pct'].median():.3f}%",
)

print(
    "Coverage max:",
    f"{grid['building_coverage_pct'].max():.3f}%",
)

print(
    "Cells with buildings:",
    int((grid["building_count"] > 0).sum()),
    "/",
    len(grid),
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\nSaving:", OUTPUT_FILE)

grid.to_file(
    OUTPUT_FILE,
    layer="buildings",
    driver="GPKG",
)

CSV_FILE = RESULTS / "heatscope_buildings.csv"

grid.drop(columns="geometry").to_csv(
    CSV_FILE,
    index=False,
)

print("Saved:", OUTPUT_FILE)
print("Saved:", CSV_FILE)

print("\nDONE.")
