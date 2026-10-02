import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math

import geopandas as gpd
import rasterio
from shapely.geometry import box

from config import BBOX, DATA_PROCESSED, GRID_SIZE_METERS, RESULTS


print("=" * 70)
print("HeatScope — 250 m Spatial Analysis Grid")
print("=" * 70)


# =========================================================
# 1. Get study-area CRS from processed Landsat
# =========================================================

lst_path = DATA_PROCESSED / "lst_celsius.tif"

with rasterio.open(lst_path) as src:

    raster_crs = src.crs
    raster_bounds = src.bounds

print("\nLandsat CRS:", raster_crs)
print("Landsat bounds:")
print(raster_bounds)


# =========================================================
# 2. Create AOI from longitude/latitude
# =========================================================

aoi_wgs84 = gpd.GeoDataFrame(
    {
        "name": ["Bengaluru_AOI"]
    },
    geometry=[
        box(
            BBOX[0],
            BBOX[1],
            BBOX[2],
            BBOX[3]
        )
    ],
    crs="EPSG:4326",
)


# Transform to Landsat projected CRS
aoi = aoi_wgs84.to_crs(raster_crs)

minx, miny, maxx, maxy = aoi.total_bounds

print("\nProjected AOI bounds:")
print(
    f"minx={minx:.2f}, "
    f"miny={miny:.2f}, "
    f"maxx={maxx:.2f}, "
    f"maxy={maxy:.2f}"
)


# =========================================================
# 3. Generate regular 250 m grid
# =========================================================

cell_size = GRID_SIZE_METERS

x_start = math.floor(minx / cell_size) * cell_size
y_start = math.floor(miny / cell_size) * cell_size

x_end = math.ceil(maxx / cell_size) * cell_size
y_end = math.ceil(maxy / cell_size) * cell_size


cells = []

cell_id = 0

y = y_start

while y < y_end:

    x = x_start

    while x < x_end:

        cell = box(
            x,
            y,
            x + cell_size,
            y + cell_size
        )

        cells.append(
            {
                "cell_id": cell_id,
                "geometry": cell,
            }
        )

        cell_id += 1
        x += cell_size

    y += cell_size


grid = gpd.GeoDataFrame(
    cells,
    crs=raster_crs,
)


# =========================================================
# 4. Keep cells intersecting the actual AOI
# =========================================================

grid = gpd.overlay(
    grid,
    aoi[["geometry"]],
    how="intersection",
)


# Reassign sequential IDs after clipping
grid = grid.reset_index(drop=True)

grid["cell_id"] = range(len(grid))


# =========================================================
# 5. Calculate geometric properties
# =========================================================

grid["area_m2"] = grid.geometry.area

grid["area_ha"] = grid["area_m2"] / 10000.0

centroids = grid.geometry.centroid

grid["centroid_x"] = centroids.x

grid["centroid_y"] = centroids.y


# =========================================================
# 6. Add geographic coordinates
# =========================================================

centroid_gdf = gpd.GeoDataFrame(
    grid[
        [
            "cell_id",
            "geometry"
        ]
    ].copy(),
    geometry=centroids,
    crs=raster_crs,
)

centroid_wgs84 = centroid_gdf.to_crs(
    "EPSG:4326"
)

grid["longitude"] = centroid_wgs84.geometry.x

grid["latitude"] = centroid_wgs84.geometry.y


# =========================================================
# 7. Quality statistics
# =========================================================

print("\nGrid statistics:")

print("Number of cells:", len(grid))

print(
    "Cell area:",
    f"{grid.area_m2.min():.2f}",
    "to",
    f"{grid.area_m2.max():.2f}",
    "m²"
)

print(
    "Mean area:",
    f"{grid.area_m2.mean():.2f}",
    "m²"
)

print(
    "Study area:",
    f"{grid.area_m2.sum() / 1_000_000:.2f}",
    "km²"
)


# =========================================================
# 8. Save GeoPackage
# =========================================================

output = RESULTS / "heatscope_grid.gpkg"

if output.exists():
    output.unlink()

grid.to_file(
    output,
    layer="grid",
    driver="GPKG",
)


# Also save GeoJSON for easy inspection
geojson = RESULTS / "heatscope_grid.geojson"

grid.to_file(
    geojson,
    driver="GeoJSON",
)


# =========================================================
# 9. Save CSV version
# =========================================================

csv_output = RESULTS / "heatscope_grid.csv"

grid.drop(
    columns="geometry"
).to_csv(
    csv_output,
    index=False,
)


print("\nSaved:")
print(" ", output)
print(" ", geojson)
print(" ", csv_output)


print("\n" + "=" * 70)
print("GRID CREATION COMPLETE")
print("=" * 70)
