import sys
from pathlib import Path
import numpy as np
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS

FILE = RESULTS / "heatscope_spatial_blocks.gpkg"

gdf = gpd.read_file(
    FILE,
    layer="spatial_blocks",
)

print("=" * 70)
print("HeatScope Grid Regularity Check")
print("=" * 70)

print("\nCells:", len(gdf))
print("CRS:", gdf.crs)

cent = gdf.geometry.centroid

x = np.sort(cent.x.unique())
y = np.sort(cent.y.unique())

dx = np.diff(x)
dy = np.diff(y)

print("\nUnique centroid X:", len(x))
print("Unique centroid Y:", len(y))

print(
    "Median X spacing:",
    np.median(dx),
    "m"
)

print(
    "Median Y spacing:",
    np.median(dy),
    "m"
)

print(
    "Minimum X spacing:",
    np.min(dx),
    "m"
)

print(
    "Maximum X spacing:",
    np.max(dx),
    "m"
)

print(
    "Minimum Y spacing:",
    np.min(dy),
    "m"
)

print(
    "Maximum Y spacing:",
    np.max(dy),
    "m"
)

geom_area = gdf.geometry.area

print("\nCell area:")
print(
    "median:",
    np.median(geom_area),
    "m²"
)
print(
    "max:",
    np.max(geom_area),
    "m²"
)
print(
    "min:",
    np.min(geom_area),
    "m²"
)

print("\nGeometry types:")
print(gdf.geometry.geom_type.value_counts())

print("\n" + "=" * 70)
print("GRID CHECK COMPLETE")
print("=" * 70)
