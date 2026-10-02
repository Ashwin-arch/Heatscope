import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio

from config import DATA_RAW, DATA_PROCESSED


def read_band(filename):
    path = DATA_RAW / filename

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    return data, profile


def save_band(filename, data, profile):
    output = DATA_PROCESSED / filename

    out_profile = profile.copy()

    out_profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
        nodata=np.nan,
    )

    with rasterio.open(output, "w", **out_profile) as dst:
        dst.write(data.astype(np.float32), 1)

    print(f"Saved: {output}")


print("=" * 70)
print("HeatScope — Satellite Preprocessing")
print("=" * 70)


# =========================================================
# LOAD RAW DATA
# =========================================================

print("\nLoading Landsat rasters...")

red_raw, profile = read_band("red.tif")
nir_raw, _ = read_band("nir.tif")
swir_raw, _ = read_band("swir.tif")
thermal_raw, _ = read_band("lst_raw.tif")
qa_raw, _ = read_band("qa.tif")

print("Shape:", red_raw.shape)
print("CRS:", profile["crs"])
print(
    "Resolution:",
    profile["transform"].a,
    "x",
    abs(profile["transform"].e),
    "meters"
)


# =========================================================
# RAW DATA VALIDITY
# =========================================================

# Landsat fill pixels are commonly represented by zero DN.
# Identify these BEFORE applying scale factors.

raw_invalid = (
    (red_raw <= 0)
    | (nir_raw <= 0)
    | (swir_raw <= 0)
    | (thermal_raw <= 0)
)


# =========================================================
# LANDSAT COLLECTION 2 LEVEL-2 SCALING
# =========================================================

print("\nApplying Landsat scale factors...")

# Surface reflectance
red = red_raw * 0.0000275 - 0.2
nir = nir_raw * 0.0000275 - 0.2
swir = swir_raw * 0.0000275 - 0.2

# Surface temperature
# Kelvin = DN * 0.00341802 + 149.0
# Celsius = Kelvin - 273.15

lst = thermal_raw * 0.00341802 + 149.0 - 273.15


# =========================================================
# QA_PIXEL CLOUD / SHADOW MASK
# =========================================================

print("Applying QA_PIXEL cloud/shadow mask...")

qa = qa_raw.astype(np.uint16)

fill = (qa & (1 << 0)) != 0
dilated_cloud = (qa & (1 << 1)) != 0
cirrus = (qa & (1 << 2)) != 0
cloud = (qa & (1 << 3)) != 0
cloud_shadow = (qa & (1 << 4)) != 0
snow = (qa & (1 << 5)) != 0

qa_invalid = (
    fill
    | dilated_cloud
    | cirrus
    | cloud
    | cloud_shadow
    | snow
)


# =========================================================
# PHYSICAL VALIDITY MASK
# =========================================================

print("Applying physical validity checks...")

invalid = raw_invalid | qa_invalid

# Reflectance must be physically reasonable.
invalid |= red < 0
invalid |= nir < 0
invalid |= swir < 0

invalid |= red > 1
invalid |= nir > 1
invalid |= swir > 1

# Thermal product sanity range.
#
# -50°C is safely below realistic Bengaluru land temperatures.
#  80°C is safely above realistic daytime surface temperatures.
#
# This mainly protects against corrupted/fill values rather
# than clipping legitimate observations.

invalid |= lst < -50
invalid |= lst > 80

# Remove non-finite values.

invalid |= ~np.isfinite(red)
invalid |= ~np.isfinite(nir)
invalid |= ~np.isfinite(swir)
invalid |= ~np.isfinite(lst)


# =========================================================
# SPECTRAL INDICES
# =========================================================

print("Calculating NDVI / NDBI / NDMI...")

eps = 1e-6

ndvi_den = nir + red
ndbi_den = swir + nir
ndmi_den = nir + swir

ndvi = np.divide(
    nir - red,
    ndvi_den,
    out=np.full_like(nir, np.nan),
    where=np.abs(ndvi_den) > eps,
)

ndbi = np.divide(
    swir - nir,
    ndbi_den,
    out=np.full_like(nir, np.nan),
    where=np.abs(ndbi_den) > eps,
)

ndmi = np.divide(
    nir - swir,
    ndmi_den,
    out=np.full_like(nir, np.nan),
    where=np.abs(ndmi_den) > eps,
)


# =========================================================
# INDEX SANITY MASK
# =========================================================

# Normalized difference indices should theoretically be
# between -1 and +1. Anything outside is numerical/data error.

invalid_index = (
    ~np.isfinite(ndvi)
    | ~np.isfinite(ndbi)
    | ~np.isfinite(ndmi)
    | (ndvi < -1)
    | (ndvi > 1)
    | (ndbi < -1)
    | (ndbi > 1)
    | (ndmi < -1)
    | (ndmi > 1)
)

invalid |= invalid_index


# =========================================================
# APPLY FINAL MASK
# =========================================================

red[invalid] = np.nan
nir[invalid] = np.nan
swir[invalid] = np.nan
lst[invalid] = np.nan
ndvi[invalid] = np.nan
ndbi[invalid] = np.nan
ndmi[invalid] = np.nan


# =========================================================
# SAVE
# =========================================================

print("\nWriting processed rasters...")

save_band("red_reflectance.tif", red, profile)
save_band("nir_reflectance.tif", nir, profile)
save_band("swir_reflectance.tif", swir, profile)
save_band("lst_celsius.tif", lst, profile)
save_band("ndvi.tif", ndvi, profile)
save_band("ndbi.tif", ndbi, profile)
save_band("ndmi.tif", ndmi, profile)


# =========================================================
# DATA QUALITY REPORT
# =========================================================

print("\n" + "-" * 70)
print("DATA QUALITY CHECK")
print("-" * 70)


def statistics(name, array):

    values = array[np.isfinite(array)]

    if len(values) == 0:
        print(f"{name}: NO VALID VALUES")
        return

    p01, p50, p99 = np.percentile(
        values,
        [1, 50, 99]
    )

    print(
        f"{name:8s} | "
        f"min={values.min():8.3f} | "
        f"P01={p01:8.3f} | "
        f"median={p50:8.3f} | "
        f"P99={p99:8.3f} | "
        f"max={values.max():8.3f}"
    )


valid_fraction = np.isfinite(lst).mean() * 100

print(f"Valid pixels: {valid_fraction:.2f}%")

statistics("NDVI", ndvi)
statistics("NDBI", ndbi)
statistics("NDMI", ndmi)
statistics("LST °C", lst)


# =========================================================
# ADDITIONAL LST SANITY CHECK
# =========================================================

lst_values = lst[np.isfinite(lst)]

if len(lst_values) > 0:

    realistic = (
        (lst_values >= 10)
        & (lst_values <= 65)
    )

    realistic_fraction = realistic.mean() * 100

    print()
    print(
        f"LST within 10–65°C: "
        f"{realistic_fraction:.2f}%"
    )


print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
