import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


MASTER_FILE = RESULTS / "heatscope_master_v2.gpkg"
WEATHER_FILE = RESULTS / "heatscope_weather_daily.csv"

OUTPUT_GPKG = RESULTS / "heatscope_features.gpkg"
OUTPUT_CSV = RESULTS / "heatscope_features.csv"


print("=" * 70)
print("HeatScope Spatial + Temporal Feature Integration")
print("=" * 70)


# =========================================================
# LOAD MASTER
# =========================================================

print("\nLoading spatial master...")

master = gpd.read_file(
    MASTER_FILE,
    layer="master",
)

print(
    "Spatial cells:",
    f"{len(master):,}"
)


# =========================================================
# LOAD WEATHER
# =========================================================

print("\nLoading NASA POWER weather...")

weather = pd.read_csv(
    WEATHER_FILE,
    parse_dates=["date"],
)

print(
    "Weather rows:",
    f"{len(weather):,}"
)


# =========================================================
# VALIDATE WEATHER
# =========================================================

required_weather = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]


for col in required_weather:

    if col not in weather.columns:

        raise RuntimeError(
            f"Missing weather column: {col}"
        )


# =========================================================
# WEATHER SUMMARY
# =========================================================

print("\nCalculating robust temporal features...")


def percentile(series, q):
    return series.dropna().quantile(q)


weather_features = {
    "weather_t2m_mean":
        weather["T2M"].mean(),

    "weather_t2m_max":
        weather["T2M_MAX"].max(),

    "weather_t2m_p95":
        percentile(
            weather["T2M_MAX"],
            0.95,
        ),

    "weather_t2m_min":
        weather["T2M_MIN"].min(),

    "weather_rh_mean":
        weather["RH2M"].mean(),

    "weather_rh_max":
        weather["RH2M"].max(),

    "weather_hot_days_30":
        (
            weather["T2M_MAX"] >= 30
        ).sum(),

    "weather_hot_days_32":
        (
            weather["T2M_MAX"] >= 32
        ).sum(),

    "weather_hot_days_35":
        (
            weather["T2M_MAX"] >= 35
        ).sum(),

    "weather_warm_nights_24":
        (
            weather["T2M_MIN"] >= 24
        ).sum(),

    "weather_max_hot32_streak":
        weather["HOT32_STREAK"].max(),

    "weather_t2m_anomaly_max":
        weather["T2M_ANOMALY"].max(),

    "weather_precip_total_mm":
        weather["PRECTOTCORR"].sum(),

    "weather_precip_mean_mm":
        weather["PRECTOTCORR"].mean(),

    "weather_wind_mean":
        weather["WS2M"].mean(),

    "weather_wind_max":
        weather["WS2M"].max(),
}


# =========================================================
# PRINT WEATHER FEATURES
# =========================================================

print("\n" + "=" * 70)
print("TEMPORAL FEATURES")
print("=" * 70)

for key, value in weather_features.items():

    print(
        f"{key:32s}: {value:.4f}"
    )


# =========================================================
# ATTACH TO EVERY SPATIAL CELL
# =========================================================

print(
    "\nAttaching regional temporal features "
    "to spatial cells..."
)

for key, value in weather_features.items():

    master[key] = value


# =========================================================
# WEATHER DATA QUALITY FLAG
# =========================================================

weather_missing = (
    weather[required_weather]
    .isna()
    .any(axis=1)
)

missing_weather_days = int(
    weather_missing.sum()
)

master["weather_missing_days"] = (
    missing_weather_days
)


# =========================================================
# CREATE OBSERVATION PERIOD FEATURES
# =========================================================

master["weather_period_days"] = len(
    weather
)

master["weather_start_year"] = (
    weather["date"].min().year
)

master["weather_start_month"] = (
    weather["date"].min().month
)

master["weather_end_year"] = (
    weather["date"].max().year
)

master["weather_end_month"] = (
    weather["date"].max().month
)


# =========================================================
# CHECK
# =========================================================

print("\n" + "=" * 70)
print("FINAL FEATURE TABLE")
print("=" * 70)

print(
    "Cells:",
    f"{len(master):,}"
)

print(
    "Columns:",
    len(master.columns)
)


# =========================================================
# MISSING VALUES
# =========================================================

print("\nMissing values:")

missing = master.isna().sum()

found_missing = False

for col, count in missing.items():

    if count > 0:

        found_missing = True

        print(
            f"{col:32s}: {count:,}"
        )


if not found_missing:

    print("No missing values.")


# =========================================================
# WEATHER CONSISTENCY CHECK
# =========================================================

print("\nWeather values across spatial cells:")

for col in [
    "weather_t2m_mean",
    "weather_t2m_p95",
    "weather_hot_days_32",
    "weather_max_hot32_streak",
]:

    unique_values = (
        master[col]
        .nunique(dropna=True)
    )

    print(
        f"{col:32s}: "
        f"{unique_values} unique value(s)"
    )


# =========================================================
# SAVE
# =========================================================

print("\nSaving GeoPackage...")

master.to_file(
    OUTPUT_GPKG,
    layer="features",
    driver="GPKG",
)

print(
    "Saved:",
    OUTPUT_GPKG
)


print("\nSaving CSV...")

master.drop(
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
print("STAGE 8 COMPLETE")
print("=" * 70)
