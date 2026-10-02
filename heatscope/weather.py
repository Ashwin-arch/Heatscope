import sys
from pathlib import Path
from datetime import date, timedelta

import requests
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CITY_LAT, CITY_LON, DAYS_BACK, RESULTS


OUTPUT_CSV = RESULTS / "heatscope_weather_daily.csv"
SUMMARY_CSV = RESULTS / "heatscope_weather_summary.csv"


# =========================================================
# CONFIGURATION
# =========================================================

END_DATE = date.today()
START_DATE = END_DATE - timedelta(
    days=DAYS_BACK
)

BASE_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)


PARAMETERS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]


print("=" * 70)
print("HeatScope NASA POWER Weather Acquisition")
print("=" * 70)

print(
    f"\nLocation: "
    f"{CITY_LAT}, {CITY_LON}"
)

print(
    f"Period: "
    f"{START_DATE} -> {END_DATE}"
)

print(
    "\nParameters:",
    ", ".join(PARAMETERS),
)


# =========================================================
# API REQUEST
# =========================================================

params = {
    "parameters": ",".join(PARAMETERS),
    "community": "AG",
    "longitude": CITY_LON,
    "latitude": CITY_LAT,
    "start": START_DATE.strftime("%Y%m%d"),
    "end": END_DATE.strftime("%Y%m%d"),
    "format": "JSON",
}


print("\nRequesting NASA POWER data...")

response = requests.get(
    BASE_URL,
    params=params,
    timeout=120,
)

print(
    "HTTP status:",
    response.status_code,
)


response.raise_for_status()

payload = response.json()


# =========================================================
# PARSE RESPONSE
# =========================================================

properties = payload["properties"]

parameter_data = properties["parameter"]

print(
    "Returned parameters:",
    ", ".join(parameter_data.keys())
)


# NASA POWER uses YYYYMMDD keys
dates = sorted(
    parameter_data["T2M"].keys()
)


records = []

for d in dates:

    record = {
        "date": pd.to_datetime(
            d,
            format="%Y%m%d",
        )
    }

    for parameter in PARAMETERS:

        value = parameter_data[
            parameter
        ].get(d, np.nan)

        record[parameter] = (
            np.nan
            if value == -999
            else value
        )

    records.append(record)


weather = pd.DataFrame(
    records
)


# =========================================================
# SORT
# =========================================================

weather = weather.sort_values(
    "date"
).reset_index(
    drop=True
)


# =========================================================
# BASIC QUALITY CHECK
# =========================================================

print("\n" + "=" * 70)
print("Weather Dataset")
print("=" * 70)

print(
    "Rows:",
    f"{len(weather):,}"
)

print(
    "Date range:",
    weather["date"].min().date(),
    "->",
    weather["date"].max().date(),
)


print("\nMissing values:")

print(
    weather.isna().sum().to_string()
)


# =========================================================
# HEAT-STRESS FEATURES
# =========================================================

print(
    "\nCalculating temporal heat-stress indicators..."
)


# ---------------------------------------------------------
# Rolling temperature statistics
# ---------------------------------------------------------

weather["T2M_7D_MEAN"] = (
    weather["T2M"]
    .rolling(
        7,
        min_periods=3,
    )
    .mean()
)

weather["T2M_7D_MAX"] = (
    weather["T2M_MAX"]
    .rolling(
        7,
        min_periods=3,
    )
    .max()
)


weather["RH2M_7D_MEAN"] = (
    weather["RH2M"]
    .rolling(
        7,
        min_periods=3,
    )
    .mean()
)


# ---------------------------------------------------------
# Heat stress index
# ---------------------------------------------------------

# Simple temperature-humidity stress proxy.
# This is NOT claimed to be an official heat index.
weather["HEAT_STRESS"] = (
    weather["T2M"]
    + 0.05
    * weather["RH2M"]
)


# ---------------------------------------------------------
# Hot-day thresholds
# ---------------------------------------------------------

weather["HOT_DAY_30"] = (
    weather["T2M_MAX"] >= 30
).astype(int)

weather["HOT_DAY_32"] = (
    weather["T2M_MAX"] >= 32
).astype(int)

weather["HOT_DAY_35"] = (
    weather["T2M_MAX"] >= 35
).astype(int)


# ---------------------------------------------------------
# Warm-night indicator
# ---------------------------------------------------------

weather["WARM_NIGHT_24"] = (
    weather["T2M_MIN"] >= 24
).astype(int)


# ---------------------------------------------------------
# Consecutive heat persistence
# ---------------------------------------------------------

hot32 = (
    weather["T2M_MAX"] >= 32
).astype(int)

groups = (
    hot32.eq(0)
    .cumsum()
)

weather["HOT32_STREAK"] = (
    hot32
    .groupby(groups)
    .cumsum()
)


# =========================================================
# TEMPERATURE ANOMALY
# =========================================================

baseline_mean = (
    weather["T2M"]
    .mean()
)

baseline_std = (
    weather["T2M"]
    .std()
)

weather["T2M_ANOMALY"] = (
    weather["T2M"]
    - baseline_mean
)

weather["T2M_ZSCORE"] = (
    weather["T2M"]
    - baseline_mean
) / baseline_std


# =========================================================
# SAVE DAILY DATA
# =========================================================

weather.to_csv(
    OUTPUT_CSV,
    index=False,
)


# =========================================================
# SUMMARY FEATURES
# =========================================================

summary = {
    "latitude": CITY_LAT,
    "longitude": CITY_LON,
    "start_date": weather["date"].min(),
    "end_date": weather["date"].max(),
    "days": len(weather),

    "t2m_mean":
        weather["T2M"].mean(),

    "t2m_max":
        weather["T2M_MAX"].max(),

    "t2m_min":
        weather["T2M_MIN"].min(),

    "t2m_p95":
        weather["T2M_MAX"].quantile(
            0.95
        ),

    "rh2m_mean":
        weather["RH2M"].mean(),

    "rh2m_min":
        weather["RH2M"].min(),

    "rh2m_max":
        weather["RH2M"].max(),

    "heat_stress_mean":
        weather["HEAT_STRESS"].mean(),

    "heat_stress_max":
        weather["HEAT_STRESS"].max(),

    "hot_days_30":
        weather["HOT_DAY_30"].sum(),

    "hot_days_32":
        weather["HOT_DAY_32"].sum(),

    "hot_days_35":
        weather["HOT_DAY_35"].sum(),

    "warm_nights_24":
        weather["WARM_NIGHT_24"].sum(),

    "max_hot32_streak":
        weather["HOT32_STREAK"].max(),

    "temperature_anomaly_max":
        weather["T2M_ANOMALY"].max(),

    "temperature_anomaly_mean":
        weather["T2M_ANOMALY"].mean(),

    "precipitation_total_mm":
        weather["PRECTOTCORR"].sum(),

    "wind_mean":
        weather["WS2M"].mean(),

    "wind_max":
        weather["WS2M"].max(),
}


summary_df = pd.DataFrame(
    [summary]
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False,
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("WEATHER SUMMARY")
print("=" * 70)

for key, value in summary.items():

    print(
        f"{key:30s}: {value}"
    )


print("\nSaved daily data:")
print(OUTPUT_CSV)

print("\nSaved summary:")
print(SUMMARY_CSV)

print("\n" + "=" * 70)
print("NASA POWER WEATHER ACQUISITION COMPLETE")
print("=" * 70)
