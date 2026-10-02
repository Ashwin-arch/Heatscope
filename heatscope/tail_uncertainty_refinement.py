import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = (
    RESULTS / "context_spatial_ml_predictions.csv"
)

OUTPUT_FILE = (
    RESULTS / "tail_uncertainty_results.csv"
)


TARGET = "actual_lst_p95"

PREDICTION = "predicted_lst_p95"

FOLD = "fold"


print("=" * 70)
print("HeatScope Tail-Aware Uncertainty Refinement")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

pred = pd.read_csv(
    INPUT_FILE
)

pred = pred[
    pred["feature_set"].eq("Context_9")
].copy()


if pred.empty:
    raise RuntimeError(
        "Context_9 predictions not found."
    )


print(
    "\nCells:",
    f"{len(pred):,}"
)


# =========================================================
# BASIC VARIABLES
# =========================================================

pred["abs_error"] = np.abs(
    pred[TARGET]
    - pred[PREDICTION]
)


# =========================================================
# GLOBAL BASELINE
# =========================================================

global_q95 = np.quantile(
    pred["abs_error"],
    0.95
)


print(
    "\nGlobal 95% residual threshold:",
    f"{global_q95:.4f} °C"
)


# =========================================================
# FUNCTION
# =========================================================

def evaluate_intervals(
    lower,
    upper,
    name,
    frame,
):

    actual = frame[TARGET]

    covered = (
        (actual >= lower)
        &
        (actual <= upper)
    )

    coverage = covered.mean()

    width = upper - lower

    q5 = frame[
        frame["target_quantile"] == "Q5"
    ]

    if len(q5) > 0:

        q5_coverage = (
            (
                (q5[TARGET] >= lower.loc[q5.index])
                &
                (q5[TARGET] <= upper.loc[q5.index])
            ).mean()
        )

    else:

        q5_coverage = np.nan


    return {
        "strategy": name,
        "overall_coverage_95":
            coverage,
        "q5_coverage_95":
            q5_coverage,
        "mean_interval_width":
            width.mean(),
        "median_interval_width":
            width.median(),
    }


# =========================================================
# QUINTILES
# =========================================================

pred[
    "target_quantile"
] = pd.qcut(
    pred[TARGET],
    q=5,
    labels=[
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ],
    duplicates="drop",
)


# =========================================================
# STRATEGY A
# GLOBAL
# =========================================================

base_lower = (
    pred[PREDICTION]
    - global_q95
)

base_upper = (
    pred[PREDICTION]
    + global_q95
)


results = []


results.append(
    evaluate_intervals(
        base_lower,
        base_upper,
        "Global_95",
        pred,
    )
)


# =========================================================
# STRATEGY B
# TARGET-REGIME CALIBRATION
# =========================================================
#
# Estimate 95% absolute error separately for each target
# quintile using out-of-fold errors.
#
# This is an exploratory tail-aware calibration, not a
# formal conformal guarantee.
# =========================================================

regime_thresholds = (
    pred
    .groupby(
        "target_quantile",
        observed=True,
    )["abs_error"]
    .quantile(0.95)
)


print(
    "\nTarget-regime 95% thresholds:"
)

for q, threshold in regime_thresholds.items():

    print(
        f"{q}: {threshold:.4f} °C"
    )


regime_width = pred[
    "target_quantile"
].map(
    regime_thresholds
).astype(float)


regime_lower = (
    pred[PREDICTION]
    - regime_width
)

regime_upper = (
    pred[PREDICTION]
    + regime_width
)


results.append(
    evaluate_intervals(
        regime_lower,
        regime_upper,
        "Target_Regime_95",
        pred,
    )
)


# =========================================================
# STRATEGY C
# PREDICTION-REGIME CALIBRATION
# =========================================================
#
# This is more operationally realistic because the actual
# target value is unavailable at deployment.
#
# We therefore create predicted-temperature quintiles.
# =========================================================

pred[
    "prediction_quantile"
] = pd.qcut(
    pred[PREDICTION],
    q=5,
    labels=[
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ],
    duplicates="drop",
)


prediction_thresholds = (
    pred
    .groupby(
        "prediction_quantile",
        observed=True,
    )["abs_error"]
    .quantile(0.95)
)


print(
    "\nPrediction-regime 95% thresholds:"
)

for q, threshold in prediction_thresholds.items():

    print(
        f"{q}: {threshold:.4f} °C"
    )


prediction_width = pred[
    "prediction_quantile"
].map(
    prediction_thresholds
).astype(float)


prediction_lower = (
    pred[PREDICTION]
    - prediction_width
)

prediction_upper = (
    pred[PREDICTION]
    + prediction_width
)


results.append(
    evaluate_intervals(
        prediction_lower,
        prediction_upper,
        "Prediction_Regime_95",
        pred,
    )
)


# =========================================================
# PRINT RESULTS
# =========================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n" + "=" * 70
)

print(
    "TAIL-AWARE UNCERTAINTY COMPARISON"
)

print(
    "=" * 70
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# TEST TARGET-REGIME COVERAGE DIRECTLY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "TARGET-REGIME COVERAGE"
)

print(
    "=" * 70
)


for strategy_name, lower, upper in [

    (
        "Global_95",
        base_lower,
        base_upper,
    ),

    (
        "Target_Regime_95",
        regime_lower,
        regime_upper,
    ),

    (
        "Prediction_Regime_95",
        prediction_lower,
        prediction_upper,
    ),

]:

    print(
        f"\n{strategy_name}"
    )

    for q in [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ]:

        idx = (
            pred["target_quantile"] == q
        )

        cov = (
            (
                (pred.loc[idx, TARGET] >= lower.loc[idx])
                &
                (pred.loc[idx, TARGET] <= upper.loc[idx])
            ).mean()
        )

        print(
            f"  {q}: {cov * 100:.2f}%"
        )


# =========================================================
# SAVE
# =========================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\n" + "=" * 70
)

print(
    "TAIL-AWARE REFINEMENT COMPLETE"
)

print(
    "=" * 70
)
