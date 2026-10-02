import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.optimize import (
    milp,
    LinearConstraint,
    Bounds,
)
from scipy.sparse import lil_matrix

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = (
    RESULTS
    / "heatscope_intervention_scores.gpkg"
)


INTERVENTIONS = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


COST = {
    "tree": 1.0,
    "shade": 1.4,
    "cool_roof": 2.0,
    "cooling": 2.5,
}


EFFECT = {
    "tree": 1.0,
    "shade": 1.2,
    "cool_roof": 1.1,
    "cooling": 1.3,
}


BUDGET_FRACTION = 0.50

EQUITY_LEVELS = [
    0.60,
    0.70,
]

TIME_LIMIT = 600

MIP_GAP = 1e-4


print("=" * 70)
print("HeatScope Optimization Refinement")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="intervention_scores",
)

gdf = gdf[
    gdf["hvi"].notna()
].copy()

gdf = gdf.reset_index(
    drop=True
)

n_cells = len(gdf)


print(
    "\nCells:",
    f"{n_cells:,}"
)


# =========================================================
# BENEFITS
# =========================================================

population_factor = np.log1p(
    gdf["population"].to_numpy(
        dtype=float
    )
)


benefit_matrix = np.zeros(
    (
        n_cells,
        len(INTERVENTIONS),
    ),
    dtype=float,
)


for j, intervention in enumerate(
    INTERVENTIONS
):

    suitability = gdf[
        f"{intervention}_suitability"
    ].to_numpy(
        dtype=float
    )

    benefit_matrix[:, j] = (
        suitability
        * population_factor
        * EFFECT[intervention]
    )


# =========================================================
# EQUITY CELLS
# =========================================================

equity_mask = (
    gdf["hvi_category"]
    .isin(
        [
            "High",
            "Very High",
        ]
    )
    .to_numpy()
)


# =========================================================
# DOMINANCE PRUNING
# =========================================================

active_pairs = []

for i in range(
    n_cells
):

    active = []

    for j in range(
        len(INTERVENTIONS)
    ):

        dominated = False

        for k in range(
            len(INTERVENTIONS)
        ):

            if j == k:
                continue

            cheaper_or_equal = (
                COST[
                    INTERVENTIONS[k]
                ]
                <=
                COST[
                    INTERVENTIONS[j]
                ]
            )

            better_or_equal = (
                benefit_matrix[i, k]
                >=
                benefit_matrix[i, j]
            )

            strictly_better = (
                COST[
                    INTERVENTIONS[k]
                ]
                <
                COST[
                    INTERVENTIONS[j]
                ]
                or
                benefit_matrix[i, k]
                >
                benefit_matrix[i, j]
            )

            if (
                cheaper_or_equal
                and better_or_equal
                and strictly_better
            ):

                dominated = True
                break


        if not dominated:

            active.append(j)


    active_pairs.append(
        active
    )


# =========================================================
# VARIABLES
# =========================================================

records = []


for i in range(
    n_cells
):

    for j in active_pairs[i]:

        intervention = INTERVENTIONS[j]

        records.append(
            {
                "cell": i,
                "cell_id":
                    int(
                        gdf.iloc[i]["cell_id"]
                    ),
                "intervention":
                    intervention,
                "cost":
                    COST[intervention],
                "benefit":
                    benefit_matrix[i, j],
                "equity":
                    bool(equity_mask[i]),
            }
        )


variables = pd.DataFrame(
    records
)


n_variables = len(
    variables
)


print(
    "Optimization variables:",
    f"{n_variables:,}"
)


# =========================================================
# OBJECTIVE
# =========================================================

objective = -variables[
    "benefit"
].to_numpy(
    dtype=float
)


integrality = np.ones(
    n_variables,
    dtype=int
)


bounds = Bounds(
    np.zeros(n_variables),
    np.ones(n_variables),
)


# =========================================================
# CELL CONSTRAINT
# =========================================================

cell_groups = (
    variables
    .groupby("cell")
    .groups
)


cell_matrix = lil_matrix(
    (
        n_cells,
        n_variables,
    )
)


for cell, positions in (
    cell_groups.items()
):

    for position in positions:

        cell_matrix[
            cell,
            position
        ] = 1


cell_constraint = LinearConstraint(
    cell_matrix.tocsr(),
    lb=-np.inf,
    ub=np.ones(n_cells),
)


# =========================================================
# BUDGET
# =========================================================

average_cost = np.mean(
    list(
        COST.values()
    )
)


reference_budget = (
    n_cells
    * average_cost
)


budget = (
    reference_budget
    * BUDGET_FRACTION
)


print(
    "Budget:",
    f"{budget:.4f}"
)


cost_vector = variables[
    "cost"
].to_numpy(
    dtype=float
)


cost_constraint = LinearConstraint(
    cost_vector[
        np.newaxis,
        :,
    ],
    lb=-np.inf,
    ub=budget,
)


# =========================================================
# SOLVE
# =========================================================

results = []


for alpha in EQUITY_LEVELS:

    print(
        "\n" + "-" * 70
    )

    print(
        f"Solving equity target: "
        f"{alpha:.0%}"
    )

    coefficient = np.zeros(
        n_variables,
        dtype=float
    )


    for idx, row in variables.iterrows():

        b = row["benefit"]

        if row["equity"]:

            coefficient[idx] = (
                (1 - alpha)
                * b
            )

        else:

            coefficient[idx] = (
                -alpha
                * b
            )


    equity_constraint = LinearConstraint(
        coefficient[
            np.newaxis,
            :,
        ],
        lb=0,
        ub=np.inf,
    )


    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[
            cell_constraint,
            cost_constraint,
            equity_constraint,
        ],
        options={
            "time_limit":
                TIME_LIMIT,
            "mip_rel_gap":
                MIP_GAP,
        },
    )


    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    print(
        "Success:",
        result.success
    )

    print(
        "Status:",
        result.message
    )


    if result.x is None:

        print(
            "No feasible solution returned."
        )

        continue


    x = result.x


    selected_mask = (
        x > 0.5
    )


    selected = variables.loc[
        selected_mask
    ].copy()


    total_benefit = selected[
        "benefit"
    ].sum()


    total_cost = selected[
        "cost"
    ].sum()


    equity_benefit = selected.loc[
        selected["equity"],
        "benefit",
    ].sum()


    equity_share = (
        equity_benefit
        /
        total_benefit
        if total_benefit > 0
        else 0
    )


    # -----------------------------------------------------
    # MIP GAP
    # -----------------------------------------------------

    dual_bound = getattr(
        result,
        "mip_dual_bound",
        np.nan,
    )


    objective_value = (
        result.fun
    )


    if np.isfinite(
        dual_bound
    ):

        primal = abs(
            objective_value
        )

        dual = abs(
            dual_bound
        )

        if primal > 0:

            gap = (
                abs(
                    primal - dual
                )
                /
                primal
            )

        else:

            gap = 0.0

    else:

        gap = np.nan


    results.append(
        {
            "budget_fraction":
                BUDGET_FRACTION,
            "equity_target":
                alpha,
            "solver_success":
                result.success,
            "solver_status":
                result.message,
            "selected_cells":
                len(selected),
            "total_cost":
                total_cost,
            "total_benefit":
                total_benefit,
            "equity_benefit":
                equity_benefit,
            "equity_benefit_share":
                equity_share,
            "mip_dual_bound":
                dual_bound,
            "relative_mip_gap":
                gap,
        }
    )


    print(
        "\nSelected cells:",
        f"{len(selected):,}"
    )

    print(
        "Total cost:",
        f"{total_cost:.4f}"
    )

    print(
        "Total benefit:",
        f"{total_benefit:.4f}"
    )

    print(
        "Equity share:",
        f"{equity_share * 100:.4f}%"
    )

    print(
        "Dual bound:",
        dual_bound
    )

    print(
        "Relative MIP gap:",
        gap
    )


# =========================================================
# SAVE
# =========================================================

output = pd.DataFrame(
    results
)


output_file = (
    RESULTS
    / "optimization_refined_50pct.csv"
)


output.to_csv(
    output_file,
    index=False,
)


print(
    "\nSaved:"
)

print(
    output_file
)


print(
    "\n" + "=" * 70
)

print(
    "OPTIMIZATION REFINEMENT COMPLETE"
)

print(
    "=" * 70
)
