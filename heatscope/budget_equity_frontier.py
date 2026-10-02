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
    RESULTS / "heatscope_intervention_eligibility.gpkg"
)

OUTPUT_FILE = (
    RESULTS / "budget_equity_frontier.csv"
)

OUTPUT_ALLOCATION = (
    RESULTS / "budget_equity_allocations.csv"
)


# =========================================================
# INTERVENTIONS
# =========================================================

INTERVENTIONS = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


# =========================================================
# SCENARIO COSTS / EFFECTS
# =========================================================
#
# Scenario parameters only.
# They are NOT measured intervention outcomes.
# =========================================================

COST = {
    "tree": 1.0,
    "shade": 1.4,
    "cool_roof": 2.0,
    "cooling": 2.5,
}


EFFECT = {
    "tree": 1.00,
    "shade": 1.20,
    "cool_roof": 1.10,
    "cooling": 1.30,
}


# =========================================================
# FIXED IMPLEMENTATION CAPACITY
# =========================================================

CAPACITY = {
    "tree": 1000,
    "shade": 1500,
    "cool_roof": 800,
    "cooling": 800,
}


BUDGET_LEVELS = [
    0.25,
    0.50,
    0.75,
    1.00,
]


EQUITY_LEVELS = [
    None,
    0.50,
    0.60,
    0.70,
    0.80,
]


TIME_LIMIT = 300

MIP_GAP = 1e-4


print("=" * 70)
print("HeatScope Budget–Equity Frontier")
print("=" * 70)


# =========================================================
# LOAD
# =========================================================

gdf = gpd.read_file(
    INPUT_FILE,
    layer="eligibility",
)

gdf = gdf[
    gdf["predicted_lst_p95"].notna()
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
# NORMALIZATION
# =========================================================

def robust_norm(series):

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    lo = x.quantile(0.01)
    hi = x.quantile(0.99)

    if hi <= lo:
        return pd.Series(
            0.5,
            index=series.index,
        )

    x = x.clip(
        lo,
        hi,
    )

    return (
        x - lo
    ) / (
        hi - lo
    )


# =========================================================
# NEED
# =========================================================

risk = robust_norm(
    gdf[
        "risk_exposure_lambda05"
    ]
)

structural = gdf[
    "structural_vulnerability"
].clip(
    0,
    1,
)

need = (
    0.60 * risk
    +
    0.40 * structural
).clip(
    0,
    1,
)


population_factor = robust_norm(
    np.log1p(
        gdf[
            "population"
        ]
    )
)


# =========================================================
# OPPORTUNITY
# =========================================================

opportunity = {

    intervention:
        gdf[
            f"{intervention}_opportunity"
        ].clip(
            0,
            1,
        )
    for intervention in INTERVENTIONS
}


# =========================================================
# ELIGIBILITY
# =========================================================

eligible = {

    intervention:
        gdf[
            f"{intervention}_eligible"
        ].astype(bool)
    for intervention in INTERVENTIONS
}


# =========================================================
# BENEFIT
# =========================================================

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

    benefit_matrix[
        :,
        j
    ] = (
        need.to_numpy()
        *
        opportunity[
            intervention
        ].to_numpy()
        *
        population_factor.to_numpy()
        *
        EFFECT[
            intervention
        ]
    )


# =========================================================
# BUILD DECISION VARIABLES
# =========================================================

records = []


for i in range(
    n_cells
):

    for j, intervention in enumerate(
        INTERVENTIONS
    ):

        if not eligible[
            intervention
        ].iloc[i]:

            continue


        b = benefit_matrix[
            i,
            j
        ]


        if b <= 0:
            continue


        records.append(
            {
                "cell": i,
                "cell_id":
                    int(
                        gdf.iloc[i][
                            "cell_id"
                        ]
                    ),
                "intervention":
                    intervention,
                "cost":
                    COST[
                        intervention
                    ],
                "benefit":
                    b,
                "equity":
                    bool(
                        gdf.iloc[i][
                            "equity_priority"
                        ]
                    ),
            }
        )


variables = pd.DataFrame(
    records
)

n_variables = len(
    variables
)


print(
    "Decision variables:",
    f"{n_variables:,}"
)


# =========================================================
# MAXIMUM IMPLEMENTABLE COST
# =========================================================

maximum_capacity_cost = sum(
    CAPACITY[i] * COST[i]
    for i in INTERVENTIONS
)


print(
    "\nMaximum capacity cost:",
    f"{maximum_capacity_cost:.2f}"
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
    np.zeros(
        n_variables
    ),
    np.ones(
        n_variables
    ),
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
        n_variables
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
    ub=np.ones(
        n_cells
    ),
)


# =========================================================
# CAPACITY CONSTRAINT
# =========================================================

capacity_matrix = lil_matrix(
    (
        len(INTERVENTIONS),
        n_variables,
    )
)


for j, intervention in enumerate(
    INTERVENTIONS
):

    positions = variables.index[
        variables[
            "intervention"
        ].eq(intervention)
    ]


    for position in positions:

        capacity_matrix[
            j,
            position
        ] = 1


capacity_constraint = LinearConstraint(
    capacity_matrix.tocsr(),
    lb=np.zeros(
        len(INTERVENTIONS)
    ),
    ub=np.array(
        [
            CAPACITY[i]
            for i in INTERVENTIONS
        ],
        dtype=float,
    ),
)


# =========================================================
# COST VECTOR
# =========================================================

cost_vector = variables[
    "cost"
].to_numpy(
    dtype=float
)


# =========================================================
# EQUITY CONSTRAINT
# =========================================================

def make_equity_constraint(
    alpha
):

    coefficient = np.zeros(
        n_variables
    )


    for idx, row in variables.iterrows():

        if row["equity"]:

            coefficient[idx] = (
                (1 - alpha)
                *
                row["benefit"]
            )

        else:

            coefficient[idx] = (
                -alpha
                *
                row["benefit"]
            )


    return LinearConstraint(
        coefficient[
            np.newaxis,
            :
        ],
        lb=0,
        ub=np.inf,
    )


# =========================================================
# SOLVER
# =========================================================

def solve(
    budget,
    equity_target,
):

    budget_constraint = LinearConstraint(
        cost_vector[
            np.newaxis,
            :
        ],
        lb=-np.inf,
        ub=budget,
    )


    constraints = [
        cell_constraint,
        capacity_constraint,
        budget_constraint,
    ]


    if equity_target is not None:

        constraints.append(
            make_equity_constraint(
                equity_target
            )
        )


    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit":
                TIME_LIMIT,
            "mip_rel_gap":
                MIP_GAP,
        },
    )


    return result


# =========================================================
# RUN
# =========================================================

results = []

allocations = []


for budget_fraction in BUDGET_LEVELS:

    budget = (
        maximum_capacity_cost
        *
        budget_fraction
    )


    print(
        "\n" + "=" * 70
    )

    print(
        f"BUDGET = {budget_fraction:.0%}"
    )

    print(
        f"Amount = {budget:.2f}"
    )


    for equity_target in EQUITY_LEVELS:

        name = (
            "No_Equity"
            if equity_target is None
            else f"Equity_{equity_target:.0%}"
        )


        print(
            f"\nSolving {name}"
        )


        result = solve(
            budget,
            equity_target,
        )


        if result.x is None:

            print(
                "NO SOLUTION"
            )

            continue


        selected = variables.loc[
            result.x > 0.5
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


        # -------------------------------------------------
        # CAPACITY USAGE
        # -------------------------------------------------

        row = {

            "budget_fraction":
                budget_fraction,

            "budget":
                budget,

            "equity_target":
                equity_target,

            "solver_status":
                (
                    "optimal"
                    if result.success
                    else result.message
                ),

            "selected_cells":
                len(selected),

            "total_cost":
                total_cost,

            "budget_utilization":
                (
                    total_cost
                    /
                    budget
                ),

            "total_benefit":
                total_benefit,

            "equity_benefit_share":
                equity_share,
        }


        for intervention in INTERVENTIONS:

            count = (
                selected[
                    "intervention"
                ]
                .eq(
                    intervention
                )
                .sum()
            )


            row[
                f"{intervention}_selected"
            ] = count


            row[
                f"{intervention}_capacity"
            ] = CAPACITY[
                intervention
            ]


            row[
                f"{intervention}_capacity_usage"
            ] = (
                count
                /
                CAPACITY[
                    intervention
                ]
            )


        results.append(
            row
        )


        if not selected.empty:

            selected = selected.copy()

            selected[
                "budget_fraction"
            ] = budget_fraction

            selected[
                "budget"
            ] = budget

            selected[
                "equity_target"
            ] = (
                np.nan
                if equity_target is None
                else equity_target
            )

            selected[
                "strategy"
            ] = name

            allocations.append(
                selected
            )


        print(
            "Status:",
            row["solver_status"]
        )

        print(
            "Selected:",
            len(selected)
        )

        print(
            "Cost:",
            f"{total_cost:.2f}"
        )

        print(
            "Benefit:",
            f"{total_benefit:.4f}"
        )

        print(
            "Equity:",
            f"{equity_share * 100:.2f}%"
        )


# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(
    results
)


allocations_df = pd.concat(
    allocations,
    ignore_index=True,
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


allocations_df.to_csv(
    OUTPUT_ALLOCATION,
    index=False,
)


print(
    "\n" + "=" * 70
)

print(
    "BUDGET–EQUITY FRONTIER"
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


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    OUTPUT_ALLOCATION
)


print(
    "\n" + "=" * 70
)

print(
    "BUDGET–EQUITY FRONTIER COMPLETE"
)

print(
    "=" * 70
)
