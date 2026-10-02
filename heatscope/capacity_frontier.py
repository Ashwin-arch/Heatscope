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
    RESULTS / "capacity_frontier_results.csv"
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
    "tree": 1.00,
    "shade": 1.20,
    "cool_roof": 1.10,
    "cooling": 1.30,
}


BASE_CAPACITY = {
    "tree": 1000,
    "shade": 1500,
    "cool_roof": 800,
    "cooling": 800,
}


CAPACITY_LEVELS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
]


EQUITY_LEVELS = [
    None,
    0.50,
    0.60,
    0.70,
]


BUDGET_FRACTION = 0.50

TIME_LIMIT = 300

MIP_GAP = 1e-4


print("=" * 70)
print("HeatScope Capacity–Equity Frontier")
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
            index=series.index
        )

    x = x.clip(
        lo,
        hi
    )

    return (
        x - lo
    ) / (
        hi - lo
    )


# =========================================================
# COMMON NEED
# =========================================================

risk = robust_norm(
    gdf[
        "risk_exposure_lambda05"
    ]
)

structural = (
    gdf[
        "structural_vulnerability"
    ]
    .clip(
        0,
        1,
    )
)

need = (
    0.60 * risk
    +
    0.40 * structural
).clip(
    0,
    1,
)


population = robust_norm(
    np.log1p(
        gdf["population"]
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
# BENEFIT MATRIX
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
        population.to_numpy()
        *
        EFFECT[
            intervention
        ]
    )


# =========================================================
# VARIABLES
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
                        gdf.iloc[i]["cell_id"]
                    ),
                "intervention":
                    intervention,
                "intervention_idx":
                    j,
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
    "\nDecision variables:",
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
        n_variables,
    )
)


for cell, positions in (
    cell_groups.items()
):

    for pos in positions:

        cell_matrix[
            cell,
            pos
        ] = 1


cell_constraint = LinearConstraint(
    cell_matrix.tocsr(),
    lb=-np.inf,
    ub=np.ones(
        n_cells
    ),
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


cost_vector = variables[
    "cost"
].to_numpy(
    dtype=float
)


budget_constraint = LinearConstraint(
    cost_vector[
        np.newaxis,
        :
    ],
    lb=-np.inf,
    ub=budget,
)


# =========================================================
# SOLVE FUNCTION
# =========================================================

def solve(
    capacity_scale,
    equity_target,
):

    constraints = [
        cell_constraint,
        budget_constraint,
    ]


    # -----------------------------------------------------
    # CAPACITY
    # -----------------------------------------------------

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


        for pos in positions:

            capacity_matrix[
                j,
                pos
            ] = 1


    capacity_values = np.array(
        [
            min(
                BASE_CAPACITY[
                    intervention
                ]
                * capacity_scale,
                eligible[
                    intervention
                ].sum()
            )
            for intervention in INTERVENTIONS
        ],
        dtype=float,
    )


    constraints.append(
        LinearConstraint(
            capacity_matrix.tocsr(),
            lb=np.zeros(
                len(INTERVENTIONS)
            ),
            ub=capacity_values,
        )
    )


    # -----------------------------------------------------
    # EQUITY
    # -----------------------------------------------------

    if equity_target is not None:

        coeff = np.zeros(
            n_variables,
            dtype=float,
        )


        for idx, row in variables.iterrows():

            if row["equity"]:

                coeff[idx] = (
                    (1 - equity_target)
                    *
                    row["benefit"]
                )

            else:

                coeff[idx] = (
                    -equity_target
                    *
                    row["benefit"]
                )


        constraints.append(
            LinearConstraint(
                coeff[
                    np.newaxis,
                    :
                ],
                lb=0,
                ub=np.inf,
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


    return result, capacity_values


# =========================================================
# RUN FRONTIER
# =========================================================

rows = []


for scale in CAPACITY_LEVELS:

    print(
        "\n" + "=" * 70
    )

    print(
        f"CAPACITY SCALE: "
        f"{scale:.2f}"
    )


    for equity in EQUITY_LEVELS:

        label = (
            "No_Equity"
            if equity is None
            else f"Equity_{equity:.0%}"
        )


        print(
            f"\nSolving {label}"
        )


        result, capacity_values = solve(
            scale,
            equity,
        )


        status = (
            "optimal"
            if result.success
            else result.message
        )


        if result.x is None:

            print(
                "No solution."
            )

            continue


        selected = variables.loc[
            result.x > 0.5
        ].copy()


        total_benefit = selected[
            "benefit"
        ].sum()


        equity_benefit = selected.loc[
            selected["equity"],
            "benefit"
        ].sum()


        equity_share = (
            equity_benefit
            /
            total_benefit
            if total_benefit > 0
            else 0
        )


        total_cost = selected[
            "cost"
        ].sum()


        row = {
            "capacity_scale":
                scale,
            "equity_target":
                equity,
            "solver_status":
                status,
            "selected_cells":
                len(selected),
            "total_cost":
                total_cost,
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
                .eq(intervention)
                .sum()
            )


            row[
                f"{intervention}_selected"
            ] = count


            row[
                f"{intervention}_capacity"
            ] = capacity_values[
                INTERVENTIONS.index(
                    intervention
                )
            ]


        rows.append(
            row
        )


        print(
            f"Status: {status}"
        )

        print(
            f"Selected: {len(selected):,}"
        )

        print(
            f"Benefit: {total_benefit:.4f}"
        )

        print(
            f"Equity: {equity_share * 100:.2f}%"
        )

        print(
            f"Cost used: {total_cost:.2f}"
        )


# =========================================================
# SAVE
# =========================================================

results = pd.DataFrame(
    rows
)


results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\n" + "=" * 70
)

print(
    "CAPACITY–EQUITY FRONTIER"
)

print(
    "=" * 70
)

print(
    results.to_string(
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
    "\n" + "=" * 70
)

print(
    "CAPACITY FRONTIER COMPLETE"
)

print(
    "=" * 70
)
