import sys
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS


INPUT_FILE = (
    RESULTS / "heatscope_intervention_scores.gpkg"
)

OUTPUT_RESULTS = (
    RESULTS / "optimization_scenario_results.csv"
)

OUTPUT_ALLOCATIONS = (
    RESULTS / "optimization_milp_allocations.csv"
)


print("=" * 70)
print("HeatScope Budget-Constrained Intervention Optimization")
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


print(
    "\nCells:",
    f"{len(gdf):,}"
)


# =========================================================
# SCENARIO PARAMETERS
# =========================================================
#
# These are deliberately explicit scenario assumptions.
# They are NOT observed temperature reductions.
# =========================================================

INTERVENTIONS = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


SCENARIO_COST = {
    "tree": 1.0,
    "shade": 1.4,
    "cool_roof": 2.0,
    "cooling": 2.5,
}


SCENARIO_EFFECT = {
    "tree": 1.0,
    "shade": 1.2,
    "cool_roof": 1.1,
    "cooling": 1.3,
}


EQUITY_LEVELS = [
    0.50,
    0.60,
    0.70,
]


BUDGET_FRACTIONS = [
    0.25,
    0.50,
    0.75,
]


# =========================================================
# CELL BENEFIT
# =========================================================
#
# We combine:
#
#   intervention suitability
#   population affected
#   scenario intervention effect
#
# The result is a RELATIVE scenario benefit.
# =========================================================

benefit_matrix = np.zeros(
    (
        len(gdf),
        len(INTERVENTIONS),
    ),
    dtype=float,
)


cost_vector = np.zeros(
    len(INTERVENTIONS),
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

    population = np.log1p(
        gdf[
            "population"
        ].to_numpy(
            dtype=float
        )
    )


    benefit_matrix[
        :,
        j
    ] = (
        suitability
        * population
        * SCENARIO_EFFECT[
            intervention
        ]
    )


    cost_vector[j] = (
        SCENARIO_COST[
            intervention
        ]
    )


# =========================================================
# EQUITY MASK
# =========================================================

equity_mask = (
    gdf[
        "hvi_category"
    ]
    .isin(
        [
            "High",
            "Very High",
        ]
    )
    .to_numpy()
)


print(
    "\nHigh/Very High cells:",
    f"{equity_mask.sum():,}"
)


print(
    "High/Very High population:",
    f"{gdf.loc[equity_mask, 'population'].sum():,.0f}"
)


# =========================================================
# DECISION VARIABLE INDEX
# =========================================================
#
# x[i,j] = 1 when intervention j is selected for cell i.
#
# Each cell may receive at most one intervention.
# =========================================================

n_cells = len(gdf)
n_interventions = len(INTERVENTIONS)
n_variables = (
    n_cells
    * n_interventions
)


def var_index(
    cell,
    intervention,
):

    return (
        cell
        * n_interventions
        + intervention
    )


# =========================================================
# MILP OBJECTIVE
# =========================================================
#
# scipy.optimize.milp minimizes by default.
# Therefore objective = -benefit.
# =========================================================

objective = (
    -benefit_matrix
    .reshape(-1)
)


integrality = np.ones(
    n_variables,
    dtype=int
)


lower_bounds = np.zeros(
    n_variables
)

upper_bounds = np.ones(
    n_variables
)


bounds = Bounds(
    lower_bounds,
    upper_bounds,
)


# =========================================================
# BASE CONSTRAINT:
# AT MOST ONE INTERVENTION PER CELL
# =========================================================

cell_constraint = lil_matrix(
    (
        n_cells,
        n_variables,
    )
)


for i in range(
    n_cells
):

    for j in range(
        n_interventions
    ):

        cell_constraint[
            i,
            var_index(i, j)
        ] = 1


cell_constraint = (
    cell_constraint.tocsr()
)


cell_constraint_obj = LinearConstraint(
    cell_constraint,
    lb=np.full(
        n_cells,
        -np.inf
    ),
    ub=np.ones(
        n_cells
    ),
)


# =========================================================
# FULL BUDGET
# =========================================================
#
# Maximum possible budget corresponds to selecting the
# cheapest intervention in every cell.
#
# We define scenarios relative to the cost of one
# intervention per cell at average cost.
# =========================================================

average_cell_cost = np.mean(
    list(
        SCENARIO_COST.values()
    )
)

full_budget = (
    n_cells
    * average_cell_cost
)


print(
    "\nReference full budget:",
    f"{full_budget:.2f}"
)


# =========================================================
# HELPER: MILP SOLVER
# =========================================================

def solve_milp(
    budget,
    equity_fraction=None,
):

    constraints = [
        cell_constraint_obj
    ]


    # -----------------------------------------------------
    # COST CONSTRAINT
    # -----------------------------------------------------

    cost_constraint = (
        cost_vector[
            np.newaxis,
            :
        ]
    )

    full_cost = np.tile(
        cost_constraint,
        (
            n_cells,
            1,
        )
    ).reshape(-1)


    cost_row = (
        full_cost[
            np.newaxis,
            :
        ]
    )


    constraints.append(
        LinearConstraint(
            cost_row,
            lb=-np.inf,
            ub=budget,
        )
    )


    # -----------------------------------------------------
    # EQUITY CONSTRAINT
    #
    # equity_benefit >= fraction * total_benefit
    #
    # Rearranged:
    #
    # (1-f)*equity_benefit
    # - f*non_equity_benefit >= 0
    # -----------------------------------------------------

    if equity_fraction is not None:

        equity_coefficients = np.zeros(
            n_variables,
            dtype=float
        )


        for i in range(
            n_cells
        ):

            for j in range(
                n_interventions
            ):

                idx = var_index(
                    i,
                    j
                )

                b = benefit_matrix[
                    i,
                    j
                ]

                if equity_mask[i]:

                    equity_coefficients[
                        idx
                    ] = (
                        1
                        - equity_fraction
                    ) * b

                else:

                    equity_coefficients[
                        idx
                    ] = (
                        -equity_fraction
                        * b
                    )


        constraints.append(
            LinearConstraint(
                equity_coefficients[
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
            "time_limit": 120,
        },
    )


    return result


# =========================================================
# BASELINE: HVI RANKING
# =========================================================
#
# Select cell/intervention combinations according to HVI,
# choosing the highest-suitability intervention in each cell.
# =========================================================

best_intervention_idx = np.argmax(
    np.column_stack(
        [
            gdf[
                f"{i}_benefit_potential"
            ].to_numpy()
            for i in INTERVENTIONS
        ]
    ),
    axis=1,
)


# =========================================================
# GREEDY OPTIONS
# =========================================================

option_rows = []


for i in range(
    n_cells
):

    for j, intervention in enumerate(
        INTERVENTIONS
    ):

        option_rows.append(
            {
                "cell": i,
                "cell_id":
                    gdf.iloc[i]["cell_id"],
                "intervention":
                    intervention,
                "intervention_idx":
                    j,
                "cost":
                    SCENARIO_COST[
                        intervention
                    ],
                "benefit":
                    benefit_matrix[
                        i,
                        j
                    ],
                "hvi":
                    gdf.iloc[i]["hvi"],
            }
        )


options = pd.DataFrame(
    option_rows
)


options[
    "benefit_cost"
] = (
    options["benefit"]
    /
    options["cost"]
)


# =========================================================
# EVALUATION
# =========================================================

def evaluate_solution(
    selected,
    strategy,
    budget,
    equity_target=None,
):

    selected = selected.copy()


    if selected.empty:

        return {
            "strategy": strategy,
            "budget": budget,
            "equity_target":
                equity_target,
            "selected_cells": 0,
            "total_cost": 0.0,
            "total_benefit": 0.0,
            "equity_benefit": 0.0,
            "equity_benefit_share": 0.0,
        }


    total_cost = selected[
        "cost"
    ].sum()


    total_benefit = selected[
        "benefit"
    ].sum()


    equity_benefit = selected.loc[
        selected["cell"].isin(
            np.where(
                equity_mask
            )[0]
        ),
        "benefit",
    ].sum()


    share = (
        equity_benefit
        /
        total_benefit
        if total_benefit > 0
        else 0
    )


    return {
        "strategy": strategy,
        "budget": budget,
        "equity_target":
            equity_target,
        "selected_cells":
            len(selected),
        "total_cost":
            total_cost,
        "total_benefit":
            total_benefit,
        "equity_benefit":
            equity_benefit,
        "equity_benefit_share":
            share,
    }


# =========================================================
# RUN SCENARIOS
# =========================================================

scenario_results = []

milp_allocations = []


for budget_fraction in BUDGET_FRACTIONS:

    budget = (
        full_budget
        * budget_fraction
    )


    print(
        "\n" + "=" * 70
    )

    print(
        f"BUDGET SCENARIO: "
        f"{budget_fraction * 100:.0f}%"
    )

    print(
        f"Budget: {budget:.2f}"
    )


    # -----------------------------------------------------
    # RANDOM BASELINE
    # -----------------------------------------------------

    rng = np.random.default_rng(
        int(
            budget_fraction * 1000
        )
    )


    random_options = options.sample(
        frac=1.0,
        random_state=int(
            budget_fraction * 1000
        ),
    ).copy()


    random_options = (
        random_options
        .sort_values(
            "cell"
        )
    )


    selected_cells = set()

    selected_rows = []

    remaining_budget = budget


    for _, row in random_options.iterrows():

        if row["cell"] in selected_cells:
            continue

        if row["cost"] <= remaining_budget:

            selected_rows.append(
                row
            )

            selected_cells.add(
                row["cell"]
            )

            remaining_budget -= (
                row["cost"]
            )


    random_selected = pd.DataFrame(
        selected_rows
    )


    scenario_results.append(
        evaluate_solution(
            random_selected,
            "Random",
            budget,
            None,
        )
    )


    # -----------------------------------------------------
    # HVI RANKING
    # -----------------------------------------------------

    hvi_options = options[
        options[
            "intervention_idx"
        ].to_numpy()
        ==
        best_intervention_idx[
            options["cell"].to_numpy()
        ]
    ].copy()


    hvi_options = (
        hvi_options
        .sort_values(
            [
                "hvi",
                "benefit_cost",
            ],
            ascending=False,
        )
    )


    remaining_budget = budget
    selected_cells = set()
    selected_rows = []


    for _, row in hvi_options.iterrows():

        if row["cost"] <= remaining_budget:

            selected_rows.append(
                row
            )

            selected_cells.add(
                row["cell"]
            )

            remaining_budget -= (
                row["cost"]
            )


    hvi_selected = pd.DataFrame(
        selected_rows
    )


    scenario_results.append(
        evaluate_solution(
            hvi_selected,
            "HVI_Ranking",
            budget,
            None,
        )
    )


    # -----------------------------------------------------
    # GREEDY
    # -----------------------------------------------------

    greedy_options = (
        options
        .sort_values(
            "benefit_cost",
            ascending=False,
        )
    )


    remaining_budget = budget
    selected_cells = set()
    selected_rows = []


    for _, row in greedy_options.iterrows():

        if row["cell"] in selected_cells:
            continue

        if row["cost"] <= remaining_budget:

            selected_rows.append(
                row
            )

            selected_cells.add(
                row["cell"]
            )

            remaining_budget -= (
                row["cost"]
            )


    greedy_selected = pd.DataFrame(
        selected_rows
    )


    scenario_results.append(
        evaluate_solution(
            greedy_selected,
            "Greedy",
            budget,
            None,
        )
    )


    # -----------------------------------------------------
    # MILP WITHOUT EQUITY
    # -----------------------------------------------------

    result = solve_milp(
        budget=budget,
        equity_fraction=None,
    )


    if result.success:

        x = result.x.reshape(
            n_cells,
            n_interventions,
        )

        selected = []


        for i in range(
            n_cells
        ):

            for j in range(
                n_interventions
            ):

                if x[i, j] > 0.5:

                    selected.append(
                        {
                            "cell":
                                i,
                            "cell_id":
                                gdf.iloc[i]["cell_id"],
                            "intervention":
                                INTERVENTIONS[j],
                            "intervention_idx":
                                j,
                            "cost":
                                SCENARIO_COST[
                                    INTERVENTIONS[j]
                                ],
                            "benefit":
                                benefit_matrix[
                                    i,
                                    j
                                ],
                        }
                    )


        milp_selected = pd.DataFrame(
            selected
        )


        scenario_results.append(
            evaluate_solution(
                milp_selected,
                "MILP_No_Equity",
                budget,
                None,
            )
        )


        temp = milp_selected.copy()

        temp[
            "budget_fraction"
        ] = budget_fraction

        temp[
            "equity_target"
        ] = np.nan

        temp[
            "strategy"
        ] = "MILP_No_Equity"

        milp_allocations.append(
            temp
        )


    else:

        print(
            "MILP without equity failed:",
            result.message
        )


    # -----------------------------------------------------
    # MILP WITH EQUITY
    # -----------------------------------------------------

    for equity_fraction in EQUITY_LEVELS:

        result = solve_milp(
            budget=budget,
            equity_fraction=
                equity_fraction,
        )


        if result.success:

            x = result.x.reshape(
                n_cells,
                n_interventions,
            )

            selected = []


            for i in range(
                n_cells
            ):

                for j in range(
                    n_interventions
                ):

                    if x[i, j] > 0.5:

                        selected.append(
                            {
                                "cell":
                                    i,
                                "cell_id":
                                    gdf.iloc[i]["cell_id"],
                                "intervention":
                                    INTERVENTIONS[j],
                                "intervention_idx":
                                    j,
                                "cost":
                                    SCENARIO_COST[
                                        INTERVENTIONS[j]
                                    ],
                                "benefit":
                                    benefit_matrix[
                                        i,
                                        j
                                    ],
                            }
                        )


            equity_selected = pd.DataFrame(
                selected
            )


            scenario_results.append(
                evaluate_solution(
                    equity_selected,
                    "MILP_Equity",
                    budget,
                    equity_fraction,
                )
            )


            temp = equity_selected.copy()

            temp[
                "budget_fraction"
            ] = budget_fraction

            temp[
                "equity_target"
            ] = equity_fraction

            temp[
                "strategy"
            ] = "MILP_Equity"

            milp_allocations.append(
                temp
            )


        else:

            print(
                f"MILP equity={equity_fraction:.2f} "
                f"failed: {result.message}"
            )


results = pd.DataFrame(
    scenario_results
)


allocations = pd.concat(
    milp_allocations,
    ignore_index=True,
)


# =========================================================
# PRINT RESULTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "OPTIMIZATION RESULTS"
)

print(
    "=" * 70
)


print(
    results[
        [
            "strategy",
            "budget",
            "equity_target",
            "selected_cells",
            "total_cost",
            "total_benefit",
            "equity_benefit_share",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# =========================================================
# INTERVENTION MIX
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "MILP INTERVENTION MIX"
)

print(
    "=" * 70
)


for strategy in [
    "MILP_No_Equity",
    "MILP_Equity",
]:

    d = allocations[
        allocations["strategy"] == strategy
    ]


    if d.empty:
        continue


    print(
        f"\n{strategy}:"
    )

    print(
        d[
            [
                "budget_fraction",
                "equity_target",
                "intervention",
            ]
        ]
        .groupby(
            [
                "budget_fraction",
                "equity_target",
                "intervention",
            ]
        )
        .size()
        .to_string()
    )


# =========================================================
# SAVE
# =========================================================

results.to_csv(
    OUTPUT_RESULTS,
    index=False,
)


allocations.to_csv(
    OUTPUT_ALLOCATIONS,
    index=False,
)


print(
    "\nSaved:"
)

print(
    OUTPUT_RESULTS
)

print(
    OUTPUT_ALLOCATIONS
)


print(
    "\n" + "=" * 70
)

print(
    "OPTIMIZATION COMPLETE"
)

print(
    "=" * 70
)
