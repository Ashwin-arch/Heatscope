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
    RESULTS
    / "heatscope_intervention_scores.gpkg"
)

OUTPUT_RESULTS = (
    RESULTS
    / "optimization_v2_results.csv"
)

OUTPUT_ALLOCATIONS = (
    RESULTS
    / "optimization_v2_allocations.csv"
)


# =========================================================
# SCENARIO
# =========================================================

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


BUDGET_FRACTIONS = [
    0.50,
    0.75,
]


EQUITY_LEVELS = [
    None,
    0.50,
    0.60,
    0.70,
]


SOLVER_TIME_LIMIT = 300


print("=" * 70)
print("HeatScope Optimization V2")
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
# BENEFIT MATRIX
# =========================================================
#
# Relative scenario benefit:
#
# suitability × log(population) × intervention effect
#
# This is NOT a measured temperature reduction.
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


    benefit_matrix[
        :,
        j
    ] = (
        suitability
        * population_factor
        * EFFECT[
            intervention
        ]
    )


# =========================================================
# DOMINANCE PRUNING
# =========================================================
#
# For a single cell:
#
# intervention A dominates B when:
#
#   cost(A) <= cost(B)
#   AND
#   benefit(A) >= benefit(B)
#
# with at least one strict inequality.
#
# A dominated intervention can never be optimal.
# =========================================================

active_pairs = []

pruned_counts = {
    intervention: 0
    for intervention in INTERVENTIONS
}


for i in range(
    n_cells
):

    active = []

    for j, intervention_j in enumerate(
        INTERVENTIONS
    ):

        dominated = False

        for k, intervention_k in enumerate(
            INTERVENTIONS
        ):

            if j == k:
                continue

            cheaper_or_equal = (
                COST[intervention_k]
                <= COST[intervention_j]
            )

            better_or_equal = (
                benefit_matrix[i, k]
                >= benefit_matrix[i, j]
            )

            strictly_better = (
                COST[intervention_k]
                < COST[intervention_j]
                or
                benefit_matrix[i, k]
                > benefit_matrix[i, j]
            )

            if (
                cheaper_or_equal
                and better_or_equal
                and strictly_better
            ):

                dominated = True
                break


        if dominated:

            pruned_counts[
                intervention_j
            ] += 1

        else:

            active.append(j)


    active_pairs.append(
        active
    )


print(
    "\nDominated choices removed:"
)

for intervention in INTERVENTIONS:

    print(
        f"{intervention:12s}: "
        f"{pruned_counts[intervention]:,}"
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
# REFERENCE BUDGET
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


print(
    "\nReference budget:",
    f"{reference_budget:.4f}"
)


# =========================================================
# BUILD REDUCED VARIABLES
# =========================================================

variable_records = []

for i in range(
    n_cells
):

    for j in active_pairs[i]:

        intervention = INTERVENTIONS[j]

        variable_records.append(
            {
                "cell": i,
                "cell_id":
                    int(
                        gdf.iloc[i]["cell_id"]
                    ),
                "j": j,
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
    variable_records
)

n_variables = len(
    variables
)


print(
    "\nOriginal variables:",
    f"{n_cells * len(INTERVENTIONS):,}"
)

print(
    "Reduced variables:",
    f"{n_variables:,}"
)

print(
    "Reduction:",
    f"{(1 - n_variables / (n_cells * len(INTERVENTIONS))) * 100:.2f}%"
)


# =========================================================
# VARIABLE INDEX
# =========================================================

variable_index = {
    row.Index: idx
    for idx, row in enumerate(
        variables.itertuples()
    )
}


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
# ONE-INTERVENTION-PER-CELL
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


for cell, rows in cell_groups.items():

    for row_position in rows:

        idx = variable_index[
            row_position
        ]

        cell_matrix[
            cell,
            idx
        ] = 1


cell_constraint = LinearConstraint(
    cell_matrix.tocsr(),
    lb=-np.inf,
    ub=np.ones(n_cells),
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
# EQUITY COEFFICIENT
# =========================================================
#
# We want:
#
# equity benefit >= alpha * total benefit
#
# equivalent to:
#
# (1-alpha)*equity - alpha*non_equity >= 0
# =========================================================

def make_equity_constraint(
    alpha
):

    coefficient = np.zeros(
        n_variables,
        dtype=float
    )


    for idx, row in variables.iterrows():

        if row["equity"]:

            coefficient[idx] = (
                (1 - alpha)
                * row["benefit"]
            )

        else:

            coefficient[idx] = (
                -alpha
                * row["benefit"]
            )


    return LinearConstraint(
        coefficient[
            np.newaxis,
            :,
        ],
        lb=0,
        ub=np.inf,
    )


# =========================================================
# SOLVER
# =========================================================

def solve(
    budget,
    equity_alpha=None,
):

    constraints = [
        cell_constraint
    ]


    constraints.append(
        LinearConstraint(
            cost_vector[
                np.newaxis,
                :,
            ],
            lb=-np.inf,
            ub=budget,
        )
    )


    if equity_alpha is not None:

        constraints.append(
            make_equity_constraint(
                equity_alpha
            )
        )


    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit":
                SOLVER_TIME_LIMIT,
            "mip_rel_gap":
                1e-5,
        },
    )


    return result


# =========================================================
# DECODE
# =========================================================

def decode_solution(
    result
):

    if result.x is None:

        return pd.DataFrame(
            columns=[
                "cell",
                "cell_id",
                "intervention",
                "cost",
                "benefit",
                "equity",
            ]
        )


    selected = []

    x = result.x

    for idx, value in enumerate(x):

        if value > 0.5:

            row = variables.iloc[idx]

            selected.append(
                {
                    "cell":
                        int(row["cell"]),
                    "cell_id":
                        int(row["cell_id"]),
                    "intervention":
                        row["intervention"],
                    "cost":
                        row["cost"],
                    "benefit":
                        row["benefit"],
                    "equity":
                        bool(row["equity"]),
                }
            )


    return pd.DataFrame(
        selected
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    selected,
    strategy,
    budget,
    equity_target=None,
    status=None,
    objective_bound=None,
):

    if selected.empty:

        return {
            "strategy":
                strategy,
            "budget":
                budget,
            "equity_target":
                equity_target,
            "solver_status":
                status,
            "selected_cells":
                0,
            "total_cost":
                0.0,
            "total_benefit":
                0.0,
            "equity_benefit":
                0.0,
            "equity_benefit_share":
                0.0,
            "objective_bound":
                objective_bound,
        }


    total_benefit = selected[
        "benefit"
    ].sum()

    equity_benefit = selected.loc[
        selected["equity"],
        "benefit",
    ].sum()

    share = (
        equity_benefit
        / total_benefit
        if total_benefit > 0
        else 0
    )


    return {
        "strategy":
            strategy,
        "budget":
            budget,
        "equity_target":
            equity_target,
        "solver_status":
            status,
        "selected_cells":
            len(selected),
        "total_cost":
            selected["cost"].sum(),
        "total_benefit":
            total_benefit,
        "equity_benefit":
            equity_benefit,
        "equity_benefit_share":
            share,
        "objective_bound":
            objective_bound,
    }


# =========================================================
# SIMPLE BASELINES
# =========================================================

all_options = variables.copy()

hvi_options = (
    all_options
    .sort_values(
        [
            "cell",
        ]
    )
)

# Keep best intervention per cell for HVI ranking.
best_by_cell = (
    all_options
    .sort_values(
        "benefit",
        ascending=False,
    )
    .groupby(
        "cell",
        as_index=False,
    )
    .first()
)

hvi_options = best_by_cell.merge(
    gdf[
        [
            "cell_id",
            "hvi",
        ]
    ],
    on="cell_id",
    how="left",
)


def fill_by_order(
    options,
    budget,
):

    selected = []

    used_cells = set()

    remaining = budget


    for _, row in options.iterrows():

        cell = int(
            row["cell"]
        )

        cost = float(
            row["cost"]
        )

        if cell in used_cells:
            continue

        if cost <= remaining:

            selected.append(
                row
            )

            used_cells.add(
                cell
            )

            remaining -= cost


    if selected:

        return pd.DataFrame(
            selected
        )

    return pd.DataFrame(
        columns=options.columns
    )


# =========================================================
# RUN
# =========================================================

result_rows = []
allocation_rows = []


for budget_fraction in BUDGET_FRACTIONS:

    budget = (
        reference_budget
        * budget_fraction
    )


    print(
        "\n" + "=" * 70
    )

    print(
        f"BUDGET: "
        f"{budget_fraction * 100:.0f}%"
    )

    print(
        f"Amount: {budget:.4f}"
    )


    # -----------------------------------------------------
    # HVI RANKING
    # -----------------------------------------------------

    hvi_sorted = (
        hvi_options
        .sort_values(
            [
                "hvi",
                "benefit",
            ],
            ascending=False,
        )
    )


    hvi_selected = fill_by_order(
        hvi_sorted[
            [
                "cell",
                "cell_id",
                "intervention",
                "cost",
                "benefit",
                "equity",
            ]
        ],
        budget,
    )


    result_rows.append(
        evaluate(
            hvi_selected,
            "HVI_Ranking",
            budget,
        )
    )


    # -----------------------------------------------------
    # GREEDY BENEFIT/COST
    # -----------------------------------------------------

    greedy_sorted = (
        all_options
        .sort_values(
            "benefit_cost"
            if "benefit_cost" in all_options.columns
            else "benefit",
            ascending=False,
        )
    )


    if "benefit_cost" not in greedy_sorted.columns:

        greedy_sorted = greedy_sorted.copy()

        greedy_sorted[
            "benefit_cost"
        ] = (
            greedy_sorted["benefit"]
            /
            greedy_sorted["cost"]
        )

        greedy_sorted = (
            greedy_sorted
            .sort_values(
                "benefit_cost",
                ascending=False,
            )
        )


    greedy_selected = fill_by_order(
        greedy_sorted,
        budget,
    )


    result_rows.append(
        evaluate(
            greedy_selected,
            "Greedy",
            budget,
        )
    )


    # -----------------------------------------------------
    # MILP
    # -----------------------------------------------------

    for alpha in EQUITY_LEVELS:

        label = (
            "MILP_No_Equity"
            if alpha is None
            else "MILP_Equity"
        )


        print(
            f"\nSolving {label}"
            + (
                ""
                if alpha is None
                else f" ({alpha:.0%})"
            )
        )


        result = solve(
            budget,
            alpha,
        )


        selected = decode_solution(
            result
        )


        status = (
            "optimal"
            if result.success
            else str(
                result.message
            )
        )


        print(
            "Status:",
            status
        )

        print(
            "Solver objective:",
            result.fun
        )


        result_rows.append(
            evaluate(
                selected,
                label,
                budget,
                alpha,
                status,
                getattr(
                    result,
                    "mip_dual_bound",
                    None,
                ),
            )
        )


        if not selected.empty:

            selected = selected.copy()

            selected[
                "budget_fraction"
            ] = budget_fraction

            selected[
                "equity_target"
            ] = (
                np.nan
                if alpha is None
                else alpha
            )

            selected[
                "strategy"
            ] = label

            allocation_rows.append(
                selected
            )


# =========================================================
# RESULTS
# =========================================================

results = pd.DataFrame(
    result_rows
)

allocations = pd.concat(
    allocation_rows,
    ignore_index=True,
)


print(
    "\n" + "=" * 70
)

print(
    "OPTIMIZATION V2 RESULTS"
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
            "solver_status",
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
# EQUITY TRADE-OFF
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "EQUITY-EFFICIENCY TRADE-OFF"
)

print(
    "=" * 70
)


for budget_fraction in BUDGET_FRACTIONS:

    d = results[
        results["budget"]
        .eq(
            reference_budget
            * budget_fraction
        )
    ]


    no_equity = d[
        d["strategy"]
        .eq(
            "MILP_No_Equity"
        )
    ]


    if no_equity.empty:
        continue


    baseline = (
        no_equity.iloc[0]
        ["total_benefit"]
    )


    print(
        f"\nBudget "
        f"{budget_fraction * 100:.0f}%:"
    )


    for alpha in [
        0.50,
        0.60,
        0.70,
    ]:

        row = d[
            d[
                "equity_target"
            ].eq(alpha)
        ]


        if row.empty:
            continue


        benefit = row.iloc[0][
            "total_benefit"
        ]


        loss_pct = (
            1
            -
            benefit
            /
            baseline
        ) * 100


        print(
            f"Equity {alpha:.0%}: "
            f"benefit={benefit:.2f} | "
            f"loss vs no-equity="
            f"{loss_pct:.2f}%"
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


if not allocations.empty:

    mix = (
        allocations
        .groupby(
            [
                "strategy",
                "budget_fraction",
                "equity_target",
                "intervention",
            ]
        )
        .size()
        .reset_index(
            name="cells"
        )
    )


    print(
        mix.to_string(
            index=False
        )
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
    "OPTIMIZATION V2 COMPLETE"
)

print(
    "=" * 70
)
