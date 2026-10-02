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
    / "heatscope_intervention_eligibility.gpkg"
)

OUTPUT_RESULTS = (
    RESULTS
    / "operational_optimizer_results.csv"
)

OUTPUT_ALLOCATIONS = (
    RESULTS
    / "operational_optimizer_allocations.csv"
)


# =========================================================
# SCENARIO PARAMETERS
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
    "tree": 1.00,
    "shade": 1.20,
    "cool_roof": 1.10,
    "cooling": 1.30,
}


BUDGET_FRACTION = 0.50


EQUITY_TARGETS = [
    None,
    0.50,
    0.60,
    0.70,
]


TIME_LIMIT = 600

MIP_GAP = 1e-4


print("=" * 70)
print("HeatScope Operational Intervention Optimizer")
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
        errors="coerce"
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
    .clip(0, 1)
)

need = (
    0.60 * risk
    +
    0.40 * structural
).clip(
    0,
    1
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

    "tree":
        gdf[
            "tree_opportunity"
        ].clip(0, 1),

    "shade":
        gdf[
            "shade_opportunity"
        ].clip(0, 1),

    "cool_roof":
        gdf[
            "cool_roof_opportunity"
        ].clip(0, 1),

    "cooling":
        gdf[
            "cooling_opportunity"
        ].clip(0, 1),
}


# =========================================================
# ELIGIBILITY
# =========================================================

eligible = {

    "tree":
        gdf[
            "tree_eligible"
        ].astype(bool),

    "shade":
        gdf[
            "shade_eligible"
        ].astype(bool),

    "cool_roof":
        gdf[
            "cool_roof_eligible"
        ].astype(bool),

    "cooling":
        gdf[
            "cooling_eligible"
        ].astype(bool),
}


# =========================================================
# OPERATIONAL BENEFIT
# =========================================================
#
# IMPORTANT:
# This is a relative scenario benefit.
# It is NOT an empirically measured temperature reduction.
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

    benefit_matrix[:, j] = (
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


# Remove non-eligible choices.

for j, intervention in enumerate(
    INTERVENTIONS
):

    mask = (
        ~eligible[
            intervention
        ].to_numpy()
    )

    benefit_matrix[
        mask,
        j
    ] = 0.0


# =========================================================
# PRINT ELIGIBILITY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "ELIGIBLE OPTIONS"
)

print(
    "=" * 70
)

for intervention in INTERVENTIONS:

    print(
        f"{intervention:12s}: "
        f"{eligible[intervention].sum():,}"
    )


# =========================================================
# VARIABLE CONSTRUCTION
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
                "intervention_idx":
                    j,
                "cost":
                    COST[
                        intervention
                    ],
                "benefit":
                    benefit_matrix[
                        i,
                        j
                    ],
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
    np.zeros(n_variables),
    np.ones(n_variables),
)


# =========================================================
# ONE INTERVENTION PER CELL
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
        :
    ],
    lb=-np.inf,
    ub=budget,
)


# =========================================================
# EQUITY CONSTRAINT
# =========================================================

def equity_constraint(
    alpha
):

    coeff = np.zeros(
        n_variables,
        dtype=float
    )


    for idx, row in variables.iterrows():

        if row["equity"]:

            coeff[idx] = (
                (1 - alpha)
                *
                row["benefit"]
            )

        else:

            coeff[idx] = (
                -alpha
                *
                row["benefit"]
            )


    return LinearConstraint(
        coeff[
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
    alpha
):

    constraints = [
        cell_constraint,
        cost_constraint,
    ]


    if alpha is not None:

        constraints.append(
            equity_constraint(
                alpha
            )
        )


    return milp(
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


# =========================================================
# DECODE
# =========================================================

def decode(
    result
):

    if result.x is None:

        return pd.DataFrame(
            columns=variables.columns
        )


    selected = variables.loc[
        result.x > 0.5
    ].copy()


    return selected


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    selected,
    strategy,
    equity_target,
    solver_status,
    gap,
):

    if selected.empty:

        return {
            "strategy":
                strategy,
            "equity_target":
                equity_target,
            "solver_status":
                solver_status,
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
            "relative_mip_gap":
                gap,
        }


    total_benefit = selected[
        "benefit"
    ].sum()


    equity_benefit = selected.loc[
        selected["equity"],
        "benefit"
    ].sum()


    return {
        "strategy":
            strategy,
        "equity_target":
            equity_target,
        "solver_status":
            solver_status,
        "selected_cells":
            len(selected),
        "total_cost":
            selected["cost"].sum(),
        "total_benefit":
            total_benefit,
        "equity_benefit":
            equity_benefit,
        "equity_benefit_share":
            equity_benefit
            /
            total_benefit
            if total_benefit > 0
            else 0,
        "relative_mip_gap":
            gap,
    }


# =========================================================
# HVI-FREE OPERATIONAL RANKING
# =========================================================
#
# Rank cells by the best operational benefit and assign
# the best eligible intervention.
# =========================================================

best_idx = np.argmax(
    benefit_matrix,
    axis=1
)


best_benefit = (
    benefit_matrix.max(
        axis=1
    )
)


ranking = pd.DataFrame(
    {
        "cell": np.arange(
            n_cells
        ),
        "cell_id":
            gdf["cell_id"].to_numpy(),
        "benefit":
            best_benefit,
        "intervention":
            [
                INTERVENTIONS[
                    i
                ]
                for i in best_idx
            ],
        "cost":
            [
                COST[
                    INTERVENTIONS[i]
                ]
                for i in best_idx
            ],
        "equity":
            gdf[
                "equity_priority"
            ].to_numpy(),
        "need":
            need.to_numpy(),
    }
)


# Zero-benefit cells are not useful.
ranking = ranking[
    ranking["benefit"] > 0
]


# =========================================================
# GREEDY
# =========================================================

greedy = (
    variables.copy()
)

greedy[
    "benefit_cost"
] = (
    greedy["benefit"]
    /
    greedy["cost"]
)


greedy = greedy.sort_values(
    "benefit_cost",
    ascending=False,
)


def greedy_select(
    budget
):

    selected = []

    used = set()

    remaining = budget


    for _, row in greedy.iterrows():

        if row["cell"] in used:
            continue

        if row["cost"] <= remaining:

            selected.append(
                row
            )

            used.add(
                row["cell"]
            )

            remaining -= (
                row["cost"]
            )


    if selected:

        return pd.DataFrame(
            selected
        )

    return pd.DataFrame(
        columns=greedy.columns
    )


# =========================================================
# RUN
# =========================================================

result_rows = []
allocation_rows = []


# ---------------------------------------------------------
# Operational ranking baseline
# ---------------------------------------------------------

ranked = ranking.sort_values(
    [
        "need",
        "benefit",
    ],
    ascending=False,
)


selected = []

used_cells = set()

remaining_budget = budget


for _, row in ranked.iterrows():

    if row["cell"] in used_cells:
        continue

    if row["cost"] <= remaining_budget:

        selected.append(
            row
        )

        used_cells.add(
            row["cell"]
        )

        remaining_budget -= (
            row["cost"]
        )


ranking_selected = pd.DataFrame(
    selected
)


result_rows.append(
    evaluate(
        ranking_selected,
        "Operational_Ranking",
        None,
        "baseline",
        np.nan,
    )
)


# ---------------------------------------------------------
# Greedy
# ---------------------------------------------------------

greedy_selected = greedy_select(
    budget
)


result_rows.append(
    evaluate(
        greedy_selected,
        "Greedy",
        None,
        "baseline",
        np.nan,
    )
)


# ---------------------------------------------------------
# MILP
# ---------------------------------------------------------

for alpha in EQUITY_TARGETS:

    label = (
        "MILP_No_Equity"
        if alpha is None
        else "MILP_Equity"
    )


    print(
        "\n" + "-" * 70
    )

    print(
        f"Solving {label}"
        +
        (
            ""
            if alpha is None
            else f" ({alpha:.0%})"
        )
    )


    result = solve(
        alpha
    )


    status = (
        "optimal"
        if result.success
        else result.message
    )


    selected = decode(
        result
    )


    dual_bound = getattr(
        result,
        "mip_dual_bound",
        np.nan,
    )


    mip_gap = np.nan


    if (
        np.isfinite(dual_bound)
        and result.fun is not None
        and abs(result.fun) > 1e-12
    ):

        mip_gap = (
            abs(
                abs(result.fun)
                -
                abs(dual_bound)
            )
            /
            abs(result.fun)
        )


    result_rows.append(
        evaluate(
            selected,
            label,
            alpha,
            status,
            mip_gap,
        )
    )


    if not selected.empty:

        selected = selected.copy()

        selected[
            "strategy"
        ] = label

        selected[
            "equity_target"
        ] = alpha

        allocation_rows.append(
            selected
        )


    print(
        "Status:",
        status
    )

    print(
        "Selected:",
        len(selected)
    )

    print(
        "Benefit:",
        f"{selected['benefit'].sum():.4f}"
    )

    print(
        "Equity share:",
        f"{(
            selected.loc[
                selected['equity'],
                'benefit'
            ].sum()
            /
            selected['benefit'].sum()
            if selected['benefit'].sum() > 0
            else 0
        ) * 100:.4f}%"
    )

    print(
        "MIP gap:",
        mip_gap
    )


# =========================================================
# RESULTS
# =========================================================

results = pd.DataFrame(
    result_rows
)


allocations = pd.concat(
    allocation_rows,
    ignore_index=True
)


print(
    "\n" + "=" * 70
)

print(
    "OPERATIONAL OPTIMIZATION RESULTS"
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
    "OPERATIONAL OPTIMIZER COMPLETE"
)

print(
    "=" * 70
)
