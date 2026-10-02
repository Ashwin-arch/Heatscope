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
    / "capacity_optimizer_results.csv"
)

OUTPUT_ALLOCATIONS = (
    RESULTS
    / "capacity_optimizer_allocations.csv"
)


INTERVENTIONS = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


# ---------------------------------------------------------
# SCENARIO PARAMETERS
# ---------------------------------------------------------

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


CAPACITY = {
    "tree": 1000,
    "shade": 1500,
    "cool_roof": 800,
    "cooling": 800,
}


EQUITY_TARGETS = [
    None,
    0.60,
    0.70,
]


BUDGET_FRACTION = 0.50

TIME_LIMIT = 600

MIP_GAP = 1e-4


print("=" * 70)
print("HeatScope Capacity-Constrained Optimization")
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


# Non-eligible options = impossible.
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
# BUILD VARIABLES
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


budget_constraint = LinearConstraint(
    cost_vector[
        np.newaxis,
        :
    ],
    lb=-np.inf,
    ub=budget,
)


# =========================================================
# CAPACITY CONSTRAINTS
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
# EQUITY
# =========================================================

def make_equity_constraint(
    alpha
):

    coefficient = np.zeros(
        n_variables,
        dtype=float,
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
        budget_constraint,
        capacity_constraint,
    ]


    if alpha is not None:

        constraints.append(
            make_equity_constraint(
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


    return variables.loc[
        result.x > 0.5
    ].copy()


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    selected,
    strategy,
    equity_target,
    status,
    gap,
):

    if selected.empty:

        return {
            "strategy":
                strategy,
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
            "equity_benefit_share":
                0.0,
            "relative_mip_gap":
                gap,
        }


    benefit = selected[
        "benefit"
    ].sum()


    equity_benefit = selected.loc[
        selected["equity"],
        "benefit",
    ].sum()


    return {
        "strategy":
            strategy,
        "equity_target":
            equity_target,
        "solver_status":
            status,
        "selected_cells":
            len(selected),
        "total_cost":
            selected["cost"].sum(),
        "total_benefit":
            benefit,
        "equity_benefit_share":
            (
                equity_benefit
                / benefit
                if benefit > 0
                else 0
            ),
        "relative_mip_gap":
            gap,
    }


# =========================================================
# GREEDY
# =========================================================

greedy = variables.copy()

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


def greedy_select():

    selected = []

    used_cells = set()

    capacity_used = {
        i: 0
        for i in INTERVENTIONS
    }

    remaining_budget = budget


    for _, row in greedy.iterrows():

        cell = int(
            row["cell"]
        )

        intervention = row[
            "intervention"
        ]

        cost = float(
            row["cost"]
        )


        if cell in used_cells:
            continue


        if (
            capacity_used[
                intervention
            ]
            >=
            CAPACITY[
                intervention
            ]
        ):
            continue


        if cost > remaining_budget:
            continue


        selected.append(
            row
        )

        used_cells.add(
            cell
        )

        capacity_used[
            intervention
        ] += 1

        remaining_budget -= cost


    return pd.DataFrame(
        selected
    )


# =========================================================
# RUN
# =========================================================

result_rows = []
allocation_rows = []


# ---------------------------------------------------------
# GREEDY BASELINE
# ---------------------------------------------------------

greedy_selected = greedy_select()


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


    dual = getattr(
        result,
        "mip_dual_bound",
        np.nan,
    )


    gap = np.nan


    if (
        np.isfinite(dual)
        and result.fun is not None
        and abs(result.fun) > 1e-12
    ):

        gap = (
            abs(
                abs(result.fun)
                -
                abs(dual)
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
            gap,
        )
    )


    print(
        "Status:",
        status
    )

    print(
        "Selected cells:",
        len(selected)
    )

    print(
        "Total benefit:",
        f"{selected['benefit'].sum():.4f}"
    )

    print(
        "Equity share:",
        f"{(
            selected.loc[
                selected["equity"],
                "benefit"
            ].sum()
            /
            selected["benefit"].sum()
            if selected["benefit"].sum() > 0
            else 0
        ) * 100:.4f}%"
    )

    print(
        "MIP gap:",
        gap
    )


    if not selected.empty:

        temp = selected.copy()

        temp[
            "strategy"
        ] = label

        temp[
            "equity_target"
        ] = alpha

        allocation_rows.append(
            temp
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
    "CAPACITY-CONSTRAINED RESULTS"
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
# CAPACITY USAGE
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "MILP CAPACITY USAGE"
)

print(
    "=" * 70
)


if not allocations.empty:

    cap = (
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
            name="selected"
        )
    )


    cap[
        "capacity"
    ] = cap[
        "intervention"
    ].map(
        CAPACITY
    )


    cap[
        "usage_pct"
    ] = (
        cap["selected"]
        /
        cap["capacity"]
        * 100
    )


    print(
        cap.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
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
    "CAPACITY OPTIMIZATION COMPLETE"
)

print(
    "=" * 70
)
