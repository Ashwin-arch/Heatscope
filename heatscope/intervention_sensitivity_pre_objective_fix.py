from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.optimize import milp, LinearConstraint, Bounds


ROOT = Path(__file__).resolve().parents[1]

GRID_PATH = (
    ROOT
    / "results"
    / "heatscope_intervention_eligibility.gpkg"
)

OUT_CSV = (
    ROOT
    / "results"
    / "intervention_sensitivity.csv"
)


# ================================================================
# Configuration
# ================================================================

SEED = 42

N_SELECTION = 100
N_EQUITY = 50

TIME_LIMIT = 20

BUDGET_SELECTION = [
    0.50,
    0.75,
    1.00,
]

BUDGET_EQUITY = [
    0.75,
    1.00,
]

EQUITY_TARGETS = [
    0.70,
    0.80,
]

COST_RANGE = 0.30
EFFECT_RANGE = 0.30
CAPACITY_RANGE = 0.20

rng = np.random.default_rng(SEED)


# ================================================================
# Load
# ================================================================

df = gpd.read_file(
    GRID_PATH,
    layer="eligibility",
)

required = [
    "cell_id",
    "population",
    "structural_vulnerability",
    "risk_exposure_lambda05",

    "tree_opportunity",
    "shade_opportunity",
    "cool_roof_opportunity",
    "cooling_opportunity",

    "tree_eligible",
    "shade_eligible",
    "cool_roof_eligible",
    "cooling_eligible",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


types = [
    "tree",
    "shade",
    "cool_roof",
    "cooling",
]


opportunity_col = {
    "tree": "tree_opportunity",
    "shade": "shade_opportunity",
    "cool_roof": "cool_roof_opportunity",
    "cooling": "cooling_opportunity",
}


eligible_col = {
    "tree": "tree_eligible",
    "shade": "shade_eligible",
    "cool_roof": "cool_roof_eligible",
    "cooling": "cooling_eligible",
}


BASE_COST = {
    "tree": 1.0,
    "shade": 1.4,
    "cool_roof": 2.0,
    "cooling": 2.5,
}


BASE_EFFECT = {
    "tree": 1.0,
    "shade": 1.2,
    "cool_roof": 1.1,
    "cooling": 1.3,
}


BASE_CAPACITY = {
    "tree": 1000,
    "shade": 1500,
    "cool_roof": 800,
    "cooling": 800,
}


# ================================================================
# Operational variables
# ================================================================

risk_raw = df[
    "risk_exposure_lambda05"
].to_numpy(float)

struct_raw = df[
    "structural_vulnerability"
].to_numpy(float)

population = (
    df["population"]
    .fillna(0)
    .to_numpy(float)
)

logpop = np.log1p(population)

pmin = np.nanpercentile(logpop, 1)
pmax = np.nanpercentile(logpop, 99)

logpop_clip = np.clip(
    logpop,
    pmin,
    pmax,
)

if pmax > pmin:
    pop_factor = (
        (logpop_clip - pmin)
        /
        (pmax - pmin)
    )
else:
    pop_factor = np.zeros(len(df))


# Canonical HeatScope equity group.
# Exactly matches budget_equity_frontier.py.
equity_priority = (
    df["equity_priority"]
    .fillna(False)
    .astype(bool)
    .to_numpy()
)


def robust_normalize(x):

    lo = np.nanpercentile(x, 1)
    hi = np.nanpercentile(x, 99)

    y = np.clip(
        x,
        lo,
        hi,
    )

    if hi <= lo:
        return np.zeros(len(x))

    return (
        (y - lo)
        /
        (hi - lo)
    )


risk = robust_normalize(
    risk_raw
)

structural = robust_normalize(
    struct_raw
)

need = (
    0.60 * risk
    +
    0.40 * structural
)


eligible = {
    t:
    df[eligible_col[t]]
    .fillna(False)
    .to_numpy(bool)
    for t in types
}


opportunity = {
    t:
    np.clip(
        df[opportunity_col[t]]
        .fillna(0.0)
        .to_numpy(float),
        0,
        1,
    )
    for t in types
}


# ================================================================
# Decision variables
# ================================================================

variables = []

for i in range(len(df)):

    for t in types:

        if eligible[t][i]:

            variables.append(
                (i, t)
            )


m = len(variables)

print("=" * 70)
print("HeatScope Intervention Sensitivity")
print("=" * 70)
print(f"Cells: {len(df)}")
print(f"Decision variables: {m}")
print(f"Selection scenarios: {N_SELECTION}")
print(f"Equity scenarios: {N_EQUITY}")
print(f"MILP time limit: {TIME_LIMIT}s")


# ================================================================
# Build static constraints
# ================================================================

rows_static = []
lb_static = []
ub_static = []


# ------------------------------------------------
# One intervention per cell
# ------------------------------------------------

cell_to_variables = {}

for j, (i, t) in enumerate(variables):

    cell_to_variables.setdefault(
        i,
        []
    ).append(j)


for indices in cell_to_variables.values():

    row = np.zeros(m)

    row[indices] = 1

    rows_static.append(row)
    lb_static.append(-np.inf)
    ub_static.append(1)


# ------------------------------------------------
# Capacity rows
# ------------------------------------------------

capacity_rows = {}

for t in types:

    row = np.zeros(m)

    for j, (_, tt) in enumerate(variables):

        if tt == t:

            row[j] = 1

    capacity_rows[t] = row


# ================================================================
# Solver
# ================================================================

def solve(
    costs,
    effects,
    capacities,
    budget_fraction,
    equity_target=None,
):

    benefit = np.zeros(m)

    for j, (i, t) in enumerate(variables):

        benefit[j] = (
            need[i]
            *
            opportunity[t][i]
            *
            population[i]
            *
            pop_factor[i]
            *
            effects[t]
        )

    rows = list(rows_static)
    lower = list(lb_static)
    upper = list(ub_static)

    # ------------------------------------------------
    # Budget
    # ------------------------------------------------

    max_capacity_cost = sum(
        costs[t] * capacities[t]
        for t in types
    )

    budget = (
        budget_fraction
        *
        max_capacity_cost
    )

    budget_row = np.zeros(m)

    for j, (_, t) in enumerate(variables):

        budget_row[j] = costs[t]

    rows.append(budget_row)
    lower.append(-np.inf)
    upper.append(budget)

    # ------------------------------------------------
    # Capacities
    # ------------------------------------------------

    for t in types:

        rows.append(
            capacity_rows[t]
        )

        lower.append(-np.inf)
        upper.append(
            capacities[t]
        )

    # ------------------------------------------------
    # Equity
    # ------------------------------------------------

    vulnerable = equity_priority

    if equity_target is not None:

        equity_row = np.zeros(m)

        for j, (i, _) in enumerate(variables):

            if vulnerable[i]:

                equity_row[j] = benefit[j]

            else:

                equity_row[j] = (
                    -equity_target
                    * benefit[j]
                )

        rows.append(equity_row)
        lower.append(0)
        upper.append(np.inf)

    A = np.vstack(rows)

    constraints = LinearConstraint(
        A,
        np.asarray(lower),
        np.asarray(upper),
    )

    result = milp(
        c=-benefit,
        integrality=np.ones(m),
        bounds=Bounds(
            np.zeros(m),
            np.ones(m),
        ),
        constraints=constraints,
        options={
            "time_limit": TIME_LIMIT,
            "presolve": True,
        },
    )

    if not result.success:

        return {
            "status": result.message,
            "selected": np.nan,
            "cost": np.nan,
            "benefit": np.nan,
            "equity": np.nan,
            **{
                f"{t}_selected": np.nan
                for t in types
            },
        }

    x = np.rint(
        result.x
    ).astype(int)

    selected = np.where(
        x == 1
    )[0]

    total_cost = 0
    total_benefit = 0
    equity_benefit = 0

    counts = {
        t: 0
        for t in types
    }

    vulnerable = equity_priority

    for j in selected:

        i, t = variables[j]

        total_cost += costs[t]

        total_benefit += benefit[j]

        counts[t] += 1

        if vulnerable[i]:

            equity_benefit += benefit[j]

    equity = (
        equity_benefit
        /
        total_benefit
        if total_benefit > 0
        else np.nan
    )

    return {
        "status": result.message,
        "selected": len(selected),
        "cost": total_cost,
        "benefit": total_benefit,
        "equity": equity,
        **{
            f"{t}_selected": counts[t]
            for t in types
        },
    }


# ================================================================
# Parameter generation
# ================================================================

def sample_parameters():

    cost_mult = {
        t: rng.uniform(
            1 - COST_RANGE,
            1 + COST_RANGE,
        )
        for t in types
    }

    effect_mult = {
        t: rng.uniform(
            1 - EFFECT_RANGE,
            1 + EFFECT_RANGE,
        )
        for t in types
    }

    capacity_mult = {
        t: rng.uniform(
            1 - CAPACITY_RANGE,
            1 + CAPACITY_RANGE,
        )
        for t in types
    }

    costs = {
        t:
        BASE_COST[t]
        *
        cost_mult[t]
        for t in types
    }

    effects = {
        t:
        BASE_EFFECT[t]
        *
        effect_mult[t]
        for t in types
    }

    capacities = {
        t:
        max(
            1,
            int(
                round(
                    BASE_CAPACITY[t]
                    *
                    capacity_mult[t]
                )
            ),
        )
        for t in types
    }

    return (
        costs,
        effects,
        capacities,
    )


# ================================================================
# Experiment
# ================================================================

records = []


# ------------------------------------------------
# A. Intervention-selection sensitivity
#
# No equity constraint.
# ------------------------------------------------

for run in range(N_SELECTION):

    costs, effects, capacities = (
        sample_parameters()
    )

    for budget_fraction in BUDGET_SELECTION:

        result = solve(
            costs,
            effects,
            capacities,
            budget_fraction,
            None,
        )

        records.append({

            "experiment":
                "selection",

            "run":
                run,

            "budget_fraction":
                budget_fraction,

            "equity_target":
                np.nan,

            **result,

            **{
                f"{t}_cost":
                costs[t]
                for t in types
            },

            **{
                f"{t}_effect":
                effects[t]
                for t in types
            },

            **{
                f"{t}_capacity":
                capacities[t]
                for t in types
            },
        })

    if (
        (run + 1) % 10
        == 0
    ):

        print(
            f"Selection "
            f"{run + 1}/{N_SELECTION}"
        )


# ------------------------------------------------
# B. Equity sensitivity
#
# Only where the deterministic frontier
# showed meaningful equity trade-offs.
# ------------------------------------------------

for run in range(N_EQUITY):

    costs, effects, capacities = (
        sample_parameters()
    )

    for budget_fraction in BUDGET_EQUITY:

        for equity_target in EQUITY_TARGETS:

            result = solve(
                costs,
                effects,
                capacities,
                budget_fraction,
                equity_target,
            )

            records.append({

                "experiment":
                    "equity",

                "run":
                    run,

                "budget_fraction":
                    budget_fraction,

                "equity_target":
                    equity_target,

                **result,

                **{
                    f"{t}_cost":
                    costs[t]
                    for t in types
                },

                **{
                    f"{t}_effect":
                    effects[t]
                    for t in types
                },

                **{
                    f"{t}_capacity":
                    capacities[t]
                    for t in types
                },
            })

    if (
        (run + 1) % 10
        == 0
    ):

        print(
            f"Equity "
            f"{run + 1}/{N_EQUITY}"
        )


# ================================================================
# Save
# ================================================================

out = pd.DataFrame(
    records
)

out.to_csv(
    OUT_CSV,
    index=False,
)


# ================================================================
# Selection stability summary
# ================================================================

print()
print("=" * 70)
print("INTERVENTION SELECTION STABILITY")
print("=" * 70)

selection = out[
    out["experiment"]
    == "selection"
]


for budget in BUDGET_SELECTION:

    s = selection[
        selection[
            "budget_fraction"
        ]
        ==
        budget
    ]

    print()
    print(
        f"Budget = {budget:.0%}"
    )

    for t in types:

        col = (
            f"{t}_selected"
        )

        print(
            f"{t:12s}"
            f" probability="
            f"{(s[col] > 0).mean():.3f}"
            f" mean="
            f"{s[col].mean():.1f}"
            f" median="
            f"{s[col].median():.1f}"
        )


# ================================================================
# Equity robustness
# ================================================================

print()
print("=" * 70)
print("BENEFIT–EQUITY ROBUSTNESS")
print("=" * 70)

equity = out[
    out["experiment"]
    == "equity"
]


for budget in BUDGET_EQUITY:

    print()
    print(
        f"Budget = {budget:.0%}"
    )

    baseline = equity[
        equity[
            "equity_target"
        ].isna()
    ]

    # There is no unconstrained record inside
    # this experiment, so compare constrained
    # solutions against deterministic baseline
    # later from budget_equity_frontier.csv.

    for target in EQUITY_TARGETS:

        s = equity[
            (
                equity[
                    "budget_fraction"
                ]
                ==
                budget
            )
            &
            (
                equity[
                    "equity_target"
                ]
                ==
                target
            )
        ]

        print(
            f"Equity {target:.0%}: "
            f"benefit median="
            f"{s['benefit'].median():.2f}"
            f" P10="
            f"{s['benefit'].quantile(.10):.2f}"
            f" P90="
            f"{s['benefit'].quantile(.90):.2f}"
            f" equity median="
            f"{s['equity'].median():.3f}"
        )


# ================================================================
# Completion
# ================================================================

print()
print("=" * 70)
print("SENSITIVITY COMPLETE")
print("=" * 70)

print(
    f"Saved: {OUT_CSV}"
)

print(
    f"Total solves: {len(out)}"
)
