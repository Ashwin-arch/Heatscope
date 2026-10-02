from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

print("=" * 90)
print("HEATSCOPE FINAL BENCHMARK VERIFICATION")
print("=" * 90)


# ================================================================
# 1. SPATIAL ML BENCHMARK
# ================================================================

ml_path = RESULTS / "spatial_ml_results.csv"

if ml_path.exists():

    ml = pd.read_csv(ml_path)

    agg = (
        ml.groupby("model")
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
        )
        .reset_index()
    )

    agg = agg.sort_values(
        "rmse_mean"
    )

    print()
    print("=" * 90)
    print("1. SPATIAL ML BENCHMARK")
    print("=" * 90)
    print(
        agg.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

else:
    print("\n[MISSING] spatial_ml_results.csv")


# ================================================================
# 2. CONTEXT IMPROVEMENT
# ================================================================

context_path = (
    RESULTS
    / "context_spatial_ml_results.csv"
)

if context_path.exists():

    ctx = pd.read_csv(context_path)

    ctx_agg = (
        ctx.groupby("feature_set")
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            q5_mae=("q5_mae", "mean"),
            q5_rmse=("q5_rmse", "mean"),
            q5_bias=("q5_bias", "mean"),
        )
    )

    print()
    print("=" * 90)
    print("2. BASE VS SPATIAL CONTEXT")
    print("=" * 90)

    print(
        ctx_agg.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    if (
        "Base_5" in ctx_agg.index
        and "Context_9" in ctx_agg.index
    ):

        base_rmse = ctx_agg.loc[
            "Base_5",
            "rmse"
        ]

        context_rmse = ctx_agg.loc[
            "Context_9",
            "rmse"
        ]

        base_r2 = ctx_agg.loc[
            "Base_5",
            "r2"
        ]

        context_r2 = ctx_agg.loc[
            "Context_9",
            "r2"
        ]

        rmse_change = (
            100
            *
            (
                base_rmse
                -
                context_rmse
            )
            /
            base_rmse
        )

        r2_change = (
            context_r2
            -
            base_r2
        )

        print()
        print(
            f"Context RMSE improvement: "
            f"{rmse_change:.2f}%"
        )

        print(
            f"Context R² improvement: "
            f"{r2_change:.4f}"
        )


# ================================================================
# 3. CAPACITY-MATCHED PLANNING BENCHMARK
# ================================================================

capacity_path = (
    RESULTS
    / "capacity_optimizer_results.csv"
)

if capacity_path.exists():

    cap = pd.read_csv(
        capacity_path
    )

    print()
    print("=" * 90)
    print("3. CAPACITY-MATCHED PLANNING")
    print("=" * 90)

    cols = [
        c for c in [
            "strategy",
            "equity_target",
            "solver_status",
            "selected_cells",
            "total_cost",
            "total_benefit",
            "equity_benefit_share",
            "relative_mip_gap",
        ]
        if c in cap.columns
    ]

    print(
        cap[cols].to_string(
            index=False
        )
    )

    try:

        greedy = cap[
            cap["strategy"]
            == "Greedy"
        ].iloc[0]

        milp = cap[
            cap["strategy"]
            == "MILP_No_Equity"
        ].iloc[0]

        improvement = (
            100
            *
            (
                milp["total_benefit"]
                -
                greedy["total_benefit"]
            )
            /
            greedy["total_benefit"]
        )

        print()
        print(
            f"MILP benefit improvement "
            f"over Greedy: "
            f"{improvement:.2f}%"
        )

    except Exception as e:
        print(
            "Could not calculate "
            "Greedy vs MILP:",
            e
        )


# ================================================================
# 4. OPERATIONAL PLANNING BENCHMARK
# ================================================================

op_path = (
    RESULTS
    / "operational_optimizer_results.csv"
)

if op_path.exists():

    op = pd.read_csv(
        op_path
    )

    print()
    print("=" * 90)
    print("4. OPERATIONAL PLANNING")
    print("=" * 90)

    print(
        op.to_string(
            index=False
        )
    )


# ================================================================
# 5. BUDGET–EQUITY FRONTIER
# ================================================================

frontier_path = (
    RESULTS
    / "budget_equity_frontier.csv"
)

if frontier_path.exists():

    f = pd.read_csv(
        frontier_path
    )

    print()
    print("=" * 90)
    print("5. PRIMARY BUDGET–EQUITY FRONTIER")
    print("=" * 90)

    for budget in sorted(
        f["budget_fraction"].unique()
    ):

        s = f[
            f["budget_fraction"]
            ==
            budget
        ]

        print(
            f"\nBudget = {budget:.0%}"
        )

        print(
            s[
                [
                    "equity_target",
                    "selected_cells",
                    "total_cost",
                    "budget_utilization",
                    "total_benefit",
                    "equity_benefit_share",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ================================================================
# 6. FINAL SENSITIVITY
# ================================================================

sens_path = (
    RESULTS
    / "intervention_sensitivity_final.csv"
)

if sens_path.exists():

    s = pd.read_csv(
        sens_path
    )

    print()
    print("=" * 90)
    print("6. FINAL SENSITIVITY")
    print("=" * 90)

    print(
        "Total rows:",
        len(s)
    )

    print()
    print(
        "Solver status:"
    )

    print(
        s["status"]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------
    # Only certified optimal cases.
    # ------------------------------------------------------------

    optimal = s[
        s["status"]
        .str.contains(
            "Optimal",
            na=False
        )
    ].copy()

    print(
        "\nCertified optimal rows:",
        len(optimal)
    )

    # ------------------------------------------------------------
    # Equity validation.
    # ------------------------------------------------------------

    e = optimal[
        (
            optimal["experiment"]
            == "equity"
        )
        &
        (
            optimal["equity_target"]
            .notna()
        )
    ]

    print()
    print(
        "Equity constraint validation:"
    )

    all_valid = True

    for (
        budget,
        target
    ), x in e.groupby(
        [
            "budget_fraction",
            "equity_target",
        ]
    ):

        violations = (
            x["equity"]
            + 1e-8
            <
            target
        ).sum()

        if violations:
            all_valid = False

        print(
            f"Budget={budget:.0%} "
            f"Target={target:.0%} "
            f"n={len(x)} "
            f"violations={violations} "
            f"min={x['equity'].min():.4f} "
            f"median={x['equity'].median():.4f}"
        )

    print()

    if all_valid:
        print(
            "PASS: zero equity violations "
            "among certified optimal solutions."
        )
    else:
        print(
            "FAIL: equity violations detected."
        )

    # ------------------------------------------------------------
    # Selection stability.
    # ------------------------------------------------------------

    selection = optimal[
        optimal["experiment"]
        == "selection"
    ]

    print()
    print(
        "Selection stability:"
    )

    for budget in sorted(
        selection["budget_fraction"]
        .unique()
    ):

        x = selection[
            selection["budget_fraction"]
            ==
            budget
        ]

        print(
            f"\nBudget = {budget:.0%}"
        )

        for t in [
            "tree",
            "shade",
            "cool_roof",
            "cooling",
        ]:

            col = (
                f"{t}_selected"
            )

            print(
                f"{t:12s}"
                f" P="
                f"{(x[col] > 0).mean():.3f}"
                f" median="
                f"{x[col].median():.1f}"
            )


# ================================================================
# 7. UNCERTAINTY CHECK
# ================================================================

unc_path = (
    RESULTS
    / "adaptive_uncertainty_results.csv"
)

if unc_path.exists():

    u = pd.read_csv(
        unc_path
    )

    print()
    print("=" * 90)
    print("7. ADAPTIVE UNCERTAINTY CHECK")
    print("=" * 90)

    coverage = (
        (
            u["lst_celsius_p95"]
            >=
            u["adaptive_lower_95"]
        )
        &
        (
            u["lst_celsius_p95"]
            <=
            u["adaptive_upper_95"]
        )
    ).mean()

    width = (
        u["adaptive_upper_95"]
        -
        u["adaptive_lower_95"]
    )

    print(
        f"95% interval empirical coverage: "
        f"{coverage:.4f}"
    )

    print(
        f"Mean interval width: "
        f"{width.mean():.4f} C"
    )

    print(
        f"Median interval width: "
        f"{width.median():.4f} C"
    )

    print(
        f"P95 interval width: "
        f"{width.quantile(.95):.4f} C"
    )


print()
print("=" * 90)
print("FINAL BENCHMARK COMPLETE")
print("=" * 90)
