from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

src = ROOT / "results" / "budget_equity_frontier.csv"
out = ROOT / "results" / "pareto_frontier.csv"

df = pd.read_csv(src)

rows = []

for budget in sorted(df["budget_fraction"].unique()):

    s = df[df["budget_fraction"] == budget].copy()

    base = s[s["equity_target"].isna()]

    if len(base) != 1:
        raise RuntimeError(
            f"Expected one unconstrained solution at {budget:.0%}"
        )

    base_benefit = float(base["total_benefit"].iloc[0])

    for _, r in s.iterrows():

        benefit = float(r["total_benefit"])

        loss = base_benefit - benefit

        loss_fraction = (
            loss / base_benefit
            if base_benefit > 0
            else np.nan
        )

        rows.append({
            "budget_fraction": budget,
            "budget": r["budget"],
            "equity_target": r["equity_target"],
            "equity_share": r["equity_benefit_share"],
            "benefit": benefit,
            "benefit_loss": loss,
            "benefit_loss_fraction": loss_fraction,
            "selected_cells": r["selected_cells"],
            "total_cost": r["total_cost"],
            "budget_utilization": r["budget_utilization"],
        })

pareto = pd.DataFrame(rows)

flags = []

for i, r in pareto.iterrows():

    same = pareto[
        pareto["budget_fraction"]
        == r["budget_fraction"]
    ]

    dominated = (
        (same["benefit"] >= r["benefit"])
        &
        (same["equity_share"] >= r["equity_share"])
        &
        (
            (same["benefit"] > r["benefit"])
            |
            (same["equity_share"] > r["equity_share"])
        )
    ).any()

    flags.append(not dominated)

pareto["pareto_efficient"] = flags

pareto.to_csv(out, index=False)

print("=" * 80)
print("HEATSCOPE PARETO FRONTIER")
print("=" * 80)

for budget in sorted(
    pareto["budget_fraction"].unique()
):

    s = pareto[
        pareto["budget_fraction"] == budget
    ]

    print(f"\nBudget = {budget:.0%}")

    print(
        s[
            [
                "equity_target",
                "equity_share",
                "benefit",
                "benefit_loss_fraction",
                "pareto_efficient",
            ]
        ]
        .sort_values("equity_share")
        .to_string(index=False)
    )

print("\nSaved:", out)
