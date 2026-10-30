#!/usr/bin/env python
"""Generate publication-quality figures for the HeatScope IEEE paper.

Every figure is drawn only from verified artifacts in results/ and data/.
Run from the repository root:

    .venv/bin/python paper/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

INK = "#1b1b1b"
ACCENT = "#b23a48"
ACCENT2 = "#1f6f8b"
ACCENT3 = "#e08a1e"
GREY = "#8a8a8a"


def audit(fig, name: str) -> list[str]:
    """Report overlapping text and text clipped by the figure boundary."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw, fh = fig.canvas.get_width_height()

    items = []
    for ax in fig.axes:
        texts = list(ax.texts)
        if ax.axison:
            texts += ax.get_xticklabels() + ax.get_yticklabels()
            if ax.get_title():
                texts.append(ax.title)
            if ax.get_xlabel():
                texts.append(ax.xaxis.label)
            if ax.get_ylabel():
                texts.append(ax.yaxis.label)
        for t in texts:
            if not t.get_text().strip():
                continue
            if t in ax.get_xticklabels() + ax.get_yticklabels():
                # ticks outside the view interval are never drawn
                pos, axis = t.get_position(), None
                if t in ax.get_xticklabels():
                    axis = ax.get_xlim()
                elif t in ax.get_yticklabels():
                    axis = ax.get_ylim()
                if axis is not None and not (axis[0] <= pos[0] <= axis[1]):
                    continue
            try:
                bb = t.get_window_extent(renderer)
            except RuntimeError:
                continue
            items.append((t.get_text()[:28].replace("\n", " "), bb))

    problems = []
    for i, (ta, ba) in enumerate(items):
        for tb, bb in items[i + 1 :]:
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > 0 and h > 0:
                inter = w * h
                smaller = min(ba.width * ba.height, bb.width * bb.height)
                if smaller > 0 and inter / smaller > 0.30:
                    problems.append(f"OVERLAP {name}: '{ta}' <-> '{tb}' ({inter / smaller:.0%})")
    for ta, ba in items:
        if ba.x0 < -2 or ba.y0 < -2 or ba.x1 > fw + 2 or ba.y1 > fh + 2:
            problems.append(
                f"CLIPPED {name}: '{ta}' extent=({ba.x0:.0f},{ba.y0:.0f})-"
                f"({ba.x1:.0f},{ba.y1:.0f}) canvas={fw}x{fh}"
            )
    return problems


def save(fig, name: str) -> None:
    problems = audit(fig, name)
    for line in problems:
        print("  !", line)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}" + ("  [layout warnings]" if problems else ""))


# ------------------------------------------------------------------ figure 1
def figure_architecture() -> None:
    """Pipeline schematic drawn from the canonical module dependency graph.

    Boxes are placed on six non-overlapping rows: each row spans
    [y - BOX_H / 2, y + BOX_H / 2] and consecutive rows are separated by a
    3.5-unit gap, so no two boxes can intersect.
    """
    BOX_H = 8.0
    rows = [
        (62.0, [(27.0, 5.5), (27.0, 36.5), (27.0, 67.5)],
         [("Landsat 8/9 C2 L2\nST_B10, NDVI,\nNDMI, NDBI", ACCENT3),
          ("Overture Maps\nbuilding\nfootprints", ACCENT3),
          ("WorldPop 100 m\npopulation", ACCENT3)]),
        (50.5, [(44.0, 28.0)],
         [("Pre-processing, cloud masking,\n250 m analysis grid", ACCENT2)]),
        (39.0, [(36.0, 32.0)],
         [("Spatial context\n(neighbourhood means)", ACCENT2)]),
        (27.5, [(30.0, 17.0), (30.0, 51.0)],
         [("Model benchmark\nHGB / XGB / RF / ET / LR", ACCENT),
          ("TreeSHAP attribution\n(9 features)", ACCENT)]),
        (16.0, [(30.0, 17.0), (30.0, 51.0)],
         [("Cross-fitted\nprediction intervals", ACCENT),
          ("Priority +\nintervention benefit", ACCENT)]),
        (5.5, [(44.0, 28.0)],
         [("Equity-constrained\nMILP allocation", ACCENT)]),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    centres = []
    for y, boxes, labels in rows:
        row_centres = []
        for (w, x0), (label, colour) in zip(boxes, labels):
            ax.add_patch(
                FancyBboxPatch(
                    (x0, y - BOX_H / 2),
                    w,
                    BOX_H,
                    boxstyle="round,pad=0,rounding_size=1.6",
                    linewidth=0.9,
                    edgecolor=colour,
                    facecolor="white",
                )
            )
            ax.text(
                x0 + w / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=6.6,
                color=INK,
                linespacing=1.3,
            )
            row_centres.append(x0 + w / 2)
        centres.append(row_centres)

    lo = lambda y: y - BOX_H / 2
    hi = lambda y: y + BOX_H / 2
    c = centres
    arrows = [
        ((50, lo(62.0)), (50, hi(50.5))),
        ((50, lo(50.5)), (50, hi(39.0))),
        ((50, lo(39.0)), (50, hi(27.5))),
        ((c[3][0] + 15.0, 27.5), (c[3][1] - 15.0, 27.5)),
        ((c[3][0], lo(27.5)), (c[4][0], hi(16.0))),
        ((c[3][1], lo(27.5)), (c[4][1], hi(16.0))),
        ((c[4][0], lo(16.0)), (42.0, hi(5.5))),
        ((c[4][1], lo(16.0)), (58.0, hi(5.5))),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.8,
                color=GREY,
                shrinkA=0,
                shrinkB=0,
            )
        )

    ax.text(
        50,
        68.4,
        "HeatScope pipeline as implemented (modules shown are those that produce reported results)",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.0,
        0.4,
        "Solid arrows: data flow. Weather retrieval exists in the codebase but is a single city-point\n"
        "series and enters no model, so it is excluded from this diagram.",
        fontsize=6.0,
        color=GREY,
        ha="left",
        va="bottom",
    )
    save(fig, "fig1_architecture.png")


# ------------------------------------------------------------------ figure 2
def figure_study_area() -> None:
    """Observed target, model prediction/residual, and equity-priority maps."""
    g = gpd.read_file(RESULTS / "heatscope_operational_priority.gpkg")
    g = g.to_crs(4326)

    panels = [
        ("lst_celsius_p95", "Observed within-cell upper-tail LST (°C)", "inferno", None),
        ("predicted_lst_p95", "Cross-validated prediction (°C)", "inferno", None),
        ("residual", "Prediction − observation (°C)", "RdBu_r", 0),
        (
            "operational_priority_lambda05",
            "Operational priority index (uncertainty-aware)",
            "viridis",
            None,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), layout="constrained")
    for ax, (col, title, cmap, center) in zip(axes.ravel(), panels):
        series = g[col]
        if center is not None:
            lim = float(np.nanpercentile(np.abs(series), 99))
            kwargs = {"cmap": cmap, "vmin": -lim, "vmax": lim}
        else:
            kwargs = {"cmap": cmap}
        g.plot(column=col, ax=ax, linewidth=0, **kwargs)
        ax.set_title(title, fontsize=7.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    lon0, lon1, lat0, lat1 = 77.45, 77.78, 12.82, 13.16
    axes[0, 0].plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0], lw=0.7, color="white")
    fig.suptitle(
        f"Study area: 250 m analysis grid over {g.shape[0]:,} cells "
        f"(Landsat 8 path 144/row 051, 2026-04-01)",
        fontsize=8.4,
    )
    save(fig, "fig2_study_area.png")


# ------------------------------------------------------------------ figure 3
def figure_models() -> None:
    """Model comparison and the neighbourhood-context ablation."""
    base = pd.read_csv(RESULTS / "spatial_ml_results.csv")
    red = pd.read_csv(RESULTS / "reduced_spatial_ml_results.csv")
    ctx = pd.read_csv(RESULTS / "context_spatial_ml_results.csv")

    agg = base.groupby("model")[["mae", "rmse", "r2"]].agg(["mean", "std"])
    redg = red.groupby("model")[["mae", "rmse", "r2"]].mean()
    ctxg = ctx.groupby("feature_set")[["mae", "rmse", "r2", "q5_mae"]].mean()

    fig = plt.figure(figsize=(7.0, 3.05), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.0])

    # (a) MAE by model, ordered
    ax = fig.add_subplot(gs[0, 0])
    order = agg[("mae", "mean")].sort_values().index
    means = agg.loc[order, ("mae", "mean")].values
    stds = agg.loc[order, ("mae", "std")].values
    colours = [ACCENT if m == "HistGradientBoosting" else GREY for m in order]
    ypos = np.arange(len(order))
    ax.barh(ypos, means, xerr=stds, color=colours, height=0.62, capsize=2, lw=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([m.replace("HistGradientBoosting", "HistGradient\nBoosting") for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("MAE (°C), mean ± SD over 5 folds")
    ax.set_title("(a) Model comparison, 11 features", loc="left")
    for y, v in zip(ypos, means):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=6.4)

    # (b) feature ablation
    ax = fig.add_subplot(gs[0, 1])
    labels = ["Linear\nreg.", "Random\nforest", "Extra\ntrees", "XGBoost", "HGB"]
    idx = ["LinearRegression", "RandomForest", "ExtraTrees", "XGBoost", "HistGradientBoosting"]
    ax.plot(range(5), redg.loc[idx, "mae"].values, "o-", color=GREY, lw=1.1, ms=3.4, label="5 features")
    ax.plot(
        range(5),
        agg.loc[idx, ("mae", "mean")].values,
        "s-",
        color=ACCENT,
        lw=1.1,
        ms=3.4,
        label="11 features",
    )
    ax.plot([4], [ctxg.loc["Base_5", "mae"]], "D", color=ACCENT2, ms=4.2, label="Context_9")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (°C)")
    ax.set_xlabel("HGB = histogram gradient boosting", fontsize=6.4)
    ax.set_title("(b) Feature ablation", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="lower left")

    # (c) hottest-quintile bias
    ax = fig.add_subplot(gs[0, 2])
    sets = ["Base_5", "Context_9"]
    q5 = ctx[ctx.feature_set.isin(sets)].groupby("feature_set")[["q5_mae", "q5_bias"]].mean()
    xpos = np.arange(2)
    ax.bar(xpos - 0.17, q5.loc[sets, "q5_mae"], 0.32, color=ACCENT2, label="MAE", lw=0)
    ax.bar(xpos + 0.17, q5.loc[sets, "q5_bias"], 0.32, color=ACCENT3, label="bias", lw=0)
    for x, (a, b) in enumerate(zip(q5.loc[sets, "q5_mae"], q5.loc[sets, "q5_bias"])):
        ax.text(x - 0.17, a + 0.02, f"{a:.2f}", ha="center", fontsize=6.2)
        ax.text(x + 0.17, b + 0.02, f"{b:.2f}", ha="center", fontsize=6.2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Base_5", "Context_9"])
    ax.set_ylabel("°C")
    ax.set_title("(c) Hottest-quintile error", loc="left")
    ax.legend(frameon=False, fontsize=6.2)
    ax.set_ylim(0, 2.15)

    fig.suptitle(
        "Out-of-fold performance under 5-fold block-grouped spatial cross-validation",
        fontsize=8.4,
    )
    save(fig, "fig3_models.png")


# ------------------------------------------------------------------ figure 4
def figure_shap() -> None:
    """Global SHAP importance and the focal/neighbourhood decomposition."""
    imp = pd.read_csv(RESULTS / "context_shap_feature_importance.csv")
    focal = imp[~imp["feature"].str.contains("_nbr_mean")].copy()
    nbr = imp[imp["feature"].str.contains("_nbr_mean")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), layout="constrained")

    ax = axes[0]
    focal = focal.sort_values("relative_importance_pct")
    ypos = np.arange(len(focal))
    ax.barh(ypos, focal["relative_importance_pct"], color=ACCENT2, height=0.6, lw=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(focal["feature"].str.replace("_mean", "", regex=False))
    ax.set_xlabel("Mean |SHAP| (% of total)")
    ax.set_title("(a) Focal-cell contributions", loc="left")
    for y, v in enumerate(focal["relative_importance_pct"]):
        ax.text(v + 0.5, y, f"{v:.2f}%", va="center", fontsize=6.2)
    ax.set_xlim(0, max(focal["relative_importance_pct"]) * 1.3)

    ax = axes[1]
    nbr = nbr.sort_values("relative_importance_pct")
    ypos = np.arange(len(nbr))
    ax.barh(ypos, nbr["relative_importance_pct"], color=ACCENT, height=0.6, lw=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(nbr["feature"].str.replace("_mean_nbr_mean", "", regex=False))
    ax.set_xlabel("Mean |SHAP| (% of total)")
    ax.set_title("(b) Neighbourhood contributions", loc="left")
    for y, v in enumerate(nbr["relative_importance_pct"]):
        ax.text(v + 0.5, y, f"{v:.2f}%", va="center", fontsize=6.2)
    ax.set_xlim(0, max(nbr["relative_importance_pct"]) * 1.3)
    ax.text(
        0.98,
        0.06,
        f"neighbourhood subtotal {nbr['relative_importance_pct'].sum():.2f}%",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.4,
        color=ACCENT,
    )

    fig.suptitle(
        "TreeSHAP attribution for the 9-feature context model (5,000 stratified cells)",
        fontsize=8.4,
    )
    save(fig, "fig4_shap.png")


# ------------------------------------------------------------------ figure 5
def figure_uncertainty() -> None:
    """Cross-fitted interval coverage overall and by observed-temperature quintile."""
    res = pd.read_csv(RESULTS / "spatial_uncertainty_results.csv")
    res = res.assign(
        covered=(res["lower_95"] <= res["actual_lst_p95"]) & (res["actual_lst_p95"] <= res["upper_95"]),
        width=res["upper_95"] - res["lower_95"],
    )
    overall = float(res["covered"].mean())
    width = float(res["width"].mean())
    per_q = res.groupby("target_quantile").agg(
        coverage=("covered", "mean"), width=("width", "mean"), n=("covered", "size")
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), layout="constrained")

    ax = axes[0]
    qs = per_q.index.astype(str).tolist()
    vals = per_q["coverage"].values * 100
    bars = ax.bar(qs, vals, color=[ACCENT if v < 95 else ACCENT2 for v in vals], width=0.62, lw=0)
    ax.axhline(95, ls="--", lw=0.9, color=INK)
    ax.axhline(overall * 100, ls=":", lw=0.9, color=GREY)
    ax.text(4.45, 95.4, "nominal 95%", fontsize=6.2, ha="right", va="bottom", color=INK)
    ax.text(
        4.45,
        overall * 100 - 1.2,
        f"overall {overall * 100:.2f}%",
        fontsize=6.2,
        ha="right",
        va="top",
        color=GREY,
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}", ha="center", fontsize=6.2)
    ax.set_ylim(80, 101)
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_xlabel("Observed within-cell upper-tail LST quintile")
    ax.set_title("(a) Coverage by temperature quintile", loc="left")

    ax = axes[1]
    ax.bar(qs, per_q["width"].values, color=GREY, width=0.62, lw=0)
    for i, v in enumerate(per_q["width"].values):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.2)
    ax.axhline(width, ls=":", lw=0.9, color=ACCENT)
    ax.text(
        4.45,
        width - 0.06,
        f"overall {width:.2f} °C",
        fontsize=6.2,
        ha="right",
        va="top",
        color=ACCENT,
    )
    ax.set_ylabel("Mean 95% interval width (°C)")
    ax.set_xlabel("Observed within-cell upper-tail LST quintile")
    ax.set_title("(b) Interval width", loc="left")
    ax.set_ylim(0, max(per_q["width"].values) * 1.25)

    fig.suptitle(
        "Cross-fitted prediction intervals (fold-wise residual quantiles, no formal conformal claim)",
        fontsize=8.4,
    )
    save(fig, "fig5_uncertainty.png")


# ------------------------------------------------------------------ figure 6
def figure_optimization() -> None:
    """Equity-benefit trade-off and the resulting intervention mix."""
    front = pd.read_csv(RESULTS / "budget_equity_frontier.csv")
    front = front[front["budget_fraction"].isin([0.25, 0.5, 0.75, 1.0])]
    none_rows = front[front["equity_target"].isna()].sort_values("budget_fraction")

    op = pd.read_csv(RESULTS / "operational_optimizer_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), layout="constrained")

    ax = axes[0]
    ax.plot(
        none_rows["equity_benefit_share"],
        none_rows["total_benefit"],
        "o-",
        color=ACCENT2,
        lw=1.2,
        ms=4,
        label="Budget sweep (no equity floor)",
    )
    for _, r in none_rows.iterrows():
        ax.annotate(
            f"{r['budget_fraction'] * 100:.0f}% budget",
            (r["equity_benefit_share"], r["total_benefit"]),
            textcoords="offset points",
            xytext=(5, -7),
            fontsize=6.2,
        )
    eq = op[op["equity_target"].notna() & (op["strategy"] == "MILP_Equity")]
    ax.plot(
        eq["equity_benefit_share"],
        eq["total_benefit"],
        "s",
        color=ACCENT,
        ms=4.5,
        label="Equity floor (full budget)",
    )
    for _, r in eq.iterrows():
        ax.annotate(
            f"α={r['equity_target']}",
            (r["equity_benefit_share"], r["total_benefit"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=6.2,
            color=ACCENT,
        )
    base = op[op["strategy"] == "MILP_No_Equity"].iloc[0]
    ax.plot(
        [base["equity_benefit_share"]],
        [base["total_benefit"]],
        "*",
        color=INK,
        ms=9,
        label="Unconstrained optimum",
    )
    ax.set_xlabel("Equity-priority share of total benefit")
    ax.set_ylabel("Total scenario benefit")
    ax.set_title("(a) Benefit-equity trade-off", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")

    ax = axes[1]
    kinds = ["tree", "shade", "cool_roof", "cooling"]
    labels = ["Urban\ntree", "Shade\nstructure", "Cool\nroof", "Cooling\ncentre"]
    colours = [ACCENT2, ACCENT3, ACCENT, GREY]
    f1 = none_rows[none_rows["budget_fraction"] == 0.25].iloc[0]
    f2 = none_rows[none_rows["budget_fraction"] == 1.0].iloc[0]
    xpos = np.arange(len(kinds))
    ax.bar(
        xpos - 0.19,
        [f1[f"{k}_selected"] for k in kinds],
        0.36,
        color=[GREY] * 4,
        lw=0,
        label="25% budget (1,482 cells)",
    )
    ax.bar(
        xpos + 0.19,
        [f2[f"{k}_selected"] for k in kinds],
        0.36,
        color=colours,
        lw=0,
        label="100% budget (4,100 cells)",
    )
    for x, (a, b) in enumerate(
        zip([f1[f"{k}_selected"] for k in kinds], [f2[f"{k}_selected"] for k in kinds])
    ):
        ax.text(x - 0.19, a + 18, f"{a:.0f}", ha="center", fontsize=6.0)
        ax.text(x + 0.19, b + 18, f"{b:.0f}", ha="center", fontsize=6.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cells assigned")
    ax.set_title("(b) Intervention mix at budget extremes", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.set_ylim(0, 1750)

    fig.suptitle(
        "Equity-constrained MILP allocation under binding budget (budget-equity frontier)",
        fontsize=8.4,
    )
    save(fig, "fig6_optimization.png")


if __name__ == "__main__":
    figure_architecture()
    figure_study_area()
    figure_models()
    figure_shap()
    figure_uncertainty()
    figure_optimization()
    print("all figures written")