# HeatScope — Claim Verification Matrix

Every quantitative claim that was proposed for the paper was checked against the repository code and
the committed artifacts. Verdicts:

- **VERIFIED** — reproducible from an artifact; the value in the paper matches the artifact.
- **VERIFIED (corrected)** — the underlying number is real, but the original wording overstated or
  misdescribed it. The paper uses the corrected wording given here.
- **NOT SUPPORTED** — the artifact does not contain the claim. Excluded from the paper.
- **PARTIAL** — supported only under a narrower statement than originally proposed.

Source-of-truth precedence used throughout: **code > artifact > draft prose**. Where the draft and the
artifact disagreed, the artifact won and the discrepancy is recorded below.

## Study area and data

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| D1 | Study area is Bengaluru, India | VERIFIED | `config.py:10-14` BBOX `[77.45, 12.82, 77.78, 13.16]`; 1,348.2515 km² |
| D2 | Analysis grid is 250 m | VERIFIED | `config.py` `GRID_SIZE_METERS=250`; `grid.py:88-113`; max cell area 62,500 m² |
| D3 | Grid contains 21,845 cells | VERIFIED | `results/heatscope_grid.csv` row count; 21,815 usable after missing-target cells |
| D4 | Total area 1,348.25 km² | VERIFIED | `sum(area_m2)` = 1.3482515 × 10⁹ m² |
| D5 | Total population 7,458,314 | VERIFIED | `sum(population)` = 7,458,313.9326 → 7,458,314; WorldPop 100 m raster, EPSG:4326 |
| D6 | Study area is the Bengaluru administrative boundary | **NOT SUPPORTED** | It is a rectangular bounding box. Corrected to "a 0.33° × 0.34° rectangular study region enclosing the Bengaluru metropolitan area" |
| D7 | Building footprints from Microsoft Global ML Building Footprints | **VERIFIED (corrected)** | `data/raw/building_dataset_links.csv:13308` resolves to `bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/...quadkey=123303312/...` = **Overture Maps Buildings theme, snapshot 2026-08-13, ODbL**. One quadkey tile only |
| D8 | Landsat scene is `LC08_L2SP_144051_20260401_02_T1`, 2026-04-01, cloud 0.23% | VERIFIED | `data/raw/scene_metadata.csv` |
| D9 | 39,624,101 valid 30 m pixels in the AOI window | VERIFIED | Raster 7541 × 7701, EPSG:32643, float32, NaN nodata |
| D10 | LST raster median 41.59 °C, P99 51.96 °C, max 70.50 °C | VERIFIED | Computed from `data/processed/lst_celsius.tif` |
| D11 | Weather variables contribute to the model | **NOT SUPPORTED** | All 21 `weather_*` columns have exactly **1 unique value** across all 21,815 cells (`integrate_weather.py:180-182` broadcasts a single NASA POWER city-point summary). Exhaustive grep of every feature list in the repository returns zero weather features. Removed from the paper entirely |
| D12 | WorldPop product is the 2025 India release | PARTIAL | Filename says `worldpop_india_2025`; only WorldPop 100 m / India v2 metadata was verifiable. Paper cites the WorldPop project DOI and records the filename as-is; see MISSING_EVIDENCE M3 |

## Target variable definition

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| T1 | Target is a p95 land-surface-temperature **extremity** (heatwave, extreme-heat day, climatology) | **NOT SUPPORTED** | `landsat_features.py:347-361` computes `np.percentile(values, 95)` over the valid 30 m pixels **inside one 250 m cell on one acquisition date**. There is no temporal stack anywhere in the repository. Corrected throughout to "within-cell upper-tail (p95) land-surface temperature on 2026-04-01" |
| T2 | Target is a spatial, within-cell upper-tail statistic | VERIFIED | As above |
| T3 | Target units are °C | VERIFIED | `preprocess.py:100` `LST_C = DN × 0.00341802 + 149.0 − 273.15` |
| T4 | LST is a proxy for human heat exposure, not air temperature | VERIFIED (corrected) | Now stated explicitly as a limitation, citing Voogt & Oke and Heaviside et al. |

## Spatial validation design

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| V1 | 5-fold spatial cross-validation | VERIFIED | `GroupKFold(n_splits=5)` on `spatial_block` in all benchmark scripts |
| V2 | 240 blocks of ~2.5 km | VERIFIED | `results/spatial_error_summary.csv`; 48 blocks per fold exactly |
| V3 | No train/test block overlap | VERIFIED | `spatial_ml_benchmark.py:334-352` asserts disjoint block sets and raises `RuntimeError` on overlap |
| V4 | Neighbourhood context features exclude the focal cell | VERIFIED | `spatial_context.py:97` skips `j == i`; no self-leakage |
| V5 | Hottest-quintile metrics are per-fold independent | **NOT SUPPORTED** | Quintiles are cut on the **pooled** target before CV (`context_spatial_benchmark.py:186-197`). Used for stratified reporting only, never as a feature. Paper reports **pooled** Q5 statistics only |

## Model performance

All values below are means over 5 folds from the named artifact. Standard deviations are reported in
the paper because they are large relative to the between-model differences.

| # | Claim | Verdict | Value in artifact | Paper |
|---|---|---|---|---|
| M1 | HGB MAE 1.2126 °C | VERIFIED | 1.2126 ± 0.0512 | 1.213 ± 0.051 |
| M2 | XGBoost MAE 1.2150 °C | VERIFIED | 1.2150 ± 0.0470 | 1.215 ± 0.047 |
| M3 | RF MAE 1.2178 °C | VERIFIED | 1.2178 ± 0.0511 | 1.218 ± 0.051 |
| M4 | ET MAE 1.2262 °C | VERIFIED | 1.2262 ± 0.0540 | 1.226 ± 0.054 |
| M5 | LR MAE 1.2901 °C | VERIFIED | 1.2901 ± 0.0684 | 1.290 ± 0.068 |
| M6 | HGB RMSE 1.5909 °C | VERIFIED | 1.5909 ± 0.0701 | 1.591 ± 0.070 |
| M7 | HGB R² 0.6356 | VERIFIED | 0.6356 ± 0.0426 | 0.636 ± 0.043 |
| M8 | HGB is the best model | **VERIFIED (corrected)** | HGB has the best mean MAE, but its advantage over XGBoost (0.0024 °C) is far smaller than the fold SD (≈0.05 °C). Paper says "best mean performance, differences not distinguishable at this sample size" |
| M9 | Reduced 5-feature HGB MAE 1.2241 °C | VERIFIED | 1.2241 | 1.224 |
| M10 | Context_9 MAE 1.1738 °C | VERIFIED | 1.1738 | 1.174 |
| M11 | Context_9 RMSE 1.5478 °C | VERIFIED | 1.5478 | 1.548 |
| M12 | Context_9 R² 0.6548 | VERIFIED | 0.6548 | 0.655 |
| M13 | Neighbourhood context improves RMSE by 3.62% | VERIFIED | 1.6060 → 1.5478 = −3.62% | 3.62% |
| M14 | Neighbourhood context improves R² by +0.0262 | VERIFIED | 0.6286 → 0.6548 | +0.026 |
| M15 | Hottest-quintile MAE improves 1.7198 → 1.6305 °C | VERIFIED | −5.19% | 5.19% |
| M16 | Hottest-quintile bias improves 1.5005 → 1.3792 °C | VERIFIED | −8.08% | 8.08% |
| M17 | The model under-predicts the hottest cells | VERIFIED (corrected) | Q5 bias is **+1.38 °C** (mean observed exceeds mean prediction). Promoted from a hidden diagnostic to an explicit reported finding |
| M18 | Mutual information identifies the most useful features | **NOT SUPPORTED** | `feature_audit.py:175-186` computes MI **fully in-sample** with no cross-validation. Paper reports MI only as a descriptive in-sample association, never as feature selection |

## Feature engineering

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| F1 | NDBI and NDMI are independent predictors | **NOT SUPPORTED** | `preprocess.py:173-195`: `ndbi_den = swir + nir` and `ndmi_den = nir + swir` are the *same* expression with opposite numerators, so **NDBI ≡ −NDMI exactly**. Verified numerically: `max |NDBI + NDMI| = 0.0` over all 39,624,101 valid pixels; Spearman ρ = −1.0000. The paper's reduced feature set drops NDBI and this identity is reported as a methodological finding |
| F2 | NDVI mean/median are correlated 0.9793 | VERIFIED | `results/feature_correlation_matrix.csv` |
| F3 | Building coverage/area correlate ≈0.999 | VERIFIED | 0.9990 |
| F4 | Population / population density correlate ≈0.998 | VERIFIED | 0.9985 |
| F5 | NDBI has the strongest univariate association with the target (ρ 0.7416) | VERIFIED | Identical to NDMI's ρ by construction; paper reports the axis once, as "built/dry vs vegetated/moist" |
| F6 | Boundary cells are full 250 m cells | **NOT SUPPORTED** | `grid.py:126-136` clips cells to the AOI polygon then renumbers; min `area_m2` = **10.54 m²**. 30 sliver cells exist, and their p95 comes from ~1 pixel. Paper reports the count and their exclusion from claims |

## SHAP attribution

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| S1 | Neighbourhood features contribute 45.43% of total attribution | **VERIFIED (corrected)** | Stored relative importances sum to **45.424447%** → **45.42%**. The 45.43% figure is recoverable only by summing the individually rounded components (16.04 + 15.34 + 7.67 + 6.38). Paper uses 45.42% |
| S2 | NDMI (focal + neighbourhood) contributes 49.38% | VERIFIED | 34.042327 + 15.336488 = 49.378815% → 49.38% |
| S3 | Attribution is computed on the operational context model | VERIFIED | `context_shap.py`, 9 features, 5,000 stratified cells, `random_state=42` |
| S4 | SHAP values are causal explanations of heat | **NOT SUPPORTED** | TreeSHAP is a model-attribution method; the code and paper describe associations between predictors and a single-date LST surface. No causal claim is made anywhere |
| S5 | SHAP additivity holds exactly | **VERIFIED (corrected)** | `context_shap.py:273-292` uses `shap.TreeExplainer(feature_perturbation="interventional")` with a **100-row background sample**, so `base + Σφ` is approximate. Measured residual: mean abs 4.59 × 10⁻⁴ °C, median 1.5 × 10⁻⁸ °C, but **max 1.0917 °C** with 19 of 5,000 cells (0.38%) above 0.01 °C. Paper reports this diagnostic |
| S6 | SHAP importances were regenerated and reproduce | VERIFIED | `paper/figures/fig4_shap.png`; `results/context_shap_feature_importance.csv` byte-identical to the pre-audit copy at `/tmp/opencode/repro/orig_ctx_shap.csv` |

## Uncertainty quantification

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| U1 | 95% intervals achieve 95.00% coverage | **VERIFIED (corrected)** | That value comes from `adaptive_uncertainty_results.csv`, whose calibration factor is the 95th percentile of `|error| / predicted_abs_error` computed **on the same cells** it then reports (`adaptive_uncertainty.py:266-272, 324-327`). Coverage is ~95% by construction. Paper does **not** headline it |
| U2 | Mean 95% interval width 5.8623 °C | **VERIFIED (corrected)** | Same artifact; the implied half-width is ~2.5× the model's actual 1.174 °C MAE. Demoted to a relative-risk ranking layer |
| U3 | Hottest-quintile coverage 88.63% | VERIFIED | `adaptive_uncertainty_results.csv` |
| U4 | Cross-fitted intervals achieve 94.94% coverage | VERIFIED | Computed from `spatial_uncertainty_results.csv` (21,815 rows): 94.9393%. **This is the paper's primary uncertainty result** |
| U5 | Cross-fitted mean 95% width 6.1356 °C | VERIFIED | 6.1356 °C, effectively constant across quintiles (6.1340–6.1381) |
| U6 | Cross-fitted hottest-quintile coverage 87.85% | VERIFIED | 87.8468% |
| U7 | Intervals have formal conformal 95% coverage | **NOT SUPPORTED** | `spatial_uncertainty.py:112-164` uses `np.quantile` with linear interpolation rather than the conformal index ⌈(n+1)(1−α)⌉, and calibration residuals for fold *k* come from models trained on folds including *k*. Paper explicitly declines the conformal claim, citing Lei et al. and Angelopoulos & Bates |
| U8 | `tail_uncertainty_refinement.py` demonstrates calibrated intervals | **NOT SUPPORTED** | Global and per-regime thresholds are derived from the same cells that are evaluated; Q5 coverage of 0.95001 for the target-regime strategy is true by construction. Excluded; reported only as the diagnostic that motivated the adaptive layer |

## Priority, eligibility, and interventions

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| O1 | Priority never uses the observed target | VERIFIED | `operational_priority.py:186-197, 524-574`; exposure enters only through out-of-fold `predicted_lst_p95` |
| O2 | 23.3% of cells are eligible for ≥1 intervention | **VERIFIED (corrected)** | 5,084 of 21,815 cells. But every gate is a conjunction of **in-sample median/P25 quantiles** (`intervention_eligibility.py:93-104`), so 23.3% is a restatement of the thresholds, not an empirical finding. Paper reports it as a property of the rule, not of Bengaluru |
| O3 | Equity priority is an independent equity criterion | **VERIFIED (corrected)** | `structural_vulnerability = robust_norm(log1p(pop_density)) × (1 − capacity_score)`, and `capacity_score` contains no population term, so equity-priority membership is a monotone function of population density. Paper describes the constraint as a **density-weighted benefit-share floor** |
| O4 | Per-intervention effect sizes determine which intervention is preferred | **NOT SUPPORTED** | `operational_intervention_benefits_v2.py:318-320` multiplies each cell's benefit by a **cell-invariant constant per intervention type**, so `EFFECT` cannot change any ranking; it only rescales totals via the budget trade-off. Paper reports effect and cost as scenario parameters and makes no effectiveness ranking claim |
| O5 | Top-100 priority overlap figures reported by the app | **NOT SUPPORTED** | `operational_priority.py:476-489` prints a set-intersection **count** (0–100) with a `%` suffix. Any overlap percentage quoted from console output is wrong. Excluded |
| O6 | HVI is a validated index | PARTIAL | `build_hvi.py:222-255` uses an unjustified expert-weighted, multiplicative form with no sensitivity analysis; `sensitivity` is a single variable (log population density). Paper presents it as one descriptive layer, not a validated index, and lists weight sensitivity as future work |

## Optimization

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| P1 | MILP beats greedy by 28.34% | **VERIFIED (corrected)** | 1616.4362 → 2074.490542 = +28.34%, from `capacity_optimizer_results.csv`. But this configuration is **capacity-bound, not budget-bound**: total cost 6,700 against a nominal budget of 18,815 (utilisation 0.356), and the MILP simply saturates all four capacity ceilings (1000+1500+800+800 = 4,100 cells) while the benefit-per-cost greedy stops at 3,629. Paper reports it as **capacity saturation**, not budget efficiency |
| P2 | Budget fraction 0.5 constrains the capacity-constrained allocation | **NOT SUPPORTED** | `capacity_optimizer.py:439-447` computes the budget but it never binds in any row of the artifact |
| P3 | The α = 0.6 equity constraint changes the capacity-constrained solution | **NOT SUPPORTED** | `MILP_Equity(0.6)` is byte-identical to `MILP_No_Equity` (share 0.621571 > 0.60, non-binding) |
| P4 | α = 0.7 costs 3.74% of benefit | VERIFIED | 2074.490542 → 1996.896382 = −3.74%, share pinned at 0.700004 |
| P5 | Equity floors cost 5.22% (α=0.6) and 18.76% (α=0.7) of benefit | VERIFIED | `operational_optimizer_results.csv`: 2470.689657 → 2341.735253 (−5.22%) → 2007.309146 (−18.76%); equity share pinned at 0.600037 / 0.700005. **Primary optimization result** |
| P6 | The unconstrained MILP optimum is verified | VERIFIED | `solver_status = optimal`, `relative_mip_gap` ≤ 7.6 × 10⁻⁵ for every equity row; the unconstrained row reproduces the operational-ranking baseline exactly (2470.689657), an independent optimality check |
| P7 | Benefits and costs are physical quantities | **NOT SUPPORTED** | `operational_optimizer.py:245` labels benefit a "relative scenario benefit"; costs are dimensionless index units. Paper never converts them to °C, currency, or health outcomes |
| P8 | Budget-equity frontier shows a genuine trade-off | VERIFIED | `budget_equity_frontier.csv`: budget `= 6700 × f` binds (utilisation 0.9999–1.0000); benefit rises 859.57 → 1502.00 → 1874.02 → 2074.49 as the equity share falls 0.940 → 0.823 → 0.725 → 0.622. α = 0.8 at f = 0.75 forces benefit to 1750.75 |
| P9 | `pareto_frontier.csv` is a global Pareto front | **NOT SUPPORTED** | `pareto_analysis.py:56-59` computes dominance **only within a fixed `budget_fraction`**, so low-budget/high-equity points are never compared against high-budget/low-equity points. Excluded |
| P10 | Equity targets ≤ 0.60 are informative | **NOT SUPPORTED** | In `budget_equity_frontier.csv`, the rows for `None`/0.5/0.6 are exact duplicates at every budget level. Paper reports only α = 0.70 and 0.80 as binding |
| P11 | Reported cost/effect/capacity values | VERIFIED | cost = 1.0 / 1.4 / 2.0 / 2.5; effect = 1.0 / 1.2 / 1.1 / 1.3; capacity = 1000 / 1500 / 800 / 800 for tree / shade / cool roof / cooling |
| P12 | Four optimizers use one consistent budget definition | **NOT SUPPORTED** | `n_cells × mean(cost) × f` (`operational_optimizer.py`, `capacity_optimizer.py`), `max_capacity_cost × f` (`budget_equity_frontier.py:589-595`), and capacity-scaled (`intervention_sensitivity.py:366-375`). Paper uses the two binding configurations and states each budget formula explicitly |

## Robustness

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| R1 | 500 Monte Carlo sensitivity runs | VERIFIED | `intervention_sensitivity_final.csv`, 500 rows, `SEED=42` |
| R2 | 475 runs certified optimal | VERIFIED | 475 `optimal`, **25 time-limited** (HiGHS status 13, 20 s limit, no `mip_rel_gap` passed) |
| R3 | All 500 runs are certified optimal | **VERIFIED (corrected)** | Only 475 of 500. The 25 failures are written as `NaN` and are not flagged in any summary artifact |
| R4 | All equity targets are met | VERIFIED | All **192 certified-optimal** equity runs satisfy their target; 8 equity runs and 17 selection runs are time-limited. Paper states the qualified form |
| R5 | Perturbation ranges | VERIFIED | cost and effect ~ U(0.7, 1.3); capacity ~ U(0.8, 1.2); 100 selection + 50 equity draws |
| R6 | Budget and equity levels are independent replications | **NOT SUPPORTED** | Each capacity draw is reused across all budget/equity levels, so the 500 runs are 150 distinct parameter draws evaluated over multiple settings. Paper reports this |
| R7 | Results reproduce on re-execution | VERIFIED | Full re-run of the benchmark, reduced benchmark, context benchmark, feature audit, context SHAP, and both uncertainty modules. Base-model fold metrics agreed to ≤4.44 × 10⁻¹⁶; context metrics exactly equal; `context_shap_feature_importance.csv` byte-identical; adaptive and cross-fitted interval results reproduced exactly |
| R8 | `results/heatscope_operational_priority.gpkg` predictions match the regenerated model output | VERIFIED | Max absolute difference **0.0** across all 21,815 cells between the 2025-09-15 artifact and the 2026-10-03 regeneration. The ML → uncertainty → priority chain is bit-reproducible |

## Software and reproducibility

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| C1 | `requirements.txt` reproduces the environment | **NOT SUPPORTED** | It lists 8 unpinned packages and omits `scikit-learn`, `shap`, `scipy`, `xgboost`, `pystac-client`, `planetary-computer`, `requests`. XGBoost is wrapped in `try/except ImportError` (`spatial_ml_benchmark.py:230-266`), so the reported XGBoost row exists only in an environment that `requirements.txt` does not describe |
| C2 | Re-running the pipeline reproduces the reported artifacts | **NOT SUPPORTED** | `satellite.py:40-41, 66-80` sets the search window from `datetime.now()` and selects the lowest-cloud scene in that window. Re-running selects a different scene and silently invalidates every downstream artifact. Paper reports the single pinned scene ID and states that the acquisition step is not idempotent |
| C3 | Repository is self-contained | **NOT SUPPORTED** | `.gitignore` excludes `data/raw/`, `data/processed/`, `results/`, `models/`, `figures/`. The public commit contains code only; all artifacts are local |
| C4 | The Streamlit app displays current results | **NOT SUPPORTED** | `demo.py:41-60` hard-codes every displayed metric in a literal dict `R`; nothing is read from `results/`. `demo_data/` holds byte-identical third copies of three artifacts |
| C5 | Pipeline has an orchestrator | **NOT SUPPORTED** | No module has an `if __name__ == "__main__"` guard, `argparse`, or `sys.argv`. Every module is a side-effecting top-level script run in a fixed manual order (documented in REPRODUCTION.md) |
| C6 | Recorded hardware and runtime | NOT RECORDED | No CPU, RAM, or wall-clock metadata is stored anywhere. See MISSING_EVIDENCE M6 |

## Citation integrity

| # | Claim | Verdict | Evidence / correction |
|---|---|---|---|
| C7 | NDBI cited to Xu 2007 / Liang 2007 | **NOT SUPPORTED** | Both wrong. NDBI = Zha, Gao & Ni, *IJRS* 24(3):583–594, 2003 |
| C8 | WorldPop DOI `10.1038/sdata.2017.41` | **NOT SUPPORTED** | Unrelated article. Correct DOI `10.1038/sdata.2017.1` |
| C9 | "Mar et al. 2020" building footprints | **NOT SUPPORTED** | No such record. Replaced with the actual Overture Maps attribution and Heris et al. 2020 as related work |
| C10 | Roberts et al. 2017 in *Methods in Ecology and Evolution* | **NOT SUPPORTED** | That DOI is a different article. Correct record is *Ecography* 40(8):913–929 |
| C11 | All cited references verified | VERIFIED | 41 of 41 references in the paper's reference list were resolved via Crossref or the official publisher page. See `paper/BIBLIOGRAPHY.md` |
| C12 | Framework is novel relative to prior work | PARTIAL | No single prior work combining block-grouped spatial CV, operationalised SHAP, and an equity-constrained MILP for heat-adaptation siting was located. Stated as "no directly comparable framework was identified in the literature we verified", not as an absolute priority claim |