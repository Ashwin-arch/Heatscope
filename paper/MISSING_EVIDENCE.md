# HeatScope — Missing Evidence and Author Action List

Items the audit could not resolve from the repository. Severity:

- **BLOCKER** — the paper cannot make the intended claim until resolved.
- **MAJOR** — a claim must be narrowed, or the code fixed, before submission.
- **MINOR** — should be fixed for reviewer comfort but does not invalidate a claim.

Author decisions taken on 2026-10-03 are recorded at the bottom so the reasoning is auditable.

---

## M1 — No verified peer-reviewed precedent for MILP allocation of heat mitigation · BLOCKER for novelty framing

The Related Work section cannot cite a peer-reviewed mixed-integer or linear program that allocates
urban heat mitigation (tree planting, shade, cool roofs, cooling centres) under budget and equity
constraints. Crossref queries returned only tangential urban-forestry and cool-materials literature.

Consequence: the optimization contribution cannot be positioned against a literature baseline. The
paper currently states only that no directly comparable framework was identified, which is honest but
weak.

Action: search Google Scholar / Scopus for urban-tree siting and heat-adaptation resource allocation
with LP/MILP formulations, and for equity-constrained environmental allocation generally. Even two or
three solid references would fix this. Do **not** substitute a loosely related paper.

## M2 — Unverified equity/canopy-cooling reference · MINOR

"Urban tree canopy has greater cooling effects in socially vulnerable communities in the US"
(*One Earth*, 2022) would be a strong equity citation, but its DOI, volume, and article number could
not be resolved. Verify directly on the publisher site, or drop it.

## M3 — WorldPop product version and licence · MAJOR

`data/raw/worldpop_india_2025.tif` carries a 2025 vintage in its filename, but only WorldPop 100 m
/ India v2 metadata was verifiable. There is no download script, no checksum, no licence record, and
no version string anywhere in the repository. Additionally `population.py:78-99` performs
centre-in-pixel selection (`all_touched=False`) with no area-weighted fractional allocation, so edge
cells can be mis-counted by up to one 100 m pixel.

Action: record the exact WorldPop release DOI, version, licence, and a SHA-256 of the GeoTIFF; and
switch to proportional-area aggregation.

## M4 — No download script for the building layer · MAJOR

`buildings.py:22` reads `data/raw/buildings/india_123303312.csv.gz`, whose origin is recoverable only
from line 13,308 of `data/raw/building_dataset_links.csv`: an Overture Maps Buildings theme snapshot
dated 2026-08-13, single quadkey 123,303,312, ODbL. The downloader is absent, so the layer cannot be
rebuilt.

Also unresolved in the aggregation itself (`buildings.py:99-187`): no de-duplication of overlapping
footprints (so `building_area_m2` can exceed true footprint area and `building_coverage_pct` can
exceed 100%), no minimum-intersection-area gate (so boundary slivers increment `building_count`), and
the `height` property is parsed away without use even though Overture supplies it — for 250 m thermal
cells, building height is arguably the most physically relevant built-form variable available.

Action: add the downloader with the pinned snapshot URL, add `unary_union` per cell, add a
minimum-area gate, and either use `height` or state explicitly that it was unavailable.

## M5 — Scene acquisition is not idempotent · MAJOR

`satellite.py:40-41, 66-80` derives the search window from `datetime.now()` and selects the
lowest-`eo:cloud_cover` item in that window with no date tie-break. Re-running on a different date
selects a different scene and silently invalidates every downstream artifact. The `scene_metadata.csv`
file records the choice but nothing downstream ever re-reads it.

Action: pin `start_date`/`end_date`, pin the chosen `item.id`, and assert that the downloaded scene
matches `scene_metadata.csv` before continuing. The paper currently reports the pinned scene
`LC08_L2SP_144051_20260401_02_T1` and states this limitation.

## M6 — No hardware or runtime metadata · MAJOR

No CPU model, core count, RAM, OS, or wall-clock timing is recorded for any stage. Reviewers of an
empirical ML paper routinely ask for this, and 5-fold CV over 21,815 cells × 5 models is not cheap.

Action: capture `platform`, `python`, package versions, per-stage wall-clock, and peak memory into a
machine-readable run manifest and cite it in the paper. Hardware can be read from the current
machine; historical runtimes must be re-measured or reported as unavailable.

## M7 — No pinned environment · MAJOR

`requirements.txt` lists 8 unpinned packages and omits `scikit-learn`, `shap`, `scipy`, `xgboost`,
`pystac-client`, `planetary-computer`, and `requests`. XGBoost is wrapped in `try/except ImportError`,
so its benchmark row is an artifact of an unpinned environment rather than a guaranteed result.

Action: publish a fully pinned `requirements-lock.txt` (or `uv.lock`) capturing the versions the
reported results were produced with.

## M8 — Four mutually inconsistent budget definitions · MAJOR

- `operational_optimizer.py:461-466` and `capacity_optimizer.py:439-447`: `n_cells × mean(cost) × f`
  → 18,815 at f = 0.5. **Never binds** in the reported capacity results; in the operational results
  total cost is 8,702.7 against a 9,407.7 budget.
- `budget_equity_frontier.py:589-595`: `max_capacity_cost × f` → 6,700 at f = 1.0. **Binds** at
  utilisation 0.9999–1.0000.
- `intervention_sensitivity.py:366-375`: capacity-scaled budget.
- `capacity_frontier.py`: `BUDGET_FRACTION = 0.50` present but inert.

Action: consolidate to one definition. The paper works around this by reporting the two binding
configurations with their formulas stated explicitly, which is defensible but will not satisfy a
careful reviewer.

## M9 — 25 of 500 sensitivity runs failed silently · MAJOR

`intervention_sensitivity.py:52` sets a 20 s limit and, unlike every other optimizer, passes no
`mip_rel_gap`. 25 of 500 runs returned HiGHS status 13 (time limit) and were written as `NaN` in
`intervention_sensitivity_final.csv`; no summary artifact flags them. The paper reports 475/500 and
qualifies the equity-violation statement accordingly.

Action: raise the limit, pass `mip_rel_gap`, and record failed runs with an explicit status instead of
`NaN`.

## M10 — Sensitivity design is not fully independent · MINOR

Each of the 150 capacity/effect/cost draws is reused across all budget and equity levels, so
`budget_fraction` and `equity_target` are not independent replications. The paper reports 500 runs over
150 distinct draws rather than implying 500 independent scenarios.

## M11 — HVI weights are unjustified and untested · MAJOR

`build_hvi.py:222-255` hard-codes exposure weights 0.50/0.30/0.20 and capacity weights 0.45/0.35/0.20
with no derivation, no citation, and no sensitivity analysis. The multiplicative form
`exposure × sensitivity × (1 − capacity)` collapses cells that are simultaneously low-vegetation,
low-population, and high-built to ≈0, and the subsequent P01–P99 rescale is a second, non-idempotent
transform. The `sensitivity` component is a single variable (log population density) and omits age,
income, health, and occupation.

Action: either state explicitly that the weights are expert-elicited and report a weight-sensitivity
analysis, or drop HVI as a headline construct. The paper treats it as one descriptive layer only.

## M12 — `equity_priority` is population density · MAJOR

`operational_priority.py:199-253`: `structural_vulnerability = robust_norm(log1p(pop_density)) × (1 −
capacity_score)`, and `capacity_score` contains no population term. Equity-priority membership is
therefore a monotone function of population density, and the optimizer's equity constraint is a
density-weighted benefit-share floor rather than an independent equity criterion.

Action: if a genuine multi-dimensional equity criterion is wanted, it needs health, demographic, or
socioeconomic layers that the repository does not currently contain. The paper describes the
constraint accurately as implemented.

## M13 — Eligibility thresholds are in-sample quantiles · MAJOR

`intervention_eligibility.py:93-104` sets every gate from in-sample P25/P50/P75 quantiles, so the
resulting "23.3% of cells are eligible for at least one intervention" is a restatement of the
thresholds rather than an empirical finding about Bengaluru.

Action: replace with externally justified thresholds (canopy targets, planning norms) or report the
rate purely as a property of the rule. The paper does the latter.

## M14 — `operational_priority.py:476-489` prints a count as a percentage · MINOR

Three "TOP-100 PRIORITY OVERLAP" console lines format a set-intersection cardinality (0–100) with a
`%` suffix. Any overlap percentage taken from that output is wrong by construction. The paper quotes
no overlap percentages.

Action: `100 * len(intersect) / 100`.

## M15 — `demo.py` hard-codes every displayed metric · MINOR

`demo.py:41-60` defines a literal dict `R` containing all reported metrics; nothing is read from
`results/`. `demo_data/` holds byte-identical third copies of three artifacts for Cloud deployment.
The public demo can therefore display stale values after any re-run.

Action: replace `R` with `pd.read_csv` against the producer CSVs; generate `demo_data/` from
`results/` in CI.

## M16 — Archived pre-fix scripts remain in the tree · MINOR

`intervention_sensitivity_before_equity_patch.py`,
`intervention_sensitivity_before_final_equity_fix.py`,
`intervention_sensitivity_pre_objective_fix.py` (875–877 lines each) and
`demo_before_google_maps.py` (812 lines) are superseded copies. Also dead:
`optimization_baseline.py`, `optimization_v2.py`, `optimization_refine.py`,
`intervention_scores.py`, `intervention_baseline.py`, `merge_features.py`,
`tail_uncertainty_refinement.py`, and `reduced_shap.py` / `shap_analysis.py` (superseded by
`context_shap.py`). Their stale CSVs still sit in `results/` and are the most likely source of
conflicting numbers.

Action: delete or move to an `archive/` directory before making the submission repository public, and
delete the orphaned CSVs.

## M17 — SHAP additivity residual is large for a small subset of cells · MINOR

`context_shap.py:273-292` builds `shap.TreeExplainer` with `feature_perturbation="interventional"`
and a 100-row background sample, then passes `check_additivity=False`. The independent check in the
same script measures a median residual of 1.5 × 10⁻⁸ °C but a **maximum of 1.0917 °C**, with 19 of
5,000 cells (0.38%) above 0.01 °C. Relative to the model's 1.174 °C MAE, the worst-case residual is
comparable to a full model error.

Cause: interventional TreeSHAP estimates E[f(x_S)] by Monte Carlo over the small background sample, so
additivity holds only with respect to the approximated expectation. Action: increase the background
sample (e.g. 200–1,000 stratified rows) and re-check, or report the residual distribution as a
diagnostic. The paper reports the residual statistics as-is.

## M18 — Thirty boundary sliver cells enter the analysis · MINOR

`grid.py:126-136` clips grid cells to the AOI polygon and renumbers them; `area_m2` ranges from
**10.54 m²** to 62,500 m². Population density and building coverage correctly divide by the clipped
area, but the p95 target for a 10.54 m² cell comes from approximately one 30 m pixel. 30 such cells
enter training, cross-validation, and optimization indistinguishable from full cells.

Action: exclude cells below a minimum area (e.g. 25% of nominal) and report the exclusion count. The
paper reports the count.

## M19 — QA_PIXEL masking is incomplete · MINOR

`preprocess.py:111-125` masks fill, dilated cloud, cirrus, cloud, cloud shadow, and snow, but never
tests the **water bit (bit 7)** and applies no cloud-confidence test. Water pixels are retained in
surface-temperature and index computation.

Action: add the water mask or state that water bodies are not excluded.

## M20 — Hottest-quintile bias is large and unexplained · MAJOR (scientific)

The model systematically **under-predicts the hottest cells by 1.38 °C** (mean observed 45.80 °C vs
mean predicted 44.42 °C in the hottest quintile; +1.50 °C before adding context features). This is
the single most consequential empirical finding in the audit: it explains why the hottest quintile is
the only group whose cross-fitted interval coverage falls below nominal (87.85%), and it means the
model systematically understates exposure precisely where exposure matters most.

Action: this is a finding to report and explain, not to hide. Candidate mechanisms worth testing:
regression-to-the-mean under squared/absolute loss on a bounded upper-tail statistic; edge-of-city
cells with few valid pixels; systematic under-representation of extreme cells in training. Note that
adding context features reduces the bias by only 8.08%.

---

## Author decisions recorded

Taken 2026-10-03, after reviewing the audit findings.

| Decision | Choice | Consequence for the paper |
|---|---|---|
| Which optimization result headlines the paper | Budget/equity-bound optimizer as primary; capacity-saturation result secondary | The 28.34% figure is retained but relabelled as capacity saturation, with the non-binding budget disclosed. The equity floors (−5.22% at α = 0.6, −18.76% at α = 0.7) and the binding budget-equity frontier become the primary allocation results |
| Which uncertainty result is the validated one | Cross-fitted `spatial_uncertainty.py` as primary; adaptive layer as a relative-risk ranking only | 94.94% coverage / 6.14 °C width / 87.85% hottest-quintile coverage are the reported intervals. The adaptive 95.00% / 5.86 °C values are not presented as validated coverage |
| How to define the target | Reframe as within-cell upper-tail exposure for a single date | No extreme-heat, heatwave, or climatology language anywhere. A multi-date composite is listed as the primary limitation and future work |
| Scope of code changes | Documentation and framing only; no pipeline changes | Items M3–M20 remain open. Every one of them is disclosed in the paper's Limitations section rather than silently omitted |
| Weather layer | Removed from the paper's method and claims | The paper states that a single city-point meteorological series was retrieved and deliberately excluded because it carries no spatial information and entered no model |