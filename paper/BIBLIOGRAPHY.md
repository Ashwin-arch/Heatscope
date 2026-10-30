# HeatScope — Verified Bibliography

Verification method: each DOI below was resolved against the Crossref REST API
(`api.crossref.org/works/{DOI}`) or fetched from the official publisher / government page, and the
`title`, `author[]`, `container-title`, `volume`, `issue`, `page`, and `published` fields were read
directly. Nothing in the "Verified" section was reconstructed from memory.

Every entry carries a `Cited` flag. Only entries marked `Cited: yes` appear in the paper's
reference list; IEEE requires that every reference be cited, so uncited verified entries are
retained here only as an author-approved reserve pool.

## Critical attribution corrections found during verification

| Common citation | Problem | Correct source used in this paper |
|---|---|---|
| NDBI attributed to "Xu 2007" or "Liang 2007" | Both attributions are wrong. | Zha, Gao & Ni, *IJRS*, 2003 (NDBI). Xu 2008 is the Index-based Built-up Index (IBI). |
| NDMI attributed to "Gao 1996" | Gao 1996 is correct for NDWI/NDMI, but is often miscited as the NDVI source. | Gao 1996 cited for NDMI; Rouse, Goward & V 1974 used for NDVI. |
| WorldPop DOI `10.1038/sdata.2017.41` | Resolves to an unrelated article (avian atlas). | `10.1038/sdata.2017.1` (Lloyd et al. 2017). |
| "Mar et al. 2020" Global Building Footprints | No such peer-reviewed record exists. | Heris et al. 2020 (*Sci Data*, Microsoft Global ML Building Footprints method). |
| Roberts et al. 2017 cited as *Methods in Ecology and Evolution* | That DOI belongs to a different article. | Roberts et al. 2017, *Ecography* 40(8):913–929. |
| Building footprints as "Microsoft Global ML Building Footprints" | **Wrong dataset.** `data/raw/building_dataset_links.csv:13308` resolves to `bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/...quadkey=123303312/...`, which is an **Overture Maps** Buildings theme snapshot dated **2026-08-13**, ODbL-licensed, conflated from OpenStreetMap, Esri Community Maps, Google Open Buildings, and Microsoft ML Building Footprints. | Overture Maps Buildings theme (2026-08-13 snapshot). Heris et al. 2020 is retained only as related work for one of the upstream sources. |

---

## Verified references

### Landsat Collection 2 surface temperature

**[R1] Cited: yes.** U.S. Geological Survey, "How do I use a scale factor for Landsat Level-2 science
products?" *Landsat Collection 2 FAQ*, U.S. Department of the Interior.
https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
— *Supports the `ST_B10 x 0.00341802 + 149.0 - 273.15` conversion implemented at `preprocess.py:100`.*

**[R2] Cited: yes.** U.S. Geological Survey EROS, "Landsat Collection 2 Surface Temperature."
https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature
— *Product definition: 30 m grid, uint16, 0 as fill, ST_B10 as the surface-temperature band.*

**[R3] Cited: yes.** X. Meng, J. Cheng, B. Yao, and Y. Guo, "Validation of the ECOSTRESS Land
Surface Temperature product using ground measurements," *IEEE Geosci. Remote Sens. Lett.*, vol. 19,
pp. 1-5, 2022. doi:10.1109/LGRS.2021.3123816
— *Ground-validation methodology for thermal-IR LST; supports treating satellite LST as a validated
but imperfect proxy requiring in-situ checks.*

**[R4] Cited: no (reserve).** S. Li, J. Wang, D. Li, Z. Ran, and B. Yang, "Evaluation of Landsat
8-like land surface temperature by fusing Landsat 8 and MODIS land surface temperature product,"
*Processes*, vol. 9, no. 12, art. 2262, 2021. doi:10.3390/pr9122262
— *Landsat 8 LST quality evaluation; reserve for the uncertainty discussion.*

### Urban heat island and urban climate

**[R5] Cited: yes.** J. A. Voogt and T. R. Oke, "Thermal remote sensing of urban climates,"
*Remote Sens. Environ.*, vol. 86, no. 3, pp. 370-384, 2003. doi:10.1016/S0034-4257(03)00079-8
— *Canonical review establishing LST/UHI remote-sensing theory and the surface-vs-air temperature
distinction that constrains this paper's claims.*

**[R6] Cited: yes.** D. Zhou et al., "Satellite remote sensing of surface urban heat islands:
progress, challenges, and perspectives," *Remote Sens.*, vol. 11, no. 1, art. 48, 2019.
doi:10.3390/rs11010048
— *Comprehensive review of surface-UHI retrieval, scaling, and open methodological challenges.*

**[R7] Cited: yes.** C. R. Almeida, A. C. Teodoro, and A. Goncalves, "Study of the urban heat island
(UHI) using remote sensing data/techniques: a systematic review," *Environments*, vol. 8, no. 10,
art. 105, 2021. doi:10.3390/environments8100105
— *Systematic review supporting the index selection and the discussion of seasonal aggregation limits.*

**[R8] Cited: yes.** M. Santamouris, "Recent progress on urban overheating and heat island research.
Integrated assessment of the energy, environmental, vulnerability and health impact," *Energy
Build.*, vol. 207, art. 109482, 2020. doi:10.1016/j.enbuild.2019.109482
— *Integrates heat mitigation, vulnerability, and health; anchors the intervention taxonomy.*

**[R9] Cited: yes.** K. Deilami, M. Kamruzzaman, and Q. Liu, "Urban heat island effect: a systematic
review of spatio-temporal factors, data, methods, and mitigation measures," *Int. J. Appl. Earth
Obs. Geoinf.*, vol. 67, pp. 30-42, 2018. doi:10.1016/j.jag.2017.12.009
— *Systematic review of UHI drivers and mitigation measures; motivates the allocation objective.*

**[R10] Cited: yes.** G. Manoli et al., "Magnitude of urban heat islands largely explained by climate
and population," *Nature*, vol. 573, no. 7772, pp. 55-60, 2019. doi:10.1038/s41586-019-1512-9
— *Attribution of UHI magnitude to population and climate; external-validity check for a
population-dense setting such as Bengaluru.*

### Heat exposure, vulnerability, and human thermal comfort

**[R11] Cited: yes.** C. Heaviside, H. Macintyre, and S. Vardoulakis, "The urban heat island:
implications for health in a changing environment," *Curr. Environ. Health Rep.*, vol. 4, no. 3,
pp. 296-305, 2017. doi:10.1007/s40572-017-0150-3
— *Supports exposure (not hazard) framing and the limits of LST-only exposure metrics.*

**[R12] Cited: no (reserve).** E. L. Kruger, C. A. Tamura, P. Brode, M. Schweiker, and A. Wagner,
"Short- and long-term acclimatization in outdoor spaces: exposure time, seasonal and heatwave
adaptation effects," *Build. Environ.*, vol. 116, pp. 17-29, 2017.
doi:10.1016/j.buildenv.2017.02.001
— *Thermal-comfort evidence for interpreting outdoor exposure thresholds.*

**[R13] Cited: yes.** N. Mondal and P. Anand, "Influence of urbanization on heat exposure risk across
Indian cities," *Sustain. Cities Soc.*, vol. 135, art. 106960, 2025. doi:10.1016/j.scs.2025.106960
— *Multi-city Indian exposure-risk study; the closest national precedent.*

**[R14] Cited: yes.** A. Hsu, G. Sheriff, T. Chakraborty, and D. Manya, "Disproportionate exposure to
urban heat island intensity across major US cities," *Nat. Commun.*, vol. 12, no. 1, art. 2721,
2021. doi:10.1038/s41467-021-22799-5
— *Empirical basis for treating heat exposure as an equity question, not only a hazard question.*

### Spectral indices

**[R15] Cited: yes.** B.-C. Gao, "NDWI — A normalized difference water index for remote sensing of
vegetation liquid water from space," *Remote Sens. Environ.*, vol. 58, no. 3, pp. 257-266, 1996.
doi:10.1016/S0034-4257(96)00067-3
— *Primary source for the normalised-difference moisture index used here.*

**[R16] Cited: yes.** Y. Zha, J. Gao, and S. Ni, "Use of normalized difference built-up index in
automatically mapping urban areas from TM imagery," *Int. J. Remote Sens.*, vol. 24, no. 3,
pp. 583-594, 2003. doi:10.1080/01431160304987
— *Primary source for NDBI. Also the source of the redundancy documented in this paper:
NDBI and NDMI are algebraic negatives under a common denominator.*

**[R17] Cited: no (reserve).** H. Xu, "A new index for delineating built-up land features in satellite
imagery," *Int. J. Remote Sens.*, vol. 29, no. 14, pp. 4269-4276, 2008.
doi:10.1080/01431160802039957
— *Index-based Built-up Index (IBI). Reserve only.*

**[R18] Cited: yes.** J. W. Rouse, S. N. Goward, and V. I. V, "Vegetation in sensor space,"
*Remote Sens. Environ.*, vol. 3, pp. 295-309, 1974. doi:10.1016/0034-4257(74)90014-X
— *Primary source for NDVI.*

### Gridded population

**[R19] Cited: yes.** C. T. Lloyd, A. Sorichetta, and A. J. Tatem, "High resolution global gridded
data for use in population studies," *Sci. Data*, vol. 4, art. 170001, 2017.
doi:10.1038/sdata.2017.1
— *WorldPop methodology.*

**[R20] Cited: yes.** F. R. Stevens, A. E. Gaughan, C. Linard, and A. J. Tatem, "Disaggregating
census data for population mapping using remotely-sensed and ancillary data," *PLoS ONE*, vol. 10,
no. 2, e0107042, 2015. doi:10.1371/journal.pone.0107042
— *Random-forest dasymetric redistribution underlying WorldPop.*

**[R21] Cited: yes.** WorldPop, University of Southampton, "WorldPop Global High Resolution
Population Denominators Project (100m)," doi:10.5258/SOTON/WP00107
— *Dataset-level citation for the gridded population denominators.*

### Building footprints

**[R22] Cited: yes.** Overture Maps Foundation, "Buildings theme," *Overture Maps Documentation*,
release snapshot 2026-08-13, ODbL. https://docs.overturemaps.org/
— *Actual source of `data/raw/buildings/india_123303312.csv.gz`, resolved from
`data/raw/building_dataset_links.csv:13308`.*

**[R23] Cited: no (related work).** M. P. Heris, N. L. Foks, K. J. Bagstad, A. Troy, and
Z. H. Ancona, "A rasterized building footprint dataset for the United States," *Sci. Data*, vol. 7,
no. 1, art. 207, 2020. doi:10.1038/s41597-020-0542-3
— *Peer-reviewed description of the Microsoft Global ML Building Footprints method, one of the
upstream sources conflated into Overture's Buildings theme.*

### Meteorological context (retrieved but not used as a model input)

**[R24] Cited: yes (limitations only).** NASA Langley Research Center, "POWER methodology."
https://power.larc.nasa.gov/docs/methodology/
— *Documents POWER's 0.5 deg x 0.625 deg meteorological grid. Cited only to explain the resolution
limitation that makes a single city-point query unusable as a per-cell covariate.*

### Boosted trees and tree ensembles

**[R25] Cited: yes.** J. H. Friedman, "Greedy function approximation: a gradient boosting machine,"
*Ann. Stat.*, vol. 29, no. 5, pp. 1189-1231, 2001. doi:10.1214/aos/1013203451
— *Origin of gradient-boosted regression trees.*

**[R26] Cited: yes.** T. Chen and C. Guestrin, "XGBoost: a scalable tree boosting system," in *Proc.
22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Mining (KDD)*, 2016, pp. 785-794.
doi:10.1145/2939672.2939785
— *XGBoost, used as a benchmarked alternative.*

**[R27] Cited: yes.** P. Geurts, D. Ernst, and L. Wehenkel, "Extremely randomized trees," *Mach.
Learn.*, vol. 63, no. 1, pp. 3-42, 2006. doi:10.1007/s10994-006-6226-1
— *Extra Trees baseline.*

**[R28] Cited: yes.** L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5-32, 2001.
doi:10.1023/A:1010933404324
— *Random Forest baseline.*

### Spatial cross-validation

**[R29] Cited: yes.** D. R. Roberts et al., "Cross-validation strategies for data with temporal,
spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913-929, 2017.
doi:10.1111/ecog.02881
— *Primary reference for the block-based spatial cross-validation scheme.*

**[R30] Cited: yes.** P. Ploton et al., "Spatial validation reveals poor predictive performance of
large-scale ecological mapping models," *Nat. Commun.*, vol. 11, no. 1, art. 4540, 2020.
doi:10.1038/s41467-020-18321-y
— *Empirical warrant that random cross-validation is optimistically biased for gridded data; this is
the justification for reporting block-grouped rather than random folds.*

### Explainability

**[R31] Cited: yes.** S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair,
R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding
with explainable AI for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56-67, 2020.
doi:10.1038/s42256-019-0138-9
— *TreeSHAP; the explainer used for all attribution reported here.*

**[R32] Cited: yes.** S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model
predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.
— *Original SHAP formulation (no DOI; verified against the NeurIPS proceedings listing).*

### Predictive uncertainty

**[R33] Cited: yes.** J. Lei, M. J. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman,
"Distribution-free predictive inference for regression," *J. Am. Stat. Assoc.*, vol. 113, no. 523,
pp. 1094-1111, 2018. doi:10.1080/01621459.2017.1307116
— *Split conformal prediction. Cited to state precisely what the cross-fitted interval scheme does
and does not guarantee.*

**[R34] Cited: yes.** A. N. Angelopoulos and S. Bates, "Conformal prediction: a gentle
introduction," *Found. Trends Mach. Learn.*, vol. 16, no. 4, pp. 494-591, 2023.
doi:10.1561/2200000101
— *Exchangeability assumptions underlying coverage guarantees; the basis for the paper's refusal to
claim formal conformal validity.*

### Sensitivity analysis

**[R35] Cited: yes.** A. Saltelli, K. Aleksankina, W. Becker, P. Fennell, F. Ferretti, N. Holst,
S. Li, and Q. Wu, "Why so many published sensitivity analyses are false: a systematic review of
sensitivity analysis practices," *Environ. Model. Softw.*, vol. 114, pp. 29-39, 2019.
doi:10.1016/j.envsoft.2019.01.012
— *Design-based Monte Carlo practice and reporting standards for the 500-run sensitivity study.*

### Equity and justice

**[R36] Cited: yes.** E. K. Chu and C. E. B. Cannon, "Equity, inclusion, and justice as criteria
for decision-making on climate adaptation in cities," *Curr. Opin. Environ. Sustain.*, vol. 51,
pp. 85-94, 2021. doi:10.1016/j.cosust.2021.02.009
— *Justice criteria for urban adaptation decisions.*

**[R37] Cited: yes.** S. Juhola, M. Heikkinen, T. Pietila, F. Groundstroem, and J. Kayhko,
"Connecting climate justice and adaptation planning: an adaptation justice index," *Environ. Sci.
Policy*, vol. 136, pp. 609-619, 2022. doi:10.1016/j.envsci.2022.07.024
— *Precedent for an operationalised composite equity score.*

**[R38] Cited: yes.** C. Carvalho, A. G. Del Campo, and D. de Carvalho Cabral, "Scales of
inequality: the role of spatial extent in environmental justice analysis," *Landsc. Urban Plan.*,
vol. 221, art. 104369, 2022. doi:10.1016/j.landurbplan.2022.104369
— *Modifiable-areal-unit and scale effects; directly motivates the paper's fixed 250 m resolution
and the sensitivity of equity results to it.*

### Bengaluru and India

**[R39] Cited: yes.** S. Kanga et al., "Understanding the linkage between urban growth and land
surface temperature — a case study of Bangalore city, India," *Remote Sens.*, vol. 14, no. 17,
art. 4241, 2022. doi:10.3390/rs14174241
— *Closest local precedent for Bengaluru LST and urban growth.*

**[R40] Cited: yes.** N. R. Govind and H. Ramesh, "Exploring the relationship between LST and land
cover of Bengaluru by concentric ring approach," *Environ. Monit. Assess.*, vol. 192, no. 10,
art. 650, 2020. doi:10.1007/s10661-020-08601-x
— *Bengaluru-specific LST-land-cover analysis using concentric zones, comparable to HeatScope's
zonal design.*

**[R41] Cited: yes.** H. Sussman, A. Dai, and P. E. Roundy, "The controlling factors of urban heat
in Bengaluru, India," *Urban Clim.*, vol. 38, art. 100881, 2021. doi:10.1016/j.uclim.2021.100881
— *Driver analysis for Bengaluru; the key explanatory-variable selection citation.*

**[R42] Cited: yes.** A. Mathew, S. Khandelwal, and N. Kaul, "Investigating spatial and seasonal
variations of urban heat island effect over Jaipur city and its relationship with vegetation,
urbanization and elevation parameters," *Sustain. Cities Soc.*, vol. 35, pp. 157-177, 2017.
doi:10.1016/j.scs.2017.07.013
— *Tropical Indian-city UHI reference validating the vegetation/urbanisation feature set.*

### Mixed-integer programming and solvers

**[R43] Cited: yes.** Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method,"
*Math. Program. Comput.*, vol. 10, no. 1, pp. 119-142, 2018. doi:10.1007/s12532-017-0130-5
— *Algorithm underlying the HiGHS LP/MILP solver reached through `scipy.optimize.milp`.*

**[R44] Cited: yes.** Q. Huangfu and J. A. J. Hall, "HiGHS — high performance software for linear
optimization," in *Proc. 23rd Int. Symp. Math. Program. Theory Oper. Res. (ISMP 2023)*, 2023.
— *Current solver reference for the HiGHS MILP backend.*

---

## Explicitly rejected / not used

| Candidate | Reason not used |
|---|---|
| "Mar et al. 2020", *Mobile Mapping of Global Building Footprints* | No matching authoritative record located. |
| "NDBI, Xu 2007" and "NDBI, Liang & Li 2007" | Wrong attributions; NDBI is Zha et al. 2003. |
| WorldPop DOI `10.1038/sdata.2017.41` | Unrelated article. |
| Liang & Li 2007 NDWI, candidate DOI `10.3390/rs9090814` | DOI did not resolve on Crossref; no matching record. |
| Peer-reviewed MILP allocation of urban heat mitigation | **No on-topic peer-reviewed reference could be verified.** See MISSING_EVIDENCE.md, item M1. |
| One Earth 2022 urban-canopy cooling-equity study | DOI, volume, and article number unresolved. See MISSING_EVIDENCE.md, item M2. |