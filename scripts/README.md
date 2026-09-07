# Analysis Scripts

These scripts reproduce all analyses for:

**A Multimodal Nomogram Predicting 5-Year Risk of Cognitive Impairment in De Novo
Parkinson's Disease** (Movement Disorders Clinical Practice, manuscript 5920300)

## Original submission pipeline

| Script | Description |
|--------|-------------|
| `00_build_cohort.py` | Builds the analytical cohort from raw PPMI data: applies inclusion/exclusion criteria, constructs the time-to-event outcome, assembles candidate predictors |
| `01_model_and_nomogram.py` | LASSO variable selection, Cox proportional hazards model fitting, bootstrap validation, cross-validation, calibration, nomogram generation |
| `02_extended_models.py` | Machine learning comparators (gradient boosting, random survival forest), biomarker sensitivity analysis, time-dependent AUC |
| `03_table1.py` | Table 1 (baseline characteristics stratified by cognitive outcome) |
| `04_flow_diagram.py` | Patient flow diagram (Figure 1) |

## Revision pipeline

Added for the peer-review revision. The primary model is unchanged: `R00` reproduces the
submitted model exactly (apparent C-index 0.733749, identical selected predictor set), and
every revision analysis is additive.

| Script | Description |
|--------|-------------|
| `R00_build_revision_data.py` | Reproduces the primary model and assembles the augmented data (GBA1/LRRK2 variant identity, longitudinal COGSTATE, baseline hallucinations) |
| `R01_corrected_validation.py` | Bootstrap optimism with LASSO selection repeated inside each replicate; complete-case sensitivity analysis; variance inflation factors |
| `R02_reviewer_analyses.py` | GBA1 variant-severity dose-response, LRRK2 G2019S restriction, COGSTATE 1-2 and 2-3 transitions, hallucination landmark analysis, age-stratified discrimination, IPCW time-dependent AUC |
| `R03_nomogram_and_figures.py` | Nomogram rebuilt so total points map to model-derived 3-, 5- and 8-year risk through the baseline survival function (Figure 5); IPCW AUC figure (Figure 3) |
| `R04_supp_tables_figures.py` | Supplementary tables and figures |
| `R12_nongenetic_nomogram.py` | Companion 16-predictor model and nomogram without GBA/LRRK2, for settings without genotyping |
| `R14_update_calculator.py` | Derives the web calculator constants; corrects the baseline-survival centring and adds the non-genetic model |
| `R16_audit_figures.py` | Bootstrap pointwise confidence band for the time-dependent AUC, numbers-at-risk annotation for the Kaplan-Meier figure, GBA1 dose-response figure on a log hazard-ratio axis |

## Data

Raw data must be obtained from the
[Parkinson's Progression Markers Initiative (PPMI)](https://www.ppmi-info.org/).

## Configuration

```bash
export PPMI_DATA_DIR="/path/to/your/PPMI/data"        # contains PPMI_Master_Merged.csv
export PPMI_NOMOGRAM_DIR="/path/to/your/project/root" # output root; defaults to the repo root
```

## Requirements

See `requirements.txt`. The reported results were verified to reproduce under Python
3.12.14 with lifelines 0.30.3, scikit-survival 0.28.0 and scikit-learn 1.9.0. A random
seed of 42 is used for all stochastic procedures.

## Web calculator

`index.html` at the repository root implements the fitted model, including the non-genetic
companion model for patients whose genotype is unknown. Risks are computed relative to the
median reference patient, S(t | medians) = S0(t)^exp(lp_median - lp_mean), and were
cross-checked against the Python model to six decimal places.
