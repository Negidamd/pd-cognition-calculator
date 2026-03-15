# Analysis Scripts

These scripts reproduce all analyses for:

**Development and Internal Validation of a Multimodal Clinical Nomogram for Predicting Time to Cognitive Impairment in De Novo Parkinson's Disease: A Longitudinal Analysis from the PPMI Cohort**

## Pipeline

| Script | Description |
|--------|-------------|
| `00_build_cohort.py` | Builds the analytical cohort from raw PPMI data: applies inclusion/exclusion criteria, constructs time-to-event outcome, assembles candidate predictors |
| `01_model_and_nomogram.py` | LASSO variable selection, Cox proportional hazards model fitting, bootstrap validation, cross-validation, calibration, nomogram generation |
| `02_extended_models.py` | Machine learning comparators (gradient boosting, random survival forest), biomarker sensitivity analysis, time-dependent AUC |
| `03_table1.py` | Generates Table 1 (baseline characteristics stratified by cognitive outcome) |
| `04_flow_diagram.py` | Generates CONSORT-style patient flow diagram (Figure 1) |
| `06_workflow_and_cover.py` | Generates Workflow_Decisions.docx and cover letter |

## Data

Raw data must be obtained from the [Parkinson's Progression Markers Initiative (PPMI)](https://www.ppmi-info.org/).

## Configuration

Set the following environment variables before running:

```bash
export PPMI_DATA_DIR="/path/to/your/PPMI/data"
export PPMI_NOMOGRAM_DIR="/path/to/your/output/directory"
```

## Requirements

- Python 3.13+
- lifelines (v0.29+)
- scikit-survival (v0.23+)
- scikit-learn (v1.5+)
- pandas, numpy, matplotlib, seaborn
- python-docx (for document generation)
