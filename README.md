# Parkinson disease cognitive impairment research calculator

Live page: https://negidamd.github.io/pd-cognition-calculator/

This is a browser calculator for the 14-predictor Cox model described in:

> Negida A. A Multimodal Nomogram Predicting 5-Year Risk of Cognitive Impairment in Parkinson Disease. Manuscript under review, *Movement Disorders Clinical Practice*.

It estimates the 3-, 5- and 8-year risk of incident cognitive impairment (mild cognitive impairment or dementia) from the first normal cognitive assessment.

**For research use only.** The model has been internally validated; external validation has not been performed. Do not use it to make decisions about individual patient care.

## Model

- **Development data:** 1,188 Parkinson's Progression Markers Initiative (PPMI) participants with normal cognition at their first cognitive assessment and at least one later assessment; 457 developed incident cognitive impairment.
- **Predictors (14):** age, sex, education, GBA carrier status, MDS-UPDRS Part II, MoCA, HVLT-R total learning, Letter-Number Sequencing, Symbol Digit Modalities Test, semantic-fluency T score, MDS-UPDRS item 1.1 (cognitive complaints), RBD Screening Questionnaire, SCOPA-AUT and modified Schwab-England ADL.
- **Clinical scores:** take them from the month of the first normal cognitive assessment or the three months before it.
- **Risk formula:** `risk(t) = 1 − S0(t)^exp(Σ βj (xj − meanj))`, where S0 is the Breslow baseline survival at the training means.
- **Internal validation:**
  - optimism-corrected C-index 0.733;
  - nested cross-validated C-index 0.736;
  - held-out time-dependent AUC 0.75, 0.78 and 0.81 at 3, 5 and 8 years.

## Files

- `index.html`: the calculator. It is self-contained; entries stay in your browser and are not sent anywhere.
- `model_parameters.json`: coefficients, training means, baseline survival at 3, 5 and 8 years, and the allowed input ranges.

The repository contains no participant-level data. PPMI data are available to qualified researchers at https://www.ppmi-info.org.
