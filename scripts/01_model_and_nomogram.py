#!/usr/bin/env python3
"""
01_model_and_nomogram.py — Cox regression with LASSO, nomogram, and validation
Predicts time to cognitive impairment (MCI/Dementia) in PD
"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── Config ──────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# NOTE: Update BASE to your local PPMI data directory
BASE = os.environ.get("PPMI_NOMOGRAM_DIR", ".")

# ─── Load analytical dataset ────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data", "analytical_dataset.csv"))
# Drop rows with missing time_years (visits not in standard mapping)
df = df.dropna(subset=['time_years']).reset_index(drop=True)
EXPECTED_N = len(df)
print(f"Loaded analytical dataset: N={len(df)}, Events={df['event'].sum()}")

# ─── Load cohort info ───────────────────────────────────────────────────────
with open(os.path.join(BASE, "logs", "cohort_info.json")) as f:
    cohort_info = json.load(f)

primary_candidates = cohort_info['primary_candidates']
print(f"Primary candidate predictors: {len(primary_candidates)}")

# ─── Handle SAA_Status_Combined (categorical → binary) ──────────────────────
if 'SAA_Status_Combined' in df.columns:
    df['SAA_positive'] = (df['SAA_Status_Combined'].astype(str).str.lower().isin(
        ['positive', '1', '1.0', 'pos'])).astype(float)
    df.loc[df['SAA_Status_Combined'].isna(), 'SAA_positive'] = np.nan
    # Replace in candidate list
    if 'SAA_Status_Combined' in primary_candidates:
        primary_candidates = [v if v != 'SAA_Status_Combined' else 'SAA_positive' for v in primary_candidates]

# ─── Prepare modeling data ───────────────────────────────────────────────────
# Remove variables with too much collinearity or redundancy
# HVLTRT1-3 are trials 1-3 of HVLT → use total (sum) instead
df['HVLT_total_learning'] = df[['HVLTRT1', 'HVLTRT2', 'HVLTRT3']].sum(axis=1, min_count=2)
# Replace individual trials with total
primary_candidates = [v for v in primary_candidates if v not in ['HVLTRT1', 'HVLTRT2', 'HVLTRT3']]
primary_candidates.append('HVLT_total_learning')

# NP3TOT_BEST and tremor_score/pigd_score are correlated; keep composites
# NP1RTOT and NP2PTOT are part sums; keep totals
# DVS_LNS is a scaled score of LNS_TOTRAW; keep raw
primary_candidates = [v for v in primary_candidates if v != 'DVS_LNS']

# Final predictor list
pred_vars = [v for v in primary_candidates if v in df.columns]
print(f"\nFinal predictor variables ({len(pred_vars)}):")
for v in pred_vars:
    print(f"  {v}")

# ─── Imputation ─────────────────────────────────────────────────────────────
X = df[pred_vars].copy()
y_time = df['time_years'].values
y_event = df['event'].values.astype(bool)

# Median imputation for missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=pred_vars)

print(f"\nAfter imputation: X shape = {X_imputed.shape}")
print(f"  Missing values remaining: {X_imputed.isna().sum().sum()}")

# ─── Step 1: LASSO-Cox for variable selection ───────────────────────────────
print("\n" + "="*70)
print("STEP 1: LASSO-Cox Variable Selection")
print("="*70)

# Standardize for LASSO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_scaled_df = pd.DataFrame(X_scaled, columns=pred_vars)

# Structured array for sksurv
y_surv = np.array([(e, t) for e, t in zip(y_event, y_time)],
                  dtype=[('event', bool), ('time', float)])

# Find optimal alpha via cross-validation
alphas = np.logspace(-3, 0, 50)

# Cross-validated LASSO
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
cv_scores = []

for alpha in alphas:
    fold_scores = []
    for train_idx, test_idx in cv.split(X_scaled, y_event.astype(int)):
        try:
            model = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=0.9,
                                            max_iter=10000, tol=1e-7)
            model.fit(X_scaled[train_idx], y_surv[train_idx])
            pred = model.predict(X_scaled[test_idx])
            ci = concordance_index_censored(y_event[test_idx], y_time[test_idx], pred)[0]
            fold_scores.append(ci)
        except Exception:
            fold_scores.append(0.5)
    cv_scores.append(np.mean(fold_scores))

best_alpha_idx = np.argmax(cv_scores)
best_alpha = alphas[best_alpha_idx]
best_cv_ci = cv_scores[best_alpha_idx]
print(f"Best alpha: {best_alpha:.4f}, CV C-index: {best_cv_ci:.4f}")

# Fit final LASSO model with best alpha
lasso_model = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=0.9,
                                      max_iter=10000, tol=1e-7)
lasso_model.fit(X_scaled, y_surv)

# Get selected variables
coefs = pd.Series(lasso_model.coef_.ravel(), index=pred_vars)
selected_vars = coefs[coefs.abs() > 0].sort_values(key=abs, ascending=False)
print(f"\nSelected variables ({len(selected_vars)}):")
for v, c in selected_vars.items():
    print(f"  {v:30s}: {c:+.4f}")

# ─── LASSO regularization path plot ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: CV C-index vs alpha
axes[0].plot(alphas, cv_scores, 'b-', linewidth=2)
axes[0].axvline(best_alpha, color='red', linestyle='--', linewidth=1.5, label=f'Best α={best_alpha:.4f}')
axes[0].set_xscale('log')
axes[0].set_xlabel('Regularization parameter (α)', fontsize=12, fontfamily='Arial')
axes[0].set_ylabel('Cross-validated C-index', fontsize=12, fontfamily='Arial')
axes[0].legend(fontsize=10, frameon=False)
axes[0].tick_params(labelsize=10)

# Panel B: Coefficient path
all_coefs = []
for alpha in alphas:
    m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=0.9, max_iter=10000, tol=1e-7)
    m.fit(X_scaled, y_surv)
    all_coefs.append(m.coef_.ravel())
all_coefs = np.array(all_coefs)

for i, var in enumerate(pred_vars):
    if var in selected_vars.index:
        axes[1].plot(alphas, all_coefs[:, i], linewidth=1.5, label=var)
    else:
        axes[1].plot(alphas, all_coefs[:, i], linewidth=0.5, color='lightgrey', alpha=0.5)
axes[1].set_xscale('log')
axes[1].axvline(best_alpha, color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Regularization parameter (α)', fontsize=12, fontfamily='Arial')
axes[1].set_ylabel('Coefficient', fontsize=12, fontfamily='Arial')
axes[1].tick_params(labelsize=10)

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "lasso_selection.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "lasso_selection.pdf"), bbox_inches='tight')
plt.close()
print("\nSaved: figures/lasso_selection.png/pdf")

# ─── Step 2: Final Cox PH model with selected variables ─────────────────────
print("\n" + "="*70)
print("STEP 2: Final Cox Proportional Hazards Model")
print("="*70)

final_vars = list(selected_vars.index)

# Build modeling dataframe with unscaled values for interpretability
model_df = X_imputed[final_vars].copy()
model_df['time_years'] = y_time
model_df['event'] = y_event.astype(int)

# Fit Cox PH
cph = CoxPHFitter(penalizer=0.01)
cph.fit(model_df, duration_col='time_years', event_col='event')

print("\nCox PH Model Summary:")
print(cph.summary.to_string())

# Save model summary
cph.summary.to_csv(os.path.join(BASE, "tables", "cox_model_summary.csv"))

# ─── Step 3: Model validation ───────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 3: Model Validation")
print("="*70)

# 3a. Apparent C-index
apparent_ci = cph.concordance_index_
print(f"Apparent C-index: {apparent_ci:.4f}")

# 3b. 10-fold cross-validated C-index
cv_cindex = []
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for train_idx, test_idx in cv.split(model_df[final_vars], model_df['event']):
    train_df = model_df.iloc[train_idx]
    test_df = model_df.iloc[test_idx]
    cv_cph = CoxPHFitter(penalizer=0.01)
    cv_cph.fit(train_df, duration_col='time_years', event_col='event')
    pred = cv_cph.predict_partial_hazard(test_df)
    ci = concordance_index(test_df['time_years'], -pred, test_df['event'])
    cv_cindex.append(ci)

cv_mean = np.mean(cv_cindex)
cv_std = np.std(cv_cindex)
print(f"10-fold CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")

# 3c. Bootstrap validation (200 iterations)
n_bootstrap = 200
boot_ci_apparent = []
boot_ci_test = []
boot_optimism = []

for b in range(n_bootstrap):
    boot_idx = np.random.choice(len(model_df), size=len(model_df), replace=True)
    oob_idx = np.array([i for i in range(len(model_df)) if i not in boot_idx])

    if len(oob_idx) < 20:
        continue

    boot_df = model_df.iloc[boot_idx]
    oob_df = model_df.iloc[oob_idx]

    try:
        boot_cph = CoxPHFitter(penalizer=0.01)
        boot_cph.fit(boot_df, duration_col='time_years', event_col='event')

        # Apparent on bootstrap
        pred_boot = boot_cph.predict_partial_hazard(boot_df)
        ci_boot = concordance_index(boot_df['time_years'], -pred_boot, boot_df['event'])

        # Test on OOB
        pred_oob = boot_cph.predict_partial_hazard(oob_df)
        ci_oob = concordance_index(oob_df['time_years'], -pred_oob, oob_df['event'])

        boot_ci_apparent.append(ci_boot)
        boot_ci_test.append(ci_oob)
        boot_optimism.append(ci_boot - ci_oob)
    except Exception:
        continue

optimism = np.mean(boot_optimism)
corrected_ci = apparent_ci - optimism
print(f"Bootstrap optimism: {optimism:.4f}")
print(f"Optimism-corrected C-index: {corrected_ci:.4f}")
print(f"Bootstrap test C-index: {np.mean(boot_ci_test):.4f} ± {np.std(boot_ci_test):.4f}")

# ─── Step 4: Calibration ────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 4: Calibration Analysis")
print("="*70)

# Time-specific calibration at 3, 5, 8 years
timepoints = [3, 5, 8]
risk_scores = cph.predict_partial_hazard(model_df).values.ravel()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
calibration_results = {}

for idx, t in enumerate(timepoints):
    ax = axes[idx]
    # Predicted survival at time t
    surv_func = cph.predict_survival_function(model_df, times=[t])
    pred_risk = 1 - surv_func.values.ravel()

    # Bin into quintiles
    quintiles = pd.qcut(pred_risk, 5, duplicates='drop')
    cal_df = pd.DataFrame({
        'pred_risk': pred_risk,
        'event': model_df['event'].values,
        'time': model_df['time_years'].values,
        'quintile': quintiles
    })

    observed = []
    predicted = []
    for q in sorted(cal_df['quintile'].unique()):
        grp = cal_df[cal_df['quintile'] == q]
        predicted.append(grp['pred_risk'].mean())

        # KM estimate of observed risk
        kmf = KaplanMeierFitter()
        kmf.fit(grp['time'], grp['event'])
        if t <= kmf.survival_function_.index.max():
            obs_surv = kmf.predict(t)
            observed.append(1 - obs_surv)
        else:
            observed.append(np.nan)

    observed = np.array(observed)
    predicted = np.array(predicted)
    mask = ~np.isnan(observed)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.scatter(predicted[mask], observed[mask], s=80, color='#2171b5', edgecolor='white', zorder=5)
    ax.plot(predicted[mask], observed[mask], '-', color='#2171b5', linewidth=1.5)
    ax.set_xlabel(f'Predicted {t}-year risk', fontsize=11, fontfamily='Arial')
    ax.set_ylabel(f'Observed {t}-year risk', fontsize=11, fontfamily='Arial')
    ax.set_xlim(-0.02, max(0.8, predicted.max() + 0.05))
    ax.set_ylim(-0.02, max(0.8, max(observed[mask]) + 0.05) if mask.any() else 0.8)
    ax.tick_params(labelsize=10)
    ax.set_aspect('equal', adjustable='box')

    calibration_results[f'{t}yr'] = {
        'predicted': predicted[mask].tolist(),
        'observed': observed[mask].tolist()
    }

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "calibration_plots.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "calibration_plots.pdf"), bbox_inches='tight')
plt.close()
print("Saved: figures/calibration_plots.png/pdf")

# ─── Step 5: Nomogram ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 5: Nomogram Construction")
print("="*70)

# Get coefficients from Cox model
cox_summary = cph.summary.copy()
cox_coefs = cox_summary['coef']

# For nomogram: convert HR/coefficients to point scores
# Points = coefficient × (value - reference) / scaling_factor
# Scale so total points span 0-100 for the most important variable

# Determine ranges for each variable
var_ranges = {}
for v in final_vars:
    var_ranges[v] = {
        'min': model_df[v].quantile(0.01),
        'max': model_df[v].quantile(0.99),
        'median': model_df[v].median(),
        'coef': cox_coefs[v],
        'hr': np.exp(cox_coefs[v]),
        'pval': cox_summary.loc[v, 'p']
    }

# Point contribution range for each variable
max_point_range = 0
for v in final_vars:
    r = var_ranges[v]
    point_range = abs(r['coef']) * (r['max'] - r['min'])
    var_ranges[v]['point_range'] = point_range
    max_point_range = max(max_point_range, point_range)

# Scale factor: map largest range to 100 points
scale = 100 / max_point_range if max_point_range > 0 else 1

# Calculate points for each variable
for v in final_vars:
    r = var_ranges[v]
    r['points_per_unit'] = r['coef'] * scale
    r['points_at_min'] = r['coef'] * r['min'] * scale
    r['points_at_max'] = r['coef'] * r['max'] * scale

# Sort variables by importance (absolute point range)
sorted_vars = sorted(final_vars, key=lambda v: var_ranges[v]['point_range'], reverse=True)

# ─── Draw nomogram ──────────────────────────────────────────────────────────
n_vars = len(sorted_vars)
fig_height = max(12, (n_vars + 3) * 0.85)
fig, ax = plt.subplots(figsize=(14, fig_height))

y_positions = np.arange(n_vars + 3, 0, -1)  # +3 for Points, Total Points, Risk lines
bar_height = 0.35

# Color palette
colors = sns.color_palette("muted", n_vars)

# Clean variable names for display
name_map = {
    'AGE_AT_VISIT': 'Age (years)',
    'SEX': 'Sex (1=Male)',
    'EDUCYRS': 'Education (years)',
    'APOE4_carrier': 'APOE ε4 carrier',
    'GBA_carrier': 'GBA mutation carrier',
    'LRRK2_carrier': 'LRRK2 mutation carrier',
    'NP3TOT_BEST': 'MDS-UPDRS Part III',
    'NHY_BEST': 'Hoehn & Yahr stage',
    'NP2PTOT': 'MDS-UPDRS Part II',
    'NP1RTOT': 'MDS-UPDRS Part I',
    'tremor_score': 'Tremor score',
    'pigd_score': 'PIGD score',
    'MCATOT': 'MoCA total',
    'HVLTRDLY': 'HVLT-R delayed recall',
    'HVLT_total_learning': 'HVLT-R total learning',
    'JLO_TOTRAW': 'JLO total raw',
    'LNS_TOTRAW': 'Letter-Number Sequencing',
    'SDMTOTAL': 'Symbol Digit Modalities',
    'DVT_SFTANIM': 'Semantic fluency (animals)',
    'NP1COG': 'Cognitive complaints (1.1)',
    'NP1HALL': 'Hallucinations (1.2)',
    'NP1DPRS': 'Depression (1.3)',
    'NP1ANXS': 'Anxiety (1.4)',
    'NP1APAT': 'Apathy (1.5)',
    'RBDSQ_TOTAL': 'RBD Screening Questionnaire',
    'SCAU_TOTAL': 'SCOPA-AUT total',
    'MSEADLG': 'Modified Schwab-England ADL',
    'SAA_positive': 'α-synuclein SAA positive',
    'ortho_hypotension': 'Orthostatic hypotension',
    'PIGD_dominant': 'PIGD-dominant subtype',
    'LEDD_TOTAL_CURRENT': 'LEDD (mg/day)',
}

# Points scale at top
ax.plot([0, 100], [y_positions[0]] * 2, 'k-', linewidth=2)
for p in range(0, 101, 10):
    ax.plot([p, p], [y_positions[0] - 0.15, y_positions[0] + 0.15], 'k-', linewidth=1.5)
    ax.text(p, y_positions[0] + 0.3, str(p), ha='center', va='bottom', fontsize=9, fontfamily='Arial')
ax.text(-5, y_positions[0], 'Points', ha='right', va='center', fontsize=11,
        fontweight='bold', fontfamily='Arial')

# Variable lines
for i, var in enumerate(sorted_vars):
    y = y_positions[i + 1]
    r = var_ranges[var]
    display_name = name_map.get(var, var)

    # Calculate point positions for this variable
    val_min, val_max = r['min'], r['max']

    # For binary variables
    if var in ['SEX', 'APOE4_carrier', 'GBA_carrier', 'LRRK2_carrier',
               'SAA_positive', 'ortho_hypotension', 'PIGD_dominant']:
        vals = [0, 1]
        labels = ['0', '1']
        if var == 'SEX':
            labels = ['F', 'M']
        elif var in ['APOE4_carrier', 'GBA_carrier', 'LRRK2_carrier', 'SAA_positive']:
            labels = ['No', 'Yes']
        elif var == 'ortho_hypotension':
            labels = ['No', 'Yes']
        elif var == 'PIGD_dominant':
            labels = ['No', 'Yes']
    else:
        # Continuous: pick ~5-7 nice tick values
        n_ticks = 6
        vals = np.linspace(val_min, val_max, n_ticks)
        # Round to sensible values
        if val_max - val_min > 10:
            vals = np.round(vals, 0)
        elif val_max - val_min > 1:
            vals = np.round(vals, 1)
        else:
            vals = np.round(vals, 2)
        vals = np.unique(vals)
        labels = [f'{v:.0f}' if abs(v) >= 1 else f'{v:.1f}' for v in vals]

    # Map values to points (0-100 scale)
    points = [r['coef'] * v * scale for v in vals]

    # Shift so minimum maps to start of line
    min_pt = min(points)
    max_pt = max(points)
    pt_range = max_pt - min_pt if max_pt != min_pt else 1

    # Map to 0-100 based on the global reference
    # Use reference point (minimum value gives 0 additional points for this var)
    ref_val = val_min if r['coef'] > 0 else val_max
    points_shifted = [(r['coef'] * (v - ref_val) * scale) for v in vals]

    # Normalize: the variable's max contribution maps to its point_range * scale
    norm_max = r['point_range'] * scale
    if norm_max > 0:
        points_norm = [p / (r['point_range'] * scale) * (r['point_range'] / max_point_range * 100) for p in points_shifted]
    else:
        points_norm = [0] * len(vals)

    # Ensure all points ≥ 0
    shift = min(points_norm)
    points_norm = [p - shift for p in points_norm]

    # Draw line
    line_min = min(points_norm)
    line_max = max(points_norm)
    ax.plot([line_min, line_max], [y, y], '-', color=colors[i % len(colors)], linewidth=2.5)

    # Tick marks and labels
    for p, lbl in zip(points_norm, labels):
        ax.plot([p, p], [y - 0.15, y + 0.15], '-', color=colors[i % len(colors)], linewidth=1.5)
        ax.text(p, y - 0.3, str(lbl), ha='center', va='top', fontsize=8, fontfamily='Arial')

    # Variable name
    ax.text(-5, y, display_name, ha='right', va='center', fontsize=10, fontfamily='Arial')

# Total Points line
y_total = y_positions[-2]
ax.plot([0, 300], [y_total, y_total], 'k-', linewidth=2)
for p in range(0, 301, 50):
    ax.plot([p / 3, p / 3], [y_total - 0.15, y_total + 0.15], 'k-', linewidth=1.5)
    ax.text(p / 3, y_total - 0.3, str(p), ha='center', va='top', fontsize=9, fontfamily='Arial')
ax.text(-5, y_total, 'Total Points', ha='right', va='center', fontsize=11,
        fontweight='bold', fontfamily='Arial')

# Risk lines (5-year and 8-year)
y_risk = y_positions[-1]
# Approximate: map total points to 5-year risk
risk_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ax.plot([0, 100], [y_risk, y_risk], 'k-', linewidth=2)
for rl in risk_levels:
    pos = rl * 100  # simplified mapping
    ax.plot([pos, pos], [y_risk - 0.15, y_risk + 0.15], 'k-', linewidth=1.5)
    ax.text(pos, y_risk - 0.3, f'{rl:.0%}', ha='center', va='top', fontsize=9, fontfamily='Arial')
ax.text(-5, y_risk, '5-year risk', ha='right', va='center', fontsize=11,
        fontweight='bold', fontfamily='Arial')

ax.set_xlim(-8, 105)
ax.set_ylim(y_positions[-1] - 1, y_positions[0] + 1)
ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "nomogram.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "nomogram.pdf"), bbox_inches='tight')
plt.close()
print("Saved: figures/nomogram.png/pdf")

# ─── Step 6: Forest plot of HRs ─────────────────────────────────────────────
print("\n" + "="*70)
print("STEP 6: Forest Plot")
print("="*70)

# Prepare forest plot data
forest_data = []
for v in sorted_vars:
    r = var_ranges[v]
    s = cox_summary.loc[v]
    forest_data.append({
        'Variable': name_map.get(v, v),
        'HR': s['exp(coef)'],
        'HR_lower': s['exp(coef) lower 95%'],
        'HR_upper': s['exp(coef) upper 95%'],
        'p': s['p'],
        'coef': s['coef']
    })

forest_df = pd.DataFrame(forest_data)

# Sort by HR
forest_df = forest_df.sort_values('HR', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(forest_df) * 0.45)))

y_pos = range(len(forest_df))
colors_forest = ['#c6dbef' if p > 0.05 else '#2171b5' for p in forest_df['p']]

for i, (_, row) in enumerate(forest_df.iterrows()):
    color = '#2171b5' if row['p'] < 0.05 else '#9ecae1'
    ax.plot([row['HR_lower'], row['HR_upper']], [i, i], '-', color=color, linewidth=2)
    ax.plot(row['HR'], i, 'o', color=color, markersize=8, markeredgecolor='white', zorder=5)

ax.axvline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(forest_df['Variable'], fontsize=10, fontfamily='Arial')
ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12, fontfamily='Arial')
ax.tick_params(labelsize=10)

# Add HR (95% CI) text on the right
for i, (_, row) in enumerate(forest_df.iterrows()):
    pstr = f'p<0.001' if row['p'] < 0.001 else f'p={row["p"]:.3f}'
    ax.text(ax.get_xlim()[1] * 1.02, i,
            f'{row["HR"]:.2f} ({row["HR_lower"]:.2f}-{row["HR_upper"]:.2f}) {pstr}',
            va='center', fontsize=8, fontfamily='Arial')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "forest_plot.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "forest_plot.pdf"), bbox_inches='tight')
plt.close()
print("Saved: figures/forest_plot.png/pdf")

# ─── Step 7: KM curves by risk groups ───────────────────────────────────────
print("\n" + "="*70)
print("STEP 7: Risk Stratification — Kaplan-Meier Curves")
print("="*70)

# Calculate risk scores
risk_scores = cph.predict_partial_hazard(model_df).values.ravel()

# Tertiles
tertile_labels = pd.qcut(risk_scores, 3, labels=['Low risk', 'Intermediate risk', 'High risk'])

fig, ax = plt.subplots(figsize=(8, 6))
palette = {'Low risk': '#2ca02c', 'Intermediate risk': '#ff7f0e', 'High risk': '#d62728'}

for grp in ['Low risk', 'Intermediate risk', 'High risk']:
    mask = tertile_labels == grp
    kmf = KaplanMeierFitter()
    kmf.fit(model_df.loc[mask, 'time_years'], model_df.loc[mask, 'event'], label=grp)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=palette[grp], linewidth=2)

# Number at risk table
ax.set_xlabel('Time from baseline (years)', fontsize=12, fontfamily='Arial')
ax.set_ylabel('Cognitive impairment-free survival', fontsize=12, fontfamily='Arial')
ax.legend(fontsize=10, frameon=False, loc='lower left')
ax.tick_params(labelsize=10)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add log-rank p-value
from lifelines.statistics import logrank_test
low_mask = tertile_labels == 'Low risk'
high_mask = tertile_labels == 'High risk'
lr = logrank_test(
    model_df.loc[low_mask, 'time_years'], model_df.loc[high_mask, 'time_years'],
    model_df.loc[low_mask, 'event'], model_df.loc[high_mask, 'event']
)
pval = lr.p_value
pstr = f'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
ax.text(0.98, 0.98, f'Log-rank {pstr}', transform=ax.transAxes,
        ha='right', va='top', fontsize=10, fontfamily='Arial')

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "km_risk_groups.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "km_risk_groups.pdf"), bbox_inches='tight')
plt.close()
print("Saved: figures/km_risk_groups.png/pdf")

# ─── Step 8: Save all results ───────────────────────────────────────────────
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

results = {
    'model': 'Cox Proportional Hazards with LASSO variable selection',
    'n_total': int(len(model_df)),
    'n_events': int(model_df['event'].sum()),
    'n_censored': int((model_df['event'] == 0).sum()),
    'n_predictors_initial': len(pred_vars),
    'n_predictors_selected': len(final_vars),
    'selected_predictors': final_vars,
    'apparent_c_index': float(apparent_ci),
    'cv_c_index_mean': float(cv_mean),
    'cv_c_index_std': float(cv_std),
    'bootstrap_optimism': float(optimism),
    'optimism_corrected_c_index': float(corrected_ci),
    'bootstrap_test_c_index_mean': float(np.mean(boot_ci_test)),
    'bootstrap_test_c_index_std': float(np.std(boot_ci_test)),
    'best_lasso_alpha': float(best_alpha),
    'lasso_cv_c_index': float(best_cv_ci),
    'logrank_p_low_vs_high': float(pval),
    'hazard_ratios': {v: {
        'HR': float(cox_summary.loc[v, 'exp(coef)']),
        'lower95': float(cox_summary.loc[v, 'exp(coef) lower 95%']),
        'upper95': float(cox_summary.loc[v, 'exp(coef) upper 95%']),
        'p': float(cox_summary.loc[v, 'p'])
    } for v in final_vars}
}

with open(os.path.join(BASE, "logs", "model_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nFinal model: {len(final_vars)} predictors")
print(f"Apparent C-index: {apparent_ci:.4f}")
print(f"10-fold CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Optimism-corrected C-index: {corrected_ci:.4f}")
print(f"Bootstrap test C-index: {np.mean(boot_ci_test):.4f} ± {np.std(boot_ci_test):.4f}")
print(f"\nSignificant predictors (p < 0.05):")
for v in final_vars:
    hr = cox_summary.loc[v, 'exp(coef)']
    p = cox_summary.loc[v, 'p']
    if p < 0.05:
        print(f"  {name_map.get(v,v):35s}: HR={hr:.3f}, p={p:.4f}")

print(f"\nLog-rank test (low vs high risk): {pstr}")
print("\n✓ Model building and validation complete.")
