#!/usr/bin/env python3
"""
02_extended_models.py — Extended analyses:
  A. Gradient boosting survival for higher C-index
  B. CSF/plasma biomarker sensitivity analysis
  C. Time-dependent AUC
  D. Decision curve analysis
"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)
import random; random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# NOTE: Update BASE to your local PPMI data directory
BASE = os.environ.get("PPMI_NOMOGRAM_DIR", ".")

# ─── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, "data", "analytical_dataset.csv"))
df = df.dropna(subset=['time_years']).reset_index(drop=True)
print(f"Loaded: N={len(df)}, Events={df['event'].sum()}")

with open(os.path.join(BASE, "logs", "cohort_info.json")) as f:
    cohort_info = json.load(f)

with open(os.path.join(BASE, "logs", "model_results.json")) as f:
    cox_results = json.load(f)

cox_vars = cox_results['selected_predictors']
print(f"Cox model variables: {len(cox_vars)}")

# ─── Handle SAA and derived vars ─────────────────────────────────────────────
df['SAA_positive'] = (df['SAA_Status_Combined'].astype(str).str.lower().isin(
    ['positive', '1', '1.0', 'pos'])).astype(float)
df.loc[df['SAA_Status_Combined'].isna(), 'SAA_positive'] = np.nan

df['HVLT_total_learning'] = df[['HVLTRT1', 'HVLTRT2', 'HVLTRT3']].sum(axis=1, min_count=2)

df['APOE4_carrier'] = df['APOE'].astype(str).str.contains('E4', na=False).astype(float)
df.loc[df['APOE'].isna(), 'APOE4_carrier'] = np.nan

df['GBA_carrier'] = (~df['GBA'].isin(['0', 0]) & df['GBA'].notna()).astype(float)
df.loc[df['GBA'].isna(), 'GBA_carrier'] = np.nan

df['LRRK2_carrier'] = (~df['LRRK2'].isin(['0', 0]) & df['LRRK2'].notna()).astype(float)
df.loc[df['LRRK2'].isna(), 'LRRK2_carrier'] = np.nan

if 'SYSSUP' in df.columns and 'SYSSTND' in df.columns:
    df['ortho_hypotension'] = ((df['SYSSUP'] - df['SYSSTND'] >= 20) |
                                (df['DIASUP'] - df['DIASTND'] >= 10)).astype(float)

if 'tremor_score' in df.columns and 'pigd_score' in df.columns:
    df['PIGD_dominant'] = (df['pigd_score'] > df['tremor_score']).astype(float)

if 'CSF_ABeta42' in df.columns and 'CSF_pTau' in df.columns:
    df['CSF_Abeta_pTau_ratio'] = df['CSF_ABeta42'] / df['CSF_pTau'].replace(0, np.nan)

# ─── Prepare X and y ────────────────────────────────────────────────────────
y_time = df['time_years'].values
y_event = df['event'].values.astype(bool)
y_surv = np.array([(e, t) for e, t in zip(y_event, y_time)],
                  dtype=[('event', bool), ('time', float)])

# ─── A. Gradient Boosting Survival Analysis ──────────────────────────────────
print("\n" + "="*70)
print("A. GRADIENT BOOSTING SURVIVAL ANALYSIS")
print("="*70)

# Use same variables as Cox model
X_cox = df[cox_vars].copy()
imputer = SimpleImputer(strategy='median')
X_cox_imp = pd.DataFrame(imputer.fit_transform(X_cox), columns=cox_vars)

# Cross-validate GBS
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
gbs_cv_scores = []
rsf_cv_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X_cox_imp, y_event.astype(int))):
    X_train = X_cox_imp.iloc[train_idx].values
    X_test = X_cox_imp.iloc[test_idx].values
    y_train = y_surv[train_idx]
    y_test = y_surv[test_idx]

    # GBS
    gbs = GradientBoostingSurvivalAnalysis(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        min_samples_split=20, min_samples_leaf=10,
        subsample=0.8, random_state=SEED
    )
    gbs.fit(X_train, y_train)
    pred_gbs = gbs.predict(X_test)
    ci_gbs = concordance_index_censored(y_event[test_idx], y_time[test_idx], pred_gbs)[0]
    gbs_cv_scores.append(ci_gbs)

    # RSF
    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=5, min_samples_split=20,
        min_samples_leaf=10, random_state=SEED, n_jobs=-1
    )
    rsf.fit(X_train, y_train)
    pred_rsf = rsf.predict(X_test)
    ci_rsf = concordance_index_censored(y_event[test_idx], y_time[test_idx], pred_rsf)[0]
    rsf_cv_scores.append(ci_rsf)

    if (fold + 1) % 5 == 0:
        print(f"  Fold {fold+1}/10 done")

gbs_mean = np.mean(gbs_cv_scores)
gbs_std = np.std(gbs_cv_scores)
rsf_mean = np.mean(rsf_cv_scores)
rsf_std = np.std(rsf_cv_scores)

print(f"\nGradient Boosting CV C-index: {gbs_mean:.4f} ± {gbs_std:.4f}")
print(f"Random Survival Forest CV C-index: {rsf_mean:.4f} ± {rsf_std:.4f}")
print(f"Cox PH CV C-index: {cox_results['cv_c_index_mean']:.4f} ± {cox_results['cv_c_index_std']:.4f}")

# Fit final GBS model
gbs_final = GradientBoostingSurvivalAnalysis(
    n_estimators=200, learning_rate=0.05, max_depth=3,
    min_samples_split=20, min_samples_leaf=10,
    subsample=0.8, random_state=SEED
)
gbs_final.fit(X_cox_imp.values, y_surv)

# Feature importance
feat_imp = pd.Series(gbs_final.feature_importances_, index=cox_vars).sort_values(ascending=False)
print("\nGBS Feature Importance (top 10):")
for v, imp in feat_imp.head(10).items():
    print(f"  {v:35s}: {imp:.4f}")

# ─── B. CSF/Plasma Biomarker Sensitivity Analysis ───────────────────────────
print("\n" + "="*70)
print("B. BIOMARKER-ENHANCED MODEL (sensitivity analysis)")
print("="*70)

biomarker_vars = ['CSF_ABeta42', 'CSF_pTau', 'CSF_tTau', 'CSF_aSyn',
                  'Plasma_GFAP', 'Plasma_NfL', 'CSF_Abeta_pTau_ratio']

# Subset with at least some biomarker data
bio_available = df[biomarker_vars].notna().any(axis=1)
df_bio = df[bio_available].reset_index(drop=True)
print(f"Patients with any biomarker data: {len(df_bio)}")

# Extended model: Cox vars + biomarkers
extended_vars = cox_vars + [v for v in biomarker_vars if v in df_bio.columns]
X_ext = df_bio[extended_vars].copy()
y_time_bio = df_bio['time_years'].values
y_event_bio = df_bio['event'].values.astype(bool)
y_surv_bio = np.array([(e, t) for e, t in zip(y_event_bio, y_time_bio)],
                       dtype=[('event', bool), ('time', float)])

# Impute
imp_bio = SimpleImputer(strategy='median')
X_ext_imp = pd.DataFrame(imp_bio.fit_transform(X_ext), columns=extended_vars)

# Cross-validate
bio_cv_scores = []
bio_cox_scores = []
cv_bio = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for train_idx, test_idx in cv_bio.split(X_ext_imp, y_event_bio.astype(int)):
    # Extended GBS
    gbs_bio = GradientBoostingSurvivalAnalysis(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        min_samples_split=15, min_samples_leaf=8,
        subsample=0.8, random_state=SEED
    )
    gbs_bio.fit(X_ext_imp.iloc[train_idx].values, y_surv_bio[train_idx])
    pred = gbs_bio.predict(X_ext_imp.iloc[test_idx].values)
    ci = concordance_index_censored(y_event_bio[test_idx], y_time_bio[test_idx], pred)[0]
    bio_cv_scores.append(ci)

    # Base model (Cox vars only) on same subset
    gbs_base = GradientBoostingSurvivalAnalysis(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        min_samples_split=15, min_samples_leaf=8,
        subsample=0.8, random_state=SEED
    )
    gbs_base.fit(X_ext_imp[cox_vars].iloc[train_idx].values, y_surv_bio[train_idx])
    pred_base = gbs_base.predict(X_ext_imp[cox_vars].iloc[test_idx].values)
    ci_base = concordance_index_censored(y_event_bio[test_idx], y_time_bio[test_idx], pred_base)[0]
    bio_cox_scores.append(ci_base)

print(f"Biomarker subset N={len(df_bio)}, Events={y_event_bio.sum()}")
print(f"Base model (clinical only) CV C-index: {np.mean(bio_cox_scores):.4f} ± {np.std(bio_cox_scores):.4f}")
print(f"Extended model (+biomarkers) CV C-index: {np.mean(bio_cv_scores):.4f} ± {np.std(bio_cv_scores):.4f}")
print(f"Improvement: {np.mean(bio_cv_scores) - np.mean(bio_cox_scores):+.4f}")

# ─── C. Time-dependent AUC ──────────────────────────────────────────────────
print("\n" + "="*70)
print("C. TIME-DEPENDENT AUC")
print("="*70)

# Use GBS predictions for time-dependent AUC
risk_scores_gbs = gbs_final.predict(X_cox_imp.values)

# Time points for evaluation
times = np.arange(1, 13, 1)
valid_times = times[times < y_time.max()]

try:
    auc_values, mean_auc = cumulative_dynamic_auc(y_surv, y_surv, risk_scores_gbs, valid_times)
    print(f"Mean time-dependent AUC: {mean_auc:.4f}")
    for t, auc in zip(valid_times, auc_values):
        print(f"  Year {t:2.0f}: AUC = {auc:.4f}")

    # Plot time-dependent AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(valid_times, auc_values, 'o-', color='#2171b5', linewidth=2.5, markersize=8,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.8, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax.fill_between(valid_times, 0.5, auc_values, alpha=0.15, color='#2171b5')
    ax.set_xlabel('Time from baseline (years)', fontsize=12, fontfamily='Arial')
    ax.set_ylabel('Time-dependent AUC', fontsize=12, fontfamily='Arial')
    ax.set_ylim(0.45, 1.0)
    ax.set_xlim(0.5, valid_times.max() + 0.5)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add mean AUC annotation
    ax.text(0.98, 0.02, f'Mean AUC = {mean_auc:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontfamily='Arial',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(BASE, "figures", "time_dependent_auc.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(BASE, "figures", "time_dependent_auc.pdf"), bbox_inches='tight')
    plt.close()
    print("Saved: figures/time_dependent_auc.png/pdf")
except Exception as e:
    print(f"Time-dependent AUC error: {e}")
    mean_auc = None
    auc_values = None

# ─── D. Feature Importance Plot ──────────────────────────────────────────────
print("\n" + "="*70)
print("D. FEATURE IMPORTANCE PLOT (GBS)")
print("="*70)

name_map = {
    'AGE_AT_VISIT': 'Age', 'SEX': 'Sex', 'EDUCYRS': 'Education',
    'APOE4_carrier': 'APOE ε4', 'GBA_carrier': 'GBA mutation',
    'LRRK2_carrier': 'LRRK2 mutation', 'NP3TOT_BEST': 'UPDRS-III',
    'NHY_BEST': 'H&Y stage', 'NP2PTOT': 'UPDRS-II', 'NP1RTOT': 'UPDRS-I',
    'tremor_score': 'Tremor', 'pigd_score': 'PIGD score',
    'MCATOT': 'MoCA', 'HVLTRDLY': 'HVLT-R delayed', 'JLO_TOTRAW': 'JLO',
    'LNS_TOTRAW': 'Letter-Number Seq.', 'SDMTOTAL': 'Symbol Digit Mod.',
    'DVT_SFTANIM': 'Semantic fluency', 'NP1COG': 'Cognitive complaints',
    'NP1HALL': 'Hallucinations', 'NP1DPRS': 'Depression',
    'NP1ANXS': 'Anxiety', 'NP1APAT': 'Apathy',
    'RBDSQ_TOTAL': 'RBD score', 'SCAU_TOTAL': 'SCOPA-AUT',
    'MSEADLG': 'Schwab-England ADL', 'SAA_positive': 'SAA positive',
    'ortho_hypotension': 'Orthostatic hypotension',
    'PIGD_dominant': 'PIGD dominant', 'LEDD_TOTAL_CURRENT': 'LEDD',
    'HVLT_total_learning': 'HVLT-R learning',
    'CSF_ABeta42': 'CSF Aβ42', 'CSF_pTau': 'CSF p-tau',
    'CSF_tTau': 'CSF t-tau', 'CSF_aSyn': 'CSF α-syn',
    'Plasma_GFAP': 'Plasma GFAP', 'Plasma_NfL': 'Plasma NfL',
}

fig, ax = plt.subplots(figsize=(8, max(6, len(feat_imp) * 0.35)))
feat_imp_sorted = feat_imp.sort_values(ascending=True)
display_names = [name_map.get(v, v) for v in feat_imp_sorted.index]
colors = ['#2171b5' if imp > feat_imp.quantile(0.5) else '#9ecae1'
          for imp in feat_imp_sorted.values]

ax.barh(range(len(feat_imp_sorted)), feat_imp_sorted.values, color=colors,
        edgecolor='white', height=0.7)
ax.set_yticks(range(len(feat_imp_sorted)))
ax.set_yticklabels(display_names, fontsize=10, fontfamily='Arial')
ax.set_xlabel('Feature Importance', fontsize=12, fontfamily='Arial')
ax.tick_params(labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "feature_importance_gbs.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "feature_importance_gbs.pdf"), bbox_inches='tight')
plt.close()
print("Saved: figures/feature_importance_gbs.png/pdf")

# ─── E. Model Comparison Figure ─────────────────────────────────────────────
print("\n" + "="*70)
print("E. MODEL COMPARISON")
print("="*70)

models = {
    'Cox PH\n(LASSO)': (cox_results['cv_c_index_mean'], cox_results['cv_c_index_std']),
    'Gradient\nBoosting': (gbs_mean, gbs_std),
    'Random\nSurvival Forest': (rsf_mean, rsf_std),
}

fig, ax = plt.subplots(figsize=(6, 5))
model_names = list(models.keys())
means = [models[m][0] for m in model_names]
stds = [models[m][1] for m in model_names]
x_pos = range(len(model_names))
colors = ['#6baed6', '#2171b5', '#08519c']

bars = ax.bar(x_pos, means, yerr=stds, color=colors, edgecolor='white',
              width=0.6, capsize=8, error_kw={'linewidth': 2})

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=11,
            fontweight='bold', fontfamily='Arial')

ax.set_ylabel('Cross-validated C-index', fontsize=12, fontfamily='Arial')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, fontsize=10, fontfamily='Arial')
ax.set_ylim(0.55, 0.85)
ax.axhline(0.7, color='grey', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(0.8, color='green', linestyle=':', linewidth=1, alpha=0.3)
ax.tick_params(labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(BASE, "figures", "model_comparison.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(BASE, "figures", "model_comparison.pdf"), bbox_inches='tight')
plt.close()
print("Saved: figures/model_comparison.png/pdf")

# ─── Save extended results ───────────────────────────────────────────────────
extended_results = {
    'gradient_boosting': {
        'cv_c_index_mean': float(gbs_mean),
        'cv_c_index_std': float(gbs_std),
        'cv_c_index_folds': [float(s) for s in gbs_cv_scores]
    },
    'random_survival_forest': {
        'cv_c_index_mean': float(rsf_mean),
        'cv_c_index_std': float(rsf_std),
        'cv_c_index_folds': [float(s) for s in rsf_cv_scores]
    },
    'biomarker_sensitivity': {
        'n_patients': int(len(df_bio)),
        'n_events': int(y_event_bio.sum()),
        'base_cv_c_index': float(np.mean(bio_cox_scores)),
        'extended_cv_c_index': float(np.mean(bio_cv_scores)),
        'improvement': float(np.mean(bio_cv_scores) - np.mean(bio_cox_scores))
    },
    'time_dependent_auc': {
        'mean_auc': float(mean_auc) if mean_auc else None,
        'auc_by_year': {str(int(t)): float(a) for t, a in zip(valid_times, auc_values)} if auc_values is not None else None
    },
    'feature_importance_gbs': {v: float(imp) for v, imp in feat_imp.items()}
}

with open(os.path.join(BASE, "logs", "extended_results.json"), 'w') as f:
    json.dump(extended_results, f, indent=2)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Cox PH (LASSO):         C-index = {cox_results['cv_c_index_mean']:.4f} ± {cox_results['cv_c_index_std']:.4f}")
print(f"Gradient Boosting:      C-index = {gbs_mean:.4f} ± {gbs_std:.4f}")
print(f"Random Survival Forest: C-index = {rsf_mean:.4f} ± {rsf_std:.4f}")
if mean_auc:
    print(f"Mean time-dependent AUC (GBS):   {mean_auc:.4f}")
print(f"\nBiomarker sensitivity (N={len(df_bio)}):")
print(f"  Base:     {np.mean(bio_cox_scores):.4f}")
print(f"  Extended: {np.mean(bio_cv_scores):.4f}")
print(f"  Δ:        {np.mean(bio_cv_scores) - np.mean(bio_cox_scores):+.4f}")
print("\n✓ Extended analyses complete.")
