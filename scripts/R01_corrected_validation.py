#!/usr/bin/env python3
"""
R01_corrected_validation.py — Revision
Reviewer/self-identified issues:
 * Leakage-corrected internal validation: LASSO selection + imputation + scaling
   are repeated INSIDE each bootstrap replicate (Harrell optimism), instead of
   selecting once on the full data. Gives an honest optimism-corrected C-index.
 * R6 — complete-case sensitivity analysis (no imputation).
 * R7 — multicollinearity of selected predictors (VIF).
Primary model / HRs are unchanged; only the validation metrics are updated.
"""
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

SEED = 42; np.random.seed(SEED)
PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
art = json.load(open(os.path.join(PROJ, "revision", "data", "primary_artifacts.json")))
pred_vars, selected, best_alpha = art['pred_vars'], art['selected'], art['best_alpha']

# Rebuild raw (unimputed) predictor matrix mirroring the primary preprocessing
df = pd.read_csv(os.path.join(PROJ, "data", "analytical_dataset.csv")).dropna(subset=['time_years']).reset_index(drop=True)
df['SAA_positive'] = (df['SAA_Status_Combined'].astype(str).str.lower().isin(['positive','1','1.0','pos'])).astype(float)
df.loc[df['SAA_Status_Combined'].isna(), 'SAA_positive'] = np.nan
df['HVLT_total_learning'] = df[['HVLTRT1','HVLTRT2','HVLTRT3']].sum(axis=1, min_count=2)
Xraw = df[pred_vars].copy()
y_time = df['time_years'].values; y_event = df['event'].values.astype(bool)
n = len(df)
results = {}

# ── (A) Leakage-corrected optimism bootstrap (selection INSIDE resampling) ───
print("="*66); print("A. Leakage-corrected internal validation (selection in bootstrap)"); print("="*66)
def pipeline_fit_predict(Xtr, ytime_tr, yev_tr, Xte):
    """Impute+scale on train, LASSO-select at fixed alpha, fit Cox; return
    (selected vars, train risk, test risk)."""
    imp = SimpleImputer(strategy='median').fit(Xtr)
    Xtr_i = pd.DataFrame(imp.transform(Xtr), columns=pred_vars)
    Xte_i = pd.DataFrame(imp.transform(Xte), columns=pred_vars)
    sc = StandardScaler().fit(Xtr_i)
    ysurv = np.array([(e,t) for e,t in zip(yev_tr, ytime_tr)], dtype=[('event',bool),('time',float)])
    lm = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=0.9, max_iter=10000, tol=1e-7)
    lm.fit(sc.transform(Xtr_i), ysurv)
    sel = [v for v,c in zip(pred_vars, lm.coef_.ravel()) if abs(c) > 0]
    if len(sel) < 2: sel = selected
    mdf = Xtr_i[sel].copy(); mdf['time_years']=ytime_tr; mdf['event']=yev_tr.astype(int)
    cph = CoxPHFitter(penalizer=0.01).fit(mdf, 'time_years', 'event')
    r_tr = cph.predict_partial_hazard(Xtr_i[sel]).values.ravel()
    r_te = cph.predict_partial_hazard(Xte_i[sel]).values.ravel()
    return sel, r_tr, r_te

# apparent C of the full pipeline on full data
sel_full, r_full, _ = pipeline_fit_predict(Xraw, y_time, y_event, Xraw)
app_c = concordance_index(y_time, -r_full, y_event)
B = 300; opt, nsel = [], []
for b in range(B):
    idx = np.random.choice(n, n, replace=True)
    try:
        sel_b, r_bb, r_bo = pipeline_fit_predict(Xraw.iloc[idx], y_time[idx], y_event[idx], Xraw)
        c_app = concordance_index(y_time[idx], -r_bb, y_event[idx])
        c_test = concordance_index(y_time, -r_bo, y_event)   # test on ORIGINAL sample
        opt.append(c_app - c_test); nsel.append(len(sel_b))
    except Exception:
        continue
optimism = float(np.mean(opt)); corrected = float(app_c - optimism)
results['corrected_validation'] = {
    'apparent_c_index_full_pipeline': float(app_c),
    'optimism_with_selection': optimism,
    'optimism_corrected_c_index': corrected,
    'n_bootstraps': len(opt),
    'mean_vars_selected_per_boot': float(np.mean(nsel)),
    'method': 'Harrell optimism; median imputation, standardization, and LASSO selection '
              '(penalty fixed at full-sample optimum) repeated within each bootstrap replicate'}
print(f"Apparent (full pipeline) C = {app_c:.4f}")
print(f"Optimism (incl. selection) = {optimism:.4f}")
print(f"Optimism-corrected C       = {corrected:.4f}  [prev reported 0.708]")
print(f"Mean vars/boot = {np.mean(nsel):.1f}")

# ── (B) R6 — complete-case sensitivity (no imputation) ───────────────────────
print("\n"+"="*66); print("B. R6 — Complete-case sensitivity (no imputation)"); print("="*66)
cc = df.dropna(subset=selected).reset_index(drop=True)
mdf_cc = cc[selected].copy(); mdf_cc['time_years']=cc['time_years']; mdf_cc['event']=cc['event'].astype(int)
cph_cc = CoxPHFitter(penalizer=0.01).fit(mdf_cc, 'time_years', 'event')
cc_c = cph_cc.concordance_index_
print(f"Complete-case N={len(cc)} events={int(cc['event'].sum())}  apparent C={cc_c:.4f}")
# compare HRs vs primary
prim = pd.read_csv(os.path.join(PROJ,'revision','data','primary_cox_summary.csv'), index_col=0)
hr_cmp = []
for v in selected:
    hr_cmp.append({'var':v,'HR_primary':float(np.exp(prim.loc[v,'coef'])),
                   'HR_completecase':float(cph_cc.summary.loc[v,'exp(coef)']),
                   'p_completecase':float(cph_cc.summary.loc[v,'p'])})
results['complete_case'] = {'n':int(len(cc)),'events':int(cc['event'].sum()),
                            'apparent_c_index':float(cc_c),'hr_comparison':hr_cmp}
cph_cc.summary.to_csv(os.path.join(PROJ,'revision','tables','cox_complete_case.csv'))
maxdiff = max(abs(h['HR_primary']-h['HR_completecase']) for h in hr_cmp)
print(f"Max |ΔHR| primary vs complete-case = {maxdiff:.3f}")

# ── (C) R7 — multicollinearity (VIF) of selected predictors ──────────────────
print("\n"+"="*66); print("C. R7 — Multicollinearity (VIF)"); print("="*66)
Xi = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(df[selected]), columns=selected)
Xc = add_constant(Xi)
vif = {selected[i]: float(variance_inflation_factor(Xc.values, i+1)) for i in range(len(selected))}
vif_sorted = dict(sorted(vif.items(), key=lambda kv: kv[1], reverse=True))
for v,val in vif_sorted.items(): print(f"  {v:22s} VIF={val:.2f}")
results['vif'] = vif_sorted
results['vif_max'] = float(max(vif.values()))
print(f"Max VIF = {max(vif.values()):.2f}  (all < 5 → no material collinearity)" if max(vif.values())<5
      else f"Max VIF = {max(vif.values()):.2f}")

json.dump(results, open(os.path.join(PROJ,'revision','logs','R01_validation.json'),'w'), indent=2)
print("\n✓ R01 complete.")
