#!/usr/bin/env python3
"""
R00_build_revision_data.py — Revision (MDCP MS 5920300)
(1) EXACTLY reproduce the primary Cox/LASSO model from the locked analytical dataset
    so that all published HRs remain identical (surgical track changes), and export
    reusable artifacts: model_df, per-patient linear predictor, baseline survival S0(t),
    selected variables.
(2) Pull augmented data required for reviewer analyses that is NOT in the locked file:
    - GBA / LRRK2 variant identity (R4 variant-level pathogenicity)
    - full longitudinal COGSTATE trajectory (R3 transitions 1->2, 2->3)
    - baseline vs incident hallucination timing (R5)
Nothing here modifies the locked analytical_dataset.csv or the primary model.
"""
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

SEED = 42
np.random.seed(SEED)

PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
MASTER = os.path.join(os.environ.get("PPMI_DATA_DIR", "."), "PPMI_Master_Merged.csv")
OUT = os.path.join(PROJ, "revision", "data"); os.makedirs(OUT, exist_ok=True)

VISIT_TO_YEARS = {0:0,1:0.25,2:0.5,3:0.75,4:1,5:2,6:3,7:3.5,8:4,9:4.5,10:5,
                  11:5.5,12:6,13:7,14:8,15:9,16:10,17:11,18:12,19:13,20:14,21:15,22:16}

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — reproduce primary model exactly (mirrors scripts/01_model_and_nomogram.py)
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(PROJ, "data", "analytical_dataset.csv"))
assert len(df) == 1180, f"locked dataset changed: {len(df)}"
df = df.dropna(subset=['time_years']).reset_index(drop=True)
N_MODEL = len(df)
print(f"[1] Modeling N after dropping missing time_years: {N_MODEL} (events={int(df['event'].sum())})")

with open(os.path.join(PROJ, "logs", "cohort_info.json")) as f:
    primary_candidates = json.load(f)['primary_candidates']

if 'SAA_Status_Combined' in df.columns:
    df['SAA_positive'] = (df['SAA_Status_Combined'].astype(str).str.lower()
                          .isin(['positive','1','1.0','pos'])).astype(float)
    df.loc[df['SAA_Status_Combined'].isna(), 'SAA_positive'] = np.nan
    primary_candidates = [v if v!='SAA_Status_Combined' else 'SAA_positive' for v in primary_candidates]

df['HVLT_total_learning'] = df[['HVLTRT1','HVLTRT2','HVLTRT3']].sum(axis=1, min_count=2)
primary_candidates = [v for v in primary_candidates if v not in ['HVLTRT1','HVLTRT2','HVLTRT3']]
primary_candidates.append('HVLT_total_learning')
primary_candidates = [v for v in primary_candidates if v != 'DVS_LNS']
pred_vars = [v for v in primary_candidates if v in df.columns]

X = df[pred_vars].copy()
y_time = df['time_years'].values
y_event = df['event'].values.astype(bool)

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=pred_vars)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
y_surv = np.array([(e,t) for e,t in zip(y_event,y_time)], dtype=[('event',bool),('time',float)])

alphas = np.logspace(-3, 0, 50)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
cv_scores = []
for alpha in alphas:
    fs = []
    for tr, te in cv.split(X_scaled, y_event.astype(int)):
        try:
            m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=0.9, max_iter=10000, tol=1e-7)
            m.fit(X_scaled[tr], y_surv[tr])
            p = m.predict(X_scaled[te])
            fs.append(concordance_index_censored(y_event[te], y_time[te], p)[0])
        except Exception:
            fs.append(0.5)
    cv_scores.append(np.mean(fs))
best_alpha = alphas[int(np.argmax(cv_scores))]
lasso = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=0.9, max_iter=10000, tol=1e-7)
lasso.fit(X_scaled, y_surv)
coefs = pd.Series(lasso.coef_.ravel(), index=pred_vars)
selected = list(coefs[coefs.abs() > 0].sort_values(key=abs, ascending=False).index)
print(f"[2] LASSO best alpha={best_alpha:.4f}; selected {len(selected)} vars")

with open(os.path.join(PROJ, "logs", "model_results.json")) as f:
    published = json.load(f)
assert set(selected) == set(published['selected_predictors']), "SELECTION DRIFT vs published model!"
print("    ✓ selected set matches published model exactly")

model_df = X_imputed[selected].copy()
model_df['time_years'] = y_time
model_df['event'] = y_event.astype(int)
cph = CoxPHFitter(penalizer=0.01)
cph.fit(model_df, duration_col='time_years', event_col='event')
app_c = cph.concordance_index_
print(f"[3] Apparent C-index reproduced: {app_c:.4f} (published {published['apparent_c_index']:.4f})")
assert abs(app_c - published['apparent_c_index']) < 1e-6, "C-index drift!"

# Reusable artifacts
lp = cph.predict_log_partial_hazard(model_df).values.ravel()          # centered linear predictor
model_df_out = model_df.copy()
model_df_out.insert(0, 'PATNO', df['PATNO'].values)
model_df_out['lp'] = lp
model_df_out['AGE_AT_VISIT_raw'] = df['AGE_AT_VISIT'].values
model_df_out.to_csv(os.path.join(OUT, "primary_model_df.csv"), index=False)

# Baseline survival S0(t) at reference (lifelines centers at means)
S0 = cph.baseline_survival_
S0.to_csv(os.path.join(OUT, "baseline_survival.csv"))
json.dump({'selected': selected, 'best_alpha': float(best_alpha),
           'apparent_c_index': float(app_c), 'pred_vars': pred_vars,
           'medians': {v: float(X_imputed[v].median()) for v in selected}},
          open(os.path.join(OUT, "primary_artifacts.json"), 'w'), indent=2)
cph.summary.to_csv(os.path.join(OUT, "primary_cox_summary.csv"))
print(f"[4] Saved primary artifacts to {OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — augmented data from master / longitudinal
# ─────────────────────────────────────────────────────────────────────────────
patnos = df['PATNO'].astype(int).tolist()
need = ['PATNO','EVENT_ID','visit','COGSTATE','COGCAT','NP1HALL','GBA','LRRK2','AGE_AT_VISIT']
mh = pd.read_csv(MASTER, usecols=lambda c: c in need, low_memory=False)
mh = mh[mh['PATNO'].isin(patnos)].copy()
mh['years'] = mh['visit'].map(VISIT_TO_YEARS)
print(f"\n[5] Master rows for cohort: {len(mh)} across {mh['PATNO'].nunique()} patients")

# (a) static variant identity per patient (first non-'0'/non-null; else '0')
def variant_id(series):
    vals = series.dropna().astype(str)
    nonwt = vals[~vals.isin(['0','0.0'])]
    if len(nonwt): return nonwt.iloc[0]
    if len(vals):  return '0'
    return np.nan
gba = mh.groupby('PATNO')['GBA'].apply(variant_id)
lrrk2 = mh.groupby('PATNO')['LRRK2'].apply(variant_id)
variants = pd.DataFrame({'PATNO': gba.index, 'GBA_variant': gba.values,
                         'LRRK2_variant': lrrk2.reindex(gba.index).values})
variants.to_csv(os.path.join(OUT, "genetic_variants.csv"), index=False)
print("[6] GBA variants:", variants['GBA_variant'].value_counts().to_dict())
print("    LRRK2 variants:", variants['LRRK2_variant'].value_counts().to_dict())

# (b) full COGSTATE trajectory (for transition analyses) — cohort patients only
cog = mh[['PATNO','visit','years','COGSTATE']].dropna(subset=['COGSTATE']).sort_values(['PATNO','visit'])
cog.to_csv(os.path.join(OUT, "cogstate_trajectory.csv"), index=False)
print(f"[7] COGSTATE trajectory rows: {len(cog)}")

# (c) baseline hallucination flag (NP1HALL>0 at first cog/baseline visit) for R5
first_visit = mh.sort_values(['PATNO','visit']).groupby('PATNO').first().reset_index()
bl_hall = first_visit[['PATNO','NP1HALL']].rename(columns={'NP1HALL':'NP1HALL_baseline'})
bl_hall['hall_baseline_pos'] = (bl_hall['NP1HALL_baseline'] > 0).astype(float)
bl_hall.to_csv(os.path.join(OUT, "baseline_hallucination.csv"), index=False)
print(f"[8] Baseline hallucination positive: {int(bl_hall['hall_baseline_pos'].sum())} / {len(bl_hall)}")

print("\n✓ R00 complete — primary model reproduced identically; augmented data saved.")
