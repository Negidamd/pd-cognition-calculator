#!/usr/bin/env python3
"""
R02_reviewer_analyses.py — Revision
 R3  COGSTATE transitions 1->2 (MCI) and 2->3 (dementia)
 R4  GBA/LRRK2 variant-level pathogenicity (MDSGene, Rossi 2025, PMID 39927608)
 R5  Hallucination temporality — landmark/sensitivity (baseline all cognitively normal)
 R8  IPCW time-dependent AUC (proper estimator, truncated at adequate at-risk N)
 R9  Age-stratified discrimination (<60 / 60-<70 / >=70)
Primary model coefficients are held fixed; the primary linear predictor (lp) is
re-used as the risk score throughout.
"""
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc

SEED=42; np.random.seed(SEED)
PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
D=os.path.join(PROJ,"revision","data")
art=json.load(open(os.path.join(D,"primary_artifacts.json"))); selected=art['selected']
mdf=pd.read_csv(os.path.join(D,"primary_model_df.csv"))     # has PATNO, selected vars, time_years, event, lp, AGE_AT_VISIT_raw
full=pd.read_csv(os.path.join(PROJ,"data","analytical_dataset.csv")).dropna(subset=['time_years']).reset_index(drop=True)
res={}

# ── R4 — variant-level pathogenicity ─────────────────────────────────────────
print("="*66); print("R4 — GBA/LRRK2 variant-level pathogenicity"); print("="*66)
gv=pd.read_csv(os.path.join(D,"genetic_variants.csv"))
def gba_class(v):
    if pd.isna(v) or str(v) in ('0','0.0'): return 'none'
    s=str(v)
    severe_tokens=['L483P','L444P','D409H','IVS','c.762-2A>G','+1G>A','fs','*','X']  # splice/null/frameshift/nonsense/severe missense
    if any(t in s for t in severe_tokens): return 'severe'
    return 'mild'   # N409S(N370S) and other non-severe missense (R535H, F255Y, R502C, R159W)
gv['GBA_group']=gv['GBA_variant'].apply(gba_class)
gv['LRRK2_G2019S']=gv['LRRK2_variant'].astype(str).str.contains('G2019S').astype(int)
gv['LRRK2_pathogenic']=(~gv['LRRK2_variant'].astype(str).isin(['0','0.0','nan'])).astype(int)
m=mdf.merge(gv,on='PATNO',how='left')
print("GBA group counts:", m['GBA_group'].value_counts().to_dict())
print("LRRK2: G2019S=%d  non-G2019S pathogenic=%d"%(int(m['LRRK2_G2019S'].sum()),
      int((m['LRRK2_pathogenic']-m['LRRK2_G2019S']).clip(lower=0).sum())))
# event rate by GBA group
grp=m.groupby('GBA_group').agg(n=('event','size'),events=('event','sum')).reset_index()
grp['event_pct']=(100*grp['events']/grp['n']).round(1); print(grp.to_string(index=False))
# Sensitivity Cox: replace binary GBA_carrier with severe/mild dummies (vs none)
sev=m.copy(); sev['GBA_mild']=(sev['GBA_group']=='mild').astype(int); sev['GBA_severe']=(sev['GBA_group']=='severe').astype(int)
cols=[c for c in selected if c!='GBA_carrier']+['GBA_mild','GBA_severe']
mm=sev[cols+['time_years','event']].dropna()
cS=CoxPHFitter(penalizer=0.01).fit(mm,'time_years','event')
res['R4_gba']={'group_counts':m['GBA_group'].value_counts().to_dict(),
   'event_rate_by_group':grp.to_dict('records'),
   'HR_GBA_mild':float(cS.summary.loc['GBA_mild','exp(coef)']),'p_GBA_mild':float(cS.summary.loc['GBA_mild','p']),
   'HR_GBA_mild_ci':[float(cS.summary.loc['GBA_mild','exp(coef) lower 95%']),float(cS.summary.loc['GBA_mild','exp(coef) upper 95%'])],
   'HR_GBA_severe':float(cS.summary.loc['GBA_severe','exp(coef)']),'p_GBA_severe':float(cS.summary.loc['GBA_severe','p']),
   'HR_GBA_severe_ci':[float(cS.summary.loc['GBA_severe','exp(coef) lower 95%']),float(cS.summary.loc['GBA_severe','exp(coef) upper 95%'])]}
print(f"GBA mild  HR={res['R4_gba']['HR_GBA_mild']:.2f} p={res['R4_gba']['p_GBA_mild']:.3f}")
print(f"GBA severe HR={res['R4_gba']['HR_GBA_severe']:.2f} p={res['R4_gba']['p_GBA_severe']:.3f}")
# LRRK2 restricted to G2019S
lr=m.copy(); lr['LRRK2_carrier']=lr['LRRK2_G2019S']
mm2=lr[selected+['time_years','event']].dropna()
cL=CoxPHFitter(penalizer=0.01).fit(mm2,'time_years','event')
res['R4_lrrk2']={'HR_G2019S_only':float(cL.summary.loc['LRRK2_carrier','exp(coef)']),
   'p':float(cL.summary.loc['LRRK2_carrier','p']),
   'ci':[float(cL.summary.loc['LRRK2_carrier','exp(coef) lower 95%']),float(cL.summary.loc['LRRK2_carrier','exp(coef) upper 95%'])],
   'n_G2019S':int(m['LRRK2_G2019S'].sum()),'n_pathogenic_total':int(m['LRRK2_pathogenic'].sum())}
print(f"LRRK2 (G2019S-only) HR={res['R4_lrrk2']['HR_G2019S_only']:.2f} p={res['R4_lrrk2']['p']:.3f}  [primary binary HR 0.71]")

# ── R3 — COGSTATE transitions ────────────────────────────────────────────────
print("\n"+"="*66); print("R3 — COGSTATE transitions"); print("="*66)
traj=pd.read_csv(os.path.join(D,"cogstate_trajectory.csv")).sort_values(['PATNO','visit'])
# 1->2 : model discrimination for MCI-specific first event (event only if first conversion == 2)
mci=mdf.copy(); mci['event_mci']=((full['event']==1)&(full['event_cogstate']==2)).astype(int).values
c_12=concordance_index(mci['time_years'], -mci['lp'], mci['event_mci'])
print(f"1->2 (MCI-specific): events={int(mci['event_mci'].sum())}  C-index(primary lp)={c_12:.3f}")
# 2->3 : among patients who ever reached COGSTATE==2, time from first '2' to first '3'
rows=[]
for pid,g in traj.groupby('PATNO'):
    g=g.reset_index(drop=True)
    two=g[g['COGSTATE']==2]
    if len(two)==0: continue
    t0=two['years'].iloc[0]
    after=g[g['years']>t0]
    three=after[after['COGSTATE']>=3]
    if len(three)>0:
        rows.append({'PATNO':pid,'t':max(three['years'].iloc[0]-t0,0.25),'event':1})
    else:
        tmax=after['years'].max() if len(after) else t0
        rows.append({'PATNO':pid,'t':max((tmax-t0),0.25),'event':0})
prog=pd.DataFrame(rows).merge(mdf[['PATNO','lp']],on='PATNO',how='inner')
c_23=concordance_index(prog['t'], -prog['lp'], prog['event']) if prog['event'].sum()>=5 else np.nan
print(f"2->3 (MCI->dementia): N reached MCI={len(prog)}  progressed={int(prog['event'].sum())}  C-index(baseline lp)={c_23:.3f}")
res['R3']={'mci_events':int(mci['event_mci'].sum()),'c_index_1to2':float(c_12),
           'n_reached_mci':int(len(prog)),'n_progressed_2to3':int(prog['event'].sum()),
           'c_index_2to3':float(c_23) if not np.isnan(c_23) else None}

# ── R5 — hallucination temporality (landmark + sensitivity) ──────────────────
print("\n"+"="*66); print("R5 — Hallucination temporality"); print("="*66)
bh=pd.read_csv(os.path.join(D,"baseline_hallucination.csv"))
m5=mdf.merge(bh[['PATNO','hall_baseline_pos']],on='PATNO',how='left')
n_bh=int(m5['hall_baseline_pos'].sum())
# (a) Landmark: exclude early converters (event within first year) so that a
#     baseline symptom cannot merely be a concurrent marker of imminent conversion.
lm=m5[~((m5['event']==1)&(m5['time_years']<=1.0))].copy()
mmL=lm[selected+['time_years','event']].dropna()
cLm=CoxPHFitter(penalizer=0.01).fit(mmL,'time_years','event')
HR_lm=float(cLm.summary.loc['NP1HALL','exp(coef)']); p_lm=float(cLm.summary.loc['NP1HALL','p'])
ci_lm=[float(cLm.summary.loc['NP1HALL','exp(coef) lower 95%']),float(cLm.summary.loc['NP1HALL','exp(coef) upper 95%'])]
# (b) Sensitivity excluding baseline-hallucination-positive patients; refit model
#     without NP1HALL (near-constant after exclusion) to confirm other predictors/discrimination hold.
keep=m5[m5['hall_baseline_pos']==0]
sel_noh=[s for s in selected if s!='NP1HALL']
mmK=keep[sel_noh+['time_years','event']].dropna()
cK=CoxPHFitter(penalizer=0.01).fit(mmK,'time_years','event')
res['R5']={'n_baseline_hallucination':n_bh,
   'landmark_exclude_events_within_1yr':int(((m5['event']==1)&(m5['time_years']<=1.0)).sum()),
   'landmark_n':int(len(lm)),'landmark_events':int(lm['event'].sum()),
   'HR_NP1HALL_landmark':HR_lm,'p_NP1HALL_landmark':p_lm,'ci_landmark':ci_lm,
   'sensitivity_exclude_baseline_hall_n':int(len(keep)),
   'sensitivity_events':int(keep['event'].sum()),
   'c_index_without_NP1HALL':float(cK.concordance_index_)}
print(f"Baseline hallucination-positive: {n_bh}")
print(f"Landmark (exclude {res['R5']['landmark_exclude_events_within_1yr']} events <=1yr; n={len(lm)}): "
      f"NP1HALL HR={HR_lm:.2f} ({ci_lm[0]:.2f}-{ci_lm[1]:.2f}) p={p_lm:.3f}  [primary 1.57]")
print(f"Excl. baseline-hall (n={len(keep)}), model w/o NP1HALL: C={res['R5']['c_index_without_NP1HALL']:.3f}  [primary 0.734]")

# ── R9 — age-stratified discrimination ───────────────────────────────────────
print("\n"+"="*66); print("R9 — Age-stratified discrimination"); print("="*66)
mdf['agebin']=pd.cut(mdf['AGE_AT_VISIT_raw'],[0,60,70,200],labels=['<60','60-<70','>=70'],right=False)
ab=[]
for b in ['<60','60-<70','>=70']:
    s=mdf[mdf['agebin']==b]
    c=concordance_index(s['time_years'], -s['lp'], s['event'])
    ab.append({'bin':b,'n':int(len(s)),'events':int(s['event'].sum()),'c_index':float(c)})
    print(f"  Age {b:7s}: n={len(s):4d} events={int(s['event'].sum()):3d}  C={c:.3f}")
res['R9']=ab

# ── R8 — IPCW time-dependent AUC (proper, truncated) ─────────────────────────
print("\n"+"="*66); print("R8 — IPCW time-dependent AUC"); print("="*66)
ytr=np.array([(bool(e),t) for e,t in zip(mdf['event'],mdf['time_years'])],dtype=[('e',bool),('t',float)])
# at-risk table; truncate where at-risk >= 50
ts=sorted(mdf['time_years'].unique())
atrisk={t:int((mdf['time_years']>=t).sum()) for t in range(1,13)}
years=[t for t in range(1,13) if atrisk[t]>=50 and (mdf.loc[mdf['event']==1,'time_years']<=t).sum()>=10]
tmax=max(years)
print("At-risk by year:",{t:atrisk[t] for t in range(1,13)})
auc_vals,_=cumulative_dynamic_auc(ytr,ytr,mdf['lp'].values,np.array(years,dtype=float))
mean_auc=cumulative_dynamic_auc(ytr,ytr,mdf['lp'].values,np.array(years,dtype=float))[1]
by_year={str(y):float(a) for y,a in zip(years,auc_vals)}
res['R8']={'method':'IPCW cumulative/dynamic AUC (Uno), apparent',
   'truncated_at_year':int(tmax),'at_risk_at_tmax':int(atrisk[tmax]),
   'mean_auc':float(mean_auc),'auc_by_year':by_year,'at_risk_by_year':{str(t):atrisk[t] for t in range(1,13)}}
for y,a in by_year.items(): print(f"  yr {y}: IPCW-AUC={a:.3f} (at risk {atrisk[int(y)]})")
print(f"Mean IPCW-AUC (yr1-{tmax}) = {mean_auc:.3f}  [prev naive mean 0.866 up to yr12]")

json.dump(res, open(os.path.join(PROJ,'revision','logs','R02_reviewer.json'),'w'), indent=2)
print("\n✓ R02 complete.")
