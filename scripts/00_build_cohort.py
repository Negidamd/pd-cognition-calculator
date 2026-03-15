#!/usr/bin/env python3
"""
00_build_cohort.py — Build analytical cohort for PD cognitive impairment nomogram
PPMI dataset: Baseline predictors → time to cognitive impairment (MCI/Dementia)

Outcome: First COGSTATE ≥ 2 (MCI or Dementia per MDS criteria)
Population: PD patients with normal cognition (COGSTATE=1) at first assessment
Design: Survival analysis with time-to-event in years
"""

import pandas as pd
import numpy as np
import os, json
from datetime import datetime

# ─── Paths ───────────────────────────────────────────────────────────────────
# NOTE: Update BASE to your local PPMI data directory
BASE = os.environ.get("PPMI_DATA_DIR", ".")
PROJECT = os.path.join(BASE, "PD_Cognition_Nomogram")
MASTER = os.path.join(BASE, "PPMI_Master_Merged.csv")

# ─── Seed ────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─── PPMI visit-to-year mapping ──────────────────────────────────────────────
VISIT_TO_YEARS = {
    0: 0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1, 5: 2, 6: 3, 7: 3.5,
    8: 4, 9: 4.5, 10: 5, 11: 5.5, 12: 6, 13: 7, 14: 8, 15: 9,
    16: 10, 17: 11, 18: 12, 19: 13, 20: 14, 21: 15, 22: 16
}

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading PPMI Master Merged...")
df = pd.read_csv(MASTER, low_memory=False)
print(f"  Total: {len(df)} rows, {df['PATNO'].nunique()} patients")

# ─── Step 1: PD cohort only ──────────────────────────────────────────────────
pd_df = df[df['COHORT'] == 1].copy()
n_pd = pd_df['PATNO'].nunique()
print(f"\n[1] PD cohort: {n_pd} patients, {len(pd_df)} rows")

# ─── Step 2: Add years_from_baseline using visit mapping ─────────────────────
pd_df['years_from_bl'] = pd_df['visit'].map(VISIT_TO_YEARS)

# ─── Step 3: Identify patients with COGSTATE data ───────────────────────────
cog_df = pd_df[pd_df['COGSTATE'].notna()].sort_values(['PATNO', 'visit'])
patients_with_cog = cog_df['PATNO'].nunique()
print(f"[2] Patients with any COGSTATE data: {patients_with_cog}")

# ─── Step 4: Get first cognitive assessment per patient ──────────────────────
first_cog = cog_df.groupby('PATNO').first().reset_index()[['PATNO', 'COGSTATE', 'visit', 'years_from_bl']]
first_cog.columns = ['PATNO', 'first_cogstate', 'first_cog_visit', 'first_cog_year']

# ─── Step 5: Restrict to patients with normal cognition at first assessment ──
normal_at_first = first_cog[first_cog['first_cogstate'] == 1]['PATNO'].values
n_normal = len(normal_at_first)
n_mci_at_first = (first_cog['first_cogstate'] == 2).sum()
n_dem_at_first = (first_cog['first_cogstate'] == 3).sum()
print(f"[3] Normal cognition at first assessment: {n_normal}")
print(f"    MCI at first assessment (excluded): {n_mci_at_first}")
print(f"    Dementia at first assessment (excluded): {n_dem_at_first}")

# ─── Step 6: Require ≥2 cognitive assessments ───────────────────────────────
cog_counts = cog_df[cog_df['PATNO'].isin(normal_at_first)].groupby('PATNO')['COGSTATE'].count()
patients_2plus = cog_counts[cog_counts >= 2].index.values
n_excluded_1visit = n_normal - len(patients_2plus)
print(f"[4] ≥2 cognitive assessments: {len(patients_2plus)} (excluded {n_excluded_1visit} with only 1)")

# ─── Step 7: Determine event and time-to-event ──────────────────────────────
results = []
for pat in patients_2plus:
    pat_cog = cog_df[cog_df['PATNO'] == pat].sort_values('visit')

    # First assessment info
    first_row = pat_cog.iloc[0]
    first_visit = first_row['visit']
    first_year = VISIT_TO_YEARS.get(first_visit, first_visit * 0.5)

    # Find first conversion (COGSTATE >= 2) AFTER first normal assessment
    subsequent = pat_cog.iloc[1:]  # skip first (which is normal)
    conversion = subsequent[subsequent['COGSTATE'] >= 2]

    if len(conversion) > 0:
        event = 1
        event_row = conversion.iloc[0]
        event_visit = event_row['visit']
        event_year = VISIT_TO_YEARS.get(event_visit, event_visit * 0.5)
        time_years = event_year - first_year
        event_cogstate = event_row['COGSTATE']
    else:
        event = 0
        last_row = pat_cog.iloc[-1]
        last_visit = last_row['visit']
        last_year = VISIT_TO_YEARS.get(last_visit, last_visit * 0.5)
        time_years = last_year - first_year
        event_cogstate = 1

    # Ensure positive time
    if time_years <= 0:
        time_years = 0.25  # minimum follow-up

    results.append({
        'PATNO': pat,
        'event': event,
        'time_years': time_years,
        'event_cogstate': event_cogstate,
        'first_cog_visit': first_visit
    })

surv_df = pd.DataFrame(results)
print(f"\n[5] Survival outcomes:")
print(f"    Events (MCI/Dementia): {surv_df['event'].sum()}")
print(f"    Censored: {(surv_df['event']==0).sum()}")
print(f"    Median follow-up (all): {surv_df['time_years'].median():.1f} years")
print(f"    Median time-to-event (converters): {surv_df[surv_df['event']==1]['time_years'].median():.1f} years")
print(f"    Event type breakdown:")
print(f"      MCI (COGSTATE=2): {(surv_df['event_cogstate']==2).sum()}")
print(f"      Dementia (COGSTATE=3): {(surv_df['event_cogstate']==3).sum()}")

# ─── Step 8: Get baseline predictors ────────────────────────────────────────
# Use the visit corresponding to first cognitive assessment (usually BL)
# For most patients this is BL; merge baseline data

# Get baseline (visit=0) data for each patient
bl_data = pd_df[pd_df['visit'] == 0].copy()
# For patients whose first cog was not at BL, also get their first cog visit data
# But prefer BL data for predictors

# If patient not in BL, use SC data
sc_data = pd_df[pd_df['EVENT_ID'] == 'SC'].copy()

# Combine: BL preferred, SC as fallback
bl_patients = set(bl_data['PATNO'].values)
sc_fallback = sc_data[~sc_data['PATNO'].isin(bl_patients)]
baseline = pd.concat([bl_data, sc_fallback], ignore_index=True)
baseline = baseline.drop_duplicates(subset='PATNO', keep='first')

print(f"\n[6] Baseline data available for: {baseline['PATNO'].nunique()} PD patients")

# ─── Step 9: Define predictor variables ──────────────────────────────────────
# Demographics
demo_vars = ['AGE_AT_VISIT', 'SEX', 'EDUCYRS']

# Genetics
genetic_vars = ['APOE', 'GBA', 'LRRK2', 'SNCA']

# Motor
motor_vars = ['NP3TOT_BEST', 'NHY_BEST', 'NP2PTOT', 'NP1RTOT',
              'tremor_score', 'pigd_score']

# Cognitive tests (baseline)
cog_vars = ['MCATOT', 'HVLTRDLY', 'HVLTRT1', 'HVLTRT2', 'HVLTRT3',
            'JLO_TOTRAW', 'LNS_TOTRAW', 'SDMTOTAL',
            'DVT_SFTANIM', 'DVS_LNS']

# Neuropsychiatric
neuropsych_vars = ['NP1COG', 'NP1HALL', 'NP1DPRS', 'NP1ANXS', 'NP1APAT']

# Sleep / RBD
sleep_vars = ['RBDSQ_TOTAL']

# Autonomic
autonomic_vars = ['SCAU_TOTAL']

# Functional
functional_vars = ['MSEADLG']

# Olfaction
olfaction_vars = ['UPSIT_PRCNTGE']

# Biomarkers - CSF
csf_vars = ['CSF_ABeta42', 'CSF_pTau', 'CSF_tTau', 'CSF_aSyn', 'CSF_NfL', 'CSF_GFAP']

# Biomarkers - Plasma
plasma_vars = ['Plasma_GFAP', 'Plasma_NfL']

# SAA
saa_vars = ['SAA_Status_Combined']

# DATScan
datscan_vars = ['DATSCAN']

# Cardiovascular
cardio_vars = ['SYSSUP', 'DIASUP', 'SYSSTND', 'DIASTND', 'HRSUP', 'HRSTND']

# LEDD
ledd_vars = ['LEDD_TOTAL_CURRENT']

# All candidate predictors
all_predictors = (demo_vars + genetic_vars + motor_vars + cog_vars +
                  neuropsych_vars + sleep_vars + autonomic_vars +
                  functional_vars + olfaction_vars + csf_vars +
                  plasma_vars + saa_vars + datscan_vars + cardio_vars + ledd_vars)

# ─── Step 10: Build analytical dataset ───────────────────────────────────────
# Merge survival outcomes with baseline predictors
analytic = surv_df.merge(baseline[['PATNO'] + [v for v in all_predictors if v in baseline.columns]],
                         on='PATNO', how='inner')

print(f"\n[7] Analytical dataset: {len(analytic)} patients")
print(f"    Events: {analytic['event'].sum()}, Censored: {(analytic['event']==0).sum()}")

# ─── Step 11: Derive additional variables ────────────────────────────────────
# Orthostatic hypotension
if 'SYSSUP' in analytic.columns and 'SYSSTND' in analytic.columns:
    analytic['OH_sys'] = analytic['SYSSUP'] - analytic['SYSSTND']  # drop in systolic
    analytic['OH_dia'] = analytic['DIASUP'] - analytic['DIASTND']
    analytic['ortho_hypotension'] = ((analytic['OH_sys'] >= 20) | (analytic['OH_dia'] >= 10)).astype(float)

# APOE4 carrier status (coded as 'E3/E4', 'E4/E4', etc.)
if 'APOE' in analytic.columns:
    analytic['APOE4_carrier'] = analytic['APOE'].astype(str).str.contains('E4', na=False).astype(float)
    analytic.loc[analytic['APOE'].isna(), 'APOE4_carrier'] = np.nan

# GBA carrier (coded as variant name or '0' for wildtype)
if 'GBA' in analytic.columns:
    analytic['GBA_carrier'] = (~analytic['GBA'].isin(['0', 0]) & analytic['GBA'].notna()).astype(float)
    analytic.loc[analytic['GBA'].isna(), 'GBA_carrier'] = np.nan

# LRRK2 carrier
if 'LRRK2' in analytic.columns:
    analytic['LRRK2_carrier'] = (~analytic['LRRK2'].isin(['0', 0]) & analytic['LRRK2'].notna()).astype(float)
    analytic.loc[analytic['LRRK2'].isna(), 'LRRK2_carrier'] = np.nan

# TD/PIGD ratio
if 'tremor_score' in analytic.columns and 'pigd_score' in analytic.columns:
    analytic['PIGD_dominant'] = (analytic['pigd_score'] > analytic['tremor_score']).astype(float)

# CSF Abeta/tau ratio
if 'CSF_ABeta42' in analytic.columns and 'CSF_pTau' in analytic.columns:
    analytic['CSF_Abeta_pTau_ratio'] = analytic['CSF_ABeta42'] / analytic['CSF_pTau'].replace(0, np.nan)

# ─── Step 12: Report missingness ────────────────────────────────────────────
print("\n[8] Variable availability in analytical dataset:")
final_vars = (demo_vars + ['APOE4_carrier', 'GBA_carrier', 'LRRK2_carrier'] +
              motor_vars + cog_vars + neuropsych_vars + sleep_vars +
              autonomic_vars + functional_vars + olfaction_vars +
              csf_vars + plasma_vars + saa_vars + datscan_vars +
              ['ortho_hypotension', 'PIGD_dominant', 'CSF_Abeta_pTau_ratio', 'LEDD_TOTAL_CURRENT'])

for v in final_vars:
    if v in analytic.columns:
        avail = analytic[v].notna().sum()
        pct = 100 * avail / len(analytic)
        print(f"  {v:30s}: {avail:4d}/{len(analytic)} ({pct:5.1f}%)")

# ─── Step 13: Select variables with ≥60% availability ───────────────────────
# Two tiers: primary model (≥70% available), extended model (with CSF/plasma)
primary_candidates = []
extended_candidates = []
for v in final_vars:
    if v in analytic.columns:
        pct = 100 * analytic[v].notna().sum() / len(analytic)
        if pct >= 70:
            primary_candidates.append(v)
        elif pct >= 30:
            extended_candidates.append(v)

print(f"\n[9] Primary model candidates (≥70% available): {len(primary_candidates)}")
for v in primary_candidates:
    print(f"    {v}")
print(f"\n    Extended candidates (30-70% available): {len(extended_candidates)}")
for v in extended_candidates:
    print(f"    {v}")

# ─── Step 14: Save analytical dataset ───────────────────────────────────────
output_path = os.path.join(PROJECT, "data", "analytical_dataset.csv")
analytic.to_csv(output_path, index=False)
print(f"\n[10] Saved analytical dataset: {output_path}")
print(f"     Shape: {analytic.shape}")
print(f"     Events: {analytic['event'].sum()}, Censored: {(analytic['event']==0).sum()}")

# Save cohort info
cohort_info = {
    'total_pd': int(n_pd),
    'patients_with_cogstate': int(patients_with_cog),
    'normal_at_first': int(n_normal),
    'mci_at_first_excluded': int(n_mci_at_first),
    'dementia_at_first_excluded': int(n_dem_at_first),
    'excluded_1_visit': int(n_excluded_1visit),
    'analytical_n': int(len(analytic)),
    'events': int(analytic['event'].sum()),
    'censored': int((analytic['event'] == 0).sum()),
    'median_followup_years': float(analytic['time_years'].median()),
    'primary_candidates': primary_candidates,
    'extended_candidates': extended_candidates
}
with open(os.path.join(PROJECT, "logs", "cohort_info.json"), 'w') as f:
    json.dump(cohort_info, f, indent=2)

# ─── Step 15: Exclusion log ─────────────────────────────────────────────────
exclusion_log = f"""# Exclusion Log — PD Cognitive Impairment Nomogram
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## CONSORT-Style Flow

1. Total PPMI participants: {df['PATNO'].nunique()}
2. PD cohort (COHORT=1): {n_pd}
3. With any COGSTATE assessment: {patients_with_cog}
4. Normal cognition (COGSTATE=1) at first assessment: {n_normal}
   - Excluded MCI at first assessment: {n_mci_at_first}
   - Excluded Dementia at first assessment: {n_dem_at_first}
5. With ≥2 cognitive assessments: {len(patients_2plus)}
   - Excluded <2 assessments: {n_excluded_1visit}
6. With baseline predictor data: {len(analytic)}

## Final Analytical Dataset
- N = {len(analytic)}
- Events (cognitive impairment): {analytic['event'].sum()} ({100*analytic['event'].sum()/len(analytic):.1f}%)
  - MCI: {(analytic['event_cogstate']==2).sum()}
  - Dementia: {(analytic['event_cogstate']==3).sum()}
- Censored: {(analytic['event']==0).sum()} ({100*(analytic['event']==0).sum()/len(analytic):.1f}%)
- Median follow-up: {analytic['time_years'].median():.1f} years
- Median time-to-event (converters): {analytic[analytic['event']==1]['time_years'].median():.1f} years
"""

with open(os.path.join(PROJECT, "logs", "exclusion_log.md"), 'w') as f:
    f.write(exclusion_log)

print("\n✓ Cohort building complete.")
print(f"  Analytical dataset: {output_path}")
print(f"  Exclusion log: {os.path.join(PROJECT, 'logs', 'exclusion_log.md')}")
