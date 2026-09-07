#!/usr/bin/env python3
"""R14 — bring webapp/index.html into line with the revised manuscript.

Fixes
  1. Baseline survival: the calculator centres the linear predictor on the MEDIAN
     reference patient but used S0(t) at covariate MEANS (lifelines default), which
     inflated every risk by a constant hazard factor. S0 is replaced by
     S(t | median patient) = S0_means(t) ^ exp(lp_median - lp_mean), for the full and
     the non-genetic model.
  2. Adds the non-genetic companion model (R12) behind a "genotype not available" box.
  3. Updates the performance statements (optimism-corrected 0.717; IPCW AUC 0.74-0.77
     through year 5, mean 0.80) and the reference line.
Writes webapp/index.html in place (backup webapp/index_v1_2026-03.html).
"""
import os, re, json, shutil
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from lifelines import CoxPHFitter

PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
HTML = os.path.join(PROJ, "webapp", "index.html"); BAK = os.path.join(PROJ, "webapp", "index_v1_2026-03.html")
if not os.path.exists(BAK): shutil.copy2(HTML, BAK)
art = json.load(open(os.path.join(PROJ, "revision", "data", "primary_artifacts.json")))
J12 = json.load(open(os.path.join(PROJ, "revision", "logs", "R12_nongenetic.json")))
selected = art["selected"]; sel_ng = J12["predictors"]

df = pd.read_csv(os.path.join(PROJ, "data", "analytical_dataset.csv")).dropna(subset=["time_years"]).reset_index(drop=True)
df["SAA_positive"] = (df["SAA_Status_Combined"].astype(str).str.lower().isin(["positive", "1", "1.0", "pos"])).astype(float)
df["HVLT_total_learning"] = df[["HVLTRT1", "HVLTRT2", "HVLTRT3"]].sum(axis=1, min_count=2)
Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[art["pred_vars"]]), columns=art["pred_vars"])
T = df["time_years"].values; E = df["event"].values.astype(int)
TIMES = [1, 2, 3, 5, 8, 10]

def model_block(vars_):
    m = Ximp[vars_].copy(); m["time_years"] = T; m["event"] = E
    cph = CoxPHFitter(penalizer=0.01).fit(m, "time_years", "event")
    coef = cph.params_; means = m[vars_].mean(); med = m[vars_].median()
    S0 = cph.baseline_survival_.iloc[:, 0]
    k = float(np.exp(float((coef * med).sum()) - float((coef * means).sum())))
    S_med = {t: float(np.interp(t, S0.index, S0.values)) ** k for t in TIMES}
    # cross-check against lifelines' own prediction for the median patient
    chk = cph.predict_survival_function(pd.DataFrame([med]))
    for t in TIMES:
        assert abs(float(np.interp(t, chk.index, chk.iloc[:, 0].values)) - S_med[t]) < 2e-3, t
    return {v: float(coef[v]) for v in vars_}, {v: float(med[v]) for v in vars_}, S_med
coef_f, med_f, S_f = model_block(selected)
coef_n, med_n, S_n = model_block(sel_ng)
assert all(abs(med_f[v] - art["medians"][v]) < 1e-9 for v in selected)
print("Full model  S(t|median):", {t: round(S_f[t], 4) for t in TIMES})
print("Non-genetic S(t|median):", {t: round(S_n[t], 4) for t in TIMES})

html = open(HTML).read()
def rep(old, new, count=1):
    global html
    assert html.count(old) == count, (html.count(old), old[:60]); html = html.replace(old, new)

# 1. baseline survival of the full model
old_s0 = re.search(r"const S0 = \{.*?\};", html, flags=re.S).group(0)
new_s0 = "const S0 = {\n" + ",\n".join(f"    {t}: {S_f[t]:.6f}" for t in TIMES) + "\n};"
rep(old_s0, new_s0)
rep("// Baseline survival at key timepoints (from fitted model)",
    "// Survival of the median reference patient, S(t | medians) = S0_means(t)^exp(lp_median - lp_mean).\n"
    "// Risk for a patient = 1 - S(t|medians)^exp(lp - lp_median). (v2: corrected centring, 2026-09-04)")

# 2. non-genetic companion model
ng_js = ("\n// Companion model WITHOUT genetic predictors (sensitivity analysis; Supplementary Figure 6)\n"
         "const NONGEN = {\n    COEFS: {\n" + ",\n".join(f"        {v}: {coef_n[v]:.6f}" for v in sel_ng) + "\n    },\n"
         "    S0: {\n" + ",\n".join(f"        {t}: {S_n[t]:.6f}" for t in TIMES) + "\n    },\n"
         "    MEDIANS: {\n" + ",\n".join(f"        {v}: {med_n[v]:g}" for v in sel_ng) + "\n    }\n};\n"
         "function activeModel() {\n"
         "    const nogen = document.getElementById('nogen') && document.getElementById('nogen').checked;\n"
         "    return nogen ? NONGEN : { COEFS: COEFS, S0: S0, MEDIANS: MEDIANS };\n}\n"
         "function toggleGenetics() {\n"
         "    const nogen = document.getElementById('nogen').checked;\n"
         "    document.getElementById('gba').disabled = nogen;\n"
         "    document.getElementById('lrrk2').disabled = nogen;\n"
         "    document.getElementById('modelNote').style.display = nogen ? 'block' : 'none';\n"
         "    if (document.getElementById('risk5yr').textContent.trim() !== '') { try { calculate(); } catch (e) {} }\n}\n")
rep("const LABELS = {", ng_js + "\nconst LABELS = {")
rep("    const vals = getInputValues();\n", "    const vals = getInputValues();\n    const M = activeModel();\n")
rep("""    let lp = 0;
    for (const key in COEFS) {
        lp += COEFS[key] * vals[key];
    }""", """    let lp = 0;
    for (const key in M.COEFS) {
        lp += M.COEFS[key] * vals[key];
    }""")
rep("""    let lp_median = 0;
    for (const key in COEFS) {
        lp_median += COEFS[key] * MEDIANS[key];
    }""", """    let lp_median = 0;
    for (const key in M.COEFS) {
        lp_median += M.COEFS[key] * M.MEDIANS[key];
    }""")
rep("""    for (const t in S0) {
        const surv = Math.pow(S0[t], relHazard);""", """    for (const t in M.S0) {
        const surv = Math.pow(M.S0[t], relHazard);""")
rep("""    for (const key in COEFS) {
        const contrib = COEFS[key] * (vals[key] - MEDIANS[key]);""", """    for (const key in M.COEFS) {
        const contrib = M.COEFS[key] * (vals[key] - M.MEDIANS[key]);""")

# checkbox in the Genetics section
rep("""                <div class="form-group">
                    <label>LRRK2 Mutation Carrier</label>
                    <select id="lrrk2">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
            </div>""", """                <div class="form-group">
                    <label>LRRK2 Mutation Carrier</label>
                    <select id="lrrk2">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group" style="grid-column: 1 / -1;">
                    <label style="display:flex;align-items:center;gap:8px;">
                        <input type="checkbox" id="nogen" onchange="toggleGenetics()" style="width:auto;">
                        Genotype not available: use the companion model without genetic predictors
                    </label>
                    <div id="modelNote" style="display:none;font-size:0.85em;color:#555;margin-top:6px;">
                        Non-genetic companion model (16 predictors; sensitivity analysis, Supplementary Figure 6).
                        Optimism-corrected C-index NGCORR. It cannot see the excess risk carried by GBA variants.
                    </div>
                </div>
            </div>""")
rep("NGCORR", f"{J12['nongenetic']['corrected_c']:.2f}")

# 3. performance statements and reference
rep("Model C-index = 0.73; time-dependent AUC = 0.82–0.92.",
    "Optimism-corrected C-index = 0.72; inverse-probability-of-censoring-weighted time-dependent AUC 0.74–0.77 through year 5 "
    "(mean 0.80 over 1–12 years). Internally validated in a single cohort only; external validation is pending.")
rep("Apparent C-index = 0.734; optimism-corrected = 0.709; mean time-dependent AUC = 0.866.",
    "Apparent C-index = 0.734; optimism-corrected = 0.717 (LASSO selection repeated in 300 bootstrap replicates); "
    "IPCW time-dependent AUC 0.74–0.77 through year 5 (mean 0.80 over 1–12 years). Risks are relative to the median reference patient.")
rep("Negida A. Development and Validation of a Clinical Nomogram for Predicting Cognitive Impairment in Parkinson's Disease. 2026.",
    "Negida A. A Multimodal Nomogram Predicting 5-Year Risk of Cognitive Impairment in De Novo Parkinson's Disease. "
    "Movement Disorders Clinical Practice, 2026 (revision under review).")
html = html.replace("<html", "<!-- v2 (2026-09-04): baseline survival re-centred on the median reference patient; "
                    "non-genetic companion model added; performance statements updated -->\n<html", 1)
open(HTML, "w").write(html)
print("webapp/index.html updated; backup at", os.path.basename(BAK))

# 4. numeric self-test of the JS logic in Python: median patient and a high-risk patient
def js_risk(coef, med, S, vals, t):
    lp = sum(coef[v] * vals[v] for v in coef); lpm = sum(coef[v] * med[v] for v in coef)
    return 1 - S[t] ** np.exp(lp - lpm)
print(f"median patient 5-yr risk: full {js_risk(coef_f, med_f, S_f, med_f, 5):.3f}, non-genetic {js_risk(coef_n, med_n, S_n, med_n, 5):.3f}")
hi = dict(med_f); hi.update({"AGE_AT_VISIT": 75, "GBA_carrier": 1, "SDMTOTAL": 30, "NP1COG": 2})
hi_n = {v: hi[v] for v in sel_ng}
print(f"75-y GBA carrier, SDMT 30, cog complaints 2: full {js_risk(coef_f, med_f, S_f, hi, 5):.3f}, non-genetic {js_risk(coef_n, med_n, S_n, hi_n, 5):.3f}")
