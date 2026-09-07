#!/usr/bin/env python3
"""R12 — Sensitivity analysis: nomogram WITHOUT genetic predictors (2026-09-04).

Same locked analytical dataset (n = 1,152), same median imputation, same Cox
specification (lifelines CoxPHFitter, ridge penalizer 0.01), same validation
recipe as the primary model, but with GBA_carrier and LRRK2_carrier removed from
the 18 LASSO-selected predictors (16 predictors remain). Intended for settings in
which genotyping is unavailable.

Outputs
  revision/logs/R12_nongenetic.json         all numbers used in the text
  revision/tables/SuppTable_nongenetic.csv  HR table (primary vs non-genetic)
  figures/SuppFig_nomogram_nongenetic.png/.pdf  nomogram + risk conversion table
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

SEED = 42; np.random.seed(SEED)
PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
FIG = os.path.join(PROJ, "figures"); LOG = os.path.join(PROJ, "revision", "logs"); TAB = os.path.join(PROJ, "revision", "tables")
art = json.load(open(os.path.join(PROJ, "revision", "data", "primary_artifacts.json")))
selected = art["selected"]; GEN = ["GBA_carrier", "LRRK2_carrier"]
sel_ng = [v for v in selected if v not in GEN]

# ── data: identical preprocessing to R03 (primary reconstruction) ─────────────
df = pd.read_csv(os.path.join(PROJ, "data", "analytical_dataset.csv")).dropna(subset=["time_years"]).reset_index(drop=True)
df["SAA_positive"] = (df["SAA_Status_Combined"].astype(str).str.lower().isin(["positive", "1", "1.0", "pos"])).astype(float)
df["HVLT_total_learning"] = df[["HVLTRT1", "HVLTRT2", "HVLTRT3"]].sum(axis=1, min_count=2)
Ximp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[art["pred_vars"]]), columns=art["pred_vars"])
T = df["time_years"].values; E = df["event"].values.astype(int)
assert len(df) == 1152 and E.sum() == 441, (len(df), E.sum())

def fit(vars_):
    m = Ximp[vars_].copy(); m["time_years"] = T; m["event"] = E
    return CoxPHFitter(penalizer=0.01).fit(m, "time_years", "event"), m

cph_p, mdf_p = fit(selected)       # primary (reproduction check)
cph_n, mdf_n = fit(sel_ng)         # non-genetic
print(f"Primary apparent C = {cph_p.concordance_index_:.4f} (published 0.734)")
print(f"Non-genetic apparent C = {cph_n.concordance_index_:.4f}")
assert abs(cph_p.concordance_index_ - art["apparent_c_index"]) < 1e-4

# ── 10-fold stratified CV and 300-rep optimism bootstrap (fixed variable set) ─
def cv_c(vars_):
    skf = StratifiedKFold(10, shuffle=True, random_state=SEED); cs = []
    for tr, te in skf.split(Ximp, E):
        m = Ximp.iloc[tr][vars_].copy(); m["time_years"] = T[tr]; m["event"] = E[tr]
        c = CoxPHFitter(penalizer=0.01).fit(m, "time_years", "event")
        r = c.predict_partial_hazard(Ximp.iloc[te][vars_]).values.ravel()
        cs.append(concordance_index(T[te], -r, E[te]))
    return float(np.mean(cs)), float(np.std(cs))
def optimism(vars_, B=300):
    n = len(df); opt = []
    m_full = Ximp[vars_].copy(); m_full["time_years"] = T; m_full["event"] = E
    app = CoxPHFitter(penalizer=0.01).fit(m_full, "time_years", "event").concordance_index_
    rng = np.random.RandomState(SEED)
    for b in range(B):
        idx = rng.choice(n, n, replace=True)
        m = Ximp.iloc[idx][vars_].reset_index(drop=True); m["time_years"] = T[idx]; m["event"] = E[idx]
        c = CoxPHFitter(penalizer=0.01).fit(m, "time_years", "event")
        r_b = c.predict_partial_hazard(m[vars_]).values.ravel()          # apparent on the bootstrap sample
        r_o = c.predict_partial_hazard(Ximp[vars_]).values.ravel()       # tested on the original sample
        opt.append(concordance_index(T[idx], -r_b, E[idx]) - concordance_index(T, -r_o, E))
    return float(app), float(np.mean(opt)), float(app - np.mean(opt))
cv_n = cv_c(sel_ng); app_n, opt_n, cor_n = optimism(sel_ng)
print(f"Non-genetic: CV C = {cv_n[0]:.3f} ± {cv_n[1]:.3f}; optimism {opt_n:.4f}; corrected C = {cor_n:.3f}")

# ── agreement with the primary model and performance within carriers ─────────
lp_p = cph_p.predict_log_partial_hazard(mdf_p[selected]).values.ravel()
lp_n = cph_n.predict_log_partial_hazard(mdf_n[sel_ng]).values.ravel()
rho = float(pd.Series(lp_p).corr(pd.Series(lp_n), method="spearman"))
def tert(x): return pd.qcut(x, 3, labels=[0, 1, 2]).astype(int)
agree = float((tert(lp_p) == tert(lp_n)).mean())
carriers = (Ximp["GBA_carrier"] == 1) | (Ximp["LRRK2_carrier"] == 1)
c_in_car = {"primary": float(concordance_index(T[carriers], -lp_p[carriers], E[carriers])),
            "nongenetic": float(concordance_index(T[carriers], -lp_n[carriers], E[carriers]))}
c_in_non = {"primary": float(concordance_index(T[~carriers], -lp_p[~carriers], E[~carriers])),
            "nongenetic": float(concordance_index(T[~carriers], -lp_n[~carriers], E[~carriers]))}
print(f"Spearman rho of linear predictors = {rho:.3f}; tertile agreement = {agree:.1%}")
print("C within carriers:", c_in_car, "| within non-carriers:", c_in_non)

# ── HR table ─────────────────────────────────────────────────────────────────
NAME = {"AGE_AT_VISIT": "Age", "SDMTOTAL": "SDMT", "NP1COG": "Cognitive complaints (1.1)", "GBA_carrier": "GBA carrier",
        "SCAU_TOTAL": "SCOPA-AUT total", "RBDSQ_TOTAL": "RBD questionnaire", "MSEADLG": "Schwab-England ADL",
        "HVLT_total_learning": "HVLT-R total learning", "DVT_SFTANIM": "Semantic fluency", "LRRK2_carrier": "LRRK2 carrier",
        "EDUCYRS": "Education", "NP2PTOT": "MDS-UPDRS II", "NP1RTOT": "MDS-UPDRS I", "NP1HALL": "Hallucinations (1.2)",
        "LNS_TOTRAW": "Letter-Number Sequencing", "SEX": "Sex (male)", "NP1APAT": "Apathy (1.5)", "PIGD_dominant": "PIGD-dominant"}
rows = []
for v in selected:
    sp = cph_p.summary.loc[v]
    if v in sel_ng:
        sn = cph_n.summary.loc[v]
        rows.append({"Predictor": NAME[v], "HR (primary)": f"{sp['exp(coef)']:.2f}",
                     "HR (non-genetic)": f"{sn['exp(coef)']:.2f}",
                     "95% CI (non-genetic)": f"{sn['exp(coef) lower 95%']:.2f}–{sn['exp(coef) upper 95%']:.2f}",
                     "p (non-genetic)": "<0.001" if sn["p"] < 0.001 else f"{sn['p']:.3f}", "coef": float(sn["coef"])})
    else:
        rows.append({"Predictor": NAME[v], "HR (primary)": f"{sp['exp(coef)']:.2f}", "HR (non-genetic)": "—",
                     "95% CI (non-genetic)": "—", "p (non-genetic)": "—", "coef": np.nan})
tab = pd.DataFrame(rows); tab.to_csv(os.path.join(TAB, "SuppTable_nongenetic.csv"), index=False)
maxdiff = max(abs(float(r["HR (primary)"]) - float(r["HR (non-genetic)"])) for r in rows if r["HR (non-genetic)"] != "—")
print(f"Max |ΔHR| primary vs non-genetic (shared predictors) = {maxdiff:.2f}")

# ── nomogram (identical construction to R03) ─────────────────────────────────
coef = cph_n.params_; means = mdf_n[sel_ng].mean(); S0 = cph_n.baseline_survival_.iloc[:, 0]
S0_at = lambda t: float(np.interp(t, S0.index, S0.values))
lo = {v: float(mdf_n[v].quantile(0.01)) for v in sel_ng}; hi = {v: float(mdf_n[v].quantile(0.99)) for v in sel_ng}
binvars = ["SEX", "PIGD_dominant"]
for v in binvars: lo[v], hi[v] = 0.0, 1.0
rng_ = {v: coef[v] * (hi[v] - lo[v]) for v in sel_ng}
scale = 100.0 / max(abs(r) for r in rng_.values())
ref = {v: (lo[v] if coef[v] > 0 else hi[v]) for v in sel_ng}
pts = lambda v, x: coef[v] * (x - ref[v]) * scale
raw_lp_min = sum(coef[v] * ref[v] for v in sel_ng); offset = raw_lp_min - float((coef * means).sum())
maxT = sum(abs(rng_[v]) * scale for v in sel_ng)
def total_to_risk(Tp, t): return 1 - S0_at(t) ** np.exp(Tp / scale + offset)
name_map = {"AGE_AT_VISIT": "Age (years)", "SEX": "Sex (1=Male)", "EDUCYRS": "Education (years)", "NP2PTOT": "MDS-UPDRS II",
            "NP1RTOT": "MDS-UPDRS I", "HVLT_total_learning": "HVLT-R total learning", "LNS_TOTRAW": "Letter-Number Seq.",
            "SDMTOTAL": "Symbol Digit Modalities", "DVT_SFTANIM": "Semantic fluency (animals)", "NP1COG": "Cognitive complaints (1.1)",
            "NP1HALL": "Hallucinations (1.2)", "NP1APAT": "Apathy (1.5)", "RBDSQ_TOTAL": "RBD questionnaire",
            "SCAU_TOTAL": "SCOPA-AUT total", "MSEADLG": "Schwab-England ADL", "PIGD_dominant": "PIGD-dominant"}
order = sorted(sel_ng, key=lambda v: abs(rng_[v]), reverse=True)
nrow = len(order); tbl_totals = list(range(80, 340, 40))
fig_h = (nrow + 2) * 0.52 + (len(tbl_totals) + 2) * 0.30 + 0.6
fig = plt.figure(figsize=(8.6, fig_h))
gs = GridSpec(2, 1, height_ratios=[(nrow + 2) * 0.52, (len(tbl_totals) + 2) * 0.30 + 0.6], hspace=0.08)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1]); axt.axis("off")
ypos = np.arange(nrow + 1, 0, -1); bh = 0.17; colors = sns.color_palette("muted", nrow)
def hline(y, x0, x1, c="k", lw=1.8): ax.plot([x0, x1], [y, y], "-", color=c, lw=lw)
def tick(x, y, c="k", lw=1.3, h=bh): ax.plot([x, x], [y - h, y + h], "-", color=c, lw=lw)
def thin(pairs, min_gap):
    kept = []; last = -1e9
    for p in sorted(pairs, key=lambda z: z[0]):
        if p[0] - last >= min_gap: kept.append(p); last = p[0]
    return kept
y0 = ypos[0]; hline(y0, 0, 100, lw=2)
for p in range(0, 101, 10):
    tick(p, y0, lw=1.4); ax.text(p, y0 + 0.30, str(p), ha="center", va="bottom", fontsize=8.5, family="Arial")
ax.text(-16, y0, "Points", ha="right", va="center", fontsize=10.5, fontweight="bold", family="Arial")
for i, v in enumerate(order):
    y = ypos[i + 1]; c = colors[i]
    if v in binvars:
        vals = [0, 1]; labs = {"SEX": ["F", "M"]}.get(v, ["No", "Yes"])
    else:
        vals = list(np.linspace(lo[v], hi[v], 6)); labs = [f"{x:.0f}" if abs(x) >= 1 else f"{x:.1f}" for x in vals]
    P = [pts(v, x) for x in vals]; p0, p1 = min(P), max(P); hline(y, p0, p1, c=c, lw=2.4)
    for px in P: tick(px, y, c=c)
    li = int(np.argmin(P)); ri = int(np.argmax(P))
    ax.text(p0 - 1.3, y, labs[li], ha="right", va="center", fontsize=8, family="Arial")
    ax.text(p1 + 1.3, y, labs[ri], ha="left", va="center", fontsize=8, family="Arial")
    for px, lb in thin([(px, lb) for px, lb in zip(P, labs) if px not in (p0, p1)], 11):
        if px - p0 >= 11 and p1 - px >= 11: ax.text(px, y - 0.32, lb, ha="center", va="top", fontsize=7.5, family="Arial")
    ax.text(-16, y, name_map.get(v, v), ha="right", va="center", fontsize=9.5, family="Arial")
ax.set_xlim(-58, 104); ax.set_ylim(ypos[-1] - 0.8, y0 + 1.2); ax.axis("off")
colx = [0.16, 0.45, 0.635, 0.82]; headers = ["Total points", "3-year risk", "5-year risk", "8-year risk"]
xL, xR = 0.02, 0.97; nrows = len(tbl_totals); ytop = 0.92; ybot = 0.06; rh = (ytop - 0.14 - ybot) / nrows; yhead = ytop
for cx, h in zip(colx, headers): axt.text(cx, yhead, h, ha="center", va="center", fontsize=9.5, fontweight="bold", family="Arial")
axt.plot([xL, xR], [yhead + 0.06] * 2, "k-", lw=1.4); axt.plot([xL, xR], [yhead - 0.06] * 2, "k-", lw=1.0)
conv = {}
for i, Tp in enumerate(tbl_totals):
    yr = yhead - 0.12 - (i + 0.5) * rh
    axt.text(colx[0], yr, str(Tp), ha="center", va="center", fontsize=9, family="Arial")
    conv[Tp] = {}
    for j, t in enumerate([3, 5, 8]):
        conv[Tp][t] = float(total_to_risk(Tp, t))
        axt.text(colx[j + 1], yr, f"{conv[Tp][t]*100:.0f}%", ha="center", va="center", fontsize=9, family="Arial")
axt.plot([xL, xR], [yhead - 0.12 - nrows * rh] * 2, "k-", lw=1.4); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
dpi_png = int(1790 / max(8.6, fig_h))
fig.savefig(os.path.join(FIG, "SuppFig_nomogram_nongenetic.png"), dpi=min(300, dpi_png), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "SuppFig_nomogram_nongenetic.pdf"), bbox_inches="tight"); plt.close()
raw_lp = (mdf_n[sel_ng] * coef).sum(axis=1); T_all = (raw_lp - raw_lp_min) * scale
med_T = float(T_all.median()); med_risk5 = float(total_to_risk(med_T, 5))
print(f"Non-genetic nomogram: maxT={maxT:.0f}; median patient {med_T:.0f} pts -> 5-yr risk {med_risk5:.1%}")

out = {"n": int(len(df)), "events": int(E.sum()), "predictors": sel_ng, "removed": GEN,
       "primary_apparent_c": float(cph_p.concordance_index_),
       "nongenetic": {"apparent_c": app_n, "cv_c_mean": cv_n[0], "cv_c_sd": cv_n[1], "optimism": opt_n, "corrected_c": cor_n},
       "spearman_lp": rho, "tertile_agreement": agree, "c_within_carriers": c_in_car, "c_within_noncarriers": c_in_non,
       "n_carriers": int(carriers.sum()), "max_abs_dHR_shared": maxdiff,
       "S0_means": {t: S0_at(t) for t in (3, 5, 8)}, "coef": {v: float(coef[v]) for v in sel_ng},
       "means": {v: float(means[v]) for v in sel_ng}, "conversion_table": conv, "median_total_points": med_T, "median_5yr_risk": med_risk5}
json.dump(out, open(os.path.join(LOG, "R12_nongenetic.json"), "w"), indent=2)
print("✓ R12 complete.")
