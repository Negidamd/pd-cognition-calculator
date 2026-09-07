#!/usr/bin/env python3
"""R19 — Regenerate Figure 2 (forest plot) with GBA1 nomenclature (2026-09-07).

Rebuilt from revision/data/primary_model_df.csv rather than the original pipeline, which
needs the raw PPMI extract. All 18 hazard ratios, confidence intervals and p-values were
verified identical to the published figure before regeneration; only the GBA label and the
axis scale change. Hazard ratios are plotted on a log axis so that protective and harmful
effects of equal magnitude are equidistant from the null (the previous linear axis
compressed the LRRK2 estimate).
"""
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
FIG, RFIG = os.path.join(PROJ, "figures"), os.path.join(PROJ, "revision", "figures")
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.linewidth": 0.9})

mdf = pd.read_csv(os.path.join(PROJ, "revision", "data", "primary_model_df.csv"))
sel = json.load(open(os.path.join(PROJ, "revision", "data", "primary_artifacts.json")))["selected"]
cph = CoxPHFitter(penalizer=0.01).fit(mdf[sel + ["time_years", "event"]], "time_years", "event")
assert abs(cph.concordance_index_ - 0.7337494243016183) < 1e-9

LAB = {"GBA_carrier": "GBA1 mutation carrier", "NP1HALL": "Hallucinations (1.2)",
       "NP1COG": "Cognitive complaints (1.1)", "PIGD_dominant": "PIGD-dominant subtype",
       "SEX": "Sex (1=Male)", "NP1APAT": "Apathy (1.5)", "RBDSQ_TOTAL": "RBD Screening Questionnaire",
       "AGE_AT_VISIT": "Age (years)", "EDUCYRS": "Education (years)", "SCAU_TOTAL": "SCOPA-AUT total",
       "NP1RTOT": "MDS-UPDRS Part I", "NP2PTOT": "MDS-UPDRS Part II",
       "DVT_SFTANIM": "Semantic fluency (animals)", "MSEADLG": "Modified Schwab-England ADL",
       "SDMTOTAL": "Symbol Digit Modalities", "HVLT_total_learning": "HVLT-R total learning",
       "LNS_TOTRAW": "Letter-Number Sequencing", "LRRK2_carrier": "LRRK2 mutation carrier"}
S = cph.summary.loc[sel]
S = S.assign(hr=S["exp(coef)"], lo=S["exp(coef) lower 95%"], hi=S["exp(coef) upper 95%"], pv=S["p"])
S = S.sort_values("hr", ascending=False)

DARK, LIGHT = "#2c5f9e", "#a8c6e5"
fig, ax = plt.subplots(figsize=(6.0, 4.9))
y = np.arange(len(S))[::-1]
for yi, (v, r) in zip(y, S.iterrows()):
    c = DARK if r.pv < 0.05 else LIGHT
    ax.plot([r.lo, r.hi], [yi, yi], "-", color=c, lw=1.9, solid_capstyle="butt")
    ax.plot(r.hr, yi, "o", ms=5.6, color=c, zorder=5)
    ps = "p < 0.001" if r.pv < 0.001 else f"p = {r.pv:.3f}"
    ax.text(3.55, yi, f"{r.hr:.2f} ({r.lo:.2f}\u2013{r.hi:.2f})  {ps}", va="center", ha="left", fontsize=7.6)
ax.axvline(1, color="grey", ls="--", lw=1)
ax.set_yticks(y); ax.set_yticklabels([LAB[v] for v in S.index], fontsize=8.2)
ax.set_ylim(-0.8, len(S) - 0.2)
ax.set_xscale("log"); ax.set_xlim(0.5, 3.2)
ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
ax.set_xticklabels(["0.50", "0.75", "1.00", "1.50", "2.00", "3.00"], fontsize=9)
ax.minorticks_off()
ax.set_xlabel("Hazard ratio for cognitive impairment (95% CI, log scale)", fontsize=9.5)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"forest_plot.{ext}"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RFIG, f"forest_plot.{ext}"), dpi=300, bbox_inches="tight")
plt.close()
print("rows:", len(S), "| significant:", int((S.pv < 0.05).sum()),
      "| HR range %.2f-%.2f" % (S.hr.min(), S.hr.max()),
      "| widest CI %.2f-%.2f" % (S.lo.min(), S.hi.max()))
