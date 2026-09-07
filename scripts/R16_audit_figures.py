
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import cumulative_dynamic_auc

PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
FIG=os.path.join(PROJ,"figures"); RFIG=os.path.join(PROJ,"revision","figures"); LOG=os.path.join(PROJ,"revision","logs")
plt.rcParams.update({"font.family":"Arial","font.size":10,"axes.linewidth":0.9})

mdf=pd.read_csv(os.path.join(PROJ,"revision","data","primary_model_df.csv"))
art=json.load(open(os.path.join(PROJ,"revision","data","primary_artifacts.json"))); sel=art["selected"]
cph=CoxPHFitter(penalizer=0.01).fit(mdf[sel+["time_years","event"]],"time_years","event")
lp=cph.predict_log_partial_hazard(mdf[sel]).values
assert abs(cph.concordance_index_-0.7337494243016183)<1e-9, cph.concordance_index_

# ── Figure 3: IPCW time-dependent AUC with bootstrap pointwise 95% CI ─────────
y=np.array([(bool(e),t) for e,t in zip(mdf.event,mdf.time_years)],dtype=[("e",bool),("t",float)])
years=np.arange(1,13,dtype=float)
auc,mean_auc=cumulative_dynamic_auc(y,y,lp,years)
ref=json.load(open(os.path.join(LOG,"R02_reviewer.json")))["R8"]
assert max(abs(a-ref["auc_by_year"][str(int(t))]) for a,t in zip(auc,years))<1e-12
rng=np.random.default_rng(42); B=500; boot=np.full((B,12),np.nan)
for b in range(B):
    idx=rng.integers(0,len(mdf),len(mdf))
    try: boot[b],_=cumulative_dynamic_auc(y[idx],y[idx],lp[idx],years)
    except Exception: pass
lo=np.nanpercentile(boot,2.5,axis=0); hi=np.nanpercentile(boot,97.5,axis=0)
atr=[ref["at_risk_by_year"][str(int(t))] for t in years]

fig,ax=plt.subplots(figsize=(6.0,4.0))
solid=years<=11
ax.fill_between(years,lo,hi,color="#4575b4",alpha=0.18,lw=0)
ax.plot(years[solid],auc[solid],"-o",color="#2c5f9e",lw=2.0,ms=5.5)
ax.plot(years[years>=11],auc[years>=11],"--o",color="#2c5f9e",lw=1.6,ms=5.0,alpha=0.55,mfc="white")
ax.axhline(mean_auc,color="#d73027",ls="--",lw=1.1)
ax.text(6.5,mean_auc+0.010,f"mean {mean_auc:.2f} (years 1\u201312)",color="#d73027",fontsize=8.2,va="bottom",ha="center")
for t,a,n in zip(years,auc,atr):
    ax.annotate(str(n),(t,0.545),ha="center",va="bottom",fontsize=6.8,color="dimgray",rotation=90)
ax.text(0.55,0.525,"n at risk:",fontsize=7.2,color="dimgray",ha="left",va="bottom")
ax.set_xlabel("Years from baseline"); ax.set_ylabel("IPCW time-dependent AUC (95% CI)")
ax.set_xticks(range(1,13)); ax.set_xlim(0.4,12.6); ax.set_ylim(0.50,0.95)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(os.path.join(FIG,f"time_dependent_auc.{ext}"),dpi=300,bbox_inches="tight")
    fig.savefig(os.path.join(RFIG,f"time_dependent_auc.{ext}"),dpi=300,bbox_inches="tight")
plt.close()
ci_w=hi-lo
print("Fig3 CI width yr1-5 %.3f-%.3f | yr12 %.3f"%(ci_w[:5].min(),ci_w[:5].max(),ci_w[11]))

# ── Figure 4: KM by risk tertile + numbers at risk, key below axes ───────────
mdf["tert"]=pd.qcut(lp,3,labels=["Low risk","Intermediate risk","High risk"])
cols={"Low risk":"#1b7837","Intermediate risk":"#e08214","High risk":"#c0392b"}
fig,(ax,axr)=plt.subplots(2,1,figsize=(6.2,4.9),gridspec_kw={"height_ratios":[4.0,1.0],"hspace":0.08})
ticks=list(range(0,15,2)); km5={}
for g in ["Low risk","Intermediate risk","High risk"]:
    s=mdf[mdf.tert==g]; km=KaplanMeierFitter().fit(s.time_years,s.event,label=g)
    km.plot_survival_function(ax=ax,color=cols[g],lw=2.0,ci_alpha=0.15,legend=False)
    km5[g]=float(km.predict(5.0))
    axr.text(-1.35,{"Low risk":2,"Intermediate risk":1,"High risk":0}[g],g,ha="right",va="center",
             fontsize=8.2,color=cols[g])
    for t in ticks:
        axr.text(t,{"Low risk":2,"Intermediate risk":1,"High risk":0}[g],
                 str(int((s.time_years>=t).sum())),ha="center",va="center",fontsize=7.6,color="black")
lr=multivariate_logrank_test(mdf.time_years,mdf.tert,mdf.event)
ax.text(0.98,0.98,"Log-rank p < 0.001" if lr.p_value<0.001 else f"Log-rank p = {lr.p_value:.3f}",
        transform=ax.transAxes,ha="right",va="top",fontsize=9)
ax.axvline(5,color="grey",ls=":",lw=0.9)
for g,dy in [("Low risk",0.035),("Intermediate risk",0.035),("High risk",0.035)]:
    ax.annotate(f"{km5[g]*100:.0f}%",(5,km5[g]),xytext=(5.35,km5[g]+dy),fontsize=8.0,color=cols[g])
ax.set_ylabel("Cognitive impairment-free survival"); ax.set_xlabel("")
ax.set_xlim(-0.3,14.5); ax.set_ylim(0,1.02); ax.set_xticks(ticks); ax.set_xticklabels([])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axr.set_xlim(-0.3,14.5); axr.set_ylim(-0.6,2.6); axr.set_yticks([]); axr.set_xticks(ticks)
axr.set_xlabel("Time from baseline (years)")
axr.text(-1.35,2.75,"Number at risk",ha="right",va="center",fontsize=8.2,fontweight="bold")
for sp in ("top","right","left"): axr.spines[sp].set_visible(False)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(os.path.join(FIG,f"km_risk_groups.{ext}"),dpi=300,bbox_inches="tight")
    fig.savefig(os.path.join(RFIG,f"km_risk_groups.{ext}"),dpi=300,bbox_inches="tight")
plt.close()
print("Fig4 5-yr survival:",{k:round(v*100,1) for k,v in km5.items()},
      "| 5-yr risk ratio high/low %.2f"%((1-km5['High risk'])/(1-km5['Low risk'])))

# ── Supp Fig 4: GBA1 dose-response, un-clipped CI, log HR axis ───────────────
g=ref=json.load(open(os.path.join(LOG,"R02_reviewer.json")))["R4_gba"]
d={x["GBA_group"]:x for x in g["event_rate_by_group"]}
pf=lambda p:"p < 0.001" if p<0.001 else f"p = {p:.3f}"
groups=[("Non-carrier or genotype\nimputed (reference)",1.0,None,None,d["none"]["n"],d["none"]["event_pct"]),
        ("Mild\n(N409S/N370S, other\nnon-severe missense)",g["HR_GBA_mild"],g["HR_GBA_mild_ci"],g["p_GBA_mild"],d["mild"]["n"],d["mild"]["event_pct"]),
        ("Severe\n(L444P/L483P, null,\nframeshift, splice)",g["HR_GBA_severe"],g["HR_GBA_severe_ci"],g["p_GBA_severe"],d["severe"]["n"],d["severe"]["event_pct"])]
cols3=["#4575b4","#fdae61","#d73027"]
fig,ax=plt.subplots(figsize=(6.4,3.6))
for i,(lab,hr,ci,p,nn,ev) in enumerate(groups):
    if ci: ax.plot([ci[0],ci[1]],[i,i],"-",color=cols3[i],lw=2.2)
    ax.plot(hr,i,"o",ms=11,color=cols3[i],zorder=5)
    txt=f"HR {hr:.2f}" if ci is None else f"HR {hr:.2f} ({ci[0]:.2f}\u2013{ci[1]:.2f}), {pf(p)}"
    ax.text(6.1,i+0.13,txt,va="center",ha="left",fontsize=8.2)
    ax.text(6.1,i-0.20,f"n = {nn}, {ev:.0f}% impaired",va="center",ha="left",fontsize=7.4,color="dimgray")
ax.axvline(1,color="grey",ls="--",lw=1)
ax.set_xscale("log"); ax.set_xlim(0.8,5.6)
ax.set_xticks([1,1.5,2,3,4,5]); ax.set_xticklabels(["1.0","1.5","2.0","3.0","4.0","5.0"])
ax.minorticks_off()
ax.set_yticks(range(3)); ax.set_yticklabels([x[0] for x in groups],fontsize=8.2)
ax.set_ylim(-0.6,2.6)
ax.set_xlabel("Adjusted hazard ratio for cognitive impairment (95% CI, log scale)",fontsize=9.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(os.path.join(FIG,f"SuppFig_GBA_doseresponse.{ext}"),dpi=300,bbox_inches="tight")
    fig.savefig(os.path.join(RFIG,f"SuppFig_GBA_doseresponse.{ext}"),dpi=300,bbox_inches="tight")
plt.close()
print("SuppFig4 severe CI upper %.2f now inside xlim 5.6"%g["HR_GBA_severe_ci"][1])
json.dump({"auc":auc.tolist(),"lo":lo.tolist(),"hi":hi.tolist(),"mean_auc":float(mean_auc),
           "ci_width":ci_w.tolist(),"km5":km5,"n_boot":B,"seed":42},
          open(os.path.join(LOG,"R16_audit_figures.json"),"w"),indent=2)
print("done")
