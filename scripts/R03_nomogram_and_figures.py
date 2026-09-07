#!/usr/bin/env python3
"""
R03_nomogram_and_figures.py — Revision
Rebuilds Figure 5 (nomogram) with a REAL, model-derived Total Points -> risk axis:
points are additive across predictors on a common scale, total points map to the
centered Cox linear predictor, and risk = 1 - S0(t)^exp(lp) at t = 3, 5, 8 years.
Also regenerates Figure 3 (time-dependent AUC) using IPCW estimates with at-risk N.
"""
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lifelines import CoxPHFitter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
FIG=os.path.join(PROJ,"revision","figures"); os.makedirs(FIG,exist_ok=True)
art=json.load(open(os.path.join(PROJ,"revision","data","primary_artifacts.json"))); selected=art['selected']

# Reconstruct primary Cox model (identical to primary) -------------------------
df=pd.read_csv(os.path.join(PROJ,"data","analytical_dataset.csv")).dropna(subset=['time_years']).reset_index(drop=True)
df['SAA_positive']=(df['SAA_Status_Combined'].astype(str).str.lower().isin(['positive','1','1.0','pos'])).astype(float)
df['HVLT_total_learning']=df[['HVLTRT1','HVLTRT2','HVLTRT3']].sum(axis=1,min_count=2)
Ximp=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(df[art['pred_vars']]),columns=art['pred_vars'])
mdf=Ximp[selected].copy(); mdf['time_years']=df['time_years'].values; mdf['event']=df['event'].values.astype(int)
cph=CoxPHFitter(penalizer=0.01).fit(mdf,'time_years','event')
coef=cph.params_                      # per-variable coefficients
means=mdf[selected].mean()            # lifelines centers baseline at means
S0=cph.baseline_survival_.iloc[:,0]   # baseline survival (at covariate means)
def S0_at(t):
    idx=S0.index;
    return float(np.interp(t, idx, S0.values))

# Nomogram point system (common additive scale) --------------------------------
lo={v:float(mdf[v].quantile(0.01)) for v in selected}
hi={v:float(mdf[v].quantile(0.99)) for v in selected}
binvars=['SEX','GBA_carrier','LRRK2_carrier','PIGD_dominant']
for v in binvars: lo[v],hi[v]=0.0,1.0
rng={v:coef[v]*(hi[v]-lo[v]) for v in selected}           # signed lp range
scale=100.0/max(abs(r) for r in rng.values())             # largest |range| -> 100 pts
ref={v:(lo[v] if coef[v]>0 else hi[v]) for v in selected} # 0-point end
pts=lambda v,x: coef[v]*(x-ref[v])*scale                  # >=0 points
raw_lp_min=sum(coef[v]*ref[v] for v in selected)
offset=raw_lp_min-float((coef*means).sum())               # lp_centered = T/scale + offset
maxT=sum(abs(rng[v])*scale for v in selected)

def total_to_risk(T,t):
    lp=T/scale+offset
    return 1-S0_at(t)**np.exp(lp)
def risk_to_total(r,t):
    s=S0_at(t)
    lp=np.log(np.log(1-r)/np.log(s))
    return (lp-offset)*scale

name_map={'AGE_AT_VISIT':'Age (years)','SEX':'Sex (1=Male)','EDUCYRS':'Education (years)',
 'GBA_carrier':'GBA carrier','LRRK2_carrier':'LRRK2 carrier','NP2PTOT':'MDS-UPDRS II',
 'NP1RTOT':'MDS-UPDRS I','HVLT_total_learning':'HVLT-R total learning','LNS_TOTRAW':'Letter-Number Seq.',
 'SDMTOTAL':'Symbol Digit Modalities','DVT_SFTANIM':'Semantic fluency (animals)',
 'NP1COG':'Cognitive complaints (1.1)','NP1HALL':'Hallucinations (1.2)','NP1APAT':'Apathy (1.5)',
 'RBDSQ_TOTAL':'RBD questionnaire','SCAU_TOTAL':'SCOPA-AUT total','MSEADLG':'Schwab-England ADL',
 'PIGD_dominant':'PIGD-dominant'}
order=sorted(selected,key=lambda v:abs(rng[v]),reverse=True)

# Draw -------------------------------------------------------------------------
# Layout: top panel = Points ruler + predictor rows (tight, so rows fill width);
# bottom panel = Total-points -> 3/5/8-year risk conversion table (unscaled).
from matplotlib.gridspec import GridSpec
nrow=len(order)
tbl_totals=list(range(80,340,40))               # representative total-point rows
fig_h=(nrow+2)*0.52 + (len(tbl_totals)+2)*0.30 + 0.6
fig=plt.figure(figsize=(8.6,fig_h))
gs=GridSpec(2,1,height_ratios=[(nrow+2)*0.52, (len(tbl_totals)+2)*0.30+0.6], hspace=0.08)
ax=fig.add_subplot(gs[0]); axt=fig.add_subplot(gs[1]); axt.axis('off')
ypos=np.arange(nrow+1,0,-1); bh=0.17
colors=sns.color_palette("muted",nrow)
def hline(y,x0,x1,c='k',lw=1.8): ax.plot([x0,x1],[y,y],'-',color=c,lw=lw)
def tick(x,y,c='k',lw=1.3,h=bh): ax.plot([x,x],[y-h,y+h],'-',color=c,lw=lw)
def thin(pairs,min_gap):
    kept=[]; last=-1e9
    for p in sorted(pairs,key=lambda z:z[0]):
        if p[0]-last>=min_gap: kept.append(p); last=p[0]
    return kept

# Points ruler (0-100)
y0=ypos[0]; hline(y0,0,100,lw=2)
for p in range(0,101,10):
    tick(p,y0,lw=1.4); ax.text(p,y0+0.30,str(p),ha='center',va='bottom',fontsize=8.5,family='Arial')
ax.text(-16,y0,'Points',ha='right',va='center',fontsize=10.5,fontweight='bold',family='Arial')

# Predictor rows — adaptive tick density + outward endpoint labels (no overlap)
for i,v in enumerate(order):
    y=ypos[i+1]; c=colors[i]
    if v in binvars:
        vals=[0,1]; labs={'SEX':['F','M']}.get(v,['No','Yes'])
    else:
        vals=list(np.linspace(lo[v],hi[v],6))
        labs=[f'{x:.0f}' if abs(x)>=1 else f'{x:.1f}' for x in vals]
    P=[pts(v,x) for x in vals]
    p0,p1=min(P),max(P); hline(y,p0,p1,c=c,lw=2.4)
    for px in P: tick(px,y,c=c)
    li=int(np.argmin(P)); ri=int(np.argmax(P))
    ax.text(p0-1.3,y,labs[li],ha='right',va='center',fontsize=8,family='Arial')
    ax.text(p1+1.3,y,labs[ri],ha='left',va='center',fontsize=8,family='Arial')
    interior=[(px,lb) for px,lb in zip(P,labs) if px not in (p0,p1)]
    for px,lb in thin(interior,11):
        if px-p0>=11 and p1-px>=11:
            ax.text(px,y-0.32,lb,ha='center',va='top',fontsize=7.5,family='Arial')
    ax.text(-16,y,name_map.get(v,v),ha='right',va='center',fontsize=9.5,family='Arial')
ax.set_xlim(-58,104); ax.set_ylim(ypos[-1]-0.8,y0+1.2); ax.axis('off')

# Conversion table (bottom panel) — Total points -> 3/5/8-year risk
colx=[0.16,0.45,0.635,0.82]; headers=['Total points','3-year risk','5-year risk','8-year risk']
xL,xR=0.02,0.97
nrows=len(tbl_totals); ytop=0.92; ybot=0.06; rh=(ytop-0.14-ybot)/nrows
yhead=ytop
for cx,h in zip(colx,headers):
    axt.text(cx,yhead,h,ha='center',va='center',fontsize=9.5,fontweight='bold',family='Arial')
axt.plot([xL,xR],[yhead+0.06]*2,'k-',lw=1.4)            # three-line: top
axt.plot([xL,xR],[yhead-0.06]*2,'k-',lw=1.0)            # under header
for i,T in enumerate(tbl_totals):
    yr=yhead-0.12-(i+0.5)*rh
    axt.text(colx[0],yr,str(T),ha='center',va='center',fontsize=9,family='Arial')
    for j,t in enumerate([3,5,8]):
        axt.text(colx[j+1],yr,f"{total_to_risk(T,t)*100:.0f}%",ha='center',va='center',fontsize=9,family='Arial')
axt.plot([xL,xR],[yhead-0.12-nrows*rh]*2,'k-',lw=1.4)   # bottom
axt.set_xlim(0,1); axt.set_ylim(0,1)

dpi_png=int(1790/max(8.6,fig_h))
fig.savefig(os.path.join(FIG,"nomogram.png"),dpi=min(300,dpi_png),bbox_inches='tight')
fig.savefig(os.path.join(FIG,"nomogram.pdf"),bbox_inches='tight'); plt.close()
print(f"Saved nomogram+table (PNG dpi={min(300,dpi_png)}, height {fig_h:.1f}in). maxT={maxT:.0f}")
print("Table totals:",tbl_totals)
print("Check 5-yr risk mapping: total=0 ->",f"{total_to_risk(0,5):.1%}",
      "| total=maxT ->",f"{total_to_risk(maxT,5):.1%}")
# sanity: median patient total points and risk
raw_lp=(mdf[selected]*coef).sum(axis=1); T_all=(raw_lp-raw_lp_min)*scale
print(f"Median total points={T_all.median():.0f} -> 5yr risk {total_to_risk(T_all.median(),5):.1%}")

# Figure 3 — IPCW time-dependent AUC -------------------------------------------
r8=json.load(open(os.path.join(PROJ,"revision","logs","R02_reviewer.json")))['R8']
yrs=[int(k) for k in r8['auc_by_year']]; aucs=[r8['auc_by_year'][str(y)] for y in yrs]
atrisk=[r8['at_risk_by_year'][str(y)] for y in yrs]
fig,ax=plt.subplots(figsize=(7.2,4.6))
solid=[y for y in yrs if atrisk[yrs.index(y)]>=100]; cut=max(solid)
ax.plot([y for y in yrs if y<=cut],[a for y,a in zip(yrs,aucs) if y<=cut],'-o',color='#2171b5',lw=2,ms=6,zorder=5)
ax.plot([y for y in yrs if y>=cut],[a for y,a in zip(yrs,aucs) if y>=cut],'--o',color='#9ecae1',lw=1.8,ms=5,zorder=5)
ax.axhline(0.5,color='grey',ls=':',lw=1)
ax.axhline(r8['mean_auc'],color='#d62728',ls='--',lw=1.2)
ax.text(yrs[-1],r8['mean_auc']+0.008,f"mean {r8['mean_auc']:.2f}",ha='right',va='bottom',fontsize=9,color='#d62728',family='Arial')
ax.set_xlabel('Years from baseline',fontsize=11,family='Arial')
ax.set_ylabel('IPCW time-dependent AUC',fontsize=11,family='Arial')
ax.set_ylim(0.5,0.95); ax.set_xticks(yrs); ax.tick_params(labelsize=9)
for y,a,n in zip(yrs,aucs,atrisk):
    ax.text(y,0.515,str(n),ha='center',va='bottom',fontsize=6.5,color='dimgray',rotation=90,family='Arial')
ax.text(yrs[0],0.505,'n at risk:',ha='left',va='bottom',fontsize=6.5,color='dimgray',family='Arial')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIG,"time_dependent_auc.png"),dpi=240,bbox_inches='tight')
fig.savefig(os.path.join(FIG,"time_dependent_auc.pdf"),bbox_inches='tight'); plt.close()
print("Saved time_dependent_auc (dashed beyond year %d where n<100)."%cut)
