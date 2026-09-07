#!/usr/bin/env python3
"""R04 — supplementary tables + supporting figures for the revision, and a
consolidated results JSON used when writing the manuscript and rebuttal."""
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import seaborn as sns
PROJ = os.environ.get("PPMI_NOMOGRAM_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # set PPMI_NOMOGRAM_DIR to your project root
T=os.path.join(PROJ,"revision","tables"); F=os.path.join(PROJ,"revision","figures")
os.makedirs(T,exist_ok=True)
v=json.load(open(os.path.join(PROJ,"revision","logs","R01_validation.json")))
r=json.load(open(os.path.join(PROJ,"revision","logs","R02_reviewer.json")))

# ── Supplementary Table: complete-case HR comparison (R6) ────────────────────
cc=pd.DataFrame(v['complete_case']['hr_comparison'])
cc['HR_primary']=cc['HR_primary'].round(2); cc['HR_completecase']=cc['HR_completecase'].round(2)
cc['p_completecase']=cc['p_completecase'].apply(lambda x: '<0.001' if x<0.001 else f'{x:.3f}')
cc.to_csv(os.path.join(T,"SuppTable_completecase.csv"),index=False)

# ── Supplementary Table: VIF (R7) ────────────────────────────────────────────
pd.DataFrame({'Variable':list(v['vif']),'VIF':[round(x,2) for x in v['vif'].values()]}
            ).to_csv(os.path.join(T,"SuppTable_VIF.csv"),index=False)

# ── Supplementary Table: GBA variant groups (R4) ─────────────────────────────
g=r['R4_gba']; rows=[]
for rec in g['event_rate_by_group']:
    rows.append({'GBA group':rec['GBA_group'],'N':rec['n'],'Events':rec['events'],'Event %':rec['event_pct']})
gba=pd.DataFrame(rows)
gba.to_csv(os.path.join(T,"SuppTable_GBA_variants.csv"),index=False)

# ── Supplementary Table: age-stratified discrimination (R9) ──────────────────
pd.DataFrame(r['R9']).rename(columns={'bin':'Age group','n':'N','events':'Events','c_index':'C-index'}
   ).round({'C-index':3}).to_csv(os.path.join(T,"SuppTable_age_stratified.csv"),index=False)

# ── Supplementary Table: N at risk / attrition per year (longitudinal rule) ──
full=pd.read_csv(os.path.join(PROJ,"data","analytical_dataset.csv")).dropna(subset=['time_years'])
ar=r['R8']['at_risk_by_year']
att=[]
for y in range(1,13):
    atrisk=ar[str(y)]; cum_ev=int(((full['event']==1)&(full['time_years']<=y)).sum())
    att.append({'Year':y,'N at risk':atrisk,'Cumulative events':cum_ev,
                'Cumulative attrition %':round(100*(1152-atrisk)/1152,1)})
pd.DataFrame(att).to_csv(os.path.join(T,"SuppTable_at_risk.csv"),index=False)

# ── Supplementary Figure: GBA variant-severity dose-response (adjusted HR) ────
def pfmt(p): return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
d={x['GBA_group']:x for x in g['event_rate_by_group']}
groups=[('Non-carrier\n(reference)',1.0,None,None,d['none']['n'],d['none']['event_pct']),
        ('Mild\n(N409S)',g['HR_GBA_mild'],g['HR_GBA_mild_ci'],g['p_GBA_mild'],d['mild']['n'],d['mild']['event_pct']),
        ('Severe\n(L444P / null / splice)',g['HR_GBA_severe'],g['HR_GBA_severe_ci'],g['p_GBA_severe'],d['severe']['n'],d['severe']['event_pct'])]
cols=['#4575b4','#fdae61','#d73027']
fig,ax=plt.subplots(figsize=(6.2,3.8))
for i,(lab,hr,ci,p,nn,ev) in enumerate(groups):
    if ci: ax.plot([ci[0],ci[1]],[i,i],'-',color=cols[i],lw=2.2)
    ax.plot(hr,i,'o',ms=12,color=cols[i],zorder=5)
    txt=f"HR {hr:.2f}" if ci is None else f"HR {hr:.2f} ({ci[0]:.2f}–{ci[1]:.2f}), {pfmt(p)}"
    ax.text(3.15,i,txt,va='center',ha='left',fontsize=8.5,family='Arial')
    ax.text(3.15,i-0.28,f"n={nn}, {ev:.0f}% impaired",va='center',ha='left',fontsize=7.5,color='dimgray',family='Arial')
ax.axvline(1,color='grey',ls='--',lw=1)
ax.set_yticks(range(3)); ax.set_yticklabels([grp[0] for grp in groups],fontsize=9,family='Arial')
ax.set_xlabel('Adjusted hazard ratio for cognitive impairment (95% CI)',fontsize=10,family='Arial')
ax.set_xlim(0.5,3.1); ax.set_ylim(-0.6,2.6)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(F,"SuppFig_GBA_doseresponse.png"),dpi=285,bbox_inches='tight')
fig.savefig(os.path.join(F,"SuppFig_GBA_doseresponse.pdf"),bbox_inches='tight'); plt.close()

# ── Supplementary Figure: age-stratified C-index (lollipop) ──────────────────
fig,ax=plt.subplots(figsize=(5.4,3.6))
ab=r['R9']; ys=[a['c_index'] for a in ab]; labs=[f"{a['bin']}\n(n={a['n']}, {a['events']} ev)" for a in ab]
pal=sns.color_palette("muted",3)
for i,(yv,cc2) in enumerate(zip(ys,pal)):
    ax.plot([i,i],[0.5,yv],'-',color=cc2,lw=2); ax.plot(i,yv,'o',ms=13,color=cc2,zorder=5)
    ax.text(i,yv+0.012,f'{yv:.3f}',ha='center',fontsize=9.5,family='Arial')
ax.axhline(0.717,color='#d62728',ls='--',lw=1.1)
ax.text(2.4,0.720,'overall 0.72',ha='right',va='bottom',fontsize=8,color='#d62728',family='Arial')
ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=8.5,family='Arial')
ax.set_ylabel('C-index',fontsize=11,family='Arial'); ax.set_ylim(0.5,0.82)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(F,"SuppFig_age_stratified.png"),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(F,"SuppFig_age_stratified.pdf"),bbox_inches='tight'); plt.close()

# ── consolidated results ─────────────────────────────────────────────────────
json.dump({'validation':v,'reviewer':r},
          open(os.path.join(PROJ,"revision","logs","revision_results.json"),'w'),indent=2)
from PIL import Image
for fn in sorted(os.listdir(F)):
    if fn.startswith('SuppFig') and fn.endswith('.png'):
        print(fn, Image.open(os.path.join(F,fn)).size)
print("R04 complete — supp tables + figures written.")
print("Complete-case HR comparison:\n", cc.to_string(index=False))
