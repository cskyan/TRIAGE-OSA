import os
import sys
"""Render author-confirmed benchmark and ablation summaries without inferred uncertainty."""
from pathlib import Path
import sys
import json
import csv

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE=Path('outputs/figures/benchmark')
OUT = HERE / 'figures'
SKILL = Path('plotting/qa')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,
 'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6})
TEAL='#167C80'; GREY='#8B95A1'; BLUE='#2F6B9A'; ORANGE='#C65D3A'
for d in ['main','supplement','panels','source_data','qa']:
    (OUT/d).mkdir(parents=True,exist_ok=True)
data = json.loads(Path('plotting/data/benchmark_ablation_data.json').read_text(encoding='utf-8'))
bench = data['benchmark']['rows']
for row in bench:
    if str(row[0]).strip() == 'TRIAGE-OSA':
        row[0] = 'TRIAGE-OSA (backbone)'
hier = data['structural_hierarchy']['rows']
abl = data['osa_deletion']['rows']
drift = data['gate_drift']['rows']
tables=[('benchmark',['Method','AUROC','AUPRC','Balanced accuracy','Precision','Recall','Specificity','F1','MCC'],bench),
 ('structural_hierarchy',['Structural configuration','Spearman','AUPRC','Top-5% enrichment','Utility'],hier),
 ('osa_deletion',['Variant','Spearman','AUPRC','Top-5% enrichment','Utility','Structural mass'],abl),
 ('gate_drift',['Variant','Gate drift'],drift)]
for name,head,rows in tables:
    with (OUT/'source_data'/f'{name}.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(head);writer.writerows(rows)

def style(ax):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#E7EBED',lw=.5)
    ax.tick_params(length=2.5,width=.5)

def panel(ax,letter,title):
    ax.set_title(title,loc='left',pad=13,fontweight='normal')
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-13,13),
                textcoords='offset points',weight='bold',size=9,ha='right',va='baseline')

def save(fig,name,folder='main'):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(OUT/'qa'/f'{name}.alignment.json'),strict=True)
    fig.savefig(OUT/folder/f'{name}.pdf')
    fig.savefig(OUT/folder/f'{name}.svg')
    fig.savefig(OUT/folder/f'{name}.png',dpi=600)
    fig.savefig(OUT/folder/f'{name}.tiff',dpi=600, pil_kwargs={'compression':'tiff_lzw'})

def benchmark_panel(ax,k,labels=True):
    vals=np.array([float(row[k]) if isinstance(row[k],(int,float)) else np.nan for row in bench])
    y=np.arange(len(vals)); cols=[GREY]*15+[TEAL]
    for yy,v,c in zip(y,vals,cols):
        if np.isfinite(v):
            ax.plot(v,yy,'o',ms=4.6 if c==TEAL else 3.6,color=c,zorder=3)
            ax.annotate(f'{v:.2f}',(v,yy),xytext=(5,0),textcoords='offset points',va='center',fontsize=7,color=c)
    ax.set_yticks(y, [r[0] for r in bench] if labels else ['']*len(y))
    ax.set_ylim(15.7,-.7);ax.set_xlim(0,1)
    ax.set_xticks([0,.25,.5,.75,1]);style(ax);ax.grid(False)
    if k in (1,2): ax.axvline(.5,color='#D1D8DB',ls=':',lw=.7,zorder=0)
    ax.set_xlabel({1:'AUROC',2:'AUPRC',8:'MCC'}[k])
    if k==8: ax.text(.035,1,'Not reported',va='center',size=7,color=GREY)

f,axs=plt.subplots(1,3,figsize=(183/25.4,112/25.4),gridspec_kw={'wspace':.24})
f.subplots_adjust(left=.23,right=.985,bottom=.14,top=.89)
for a,k,l,t in zip(axs,[1,2,8],'abc',['Ranking discrimination','Positive-pair retrieval','Decision balance']):
    benchmark_panel(a,k,a is axs[0]);panel(a,l,t)
save(f,'Fig3_Bernett_benchmark');plt.close(f)
if '--benchmark-only' in sys.argv:
    raise SystemExit(0)
for k,l in zip([1,2,8],'abc'):
    f,a=plt.subplots(figsize=(90/25.4,112/25.4));f.subplots_adjust(left=.36,right=.96,top=.94,bottom=.12)
    benchmark_panel(a,k);save(f,f'Fig3{l}',folder='panels');plt.close(f)

short=['Full','No participant context','Shuffled participant context','No OSA context',
 'No structural-mass constraint','No missingness / coverage','No gate delta','No gate-drift regularization',
 'No residual','Static fusion','Equal fusion']

def hierarchy(ax):
    vals=np.array([r[4] for r in hier]);y=np.arange(len(vals))
    names=['L1 + L2 + L3','L1 + L3','L3 only','L1 + L2','L1 only']
    for i,v in enumerate(vals):
        ax.plot(v,i,'o',color=TEAL if i==0 else BLUE,ms=5)
        ax.text(v+.0017,i,f'{v:.4f}',va='center',size=7)
    ax.set_yticks(y,names);ax.set_ylim(4.6,-.6);ax.set_xlim(.625,.695)
    ax.set_xlabel('Composite discovery utility');style(ax);ax.grid(False)

def deletion(ax):
    delta=np.array([r[4]-abl[0][4] for r in abl]);y=np.arange(len(delta))
    for i,v in enumerate(delta):
        ax.hlines(i,v,0,color='#D3DBDF',lw=1)
        ax.plot(v,i,'o',color=TEAL if i==0 else ORANGE,ms=4)
    ax.axvline(0,color=GREY,lw=.7,ls=':')
    ax.set_yticks(y,short);ax.set_ylim(10.7,-.7);ax.set_xlim(-.57,.045)
    ax.set_xlabel('Utility difference from full');style(ax)

def mass(ax):
    values=[r[5] for r in abl];y=np.arange(len(values))
    for i,v in enumerate(values): ax.plot(v,i,'o',ms=4,color=TEAL if i==0 else (ORANGE if i==4 else GREY))
    ax.axvline(.1,color=BLUE,ls=':',lw=.9)
    ax.set_yticks(y,['']*len(y));ax.set_ylim(10.7,-.7);ax.set_xlim(0,.72)
    ax.set_xlabel('Combined molecular weight');style(ax)


def strong(ax):
    import csv
    p=Path('plotting/data/strong_controls.tsv')
    rows={r['method']:r for r in csv.DictReader(p.open(),delimiter='\t')}
    keys=['full','static_constrained_converged','context_anchor_direct']
    names=['Full','Converged constrained static','Direct context anchor']
    for i,k in enumerate(keys):
        v=float(rows[k]['multi_axis_utility']); ax.plot(v,i,'o',ms=5,color=TEAL if i==0 else GREY)
        ax.annotate(f'{v:.4f}',(v,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=7)
    ax.set_yticks(range(3),names);ax.set_ylim(2.6,-.6);ax.set_xlim(.645,.690)
    ax.set_xlabel('Composite discovery utility');style(ax);ax.grid(False)

f=plt.figure(figsize=(183/25.4,190/25.4))
gs=f.add_gridspec(3,2,left=.30,right=.98,bottom=.075,top=.94,hspace=.65,wspace=.38,height_ratios=[.7,1.4,.65])
a=f.add_subplot(gs[0,:]);hierarchy(a);panel(a,'a','Molecular configurations with context retained')
b=f.add_subplot(gs[1,0]);deletion(b);panel(b,'b','OSA-specific component deletion')
c=f.add_subplot(gs[1,1]);mass(c);panel(c,'c','Molecular retention')
d=f.add_subplot(gs[2,:]); strong(d); panel(d,'d','Converged controls delimit the adaptive gain')
save(f,'Fig4_Two_level_ablation');plt.close(f)
for fun,l in [(hierarchy,'a'),(deletion,'b'),(mass,'c'),(strong,'d')]:
    f,a=plt.subplots(figsize=(120/25.4,95/25.4));f.subplots_adjust(left=.48 if l=='b' else (.40 if l=='d' else .25),right=.97,top=.93,bottom=.17)
    fun(a);save(f,f'Fig4{l}',folder='panels');plt.close(f)

(OUT/'source_data'/'benchmark_ablation_data.json').write_text(json.dumps({n:{'headers':h,'rows':r} for n,h,r in tables},indent=2,ensure_ascii=False),encoding='utf8')
(OUT/'qa'/'data_contract.md').write_text('''# Figure source contract
Author-confirmed completed experiments. Intra1 training, Intra0 validation, Intra2 testing.
All 16 benchmark rows, all five hierarchy rows and all 11 deletion rows retained.
Eight benchmark metrics and all ablation variables retained in supplementary tables.
Plots select AUROC/AUPRC/MCC for benchmark and utility/structural mass for ablation.
No per-run values supplied for newly added rows. No error bars, inferred tests, ROC/PR curves or fabricated observations.
Summary differences are descriptive and are not paired statistical tests.
Undefined MCC and N/A gate drift remain missing, never zero.
Style-only inheritance from existing OSA figures: teal full model, blue structural evidence, muted orange deletion, grey baseline.
''',encoding='utf8')
print('Rendered Fig3 and Fig4; seven individual panels; four source tables.')
