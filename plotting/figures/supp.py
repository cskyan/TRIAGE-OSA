import os
import sys
"""Reproducible figure assembly from existing, immutable OSA results. No training."""
from pathlib import Path
import json
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.transforms import Bbox, ScaledTranslation


ROOT=Path(os.environ.get('OSA_DATA_ROOT', '.'))
BASE = ROOT / 'results/discovery_preprocess_v2'
OUT=Path('outputs/figures/supp')
for d in ['main_figures','supplementary_figures','panels','source_data','qa','documents']:
    (OUT/d).mkdir(parents=True, exist_ok=True)
TEAL, BLUE, CORAL, PURPLE, GREY, DARK = '#147D86','#3A70A0','#C8644B','#8A6DA8','#9AA6B2','#253340'
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
 'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white',
 'text.color':DARK,'axes.labelcolor':DARK,'xtick.color':DARK,'ytick.color':DARK})
W = 183/25.4
records=[]
def read(stage, name, copy=True):
    p=BASE/stage/name
    if copy: shutil.copy2(p, OUT/'source_data'/(stage.split('_')[0]+'_'+name))
    return pd.read_csv(p,sep='\t',low_memory=False)
def js(stage): return json.loads((BASE/stage/'run_summary.json').read_text(encoding='utf-8'))
def tidy(ax):
    ax.set_axisbelow(True); ax.grid(axis='x',color='#E7EDF0',lw=.6)
def label(fig,ax,letter,title):
    ax.set_title(title,loc='left',pad=13,fontweight='bold')
    ax.text(0,1,letter,transform=ax.transAxes+ScaledTranslation(-20/72,13/72,fig.dpi_scale_trans),
            va='bottom',ha='left',fontsize=11,fontweight='bold')
def save(fig,ident,slug,axes,caption,claim,category='main',extras=None):
    extras=extras or {}
    fig.canvas.draw()
    dest=OUT/('main_figures' if category=='main' else 'supplementary_figures')
    stem=ident+'_'+slug
    fig.savefig(dest/(stem+'.pdf'))
    fig.savefig(dest/(stem+'.svg'))
    fig.savefig(dest/(stem+'.png'),dpi=600)
    fig.savefig(dest/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    paneldir=OUT/'panels'/ident; paneldir.mkdir(exist_ok=True)
    for k,ax in enumerate(axes):
        keep=[ax]+extras.get(k,[])
        visibility={a:a.get_visible() for a in fig.axes}
        for a in fig.axes:a.set_visible(a in keep)
        fig.canvas.draw()
        bbox=Bbox.union([a.get_tightbbox(fig.canvas.get_renderer()) for a in keep]).transformed(fig.dpi_scale_trans.inverted()).padded(.07)
        for ext,dpi in [('pdf',None),('svg',None),('png',600),('tiff',600)]:
            kw={'dpi':dpi} if dpi else {}
            if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
            fig.savefig(paneldir/(ident+chr(97+k)+'.'+ext),bbox_inches=bbox,**kw)
        for a,v in visibility.items():a.set_visible(v)
    records.append({'id':ident,'file':str((dest/(stem+'.png')).relative_to(OUT)),
      'caption':caption,'claim':claim,'category':category,'panels':len(axes),'archetype':slug})
    plt.close(fig); print('SAVED',ident,flush=True)
def box(ax,x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.02',
       facecolor=color,edgecolor='none',transform=ax.transAxes))
    ax.text(x+.02,y+h-.035,title,transform=ax.transAxes,va='top',fontsize=9,fontweight='bold')
    ax.text(x+.02,y+h-.115,body,transform=ax.transAxes,va='top',fontsize=7.7,linespacing=1.6)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,transform=ax.transAxes,arrowstyle='-|>',mutation_scale=12,lw=1,color=GREY))

TRAIN='06_model_training/triage_osa_patient_information_v4'
cv=read(TRAIN,'cross_validation_metrics.tsv')
test=cv.loc[cv.partition.eq('test')].copy()
assert not test.duplicated(['outer_fold','model']).any()
assert test.outer_fold.nunique()==4
test.to_csv(OUT/'source_data/Fig2_test_only_metrics.tsv',sep='\t',index=False)
availability=read(TRAIN,'patient_information_availability.tsv')
summary=js(TRAIN)
intervals=read('29_publication_robustness_v1','participant_conditional_intervals.tsv')
read('29_publication_robustness_v1','paired_modality_differences.tsv')
read('29_publication_robustness_v1','pair_readout_rank_sensitivity.tsv')

ST22='22_consolidated_evidence_triage_osa_v4'
rec=read('19_known_ppi_complex_recovery_triage_osa_v4','ranking_recovery_metrics.tsv')
pareto=read(ST22,'pareto_front_candidates.tsv');modules=read(ST22,'mechanism_modules.tsv');proteins=read(ST22,'protein_evidence_matrix.tsv')
gene=proteins.set_index('UniProt').gene_symbol.to_dict()

sel=rec[(rec.reference=='IntAct_all_training_excluded')&np.isclose(rec.fraction,.01)].set_index('model')
# Supplement S1: every internal model, test-only, four different metrics.
allmodels=test.model.drop_duplicates().tolist()
modelnames={'TRIAGE_OSA_specialized':'TRIAGE-OSA','molecular_context_without_patient_information':'Molecular reference','patient_information_only':'Patient info only','frozen_TRIAGE_p_PPI':'Backbone PPI','equal_L1_L2_L3':'Equal L1–L2–L3','context_only_anchor':'Context anchor','L3_only':'L3 readout'}
fig,axs=plt.subplots(2,2,figsize=(W,6.8));fig.subplots_adjust(left=.30,right=.97,top=.94,bottom=.10,wspace=.70,hspace=.43)
for ax,(col,name),letter in zip(axs.flat,[('Spearman','Spearman'),('weak_top20_AUPRC','Weak-label AUPRC'),('top5_enrichment_ratio','Top-5% enrichment'),('multi_axis_utility','Composite discovery utility')],'abcd'):
    for i,m in enumerate(allmodels):
        vals=test.loc[test.model==m,col].to_numpy();ax.scatter(vals,np.arange(len(vals))*.06+i-.09,color=TEAL if m==allmodels[0] else GREY,s=14,alpha=.8)
        ax.errorbar(vals.mean(),i,xerr=vals.std(ddof=1),fmt='|',color=DARK,capsize=2,ms=9)
    ax.set_yticks(range(len(allmodels)),[modelnames.get(m,m.replace('_',' ')) for m in allmodels]);ax.invert_yaxis();ax.set_xlabel(name);tidy(ax);label(fig,ax,letter,name)
save(fig,'FigS1','complete_internal_baselines',list(axs.flat),
 'All recorded internal models using test partitions only. Points are the four outer folds; whiskers are mean ± fold SD. These baseline/readout comparisons are not experiments retraining the original three-layer backbone after branch removal. Weak-label AUPRC and composite utility are not clinical outcome metrics.',
 'All internal controls and four-fold distributions are retained; these comparisons alone do not document retraining ablations.','supplement')

# Supplement S2: all 80 exploratory estimates, including failed controls.
axorder=['OSA_state','severity','AHI','hypertension','EDS']
variants=intervals.variant.drop_duplicates().tolist()
nice={v:v.replace('_',' ') for v in variants};nice.update({'full_tri_modal':'Full tri-modal readout','fold_selected250_expression_only':'Fold-selected 250 expression','LOPO_training_mean_diagnostic':'LOPO training-mean diagnostic','age_sex_ridge_signal_component':'Age/sex ridge: signal component','age_sex_ridge_raw_prediction':'Age/sex ridge: raw prediction','frozen250_ridge_signal_component':'Fixed-universe ridge: signal component','frozen250_ridge_raw_prediction':'Fixed-universe ridge: raw prediction'})
matrix=intervals.pivot(index='variant',columns='axis',values='estimate').loc[variants,axorder]
fig,ax=plt.subplots(figsize=(W,7.4));fig.subplots_adjust(left=.46,right=.97,top=.93,bottom=.09)
im=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(5),['OSA\nAUROC','Severity\nrho','AHI\nrho','HTN\nAUROC','EDS\nAUROC']);ax.set_yticks(range(len(variants)),[nice[v] for v in variants])
for i in range(len(variants)):
    for j in range(5):
        v=matrix.iloc[i,j];ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if abs(v)>.6 else DARK,fontsize=7)
ax.spines[['left','bottom']].set_visible(False);ax.set_title('All participant readouts and diagnostics',loc='left',pad=14,fontweight='bold')
save(fig,'FigS2','all_participant_readouts',[ax],
 'All 16 variants across all five axes (32 participants). Cells report AUROC for binary axes and Spearman rho for continuous/ordinal axes; color encodes the numerical value, not comparable biological importance across metrics. Values are not sign-flipped. Fixed-universe readouts are selection-conditioned. Raw leave-one-out ridge predictions can inherit a changing training-target intercept; the training-mean diagnostic and centered signal component are shown alongside raw predictions. Centered outputs are not probabilities.',
 'All axes and controls are retained, including hypertension, EDS and negative diagnostic correlations.','supplement')

# Supplement S3: conditional intervals, not full-pipeline confidence intervals.
fig,axs=plt.subplots(1,2,figsize=(W,3.7));fig.subplots_adjust(left=.17,right=.97,top=.86,bottom=.20,wspace=.65)
for ax,axeset,letter,title in zip(axs,[['OSA_state','hypertension','EDS'],['severity','AHI']],['a','b'],['Binary axes','Continuous / ordinal axes']):
    for i,key in enumerate(axeset):
        r=intervals[(intervals.variant=='full_tri_modal')&(intervals.axis==key)].iloc[0]
        ax.plot([r.conditional_interval_low,r.conditional_interval_high],[i,i],color=TEAL,lw=2)
        ax.scatter(r.estimate,i,color=TEAL,s=30)
    ax.set_yticks(range(len(axeset)),axeset);ax.set_ylim(len(axeset)-.5,-.5);ax.set_xlim(0,1);ax.axvline(.5 if letter=='a' else 0,color=GREY,ls='--',lw=.8);ax.set_xlabel('AUROC' if letter=='a' else 'Spearman rho');tidy(ax);label(fig,ax,letter,title)
save(fig,'FigS3','conditional_uncertainty',list(axs),
 'Full-readout point estimates and 95% descriptive intervals from 2,000 resamples of participants conditional on existing out-of-fold scores. These intervals do not re-run feature selection or model fitting, do not account for overlapping training folds, and are not full-pipeline or confirmatory confidence intervals. All intervals for all variants and paired differences are provided as source tables.',
 'Intervals describe uncertainty of the available scores; they do not resolve candidate-selection bias or replace external validation.','supplement')

# Supplement S4: every Pareto pair and all evidence axes, no missing=negative.
spec=[('CPAP 1','cpap_GSE133601'),('CPAP 2','cpap_GSE49800'),('Adipose state','adipose_state'),('Adipose CPAP','adipose_CPAP'),('Pediatric scRNA','pediatric_scRNA'),('Pediatric proteome','pediatric_proteome'),('Known network','known_network'),('GSE38792','independent_GSE38792'),('GSE293433','independent_GSE293433')]
def truth(s):return s.astype(str).str.lower().isin(['true','yes','1'])
mat=np.zeros((len(pareto),len(spec)),dtype=int)
for j,(_,key) in enumerate(spec):mat[:,j]=np.where(truth(pareto[key+'_evaluable']),np.where(truth(pareto[key+'_supportive']),2,1),0)
ev=pd.DataFrame(mat,columns=[v[0] for v in spec]);ev.insert(0,'pair_id',pareto.pair_id);ev.to_csv(OUT/'source_data/FigS4_evidence_states.tsv',sep='\t',index=False)
fig,ax=plt.subplots(figsize=(W,9.0));fig.subplots_adjust(left=.37,right=.97,top=.91,bottom=.16)
ax.imshow(mat,aspect='auto',cmap=ListedColormap(['#E5E9ED','#D5B4A7',TEAL]),vmin=-.5,vmax=2.5)
ax.set_yticks(range(len(pareto)),pareto.gene_A.astype(str)+' – '+pareto.gene_B.astype(str));ax.set_xticks(range(len(spec)),[x[0] for x in spec],rotation=90,ha='center',rotation_mode='default')
ax.set_xticks(np.arange(-.5,len(spec),1),minor=True);ax.set_yticks(np.arange(-.5,len(pareto),1),minor=True);ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',length=0)
ax.set_title('All 32 Pareto pairs: external evidence states',loc='left',pad=15,fontweight='bold')
ax.legend(handles=[Patch(color='#E5E9ED',label='Not evaluable'),Patch(color='#D5B4A7',label='Not supportive'),Patch(color=TEAL,label='Supportive')],loc='lower right',bbox_to_anchor=(1,1.045),ncol=3,fontsize=7.0,frameon=False)
save(fig,'FigS4','complete_Pareto_evidence',[ax],
 'All 32 Pareto pairs across nine evidence axes. Grey = not evaluable; muted brown = evaluable but not supportive under the existing rule; teal = supportive. Not supportive does not necessarily mean opposite-direction evidence. These flags are not independent hypothesis tests or proof of interaction. Pediatric proteomics measured only 4/250 primary proteins and 0/312 top-1% dual-detected pairs, precluding its primary pair endpoint.',
 'Pair-level evidence distinguishes unavailable, unsupported and supported axes, including conflicting evidence and coverage limitations.','supplement')

# Supplement S5: all generic network rankings plus version stability.
stability=read(ST22,'version_stability_metrics.tsv')
fig,axs=plt.subplots(1,2,figsize=(W,4.6));fig.subplots_adjust(left=.27,right=.97,top=.87,bottom=.17,wspace=.95)
a,b=axs
rr=sel.sort_values('top_enrichment_ratio',ascending=False)
for i,(m,r) in enumerate(rr.iterrows()):a.scatter(r.top_enrichment_ratio,i,color=TEAL if m=='full_OSA_priority' else GREY,s=24)
rn={'full_OSA_priority':'TRIAGE-OSA','context_free_p_PPI':'Backbone PPI','context_only_r_OSA':'OSA context','bidirectional_conservative':'Conservative','priority_bidirectional_conservative':'Conservative','PPI25_context75':'PPI 25 / context 75','PPI75_context25':'PPI 75 / context 25','equal_L1_L2_L3':'Equal L1–L2–L3'}
a.set_yticks(range(len(rr)),[rn.get(m,m.replace('_',' ')) for m in rr.index]);a.invert_yaxis();a.axvline(1,color=GREY,ls='--',lw=.8);a.set_xlabel('Top-1% enrichment');tidy(a);label(fig,a,'a','All ranking comparators')
ss=stability[stability.comparison!='all_pair_rank_Spearman'];sn=['Top-1% pairs','Top-5% pairs','Pareto pairs','Assay shortlist','Module proteins']
b.barh(range(len(ss)),ss.retention_fraction_of_v3_2,color=TEAL,height=.55);b.set_yticks(range(len(ss)),sn);b.invert_yaxis();b.set_xlim(0,1.14);b.set_xlabel('Retention after participant conditioning')
for i,r in enumerate(ss.itertuples()):b.text(r.retention_fraction_of_v3_2+.04,i,f'{r.retention_fraction_of_v3_2:.0%}',va='center',fontsize=7)
label(fig,b,'b','Candidate stability')
save(fig,'FigS5','network_controls_and_stability',list(axs),
 'a, All recorded top-1% ranking comparators on training-overlap-excluded IntAct-all references. b, Set retention from V3.2 to V4; all-pair rank Spearman = 0.95249. Stable Pareto/module membership despite changed top rankings does not by itself demonstrate additional biological validity. Generic interaction recovery is reference-limited and cannot establish OSA-specific binding.',
 'Network controls and between-version stability describe robustness, not additional evidence of treatment efficacy.','supplement')


(OUT/'figure_manifest.json').write_text(json.dumps({'records':records},indent=2,ensure_ascii=False),encoding='utf8')
