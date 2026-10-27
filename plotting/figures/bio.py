import os
import sys
"""Biology-led OSA figures; frozen results only, no training or candidate reselection."""
from pathlib import Path
import sys,json,shutil,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,ListedColormap,Normalize
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox,ScaledTranslation


ROOT=Path(os.environ.get('OSA_DATA_ROOT', '.'));BASE=ROOT/'results/discovery_preprocess_v2'
OUT=Path('outputs/figures/bio');OLD=BASE/'30_manuscript_figure_package_v1'
for d in ['main_figures','supplementary_figures','panels','source_data','qa','documents']:(OUT/d).mkdir(parents=True,exist_ok=True)
W=183/25.4
DARK='#263746';TEAL='#167D8D';BLUE='#447DA9';CORAL='#BD6257';PURPLE='#927AB1';GREY='#A8B3BE';PALE='#E9EEF2'
DIV=LinearSegmentedColormap.from_list('bio_div',['#315F8C','#7EA9C6','#FAF9F6','#CC9586','#A54043'])
DIV.set_bad('#D9DEE3')
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'legend.frameon':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,
 'svg.fonttype':'none','text.color':DARK,'axes.labelcolor':DARK,'xtick.color':DARK,'ytick.color':DARK})
records=[];audit={'training_started':False,'candidate_membership_changed':False,'transformations':[]}
def read(stage,name):
    p=BASE/stage/name
    x=pd.read_csv(p,sep='\t',low_memory=False)
    return x
def js(stage):return json.loads((BASE/stage/'run_summary.json').read_text(encoding='utf-8'))
def source(x,name):x.to_csv(OUT/'source_data'/name,sep='\t',index=True)
def lab(fig,ax,l,title):
    ax.set_title(title,loc='left',pad=10,fontweight='bold')
    ax.text(0,1,l,transform=ax.transAxes+ScaledTranslation(-17/72,10/72,fig.dpi_scale_trans),fontweight='bold',fontsize=10,va='bottom')
def heat(ax,x,labels=None,cols=None,lim=2,cmap=DIV):
    im=ax.imshow(np.asarray(x,float),aspect='auto',interpolation='nearest',cmap=cmap,vmin=-lim,vmax=lim,rasterized=True)
    ax.set_yticks(range(len(x)),labels if labels is not None else ['']*len(x))
    if cols is None:ax.set_xticks([])
    else:ax.set_xticks(range(len(cols)),cols,rotation=90,rotation_mode='anchor',ha='right',va='center')
    ax.tick_params(length=0,pad=3)
    for sp in ax.spines.values():sp.set_visible(False)
    return im
def colorbar(fig,im,rect,title,ticks):
    cax=fig.add_axes(rect);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=ticks)
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=7.0,length=2);cb.set_label(title,fontsize=7.0,labelpad=2)
    return cax
def save(fig,ident,slug,axes,extras,claim,caption,category='main'):
    fig.canvas.draw()
    dest=OUT/('main_figures' if category=='main' else 'supplementary_figures');stem=ident+'_'+slug
    fig.savefig(dest/(stem+'.pdf'));fig.savefig(dest/(stem+'.svg'))
    fig.savefig(dest/(stem+'.png'),dpi=600);fig.savefig(dest/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    pdest=OUT/'panels'/ident;pdest.mkdir(exist_ok=True)
    for i,ax in enumerate(axes):
        keep=[ax]+extras.get(i,[]);vis={a:a.get_visible() for a in fig.axes}
        text_vis={t:t.get_visible() for t in fig.texts}
        for t in fig.texts:t.set_visible(False)
        for a in fig.axes:a.set_visible(a in keep)
        fig.canvas.draw()
        bb=Bbox.union([a.get_tightbbox(fig.canvas.get_renderer()) for a in keep]).transformed(fig.dpi_scale_trans.inverted()).padded(.06)
        for ext,dpi in [('pdf',None),('svg',None),('png',600),('tiff',600)]:
            kw={'bbox_inches':bb}
            if dpi:kw['dpi']=dpi
            if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
            fig.savefig(pdest/(ident+chr(97+i)+'.'+ext),**kw)
        for a,v in vis.items():a.set_visible(v)
        for t,v in text_vis.items():t.set_visible(v)
    records.append({'id':ident,'file':str((dest/(stem+'.png')).relative_to(OUT)),'claim':claim,'caption':caption,'panels':len(axes),'category':category})
    plt.close(fig);print('SAVED',ident,flush=True)
def box(ax,x,y,w,h,title,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.018',facecolor=color,edgecolor='none'))
    ax.text(x+.018,y+h-.035,title,fontsize=8,fontweight='bold',va='top')
    ax.text(x+.018,y+h-.10,text,fontsize=7,va='top',linespacing=1.65)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color=GREY,lw=.85))

C=read('04_candidates_v1','candidate_proteins_primary_250.tsv').sort_values('discovery_rank').set_index('UniProt_accession')
assert len(C)==250 and C.index.is_unique
M=pd.read_csv(ROOT/'metadata/discovery_patient_information.tsv',sep='\t');M=M[M.dataset.eq('GSE75097')].copy()
M['group_order']=M.study_group.map({'PS':0,'MSO':1,'VSO':2});M=M.sort_values(['group_order','AHI_events_h','sample_id']).set_index('sample_id')
assert len(M)==32 and M.study_group.value_counts().to_dict()=={'MSO':15,'VSO':11,'PS':6}
E=read('03_mapping/GSE75097','protein_expression_primary_samples.tsv.gz').set_index('UniProt_accession').loc[C.index,M.index]
assert np.isfinite(E.to_numpy()).all()
Z=E.sub(E.mean(axis=1),axis=0).div(E.std(axis=1,ddof=1).replace(0,np.nan),axis=0)
assert np.isfinite(Z.to_numpy()).all()
source(M,'patient_annotations_32.tsv');source(C,'frozen_primary_250.tsv');source(E,'expression_250_by_32_log2.tsv');source(Z,'expression_250_by_32_row_z.tsv')
ST22='22_consolidated_evidence_triage_osa_v4';P=read(ST22,'protein_evidence_matrix.tsv').set_index('UniProt')
mods=read(ST22,'mechanism_modules.tsv');pairs=read(ST22,'pareto_front_candidates.tsv')
module_ids=[];module_labels=[]
for r in mods.itertuples():
    ids=sorted(r.proteins.split('|'),key=lambda u:float(C.loc[u,'discovery_rank']))
    module_ids+=ids;module_labels +=[r.module_id]*len(ids)
assert len(module_ids)==17
cpap={};cpap_ev={}
for ds in ['GSE133601','GSE49800']:
    cpap_ev[ds]=read('10_cpap_validation_preprocess_v1/'+ds,'candidate_validation_evidence.tsv').set_index('UniProt_accession')
    t=read('10_cpap_validation_preprocess_v1/'+ds,'candidate_participant_deltas.tsv.gz').set_index('UniProt_accession')
    cs=sorted([k for k in t.columns if k.startswith('subject_')],key=lambda k:k[8:])
    # Same discovery orientation used by the existing CPAP endpoint; no centering of changes.
    cpap[ds]=t.loc[C.index,cs].mul(-np.sign(C.OSA_beta),axis=0)
    source(cpap[ds],ds+'_all250_oriented_log2_changes.tsv')
audit['transformations']=[{'name':'expression_heatmap','rule':'row z=(log2 expression - row mean)/row sample SD over all 32 participants; display saturation +/-2; no values discarded'},
 {'name':'CPAP_heatmap','rule':'oriented delta=-sign(discovery OSA beta)*(post-pre); no centering; saturation +/-0.6 log2; missing values grey'},
 {'name':'display_selection','rule':'Fig2 first30 by frozen discovery rank; Fig3/5 all17 existing module proteins; complete250 views in S6/S7. No selection by new P values.'}]

def annotated_expression(ident,ids,allview=False):
    dat=Z.loc[ids];n=len(ids)
    if allview:
        frozen_order=pd.read_csv(BASE/'30_manuscript_figure_package_v2/source_data/FigS6_display_expression.tsv',sep='\t',index_col=0).index;dat=dat.loc[frozen_order];ids=dat.index.tolist()
    source(dat,ident+'_display_expression.tsv')
    fig=plt.figure(figsize=(W,7.6 if allview else 7.6))
    gs=fig.add_gridspec(1,5,left=.135,right=.96,bottom=.18,top=.70,wspace=.13)
    a=fig.add_subplot(gs[0,:4]);b=fig.add_subplot(gs[0,4]);pos=a.get_position()
    labels=C.loc[ids,'gene_symbol_current'].tolist() if not allview else None
    im=heat(a,dat,labels,[f'{i+1:02d}' for i in range(32)],lim=2)
    if allview:a.set_yticks([0,n-1],['1','250']);a.set_ylabel('All frozen candidates; clustered rows')
    for x in [5.5,20.5]:a.axvline(x,color='white',lw=1.5)
    a.set_xlabel('Participants ordered by study group, then AHI',labelpad=9)
    # Metadata strips remain separate axes with explicit units and ranges.
    extras=[];start=.735;h=.022;gap=.013
    specs=[('Study group',M.group_order.to_numpy(),ListedColormap([GREY,BLUE,CORAL]),0,2),
      ('AHI (events/h)',M.AHI_events_h.to_numpy(float),'YlOrBr',0,float(M.AHI_events_h.max())),
      ('Age (years)',M.age_years.to_numpy(float),'Greys',float(M.age_years.min()),float(M.age_years.max())),
      ('Sex',M.sex.map({'female':0,'male':1}).to_numpy(float),ListedColormap([PURPLE,TEAL]),0,1),
      ('Hypertension',M.hypertension.to_numpy(float),ListedColormap(['#E6E8EB','#5E657A']),0,1),
      ('EDS',M.EDS.to_numpy(float),ListedColormap(['#E6E8EB','#5E657A']),0,1)]
    for i,(name,vals,cmap,lo,hi) in enumerate(specs):
        ax=fig.add_axes([pos.x0,start+(5-i)*(h+gap),pos.width,h]);ax.imshow(vals[None,:],aspect='auto',cmap=cmap,vmin=lo,vmax=hi);ax.set_xticks([]);ax.set_yticks([0],[name],fontsize=7.0);ax.tick_params(length=0)
        for sp in ax.spines.values():sp.set_visible(False)
        extras.append(ax)
    topax=extras[0];lab(fig,topax,'a','Participant context and protein-mapped transcript abundance')
    topax.text(1.02,.5,'PS / MSO / VSO',transform=topax.transAxes,fontsize=7.0,va='center')
    for idx,txt in [(1,f'0–{M.AHI_events_h.max():.0f}'),(2,f'{M.age_years.min():.0f}–{M.age_years.max():.0f}'),(3,'Female / male'),(4,'No / yes'),(5,'No / yes')]:extras[idx].text(1.02,.5,txt,transform=extras[idx].transAxes,fontsize=7.0,va='center')
    cols=['OSA_moderated_t','severe_moderated_t','AHI_moderated_t','IH_moderated_t']
    eff=C.loc[ids,cols];im2=heat(b,eff,None,['OSA','Severity','AHI','IH'],lim=3);b.set_yticks([]);lab(fig,b,'b','Discovery effects')
    cb1=colorbar(fig,im,[pos.x0,.085,.22,.013],'Expression row z-score',[-2,0,2]);cb2=colorbar(fig,im2,[.73,.085,.22,.013],'Moderated t',[-3,0,3])
    fig.text(.45,.015,'Prioritized candidates, not FDR-significant DEGs; grey = unavailable',ha='center',fontsize=7,color=DARK)
    # Footer belongs to the composite; panel exports retain scales and annotations.
    save(fig,ident,'annotated_discovery_heatmap',[a,b],{0:extras+[cb1],1:[cb2]},
      'Participant annotations accompany candidate expression patterns; discovery-selected group patterns are not independent predictive validation.',
      ('All 250 frozen candidates; rows ordered by average-linkage Euclidean clustering of row-z expression.' if allview else 'The first 30 candidates by the existing frozen discovery rank; no reselection using these plots or external outcomes.')+
      ' Columns show all 32 development participants ordered by PS/MSO/VSO and increasing AHI within group; numerical labels map to sample IDs in Source Data. Expression is log2 transcript abundance mapped to proteins, row-standardized across participants, with colors saturated at +/-2. Metadata use observed individual values; BMI is unavailable. The right matrix shows original moderated t statistics (color saturation +/-3); grey indicates unavailable IH measurements. No selected protein meets FDR <0.05 on the four displayed axes. Selection-conditioned patterns are descriptive, not independent clinical accuracy.', 'supplement' if allview else 'main')

annotated_expression('Fig2',C.index[:30].tolist())

def cpap_heatmap(ident,ids,allview=False):
    matrices=[cpap[d].loc[ids] for d in ['GSE133601','GSE49800']];data=pd.concat(matrices,axis=1)
    data.columns=[d+':'+s.replace('subject_','') for d in ['GSE133601','GSE49800'] for s in cpap[d].columns]
    source(data,ident+'_oriented_CPAP_heatmap.tsv')
    fig=plt.figure(figsize=(W,7.1 if not allview else 7.4));gs=fig.add_gridspec(2,6,left=.14,right=.96,top=.88,bottom=.16,height_ratios=[1.9,1],hspace=.62,wspace=.7)
    a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,:3]);c=fig.add_subplot(gs[1,3:])
    labels=[C.loc[u,'gene_symbol_current'] for u in ids] if not allview else None
    im=heat(a,data,labels,[x.split(':')[1] for x in data.columns],lim=.6)
    a.axvline(14.5,color='white',lw=2)
    if not allview:
        for y in [10.5,14.5]:a.axhline(y,color='white',lw=1.5)
        for mid,yy in [('M001',5),('M002',12.5),('M003',15.5)]:a.text(1.013,1-(yy+.5)/17,mid,transform=a.transAxes,fontsize=7.0,va='center',rotation=90)
    else:a.set_yticks([0,249],['1','250']);a.set_ylabel('Frozen discovery rank')
    a.set_xlabel('Participant within cohort',labelpad=5)
    pos=a.get_position();st=fig.add_axes([pos.x0,.895,pos.width,.018]);st.imshow(np.array([0]*15+[1]*18)[None,:],aspect='auto',cmap=ListedColormap([BLUE,PURPLE]));st.axis('off')
    st.text(7,0,'GSE133601 · n = 15',ha='center',va='center',color='white',fontsize=7.0);st.text(23.5,0,'GSE49800 · n = 18',ha='center',va='center',color='white',fontsize=7.0)
    lab(fig,st,'a','Paired molecular response across all 33 participants')
    cb=colorbar(fig,im,[.28,.065,.27,.012],'Oriented post − pre (log2)',[-.6,0,.6])
    # Descriptive component means use every participant, no fitted subtypes or cutoffs.
    rows=[]
    for j,r in enumerate(mods.itertuples()):
        uu=r.proteins.split('|')
        for k,ds in enumerate(['GSE133601','GSE49800']):
            values=cpap[ds].loc[uu].mean(axis=0,skipna=True)
            for n,(sid,v) in enumerate(values.items()):
                b.scatter(j+(k-.5)*.24+(n-(len(values)-1)/2)*.006,v,s=12,color=[BLUE,PURPLE][k],alpha=.8,edgecolors='none')
                rows.append({'module':r.module_id,'dataset':ds,'subject':sid,'mean_oriented_change':v,'measured_proteins':int(cpap[ds].loc[uu,sid].notna().sum())})
            b.plot([j+(k-.5)*.24-.065,j+(k-.5)*.24+.065],[values.median()]*2,color=DARK,lw=1.2)
    b.axhline(0,color=GREY,lw=.8,ls='--');b.set_xticks(range(3),mods.module_id);b.set_ylabel('Mean oriented change (log2)');lab(fig,b,'b','Component-level responses')
    s=js('11_cpap_reversal_endpoint_triage_osa_v4'); sf=js('12_cpap_subject_signflip_triage_osa_v4')
    for i,ds in enumerate(['GSE133601','GSE49800']):
        for dy,obj,rkey,pkey,col in [(-.12,s['cohorts'][ds],'primary_top_1pct_enrichment_ratio','top_1pct_permutation_P',TEAL),(.12,sf['cohorts'][ds],'observed_top_1pct_enrichment_ratio_unmoderated_t','top_1pct_one_sided_signflip_P',PURPLE)]:
            c.scatter(obj[rkey],i+dy,s=24,color=col);c.text(1.23,i+dy,f'P={obj[pkey]:.3f}',fontsize=7.0,va='center',color=col)
    c.set_yticks([0,1],['GSE133601','GSE49800']);c.invert_yaxis();c.axvline(1,ls='--',lw=.8,color=GREY);c.set_xlim(.72,1.51);c.set_xlabel('Top-1% reversal enrichment');lab(fig,c,'c','Global endpoint boundary')
    c.legend(handles=[Line2D([],[],color=TEAL,marker='o',ls='',label='Primary'),Line2D([],[],color=PURPLE,marker='o',ls='',label='Sign-flip')],loc='lower center',bbox_to_anchor=(.5,-.5),ncol=2,fontsize=7.0)
    source(pd.DataFrame(rows),ident+'_module_participant_means.tsv')
    save(fig,ident,'CPAP_response_atlas',[a,b,c],{0:[st,cb]},
      'Response directions vary across proteins and participants; these patterns do not establish clinical benefit subtypes.',
      ('All 250 primary proteins in frozen discovery-rank order.' if allview else 'All 17 proteins from the three existing candidate modules, grouped by component and ordered by frozen discovery rank.')+
      ' a, Paired log2 post-minus-pre changes multiplied by minus the sign of the discovery OSA coefficient. Positive means molecular reversal; values are not centered. Common color saturation is +/-0.6; unavailable measurements are grey. These are protein-mapped transcript measurements, not proteomic abundance. b, Every participant mean across measured proteins within each existing component; bars show medians, not confidence intervals. Components were selected using previously inspected evidence; these are post-selection descriptions, not validated response subtypes. c, Existing primary and sign-flip endpoint tests (10,000 permutations; nominal one-sided P). Pooled primary P=0.453 and pooled sign-flip P=0.520 do not support a shared global reversal claim.', 'supplement' if allview else 'main')

cpap_heatmap('Fig3',module_ids)

# Fig4: pathway-scale signal, candidate-scale concordance, and pair-level boundary.
ST21='21_independent_validation_endpoint_triage_osa_v4'
path=read(ST21,'reactome_pathway_axis_results.tsv');path=path[path.level=='axis'].copy()
node=read(ST21,'candidate_node_evidence_primary_250.tsv').set_index('UniProt_accession')
perm=read(ST21,'node_aware_permutation_summary.tsv')
axesorder=['hypoxia_HIF_ROS','NFkB_innate_inflammation','antigen_lymphocyte_signalling','PI3K_AKT_JAK_STAT_TGFb','proteostasis_ER_stress','PPAR_adipose_lipid_metabolism']
names=['Hypoxia / HIF / ROS','NF-κB / innate immunity','Antigen / lymphocyte','PI3K–AKT / JAK–STAT / TGF-β','Proteostasis / ER stress','PPAR / lipid metabolism']
fig=plt.figure(figsize=(W,6.0));gs=fig.add_gridspec(2,6,left=.14,right=.95,top=.92,bottom=.13,height_ratios=[1.3,1],hspace=.64,wspace=.65)
a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,:3]);c=fig.add_subplot(gs[1,3:])
eff=path.pivot(index='axis',columns='dataset',values='competitive_effect').loc[axesorder,['GSE38792','GSE293433']]
q=path.pivot(index='axis',columns='dataset',values='BH_FDR_axis_family').loc[axesorder,['GSE38792','GSE293433']]
for i in range(6):
    a.axhspan(i-.43,i+.43,color='#F2F5F7',zorder=0)
    a.text(-.85,i,names[i],ha='right',va='center',fontsize=7)
    for j in range(2):
        value=float(eff.iloc[i,j]);qq=float(q.iloc[i,j]);a.scatter(j,i,s=50+70*min(-np.log10(max(qq,1e-12)),4),c=[DIV((value+.7)/1.4)],edgecolors='white',lw=.7,zorder=3)
        a.text(j+.18,i,f'{value:+.2f}   q={qq:.3g}',fontsize=7.0,va='center')
a.set_xlim(-2.3,1.8);a.set_ylim(5.7,-1.6);a.axis('off');lab(fig,a,'a','Pathway effects across external cohorts')
for xx,tt,dd,co in [(0,'Visceral adipose','GSE38792 · n = 18',BLUE),(1,'PBMC','GSE293433 · n = 10',PURPLE)]:
    a.text(xx,-1.30,tt,ha='center',va='center',fontsize=7,color=co)
    a.text(xx,-.82,dd,ha='center',va='center',fontsize=7.0,color=co)
# All complete candidate pairs; no p-value-driven gene filtering or labeling.
nc=node[['GSE38792_oriented_t','GSE293433_oriented_t']].dropna();x=nc.iloc[:,0];y=nc.iloc[:,1]
mask=(x>0)&(y>0);b.scatter(x,y,s=12,c=np.where(mask,TEAL,'#B9C1C8'),alpha=.72,edgecolors='none');b.axhline(0,color=GREY,lw=.7);b.axvline(0,color=GREY,lw=.7)
b.set_xlabel('Adipose: direction-aligned t');b.set_ylabel('PBMC: direction-aligned t');lab(fig,b,'b',f'Candidate concordance · n = {len(nc)}')
b.text(.02,.97,f'Both aligned: {int(mask.sum())}/{len(nc)}',transform=b.transAxes,va='top',fontsize=7.0)
top=perm[(perm.model=='full_TRIAGE_OSA')&(perm.statistic=='top_1pct_support_enrichment_ratio')]
for i,r in enumerate(top.itertuples()):
    c.plot([r.null_q025,r.null_q975],[i,i],lw=4,color='#C9D5DC',solid_capstyle='round');c.scatter(r.observed,i,s=25,color=TEAL,zorder=3);c.text(.69,i+.27,f'P={r.one_sided_P:.3f}',fontsize=7.0)
c.set_yticks(range(len(top)),['Adipose','PBMC','Joint']);c.set_ylim(2.55,-.55);c.set_xlim(.66,1.33);c.axvline(1,color=GREY,ls='--',lw=.8);c.set_xlabel('Top-1% pair enrichment');lab(fig,c,'c','Pair-level external test')
source(path,'Fig4_pathway_axis_results.tsv');source(nc,'Fig4_complete_candidate_effects.tsv');source(top,'Fig4_pair_endpoint_null_intervals.tsv')
save(fig,'Fig4','cross_cohort_biological_programs',[a,b,c],{},
 'Pathway direction and candidate support depend on the cohort; local evidence does not establish success of the overall pair endpoint.',
 'a, Six prespecified pathway axes. Color shows competitive effect (blue negative/red positive); bubble area increases with -log10(BH q), capped at 4. Exact q values retain the registered directional/two-sided test: q=1 does not establish a significant opposite effect. b, All candidates with both independent oriented t values; positive is concordant with discovery OSA direction. No effect-size threshold or P-value exclusion is applied. c, Frozen-ranking pair enrichment, with central 95% node-aware permutation null intervals (not effect confidence intervals), 10,000 permutations. Cohort, tissue, population and assay differences cannot be disentangled by these two cohorts alone.')

# Fig5: compact evidence-colored 17-node network with all protein-level support.
edges=pairs[pd.to_numeric(pairs.triage_osa_rank_primary_250)<=1557].copy();assert len(edges)==15
source(edges,'Fig5_exact_module_edges.tsv')
fig=plt.figure(figsize=(W,6.2));gs=fig.add_gridspec(2,6,left=.075,right=.965,top=.91,bottom=.17,height_ratios=[2.2,.75],hspace=.32,wspace=.5)
a=fig.add_subplot(gs[0,:3]);b=fig.add_subplot(gs[0,3:]);c=fig.add_subplot(gs[1,:])
a.axis('off');lab(fig,a,'a','Candidate interaction modules')
gene=C.gene_symbol_current.to_dict();inv={v:k for k,v in gene.items()}
# Deterministic module-aware geometry; position is a display choice, not a learned embedding.
gpos={'SCAMP5':(0,1),'EOGT':(-.75,1.85),'ARL6IP5':(-1.22,2.40),'CSF2RB':(-.1,2.43),
 'CRYZ':(.75,1.85),'TIMMDC1':(1.13,1.04),'SLC23A2':(.85,.42),'ATP5MK':(.16,.10),'PIGK':(-.65,.36),'SCYL2':(-1.22,1.08),'VAMP7':(1.05,2.55),
 'UBE2Q2':(-.25,-1.02),'BRCA1':(.72,-.76),'HELZ2':(-1.14,-.70),'TDRD3':(-.4,-1.72),'IL17RB':(.45,-2.35),'VIT':(-.7,-2.35)}
pos={u:gpos[gene[u]] for u in module_ids};norm=Normalize(-.3,.3)
for r in edges.itertuples():
    u,v=r.protein_A,r.protein_B;known=str(r.intact_all_positive).lower()=='yes'
    p0=np.array(pos[u]);p1=np.array(pos[v]);direction=(p1-p0)/np.linalg.norm(p1-p0)
    p0=p0+direction*(.22 if gene[u]=='SCAMP5' else .10);p1=p1-direction*(.22 if gene[v]=='SCAMP5' else .10)
    a.plot([p0[0],p1[0]],[p0[1],p1[1]],color=TEAL if known else '#AEB9C1',ls='-' if known else (0,(2,2)),lw=1.8 if known else .9,zorder=1)
for u,(x,y) in pos.items():
    a.scatter(x,y,s=95+30*float(P.loc[u,'pareto_pair_count']),c=[DIV(norm(C.loc[u,'OSA_beta']))],edgecolors='white',lw=.8,zorder=3)
    special={'EOGT':(x-.20,y,'right','center'),'CRYZ':(x+.20,y,'left','center'),'TIMMDC1':(x+.19,y-.06,'left','center'),'ATP5MK':(x,y-.18,'center','top'),'TDRD3':(x,y-.18,'center','top')}
    special.update({'PIGK':(x,y+.27,'center','bottom'),'UBE2Q2':(x,y+.29,'center','bottom')})
    if gene[u]=='SCAMP5':a.text(x,y+.50,gene[u],ha='center',va='bottom',fontsize=7.0,color=DARK,fontweight='bold',zorder=4)
    else:
        tx,ty,ha,va=special.get(gene[u],(x,y+.18,'center','bottom'));a.text(tx,ty,gene[u],ha=ha,va=va,fontsize=7.0)
a.set_xlim(-1.75,2.0);a.set_ylim(-2.7,3.1)
for text,x,y in [('M001',-1.58,2.83),('M002',-1.58,-.18),('M003',-1.58,-2.01)]:a.text(x,y,text,fontsize=7,fontweight='bold',color=TEAL)
vals=[]
for u in module_ids:
    sgn=np.sign(C.loc[u,'OSA_beta']);vals.append([-sgn*cpap_ev['GSE133601'].loc[u,'moderated_t'],-sgn*cpap_ev['GSE49800'].loc[u,'moderated_t'],node.loc[u,'GSE38792_oriented_t'],node.loc[u,'GSE293433_oriented_t']])
ev=pd.DataFrame(vals,index=module_ids,columns=['CPAP 1','CPAP 2','Adipose','PBMC']);source(ev,'Fig5_direction_aligned_support_t.tsv')
im=heat(b,ev,[gene[u] for u in module_ids],ev.columns,lim=3)
for yy in [10.5,14.5]:b.axhline(yy,color='white',lw=1.6)
lab(fig,b,'b','Treatment and external-cohort support')
c.axis('off');lab(fig,c,'c','What each evidence layer contributes')
for xx,title,body in [(0,'Disease association','Node color: OSA log2 effect\nNode size: Pareto-pair degree'),(.36,'Interaction evidence','Solid teal: IntAct-supported\nDashed grey: model hypothesis'),(.73,'External support','Heatmap: oriented t\nGrey: unavailable measurement')]:
    c.text(xx,.88,title,fontsize=7.5,fontweight='bold',transform=c.transAxes);c.text(xx,.65,body,fontsize=7.0,va='top',linespacing=1.65,transform=c.transAxes)
cb1=colorbar(fig,plt.cm.ScalarMappable(norm=norm,cmap=DIV),[.13,.045,.24,.013],'Discovery OSA log2 effect',[-.3,0,.3])
cb2=colorbar(fig,im,[.61,.045,.25,.013],'Direction-aligned t',[-3,0,3])
save(fig,'Fig5','evidence_colored_mechanism_network',[a,b,c],{0:[cb1],1:[cb2]},
 'Candidate networks connect disease associations, treatment changes and external cohort evidence; predicted edges remain distinct from verified interactions.',
 'a, All 17 proteins and 15 edges from the existing Pareto/top-5% components. Node color is the discovery OSA log2 coefficient (saturation +/-0.3); area scales with the number of Pareto pairs involving that protein. The single IntAct-supported edge is solid teal; remaining edges are dashed hypotheses, not experimentally established OSA interactions. Geometric layout is deterministic and has no statistical meaning. b, The same proteins in module order: CPAP t statistics are multiplied by minus the discovery coefficient sign, while independent-cohort t values are aligned with its sign. Positive therefore means expected reversal or disease concordance, respectively; scales are capped at +/-3 and missing values are grey. Components were derived using previously inspected evidence: this is an evidence summary, not a new validation. No pathway or clinical-causality labels are assigned without support.')

annotated_expression('FigS6',C.index.tolist(),True)
cpap_heatmap('FigS7',C.index.tolist(),True)

(OUT/'figure_manifest.json').write_text(json.dumps({'records':records,'audit':audit},indent=2,ensure_ascii=False),encoding='utf8')
