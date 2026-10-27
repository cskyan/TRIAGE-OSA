import os
import sys
"""Replace the text schematic with an observed cohort and assay-coverage atlas."""
from pathlib import Path
import json, sys, shutil, re, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox, ScaledTranslation
from matplotlib.patches import Patch

ROOT=Path(os.environ.get('OSA_DATA_ROOT', '.'))
B=ROOT/'results/discovery_preprocess_v2'
O=Path('outputs/figures/coverage')
for d in ['main_figures','panels/Fig1','source_data','qa']: (O/d).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,'plotting/qa')
from audit_panel_alignment import require_matplotlib_panel_alignment
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':8,'axes.labelsize':7,
    'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'text.color':'#263746',
    'axes.labelcolor':'#263746','xtick.color':'#263746','ytick.color':'#263746'})
BLUE='#447DA9';TEAL='#167D8D';PURPLE='#927AB1';GOLD='#B18C50';PALE='#EDF1F4';DARK='#263746'
def read(stage,name):return pd.read_csv(B/stage/name,sep='\t',low_memory=False)
def js(stage):return json.loads((B/stage/'run_summary.json').read_text(encoding='utf-8'))
def source(x,name):x.to_csv(O/'source_data'/name,sep='\t',index=False)
def label(ax,key,title):
    ax.set_title(title,loc='left',pad=12,fontweight='bold')
    ax.text(0,1,key,transform=ax.transAxes+ScaledTranslation(-16/72,12/72,fig.dpi_scale_trans),fontsize=10,fontweight='bold',va='bottom')

# Contract: complementary sampling designs and uneven patient/assay coverage bound inference.
# Counts describe analyzed records, not pooled independent subjects; no significance testing.
D=pd.read_csv(ROOT/'metadata/discovery_patient_information.tsv',sep='\t')
DM=D[D.dataset.eq('GSE75097')].copy()
CM=read('25_cpap_subject_clinical_package_triage_osa_v4','clinical_subject_manifest.tsv')
S10=js('10_cpap_validation_preprocess_v1');S13=js('13_adipose_validation_preprocess_v1')
S15=js('15_scrna_validation_preprocess_v1');S17=js('17_pediatric_proteome_preprocess_v1')
S20=js('20_independent_validation_preprocess_v1')
registry=pd.read_csv(ROOT/'metadata/dataset_registry.tsv',sep='\t').set_index('dataset')
cohorts=[
 ('GSE75097','Adult PBMC',len(DM),len(DM),'Discovery',BLUE,'PS 6 / MSO 15 / VSO 11'),
 ('GSE71356','Adult PBMC',int(registry.loc['GSE71356','manifest_subjects']),int(registry.loc['GSE71356','observed_assays']),'Discovery',BLUE,'IH crossover'),
 ('GSE133601','Adult PBMC',S10['cohorts']['GSE133601']['subjects'],S10['cohorts']['GSE133601']['assays'],'Paired CPAP',PURPLE,'Pre / post'),
 ('GSE49800','Adult PBL',S10['cohorts']['GSE49800']['subjects'],S10['cohorts']['GSE49800']['assays'],'Paired CPAP',PURPLE,'Pre / post'),
 ('GSE135917','Subcutaneous fat',S13['case_control']['controls']+S13['case_control']['OSA']+S13['paired_CPAP']['subjects'],18+S13['paired_CPAP']['assays'],'Adult tissue',TEAL,'18 state + 24 paired'),
 ('GSE38792','Visceral fat',S20['GSE38792']['samples'],S20['GSE38792']['samples'],'Adult tissue',TEAL,'8 control / 10 OSA'),
 ('GSE293433','Adult PBMC',S20['GSE293433']['samples'],S20['GSE293433']['samples'],'Adult tissue',TEAL,'5 control / 5 OSA'),
 ('GSE214865','Pediatric PBMC',S15['participants_processed'],S15['participants_processed'],'Pediatric',GOLD,'11 control / 11 OSA'),
 ('PXD032734','Pediatric serum',S17['participants'],S17['participants'],'Pediatric',GOLD,'TMT: 4 groups of 3')]
cohort_table=pd.DataFrame(cohorts,columns=['dataset','tissue','participants','samples','role','color','design'])
assert cohort_table.participants.tolist()==[32,4,15,18,42,18,10,22,12]
source(cohort_table,'Fig1_analyzed_cohort_counts.tsv')

C=read('04_candidates_v1','candidate_proteins_primary_250.tsv').sort_values('discovery_rank').set_index('UniProt_accession')
P=read('22_consolidated_evidence_triage_osa_v4','protein_evidence_matrix.tsv').set_index('UniProt').reindex(C.index)
sc=read('16_scrna_cell_context_endpoint_triage_osa_v4','candidate_scrna_node_support.tsv')
availability=[];covlabels=[]
for txt,col in [('CPAP 1 | PBMC','GSE133601_available'),('CPAP 2 | PBL','GSE49800_available'),('Subcutaneous fat','adipose_available'),('Visceral fat','GSE38792_available'),('Adult PBMC','GSE293433_available')]:
    availability.append(P[col].eq('yes').to_numpy());covlabels.append(txt)
for lineage,txt in [('T_NK','Pediatric T / NK'),('B','Pediatric B'),('Myeloid','Pediatric myeloid')]:
    t=sc[sc.lineage.eq(lineage)].set_index('UniProt_accession').reindex(C.index)
    availability.append(t.available.eq('yes').to_numpy());covlabels.append(txt)
availability.append(P.proteome_detected.eq('yes').to_numpy());covlabels.append('Pediatric serum proteins')
A=np.asarray(availability,int);counts=A.sum(axis=1)
assert counts.tolist()==[243,236,236,236,204,206,215,213,4],counts
# Display only: order by decreasing number of assays available, then frozen discovery rank.
order=np.lexsort((C.discovery_rank.to_numpy(),-A.sum(axis=0)))
cov=pd.DataFrame(A.T,index=C.index,columns=covlabels).reset_index(names='UniProt')
cov['display_order']=np.argsort(order)+1;source(cov,'Fig1_primary_250_assay_availability.tsv')

fields=[('Age','age_years','age_years'),('Sex','sex','sex'),('BMI','BMI_kg_m2','BMI_kg_m2'),
 ('AHI','AHI_events_h','baseline_AHI_events_h'),('HTN','hypertension','hypertension'),
 ('EDS','EDS',None),('ESS',None,'baseline_ESS_score'),('CPAP use',None,'CPAP_adherence_hours_night')]
clinical=[];denoms=[len(DM),15,18]
for i,(name,dcol,ccol) in enumerate(fields):
    row=[int(DM[dcol].notna().sum()) if dcol else 0]
    for ds in ['GSE133601','GSE49800']:
        t=CM[CM.dataset.eq(ds)];row.append(int(t[ccol].notna().sum()) if ccol else 0)
    clinical.append(row)
K=np.asarray(clinical).T; frac=K/np.asarray(denoms)[:,None];frac[0,-1]=np.nan
source(pd.DataFrame([{'cohort':ds,'field':field[0],'available_n':int(K[i,j]),'denominator':denoms[i],
 'applicable':not(i==0 and j==7)} for i,ds in enumerate(['GSE75097','GSE133601','GSE49800']) for j,field in enumerate(fields)]),'Fig1_individual_clinical_field_availability.tsv')

W=183/25.4
fig=plt.figure(figsize=(W,8.5))
gs=fig.add_gridspec(3,1,left=.24,right=.96,top=.945,bottom=.115,height_ratios=[2.2,1,1.65],hspace=.48)
a=fig.add_subplot(gs[0]);b=fig.add_subplot(gs[1]);c=fig.add_subplot(gs[2])
label(a,'a','Analyzed cohorts and sampling designs')
a.set_xlim(0,76);a.set_ylim(9.1,-.6);a.axis('off')
for i,r in cohort_table.iterrows():
    if i%2==0:a.axhspan(i-.42,i+.42,color='#F6F8F9',zorder=0)
    a.text(-1,i-.17,r.dataset,ha='right',va='center',fontsize=7,fontweight='bold')
    a.text(-1,i+.28,r.tissue,ha='right',va='center',fontsize=7,color='#677782')
    # One dot per biological participant; positions do not encode outcomes.
    a.scatter(np.arange(r.participants)+.6,np.repeat(i,r.participants),s=8,color=r.color,linewidths=0)
    a.text(44.3,i,str(r.participants),ha='left',va='center',fontweight='bold',color=r.color,fontsize=7)
    a.text(51,i-.10,str(r.samples),va='center',fontsize=7,fontweight='bold')
    a.text(57,i-.10,r.design,va='center',fontsize=7)
    if r.dataset=='GSE214865':a.text(51,i+.23,'libraries',fontsize=5.5,color='#677782',va='center')
    if r.dataset=='PXD032734':a.text(51,i+.23,'channels',fontsize=5.5,color='#677782',va='center')
a.text(0,-.68,'One dot = one participant',fontsize=7,color='#677782',va='bottom')
a.text(44.3,-.68,'n',fontsize=7,va='bottom');a.text(51,-.68,'Samples / design',fontsize=7,va='bottom')
for x,txt,col in [(0,'Discovery',BLUE),(18,'Paired CPAP',PURPLE),(40,'Adult tissue',TEAL),(61,'Pediatric',GOLD)]:
    a.scatter([x],[8.83],s=16,color=col,linewidths=0);a.text(x+1.7,8.83,txt,va='center',fontsize=7)

label(b,'b','Patient-level clinical information actually linked')
cmap=ListedColormap([PALE,TEAL]);cmap.set_bad('#FFFFFF')
b.imshow(frac,aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
b.set_yticks(range(3),['Discovery | n = 32','CPAP 1 | n = 15','CPAP 2 | n = 18'])
b.set_xticks(range(len(fields)),[r[0] for r in fields]);b.tick_params(length=0,pad=5)
for i in range(3):
    for j in range(8):
        b.text(j,i,'N/A' if not np.isfinite(frac[i,j]) else f'{K[i,j]}/{denoms[i]}',ha='center',va='center',fontsize=7,color='white' if frac[i,j]>0.5 else DARK)
for y in [.5,1.5]:b.axhline(y,color='white',lw=2)
for x in np.arange(.5,7.5):b.axvline(x,color='white',lw=2)
for sp in b.spines.values():sp.set_visible(False)

label(c,'c','Assay coverage of the same 250 frozen candidates')
c.imshow(A[:,order],aspect='auto',cmap=ListedColormap([PALE,TEAL]),vmin=0,vmax=1,interpolation='nearest',rasterized=True,extent=[0,250,8.5,-.5])
c.set_xlim(0,280);c.set_ylim(8.5,-.5)
c.set_yticks(range(9),covlabels);c.tick_params(axis='y',length=0,pad=5)
c.set_xticks([0,50,100,150,200,250]);c.set_xticklabels(['1','50','100','150','200','250'])
c.set_xlabel('Candidates ordered by assay availability (display only)',labelpad=6)
for y in np.arange(.5,8.5):c.plot([0,250],[y,y],color='white',lw=1)
for i,n in enumerate(counts):c.text(254,i,f'{n}/250',va='center',fontsize=7,fontweight='bold',color=GOLD if i==8 else DARK)
for sp in c.spines.values():sp.set_visible(False)
c.legend(handles=[Patch(facecolor=TEAL,label='Available'),Patch(facecolor=PALE,label='Unavailable')],loc='upper left',bbox_to_anchor=(0,-.25),ncol=2,frameon=False,fontsize=7)
fig.text(.24,.014,'Coverage is not validation success; transcript assays are not protein-abundance measurements.',fontsize=7,color='#677782')

caption=('a, Biological participant counts in the analyzed cohorts; each dot denotes one participant, not one sample. '
 'GSE75097 shows the 32 frozen primary-analysis participants, not all deposited samples. GSE71356 has four participants and 16 crossover assays. '
 'GSE135917 comprises 18 case-control participants and a separate 24-person paired cohort (66 samples). GSE214865 counts participant libraries; PXD032734 counts participant TMT channels, not LC fractions. '
 'b, Nonmissing individual fields in the locally assembled discovery and CPAP manifests; entries are available participants/total. Zero denotes absent linked fields, not absence of disease. '
 'N/A marks CPAP adherence in the untreated discovery cohort. EDS is a binary annotation and is not interchangeable with the continuous Epworth Sleepiness Scale (ESS). Published cohort summaries were not assigned to individuals. '
 'c, Availability of all 250 frozen candidates under each existing assay-specific mapping/filtering rule. Columns are ordered by decreasing number of available assays and then discovery rank, solely for display. '
 'Rows denote RNA-derived evidence except pediatric serum proteins. Missingness may reflect mapping or filtering rather than biological absence. '
 'Counts are descriptive; no statistical test or new cohort pooling was performed. These panels describe the evidence envelope, not clinical prediction performance or successful external validation.')
claim='Cohorts provide complementary evidence; participants, available clinical fields and candidate measurement coverage are counted separately.'
contract={'question':'Which biological and patient-information evidence is actually available?',
 'claim':claim,'archetype':'quantitative grid with cohort unit marks',
 'panels':{'a':'Analyzed sampling design; participant is the biological unit','b':'Linked clinical variables; nonmissing counts, not diagnoses','c':'All 250 candidate availability under fixed preprocessing'},
 'backend':'python','width_mm':183,'statistics':'Descriptive counts; no p values or error bars applicable',
 'AI_generated_image_assets':False,'image2_decision':'Optional schematic assets were considered; native quantitative/vector rendering avoids decorative repetition of the author model framework.',
 'policy_reference':{'url':'https://www.nature.com/nature-portfolio/editorial-policies/ai','access_date':'2026-09-02'},
 'exclusions':'No new exclusions. GSE75097 uses the already frozen primary sample set. Availability matrix includes all 250 candidates.',
 'clinical_crosswalk':'No cohort-average values substituted for individual records',
 'placement':'After the author framework/detail figure, or as a resource figure in Supplementary Information; final figure numbering remains author-controlled.'}
(O/'qa/Fig1_revision_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')

fig.canvas.draw();require_matplotlib_panel_alignment(fig,axes=[a,b,c],panel_ids=['a','b','c'],json_out=O/'qa/Fig1_alignment.json',strict=True)
# Preserve the author-reviewed previous schematic exactly once before replacing it.
archive=O/'archive/Fig1_text_schematic';archive.mkdir(parents=True,exist_ok=True)
stem='Fig1_cohort_to_biological_evidence'
for p in list((O/'main_figures').glob(stem+'.*'))+list((O/'panels/Fig1').glob('*')):
    dest=archive/p.name
    if not dest.exists():shutil.copy2(p,dest)
fig.savefig(O/'main_figures'/(stem+'.pdf'))
fig.savefig(O/'main_figures'/(stem+'.svg'))
fig.savefig(O/'main_figures'/(stem+'.png'),dpi=600)
fig.savefig(O/'main_figures'/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
for i,ax in enumerate([a,b,c]):
    for other in [a,b,c]:other.set_visible(other is ax)
    for t in fig.texts:t.set_visible(False)
    fig.canvas.draw();require_matplotlib_panel_alignment(fig,axes=[ax],json_out=O/'qa'/f'Fig1{chr(97+i)}_alignment.json',strict=True)
    bb=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(.13)
    for ext,dpi in [('pdf',None),('svg',None),('png',600),('tiff',600)]:
        kw={'bbox_inches':bb}
        if dpi:kw['dpi']=dpi
        if ext=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(O/'panels/Fig1'/f'Fig1{chr(97+i)}.{ext}',**kw)
plt.close(fig)
