"""Audit published-panel predictions and evidence-removal stability; no retraining.

Preserves existing result directories. Python 3.7+, numpy/pandas/scipy.
"""
import argparse
import ast
import gzip
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'model/osa_adapter'))
import osa_published_comparison as pc


def read(p): return pd.read_csv(p,sep='\t',low_memory=False)
def save(x,p): pd.DataFrame(x).to_csv(p,sep='\t',index=False)
def truth(x): return x.astype(str).str.lower().isin(['true','yes','1'])


def checked_pareto(root):
    # Reuse the exact canonical algorithm without importing its GPU dependencies.
    p=root/'model/osa_adapter/predict_triage_osa.py'
    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pareto_front_mask')
    code=ast.Module(body=[fn],type_ignores=[]) if sys.version_info>=(3,8) else ast.Module(body=[fn])
    scope={'np':np}; exec(compile(code,str(p),'exec'),scope)
    return scope['pareto_front_mask']


def component_sizes(frame):
    links={}
    for a,b in zip(frame.protein_A,frame.protein_B):
        links.setdefault(a,set()).add(b); links.setdefault(b,set()).add(a)
    unseen=set(links); sizes=[]
    while unseen:
        todo=[min(unseen)]; seen=set()
        while todo:
            a=todo.pop()
            if a in seen: continue
            seen.add(a); todo.extend(links[a]-seen)
        unseen-=seen; sizes.append(len(seen))
    return sorted(sizes,reverse=True)


def evidence_stability(root,out):
    p=root/'results/discovery_preprocess_v2/22_consolidated_evidence_triage_osa_v4'
    x=read(p/'pair_evidence_matrix.tsv.gz')
    if len(x)!=31125 or x.pair_id.duplicated().any(): raise RuntimeError('Unexpected primary universe')
    front=checked_pareto(root)
    axes=[c[:-11] for c in x.columns if c.endswith('_supportive')]
    if len(axes)!=9: raise RuntimeError('Expected nine support axes')
    support=np.column_stack([truth(x[a+'_supportive']) for a in axes]).astype(int)
    available=np.column_stack([truth(x[a+'_evaluable']) for a in axes]).astype(int)
    if np.any(support>available): raise RuntimeError('Support without availability')
    xy=x[['structural_percentile','osa_context_percentile']].to_numpy(float)
    base=front(np.column_stack([xy,support.sum(1)/9.]))
    if not np.array_equal(base,truth(x.pareto_front)): raise RuntimeError('Canonical Pareto does not reproduce archive')
    rank=x.triage_osa_rank_primary_250.to_numpy(float)
    original=set(x.loc[base&(rank<=1557),'pair_id'])
    if len(original)!=15: raise RuntimeError('Expected 15 archived module edges')
    groups={'CPAP':['cpap_GSE133601','cpap_GSE49800','adipose_CPAP'],
            'adult_tissues':['adipose_state','independent_GSE38792','independent_GSE293433'],
            'cell_context':['pediatric_scRNA'],'proteomics':['pediatric_proteome'],
            'known_network':['known_network']}
    settings=[('all',[])]+[('drop_'+a,[a]) for a in axes]+[('drop_family_'+k,v) for k,v in groups.items()]
    rows=[]; memberships=[]
    for name,drop in settings:
        keep=np.array([a not in drop for a in axes]); count=support[:,keep].sum(1); avail=available[:,keep].sum(1)
        for normalization in ['fixed_nine','available_axes']:
            breadth=count/9. if normalization=='fixed_nine' else np.divide(count,avail,out=np.zeros(len(x)),where=avail>0)
            mask=front(np.column_stack([xy,breadth]))
            for fraction in [.01,.05,.10]:
                cutoff=int(np.ceil(fraction*len(x))); selected=mask&(rank<=cutoff); edges=x.loc[selected]
                ids=set(edges.pair_id); union=ids|original
                rows.append(dict(setting=name,normalization=normalization,top_fraction=fraction,
                    pareto_pairs=int(mask.sum()),shortlist_pairs=len(ids),retained_original=len(ids&original),
                    original_retention=len(ids&original)/15.,jaccard_vs_original=len(ids&original)/len(union),
                    component_sizes='|'.join(map(str,component_sizes(edges))),
                    nodes=len(set(edges.protein_A)|set(edges.protein_B))))
                memberships.extend(dict(setting=name,normalization=normalization,top_fraction=fraction,
                    pair_id=pid,original_edge=pid in original) for pid in sorted(ids))
    save(rows,out/'evidence_removal_summary.tsv'); save(memberships,out/'evidence_removal_membership.tsv')
    baseline=x[x.pair_id.isin(original)][['pair_id','gene_A','gene_B','triage_osa_rank_primary_250',
        'evidence_evaluable_axis_count','evidence_supportive_axis_count']].copy()
    mem=pd.DataFrame(memberships)
    selected=mem[(mem.normalization=='fixed_nine')&(mem.top_fraction==.05)&mem.setting.str.startswith('drop_family_')]
    baseline['family_deletion_retention_count']=baseline.pair_id.map(selected.pair_id.value_counts()).fillna(0).astype(int)
    baseline['family_deletions']=len(groups)
    save(baseline,out/'original_candidate_evidence_dependence.tsv')
    # Same-fold seed comparisons are conditional on that fold's available pair set.
    predroot=root/'results/discovery_preprocess_v2/31_osa_controlled_experiments_v1/osa_ablation_suite'
    seedrows=[]; coverage=[]
    for fold in range(1,5):
        preds=[]
        for seed in [20260716,20260717,20260718]:
            f=predroot/('seed_%d/full/fold_%d/predictions.tsv.gz'%(seed,fold))
            z=read(f); z=z[z.pair_id.isin(set(x.pair_id))].set_index('pair_id')
            coverage.append(dict(fold=fold,seed=seed,pairs_available=len(z),complete_primary_universe=len(z)==len(x)))
            preds.append((seed,z))
        common=sorted(set.intersection(*[set(z.index) for _,z in preds]))
        for i in range(len(preds)):
            for j in range(i+1,len(preds)):
                s,a=preds[i]; t,b=preds[j]; a=a.loc[common]; b=b.loc[common]
                for fraction in [.01,.05,.10]:
                    k=int(np.ceil(len(common)*fraction)); ia=set(a.logit.nlargest(k).index); ib=set(b.logit.nlargest(k).index)
                    seedrows.append(dict(fold=fold,seed_A=s,seed_B=t,common_pairs=len(common),top_fraction=fraction,
                        rank_rho=float(spearmanr(a.logit,b.logit)[0]),jaccard=len(ia&ib)/len(ia|ib),
                        interpretation='same-fold seed stability on saved pair subset; not complete ranking stability'))
    save(seedrows,out/'conditional_seed_stability.tsv'); save(coverage,out/'prediction_universe_coverage.tsv')


def metrics_audit(root,out):
    base=root/'results/discovery_preprocess_v2'
    folders=[base/'36_published_osa_comparison_v2']+list((base/'36_published_osa_comparison_crossparticipant_v1').glob('*'))
    rows=[]
    for folder in folders:
        p=folder/'out_of_subject_predictions.tsv.gz'
        if not p.is_file(): continue
        z=read(p); reported=read(folder/'sample_endpoint_results.tsv')
        for (dataset,panel),g in z.groupby(['dataset','panel_id']):
            if g.sample_id.duplicated().any(): raise RuntimeError('Duplicate sample/panel')
            y=g.label.to_numpy(int); score=g.probability.to_numpy(float)
            if not np.isfinite(score).all() or not np.isin(y,[0,1]).all() or np.any((score<0)|(score>1)):
                raise RuntimeError('Invalid labels/probabilities')
            met=pc._metrics(y,score); old=reported[(reported.dataset==dataset)&(reported.panel_id==panel)].iloc[0]
            ties=any(len(set(v.label))>1 for _,v in g.groupby('probability'))
            rows.append(dict(source=folder.name,dataset=dataset,panel_id=panel,AUROC=met['AUROC'],
                AUPRC=met['AUPRC'],AUROC_recalculation_delta=met['AUROC']-old.AUROC,
                AUPRC_recalculation_delta=met['AUPRC']-old.AUPRC,cross_label_ties=ties,
                extreme_AUC=met['AUROC']<.1 or met['AUROC']>.95,n_samples=len(g),n_subjects=g.subject_id.nunique()))
    save(rows,out/'prediction_metric_audit.tsv')


def convergence_audit(root,out):
    gpl=pc._gpl6244_map(root/'annotations/GPL6244.annot.gz')
    entrez=pc._entrez_map(root/'annotations/hgnc_complete_set_20260714.tsv')
    archive=read(root/'results/discovery_preprocess_v2/36_published_osa_comparison_v2/out_of_subject_predictions.tsv.gz')
    rows=[]; preds=[]; source=[]; mappings=[]
    for spec in pc.DATASETS:
        print('Audit raw matrix and published panels:',spec['dataset'],flush=True)
        genes=pc._gene_matrix(pc._series_matrix(root/spec['matrix']),entrez if spec['platform']=='ENTREZ_CUSTOM' else gpl,spec['platform'])
        meta=pc._metadata(root,spec['dataset'],spec['subset']); meta=meta.set_index('sample_id')
        if not set(meta.index)<=set(genes.index): raise RuntimeError('Missing metadata samples')
        genes=genes.loc[meta.index]; y=(meta[spec['label_column']].str.lower()==spec['positive'].lower()).to_numpy(int)
        groups=meta.subject_id.fillna(pd.Series(meta.index,index=meta.index)).to_numpy(str)
        with gzip.open(str(root/spec['matrix']),'rt',encoding='utf-8') as f:
            lines=[line.rstrip().split('\t') for line in f if line.startswith('!Sample_')]
        accessions=next(line[1:] for line in lines if line[0]=='!Sample_geo_accession')
        chars=[line for line in lines if line[0] in ['!Sample_title','!Sample_characteristics_ch1']]
        for i,s in enumerate(accessions):
            s=s.strip('"')
            if s in meta.index:
                source.append(dict(dataset=spec['dataset'],sample_id=s,subject_id=groups[list(meta.index).index(s)],
                    metadata_label=meta.loc[s,spec['label_column']],geo_fields=' | '.join(c[i+1].strip('"') for c in chars)))
        for panel in [p for p in pc.PUBLISHED if p['task']==spec['task']]:
            names=[g for g in panel['genes'].split(',') if g in genes.columns]
            mappings.append(dict(dataset=spec['dataset'],panel_id=panel['panel_id'],requested=panel['genes'],available=','.join(names)))
            if len(names)<2: continue
            x=genes[names].to_numpy(float); optpred=np.full(len(y),np.nan); maxgrad=0.; maxdelta=0.
            old=archive[(archive.dataset==spec['dataset'])&(archive.panel_id==panel['panel_id'])].set_index('sample_id').loc[meta.index]
            if not np.array_equal(old.label,y): raise RuntimeError('Archive labels mismatch metadata')
            if not np.array_equal(old.subject_id.to_numpy(str),groups): raise RuntimeError('Archive groups mismatch metadata')
            for group in sorted(set(groups)):
                test=groups==group; train=~test
                fitted=pc._fit_ridge_logistic(x[train],y[train]); beta,mean,sd=fitted
                xx=(np.where(np.isfinite(x[train]),x[train],mean)-mean)/sd
                design=np.column_stack([np.ones(train.sum()),xx]); yy=y[train]; n=len(yy)
                w=np.where(yy==1,n/(2.*sum(yy==1)),n/(2.*sum(yy==0)))
                def objective(b):
                    logits=design.dot(b)
                    loss=np.mean(w*(np.logaddexp(0,logits)-yy*logits))+.5*np.sum(b[1:]**2)/n
                    grad=design.T.dot(w*(pc._sigmoid(logits)-yy))/n; grad[1:]+=b[1:]/n
                    return loss,grad
                fitted_opt=minimize(objective,beta,jac=True,method='L-BFGS-B',options={'gtol':1e-10,'ftol':1e-14,'maxiter':5000})
                if np.max(abs(objective(fitted_opt.x)[1]))>1e-6: raise RuntimeError('Independent optimizer not converged')
                optpred[test]=pc._predict((fitted_opt.x,mean,sd),x[test])
                reproduced=pc._predict(fitted,x[test]); maxdelta=max(maxdelta,float(np.max(abs(reproduced-old.probability.to_numpy()[test]))))
                maxgrad=max(maxgrad,float(np.max(abs(objective(beta)[1]))))
            rows.append(dict(dataset=spec['dataset'],panel_id=panel['panel_id'],n_samples=len(y),n_subjects=len(set(groups)),
                original_AUC=pc._roc_auc(y,old.probability),converged_same_objective_AUC=pc._roc_auc(y,optpred),
                max_probability_difference=float(np.max(abs(optpred-old.probability.to_numpy()))),
                original_max_gradient=maxgrad,archive_reproduction_max_difference=maxdelta))
            preds.extend(dict(dataset=spec['dataset'],panel_id=panel['panel_id'],sample_id=s,label=int(y[i]),
                original_probability=float(old.probability.iloc[i]),converged_probability=float(optpred[i])) for i,s in enumerate(meta.index))
    save(rows,out/'published_optimizer_audit.tsv'); save(preds,out/'published_optimizer_predictions.tsv')
    save(source,out/'geo_label_source_crosswalk.tsv'); save(mappings,out/'published_gene_mapping.tsv')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default=str(ROOT)); args=p.parse_args()
    root=Path(args.root); out=root/'results/discovery_preprocess_v2/41_public_evidence_closeout_v1'
    if out.exists(): raise RuntimeError('Output exists; preserve previous audit')
    out.mkdir()
    (out/'analysis_contract.json').write_text(json.dumps(dict(
        scope='existing public data audit and descriptive stability; no model selection',
        original_outputs_modified=False,top_fractions=[.01,.05,.10],
        evidence_normalization=['fixed_nine','available_axes'],
        optimization='same class-balanced ridge objective, independent converged solver; diagnostic only',
        no_empty_shortlist_fallback=True),indent=2),encoding='utf-8')
    metrics_audit(root,out); evidence_stability(root,out); convergence_audit(root,out)
    (out/'run_summary.json').write_text(json.dumps(dict(status='PASS_AUDIT_EXECUTION',
        manuscript_ready=False,complete_ranking_stability='separate Stage42 analysis; not measured by this script',
        biological_advantage_established=False),indent=2),encoding='utf-8')
    print('Output:',out)

if __name__=='__main__': main()
