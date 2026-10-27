"""Common-model leave-pair-out sensitivity for the two small OSA cohorts.

All comparisons use one control and one case scored by the SAME training fit.
This post hoc diagnostic is not a replacement selected for favourable AUROC.
CPAP grouped LOSO already tests both states together and is not changed here.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'model/osa_adapter'))
import osa_published_comparison as pc


def fit(x,y):
    mean=np.nanmean(x,axis=0); sd=np.nanstd(x,axis=0); sd[sd<1e-8]=1.
    xx=(np.where(np.isfinite(x),x,mean)-mean)/sd
    a=np.column_stack([np.ones(len(y)),xx]); n=len(y)
    w=np.where(y==1,n/(2.*sum(y==1)),n/(2.*sum(y==0)))
    def objective(beta):
        logits=a.dot(beta); grad=a.T.dot(w*(pc._sigmoid(logits)-y))/n; grad[1:]+=beta[1:]/n
        return np.mean(w*(np.logaddexp(0,logits)-y*logits))+.5*np.sum(beta[1:]**2)/n,grad
    res=minimize(objective,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'gtol':1e-10,'ftol':1e-14,'maxiter':3000})
    if max(abs(objective(res.x)[1]))>1e-6: raise RuntimeError('Convergence failure')
    return res.x,mean,sd


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default=str(ROOT)); args=p.parse_args()
    root=Path(args.root); base=root/'results/discovery_preprocess_v2'; out=base/'43_auc_protocol_diagnostic_v1'
    if out.exists(): raise RuntimeError('Preserve previous output')
    out.mkdir(); (out/'analysis_contract.json').write_text(json.dumps(dict(
        purpose='diagnose cross-fit score comparability; all panels and methods',
        evaluation='exhaustive leave-one-case-and-one-control-out; no patient split across train/test',
        solver='converged class-balanced ridge logistic, penalty=1, same as original objective',
        uncertainty='descriptive only; pair comparisons share subjects; no pair-level confidence interval',
        original_results_overwritten=False),indent=2),encoding='utf-8')
    rawrank=pd.read_csv(root/pc.RANKING,sep='\t',low_memory=False)
    controls=pd.read_csv(base/'38_external_strong_controls_v1/rankings_primary_250.tsv.gz',sep='\t').set_index('pair_id')
    node=pd.read_csv(root/pc.NODE_EVIDENCE,sep='\t'); directions_map=dict(zip(node.UniProt_accession,node.adult_signed))
    method_data={}
    for method in ['Full','Static_constrained','Direct_anchor']:
        rank=rawrank.copy(); rank['score']=rank.pair_id.map(controls[method]); rank=rank.sort_values('score',ascending=False,kind='mergesort')
        genes=[]; directions={}; edges=[]
        for r in rank.itertuples(index=False):
            for g,pr in [(r.gene_A,r.protein_A),(r.gene_B,r.protein_B)]:
                if g not in genes: genes.append(g)
                if g not in directions: directions[g]=directions_map.get(pr,0.)
            edges.append((r.gene_A,r.gene_B,r.score))
        method_data[method]=(genes,directions,edges)
    gpl=pc._gpl6244_map(root/'annotations/GPL6244.annot.gz'); rows=[]; detailed=[]
    for spec in pc.DATASETS[:2]:
        gene=pc._gene_matrix(pc._series_matrix(root/spec['matrix']),gpl,spec['platform'])
        meta=pc._metadata(root,spec['dataset'],spec['subset']).set_index('sample_id'); gene=gene.loc[meta.index]
        if meta.subject_id.duplicated().any(): raise RuntimeError('Expected one sample per participant')
        y=(meta.disease_group.str.lower()=='osa').to_numpy(int)
        for pub in [p for p in pc.PUBLISHED if p['task']=='OSA_status']:
            available=[g for g in pub['genes'].split(',') if g in gene.columns]
            representations=[('Published','genes',available,None,None)]
            for method,(order,directions,edges) in method_data.items():
                selected=[g for g in order if g in gene.columns][:len(available)]
                representations.extend([(method,'genes',selected,directions,edges),(method,'pair_summary',selected,directions,edges)])
            for method,mode,names,directions,edges in representations:
                x=gene[names].to_numpy(float)
                def predict(train,test):
                    a=x[train]; b=x[test]
                    if mode=='pair_summary':
                        m=np.nanmean(a,axis=0); s=np.nanstd(a,axis=0); s[s<1e-8]=1.
                        a=pc._pair_summary_features((np.where(np.isfinite(a),a,m)-m)/s,names,directions,edges,'OSA_status')
                        b=pc._pair_summary_features((np.where(np.isfinite(b),b,m)-m)/s,names,directions,edges,'OSA_status')
                    return pc._predict(fit(a,y[train]),b)
                loo=np.empty(len(y)); idx=np.arange(len(y))
                for i in idx: loo[i]=predict(idx!=i,idx==i)[0]
                wins=[]
                for a in np.where(y==1)[0]:
                    for b in np.where(y==0)[0]:
                        test=np.array([a,b]); train=(idx!=a)&(idx!=b); prob=predict(train,test)
                        win=float(prob[0]>prob[1])+.5*float(prob[0]==prob[1]); wins.append(win)
                        detailed.append(dict(dataset=spec['dataset'],panel=pub['panel_id'],method=method,representation=mode,
                            case_sample=meta.index[a],control_sample=meta.index[b],case_probability=prob[0],control_probability=prob[1],concordance=win))
                rows.append(dict(dataset=spec['dataset'],panel=pub['panel_id'],method=method,representation=mode,
                    genes=','.join(names),n_subjects=len(y),n_cases=int(sum(y)),n_controls=int(sum(1-y)),
                    pooled_LOSO_AUC=pc._roc_auc(y,loo),common_model_LPO_AUC=float(np.mean(wins)),
                    n_dependent_comparisons=len(wins)))
                print(spec['dataset'],pub['panel_id'],method,mode,'LOSO/LPO',rows[-1]['pooled_LOSO_AUC'],rows[-1]['common_model_LPO_AUC'],flush=True)
    pd.DataFrame(rows).to_csv(out/'all_panel_protocol_sensitivity.tsv',sep='\t',index=False)
    pd.DataFrame(detailed).to_csv(out/'common_model_pair_predictions.tsv.gz',sep='\t',index=False,compression='gzip')
    (out/'run_summary.json').write_text(json.dumps(dict(status='PASS_PROTOCOL_SENSITIVITY',n_settings=len(rows),
        interpretation='post hoc diagnostic, no protocol selection and no independent pair replicates'),indent=2),encoding='utf-8')

if __name__=='__main__': main()
