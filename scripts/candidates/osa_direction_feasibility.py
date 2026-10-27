"""Discovery-only direction feasibility; no external outcomes or backbone update.

Fixed repeated two-fold splits, train-only protein selection and normalization.
Expression prediction from phenotype is NOT clinical phenotype prediction.
Python 3.7+, numpy/pandas/scipy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEEDS=(20260921,20260922,20260923,20260924,20260925)


def split(meta,seed):
    rng=np.random.RandomState(seed); a=[]; b=[]
    for _,g in meta.groupby('study_group',sort=True):
        ids=sorted(g.sample_id.astype(str)); rng.shuffle(ids)
        a.extend(ids[::2]); b.extend(ids[1::2])
    return a,b


def effect(z,values,binary=False):
    v=np.asarray(values,float); valid=np.isfinite(v)
    if binary:
        if (v[valid]==0).sum()<2 or (v[valid]==1).sum()<2:
            raise RuntimeError('At least two controls/cases required for each effect estimate')
        return z[valid&(v==1)].mean(0)-z[valid&(v==0)].mean(0)
    v=v[valid]; v=v-v.mean()
    if len(v)<4 or np.sum(v*v)<1e-8: raise RuntimeError('Phenotype has insufficient variation')
    return v.dot(z[valid])/np.sum(v*v)


def fit_predict(xtrain,xtest,y,penalty):
    mean=np.nanmean(xtrain,axis=0)
    if not np.isfinite(mean).all(): raise RuntimeError('No observed training values for a covariate')
    xtrain=np.where(np.isfinite(xtrain),xtrain,mean)
    xtest=np.where(np.isfinite(xtest),xtest,mean)
    sd=xtrain.std(0); sd=np.where(sd>1e-8,sd,1.)
    a=np.column_stack([np.ones(len(xtrain)),(xtrain-mean)/sd])
    b=np.column_stack([np.ones(len(xtest)),(xtest-mean)/sd])
    reg=np.eye(a.shape[1])*penalty; reg[0,0]=0
    beta=np.linalg.solve(a.T.dot(a)+reg,a.T.dot(y))
    return b.dot(beta)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    args=parser.parse_args(); root=Path(args.root).resolve()
    base=root/'results/discovery_preprocess_v2'
    out=base/'40_direction_feasibility_v1'
    if out.exists(): raise RuntimeError('Output already exists; preserve previous feasibility results')
    md=pd.read_csv(root/'metadata/discovery_patient_information.tsv',sep='\t')
    md=md[(md.dataset=='GSE75097')&md.study_group.isin(['PS','MSO','VSO'])].copy()
    if len(md)!=32 or md.sample_id.duplicated().any(): raise RuntimeError('Unexpected discovery participants')
    md=md.set_index('sample_id',drop=False)
    raw=pd.read_csv(base/'03_mapping/GSE75097/protein_expression_primary_samples.tsv.gz',sep='\t').drop_duplicates('UniProt_accession').set_index('UniProt_accession')
    y=raw.loc[:,md.index].T.apply(pd.to_numeric,errors='coerce')
    existing=set(pd.read_csv(base/'09_pair_context_ranking_triage_osa_v4/full_discovery_node_evidence.tsv',sep='\t').UniProt_accession)
    variables=pd.DataFrame(index=md.index)
    variables['age']=pd.to_numeric(md.age_years,errors='coerce')
    sex=md.sex.astype(str).str.lower(); variables['sex']=sex.map({'male':1.,'female':0.,'m':1.,'f':0.})
    variables['OSA']=(md.study_group!='PS').astype(float)
    variables['severity']=md.study_group.map({'PS':0.,'MSO':1.,'VSO':2.})
    variables['AHI']=pd.to_numeric(md.AHI_events_h,errors='coerce')
    out.mkdir()
    contract=dict(seeds=list(SEEDS),split='stratified two-fold with both directions; repeated folds are dependent',
        primary='train-selected 250 proteins; heldout expression MSE vs training mean; ridge penalty=1',
        sensitivity='all measurable proteins and old candidates; ridge penalty=10',
        target='expression conditional on observed phenotype, NOT diagnosis/prognosis',
        selection='within training participants for each axis',external_outcomes_read=False,
        IH='not used: 4 participants and sparse candidate coverage; no artificial imputation of biological effects',
        prior_inspection='discovery data previously inspected; exploratory feasibility, not independent confirmation')
    (out/'analysis_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    rows=[]; memberships=[]; selection=[]
    for seed in SEEDS:
        halves=split(md,seed)
        for direction in (0,1):
            train,test=halves[direction],halves[1-direction]
            assert not set(train)&set(test)
            for role,ids in [('train',train),('test',test)]:
                memberships.extend(dict(seed=seed,direction=direction,role=role,sample_id=s) for s in ids)
            yt=y.loc[train].to_numpy(float); ye=y.loc[test].to_numpy(float)
            keep=np.isfinite(yt).all(0)&(np.nanstd(yt,axis=0)>1e-8)
            proteins=np.asarray(y.columns)[keep]; yt=yt[:,keep]; ye=ye[:,keep]
            mean=yt.mean(0); sd=yt.std(0,ddof=1)
            zt=(yt-mean)/sd; ze=(ye-mean)/sd
            available=np.isfinite(ze).all(0)
            for axis in ('OSA','severity','AHI'):
                a=effect(zt,variables.loc[train,axis],axis=='OSA')
                b=effect(ze,variables.loc[test,axis],axis=='OSA')
                order=np.argsort(-abs(a),kind='mergesort')[:250]
                panels={'train_selected_250':order,'all_measurable':np.arange(len(proteins)),
                        'old_candidates_conditional':np.where(np.isin(proteins,list(existing)))[0]}
                selection.extend(dict(seed=seed,direction=direction,axis=axis,protein=p,train_rank=k+1)
                                 for k,p in enumerate(proteins[order]))
                features={'demographics':['age','sex'],'axis_only':[axis],'demographics_plus_axis':['age','sex',axis]}
                preds={'training_mean':np.zeros_like(ze)}
                for name,cols in features.items():
                    for penalty in (1.,10.):
                        preds[name+'_ridge'+str(int(penalty))]=fit_predict(
                            variables.loc[train,cols].to_numpy(float),variables.loc[test,cols].to_numpy(float),zt,penalty)
                for panel,idx in panels.items():
                    idx=idx[available[idx]]
                    baseline=float(np.mean(ze[:,idx]**2))
                    common=dict(seed=seed,direction=direction,axis=axis,panel=panel,n_train=len(train),n_test=len(test),
                        n_proteins=len(idx),signed_effect_rho=float(spearmanr(a[idx],b[idx])[0]),
                        absolute_effect_rho=float(spearmanr(abs(a[idx]),abs(b[idx]))[0]),
                        direction_agreement=float(np.mean(np.sign(a[idx])==np.sign(b[idx]))))
                    for name,pred in preds.items():
                        mse=float(np.mean((pred[:,idx]-ze[:,idx])**2))
                        rows.append(dict(common,model=name,MSE=mse,relative_MSE_improvement=1-mse/baseline))
    frame=pd.DataFrame(rows); frame.to_csv(out/'feasibility_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(memberships).to_csv(out/'participant_splits.tsv',sep='\t',index=False)
    pd.DataFrame(selection).to_csv(out/'training_only_protein_selection.tsv',sep='\t',index=False)
    summary=frame.groupby(['panel','axis','model'])[['MSE','relative_MSE_improvement','signed_effect_rho','absolute_effect_rho','direction_agreement']].mean()
    summary.to_csv(out/'descriptive_summary.tsv',sep='\t')
    (out/'run_summary.json').write_text(json.dumps(dict(status='PASS_FEASIBILITY_EXECUTION',
        external_outcomes_read=False,participants=32,splits=10,rows=len(frame),
        inference='descriptive repeated-split diagnostic; not clinical validation'),indent=2),encoding='utf-8')
    print(summary.loc['train_selected_250'].to_string())
    print('Output:',out)


if __name__=='__main__': main()
