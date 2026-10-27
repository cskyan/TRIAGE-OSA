"""Post hoc, dimension-matched representation audit; original Stage36 is untouched.

Run from the OSA project root. No backbone/adapter fitting or ranking changes.
The contract is saved before evaluating outcomes. All twelve comparisons remain.
"""
from pathlib import Path
import sys, json, hashlib, argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model/osa_adapter'))
import osa_published_comparison as pc


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def node_summary(z, genes, directions, edges, task):
    """Keep two original linear summaries; replace edge disagreement by node MAD.

    Weighted degree is inherited from the same graph, but no neighbour identity
    enters the third statistic. This is a degree-preserving node-summary control,
    not a topology-independent or information-free baseline.
    """
    direction = np.asarray([directions.get(g, 0.) for g in genes], float)
    direction[~np.isfinite(direction)] = 0
    signed = z * direction * (-1 if task == 'CPAP_state' else 1)
    degree = np.zeros(len(genes)); index = {g:i for i,g in enumerate(genes)}
    for a,b,w in edges:
        if a in index and b in index:
            degree[index[a]] += max(float(w), 1e-8)/2
            degree[index[b]] += max(float(w), 1e-8)/2
    if degree.sum() == 0:
        original = pc._pair_summary_features(z, genes, directions, edges, task)
        return original  # identical fallback, no invented edge information
    degree /= degree.sum()
    activity = signed.sum(axis=1)/max(np.abs(direction).sum(),1e-8)
    mean = signed @ degree
    dispersion = np.abs(signed - mean[:,None]) @ degree
    return np.column_stack([activity, mean, dispersion])


def predict(x,y,groups,genes,directions,edges,task,mode):
    pred = np.full(len(y),np.nan)
    for group in sorted(set(groups)):
        test=groups==group; train=~test
        mu=np.nanmean(x[train],axis=0); sd=np.nanstd(x[train],axis=0)
        sd[~np.isfinite(sd)|(sd<1e-8)]=1
        a=(np.where(np.isfinite(x[train]),x[train],mu)-mu)/sd
        b=(np.where(np.isfinite(x[test]),x[test],mu)-mu)/sd
        if mode=='node_pca3':
            _,_,vt=np.linalg.svd(a,full_matrices=False)
            k=min(3,len(vt)); aa=a@vt[:k].T; bb=b@vt[:k].T
            aa=np.pad(aa,((0,0),(0,3-k))); bb=np.pad(bb,((0,0),(0,3-k)))
        else:
            fn=pc._pair_summary_features if mode=='pair3' else node_summary
            aa=fn(a,genes,directions,edges,task); bb=fn(b,genes,directions,edges,task)
        pred[test]=pc._predict(pc._fit_ridge_logistic(aa,y[train]),bb)
    return pred


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/discovery_preprocess_v2/47_representation_capacity_control_v1')
    args=ap.parse_args(); out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/discovery_preprocess_v2/36_published_osa_comparison_v2'
    contract=dict(design='post_hoc_capacity_sensitivity',methods=['pair3','degree_node3','node_pca3'],
        folds='leave_one_participant_out; all visits kept together',
        genes='same discovery-ranked measurable genes and sizes as archived Stage36',
        preprocessing='training-fold mean imputation and population-SD scaling; PCA fit on training only',
        classifier='original class-balanced ridge logistic, penalty=1, 1500 steps, learning_rate=0.05',
        selection='no outcome-driven feature, parameter, panel or method selection',
        uncertainty='5000 paired participant-cluster bootstrap draws of fixed OOF predictions; no refitting',
        interpretation='tests representation sensitivity, not clinical utility or direct PPI accuracy',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/pc.RANKING,ROOT/pc.NODE_EVIDENCE,source/'matched_panel_design.tsv']})
    (out/'analysis_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    designs=pd.read_csv(source/'matched_panel_design.tsv',sep='\t')
    coverage=pd.read_csv(source/'panel_gene_coverage.tsv',sep='\t')
    archived=pd.read_csv(source/'out_of_subject_predictions.tsv.gz',sep='\t')
    directions,edges=pc._triage_graph(ROOT/pc.RANKING,ROOT/pc.NODE_EVIDENCE)
    gpl=pc._gpl6244_map(ROOT/'annotations/GPL6244.annot.gz')
    entrez=pc._entrez_map(ROOT/'annotations/hgnc_complete_set_20260714.tsv')
    rows=[]; allpred=[]; checks=[]
    for spec in pc.DATASETS:
        print('DATASET',spec['dataset'],flush=True)
        matrix=pc._gene_matrix(pc._series_matrix(ROOT/spec['matrix']),entrez if spec['platform']=='ENTREZ_CUSTOM' else gpl,spec['platform'])
        meta=pc._metadata(ROOT,spec['dataset'],spec['subset'])
        samples=[s for s in meta.sample_id.astype(str) if s in matrix.index]
        meta=meta.set_index('sample_id').loc[samples].reset_index(); matrix=matrix.loc[samples]
        y=(meta[spec['label_column']].str.lower()==spec['positive'].lower()).astype(int).to_numpy()
        groups=meta.subject_id.fillna(meta.sample_id).astype(str).to_numpy()
        for _,d in designs[designs.dataset==spec['dataset']].iterrows():
            cov=coverage[(coverage.dataset==spec['dataset'])&(coverage.panel_id==d.TRIAGE_pair_aware_panel)]
            assert len(cov)==1
            genes=cov.iloc[0].available_gene_symbols.split(','); x=matrix[genes].to_numpy(float)
            predictions={mode:predict(x,y,groups,genes,directions,edges,spec['task'],mode) for mode in contract['methods']}
            old=archived[(archived.dataset==spec['dataset'])&(archived.panel_id==d.TRIAGE_pair_aware_panel)].set_index('sample_id').loc[samples]
            err=float(np.max(np.abs(predictions['pair3']-old.probability.to_numpy())))
            assert np.array_equal(y,old.label.to_numpy()) and err<1e-7, (spec['dataset'],d.published_panel,err)
            # Algebra check: original joint activity is the weighted-degree mean.
            z=np.nan_to_num(x)
            assert np.allclose(pc._pair_summary_features(z,genes,directions,edges,spec['task'])[:,:2],node_summary(z,genes,directions,edges,spec['task'])[:,:2])
            checks.append(dict(dataset=spec['dataset'],panel=d.published_panel,max_probability_error=err,degree_identity='PASS'))
            for mode,pred in predictions.items():
                row=dict(dataset=spec['dataset'],task=spec['task'],panel=d.published_panel,method=mode,n_genes=len(genes),n_features=3,n_participants=len(set(groups)))
                row.update(pc._metrics(y,pred)); rows.append(row)
                for i,sample in enumerate(samples): allpred.append(dict(dataset=spec['dataset'],panel=d.published_panel,method=mode,sample_id=sample,subject_id=groups[i],label=int(y[i]),probability=float(pred[i])))
            for mode in ['degree_node3','node_pca3']:
                lo,hi,_=pc._bootstrap_auc_delta(y,predictions['pair3'],predictions[mode],groups)
                rows[-(3-contract['methods'].index(mode))].update(pair_minus_control=pc._roc_auc(y,predictions['pair3'])-pc._roc_auc(y,predictions[mode]),delta_CI_low=lo,delta_CI_high=hi)
            print(spec['dataset'],d.published_panel, {m:round(pc._roc_auc(y,p),4) for m,p in predictions.items()},flush=True)
    pd.DataFrame(rows).to_csv(out/'capacity_results.tsv',sep='\t',index=False)
    pd.DataFrame(allpred).to_csv(out/'participant_predictions.tsv.gz',sep='\t',index=False)
    pd.DataFrame(checks).to_csv(out/'reproduction_checks.tsv',sep='\t',index=False)
    (out/'run_summary.json').write_text(json.dumps(dict(status='PASS',settings=len(checks),models=len(rows),original_rankings_changed=False),indent=2))

if __name__=='__main__': main()
