"""Complete-universe inference of existing Stage31 Full adapters, no training.

Reconstruct each fold's discovery features; first reproduce archived logits.
The resulting fold/seed stability is conditional on the original candidate set.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'model/osa_adapter'))


def summarize_shortlists(base,out):
    scores=pd.read_csv(out/'complete_primary_logits.tsv.gz',sep='\t').set_index('pair_id')
    ranks=scores.rank(ascending=False,method='first')
    original=pd.read_csv(base/'22_consolidated_evidence_triage_osa_v4/pair_evidence_matrix.tsv.gz',sep='\t',low_memory=False).set_index('pair_id').loc[ranks.index]
    front=original.pareto_front.astype(str).str.lower().eq('true')
    records=[]
    for unit in ranks.columns:
        for fraction in [.01,.05,.10]:
            selected=front&(ranks[unit]<=int(np.ceil(len(ranks)*fraction)))
            records.extend(dict(unit=unit,fraction=fraction,pair_id=pid,rank=int(ranks.loc[pid,unit])) for pid in ranks.index[selected])
    membership=pd.DataFrame(records); membership.to_csv(out/'fixed_evidence_shortlist_membership.tsv',sep='\t',index=False)
    summary=original.loc[front,['gene_A','gene_B','triage_osa_rank_primary_250']].copy()
    for fraction in [.01,.05,.10]:
        counts=membership[membership.fraction==fraction].pair_id.value_counts()
        summary['shortlist_top%d_frequency'%int(fraction*100)]=summary.index.map(counts).fillna(0)/len(ranks.columns)
    summary.to_csv(out/'fixed_evidence_shortlist_frequency.tsv',sep='\t')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default=str(ROOT)); p.add_argument('--device',default='cpu')
    p.add_argument('--summarize-existing',action='store_true'); args=p.parse_args()
    if args.summarize_existing:
        base=Path(args.root)/'results/discovery_preprocess_v2'; summarize_shortlists(base,base/'42_complete_candidate_stability_v1'); return
    import torch
    from triage_osa_model import TRIAGEOSAModel
    from osa_context_data import load_discovery_inputs, compute_node_evidence, build_pair_context, assemble_input_matrix
    from train_triage_osa import model_predictions
    torch.set_num_threads(1)
    root=Path(args.root); base=root/'results/discovery_preprocess_v2'
    source=base/'31_osa_controlled_experiments_v1/osa_ablation_suite'
    out=base/'42_complete_candidate_stability_v1'
    if out.exists(): raise RuntimeError('Output exists; preserve previous results')
    pairs=pd.read_csv(base/'08_pair_inference_sensitivity_v1/pair_scores.tsv.gz',sep='\t',low_memory=False)
    proteins=sorted(set(pairs.protein_A)|set(pairs.protein_B))
    expression,metadata,_=load_discovery_inputs(root,proteins)
    assign=pd.read_csv(source/'fold_assignments.tsv',sep='\t')
    primary=pd.read_csv(base/'22_consolidated_evidence_triage_osa_v4/pair_evidence_matrix.tsv.gz',sep='\t',low_memory=False)
    primary_idx=np.where(pairs.pair_id.isin(set(primary.pair_id)))[0]
    if len(primary_idx)!=31125: raise RuntimeError('Incomplete primary universe')
    out.mkdir(); allscores={}; checks=[]
    for fold in range(1,5):
        def trainids(dataset):
            return assign.loc[(assign.dataset==dataset)&(assign.fold!=fold-1),'unit_id'].astype(str).tolist()
        node=compute_node_evidence(expression,metadata,proteins,trainids('GSE75097'),trainids('GSE71356'))
        context=build_pair_context(pairs,node,information_fusion=True)
        features=assemble_input_matrix(pairs,context)
        for seed in [20260716,20260717,20260718]:
            print('Complete ranking: fold=%d seed=%d'%(fold,seed),flush=True)
            unit=source/('seed_%d/full/fold_%d'%(seed,fold))
            ck=torch.load(str(unit/'adapter.pt'),map_location='cpu',weights_only=False)
            model=TRIAGEOSAModel.from_config(ck['model_config']); model.load_state_dict(ck['state_dict'],strict=True); model.to(args.device).eval()
            logits=model_predictions(model,features,np.arange(len(pairs)),4096,torch.device(args.device),torch)
            full=pd.Series(logits,index=pairs.pair_id)
            old=pd.read_csv(unit/'predictions.tsv.gz',sep='\t').set_index('pair_id')
            diff=float(np.max(abs(full.loc[old.index].to_numpy()-old.logit.to_numpy())))
            targetdiff=float(np.max(abs(context.set_index(pairs.pair_id).loc[old.index,'context_anchor'].to_numpy()-old.weak_target.to_numpy())))
            if diff>5e-5 or targetdiff>1e-6: raise RuntimeError('Archived reconstruction mismatch: %s %s'%(diff,targetdiff))
            name='fold%d_seed%d'%(fold,seed); allscores[name]=full.loc[pairs.iloc[primary_idx].pair_id].to_numpy()
            checks.append(dict(unit=name,archived_logit_max_difference=diff,target_max_difference=targetdiff))
    scores=pd.DataFrame(allscores,index=pairs.iloc[primary_idx].pair_id); scores.index.name='pair_id'
    scores.to_csv(out/'complete_primary_logits.tsv.gz',sep='\t',compression='gzip')
    ranks=scores.rank(ascending=False,method='first'); rows=[]
    for i,a in enumerate(scores.columns):
        for b in scores.columns[i+1:]:
            for fraction in [.01,.05,.10]:
                k=int(np.ceil(len(scores)*fraction)); aa=set(ranks.index[ranks[a]<=k]); bb=set(ranks.index[ranks[b]<=k])
                rows.append(dict(unit_A=a,unit_B=b,comparison='same_fold_seed' if a.split('_')[0]==b.split('_')[0] else 'cross_fold',
                    top_fraction=fraction,rank_rho=float(spearmanr(scores[a],scores[b])[0]),jaccard=len(aa&bb)/len(aa|bb)))
    pd.DataFrame(rows).to_csv(out/'ranking_stability.tsv',sep='\t',index=False)
    summary=primary.set_index('pair_id')[['gene_A','gene_B','triage_osa_rank_primary_250','pareto_front']].reindex(ranks.index)
    summary['rank_median']=ranks.median(axis=1); summary['rank_min']=ranks.min(axis=1); summary['rank_max']=ranks.max(axis=1)
    for fraction in [.01,.05,.10]: summary['top%d_frequency'%int(fraction*100)]=(ranks<=int(np.ceil(len(ranks)*fraction))).mean(axis=1)
    summary.to_csv(out/'candidate_rank_stability.tsv.gz',sep='\t',compression='gzip')
    pd.DataFrame(checks).to_csv(out/'archive_reconstruction_checks.tsv',sep='\t',index=False)
    summarize_shortlists(base,out)
    (out/'run_summary.json').write_text(json.dumps(dict(status='PASS_COMPLETE_UNIVERSE_INFERENCE',units=12,pairs=31125,
        retrained=False,interpretation='conditional on original candidates; not full participant-resampling pipeline or independent clinical replication'),indent=2),encoding='utf-8')

if __name__=='__main__': main()
