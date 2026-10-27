"""Exploratory model-ranked pathway-anchored evidence, not whole-pathway enrichment.

Uses existing primary candidates, axes, external node scores and matching blocks.
All methods select the same number of measurable pairs within each axis universe.
No new model fit, no pathway selection by outcomes, no clinical confidence intervals.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'model/osa_adapter'))
from osa_external_strong_controls import bh
AXES=['hypoxia_HIF_ROS','NFkB_innate_inflammation','antigen_lymphocyte_signalling',
      'PI3K_AKT_JAK_STAT_TGFb','proteostasis_ER_stress','PPAR_adipose_lipid_metabolism']
METHODS=['Full','Static_constrained','Direct_anchor','PPI_only']


def finalize(out):
    for name in ['analysis_contract.json','pathway_coverage.tsv','method_pathway_support.tsv',
                 'paired_method_randomization.tsv','selected_pathway_pairs.tsv']:
        if not (out/name).is_file(): raise RuntimeError('Incomplete output: '+name)
    x=pd.read_csv(out/'paired_method_randomization.tsv',sep='\t')
    (out/'run_summary.json').write_text(json.dumps(dict(status='PASS_EXPLORATORY_PATHWAY_LINKAGE',primary_tests=len(x),
        no_new_training=True,positive_tests=int(((x.BH_FDR_all_primary_comparisons<.05)&(x.difference>0)).sum()),
        negative_tests=int(((x.BH_FDR_all_primary_comparisons<.05)&(x.difference<0)).sum())),indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default=str(ROOT)); p.add_argument('--permutations',type=int,default=10000)
    p.add_argument('--finalize-existing',action='store_true',help='Recover a final-summary write after all result tables exist')
    args=p.parse_args()
    root=Path(args.root); base=root/'results/discovery_preprocess_v2'; out=base/'44_candidate_pathway_linkage_v1'
    if args.finalize_existing:
        finalize(out); return
    if out.exists(): raise RuntimeError('Preserve existing analysis')
    out.mkdir(); (out/'analysis_contract.json').write_text(json.dumps(dict(
        primary_fraction=.05,sensitivity_fractions=[.01,.10],axes=AXES,methods=METHODS,
        minimum_axis_nodes=5,minimum_axis_pairs=20,permutations=args.permutations,seed=20260917,
        selection='top fractions within common measurable pathway-anchored pairs; at least one annotated endpoint',
        statistic='mean sqrt(node_score_A * node_score_B), Full minus each control',
        null='shared node permutations within archived coverage/length blocks crossed with pathway membership',
        inference='exploratory matched-node randomization, not participant confidence intervals or physical binding',
        shortlist_used=False,external_outcomes_used_to_train=False),indent=2),encoding='utf-8')
    def read(p): return pd.read_csv(p,sep='\t',low_memory=False)
    frame=read(base/'22_consolidated_evidence_triage_osa_v4/pair_evidence_matrix.tsv.gz')
    ranks=read(base/'38_external_strong_controls_v1/rankings_primary_250.tsv.gz').set_index('pair_id').loc[frame.pair_id]
    protein=read(base/'22_consolidated_evidence_triage_osa_v4/protein_evidence_matrix.tsv').set_index('UniProt')
    nodes=read(base/'21_independent_validation_endpoint_triage_osa_v4/candidate_node_evidence_primary_250.tsv').set_index('UniProt_accession')
    ids=sorted(nodes.index); index={p:i for i,p in enumerate(ids)}; nodes=nodes.loc[ids]
    left=np.array([index[p] for p in frame.protein_A]); right=np.array([index[p] for p in frame.protein_B])
    method_scores={m:ranks[m].to_numpy(float) for m in METHODS[:-1]}; method_scores['PPI_only']=frame.p_PPI.to_numpy(float)
    member_axes=protein.reindex(ids).pathway_axes.fillna('').str.split('|')
    results=[]; coverage=[]; comparisons=[]; selectedrows=[]
    for di,dataset in enumerate(['GSE38792','GSE293433']):
        available=nodes[dataset+'_available'].eq('yes').to_numpy()
        score=nodes[dataset+'_node_score'].fillna(0).to_numpy(float)
        blocks=read(base/('38_external_strong_controls_v1/'+dataset+'_matching_blocks.tsv'))
        blockmap=dict(zip(blocks.UniProt_accession,blocks.block_id))
        for ai,axis in enumerate(AXES):
            inside=np.array([axis in a for a in member_axes]); eligible=available[left]&available[right]&(inside[left]|inside[right])
            count=int(np.sum(inside&available)); n=int(eligible.sum())
            cov=dict(dataset=dataset,axis=axis,measurable_axis_nodes=count,measurable_anchored_pairs=n)
            if count<5 or n<20:
                coverage.append(dict(cov,status='NOT_EVALUABLE_PRESPECIFIED_COVERAGE')); continue
            strata={}
            for i,ident in enumerate(ids):
                if available[i]:
                    if ident not in blockmap: raise RuntimeError('Missing matched block')
                    strata.setdefault((blockmap[ident],bool(inside[i])),[]).append(i)
            groups=[np.array(v) for v in strata.values() if len(v)>1]
            movable=sum(len(g) for g in groups); movable_inside=sum(int(inside[g].sum()) for g in groups)
            coverage.append(dict(cov,status='EVALUATED',movable_nodes=movable,movable_axis_nodes=movable_inside))
            ordered={m:np.where(eligible)[0][np.argsort(-s[eligible],kind='mergesort')] for m,s in method_scores.items()}
            observed=np.sqrt(score[left]*score[right]); sets={}
            for fraction in [.01,.05,.10]:
                k=int(np.ceil(n*fraction))
                for m in METHODS:
                    pick=ordered[m][:k]
                    results.append(dict(cov,method=m,fraction=fraction,selected_pairs=k,
                        selected_nodes=len(set(left[pick])|set(right[pick])),mean_support=float(observed[pick].mean())))
                    if fraction==.05:
                        sets[m]=pick
                        selectedrows.extend(dict(dataset=dataset,axis=axis,method=m,pair_id=frame.iloc[j].pair_id,
                            gene_A=frame.iloc[j].gene_A,gene_B=frame.iloc[j].gene_B,support=observed[j]) for j in pick)
            union=np.unique(np.concatenate(list(sets.values()))); loc={j:i for i,j in enumerate(union)}
            weights=np.zeros((len(METHODS),len(union)))
            for mi,m in enumerate(METHODS): weights[mi,[loc[j] for j in sets[m]]]=1./len(sets[m])
            means=weights.dot(observed[union]); null=np.empty((args.permutations,4)); rng=np.random.RandomState(20260917+di*100+ai)
            for iteration in range(args.permutations):
                perm=score.copy()
                for g in groups: perm[g]=score[rng.permutation(g)]
                null[iteration]=weights.dot(np.sqrt(perm[left[union]]*perm[right[union]]))
            for mi,m in enumerate(METHODS[1:],1):
                delta=means[0]-means[mi]; nd=null[:,0]-null[:,mi]
                pvalue=min(1.,2*min((1+np.sum(nd>=delta))/(len(nd)+1.),(1+np.sum(nd<=delta))/(len(nd)+1.)))
                comparisons.append(dict(dataset=dataset,axis=axis,comparison='Full_minus_'+m,Full=means[0],control=means[mi],
                    difference=delta,randomization_P=pvalue,null_q025=np.quantile(nd,.025),null_q975=np.quantile(nd,.975),
                    interval_kind='matched-node null quantiles, NOT effect confidence intervals',top_fraction=.05))
            print(dataset,axis,'nodes',count,'pairs',n,'done',flush=True)
    if comparisons:
        for r,q in zip(comparisons,bh([r['randomization_P'] for r in comparisons])): r['BH_FDR_all_primary_comparisons']=q
    for name,data in [('pathway_coverage',coverage),('method_pathway_support',results),('paired_method_randomization',comparisons),('selected_pathway_pairs',selectedrows)]:
        pd.DataFrame(data).to_csv(out/(name+'.tsv'),sep='\t',index=False)
    finalize(out)

if __name__=='__main__': main()
