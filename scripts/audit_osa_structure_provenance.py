"""Audit archived sequences and residue evidence against the two structure figures.

No inference, training, ranking or source-file modification is performed.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def rows(path):
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def fasta(path):
    out={}; key=None
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            key=line[1:].split()[0];out[key]=''
        elif key:out[key]+=line.strip()
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--manuscript',type=Path,default=Path('resources/manuscript'))
    args=ap.parse_args();root=args.root
    stage=root/'results/discovery_preprocess_v2/08_pair_inference_primary_v1'
    fig=args.manuscript/'figures/structure_localization_draft'
    out=root/'results/discovery_preprocess_v2/45_structure_sequence_provenance_v1'
    out.mkdir(parents=True,exist_ok=True)
    prep=json.loads((root/'model_assets/osa_sequence_preparation.json').read_text())
    contract=json.loads((stage/'inference_contract.json').read_text())
    run=json.loads((stage/'run_summary.json').read_text())
    checks=[];sources=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    def source(path,expected=None):
        sha=digest(path)
        sources.append(dict(path=str(path),sha256=sha,archived_sha256=expected or '',matches_archive=sha==expected if expected else None))
        if expected:check('archive_hash:'+path.name,sha==expected)
    for name in ['full_fasta','native_fasta','manifest']:
        source(root/'model_assets'/Path(prep[name]['path']).name,prep[name]['sha256'])
    for name in ['pair_scores','evidence_manifest']:source(root/run[name]['path'],run[name]['sha256'])
    source(root/contract['registry'],contract['registry_sha256'])
    ckpt=root/contract['checkpoint']
    if ckpt.exists():source(ckpt,contract['checkpoint_sha256'])
    else:check('checkpoint_available',False,str(ckpt))
    full=fasta(root/'model_assets/osa_frozen_750_full.fasta')
    native=fasta(root/'model_assets/osa_frozen_750_native512.fasta')
    manifest={r['UniProt_accession']:r for r in rows(root/'model_assets/osa_sequence_manifest.tsv')}
    scores={r['pair_id']:r for r in rows(stage/'pair_scores.tsv.gz')}
    shards={r['path']:r for r in rows(stage/'evidence_manifest.tsv')}
    aa3=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))
    summaries=[];mapping_rows=[]
    for gene in ['UBE2Q2','VAMP7']:
        path=fig/'source_data'/f'{gene}_input.json';source(path)
        inp=json.loads(path.read_text());uid=inp['uid'];seq=full[uid];r=scores[inp['pair_id']]
        check(gene+':sequence_manifest',hashlib.sha256(seq.encode()).hexdigest()==manifest[uid]['sequence_sha256'])
        check(gene+':native_prefix',native[uid]==seq[:512])
        check(gene+':render_sequence',seq==inp['sequence'])
        emb_path=root/'model_assets/esm2'/f'{uid}.npy';source(emb_path)
        emb=np.load(emb_path,mmap_mode='r',allow_pickle=False)
        check(gene+':embedding_shape',emb.shape==(len(native[uid]),1280),emb.shape)
        side='A' if r['protein_A']==uid else 'B'
        check(gene+':pair_identity',r['protein_'+side]==uid)
        shard_path=root/r['evidence_shard'];source(shard_path,shards[r['evidence_shard']]['sha256'])
        z=np.load(shard_path,allow_pickle=False);idx=int(r['evidence_row'])
        check(gene+':shard_pair_identity',str(z['pair_id'][idx])==inp['pair_id'])
        off=z[f'p_residue_{side}_offsets'];raw=z[f'p_residue_{side}_values'][off[idx]:off[idx+1]].astype(float)
        off=z[f'topk_{side}_offsets'];top=z[f'topk_{side}_indices'][off[idx]:off[idx+1]]
        check(gene+':score_exact',np.array_equal(raw,np.asarray(inp['scores'],float)))
        check(gene+':stored_npy_exact',np.array_equal(raw,np.load(fig/'source_data'/f'{gene}_residue_scores.npy')))
        check(gene+':topk_indices_exact',np.array_equal(top+1,np.asarray(inp['topk'])), 'stored zero-based indices + 1 versus one-based figure indices; raw={} figure={}'.format(top.tolist(),inp['topk']))
        check(gene+':length_and_crop',len(raw)==len(native[uid])==int(r['used_length_'+side]))
        check(gene+':checkpoint_identity',r['checkpoint_sha256']==contract['checkpoint_sha256'])
        mapping=rows(fig/'source_data'/f'{gene}_coordinate_mapping.tsv')
        pdb_path=fig/'structures'/f"{inp['pdb']}.pdb";source(pdb_path)
        atoms={}
        for line in pdb_path.read_text().splitlines():
            if line.startswith('ENDMDL'):break
            if line.startswith('ATOM  ') and line[21:22]==inp['chain']:
                atoms[(int(line[22:26]),line[26:27].strip())]=line[17:20].strip()
        matched=0;selected=0;mismatches=[]
        for m in mapping:
            a=int(m['author_residue']);ins=m['insertion'];u=int(m['uniprot_residue']) if m['uniprot_residue'] else None
            check(f'{gene}:coordinate_identity:{a}{ins}',atoms.get((a,ins))==m['resname'])
            ok=u is not None and 1<=u<=len(seq) and aa3.get(m['resname'])==seq[u-1]
            check(f'{gene}:mapping_flag:{a}{ins}',ok==(m['sequence_match'].lower()=='true'))
            if ok:
                matched+=1;expected_top=int(u-1 in top);selected+=expected_top
                check(f'{gene}:mapped_score:{a}{ins}',np.isclose(float(m['score']),raw[u-1],rtol=1e-6,atol=1e-9))
                check(f'{gene}:mapped_topk:{a}{ins}',int(m['topk'])==expected_top)
            elif u is not None:mismatches.append(dict(author_residue=a,sequence_position=u,structure_residue=m['resname'],sequence_residue=seq[u-1]))
            mapping_rows.append(dict(gene=gene,accession=uid,pdb=inp['pdb'],chain=inp['chain'],author_residue=a,insertion=ins,sequence_position=u or '',sequence_residue=seq[u-1] if u else '',coordinate_residue=m['resname'],sequence_match=ok,score=raw[u-1] if ok else '',top16=int(u-1 in top) if ok else 0))
        summaries.append(dict(gene=gene,accession=uid,pair_id=inp['pair_id'],sequence_length=len(seq),used_length=len(raw),sequence_sha256=manifest[uid]['sequence_sha256'],render_sequence_identical=seq==inp['sequence'],mapped_residues=matched,mapped_top16=selected,mismatches=mismatches,embedding_sha256=digest(emb_path),checkpoint_sha256=r['checkpoint_sha256']))
    report=dict(status='PASS_ARCHIVED_SEQUENCE_SCORE_COORDINATE_CONSISTENCY' if all(c['passed'] for c in checks) else 'FAIL',cases=summaries,checks=checks,sources=sources,
        remaining_provenance_boundary='The archived inference consumed ESM arrays, not FASTA directly. No per-array historical input-sequence hash has been identified. Current array hashes are recorded, but do not retrospectively certify the ESM generation event. Sequence manifest, archived asset audit, array shape, score shard identity and coordinate consistency are checked separately.',
        model_changes=False,ranking_changes=False,structure_changes=False)
    (out/'audit_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'verified_residue_mapping.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mapping_rows[0]),delimiter='\t');w.writeheader();w.writerows(mapping_rows)
    print(json.dumps({k:report[k] for k in ['status','cases','remaining_provenance_boundary']},ensure_ascii=False,indent=2))
    print('FAILED CHECKS',json.dumps([c for c in checks if not c['passed']],ensure_ascii=False))
if __name__=='__main__':main()
