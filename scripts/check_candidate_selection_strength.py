"""Verify documented discovery selection strength against archived outputs."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
base = root/'results/discovery_preprocess_v2'
source = root/'scripts/osa_discovery_candidate_ranking.py'
fn = next(n for n in ast.parse(source.read_text(encoding='utf-8-sig')).body
          if isinstance(n, ast.FunctionDef) and n.name == 'midrank_strength')
env = {'Any': object}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), env)
strength = env['midrank_strength']
assert np.allclose(strength(pd.Series([.1,.01,.01]),pd,np),[1/6,2/3,2/3])
x = pd.read_csv(base/'04_discovery_v2/protein_context_evidence.tsv.gz',sep='\t')
adult = x[x.OSA_available.eq('yes') & x.severe_available.eq('yes') & x.AHI_available.eq('yes')].copy()
for a in ['OSA','severe','AHI']:
    adult[a+'_strength'] = strength(adult[a+'_P_value'], pd, np)
ih = x[x.IH_available.eq('yes')].copy()
ih['IH_strength'] = strength(ih.IH_P_value, pd, np)
adult['IH_strength_or_neutral'] = adult.UniProt_accession.map(ih.set_index('UniProt_accession').IH_strength).fillna(.5)
adult['adult_OSA_axis'] = .5*adult.OSA_strength+.25*adult.severe_strength+.25*adult.AHI_strength
adult['discovery_score'] = .75*adult.adult_OSA_axis+.25*adult.IH_strength_or_neutral
adult = adult.sort_values(['discovery_score','adult_OSA_axis','OSA_strength','UniProt_accession'],ascending=[False,False,False,True])
old = pd.read_csv(base/'04_candidates_v1/protein_ranking_all.tsv.gz',sep='\t')
assert adult.UniProt_accession.tolist() == old.UniProt_accession.tolist()
for c in ['OSA_strength','severe_strength','AHI_strength','IH_strength_or_neutral','discovery_score']:
    assert np.allclose(adult[c],old[c],atol=1e-12), c
out=base/'46_shortlist_rule_sensitivity_v2'
report={'status':'PASS','adult_background':len(adult),'IH_background':len(ih),
        'all_archived_ranks_reproduced':True,'clipping':1e-300,'ties':'average',
        'formula':'(ascending_midrank(-log10(max(P, 1e-300))) - 0.5)/N'}
(out/'selection_strength_check.json').write_text(json.dumps(report,indent=2))
print(report)
