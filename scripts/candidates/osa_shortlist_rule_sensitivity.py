"""Reproduce and perturb archived shortlist rules without model fitting.

Run: python scripts/candidates/osa_shortlist_rule_sensitivity.py --root /path/to/OSA
The rule definitions are encoded below; an archival protocol note is optional.
Requires numpy and pandas. Writes only a new Stage46 result directory.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def truth(x):
    return x.astype(str).str.lower().isin(['true', 'yes', '1'])


def components(frame):
    links = {}
    for a, b in zip(frame.protein_A, frame.protein_B):
        links.setdefault(a, set()).add(b)
        links.setdefault(b, set()).add(a)
    todo = set(links)
    sizes = []
    while todo:
        stack, seen = [min(todo)], set()
        while stack:
            a = stack.pop()
            if a not in seen:
                seen.add(a)
                stack.extend(links[a] - seen)
        todo -= seen
        sizes.append(len(seen))
    return '|'.join(map(str, sorted(sizes, reverse=True)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--output-stage', default='46_shortlist_rule_sensitivity_v2')
    ap.add_argument('--protocol-note', type=Path, help='Optional archival protocol note to fingerprint.')
    args = ap.parse_args()
    root = args.root.resolve()
    if args.protocol_note is not None and not args.protocol_note.is_file():
        ap.error('--protocol-note must refer to an existing file')
    base = root / 'results/discovery_preprocess_v2'
    out = base / args.output_stage
    if out.exists():
        raise RuntimeError('Output already exists; retain it and choose a new output-stage')
    inputs = []

    def read(rel):
        p = base / rel
        inputs.append(p)
        return pd.read_csv(p, sep='\t', low_memory=False)

    x = read('22_consolidated_evidence_triage_osa_v4/pair_evidence_matrix.tsv.gz')
    assert len(x) == 31125 and not x.pair_id.duplicated().any()
    codepath = root / 'model/osa_adapter/predict_triage_osa.py'
    tree = ast.parse(codepath.read_text(encoding='utf-8-sig'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pareto_front_mask')
    scope = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(codepath), 'exec'), scope)
    front = scope['pareto_front_mask']
    axes = [c[:-11] for c in x if c.endswith('_supportive')]
    assert len(axes) == 9
    original_support = pd.DataFrame({a: truth(x[a+'_supportive']) for a in axes})
    availability = pd.DataFrame({a: truth(x[a+'_evaluable']) for a in axes})
    assert not (original_support & ~availability).any().any()
    strict = original_support.copy()
    direction_rows, checks = [], []

    def endpoint_values(df, field):
        assert not df.UniProt_accession.duplicated().any()
        s = df.set_index('UniProt_accession')[field]
        return x.protein_A.map(s).to_numpy(float), x.protein_B.map(s).to_numpy(float)

    def expression_axis(axis, df, tfield, scorefield, stored):
        ta, tb = endpoint_values(df, tfield)
        sa, sb = endpoint_values(df, scorefield)
        ev = availability[axis].to_numpy()
        assert np.isfinite(ta[ev]).all() and np.isfinite(tb[ev]).all(), axis
        reconstructed = np.sqrt(sa * sb)
        assert np.allclose(reconstructed[ev], x.loc[ev, stored], atol=1e-10), axis
        assert np.array_equal((reconstructed > .5) & ev, original_support[axis]), axis
        strict[axis] = original_support[axis] & (ta > 0) & (tb > 0)
        checks.append({'axis': axis, 'eligible': int(ev.sum()), 'score_reconstruction': 'PASS'})
        direction_rows.append(pd.DataFrame({'pair_id': x.pair_id, 'axis': axis,
            'evaluable': ev, 'oriented_t_A': ta, 'oriented_t_B': tb,
            'relative_support': reconstructed, 'original_supportive': original_support[axis],
            'direction_qualified_supportive': strict[axis]}))

    for ds in ['GSE133601', 'GSE49800']:
        df = read('11_cpap_reversal_endpoint_triage_osa_v4/'+ds+'/candidate_node_reversal.tsv')
        expression_axis('cpap_'+ds, df, 'reversal_oriented_moderated_t', 'node_reversal_score', 'cpap_'+ds+'_support')
    df = read('14_adipose_cross_tissue_endpoint_triage_osa_v4/candidate_adipose_node_support.tsv')
    for kind in ['state', 'CPAP']:
        expression_axis('adipose_'+kind, df, kind+'_oriented_moderated_t', kind+'_node_score', 'adipose_'+kind+'_support')
    df = read('21_independent_validation_endpoint_triage_osa_v4/candidate_node_evidence_primary_250.tsv')
    for ds in ['GSE38792', 'GSE293433']:
        expression_axis('independent_'+ds, df, ds+'_oriented_t', ds+'_node_score', 'independent_'+ds+'_pair_support')
    df = read('16_scrna_cell_context_endpoint_triage_osa_v4/candidate_scrna_node_support.tsv')
    original_lineages, qualified_lineages = [], []
    for lineage, sub in df.groupby('lineage', sort=True):
        sa, sb = endpoint_values(sub, 'node_score')
        ta, tb = endpoint_values(sub, 'oriented_moderated_t')
        aa, ab = endpoint_values(sub.assign(eligible=truth(sub.available).astype(float)), 'eligible')
        eligible = (aa == 1) & (ab == 1)
        val = np.where(eligible, np.sqrt(sa * sb), np.nan)
        assert np.isfinite(val[eligible]).all()
        qualified = np.where(eligible, np.where((ta > 0) & (tb > 0), val, 0.), np.nan)
        original_lineages.append(val)
        qualified_lineages.append(qualified)
        direction_rows.append(pd.DataFrame({'pair_id': x.pair_id, 'axis': 'pediatric_scRNA_'+lineage,
            'evaluable': eligible, 'oriented_t_A': ta, 'oriented_t_B': tb,
            'relative_support': val, 'original_supportive': val > .5,
            'direction_qualified_supportive': qualified > .5}))
    arr = np.column_stack(original_lineages)
    n = np.isfinite(arr).sum(1)
    score = np.divide(np.nansum(arr, axis=1), n, out=np.full(len(x), np.nan), where=n > 0)
    qscore = np.divide(np.nansum(np.column_stack(qualified_lineages), axis=1), n,
                       out=np.full(len(x), np.nan), where=n > 0)
    assert np.array_equal(n, x.scrna_available_lineage_count)
    assert np.allclose(score, x.scrna_mean_support, equal_nan=True, atol=1e-10)
    assert np.array_equal(score > .5, original_support.pediatric_scRNA)
    strict['pediatric_scRNA'] = qscore > .5
    assert not (strict & ~original_support).any().any()
    checks.append({'axis': 'pediatric_scRNA', 'eligible': int((n > 0).sum()), 'score_reconstruction': 'PASS'})
    xy = x[['structural_percentile', 'osa_context_percentile']].to_numpy(float)
    ranks = x.triage_osa_rank_primary_250.to_numpy(float)
    baseline = front(np.column_stack([xy, original_support.sum(axis=1)/9]))
    assert np.array_equal(baseline, truth(x.pareto_front)) and baseline.sum() == 32
    orig = set(x.loc[baseline & (ranks <= 1557), 'pair_id'])
    assert len(orig) == 15

    def breadth(flags, grouping):
        if grouping == 'none':
            return flags.sum(axis=1).to_numpy(float)/9
        if grouping == 'available':
            return np.divide(flags.sum(axis=1), availability.sum(axis=1),
                             out=np.zeros(len(x)), where=availability.sum(axis=1) > 0)
        grouped = ['adipose_state', 'independent_GSE38792']
        return (flags.drop(columns=grouped).sum(axis=1)+flags[grouped].mean(axis=1)).to_numpy(float)/(10-len(grouped))

    settings = [('original', original_support, 'none', 'primary'),
        ('direction', strict, 'none', 'primary'),
        ('source_grouped', original_support, 'source', 'primary'),
        ('direction_source_grouped', strict, 'source', 'primary'),
        ('available_axes', original_support, 'available', 'secondary')]
    summaries, memberships = [], []
    allflags = x[['pair_id', 'protein_A', 'protein_B', 'gene_A', 'gene_B', 'triage_osa_rank_primary_250']].copy()
    for name, flags, grouping, role in settings:
        br = breadth(flags, grouping)
        values = np.column_stack([xy, br])
        mask = front(values)
        # Independent direct dominance checks for every returned front point.
        for idx in np.flatnonzero(mask):
            assert not np.any(np.all(values >= values[idx], axis=1) & np.any(values > values[idx], axis=1))
        keep = mask & (ranks <= 1557)
        edges = x.loc[keep]
        ids = set(edges.pair_id)
        summaries.append(dict(setting=name, role=role, pareto_pairs=int(mask.sum()),
            selected_pairs=len(ids), retained_original=len(ids & orig), lost_original=len(orig-ids),
            added_pairs=len(ids-orig), jaccard=len(ids & orig)/len(ids | orig),
            nodes=len(set(edges.protein_A)|set(edges.protein_B)), component_sizes=components(edges)))
        selected = edges[['pair_id', 'protein_A', 'protein_B', 'gene_A', 'gene_B', 'triage_osa_rank_primary_250']].copy()
        selected['setting'], selected['original_edge'] = name, selected.pair_id.isin(orig)
        memberships.append(selected)
        allflags[name+'_breadth'] = br
        allflags[name+'_pareto'] = mask
        allflags[name+'_selected'] = keep
    stab = read('42_complete_candidate_stability_v1/candidate_rank_stability.tsv.gz').set_index('pair_id')
    dep = read('41_public_evidence_closeout_v1/original_candidate_evidence_dependence.tsv').set_index('pair_id')
    review = x.loc[x.pair_id.isin(orig), ['pair_id','gene_A','gene_B','protein_A','protein_B',
        'triage_osa_rank_primary_250','evidence_evaluable_axis_count','evidence_supportive_axis_count']].copy()
    for col in ['rank_min','rank_max','top5_frequency']:
        review[col] = review.pair_id.map(stab[col])
    review['top5_runs_of_12'] = np.rint(review.top5_frequency*12).astype(int)
    review['family_deletion_retention_count'] = review.pair_id.map(dep.family_deletion_retention_count)
    for name, _, _, _ in settings:
        review[name+'_selected'] = review.pair_id.map(allflags.set_index('pair_id')[name+'_selected'])
    review['primary_alternative_rules_retained'] = review[[n+'_selected' for n in ['direction','source_grouped','direction_source_grouped']]].sum(axis=1)
    for a in axes:
        review[a+'_supportive'] = review.pair_id.map(pd.Series(original_support[a].to_numpy(), index=x.pair_id))
        review[a+'_direction_supportive'] = review.pair_id.map(pd.Series(strict[a].to_numpy(), index=x.pair_id))
    out.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(out/'rule_summary.tsv', sep='\t', index=False)
    pd.concat(memberships).to_csv(out/'selected_pairs_by_rule.tsv', sep='\t', index=False)
    allflags.to_csv(out/'complete_rule_membership.tsv.gz', sep='\t', index=False)
    review.to_csv(out/'original_15_candidate_rule_review.tsv', sep='\t', index=False)
    pd.concat(direction_rows).to_csv(out/'complete_endpoint_direction_audit.tsv.gz', sep='\t', index=False)
    pd.DataFrame(checks).to_csv(out/'reconstruction_checks.tsv', sep='\t', index=False)
    inputs += [codepath, Path(__file__).resolve()]
    if args.protocol_note is not None:
        inputs.append(args.protocol_note.resolve())
    manifest = []
    for p in dict.fromkeys(inputs):
        try:
            label = p.relative_to(root).as_posix()
        except ValueError:
            label = p.name
        manifest.append({'path': label, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (out/'provenance.json').write_text(json.dumps({'status':'PASS', 'pairs':len(x),
        'original_pareto':32, 'original_edges':15, 'training_performed':False,
        'original_results_modified':False, 'inputs':manifest}, indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(review[['gene_A','gene_B','top5_runs_of_12','primary_alternative_rules_retained']].to_string(index=False))


if __name__ == '__main__':
    main()
