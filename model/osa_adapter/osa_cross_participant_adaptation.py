"""Discovery-only cross-participant supervision pilot (Python 3.7+).

Conditional on the existing candidate universe. Not de-novo participant
generalization. External outcomes are never loaded by this module.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit

BASE = "results/discovery_preprocess_v2"
STAGE = "39_cross_participant_adaptation_v1"
METHODS = ("Full", "Static_constrained", "Direct_anchor", "Self_copy")


def write_json(path, value):
    from train_triage_osa import write_json as save
    save(path, value)


def disjoint_halves(meta, sample_ids, ih_ids, seed):
    """Stratified deterministic split; no outcome-driven seed search."""
    rng = np.random.RandomState(seed)
    a, b = [], []
    subset = meta.loc[meta.sample_id.astype(str).isin(sample_ids)]
    for _, rows in subset.groupby("study_group", sort=True):
        ids = sorted(rows.sample_id.astype(str))
        rng.shuffle(ids)
        if len(ids) < 2:
            raise RuntimeError("Cannot split a disease stratum into independent evidence subsets")
        a.extend(ids[::2]); b.extend(ids[1::2])
    ih = sorted(ih_ids); rng.shuffle(ih)
    if len(ih) < 2:
        raise RuntimeError("Insufficient IH participants for disjoint supervision")
    assert not set(a) & set(b)
    return (a, ih[::2]), (b, ih[1::2])


def expert_matrix(x):
    from triage_osa_model import INPUT_FEATURE_COLUMNS
    ix = {c: i for i, c in enumerate(INPUT_FEATURE_COLUMNS)}
    gates = x[:, [ix['g_L1'], ix['g_L2'], ix['g_L3']]].clip(1e-6)
    gates = .1 + .7 * gates / gates.sum(axis=1, keepdims=True)
    tri = (gates * x[:, [ix['logit_L1'], ix['logit_L2'], ix['logit_L3']]]).sum(axis=1)
    anchor = x[:, ix['context_anchor']].clip(1e-5, 1-1e-5)
    return np.column_stack([x[:, 0], tri, np.log(anchor/(1-anchor))])


def initialize_from_static(model, weights, torch):
    """Invert constrained expert parameterization; zero all adaptive corrections.

    Boundary simplex weights are approximated within floating-point tolerance.
    This is an initialization, not a constraint on external performance.
    """
    w = np.asarray(weights, float)
    if w.shape != (3,) or not np.isfinite(w).all() or abs(w.sum()-1)>1e-6:
        raise RuntimeError('Invalid static initialization')
    floors = model.expert_floors.detach().cpu().numpy()
    if np.any(w < floors-1e-7) or w[2] > 1-model.minimum_structural_mass+1e-7:
        raise RuntimeError('Static weights violate model constraints')
    r3 = np.clip((w[2]-floors[2])/(1-model.minimum_structural_mass-floors[2]),1e-6,1-1e-6)
    split = np.maximum(w[:2]-floors[:2],1e-8)
    split = split/split.sum()
    raw = np.r_[split*(1-r3),r3]
    with torch.no_grad():
        for layer in (model.osa_gate_delta,model.osa_residual,model.expert_gate):
            layer.weight.zero_(); layer.bias.zero_()
        model.expert_gate.bias.copy_(torch.as_tensor(np.log(raw),dtype=model.expert_gate.bias.dtype,
                                                     device=model.expert_gate.bias.device))


def run(args):
    import torch
    from triage_osa_model import TRIAGEOSAModel
    from osa_context_data import load_discovery_inputs, compute_node_evidence, build_pair_context, assemble_input_matrix
    from osa_controlled_experiments import reconstructed_folds, convex_static_fit
    from train_triage_osa import fit_adapter, model_predictions, set_seeds, prediction_metrics, sha256_file
    root = Path(args.root).resolve()
    source = root / BASE / '31_osa_controlled_experiments_v1/osa_ablation_suite'
    with (source/'experiment_contract.json').open() as f:
        old = json.load(f)
    seeds = [int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise RuntimeError('Seeds must be unique')
    if args.smoke:
        seeds = seeds[:1]
    out = root / BASE / (STAGE + ('_smoke' if args.smoke else ''))
    if not args.check_env and out.exists() and any(out.iterdir()):
        raise RuntimeError('Stage39 output exists; use a fresh verified run, not --force to erase results')
    pairs = pd.read_csv(root/args.pair_scores, sep='\t')
    proteins = sorted(set(pairs.protein_A) | set(pairs.protein_B))
    expr, meta, paths = load_discovery_inputs(root, proteins)
    cfg = dict(old['optimization'], selection_metric='soft_BCE', experiment_variant='full')
    if args.smoke:
        cfg.update(max_epochs=2, patience=2)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for local checks')
    device = torch.device(args.device)
    # Materialize once: this also verifies historical input provenance.
    folds = list(reconstructed_folds(args, root, source))
    for fold, _, _, _, _, _, _, members in folds:
        disjoint_halves(meta['GSE75097'], members('GSE75097', False),
                        members('GSE71356', False), old['split_seed']+fold)
    if args.check_env:
        print('Cross-participant preflight: PASS; four outer folds; disjoint inner evidence; no external inputs')
        return 0
    out.mkdir(parents=True)
    write_json(out/'analysis_contract.json', dict(
        status='DEVELOPMENT_PROTOCOL', conditional_candidate_universe=True,
        external_data_read=False, architecture='existing tri-level adapter; cross-participant supervision',
        selection='negative soft BCE on validation proteins against supervision-participant anchor; epoch-zero converged static initialization eligible',
        outer_test='heldout participants and test proteins; not used for selection',
        ensemble='equal mean probability of every prespecified fold, direction and seed; no performance filtering',
        optimization=cfg, seeds=seeds, split_seed=old['split_seed'],
        methods=list(METHODS), smoke=bool(args.smoke),
        Self_copy='same architecture and input participants, own-input anchor target',
        Direct_anchor='input-participant anchor, never heldout target'))
    write_json(out/'run_summary.json', dict(status='RUNNING', external_data_read=False))
    sums = {m: np.zeros(len(pairs)) for m in METHODS}
    records, membership, selections, count = [], [], [], 0
    for fold, fp, _, contexts, _, splits, _, members in folds[:1 if args.smoke else 4]:
        if not pairs.pair_id.equals(fp.pair_id):
            raise RuntimeError('Pair order changed')
        halves = disjoint_halves(meta['GSE75097'], members('GSE75097', False),
                                 members('GSE71356', False), old['split_seed']+fold)
        outer = set(members('GSE75097', True))
        for direction in (0, 1):
            inp, sup = halves[direction], halves[1-direction]
            if outer & (set(inp[0]) | set(sup[0])):
                raise RuntimeError('Participant leakage')
            for role, units in [('input', inp), ('supervision', sup),
                                ('outer_test', (members('GSE75097', True), members('GSE71356', True)))]:
                for dataset, ids in zip(('GSE75097','GSE71356'), units):
                    membership.extend(dict(fold=fold, direction=direction, role=role, dataset=dataset, unit_id=u) for u in ids)
            nodes = [compute_node_evidence(expr, meta, proteins, *u) for u in (inp, sup)]
            cs = [build_pair_context(pairs, n, information_fusion=True) for n in nodes]
            x = assemble_input_matrix(pairs, cs[0])
            y = cs[1].context_anchor.to_numpy(dtype=np.float32)
            own = cs[0].context_anchor.to_numpy(dtype=np.float32)
            held = contexts['heldout'].context_anchor.to_numpy(dtype=float)
            experts = expert_matrix(x)
            for seed in seeds:
                unit = out/('fold_%d_direction_%d_seed_%d' % (fold,direction,seed)); unit.mkdir()
                for label, node in zip(('input','supervision'), nodes):
                    write_json(unit/(label+'_availability.json'), node.attrs)
                predictions = {}
                static = convex_static_fit(experts[splits['train']], y[splits['train']], seed+fold, constrained=True)
                if not static['converged']:
                    raise RuntimeError('Static control failed convergence')
                write_json(unit/'static_fit.json', static)
                predictions['Static_constrained'] = expit(experts.dot(static['weights']))
                predictions['Direct_anchor'] = own
                for method, target in [('Full', y), ('Self_copy', own)]:
                    set_seeds(seed+fold, torch)
                    initial_fit = static if method == 'Full' else convex_static_fit(
                        experts[splits['train']],target[splits['train']],seed+fold,constrained=True)
                    if not initial_fit['converged']:
                        raise RuntimeError('Initialization solver did not converge')
                    write_json(unit/(method+'_initial_static_fit.json'), initial_fit)
                    config = dict(cfg, seed=seed+fold, static_initial_weights=initial_fit['weights'])
                    model, history, epoch, best = fit_adapter(x, target, splits['train'], splits['validation'],
                        config, device, torch, TRIAGEOSAModel,
                        progress_prefix='[cross-participant %s fold=%d direction=%d seed=%d]' % (method,fold,direction,seed),
                        validation_targets=y)
                    z = model_predictions(model, x, np.arange(len(pairs)), cfg['batch_size'], device, torch)
                    predictions[method] = expit(z)
                    selections.append(dict(fold=fold,direction=direction,seed=seed,method=method,
                                           best_epoch=epoch,validation_score=best,static_fallback=epoch==0))
                    pd.DataFrame(selections).to_csv(out/'model_selection.tsv',sep='\t',index=False)
                    pd.DataFrame(history).to_csv(unit/(method+'_history.tsv'),sep='\t',index=False)
                    torch.save(dict(model_config=model.export_config(), state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                                    best_epoch=epoch, selection_score=best, training_config=config), str(unit/(method+'_adapter.pt')))
                frame = pairs[['pair_id','protein_A','protein_B']].copy()
                frame['input_anchor'],frame['supervision_anchor'],frame['outer_test_anchor'] = own,y,held
                frame['partition'] = 'excluded_cross_partition'
                for role, idx in splits.items():
                    frame.loc[idx,'partition'] = role
                for method, p in predictions.items():
                    if not np.isfinite(p).all():
                        raise RuntimeError('Nonfinite prediction')
                    frame[method] = p; sums[method] += p
                    for target_name, target in [('supervision',y),('outer_test',held)]:
                        idx = splits['test']; pp = p[idx].clip(1e-6,1-1e-6)
                        records.append(dict(fold=fold,direction=direction,seed=seed,method=method,target=target_name,
                            **prediction_metrics(target[idx], np.log(pp/(1-pp)), x[idx,0])))
                frame.to_csv(unit/'predictions.tsv.gz',sep='\t',index=False)
                count += 1
                pd.DataFrame(records).to_csv(out/'metrics.tsv',sep='\t',index=False)
                print('[completed ensemble units] %d' % count, flush=True)
    pd.DataFrame(membership).to_csv(out/'participant_roles.tsv',sep='\t',index=False)
    ranking = pairs[['pair_id','protein_A','protein_B','gene_A','gene_B']].copy()
    primary_ids = set(pd.read_csv(root/BASE/'04_candidates_v1/candidate_proteins_primary_250.tsv',sep='\t').UniProt_accession)
    ranking['primary_pair'] = np.where(ranking.protein_A.isin(primary_ids)&ranking.protein_B.isin(primary_ids),'yes','no')
    for method in METHODS:
        ranking[method] = sums[method]/count
    ranking.to_csv(out/'rankings_500.tsv.gz',sep='\t',index=False)
    primary = ranking[ranking.primary_pair=='yes'].copy()
    for method in METHODS:
        primary['rank_'+method] = primary[method].rank(method='first',ascending=False).astype(int)
    primary.to_csv(out/'rankings_primary_250.tsv.gz',sep='\t',index=False)
    for method in METHODS:
        export = primary.copy()
        export['triage_osa_score'] = export[method]
        export['triage_osa_rank_primary_250'] = export['rank_'+method]
        export.to_csv(out/('comparison_'+method+'.tsv.gz'),sep='\t',index=False)
    full_node = compute_node_evidence(expr,meta,proteins,meta['GSE75097'].sample_id.astype(str).tolist(),
                                     sorted(set(meta['GSE71356'].subject_id.astype(str))))
    full_node.to_csv(out/'full_discovery_node_evidence.tsv',sep='\t',index=False)
    write_json(out/'input_manifest.json', {k:sha256_file(p) for k,p in paths.items()})
    write_json(out/'run_summary.json',dict(status='PASS_SMOKE' if args.smoke else 'PASS_CROSS_PARTICIPANT_EXECUTION',
        ensemble_units=count, external_data_read=False, conditional_candidate_universe=True,
        full_replaces_production=False, all_static_converged=True, ranking_sha256=sha256_file(out/'rankings_primary_250.tsv.gz'),
        static_fallback_units=sum(int(s['static_fallback']) for s in selections if s['method']=='Full'),
        meaning='execution complete, not a performance gate; Full denotes the new cross-participant ensemble'))
    print('Output: '+str(out),flush=True)
    return 0
