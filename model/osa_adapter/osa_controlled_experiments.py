#!/usr/bin/env python3
"""Controlled experiments called by train_triage_osa.py, not a new training engine.

Python 3.7 compatible. No external OSA endpoint is used. Numerical completion
is not biological success. Existing production checkpoints are read-only.
"""
from __future__ import print_function

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0.0"
BASE = "results/discovery_preprocess_v2/06_model_training/esm_only_adaptation_v1/triage_tuna_esm_only_adapted.pt"
V4 = "results/discovery_preprocess_v2/06_model_training/triage_osa_patient_information_v4/triage_osa_specialized.pt"


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def safe_directory(root, path):
    path = resolve(root, path).resolve()
    allowed = (root / "results" / "discovery_preprocess_v2").resolve()
    if allowed not in path.parents or not path.relative_to(allowed).parts[0].startswith("31_"):
        raise RuntimeError("Experiment outputs must be inside results/discovery_preprocess_v2/31_*")
    return path


def write_json(path, value):
    from train_triage_osa import json_safe
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def average_precision_ties(y, score):
    """Threshold-group AP: tied scores must not depend on file row order."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    ends = np.r_[np.where(np.diff(score) != 0)[0], len(y) - 1]
    tp = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / y.sum())


def metrics(target, logits, base):
    from train_triage_osa import prediction_metrics
    result = prediction_metrics(target, logits, base)
    result["weak_top20_AUPRC"] = average_precision_ties(
        target >= np.quantile(target, .8), logits)
    return result


def variant_features(features, context, variant):
    from triage_osa_model import INPUT_FEATURE_COLUMNS, PATIENT_INFORMATION_FEATURE_COLUMNS, CONTEXT_FEATURE_COLUMNS
    x = np.asarray(features, dtype=np.float32).copy()
    idx = dict((name, i) for i, name in enumerate(INPUT_FEATURE_COLUMNS))
    if variant == "no_patient":
        x[:, [idx[c] for c in PATIENT_INFORMATION_FEATURE_COLUMNS]] = 0.0
        x[:, idx["context_anchor"]] = np.asarray(context["molecular_context_anchor"], dtype=np.float32)
    if variant == "no_context":
        x[:, [idx[c] for c in CONTEXT_FEATURE_COLUMNS]] = 0.0
        x[:, idx["context_anchor"]] = .5
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite experiment features")
    return x


def variant_config(optimization, variant, seed):
    config = dict(optimization, experiment_variant=variant, seed=int(seed))
    if variant in ("equal_fusion", "static_fusion"):
        # These comparators are deliberately ordinary unconstrained fusion.
        # Keep data BCE/ranking losses but do not impose adaptive-model penalties.
        for key in ("constraint_triggered_PPI_retention_weight", "gate_drift_weight",
                    "expert_entropy_weight", "structural_dual_initial",
                    "structural_dual_learning_rate", "structural_dual_max"):
            config[key] = 0.0
    return config


def self_test(torch):
    from triage_osa_model import TRIAGEOSAModel, INPUT_FEATURE_COLUMNS, EXPERIMENT_VARIANTS
    from train_triage_osa import fit_adapter
    n = len(INPUT_FEATURE_COLUMNS)
    rng = np.random.RandomState(41)
    x = rng.normal(size=(32, n)).astype(np.float32)
    x[:, 4:7] = 1.0 / 3
    anchor = INPUT_FEATURE_COLUMNS.index("context_anchor")
    x[:, anchor] = rng.uniform(.15, .85, len(x))
    context = {"molecular_context_anchor": np.full(len(x), .4)}
    outputs = {}
    for variant in EXPERIMENT_VARIANTS:
        model = TRIAGEOSAModel(np.zeros(n), np.ones(n), experiment_variant=variant).eval()
        xx = variant_features(x, context, variant)
        out = model(torch.from_numpy(xx))
        assert torch.isfinite(out["adapted_logit"]).all()
        assert torch.allclose(out["expert_weights"].sum(1), torch.ones(len(x)))
        if variant not in ("equal_fusion", "static_fusion"):
            assert out["structural_expert_mass"].min() >= .1 - 1e-6
        if variant in ("no_gate_delta", "equal_fusion", "static_fusion"):
            assert out["gate_delta"].abs().max() == 0
        if variant in ("no_residual", "equal_fusion", "static_fusion"):
            assert out["context_residual"].abs().max() == 0
        if variant == "equal_fusion":
            assert torch.allclose(out["expert_weights"], torch.full((len(x), 3), 1.0 / 3))
        if variant == "static_fusion":
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3
        if variant != "equal_fusion":
            out["adapted_logit"].square().mean().backward()
            assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
        restored = TRIAGEOSAModel.from_config(model.export_config()).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(out["adapted_logit"], restored(torch.from_numpy(xx))["adapted_logit"])
        outputs[variant] = "PASS"
    changed = x.copy()
    from triage_osa_model import PATIENT_INFORMATION_FEATURE_COLUMNS, CONTEXT_FEATURE_COLUMNS
    for name in PATIENT_INFORMATION_FEATURE_COLUMNS:
        changed[:, INPUT_FEATURE_COLUMNS.index(name)] += 100
    changed[:, anchor] = .9
    assert np.array_equal(variant_features(x, context, "no_patient"), variant_features(changed, context, "no_patient"))
    for name in CONTEXT_FEATURE_COLUMNS:
        changed[:, INPUT_FEATURE_COLUMNS.index(name)] += 100
    assert np.array_equal(variant_features(x, context, "no_context"), variant_features(changed, context, "no_context"))
    assert abs(average_precision_ties([1, 0], [0, 0]) - .5) < 1e-12
    # Real optimizer/selection/save-restore exercise on tiny synthetic arrays.
    config = dict(hidden_dimension=8, dropout=0., minimum_mean_structural_mass=.1,
                  learning_rate=.001, weight_decay=.0001, batch_size=8, max_epochs=2,
                  patience=2, minimum_ranking_target_gap=.1, bce_weight=1., ranking_weight=.25,
                  constraint_triggered_PPI_retention_weight=.1, gate_drift_weight=.02,
                  expert_entropy_weight=.02, minimum_expert_entropy_normalized=.2)
    for variant in EXPERIMENT_VARIANTS:
        fitted, history, epoch, _ = fit_adapter(
            variant_features(x, context, variant), x[:, anchor], np.arange(16),
            np.arange(16, 24), variant_config(config, variant, 41), torch.device("cpu"),
            torch, TRIAGEOSAModel, progress_prefix="[self-test " + variant + "]")
        assert epoch == (0 if variant == "equal_fusion" else len(history)) or epoch == 1
        assert torch.isfinite(fitted(torch.from_numpy(x[24:]))["adapted_logit"]).all()
    print("Experiment self-test: PASS (7 variants, masking, roundtrip, optimizer, tied AP)")
    return outputs


def run_suite(args, root, out, torch):
    from triage_osa_model import TRIAGEOSAModel, EXPERIMENT_VARIANTS
    from train_triage_osa import (fit_adapter, model_predictions, model_expert_weights,
                                set_seeds, sha256_file, EXPECTED_BASE_SHA256)
    from osa_context_data import (load_discovery_inputs, stratified_sample_folds,
        subject_folds, stable_bucket, role_for_fold, compute_node_evidence,
        build_pair_context, assemble_input_matrix)
    variants = list(EXPERIMENT_VARIANTS) if args.variants == "all" else args.variants.split(",")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(EXPERIMENT_VARIANTS):
        raise RuntimeError("Invalid or duplicate --variants")
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError("Provide unique paired training seeds")
    if args.smoke:
        seeds = seeds[:1]
    lock_path = resolve(root, args.training_lock)
    lock = read_json(lock_path)
    optimization = dict(lock["optimization"])
    if not lock["trainable_innovation"].get("patient_information_conditioning"):
        raise RuntimeError("Controlled experiments require the V4 information contract")
    split_seed = int(optimization["seed"])
    if args.smoke:
        optimization.update(max_epochs=2, patience=2, batch_size=512)
    paths = {"training_lock": lock_path, "pair_scores": resolve(root, args.pair_scores),
             "base_checkpoint": root / BASE}
    if sha256_file(paths["base_checkpoint"]) != EXPECTED_BASE_SHA256:
        raise RuntimeError("Unexpected actual OSA backbone")
    pairs = pd.read_csv(paths["pair_scores"], sep="\t", low_memory=False)
    pairs[["protein_A", "protein_B"]] = pairs[["protein_A", "protein_B"]].astype(str)
    proteins = sorted(set(pairs.protein_A) | set(pairs.protein_B))
    canonical = [tuple(sorted(p)) for p in zip(pairs.protein_A, pairs.protein_B)]
    if (len(pairs) != len(proteins) * (len(proteins) - 1) // 2 or
            len(set(canonical)) != len(pairs) or any(a == b for a, b in canonical) or pairs.pair_id.duplicated().any()):
        raise RuntimeError("Incomplete/duplicate/self pair universe")
    if not args.smoke and (len(proteins), len(pairs)) != (500, 124750):
        raise RuntimeError("Formal suite requires the frozen 500-protein universe")
    if set(pairs.checkpoint_sha256.astype(str)) != {EXPECTED_BASE_SHA256}:
        raise RuntimeError("Pair scores use a different backbone")
    expression, metadata, input_paths = load_discovery_inputs(root, proteins)
    paths.update(input_paths)
    if sha256_file(input_paths["discovery_patient_information"]) != lock["patient_information_manifest"]["sha256"]:
        raise RuntimeError("Patient-information input differs from V4 contract")
    # Content hashes are internal resume guards, not repeated user-facing tasks.
    paths.update({"experiment_code": Path(__file__), "model_code": Path(__file__).with_name("triage_osa_model.py"),
                  "trainer_code": Path(__file__).with_name("train_triage_osa.py"),
                  "context_code": Path(__file__).with_name("osa_context_data.py")})
    contract = dict(version=VERSION, smoke=bool(args.smoke), variants=variants, seeds=seeds,
        split_seed=split_seed, optimization=optimization,
        inputs=dict((k, sha256_file(v)) for k, v in paths.items()),
        target="Same development-participant V4 weak anchor for every variant",
        selection="Validation-protein multi-axis utility; never held-out participants or external endpoints",
        interpretation="Protein-disjoint, selection-conditioned weak-target comparison; not clinical prediction",
        no_patient_scope="Patient-derived channels and mixed-anchor component removed; existing demographic adjustment retained",
        external_validation_read=False, replace_frozen_V4=False,
        performance_filter=False, static_fusion_constraints="Ordinary simplex only, no adaptive safety regularizers")
    if args.check_env:
        self_test(torch)
        print("Controlled experiment preflight: PASS; {} proteins; {} variants; {} seeds".format(len(proteins), len(variants), len(seeds)))
        print("Output: " + str(out))
        return 0
    if out.exists() and (out / "experiment_contract.json").exists():
        if args.resume_experiments:
            if read_json(out / "experiment_contract.json") != contract:
                raise RuntimeError("Resume contract differs; use a new 31_* experiment directory")
        elif args.force:
            shutil.rmtree(str(out))  # safe_directory verified exact Stage 31 subtree
        else:
            raise RuntimeError("Experiment exists; use --resume-experiments or --force")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "experiment_contract.json", contract)
    write_json(out / "run_summary.json", dict(status="RUNNING", completed_units=0, external_validation_read=False))
    sample = stratified_sample_folds(metadata["GSE75097"], folds=4, seed=split_seed)
    subjects = subject_folds(metadata["GSE71356"], seed=split_seed)
    protein = pd.DataFrame(dict(dataset="protein_universe", unit_id=proteins,
        stratum="UniProt", fold=[stable_bucket(p, 4, split_seed) for p in proteins]))
    pd.concat([sample, subjects, protein], ignore_index=True).to_csv(out / "fold_assignments.tsv", sep="\t", index=False)
    all_metrics, completed = [], 0
    for fold in range(1 if args.smoke else 4):
        def units(table, test):
            return [str(r.unit_id) for r in table.itertuples() if (int(r.fold) == fold) == test]
        node = compute_node_evidence(expression, metadata, proteins, units(sample, False), units(subjects, False))
        heldout_node = compute_node_evidence(expression, metadata, proteins, units(sample, True), units(subjects, True))
        context = build_pair_context(pairs, node, information_fusion=True)
        heldout = build_pair_context(pairs, heldout_node, information_fusion=True)
        features = assemble_input_matrix(pairs, context)
        target = context.context_anchor.to_numpy(dtype=np.float32).copy()
        heldout_target = heldout.context_anchor.to_numpy(dtype=np.float32)
        roles = dict((p, role_for_fold(f, fold, 4)) for p, f in zip(protein.unit_id, protein.fold))
        role = np.array([roles[a] if roles[a] == roles[b] else "excluded_cross_partition" for a, b in canonical])
        splits = dict((r, np.where(role == r)[0]) for r in ("train", "validation", "test"))
        if min(map(len, splits.values())) < 2:
            raise RuntimeError("Empty protein split")
        pd.DataFrame(dict(pair_id=pairs.pair_id, partition=role)).to_csv(out / ("fold_%d_pair_partitions.tsv.gz" % (fold + 1)), sep="\t", index=False, compression="gzip")
        for seed in seeds:
            for variant in variants:
                unit = out / ("seed_%d" % seed) / variant / ("fold_%d" % (fold + 1))
                done = unit / "completed.json"
                if args.resume_experiments and done.is_file():
                    files = ("metrics.tsv", "predictions.tsv.gz", "adapter.pt", "history.tsv")
                    if not all((unit / f).is_file() for f in files):
                        raise RuntimeError("Incomplete completed experiment unit: " + str(unit))
                    all_metrics.extend(pd.read_csv(unit / "metrics.tsv", sep="\t").to_dict("records"))
                    completed += 1
                    continue
                unit.mkdir(parents=True, exist_ok=True)
                set_seeds(seed + fold, torch)
                x = variant_features(features, context, variant)
                model, history, epoch, best = fit_adapter(x, target, splits["train"], splits["validation"],
                    variant_config(optimization, variant, seed + fold), torch.device(args.device), torch,
                    TRIAGEOSAModel, progress_prefix="[{} seed={} fold={}]".format(variant, seed, fold + 1))
                pd.DataFrame(history, columns=list(history[0]) if history else ["epoch", "train_loss"]).to_csv(unit / "history.tsv", sep="\t", index=False)
                predictions, rows = [], []
                for partition, idx in splits.items():
                    logit = model_predictions(model, x, idx, optimization["batch_size"], torch.device(args.device), torch)
                    weights = model_expert_weights(model, x, idx, optimization["batch_size"], torch.device(args.device), torch)
                    if not np.isfinite(logit).all() or not np.isfinite(weights).all():
                        raise RuntimeError("Non-finite fitted outputs")
                    identity = dict(variant=variant, seed=seed, fold=fold + 1, partition=partition, best_epoch=epoch,
                                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
                    row = dict(identity, **metrics(target[idx], logit, features[idx, 0]))
                    row.update(structural_mass=float(weights[:, :2].sum(1).mean()),
                               heldout_participant_target_rho=__import__("train_triage_osa").spearman_correlation(heldout_target[idx], logit))
                    rows.append(row)
                    frame = pairs.iloc[idx][["pair_id", "protein_A", "protein_B"]].copy()
                    frame["partition"], frame["weak_target"], frame["logit"] = partition, target[idx], logit
                    frame["heldout_participant_weak_target"] = heldout_target[idx]
                    frame["base_logit"] = features[idx, 0]
                    predictions.append(frame)
                pd.DataFrame(rows).to_csv(unit / "metrics.tsv", sep="\t", index=False)
                pd.concat(predictions, ignore_index=True).to_csv(unit / "predictions.tsv.gz", sep="\t", index=False, compression="gzip")
                torch.save(dict(checkpoint_type="OSA_controlled_adapter_not_production", model_config=model.export_config(),
                    state_dict=dict((k, v.detach().cpu()) for k, v in model.state_dict().items()),
                    variant=variant, seed=seed, fold=fold + 1, best_epoch=epoch, selection_score=best,
                    frozen_base_checkpoint=BASE, eligible_for_production=False), str(unit / "adapter.pt"))
                write_json(done, dict(status="COMPLETE_NOT_PERFORMANCE_GATE", variant=variant, seed=seed, fold=fold + 1))
                all_metrics.extend(rows)
                completed += 1
                pd.DataFrame(all_metrics).to_csv(out / "all_metrics.tsv", sep="\t", index=False)
                write_json(out / "run_summary.json", dict(status="RUNNING", completed_units=completed,
                    total_units=len(seeds) * len(variants) * (1 if args.smoke else 4)))
                del model
    frame = pd.DataFrame(all_metrics)
    frame.to_csv(out / "all_metrics.tsv", sep="\t", index=False)
    test = frame.loc[frame.partition == "test"].copy()
    report_metrics = ["Spearman", "weak_top20_AUPRC", "top5_enrichment_ratio", "multi_axis_utility", "structural_mass", "heldout_participant_target_rho"]
    test.groupby("variant")[report_metrics].agg(["mean", "std", "min", "max"]).to_csv(out / "test_summary.tsv", sep="\t")
    full = test.loc[test.variant == "full", ["seed", "fold"] + report_metrics]
    difference = test.merge(full, on=["seed", "fold"], suffixes=("", "_full"), validate="many_to_one")
    for name in report_metrics:
        difference["delta_" + name + "_vs_full"] = difference[name] - difference[name + "_full"]
    difference.to_csv(out / "paired_differences_vs_full.tsv", sep="\t", index=False)
    summary = dict(status="PASS_SMOKE_EXECUTION" if args.smoke else "PASS_CONTROLLED_EXPERIMENT_EXECUTION",
        variants=variants, seeds=seeds, folds=1 if args.smoke else 4, completed_units=completed,
        biological_advantage_established=False, external_validation_read=False, frozen_V4_changed=False,
        full_control_available=not full.empty, checkpoint_selection_eligible=False,
        note="Descriptive paired fold/seed contrasts; overlapping folds are not independent clinical replicates. No production refit.")
    write_json(out / "run_summary.json", summary)
    print("Controlled experiments: " + summary["status"])
    print("Output: " + str(out))
    return 0


def binary_metrics(labels, logits):
    from train_triage_osa import roc_auc
    y = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(logits).all():
        raise RuntimeError("Invalid benchmark labels/scores")
    p = logits >= 0  # Prespecified probability threshold 0.5; no test optimization.
    tp, tn = float(np.sum(p & (y == 1))), float(np.sum(~p & (y == 0)))
    fp, fn = float(np.sum(p & (y == 0))), float(np.sum(~p & (y == 1)))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(n=len(y), positives=int(y.sum()), AUROC=roc_auc(y, logits),
        AUPRC=average_precision_ties(y, logits), MCC=(tp * tn - fp * fn) / denominator if denominator else 0.0,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        accuracy=(tp + tn) / max(1, len(y)), threshold=.5)


def retention_identity(root, torch):
    from osa_triage_adapter import load_frozen_checkpoint
    from triage_osa_model import TRIAGEOSAModel
    model, cfg, meta = load_frozen_checkpoint(root / BASE, torch.device("cpu"))
    payload = torch.load(str(root / V4), map_location="cpu")
    embedded, _ = TRIAGEOSAModel.split_unified_state_dict(payload["model_state_dict"])
    state = model.state_dict()
    mismatch = sorted(set(state) ^ set(embedded))
    mismatch += [k for k in set(state) & set(embedded) if not torch.equal(state[k], embedded[k])]
    config_mismatch = [k for k, v in cfg.to_dict().items() if payload["base_model_config"].get(k) != v]
    duplicate = copy.deepcopy(model)
    duplicate.load_state_dict(embedded, strict=True)
    rng = torch.Generator().manual_seed(41)
    inputs = dict(resA=torch.randn(1, 9, int(cfg.d_res_in), generator=rng),
                  resB=torch.randn(1, 11, int(cfg.d_res_in), generator=rng))
    inputs["resA"][:, :, 1280:] = 0
    inputs["resB"][:, :, 1280:] = 0
    with torch.no_grad():
        a, b = model(**inputs), duplicate(**inputs)
    keys = ["logit_triage", "logit_res_A", "logit_res_B", "S_interface"]
    maxdiff = dict((k, float((a[k] - b[k]).abs().max())) for k in keys)
    result = dict(status="PASS_BACKBONE_IDENTITY" if not mismatch and not config_mismatch and max(maxdiff.values()) < 1e-6 else "FAIL_BACKBONE_IDENTITY",
        tensors=len(state), mismatched_tensors=mismatch, mismatched_config_fields=config_mismatch,
        synthetic_max_abs_difference=maxdiff, actual_benchmark_evaluated=False,
        meaning="Weight/config and synthetic-output equivalence only; not residue/pair accuracy or generalization")
    return result, model, cfg


def support_perturbed_forward(model, inputs, mode, fraction, seed, torch):
    """Remove an equal number of selected L2 supports and renormalize weights.

    Hooks change only the pooled L2 evidence fed to interface_vec. Encoders,
    masks, L1 maps, L3 and checkpoint tensors are untouched. Not retraining.
    """
    from triage_model import select_topm_interface
    cache, handles = {}, []
    def capture(name):
        def hook(module, values, output):
            cache[name] = values[0] if name in ("HA", "HB") else output
        return hook
    def replace_pool(module, values):
        ha, hb = cache["HA"], cache["HB"]
        scores = cache["scores"].clamp(-30, 30)
        index, logits = select_topm_interface(scores, inputs.get("maskA"), inputs.get("maskB"),
                                              model.cfg.topm, model.cfg.block_size)
        fa = ha.gather(1, index[..., 0].unsqueeze(-1).expand(-1, -1, ha.shape[-1]))
        fb = hb.gather(1, index[..., 1].unsqueeze(-1).expand(-1, -1, hb.shape[-1]))
        weights = torch.softmax(logits, dim=1)
        keep = torch.ones_like(weights)
        rng = np.random.RandomState(seed)
        for row in range(len(weights)):
            valid = torch.where(logits[row] > -1e8)[0]
            count = min(max(0, len(valid) - 1), int(math.ceil(fraction * len(valid))))
            if count:
                if mode == "random":
                    chosen = valid[torch.as_tensor(rng.permutation(len(valid))[:count], device=valid.device)]
                else:
                    order = torch.argsort(logits[row, valid], descending=(mode == "top"))
                    chosen = valid[order[:count]]
                keep[row, chosen] = 0
        weights = weights * keep
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        pooled = (torch.cat([fa, fb, logits.unsqueeze(-1)], dim=-1) * weights.unsqueeze(-1)).sum(1)
        return (pooled,)
    handles.append(model.res_head_A.register_forward_hook(capture("HA")))
    handles.append(model.res_head_B.register_forward_hook(capture("HB")))
    handles.append(model.l2_refine.register_forward_hook(capture("scores")))
    handles.append(model.interface_vec.register_forward_pre_hook(replace_pool))
    try:
        return model(**inputs)
    finally:
        for handle in handles:
            handle.remove()


def run_benchmarks(args, root, out, adapted, cfg, torch):
    """Reuse original native data loaders and model; require explicit holdout IDs.

    The manifest is evidence supplied after exposure auditing, not an automatic
    declaration that a named public dataset is independent.
    """
    from osa_triage_adapter import load_frozen_checkpoint, swapped_inputs
    from train_triage import TriagePairDataset, TriageSiteDataset, collate_pad, batch_to_device
    import train_triage as native_loader
    from train_triage_osa import sha256_file
    lock_path = resolve(root, args.benchmark_lock)
    lock = read_json(lock_path)
    if lock.get("exposure_audit_complete") is not True or not lock.get("exposure_audit_note"):
        raise RuntimeError("Complete the original/adaptation/model-selection exposure audit before benchmark evaluation")
    required_phases = {"original_training", "original_selection", "esm_adaptation_training", "esm_adaptation_selection"}
    exposures = lock.get("exposure_manifests", [])
    if set(x["phase"] for x in exposures) != required_phases:
        raise RuntimeError("Exposure manifests must cover all four original/adaptation training and selection phases")
    seen_proteins, seen_units, input_hashes = set(), set(), {}
    for entry in exposures:
        path = resolve(root, entry["path"])
        table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if not {"protein_id", "unit_id"}.issubset(table.columns) or table.empty:
            raise RuntimeError("Exposure TSV requires nonempty protein_id and unit_id columns")
        seen_proteins.update(p for p in table.protein_id if p)
        seen_units.update(p for p in table.unit_id if p)
        input_hashes[entry["phase"]] = sha256_file(path)
    if not seen_proteins or not seen_units:
        raise RuntimeError("Empty exposure inventory")
    if not lock.get("datasets"):
        raise RuntimeError("No locked benchmark datasets; identity alone is not an accuracy test")
    original_path = resolve(root, lock["original_checkpoint"])
    original, original_cfg, _ = load_frozen_checkpoint(original_path, torch.device("cpu"))
    if original_cfg.d_res_in != cfg.d_res_in:
        raise RuntimeError("Backbones have different input dimensions; matched-input comparison needs an explicit adapter")
    models = {"original_matched_ESM_input": original, "actual_OSA_ESM_backbone": adapted}
    device = torch.device(args.device)
    for model in models.values():
        model.eval().to(device)
    summary_rows = []
    datasets_used = set()
    for spec in lock["datasets"]:
        name = str(spec["name"])
        if not name or Path(name).name != name or name in datasets_used or name in (".", ".."):
            raise RuntimeError("Invalid/duplicate benchmark dataset name")
        datasets_used.add(name)
        if spec["type"] not in ("pair_fourpack", "site_global"):
            raise RuntimeError("Supported native loaders: pair_fourpack and site_global")
        directory = resolve(root, spec["directory"])
        esm_dir = resolve(root, spec["esm_dir"])
        test_path = resolve(root, spec["test_ids_tsv"])
        test = pd.read_csv(test_path, sep="\t", dtype=str, keep_default_na=False)
        if not {"unit_id", "protein_A", "protein_B"}.issubset(test.columns) or test.empty or test.unit_id.duplicated().any():
            raise RuntimeError("Test manifest requires unique unit_id, protein_A, protein_B")
        test_proteins = set(test.protein_A) | set(test.protein_B)
        # Require a common canonical namespace, including aliases/chain mappings.
        if not spec.get("canonical_id_audit_note"):
            raise RuntimeError("Canonical ID/homology scope audit note required")
        aliases = pd.read_csv(resolve(root, spec["canonical_id_map_tsv"]), sep="\t", dtype=str, keep_default_na=False)
        if not {"protein_id", "canonical_id"}.issubset(aliases.columns) or aliases.protein_id.duplicated().any():
            raise RuntimeError("Canonical mapping requires unique protein_id and canonical_id")
        canonical_map = dict(zip(aliases.protein_id, aliases.canonical_id))
        needed = test_proteins | seen_proteins
        if needed - set(canonical_map) or any(not canonical_map[p] for p in needed):
            raise RuntimeError("Canonical mapping does not cover test and all exposure proteins")
        overlap = {canonical_map[p] for p in test_proteins} & {canonical_map[p] for p in seen_proteins}
        if overlap or set(test.unit_id) & seen_units:
            raise RuntimeError("Benchmark overlaps training/selection exposure: " + str(sorted(overlap)[:10]))
        target_dir = out / name
        if target_dir.exists():
            if not args.force:
                raise RuntimeError("Benchmark output exists; use --force or a new 31_* directory")
            shutil.rmtree(str(target_dir))
        target_dir.mkdir(parents=True)
        dcfg = copy.deepcopy(cfg)
        for field in ("use_pair_pssm", "use_pair_dssp_ss", "use_pair_dssp_rsa", "use_site_pssm", "use_site_dssp_ss", "use_site_dssp_rsa"):
            setattr(dcfg, field, False)
        dcfg.use_esm = True
        # Use actual deployed OSA truncation for BOTH versions, including site tests.
        dcfg.max_site_len = int(cfg.max_pair_len)
        selected = test.iloc[:2] if args.smoke else test
        selected_proteins = set(selected.protein_A) | set(selected.protein_B)
        def selected_esm_loader(path, max_len=0):
            # The legacy loader eagerly reads every embedding in the directory.
            # Load ONLY locked test proteins (two units during smoke), preserving
            # the original dataset/model interfaces and feature order.
            result = {}
            for uid in sorted(selected_proteins):
                file = Path(path) / (uid + ".npy")
                if not file.is_file():
                    raise RuntimeError("Missing locked test embedding: " + str(file))
                array = np.load(file, allow_pickle=False, mmap_mode="r")
                if array.ndim != 2 or array.shape[1] != 1280 or not len(array):
                    raise RuntimeError("Invalid ESM shape: " + str(file))
                used = array[:max_len] if max_len else array
                result[uid] = np.array(used, dtype=np.float32, copy=True)
            return result
        def build_dataset(dataset_class):
            saved_loader = native_loader.load_esm_map
            native_loader.load_esm_map = selected_esm_loader
            try:
                return dataset_class(dcfg)
            finally:
                native_loader.load_esm_map = saved_loader
        if spec["type"] == "pair_fourpack":
            dcfg.pair_fourpack_dir, dcfg.pair_esm_dir = str(directory), str(esm_dir)
            ds = build_dataset(TriagePairDataset)
            if ds.y_col is None:
                raise RuntimeError("Pair benchmark requires explicit binary labels; absent contacts cannot define negatives")
            labels = pd.to_numeric(ds.rows[ds.y_col], errors="coerce")
            if not labels.isin([0, 1]).all():
                raise RuntimeError("Nonbinary/missing pair ground truth")
            ids = [str(x) for x in ds.rows[ds.pair_col]]
        else:
            dcfg.site_global_dir, dcfg.site_esm_dir = str(directory), str(esm_dir)
            ds = build_dataset(TriageSiteDataset)
            ids = [str(x[0]) for x in ds.pairs]
            if spec.get("contact_index_base") != 1:
                raise RuntimeError("Native site loader requires explicitly audited one-based contact indices")
            for _, _, _, contacts in ds.pairs:
                values = contacts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 1).any() or (values != np.floor(values)).any():
                    raise RuntimeError("Invalid native contact indices; do not silently drop annotations")
        if len(set(ids)) != len(ids) or set(test.unit_id) - set(ids):
            raise RuntimeError("Dataset IDs are duplicate or locked test IDs are missing")
        lookup = dict((key, i) for i, key in enumerate(ids))
        prediction_rows, chain_rows, coverage = [], [], []
        for number, entry in enumerate(selected.itertuples(), 1):
            index = lookup[entry.unit_id]
            item = ds[index]
            if (str(item["protein_A"]), str(item["protein_B"])) != (entry.protein_A, entry.protein_B):
                raise RuntimeError("Locked test sample/chain identity mismatch")
            for side in ("A", "B"):
                uid = str(item["protein_" + side])
                raw_path = esm_dir / (uid + ".npy")
                if not raw_path.is_file():
                    raise RuntimeError("Missing per-residue embedding: " + str(raw_path))
                raw = np.load(raw_path, allow_pickle=False)
                if raw.ndim != 2 or raw.shape[1] != 1280 or raw.shape[0] == 0 or not np.isfinite(raw).all():
                    raise RuntimeError("Invalid per-residue ESM embedding")
                if raw.shape[0] != ds.esm[uid].shape[0] and raw.shape[0] < cfg.max_pair_len:
                    raise RuntimeError("Unexpected embedding truncation")
                coverage.append(dict(unit_id=entry.unit_id, protein=uid, side=side,
                    original_length=raw.shape[0], evaluated_length=item["res" + side].shape[0],
                    truncated=raw.shape[0] > cfg.max_pair_len))
                item["res" + side][:, 1280:] = 0
                item["coords" + side] = torch.full((item["res" + side].shape[0], 3), float("nan"))
                if spec["type"] == "site_global":
                    # Preserve ground truth from ALL annotated contacts, including
                    # partners outside the evaluated prefix; cropping must not
                    # turn real interface residues into artificial negatives.
                    contacts = ds.pairs[index][3]
                    column = ds.i_col if side == "A" else ds.j_col
                    positions = pd.to_numeric(contacts[column]).to_numpy(dtype=int) - 1
                    if (positions >= len(raw)).any():
                        raise RuntimeError("Contact index exceeds full embedding length")
                    label = torch.zeros(item["res" + side].shape[0])
                    label[positions[positions < len(label)]] = 1
                    item["y_res_" + side] = label
            batch = batch_to_device(collate_pad([item]), device)
            inputs = dict((key, batch[key]) for key in ("resA", "resB", "maskA", "maskB", "coordsA", "coordsB"))
            for model_name, model in models.items():
                with torch.no_grad():
                    ab, ba = model(**inputs), model(**swapped_inputs(inputs))
                    if spec["type"] == "pair_fourpack":
                        row = dict(unit_id=entry.unit_id, protein_A=entry.protein_A, protein_B=entry.protein_B,
                            model=model_name, output="PPI_logit_symmetric", label=int(item["y_pair"].item()),
                            logit=float(.5 * (ab["logit_triage"] + ba["logit_triage"])))
                        prediction_rows.append(row)
                        if lock.get("l2_support_sensitivity", False):
                            seed = int(hashlib.sha256(entry.unit_id.encode()).hexdigest()[:8], 16)
                            for mode in ("top", "random", "bottom"):
                                aa = support_perturbed_forward(model, inputs, mode, .2, seed, torch)
                                bb = support_perturbed_forward(model, swapped_inputs(inputs), mode, .2, seed, torch)
                                rr = dict(row, output="L2_remove_20pct_" + mode,
                                    logit=float(.5 * (aa["logit_triage"] + bb["logit_triage"])))
                                prediction_rows.append(rr)
                    else:
                        for side, other in (("A", "B"), ("B", "A")):
                            logits = (.5 * (ab["logit_res_" + side] + ba["logit_res_" + other]))[0].cpu().numpy()
                            labels = item["y_res_" + side].numpy().astype(int)
                            chain_rows.append(dict(unit_id=entry.unit_id, side=side, model=model_name, **binary_metrics(labels, logits)))
                            for position, (label, logit) in enumerate(zip(labels, logits), 1):
                                prediction_rows.append(dict(unit_id=entry.unit_id, side=side, residue_index=position,
                                    model=model_name, output="L1_residue_logit_symmetric", label=int(label), logit=float(logit)))
            if number % 25 == 0:
                print("[retention {}] {}/{}".format(name, number, len(selected)), flush=True)
        pd.DataFrame(coverage).to_csv(target_dir / "coverage_and_truncation.tsv", sep="\t", index=False)
        pred = pd.DataFrame(prediction_rows)
        pred.to_csv(target_dir / "predictions.tsv.gz", sep="\t", index=False, compression="gzip")
        if chain_rows:
            pd.DataFrame(chain_rows).to_csv(target_dir / "per_chain_metrics.tsv", sep="\t", index=False)
        rows = []
        for (model_name, output), frame in pred.groupby(["model", "output"]):
            rows.append(dict(dataset=name, model=model_name, output=output,
                             **binary_metrics(frame.label.to_numpy(), frame.logit.to_numpy())))
        pd.DataFrame(rows).to_csv(target_dir / "metrics.tsv", sep="\t", index=False)
        summary_rows.extend(rows)
        write_json(target_dir / "input_audit.json", dict(test_manifest_sha256=sha256_file(test_path),
            exposure_input_sha256=input_hashes, matched_input_policy="ESM 1280 + zero ancillary channels; absent coordinates; deployed max_pair_len",
            max_length=cfg.max_pair_len, test_units=len(selected), protein_overlap=0,
            canonical_id_audit_note=spec["canonical_id_audit_note"],
            homology_independence_proven=False, note="Zero mapped-ID overlap is not proof of sequence-homology independence"))
    pd.DataFrame(summary_rows).to_csv(out / "benchmark_metrics.tsv", sep="\t", index=False)
    write_json(out / "run_summary.json", dict(status="PASS_BENCHMARK_SMOKE_EXECUTION" if args.smoke else "PASS_BENCHMARK_EXECUTION",
        datasets=sorted(datasets_used), v4_embedded_backbone_identical=True, superiority_claim=False,
        clinical_prediction=False, test_threshold_selected=False,
        benchmark_lock=lock, note="Original checkpoint evaluated under matched deployed ESM-only inputs, not native-feature published performance"))
    print("Backbone benchmark execution complete: " + str(out))
    return 0


def audit_suite(source, output):
    """Recompute every reported metric; never delete an unfavourable unit."""
    summary = read_json(source / "run_summary.json")
    contract = read_json(source / "experiment_contract.json")
    if summary["status"] != "PASS_CONTROLLED_EXPERIMENT_EXECUTION":
        raise RuntimeError("Formal source suite has not completed")
    output.mkdir(parents=True, exist_ok=True)
    combined = pd.read_csv(source / "all_metrics.tsv", sep="\t")
    expected = len(contract["variants"]) * len(contract["seeds"]) * int(summary["folds"])
    if len(combined) != expected * 3 or combined.duplicated(["variant", "seed", "fold", "partition"]).any():
        raise RuntimeError("Aggregate metric units missing/duplicated")
    diagnostics, references = [], {}
    names = ["soft_BCE", "Spearman", "weak_top20_AUPRC", "top5_enrichment_ratio", "multi_axis_utility"]
    for fold in range(1, int(summary["folds"]) + 1):
        partitions = pd.read_csv(source / ("fold_%d_pair_partitions.tsv.gz" % fold), sep="\t")
        partitions = partitions.loc[partitions.partition.isin(["train", "validation", "test"])]
        expected_ids = dict(zip(partitions.pair_id, partitions.partition))
        for seed in contract["seeds"]:
            for variant in contract["variants"]:
                unit = source / ("seed_%d" % seed) / variant / ("fold_%d" % fold)
                required = ["completed.json", "adapter.pt", "history.tsv", "metrics.tsv", "predictions.tsv.gz"]
                if not all((unit / f).is_file() and (unit / f).stat().st_size > 0 for f in required):
                    raise RuntimeError("Incomplete synchronized experiment: " + str(unit))
                pred = pd.read_csv(unit / "predictions.tsv.gz", sep="\t").sort_values("pair_id").reset_index(drop=True)
                if pred.pair_id.duplicated().any() or dict(zip(pred.pair_id, pred.partition)) != expected_ids:
                    raise RuntimeError("Pair/partition mismatch: " + str(unit))
                if not np.isfinite(pred[["weak_target", "logit", "base_logit", "heldout_participant_weak_target"]]).all().all():
                    raise RuntimeError("Nonfinite prediction/target: " + str(unit))
                fingerprint = pred[["pair_id", "protein_A", "protein_B", "partition", "weak_target", "heldout_participant_weak_target"]]
                if fold not in references:
                    references[fold] = fingerprint
                elif not fingerprint.equals(references[fold]):
                    raise RuntimeError("Variants/seeds do not share targets, identities and partitions")
                proteins_by_role = {}
                for role, part in pred.groupby("partition"):
                    proteins_by_role[role] = set(part.protein_A) | set(part.protein_B)
                    computed = metrics(part.weak_target.to_numpy(), part.logit.to_numpy(), part.base_logit.to_numpy())
                    reported = combined.loc[(combined.seed == seed) & (combined.fold == fold) &
                        (combined.variant == variant) & (combined.partition == role)]
                    if len(reported) != 1 or any(not np.isclose(computed[k], float(reported.iloc[0][k]), atol=2e-6, rtol=2e-6) for k in names):
                        raise RuntimeError("Recomputed metrics differ from summary: " + str(unit))
                if any(proteins_by_role[a] & proteins_by_role[b] for a, b in
                       (("train", "validation"), ("train", "test"), ("validation", "test"))):
                    raise RuntimeError("Protein overlap across partitions")
                history = pd.read_csv(unit / "history.tsv", sep="\t")
                trained = len(history) > 0
                last_gain = float(history.selection_score.iloc[-1] - history.selection_score.iloc[max(0, len(history) - 11)]) if trained else 0.
                best_epoch = int(history.loc[history.selection_score.idxmax(), "epoch"]) if trained else 0
                diagnostics.append(dict(variant=variant, seed=seed, fold=fold, epochs=len(history), best_epoch=best_epoch,
                    reached_epoch_cap=len(history) == int(contract["optimization"]["max_epochs"]),
                    last10_validation_score_gain=last_gain,
                    still_improving_at_cap=trained and len(history) == int(contract["optimization"]["max_epochs"]) and last_gain > .001))
        print("[suite audit] fold {}/{} complete".format(fold, summary["folds"]), flush=True)
    table = pd.DataFrame(diagnostics)
    table.to_csv(output / "unit_integrity_and_convergence.tsv", sep="\t", index=False)
    test = combined.loc[combined.partition == "test"]
    cols = names + ["structural_mass", "heldout_participant_target_rho"]
    test.groupby("variant")[cols].mean().to_csv(output / "test_means.tsv", sep="\t")
    result = dict(status="PASS_INTEGRITY_WITH_INTERPRETATION_WARNINGS", units=expected, metric_rows=len(combined),
        identical_targets_across_variants_and_seeds=True, protein_disjoint=True,
        still_improving_at_cap=int(table.still_improving_at_cap.sum()),
        full_heldout_participant_target_rho=float(test.loc[test.variant == "full", "heldout_participant_target_rho"].mean()),
        full_minus_no_patient_utility=float(test.loc[test.variant == "full", "multi_axis_utility"].mean() - test.loc[test.variant == "no_patient", "multi_axis_utility"].mean()),
        full_minus_no_patient_AUPRC=float(test.loc[test.variant == "full", "weak_top20_AUPRC"].mean() - test.loc[test.variant == "no_patient", "weak_top20_AUPRC"].mean()),
        warnings=["Original static fusion may be under-optimized; retain it as fixed-budget comparison only",
                  "Weak target is also an input anchor; high internal correlation is not patient-level prediction",
                  "Negative held-out participant background correlation does not establish generalization",
                  "Do not drop unfavourable endpoints or treat repeated folds/seeds as independent patients"])
    write_json(output / "run_summary.json", result)
    return result


def convex_static_fit(x, y, seed, constrained=False):
    """Global 3-expert convex control, trained without validation/test outcomes.

    Same BCE + 0.25 ranking objective family; two prespecified random training
    pairings approximate the stochastic ranking term. Direct simplex weights
    avoid the slow softmax-bias parameterization. Frank-Wolfe gap certifies
    convergence of this finite convex objective, not biological advantage.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) != len(y) or not np.isfinite(x).all():
        raise RuntimeError("Invalid convex fusion input")
    rng = np.random.RandomState(int(seed))
    differences = []
    for _ in range(2):
        permutation = rng.permutation(len(y))
        gap = y - y[permutation]
        valid = np.abs(gap) >= .1
        differences.append((x[valid] - x[permutation[valid]]) * np.sign(gap[valid])[:, None])
    d = np.concatenate(differences, axis=0)
    lower = np.full(3, .01) if constrained else np.zeros(3)
    upper = np.array([.98, .98, .90]) if constrained else np.ones(3)
    def objective(w):
        z = x.dot(w)
        value = np.mean(np.logaddexp(0., z) - y * z)
        gradient = x.T.dot(expit(z) - y) / len(y)
        if len(d):
            dz = d.dot(w)
            value += .25 * np.mean(np.logaddexp(0., -dz))
            gradient -= .25 * d.T.dot(expit(-dz)) / len(d)
        return float(value), gradient
    def linear_oracle(gradient):
        v = lower.copy()
        remaining = 1. - v.sum()
        for i in np.argsort(gradient):
            take = min(remaining, upper[i] - v[i])
            v[i] += take
            remaining -= take
        return v
    attempts = []
    starts = [np.ones(3) / 3., np.array([.05, .05, .90])]
    for start in starts:
        fit = minimize(objective, start, method="SLSQP", jac=True,
            bounds=list(zip(lower, upper)), constraints=[dict(type="eq", fun=lambda w: w.sum() - 1., jac=lambda w: np.ones(3))],
            options=dict(maxiter=500, ftol=1e-12))
        value, gradient = objective(fit.x)
        gap = max(0., float(gradient.dot(fit.x - linear_oracle(gradient))))
        feasible = abs(fit.x.sum() - 1.) < 1e-7 and np.all(fit.x >= lower - 1e-7) and np.all(fit.x <= upper + 1e-7)
        attempts.append(dict(weights=fit.x.tolist(), objective=value, duality_gap=gap,
            iterations=int(fit.nit), solver_success=bool(fit.success), feasible=bool(feasible),
            converged=bool(feasible and gap <= 1e-6)))
    acceptable = [r for r in attempts if r["converged"]]
    chosen = min(acceptable or attempts, key=lambda r: r["objective"])
    return dict(chosen, attempts=attempts, ranking_comparisons=len(d), training_rows=len(y),
        constrained=constrained, selection="training convex objective only; test and validation not used")


def reconstructed_folds(args, root, source):
    """Rebuild the OLD fixed folds; any target drift stops the supplement."""
    from osa_context_data import load_discovery_inputs, compute_node_evidence, build_pair_context, assemble_input_matrix
    from train_triage_osa import sha256_file
    contract = read_json(source / "experiment_contract.json")
    pairs_path = resolve(root, args.pair_scores)
    if sha256_file(pairs_path) != contract["inputs"]["pair_scores"]:
        raise RuntimeError("Frozen pair inputs changed")
    pairs = pd.read_csv(pairs_path, sep="\t", low_memory=False)
    proteins = sorted(set(pairs.protein_A) | set(pairs.protein_B))
    expr, meta, paths = load_discovery_inputs(root, proteins)
    if any(sha256_file(path) != contract["inputs"][key] for key, path in paths.items()):
        raise RuntimeError("Discovery inputs changed since completed suite")
    folds = pd.read_csv(source / "fold_assignments.tsv", sep="\t")
    for fold in range(4):
        def members(dataset, heldout, fold=fold):
            rows = folds.loc[(folds.dataset == dataset) & ((folds.fold == fold) == heldout)]
            return rows.unit_id.astype(str).tolist()
        nodes, contexts = {}, {}
        for role, held in (("development", False), ("heldout", True)):
            nodes[role] = compute_node_evidence(expr, meta, proteins, members("GSE75097", held), members("GSE71356", held))
            contexts[role] = build_pair_context(pairs, nodes[role], information_fusion=True)
        partitions = pd.read_csv(source / ("fold_%d_pair_partitions.tsv.gz" % (fold + 1)), sep="\t").set_index("pair_id")
        partition = pairs.pair_id.map(partitions.partition)
        split = dict((role, np.where(partition.to_numpy() == role)[0]) for role in ("train", "validation", "test"))
        yield fold + 1, pairs, nodes, contexts, assemble_input_matrix(pairs, contexts["development"]), split, meta, members


def run_fusion_and_context_supplements(args, root, source, output):
    from triage_osa_model import INPUT_FEATURE_COLUMNS
    from train_triage_osa import spearman_correlation
    output.mkdir(parents=True, exist_ok=True)
    contract = read_json(source / "experiment_contract.json")
    ix = dict((name, i) for i, name in enumerate(INPUT_FEATURE_COLUMNS))
    rows, node_rows, target_rows, solve_rows = [], [], [], []
    for fold, pairs, nodes, contexts, features, splits, meta, members in reconstructed_folds(args, root, source):
        target = contexts["development"].context_anchor.to_numpy(dtype=float)
        held = contexts["heldout"].context_anchor.to_numpy(dtype=float)
        for label in ["OSA_raw", "severity_raw", "AHI_raw", "IH_raw", "hypertension_signed", "EDS_signed", "discovery_anchor"]:
            a = nodes["development"][label].to_numpy(dtype=float)
            b = nodes["heldout"][label].to_numpy(dtype=float)
            node_rows.append(dict(fold=fold, axis=label, proteins=len(a), finite_development=int(np.isfinite(a).sum()),
                finite_heldout=int(np.isfinite(b).sum()), Spearman=spearman_correlation(a, b),
                absolute_effect_Spearman=spearman_correlation(np.abs(a), np.abs(b)),
                sign_agreement=float(np.mean(np.sign(a[np.isfinite(a) & np.isfinite(b)]) == np.sign(b[np.isfinite(a) & np.isfinite(b)])))
                    if np.any(np.isfinite(a) & np.isfinite(b)) else float("nan")))
        held_meta = meta["GSE75097"].loc[meta["GSE75097"].sample_id.isin(members("GSE75097", True))]
        for role, idx in splits.items():
            target_rows.append(dict(fold=fold, partition=role, pairs=len(idx),
                heldout_participants=len(held_meta), heldout_PS=int((held_meta.study_group == "PS").sum()),
                heldout_IH_subjects=len(members("GSE71356", True)),
                development_vs_heldout_target_rho=spearman_correlation(target[idx], held[idx]),
                development_target_sd=float(np.std(target[idx])), heldout_target_sd=float(np.std(held[idx]))))
        def logit(p):
            p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
            return np.log(p / (1 - p))
        gates = features[:, [ix["g_L1"], ix["g_L2"], ix["g_L3"]]].clip(1e-6)
        gates = .1 + .7 * gates / gates.sum(1, keepdims=True)
        tri = np.sum(gates * features[:, [ix["logit_L1"], ix["logit_L2"], ix["logit_L3"]]], axis=1)
        experts = np.column_stack([features[:, 0], tri, logit(target)])
        direct = dict(context_anchor_direct=logit(target),
            molecular_anchor_direct=logit(contexts["development"].molecular_context_anchor),
            patient_anchor_direct=logit(contexts["development"].clinical_information_anchor),
            frozen_PPI_direct=features[:, 0], frozen_tri_level_direct=tri)
        for seed in contract["seeds"]:
            # Confirm feature/target identity before fitting a new comparator.
            old = pd.read_csv(source / ("seed_%d/full/fold_%d/predictions.tsv.gz" % (seed, fold)), sep="\t").set_index("pair_id")
            indices = np.concatenate(list(splits.values()))
            saved = old.loc[pairs.iloc[indices].pair_id]
            if not np.allclose(saved.weak_target, target[indices], atol=1e-6, rtol=1e-6) or not np.allclose(saved.heldout_participant_weak_target, held[indices], atol=1e-6, rtol=1e-6):
                raise RuntimeError("Reconstructed weak targets differ from completed suite")
            methods = dict(direct)
            for constrained in (False, True):
                name = "static_constrained_converged" if constrained else "static_simplex_converged"
                fit = convex_static_fit(experts[splits["train"]], target[splits["train"]], seed + fold - 1, constrained)
                record = dict(seed=seed, fold=fold, method=name, **fit)
                solve_rows.append(record)
                fit_dir = output / ("seed_%d" % seed) / ("fold_%d" % fold)
                fit_dir.mkdir(parents=True, exist_ok=True)
                write_json(fit_dir / (name + ".json"), record)
                methods[name] = experts.dot(np.asarray(fit["weights"]))
                print("[static convergence] {} seed={} fold={} gap={:.3g} converged={}".format(name, seed, fold, fit["duality_gap"], fit["converged"]), flush=True)
            predictions = []
            for role, idx in splits.items():
                for method, logits in methods.items():
                    row = dict(method=method, seed=seed, fold=fold, partition=role,
                        heldout_participant_target_rho=spearman_correlation(held[idx], logits[idx]), **metrics(target[idx], logits[idx], features[idx, 0]))
                    rows.append(row)
                frame = pairs.iloc[idx][["pair_id", "protein_A", "protein_B"]].copy()
                frame["partition"], frame["weak_target"], frame["heldout_target"] = role, target[idx], held[idx]
                for method, logits in methods.items():
                    frame[method] = logits[idx]
                predictions.append(frame)
            pd.concat(predictions, ignore_index=True).to_csv(fit_dir / "comparator_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
        if args.smoke:
            break
    pd.DataFrame(node_rows).to_csv(output / "node_context_stability.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(output / "pair_target_stability.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(output / "comparator_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(solve_rows).to_csv(output / "solver_audit.tsv", sep="\t", index=False)
    original = pd.read_csv(source / "all_metrics.tsv", sep="\t").rename(columns={"variant": "method"})
    combined = pd.concat([original, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(output / "all_comparisons_including_original.tsv", sep="\t", index=False)
    selected = combined.loc[combined.partition == "test"]
    selected.groupby("method")[["Spearman", "weak_top20_AUPRC", "multi_axis_utility", "heldout_participant_target_rho"]].mean().to_csv(output / "test_means_with_strong_controls.tsv", sep="\t")
    result = dict(status="PASS_CONVERGED_CONTROLS" if all(r["converged"] for r in solve_rows) else "INCOMPLETE_SOLVER_CONVERGENCE",
        fits=len(solve_rows), converged=sum(int(r["converged"]) for r in solve_rows),
        posthoc_optimization_fairness_sensitivity=True, original_results_retained=True,
        validation_or_test_used_to_fit=False, external_validation_read=False,
        direct_baselines_are_deterministic=True, patient_prediction_claim=False,
        note="Controls have train-only convex convergence checks; original 60-epoch static runs remain fixed-budget references")
    write_json(output / "run_summary.json", result)
    return result


def osa_l2_sensitivity(args, root, output, torch):
    """Label-free perturbation on an outcome-independent OSA pair sample."""
    from osa_triage_adapter import load_frozen_checkpoint, swapped_inputs
    from run_osa_symmetric_inference import ESMFeatureCache, make_batch
    n = min(args.l2_pairs, 4) if args.smoke else int(args.l2_pairs)
    if n < 1:
        raise RuntimeError("--l2-pairs must be positive")
    pairs = pd.read_csv(resolve(root, args.pair_scores), sep="\t", low_memory=False)
    pairs["sample_key"] = pairs.pair_id.astype(str).map(lambda p: hashlib.sha256(("OSA-L2-fixed-20260716|" + p).encode()).hexdigest())
    selected = pairs.sort_values("sample_key").iloc[:n].copy()
    esm_dir = root / "model_assets" / "esm2"
    proteins = set(selected.protein_A) | set(selected.protein_B)
    missing = [p for p in sorted(proteins) if not (esm_dir / (p + ".npy")).is_file()]
    output.mkdir(parents=True, exist_ok=True)
    selected[["pair_id", "protein_A", "protein_B", "sample_key"]].to_csv(output / "locked_pair_sample.tsv", sep="\t", index=False)
    if missing:
        result = dict(status="PENDING_ESM_ASSETS", missing_proteins=missing, expected_directory=str(esm_dir),
            note="No replacement sampling allowed due to missing assets")
        write_json(output / "run_summary.json", result)
        return result
    model, cfg, _ = load_frozen_checkpoint(root / BASE, torch.device(args.device))
    cache = ESMFeatureCache(np, esm_dir, cfg.max_pair_len, 32)
    rows = []
    for number, row in enumerate(selected.to_dict("records"), 1):
        # Registry may report length under another column; compare the cached
        # actual prefix and saved PPI output before accepting any perturbation.
        for side in ("A", "B"):
            key = "length_" + side
            if key not in row:
                row[key] = cache.get(row["protein_" + side])[1]
        inputs, la, lb = make_batch(torch, np, cache, [row], cfg.d_res_in, torch.device(args.device))
        with torch.no_grad():
            ab, ba = model(**inputs), model(**swapped_inputs(inputs))
            base = float((.5 * (ab["logit_triage"] + ba["logit_triage"])).item())
            if not np.isclose(base, float(row["logit_PPI"]), atol=.002, rtol=1e-4):
                raise RuntimeError("Recomputed OSA PPI differs from frozen inference: " + str(row["pair_id"]))
            baseline_l2 = float((.5 * (ab["logit_interface"] + ba["logit_interface"])).item())
            seed = int(row["sample_key"][:8], 16)
            for mode in ("top", "random", "bottom"):
                for repeat in range(3 if mode == "random" else 1):
                    aa = support_perturbed_forward(model, inputs, mode, .2, (seed + repeat) % (2 ** 32), torch)
                    bb = support_perturbed_forward(model, swapped_inputs(inputs), mode, .2, (seed + repeat) % (2 ** 32), torch)
                    for key in ("logit_res_A", "logit_res_B", "logit_pair_raw"):
                        if not torch.equal(ab[key], aa[key]) or not torch.equal(ba[key], bb[key]):
                            raise RuntimeError("L2 perturbation changed native L1 or L3 computation")
                    changed = float((.5 * (aa["logit_triage"] + bb["logit_triage"])).item())
                    changed_l2 = float((.5 * (aa["logit_interface"] + bb["logit_interface"])).item())
                    rows.append(dict(pair_id=row["pair_id"], protein_A=row["protein_A"], protein_B=row["protein_B"],
                        mode=mode, repeat=repeat, support_fraction_removed=.2, used_length_A=la[0], used_length_B=lb[0],
                        base_PPI_logit=base, perturbed_PPI_logit=changed, delta_PPI_logit=changed - base,
                        delta_L2_logit=changed_l2 - baseline_l2, absolute_PPI_change=abs(changed - base)))
        if number % 16 == 0:
            print("[OSA L2 sensitivity] {}/{}".format(number, len(selected)), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pair_perturbations.tsv", sep="\t", index=False)
    per_pair = frame.groupby(["pair_id", "mode"])[["delta_PPI_logit", "delta_L2_logit", "absolute_PPI_change"]].mean().reset_index()
    per_pair.to_csv(output / "per_pair_averaged_random.tsv", sep="\t", index=False)
    per_pair.groupby("mode")[["delta_PPI_logit", "delta_L2_logit", "absolute_PPI_change"]].agg(["mean", "median", "std"]).to_csv(output / "perturbation_summary.tsv", sep="\t")
    result = dict(status="PASS_L2_PERTURBATION_EXECUTION", pairs=len(selected), random_repeats=3,
        labels_used=False, contact_accuracy_claim=False, no_OSA_outcome_used_for_sampling=True,
        L1_L3_outputs_unchanged=True, note="Internal dependence on selected L2 evidence; not true interaction/contact validation")
    write_json(output / "run_summary.json", result)
    return result


def collect_benchmark_inventory(triage_root, output):
    """Collect small split/protocol records, never invent an exposure manifest."""
    triage_root = Path(triage_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if triage_root.is_dir():
        for folder in ("triage_runs", "triage_if_extra", "pair_data"):
            directory = triage_root / folder
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in (".tsv", ".csv", ".json", ".txt"):
                    continue
                if not any(word in path.name.lower() for word in ("split", "partition", "manifest", "contract", "overlap")):
                    continue
                relative = path.relative_to(triage_root)
                # Avoid recursive symlink targets outside the user-scoped source.
                if triage_root.resolve() not in path.resolve().parents:
                    continue
                copied = path.stat().st_size <= 2 * 1024 * 1024
                if copied:
                    destination = output / "source_records" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(path), str(destination))
                rows.append(dict(source=str(path), relative_path=str(relative), bytes=path.stat().st_size, copied=copied))
    pd.DataFrame(rows, columns=["source", "relative_path", "bytes", "copied"]).to_csv(output / "split_record_inventory.tsv", sep="\t", index=False)
    result = dict(status="COLLECTED_EXPOSURE_AUDIT_REQUIRED" if rows else "PENDING_ORIGINAL_PROJECT_RECORDS",
        records=len(rows), root=str(triage_root), independent_holdout_certified=False,
        note="Records require review across every training/selection stage; collection is not proof of an independent benchmark")
    write_json(output / "run_summary.json", result)
    return result


def audit_historical_tuna_split(shapes_path, split_path, labels_path, output):
    """Export observed exposure, not an invented all-stage independence certificate."""
    output.mkdir(parents=True, exist_ok=True)
    shapes = pd.read_csv(shapes_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"pair_id", "A_uid", "B_uid"}
    if not required.issubset(shapes.columns) or shapes.empty or shapes.pair_id.duplicated().any():
        raise RuntimeError("Historical shapes require unique pair_id, A_uid and B_uid")
    if (shapes[list(required)] == "").any().any():
        raise RuntimeError("Empty historical pair identifiers")
    saved = read_json(split_path)
    if saved.get("dataset_fingerprint", {}).get("n_rows") != len(shapes):
        raise RuntimeError("Historical split/shapes row count mismatch")
    groups, used, registries = {}, set(), []
    for role in ("train", "val", "test"):
        spec = saved["splits"][role]
        indices = spec["indices"]
        if any(type(i) is not int or i < 0 or i >= len(shapes) for i in indices):
            raise RuntimeError("Invalid historical split indices")
        if len(indices) != len(set(indices)) or used & set(indices):
            raise RuntimeError("Duplicated/overlapping historical row indices")
        used.update(indices)
        frame = shapes.iloc[indices].copy()
        if frame.pair_id.tolist() != spec["pair_ids"] or len(frame) != spec["n"]:
            raise RuntimeError("Historical split pair identities/order mismatch")
        frame["partition"] = role
        registries.append(frame)
        groups[role] = set(frame.A_uid) | set(frame.B_uid)
    if len(used) != len(shapes):
        raise RuntimeError("Historical split does not partition the complete registry")
    registry = pd.concat(registries, ignore_index=True)
    labels_status = "MISSING_EXPLICIT_LABEL_SOURCE"
    label_counts = []
    if labels_path.is_file():
        # Original TUnA interaction file is headerless; never infer negatives
        # from missing residue contacts or from previously predicted scores.
        labels = pd.read_csv(labels_path, sep="\t", header=None, dtype=str, keep_default_na=False)
        if labels.shape[1] != 3:
            raise RuntimeError("Expected three-column headerless TUnA interaction labels")
        labels.columns = ["A", "B", "label"]
        y = pd.to_numeric(labels.label, errors="coerce")
        if not y.isin([0, 1]).all() or (labels[["A", "B"]] == "").any().any():
            raise RuntimeError("Missing/nonbinary historical interaction labels")
        mapping = {}
        for a, b, value in zip(labels.A, labels.B, y):
            key = tuple(sorted((a, b)))
            if key in mapping and mapping[key] != int(value):
                raise RuntimeError("Conflicting symmetric pair labels")
            mapping[key] = int(value)
        keys = [tuple(sorted((a, b))) for a, b in zip(registry.A_uid, registry.B_uid)]
        if any(k not in mapping for k in keys):
            raise RuntimeError("Explicit label file does not cover historical pair registry")
        registry["label"] = [mapping[k] for k in keys]
        for role, frame in registry.groupby("partition"):
            spec = saved["splits"][role]
            label_counts.append(dict(partition=role, explicit_positive=int(frame.label.sum()),
                recorded_positive=int(spec["positive"]), explicit_negative=int((frame.label == 0).sum()),
                recorded_negative=int(spec["negative"])))
        labels_status = ("LABEL_COUNTS_DIFFER_FROM_HISTORICAL_RECORD" if any(
            r["explicit_positive"] != r["recorded_positive"] for r in label_counts)
            else "PASS_EXPLICIT_BINARY_LABELS")
    else:
        registry["label"] = np.nan
    pd.DataFrame(label_counts, columns=["partition", "explicit_positive", "recorded_positive",
        "explicit_negative", "recorded_negative"]).to_csv(output / "label_count_audit.tsv", sep="\t", index=False)
    contact_mismatches = None
    contacts_path = shapes_path.parent / "tuna.contacts.tsv"
    if contacts_path.is_file() and registry.label.notna().all():
        contact = pd.read_csv(contacts_path, sep="\t", dtype=str, keep_default_na=False)
        if {"pair_id", "label"}.issubset(contact.columns):
            values = pd.to_numeric(contact.label, errors="coerce")
            if not values.isin([0, 1]).all():
                raise RuntimeError("Nonbinary contact-file pair labels")
            contact["label"] = values
            # Reproduce the native loader's observed label semantics solely for audit.
            # Missing contacts are NOT accepted as ground-truth negatives.
            native = contact.groupby("pair_id").label.max()
            registry["native_contact_label_observed"] = registry.pair_id.map(native).fillna(0).astype(int)
            registry["native_contact_record_present"] = registry.pair_id.isin(native.index)
            difference = registry.loc[registry.label != registry.native_contact_label_observed].copy()
            difference.to_csv(output / "native_vs_explicit_label_differences.tsv", sep="\t", index=False)
            contact_mismatches = len(difference)
    exposed = groups["train"] | groups["val"]
    registry["A_seen_in_observed_train_or_val"] = registry.A_uid.isin(exposed)
    registry["B_seen_in_observed_train_or_val"] = registry.B_uid.isin(exposed)
    registry["both_proteins_unseen_in_observed_train_or_val"] = ~(
        registry.A_seen_in_observed_train_or_val | registry.B_seen_in_observed_train_or_val)
    # This table includes all rows, including unfavorable/overlapping ones.
    registry.to_csv(output / "historical_pair_partition_registry.tsv.gz", sep="\t", index=False, compression="gzip")
    test = registry.loc[registry.partition == "test"].copy()
    test.to_csv(output / "historical_pair_test_registry.tsv", sep="\t", index=False)
    prediction_path = split_path.parent / "tuna_test_fusion_predictions.tsv"
    historical_test_label_mismatches = None
    if prediction_path.is_file() and test.label.notna().all():
        old = pd.read_csv(prediction_path, sep="\t", dtype={"pair_id": str})
        required_old = {"pair_id", "protein_A", "protein_B", "label", "logit_triage"}
        if not required_old.issubset(old.columns) or old.pair_id.duplicated().any() or set(old.pair_id) != set(test.pair_id):
            raise RuntimeError("Historical test prediction identity coverage mismatch")
        compare = old.merge(test[["pair_id", "A_uid", "B_uid", "label"]], on="pair_id", suffixes=("_historical", "_explicit"), validate="one_to_one")
        if not ((compare.protein_A == compare.A_uid) & (compare.protein_B == compare.B_uid)).all():
            raise RuntimeError("Historical test prediction protein identity mismatch")
        compare["label_disagrees"] = compare.label_historical != compare.label_explicit
        historical_test_label_mismatches = int(compare.label_disagrees.sum())
        compare[["pair_id", "protein_A", "protein_B", "label_historical", "label_explicit",
            "logit_triage", "label_disagrees"]].to_csv(output / "historical_test_label_audit.tsv", sep="\t", index=False)
        scores = []
        for source in ("historical", "explicit"):
            scores.append(dict(label_source=source, prediction_source="old_original_project_export_NOT_OSA_backbone",
                independence="not_certified", **binary_metrics(compare["label_" + source].to_numpy(), compare.logit_triage.to_numpy())))
        pd.DataFrame(scores).to_csv(output / "historical_predictions_label_sensitivity.tsv", sep="\t", index=False)
    rows = []
    for role in ("train", "val", "test"):
        rows.append(dict(partition=role, pairs=int((registry.partition == role).sum()), proteins=len(groups[role])))
    pd.DataFrame(rows).to_csv(output / "historical_split_counts.tsv", sep="\t", index=False)
    result = dict(status="AUDIT_COMPLETE_LABEL_LINEAGE_REVIEW_REQUIRED" if labels_status == "LABEL_COUNTS_DIFFER_FROM_HISTORICAL_RECORD" else "PASS_HISTORICAL_SPLIT_AUDIT", labels_status=labels_status,
        native_contact_vs_explicit_label_mismatches=contact_mismatches,
        historical_test_label_mismatches=historical_test_label_mismatches,
        seed=saved.get("seed"), declared_split_mode=saved.get("split_mode"),
        train_val_protein_overlap=len(groups["train"] & groups["val"]),
        train_test_protein_overlap=len(groups["train"] & groups["test"]),
        val_test_protein_overlap=len(groups["val"] & groups["test"]),
        test_proteins=len(groups["test"]), test_pairs=len(test),
        test_pairs_both_proteins_unseen=int(test.both_proteins_unseen_in_observed_train_or_val.sum()),
        all_pairs_both_proteins_unseen=int(registry.both_proteins_unseen_in_observed_train_or_val.sum()),
        all_training_stages_accounted_for=False, independent_holdout_certified=False,
        note="Observed historical split only; checkpoint lineage and earlier structural training still require audit")
    write_json(output / "run_summary.json", result)
    return result


def prepare_benchmarks(args, root, parent):
    """Small CPU material package; never declares missing exposure data complete."""
    triage = Path(args.triage_root).resolve()
    out = parent / "benchmark_preparation_v1"
    split = triage / "triage_if_extra/tuna_fusion_analysis/tuna_split_seed42_protein_component.json"
    shapes = triage / "pair_data/TUnA_Intra0/fourpack/tuna.shapes.tsv"
    labels = triage / "pair_data/Intra0_interaction.tsv"
    if args.check_env:
        print("Benchmark preparation: CPU-only; no embeddings or model inference")
        print("Original project: {}".format(triage))
        for path in (split, shapes, labels):
            print("{}: {}".format("FOUND" if path.is_file() else "MISSING", path))
        return 0
    if out.exists() and not args.force:
        raise RuntimeError("Benchmark preparation exists; --force refreshes only this metadata package")
    if out.exists():
        safe_directory(root, out)
        shutil.rmtree(str(out))
    out.mkdir(parents=True)
    report = dict(status="PENDING_HISTORICAL_SPLIT_INPUTS")
    if split.is_file() and shapes.is_file():
        try:
            report = audit_historical_tuna_split(shapes, split, labels, out / "historical_tuna")
        except Exception as error:
            report = dict(status="FAILED_HISTORICAL_SPLIT_AUDIT", error=str(error))
    # Copy full explicit source records, including the previously size-skipped split.
    # Raw embeddings and checkpoints remain in place; no expensive inference occurs.
    records = [split, shapes, labels,
        triage / "site_data/site_global/contacts.tsv",
        triage / "site_data/site_global/proteins.fasta",
        triage / "site_data/site_global/seq.all.fasta",
        triage / "pair_data/TUnA_Intra0/fourpack/tuna.seq.fasta",
        triage / "Dest_prepared/train.txt", triage / "Dest_prepared/val.txt",
        triage / "Dest_prepared/test.txt", triage / "Dest_prepared/manifest.tsv",
        triage / "train_triage.py", triage / "triage_config.py"]
    # new2/Dest_prepared is referenced by the saved original project configuration.
    dest = triage.parent / "new2/Dest_prepared"
    records.extend(dest / name for name in ("train.txt", "val.txt", "test.txt", "manifest.tsv",
        "dest_manifest.tsv", "dest_manifest.json", "deepppisp_split_manifest.json"))
    records.extend([root / "model/triage_server_candidate/train_triage.py",
        root / "model/osa_adapter/train_osa_esm_only_adaptation.py",
        root / "results/discovery_preprocess_v2/06_model_training/esm_only_adaptation_v1/feature_adaptation_contract.json"])
    rows = []
    for path in records:
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        copied = False
        if exists and size <= 64 * 1024 * 1024 and triage.parent in path.resolve().parents:
            target = out / "source_records" / path.relative_to(triage.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
            copied = True
        rows.append(dict(path=str(path), exists=exists, bytes=size, copied=copied,
                         role="source_record", note="" if copied else "missing or over 64 MiB or external symlink"))
    for path in (triage / "pair_data/TUnA_Intra0/fourpack/emb/esm2",
                 triage / "esm_embeddings/site", triage / "esm_embeddings/dest",
                 dest / "labels", dest / "seq"):
        rows.append(dict(path=str(path), exists=path.is_dir(), bytes=None, copied=False,
            role="native_asset_directory", files=sum(1 for f in path.glob("*") if f.is_file()) if path.is_dir() else 0,
            note="Inventory only; asset content/identity not yet certified"))
    pd.DataFrame(rows).to_csv(out / "native_asset_inventory.tsv", sep="\t", index=False)
    checkpoint_rows = []
    checkpoint_paths = {"original_engineering": triage / "triage_runs/triage_final.pt",
        "original_tuna": triage / "triage_runs/tuna_from_engineering/triage_tuna_pair_finetuned.pt",
        "esm_adaptation": root / BASE, "OSA_V4": root / V4}
    for name, path in checkpoint_paths.items():
        record = dict(name=name, path=str(path), exists=path.is_file(), status="MISSING")
        if path.is_file():
            try:
                import torch
                # User-owned project checkpoints. Metadata only; never instantiate,
                # evaluate or alter a model, and never write checkpoint weights.
                try:
                    payload = torch.load(str(path), map_location="cpu", weights_only=False)
                except TypeError:  # PyTorch 1.13 / Python 3.7 compatibility.
                    payload = torch.load(str(path), map_location="cpu")
                if not isinstance(payload, dict):
                    raise RuntimeError("Checkpoint is not a metadata dictionary")
                metadata = dict((k, payload[k]) for k in ("stage", "epoch", "config", "best_metric",
                    "feature_spec", "base_checkpoint", "base_checkpoint_sha256", "base_model_config",
                    "training_contract", "feature_adaptation_contract") if k in payload)
                metadata["available_top_level_keys"] = sorted(str(k) for k in payload)
                metadata["checkpoint_path"] = str(path)
                write_json(out / (name + "_checkpoint_metadata.json"), metadata)
                record["status"] = "METADATA_EXTRACTED_NO_INFERENCE"
                del payload
            except Exception as error:
                record["status"] = "METADATA_READ_FAILED"
                record["error"] = str(error)
        checkpoint_rows.append(record)
    pd.DataFrame(checkpoint_rows).to_csv(out / "checkpoint_metadata_inventory.tsv", sep="\t", index=False)
    requirements = [
        ("checkpoint_lineage", "Original pretraining and model-selection exposures before TUnA", "Earlier checkpoints/run configuration and protein/unit membership"),
        ("adaptation_exposure", "Formal ESM adaptation partition membership", "Saved or source-reconstructed split with exact dataset/config agreement"),
        ("canonical_mapping", "Common protein identity across PDB chains and accessions", "Audited mapping and sequence/homology scope; no guessed aliases"),
        ("residue_ground_truth", "Real residue labels and native inputs", "Site contacts with index convention, or explicit residue labels with sequence/partner protocol"),
        ("pair_ground_truth", "Unexposed labeled protein pairs", "Both endpoints outside all training/selection exposure; old TUnA split alone is insufficient")]
    pd.DataFrame(requirements, columns=["requirement", "purpose", "required_material"]).to_csv(out / "remaining_requirements.tsv", sep="\t", index=False)
    summary = dict(status="BENCHMARK_PREPARATION_COMPLETE_HOLDOUT_NOT_CERTIFIED",
        historical_tuna=report, records=len(rows), copied_source_files=sum(int(r["copied"]) for r in rows),
        checkpoint_metadata=checkpoint_rows,
        training_performed=False, inference_performed=False, independent_holdout_certified=False,
        benchmark_lock_generated=False, frozen_V4_changed=False,
        next_step="Review label source and checkpoint exposure lineage; no automatic relaxation to an overlapping benchmark")
    if report.get("labels_status") == "LABEL_COUNTS_DIFFER_FROM_HISTORICAL_RECORD":
        summary["status"] = "BENCHMARK_PREPARATION_COMPLETE_LABEL_LINEAGE_REVIEW_REQUIRED"
    if report["status"].startswith("FAILED"):
        summary["status"] = "BENCHMARK_PREPARATION_NEEDS_ATTENTION"
    write_json(out / "run_summary.json", summary)
    print("Benchmark preparation: " + summary["status"])
    print("Historical test pairs with two unexposed proteins: " + str(report.get("test_pairs_both_proteins_unseen", "unknown")))
    print("Output: " + str(out))
    print("No training or inference; true residue/pair benchmarks remain pending exposure certification.")
    return 1 if report["status"].startswith("FAILED") else 0


def complete_supplements(args, root, parent, torch):
    """One server command; independent pending tasks do not hide completed ones."""
    import scipy
    from train_triage_osa import sha256_file
    source = parent / "osa_ablation_suite"
    if not (source / "run_summary.json").is_file():
        raise RuntimeError("Missing completed formal suite")
    source_status = read_json(source / "run_summary.json")
    if source_status.get("completed_units") != 84 or source_status.get("status") != "PASS_CONTROLLED_EXPERIMENT_EXECUTION":
        raise RuntimeError("Complete the locked seven-configuration, four-fold, three-seed suite first")
    output = parent / ("supplement_completion_v1_smoke" if args.smoke else "supplement_completion_v1")
    protocol = dict(version="1.1.0", smoke=bool(args.smoke), l2_pairs=int(args.l2_pairs),
        source_contract=sha256_file(source / "experiment_contract.json"), source_metrics=sha256_file(source / "all_metrics.tsv"),
        source_code=sha256_file(Path(__file__)), scipy_version=scipy.__version__,
        strong_controls="Train-only convex simplex, constrained simplex; direct anchor and backbone controls",
        context_diagnostic="Same frozen folds and candidates; report negative correlations unchanged",
        benchmark_lock=str(args.benchmark_lock), external_validation_read=False,
        original_outputs_retained=True, inference="Post-hoc technical fairness checks, not newly prespecified confirmation")
    if args.check_env:
        print("Supplement completion preflight: PASS")
        print("SciPy: {} ; formal source units: 84 ; device: {}".format(scipy.__version__, args.device))
        print("ESM directory present: {}".format((root / "model_assets/esm2").is_dir()))
        print("Original project present: {}".format(Path(args.triage_root).is_dir()))
        print("True residue/pair benchmark manifest: {}".format(args.benchmark_lock or "pending; not assumed"))
        return 0
    if output.exists():
        if args.resume_experiments:
            if not (output / "supplement_contract.json").is_file() or read_json(output / "supplement_contract.json") != protocol:
                raise RuntimeError("Supplement resume contract changed")
        elif args.force:
            shutil.rmtree(str(output))  # parent constrained to Stage 31 by run()
        else:
            raise RuntimeError("Supplement output exists; use --resume-experiments")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "supplement_contract.json", protocol)
    results = {}
    def step(name, function):
        path = output / name
        path.mkdir(parents=True, exist_ok=True)
        done = path / "step_complete.json"
        if args.resume_experiments and done.is_file():
            results[name] = read_json(done)
            return
        print("[supplement] " + name, flush=True)
        try:
            result = function(path)
        except Exception as error:
            result = dict(status="FAILED", error=type(error).__name__ + ": " + str(error))
            write_json(path / "failure.json", result)
        results[name] = result
        # Pending/failed steps will be tried again on resume, not silently skipped.
        if result["status"].startswith(("PASS", "COLLECTED")):
            write_json(done, result)
        write_json(output / "run_summary.json", dict(status="RUNNING", steps=results))
    step("01_suite_audit", lambda path: audit_suite(source, path))
    if results["01_suite_audit"]["status"] == "FAILED":
        write_json(output / "run_summary.json", dict(status="FAILED_SOURCE_AUDIT", steps=results))
        return 1
    step("02_fusion_convergence_and_context", lambda path: run_fusion_and_context_supplements(args, root, source, path))
    def identity(path):
        result, model, cfg = retention_identity(root, torch)
        write_json(path / "run_summary.json", result)
        return result
    step("03_backbone_identity", identity)
    step("04_OSA_L2_sensitivity", lambda path: osa_l2_sensitivity(args, root, path, torch))
    step("05_benchmark_inventory", lambda path: collect_benchmark_inventory(args.triage_root, path))
    if args.benchmark_lock:
        def benchmarks(path):
            identity_result, model, cfg = retention_identity(root, torch)
            if identity_result["status"] != "PASS_BACKBONE_IDENTITY":
                raise RuntimeError("Backbone identity failed")
            run_benchmarks(args, root, path, model, cfg, torch)
            return read_json(path / "run_summary.json")
        step("06_true_backbone_benchmarks", benchmarks)
    else:
        results["06_true_backbone_benchmarks"] = dict(status="PENDING_AUDITED_HOLDOUT_MANIFEST",
            note="Do not infer accuracy from weight identity or from label-free L2 perturbation")
    failed = [k for k, r in results.items() if r["status"].startswith(("FAIL", "INCOMPLETE"))]
    pending = [k for k, r in results.items() if r["status"].startswith("PENDING")]
    status = "NEEDS_ATTENTION" if failed else ("AVAILABLE_SUPPLEMENTS_COMPLETE_BENCHMARK_PENDING" if pending else "PASS_SUPPLEMENT_EXECUTION")
    write_json(output / "run_summary.json", dict(status=status, steps=results, failed=failed, pending=pending,
        biological_superiority_claim=False, original_results_deleted=False, frozen_V4_changed=False))
    print("Supplement completion: " + status)
    print("Output: " + str(output))
    return 1 if failed else 0


def run(args):
    root = Path(args.root).resolve()
    if args.force and args.resume_experiments:
        raise RuntimeError("Choose --force OR --resume-experiments, not both")
    if getattr(args, "prepare_benchmarks", False):
        return prepare_benchmarks(args, root, safe_directory(root, args.experiment_dir))
    import torch
    if args.experiment_self_test:
        self_test(torch)
        return 0
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for local tests")
    out = safe_directory(root, args.experiment_dir)
    if getattr(args, "audit_experiments", False):
        audit_suite(out / "osa_ablation_suite", out / "postrun_audit")
        return 0
    if getattr(args, "complete_supplements", False):
        return complete_supplements(args, root, out, torch)
    if args.backbone_retention:
        out = out / "backbone_retention"
        result, model, cfg = retention_identity(root, torch)
        print("Backbone identity: " + result["status"])
        if result["status"].startswith("FAIL"):
            raise RuntimeError("V4 backbone differs from locked ESM-only checkpoint")
        if args.check_env:
            print("Benchmark lock: " + str(args.benchmark_lock or "not supplied; accuracy tests not yet authorized by a holdout manifest"))
            return 0
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "backbone_identity.json", result)
        if args.benchmark_lock:
            benchmark_out = out / ("smoke" if args.smoke else "benchmarks")
            benchmark_out.mkdir(parents=True, exist_ok=True)
            return run_benchmarks(args, root, benchmark_out, model, cfg, torch)
        write_json(out / "run_summary.json", dict(status="PASS_IDENTITY_BENCHMARK_INPUT_REQUIRED", identity=result,
            next_step="Provide --benchmark-lock after auditing all training/adaptation/selection exposure"))
        print("Residue/pair accuracy pending: a locked holdout manifest is required, not assumed.")
        return 0
    out = out / "osa_ablation_suite"
    if args.smoke:
        out = out / "smoke"
    return run_suite(args, root, out, torch)
