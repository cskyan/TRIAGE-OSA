#!/usr/bin/env python3
"""Paired fusion-capacity controls using the archived discovery protocol.

No production model, old result directory or manuscript is changed. Training
and external evaluation are separate phases. Python 3.7+ compatible.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit

VERSION = "1.0.0"
DEFAULT_CONFIG = "config/osa_fusion_capacity_control_v1.json"
BASE = "results/discovery_preprocess_v2"
SOURCE_ORDER = BASE + "/38_external_strong_controls_v1/rankings_primary_250.tsv.gz"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    from train_triage_osa import json_safe
    path = Path(path)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output_artifacts(folder, excluded_directories=()):
    return dict((p.relative_to(folder).as_posix(), sha(p))
        for p in sorted(folder.rglob("*")) if p.is_file()
        and p != folder/"run_summary.json"
        and not any(part in excluded_directories for part in p.relative_to(folder).parts))


def verify_artifacts(folder, artifacts):
    if not artifacts:
        raise RuntimeError("Missing completed-output artifact manifest")
    for relative, digest in artifacts.items():
        path = folder/relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError("Completed output changed or missing: " + str(path))


def resolve(root, path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def safe_output(root, path):
    value = resolve(root, path)
    allowed = (root / BASE).resolve()
    if allowed not in value.parents or not value.relative_to(allowed).parts[0].startswith("50_"):
        raise RuntimeError("Outputs must be in a new results/discovery_preprocess_v2/50_* directory")
    return value


def load_checkpoint(path, torch):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def model_from_config(config):
    if config.get("model_type") == "generic_fusion_mlp":
        from osa_fusion_capacity_models import GenericFusionMLP
        return GenericFusionMLP.from_config(config)
    from triage_osa_model import TRIAGEOSAModel
    return TRIAGEOSAModel.from_config(config)


def make_model(spec, mean, std):
    if spec["architecture"] == "structured":
        from triage_osa_model import TRIAGEOSAModel
        return TRIAGEOSAModel(mean, std, hidden_dim=spec["hidden_dim"], dropout=0.10)
    from osa_fusion_capacity_models import GenericFusionMLP
    return GenericFusionMLP(mean, std, hidden_dim=spec["hidden_dim"], dropout=0.10,
                            residual_skip=spec.get("residual_skip", False))


def ranking_loss(logits, target, permutation, gap, torch):
    difference = target - target[permutation]
    mask = difference.abs() >= float(gap)
    if bool(mask.any().item()):
        return torch.nn.functional.softplus(-torch.sign(difference[mask]) *
            (logits[mask] - logits[permutation][mask])).mean()
    return logits.sum() * 0.0


def fit_common(features, target, train, validation, spec, optimization, objective,
               seed, learning_rate, device, torch, fixed_epochs=None, prefix=""):
    """No fabricated mass/gate variables. Both architectures use this loss."""
    from train_triage_osa import set_seeds, model_predictions, tensor_batches
    from osa_controlled_experiments import metrics
    set_seeds(seed, torch)
    mean = features[train].mean(axis=0).astype(np.float32)
    std = features[train].std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    model = make_model(spec, mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate),
                                 weight_decay=float(optimization["weight_decay"]))
    # Model initialization and dropout never determine the rank-pair RNG.
    batch_rng = np.random.RandomState(int(seed))
    pair_rng = np.random.RandomState(int(seed) + 7000003)
    set_seeds(seed, torch)
    history, best_state, best_epoch, stale = [], None, 0, 0
    best_score = -float("inf")
    epochs = int(fixed_epochs or optimization["max_epochs"])
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for ids in tensor_batches(train, optimization["batch_size"], batch_rng):
            x = torch.from_numpy(features[ids]).to(device)
            y = torch.from_numpy(target[ids].astype(np.float32)).to(device)
            z = model(x)["adapted_logit"]
            permutation = torch.as_tensor(pair_rng.permutation(len(ids)), dtype=torch.long, device=device)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            rank = ranking_loss(z, y, permutation, objective["minimum_ranking_target_gap"], torch)
            loss = objective["bce_weight"] * bce + objective["ranking_weight"] * rank
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        if validation is not None and len(validation):
            z = model_predictions(model, features, validation, optimization["batch_size"], device, torch)
            result = metrics(target[validation], z, features[validation, 0])
            score = float(result["multi_axis_utility"])
        else:
            result = {"soft_BCE": float("nan"), "Spearman": float("nan")}
            score = float(epoch)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(epoch_losses)),
            selection_score=score, validation_soft_BCE=result["soft_BCE"],
            validation_Spearman=result["Spearman"], elapsed_seconds=time.time()-started))
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation selection score")
        if score > best_score + optimization["minimum_improvement"]:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print("{} epoch={} loss={:.6f} validation={:.6f}".format(
                prefix, epoch, history[-1]["train_loss"], score), flush=True)
        if fixed_epochs is None and stale >= optimization["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    return model, history, int(best_epoch), float(best_score)


def save_model(path, model, metadata, torch):
    torch.save(dict(model_config=model.export_config(),
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        metadata=metadata, eligible_for_production=False), str(path))


def restore_model(path, device, torch):
    value = load_checkpoint(path, torch)
    model = model_from_config(value["model_config"])
    model.load_state_dict(value["state_dict"], strict=True)
    return model.to(device)


def primary_mask(pairs, primary_proteins):
    return (pairs.protein_A.isin(primary_proteins) & pairs.protein_B.isin(primary_proteins)).to_numpy()


def prediction_tables(model, features, target, heldout, pairs, splits, is_primary,
                      identity, batch_size, device, torch):
    from train_triage_osa import model_predictions, spearman_correlation
    from osa_controlled_experiments import metrics
    rows, frames = [], []
    for partition, indices in splits.items():
        z = model_predictions(model, features, indices, batch_size, device, torch)
        frame = pairs.iloc[indices][["pair_id", "protein_A", "protein_B"]].copy()
        frame["partition"], frame["weak_target"], frame["logit"] = partition, target[indices], z
        frame["base_logit"] = features[indices, 0]
        frame["heldout_participant_weak_target"] = heldout[indices]
        frame["primary_pair"] = is_primary[indices]
        frames.append(frame)
        for scope, keep in (("all_500", np.ones(len(indices), dtype=bool)),
                            ("primary_250", is_primary[indices])):
            if keep.sum() < 2:
                continue
            row = dict(identity, partition=partition, scope=scope,
                       **metrics(target[indices][keep], z[keep], features[indices, 0][keep]))
            row["heldout_participant_target_rho"] = spearman_correlation(heldout[indices][keep], z[keep])
            rows.append(row)
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def verify_fold(source, fold, seeds, pairs, features, target, heldout, splits, torch):
    """Check all identities/targets; replay up to256 archived Full predictions."""
    from train_triage_osa import model_predictions
    included = np.concatenate([splits[r] for r in ("train", "validation", "test")])
    expected_role = dict((str(pairs.iloc[i].pair_id), r) for r, ids in splits.items() for i in ids)
    index = dict((str(p), i) for i, p in enumerate(pairs.pair_id))
    records = []
    for seed in seeds:
        unit = source / ("seed_%d/full/fold_%d" % (seed, fold))
        saved = pd.read_csv(unit / "predictions.tsv.gz", sep="\t")
        if saved.pair_id.duplicated().any() or dict(zip(saved.pair_id.astype(str), saved.partition)) != expected_role:
            raise RuntimeError("Archived pair identities or partitions differ")
        ids = np.asarray([index[str(p)] for p in saved.pair_id], dtype=int)
        for observed, expected in ((saved.weak_target.to_numpy(), target[ids]),
                (saved.heldout_participant_weak_target.to_numpy(), heldout[ids]),
                (saved.base_logit.to_numpy(), features[ids, 0])):
            if not np.allclose(observed, expected, atol=2e-7, rtol=2e-7):
                raise RuntimeError("Archived target/backbone reconstruction drift")
        checkpoint = unit / "adapter.pt"
        model = restore_model(checkpoint, torch.device("cpu"), torch)
        sampled = np.arange(min(256, len(ids)))
        replay = model_predictions(model, features, ids[sampled], 512, torch.device("cpu"), torch)
        error = float(np.max(np.abs(replay - saved.logit.to_numpy()[sampled])))
        if error > 2e-5:
            raise RuntimeError("Archived Full replay error exceeds2e-5: {}".format(error))
        records.append(dict(fold=fold, seed=seed, rows=len(included), replay_rows=len(sampled),
                            max_logit_error=error, checkpoint_sha256=sha(checkpoint)))
    return records


def data_folds(root, config):
    from osa_controlled_experiments import reconstructed_folds
    args = SimpleNamespace(pair_scores=config["pair_scores"])
    return reconstructed_folds(args, root, resolve(root, config["source_suite"]))


def source_contract(root, config, config_path):
    from osa_context_data import load_discovery_inputs
    pairs_path = resolve(root, config["pair_scores"])
    source = resolve(root, config["source_suite"])
    old = read_json(source / "experiment_contract.json")
    status = read_json(source / "run_summary.json")
    if status["status"] != "PASS_CONTROLLED_EXPERIMENT_EXECUTION":
        raise RuntimeError("The original formal Stage31 suite must be complete")
    pairs = pd.read_csv(pairs_path, sep="\t", low_memory=False)
    proteins = sorted(set(pairs.protein_A.astype(str)) | set(pairs.protein_B.astype(str)))
    if (len(proteins), len(pairs)) != (500, 124750) or pairs.pair_id.duplicated().any():
        raise RuntimeError("Expected complete original500-protein pair universe")
    canonical = [tuple(sorted((str(a), str(b)))) for a,b in zip(pairs.protein_A,pairs.protein_B)]
    if len(set(canonical)) != len(pairs) or any(a==b for a,b in canonical):
        raise RuntimeError("Duplicate or self protein pairs")
    if set(pairs.checkpoint_sha256.astype(str)) != {old["inputs"]["base_checkpoint"]}:
        raise RuntimeError("Stored scores do not identify the original backbone")
    if sha(pairs_path) != old["inputs"]["pair_scores"]:
        raise RuntimeError("Pair input hash differs from completed Stage31")
    _, _, paths = load_discovery_inputs(root, proteins)
    if any(sha(p) != old["inputs"][k] for k, p in paths.items()):
        raise RuntimeError("Discovery data differ from completed Stage31")
    if not set(config["seeds"]).issubset(set(old["seeds"])):
        raise RuntimeError("Replay requires seeds present in original suite")
    files = {"config": config_path, "source_contract": source / "experiment_contract.json",
             "pair_scores": pairs_path, "training_lock": resolve(root, config["source_training_lock"])}
    files["primary_membership"] = root/BASE/"04_candidates_v1/candidate_proteins_primary_250.tsv"
    files["primary_row_order"] = root/SOURCE_ORDER
    files.update(paths)
    files["fold_assignments"] = source / "fold_assignments.tsv"
    for fold in range(1, 5):
        files["fold_%d_pairs" % fold] = source / ("fold_%d_pair_partitions.tsv.gz" % fold)
        for seed in config["seeds"]:
            unit=source/("seed_%d/full/fold_%d" % (seed,fold))
            for name in ("predictions.tsv.gz","adapter.pt"):
                files["archive_%d_%d_%s" % (fold,seed,name)] = unit/name
    for name in ("osa_fusion_capacity_experiments.py", "osa_fusion_capacity_models.py",
                 "triage_osa_model.py", "train_triage_osa.py", "osa_context_data.py",
                 "osa_controlled_experiments.py"):
        files[name] = Path(__file__).with_name(name)
    if sha(files["training_lock"]) != old["inputs"]["training_lock"]:
        raise RuntimeError("Original training lock changed")
    return old, dict((key, sha(path)) for key, path in files.items())


def check_capacity(config):
    from triage_osa_model import INPUT_FEATURE_COLUMNS
    n = len(INPUT_FEATURE_COLUMNS)
    rows = []
    for name, spec in config["models"].items():
        model = make_model(spec, np.zeros(n), np.ones(n))
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append(dict(method=name, trainable_parameters=count,
                         objective=spec["objective"], architecture=spec["architecture"]))
    counts = dict((r["method"], r["trainable_parameters"]) for r in rows)
    if counts != {"Full":2824, "Concat_matched":2823, "Concat_warm":2823,
                  "Concat_wide":5721, "Native_full":2824}:
        raise RuntimeError("Unexpected active model capacity: " + str(counts))
    return rows


def external_source_files(root):
    required = [root/"config/osa_cpap_validation_lock.json",
        root/BASE/"04_candidates_v1/candidate_proteins_primary_250.tsv",
        root/BASE/"20_independent_validation_preprocess_v1/candidate_independent_evidence_500.tsv",
        root/BASE/"13_adipose_validation_preprocess_v1/candidate_adipose_evidence.tsv"]
    for cohort in ("GSE133601", "GSE49800"):
        required += [root/BASE/"10_cpap_validation_preprocess_v1"/cohort/name
            for name in ("candidate_validation_evidence.tsv", "candidate_participant_deltas.tsv.gz")]
    # Python 3.7 can keep __file__ relative when the script is launched by path.
    # Normalize before recording project-relative provenance names.
    script_path = Path(__file__).resolve()
    required += [script_path, script_path.with_name("osa_external_strong_controls.py")]
    required += [root/"scripts/validation"/name for name in ("osa_cpap_reversal_endpoint.py",
        "osa_cpap_subject_signflip.py", "osa_independent_validation.py", "osa_adipose_cross_tissue_endpoint.py")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("Missing existing external inputs/code: " + str(missing))
    return [p.resolve() for p in required]


def final_refit(root, out, config, cv, primary_proteins, device, torch, resume):
    from osa_context_data import load_discovery_inputs, compute_node_evidence, build_pair_context, assemble_input_matrix
    from train_triage_osa import fit_adapter, model_predictions, set_seeds
    from triage_osa_model import TRIAGEOSAModel
    old = read_json(resolve(root, config["source_suite"]) / "experiment_contract.json")
    pairs = pd.read_csv(resolve(root, config["pair_scores"]), sep="\t", low_memory=False)
    proteins = sorted(set(pairs.protein_A) | set(pairs.protein_B))
    expression, meta, _ = load_discovery_inputs(root, proteins)
    node = compute_node_evidence(expression, meta, proteins,
        meta["GSE75097"].sample_id.astype(str).tolist(), meta["GSE71356"].subject_id.astype(str).unique().tolist())
    context = build_pair_context(pairs, node, information_fusion=True)
    features = assemble_input_matrix(pairs, context)
    target = context.context_anchor.to_numpy(dtype=np.float32)
    indices = np.arange(len(pairs))
    ranks = pairs[["pair_id", "protein_A", "protein_B"]].copy()
    ranks["primary_pair"] = primary_mask(pairs, primary_proteins)
    choices = []
    for name, spec in config["models"].items():
        table = cv.loc[cv.method == name]
        means = table.groupby("learning_rate").validation_utility.mean().sort_index()
        rate = float(means.idxmax())
        epochs = max(1, int(np.median(table.loc[table.learning_rate == rate, "best_epoch"])))
        choices.append(dict(method=name, learning_rate=rate, fixed_epochs=epochs,
                            selection="mean discovery validation utility only"))
        probabilities = []
        for seed in config["seeds"]:
            unit = out / "final_refits" / name / ("seed_%d" % seed)
            done = unit / "completed.json"
            if not (resume and done.is_file()):
                unit.mkdir(parents=True, exist_ok=True)
                if spec["objective"] == "archived_native":
                    native = dict(old["optimization"], seed=seed, learning_rate=rate)
                    set_seeds(seed, torch)
                    model, history, epoch, score = fit_adapter(features, target, indices, None,
                        native, device, torch, TRIAGEOSAModel, fixed_epochs=epochs,
                        progress_prefix="[refit {} seed={}]".format(name, seed))
                else:
                    model, history, epoch, score = fit_common(features, target, indices, None, spec,
                        config["common_optimization"], config["common_objective"], seed, rate,
                        device, torch, fixed_epochs=epochs, prefix="[refit {} seed={}]".format(name, seed))
                pd.DataFrame(history).to_csv(unit/"history.tsv", sep="\t", index=False)
                save_model(unit/"adapter.pt", model, dict(method=name, seed=seed, epochs=epochs,
                    learning_rate=rate, final_selection_uses_external=False), torch)
                z = model_predictions(model, features, indices, 4096, device, torch)
                np.save(str(unit/"probabilities.npy"), expit(z))
                write_json(done, dict(probabilities_sha256=sha(unit/"probabilities.npy"),
                    checkpoint_sha256=sha(unit/"adapter.pt"), learning_rate=rate, epochs=epochs))
                del model
            manifest = read_json(done)
            if (manifest["learning_rate"] != rate or manifest["epochs"] != epochs or
                    sha(unit/"probabilities.npy") != manifest["probabilities_sha256"] or
                    sha(unit/"adapter.pt") != manifest["checkpoint_sha256"]):
                raise RuntimeError("Final-refit resume mismatch")
            probabilities.append(np.load(str(unit/"probabilities.npy"), allow_pickle=False))
        ranks[name] = np.mean(probabilities, axis=0)
    ranks["Direct_anchor"] = target
    ranks.to_csv(out/"scores_all_500.tsv.gz", sep="\t", index=False)
    primary = ranks.loc[ranks.primary_pair].copy()
    order = pd.read_csv(root/SOURCE_ORDER, sep="\t")
    if set(primary.pair_id) != set(order.pair_id) or len(primary) != 31125:
        raise RuntimeError("Primary universe changed")
    primary = primary.set_index("pair_id").loc[order.pair_id].reset_index()
    if not np.array_equal(primary[["protein_A", "protein_B"]], order[["protein_A", "protein_B"]]):
        raise RuntimeError("Primary protein endpoints changed")
    names = list(config["models"]) + ["Direct_anchor"]
    for name in names:
        primary["rank_"+name] = primary[name].rank(method="first", ascending=False).astype(int)
    primary.to_csv(out/"rankings_primary_250.tsv.gz", sep="\t", index=False)
    pd.DataFrame(choices).to_csv(out/"final_refit_choices.tsv", sep="\t", index=False)
    overlap = []
    for budget in (312, 1557):
        reference = set(primary.loc[primary.rank_Full <= budget, "pair_id"])
        for name in names:
            selected = set(primary.loc[primary["rank_"+name] <= budget, "pair_id"])
            overlap.append(dict(method=name, budget=budget, intersection=len(reference & selected),
                Jaccard=len(reference & selected)/float(len(reference | selected))))
    pd.DataFrame(overlap).to_csv(out/"top_set_overlaps.tsv", sep="\t", index=False)
    primary.loc[primary[["rank_"+n for n in names]].min(axis=1) <= 1557].to_csv(
        out/"top_pair_method_comparisons.tsv.gz", sep="\t", index=False)
    return sha(out/"rankings_primary_250.tsv.gz")


def run_fit(args, root, config, config_path, out, torch):
    from osa_controlled_experiments import metrics
    from train_triage_osa import fit_adapter, model_predictions, set_seeds
    from triage_osa_model import TRIAGEOSAModel
    old, fingerprints = source_contract(root, config, config_path)
    # Check availability only. No external evidence values are loaded for fitting.
    external_source_files(root)
    capacities = check_capacity(config)
    primary_table = pd.read_csv(root/BASE/"04_candidates_v1/candidate_proteins_primary_250.tsv", sep="\t")
    primary_proteins = set(primary_table.UniProt_accession.astype(str))
    if len(primary_proteins) != 250:
        raise RuntimeError("Primary protein membership invalid")
    contract = dict(version=VERSION, smoke=bool(args.smoke), configuration=config,
        inputs=fingerprints, active_parameters=capacities, external_outcomes_read_during_fit=False,
        software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                      torch=torch.__version__, platform=platform.platform()),
        analysis_type="post_hoc_common_objective_capacity_control")
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if not args.check_env and out.exists() and any(out.iterdir()):
        if not args.resume or not (out/"analysis_contract.json").is_file():
            raise RuntimeError("Output exists. Use --resume with identical code/inputs; never overwrite old experiments")
        if read_json(out/"analysis_contract.json") != contract:
            raise RuntimeError("Resume contract differs; choose a new50_* directory")
        if (out/"run_summary.json").is_file():
            completed = read_json(out/"run_summary.json")
            if completed["status"] in ("PASS_SMOKE_FIT", "PASS_FUSION_CAPACITY_FIT"):
                verify_artifacts(out, completed.get("artifacts"))
                print("Completed fit and all recorded artifacts verified; retained unchanged", flush=True)
                return 0
    if not args.check_env:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/"analysis_contract.json", contract)
        write_json(out/"run_summary.json", dict(status="RUNNING", external_outcomes_read_during_fit=False))
        pd.DataFrame(capacities).to_csv(out/"active_parameter_counts.tsv", sep="\t", index=False)
    all_trials, selected_metrics, replay_rows = [], [], []
    device = torch.device(args.device)
    for fold, pairs, nodes, contexts, features, splits, meta, members in data_folds(root, config):
        if args.smoke and fold != 1:
            break
        target = contexts["development"].context_anchor.to_numpy(dtype=np.float32)
        heldout = contexts["heldout"].context_anchor.to_numpy(dtype=np.float32)
        replay_rows.extend(verify_fold(resolve(root,config["source_suite"]), fold, config["seeds"],
                           pairs, features, target, heldout, splits, torch))
        print("[fold {}] original identities, targets and checkpoint replay: PASS".format(fold), flush=True)
        if args.check_env:
            continue
        is_primary = primary_mask(pairs, primary_proteins)
        for seed in config["seeds"]:
            actual_seed = int(seed) + fold - 1
            for name, spec in config["models"].items():
                unit = out/"cv"/("fold_%d" % fold)/("seed_%d" % seed)/name
                done = unit/"completed.json"
                if args.resume and done.is_file():
                    mark = read_json(done)
                    for f, digest in mark["artifacts"].items():
                        if sha(unit/f) != digest:
                            raise RuntimeError("Completed unit modified: " + str(unit/f))
                    selected_metrics.extend(pd.read_csv(unit/"metrics.tsv",sep="\t").to_dict("records"))
                    all_trials.extend(pd.read_csv(unit/"tuning_summary.tsv",sep="\t").to_dict("records"))
                    continue
                unit.mkdir(parents=True, exist_ok=True)
                rates = ([float(old["optimization"]["learning_rate"])] if spec["objective"] == "archived_native"
                         else config["common_optimization"]["learning_rates"])
                tuning = []
                for rate in rates:
                    trial = unit/("lr_%.8g" % rate)
                    trial.mkdir(exist_ok=True)
                    marker = trial/"completed.json"
                    if args.resume and marker.is_file():
                        trial_record = read_json(marker)
                        if trial_record["contract_hash"] != contract_hash or sha(trial/"adapter.pt") != trial_record["checkpoint_sha256"]:
                            raise RuntimeError("Trial resume mismatch")
                        tuning.append(trial_record["result"])
                        continue
                    prefix = "[{} fold={} seed={} lr={}]".format(name,fold,seed,rate)
                    started = time.time()
                    if spec["objective"] == "archived_native":
                        native = dict(old["optimization"], seed=actual_seed)
                        if args.smoke:
                            native.update(max_epochs=2,patience=2)
                        set_seeds(actual_seed,torch)
                        model, history, epoch, score = fit_adapter(features,target,splits["train"],splits["validation"],
                            native,device,torch,TRIAGEOSAModel,progress_prefix=prefix)
                    else:
                        model, history, epoch, score = fit_common(features,target,splits["train"],splits["validation"],
                            spec,config["common_optimization"],config["common_objective"],actual_seed,rate,device,torch,prefix=prefix)
                    result = dict(method=name,fold=fold,seed=seed,learning_rate=rate,best_epoch=epoch,
                        validation_utility=score,epochs_executed=len(history),wall_seconds=time.time()-started,
                        reached_epoch_cap=len(history)==int(native["max_epochs"] if spec["objective"]=="archived_native" else config["common_optimization"]["max_epochs"]))
                    pd.DataFrame(history).to_csv(trial/"history.tsv",sep="\t",index=False)
                    save_model(trial/"adapter.pt",model,result,torch)
                    write_json(marker,dict(result=result,contract_hash=contract_hash,checkpoint_sha256=sha(trial/"adapter.pt")))
                    tuning.append(result)
                    del model
                tuning = pd.DataFrame(tuning).sort_values("learning_rate",kind="mergesort")
                chosen = tuning.loc[tuning.validation_utility.idxmax()].to_dict()
                model = restore_model(unit/("lr_%.8g" % chosen["learning_rate"])/"adapter.pt",device,torch)
                rows, predictions = prediction_tables(model,features,target,heldout,pairs,splits,is_primary,
                    dict(method=name,fold=fold,seed=seed,best_epoch=int(chosen["best_epoch"]),
                         learning_rate=float(chosen["learning_rate"])),2048,device,torch)
                rows.to_csv(unit/"metrics.tsv",sep="\t",index=False)
                predictions.to_csv(unit/"predictions.tsv.gz",sep="\t",index=False)
                tuning.to_csv(unit/"tuning_summary.tsv",sep="\t",index=False)
                save_model(unit/"selected_adapter.pt",model,chosen,torch)
                artifacts = {f:sha(unit/f) for f in ("metrics.tsv","predictions.tsv.gz","tuning_summary.tsv","selected_adapter.pt")}
                write_json(done,dict(contract_hash=contract_hash,artifacts=artifacts))
                selected_metrics.extend(rows.to_dict("records"))
                all_trials.extend(tuning.to_dict("records"))
                del model
                pd.DataFrame(selected_metrics).to_csv(out/"cv_metrics.tsv",sep="\t",index=False)
    if args.check_env:
        print("Preflight PASS: data reconstruction, archived prediction replay and active parameter counts")
        print(pd.DataFrame(capacities).to_string(index=False))
        return 0
    pd.DataFrame(replay_rows).to_csv(out/"archived_replay_checks.tsv",sep="\t",index=False)
    trials = pd.DataFrame(all_trials)
    trials.to_csv(out/"all_tuning_trials.tsv",sep="\t",index=False)
    summary = pd.DataFrame(selected_metrics)
    summary.to_csv(out/"cv_metrics.tsv",sep="\t",index=False)
    test = summary.loc[summary.partition == "test"].copy()
    report = ["soft_BCE","Spearman","weak_top20_AUPRC","structural_retention_Spearman",
              "top5_enrichment_ratio","multi_axis_utility","heldout_participant_target_rho"]
    test.groupby(["method","scope"])[report].agg(["mean","std","min","max"]).to_csv(out/"test_summary.tsv",sep="\t")
    reference = test.loc[test.method=="Full",["fold","seed","scope"]+report]
    paired = test.merge(reference,on=["fold","seed","scope"],suffixes=("","_Full"),validate="many_to_one")
    for metric in report:
        paired["difference_"+metric] = paired[metric] - paired[metric+"_Full"]
    paired.to_csv(out/"paired_differences_vs_common_Full.tsv",sep="\t",index=False)
    # Four folds share discovery data; no iid12-run significance test is added.
    paired.groupby(["method","scope","fold"])[["difference_"+m for m in report]].mean().to_csv(out/"fold_mean_differences.tsv",sep="\t")
    ranking_hash = final_refit(root,out,config,trials,primary_proteins,device,torch,args.resume)
    write_json(out/"run_summary.json",dict(status="PASS_SMOKE_FIT" if args.smoke else "PASS_FUSION_CAPACITY_FIT",
        version=VERSION,ranking_sha256=ranking_hash,external_outcomes_read_during_fit=False,
        methods=list(config["models"])+["Direct_anchor"],
        primary_reference="Full under common BCE-plus-ranking objective",
        native_reference="Native_full uses archived objective and original budget; not objective-matched",
        original_production_checkpoint_changed=False,clinical_validation=False,
        artifacts=output_artifacts(out, ("external_evaluation",)),
        note="Execution completion is not a superiority or performance gate"))
    print("FIT COMPLETE: " + str(out),flush=True)
    return 0


def run_external(args,root,config,out):
    from osa_external_strong_controls import cpap, tissue_controls
    source = read_json(out/"run_summary.json")
    expected = "PASS_SMOKE_FIT" if args.smoke else "PASS_FUSION_CAPACITY_FIT"
    if source["status"] != expected or source["external_outcomes_read_during_fit"]:
        raise RuntimeError("A matching completed fit is required before external evaluation")
    ranking = out/"rankings_primary_250.tsv.gz"
    if sha(ranking) != source["ranking_sha256"]:
        raise RuntimeError("Ranking modified after fitting")
    destination = out/"external_evaluation"
    destination.mkdir(exist_ok=True)
    methods = tuple(source["methods"])
    required = external_source_files(root)
    fingerprints = dict((p.relative_to(root).as_posix(), sha(p)) for p in required)
    import scipy
    extconfig = dict(version=VERSION,ranking_sha256=sha(ranking),methods=methods,
        inputs_and_code=fingerprints,
        software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        permutations=config["external"]["permutations"],bootstrap=config["external"]["participant_bootstrap"],
        post_hoc=True,model_selection_using_external=False,
        primary_comparison="Full(common objective) versus Concat_matched",
        correction="BH across every Full-versus-control CPAP endpoint comparison; tissue family separate",
        full_meaning=source["primary_reference"],native_full_meaning=source["native_reference"])
    if (destination/"analysis_contract.json").exists():
        old=read_json(destination/"analysis_contract.json")
        # JSON converts tuples to lists.
        if json.loads(json.dumps(extconfig)) != old:
            raise RuntimeError("External resume contract differs")
        if (destination/"run_summary.json").is_file() and args.resume:
            completed=read_json(destination/"run_summary.json")
            if completed["status"].startswith("PASS_"):
                verify_artifacts(destination, completed.get("artifacts"))
                print("External evaluation already complete; retained unchanged")
                return 0
        if not args.resume:
            raise RuntimeError("External output exists; use --resume")
    write_json(destination/"analysis_contract.json",extconfig)
    write_json(destination/"run_summary.json",dict(status="RUNNING"))
    frame=pd.read_csv(ranking,sep="\t")
    print("[external] fitted models fixed; now reading previously inspected external evidence",flush=True)
    cpap(root,destination,frame,config["external"]["permutations"],config["external"]["participant_bootstrap"],methods=methods)
    tissue_controls(root,destination,frame,config["external"]["permutations"],methods=methods)
    write_json(destination/"run_summary.json",dict(status="PASS_SMOKE_EXTERNAL" if args.smoke else "PASS_FUSION_CAPACITY_EXTERNAL",
        ranking_sha256=sha(ranking),primary_reference=source["primary_reference"],
        artifacts=output_artifacts(destination),
        methods=methods,clinical_validation=False,superiority_assumed=False))
    print("EXTERNAL COMPLETE: " + str(destination),flush=True)
    return 0


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--config",default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--check-env",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--external-only",action="store_true")
    return parser.parse_args()


def main():
    args=parse_args()
    root=Path(args.root).resolve()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/validation"))
    config_path=resolve(root,args.config)
    config=read_json(config_path)
    if args.smoke:
        config["seeds"]=config["seeds"][:1]
        config["common_optimization"].update(max_epochs=2,patience=2,learning_rates=[0.001])
        config["external"].update(permutations=100,participant_bootstrap=100)
    location=args.output_dir or (config["output_dir"] + ("_smoke" if args.smoke else ""))
    out=safe_output(root,location)
    print("Python: " + sys.executable,flush=True)
    if args.external_only:
        if args.check_env:
            raise RuntimeError("--external-only and --check-env are separate phases")
        return run_external(args,root,config,out)
    import torch
    if args.threads<1:
        raise RuntimeError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use the verified environment")
    return run_fit(args,root,config,config_path,out,torch)


if __name__ == "__main__":
    raise SystemExit(main())
