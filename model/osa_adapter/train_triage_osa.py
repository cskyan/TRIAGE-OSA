#!/usr/bin/env python3
"""Continue TRIAGE-Fuse training with discovery-only OSA specialization.

Formal training requires frozen TRIAGE inference for all 124,750 pairs in the
500-protein sensitivity universe.  GSE75097 and GSE71356 are the only disease
data read.  CPAP, adipose, single-cell, proteome and known-PPI resources are
neither opened nor used for selection.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "4.0.0"
EXPECTED_BASE_SHA256 = "87a161435023070443c675ebab3539268f9537a8a925f782d081061e1a853a56"
DEFAULT_SEED = 20260716


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return dict((str(key), json_safe(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(json_safe(payload), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(values.shape[0], dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return float("nan")
    x = rankdata(first[valid])
    y = rankdata(second[valid])
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / (np.arange(len(ordered), dtype=float) + 1.0)
    return float(np.sum(precision * ordered) / positives)


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    ranks = rankdata(scores) + 1.0
    statistic = ranks[positive].sum() - positive.sum() * (positive.sum() + 1.0) / 2.0
    return float(statistic / (positive.sum() * negative.sum()))


def prediction_metrics(target, logit, base_logit=None):
    target = np.asarray(target, dtype=float)
    logit = np.asarray(logit, dtype=float)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    eps = 1e-7
    bce = -np.mean(
        target * np.log(np.clip(probability, eps, 1.0 - eps))
        + (1.0 - target) * np.log(np.clip(1.0 - probability, eps, 1.0 - eps))
    )
    threshold = float(np.quantile(target, 0.80))
    label = (target >= threshold).astype(int)
    top_n = max(1, int(math.ceil(0.05 * len(target))))
    top_index = np.argsort(-probability, kind="mergesort")[:top_n]
    structural_retention = (
        spearman_correlation(logit, base_logit)
        if base_logit is not None else float("nan")
    )
    multi_axis_utility = (
        0.75 * spearman_correlation(target, probability)
        + 0.25 * structural_retention
        - 0.15 * float(bce)
        if math.isfinite(structural_retention) else float("nan")
    )
    return {
        "soft_BCE": float(bce),
        "Spearman": spearman_correlation(target, probability),
        "weak_top20_AUPRC": average_precision(label, probability),
        "weak_top20_AUROC": roc_auc(label, probability),
        "top5_target_mean": float(np.mean(target[top_index])),
        "overall_target_mean": float(np.mean(target)),
        "top5_enrichment_ratio": float(
            np.mean(target[top_index]) / max(np.mean(target), 1e-12)
        ),
        "structural_retention_Spearman": structural_retention,
        "multi_axis_utility": multi_axis_utility,
        "pairs": int(len(target)),
    }


def set_seeds(seed, torch):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_batches(indices, batch_size, rng):
    order = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        yield order[start:start + int(batch_size)]


def model_predictions(model, features, indices, batch_size, device, torch):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_index = indices[start:start + int(batch_size)]
            x = torch.from_numpy(features[batch_index]).to(device=device)
            outputs.append(model(x)["adapted_logit"].detach().cpu().numpy())
    if not outputs:
        return np.asarray([], dtype=float)
    return np.concatenate(outputs)


def model_expert_weights(model, features, indices, batch_size, device, torch):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch_index = indices[start:start + int(batch_size)]
            x = torch.from_numpy(features[batch_index]).to(device=device)
            outputs.append(model(x)["expert_weights"].detach().cpu().numpy())
    if not outputs:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def fit_adapter(
    features,
    targets,
    train_indices,
    validation_indices,
    config,
    device,
    torch,
    model_class,
    fixed_epochs=None,
    progress_prefix="",
    validation_targets=None,
):
    train_values = features[train_indices]
    input_mean = np.nanmean(train_values, axis=0).astype(np.float32)
    input_std = np.nanstd(train_values, axis=0).astype(np.float32)
    input_std = np.where(input_std > 1e-6, input_std, 1.0).astype(np.float32)
    model_options = {}
    if "experiment_variant" in config:
        model_options["experiment_variant"] = config["experiment_variant"]
    model = model_class(
        input_mean=input_mean,
        input_std=input_std,
        hidden_dim=int(config["hidden_dimension"]),
        dropout=float(config["dropout"]),
        minimum_structural_mass=float(config["minimum_mean_structural_mass"]),
        **model_options
    ).to(device)
    if config.get("static_initial_weights") is not None:
        from osa_cross_participant_adaptation import initialize_from_static
        initialize_from_static(model, config["static_initial_weights"], torch)
    if config.get("experiment_variant") == "equal_fusion":
        # A fixed comparator is evaluated, never given artificial train epochs.
        return model, [], 0, float("nan")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = int(config["batch_size"])
    max_epochs = int(fixed_epochs or config["max_epochs"])
    patience = int(config["patience"])
    rng = np.random.RandomState(int(config["seed"]))
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    history = []
    stale = 0
    if config.get("static_initial_weights") is not None:
        if validation_indices is None or not len(validation_indices):
            raise RuntimeError("Static initialization requires a discovery validation partition")
        z0 = model_predictions(model, features, validation_indices, batch_size, device, torch)
        target0 = targets if validation_targets is None else validation_targets
        m0 = prediction_metrics(target0[validation_indices], z0, features[validation_indices, 0])
        best_score = (-float(m0["soft_BCE"]) if config.get("selection_metric") == "soft_BCE"
                      else float(m0["multi_axis_utility"]))
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print("{} epoch=0 static initialization validation score={:.6f}".format(progress_prefix,best_score),flush=True)
    structural_dual = float(config.get("structural_dual_initial", 0.50))

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        structural_masses = []
        retention_activations = []
        epoch_structural_dual = float(structural_dual)
        for batch_index in tensor_batches(train_indices, batch_size, rng):
            x = torch.from_numpy(features[batch_index]).to(device=device)
            y = torch.from_numpy(targets[batch_index].astype(np.float32)).to(device=device)
            output = model(x)
            logits = output["adapted_logit"]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            permutation = torch.randperm(logits.shape[0], device=device)
            difference = y - y[permutation]
            mask = torch.abs(difference) >= float(config["minimum_ranking_target_gap"])
            if bool(mask.any().item()):
                ranking = torch.nn.functional.softplus(
                    -torch.sign(difference[mask])
                    * (logits[mask] - logits[permutation][mask])
                ).mean()
            else:
                ranking = logits.sum() * 0.0
            # V3.2: structural distillation is a safety constraint, not a
            # competing objective.  It is active only for pairs whose learned
            # structural mass falls below the locked minimum.  This removes
            # the always-on pull toward frozen PPI that over-constrained V3.1.
            expert_weights = output["expert_weights"]
            structural_mass = expert_weights[:, 0] + expert_weights[:, 1]
            structural_deficit = torch.relu(
                float(config["minimum_mean_structural_mass"])
                - structural_mass
            )
            retention_weight = (
                (1.0 - y).detach() * structural_deficit.detach()
            )
            retention = torch.mean(
                retention_weight * torch.square(logits - output["base_logit"])
            )
            gate_drift = torch.mean(
                torch.square(output["context_gates"] - output["base_gates"])
            )
            structural_mass_penalty = torch.square(
                torch.relu(
                    float(config["minimum_mean_structural_mass"])
                    - torch.mean(structural_mass)
                )
            )
            normalized_entropy = -torch.sum(
                expert_weights * torch.log(expert_weights.clamp_min(1e-8)), dim=1
            ) / math.log(3.0)
            entropy_penalty = torch.mean(
                torch.square(
                    torch.relu(
                        float(config["minimum_expert_entropy_normalized"])
                        - normalized_entropy
                    )
                )
            )
            loss = (
                float(config["bce_weight"]) * bce
                + float(config["ranking_weight"]) * ranking
                + float(config["constraint_triggered_PPI_retention_weight"]) * retention
                + float(config["gate_drift_weight"]) * gate_drift
                + epoch_structural_dual * structural_mass_penalty
                + float(config["expert_entropy_weight"]) * entropy_penalty
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            structural_masses.append(
                float(structural_mass.detach().mean().cpu().item())
            )
            retention_activations.append(
                float((structural_deficit > 0.0).float().mean().detach().cpu().item())
            )

        mean_train_structural_mass = float(np.mean(structural_masses))
        structural_dual = min(
            float(config.get("structural_dual_max", 5.0)),
            max(
                0.0,
                structural_dual
                + float(config.get("structural_dual_learning_rate", 2.0))
                * (
                    float(config["minimum_mean_structural_mass"])
                    - mean_train_structural_mass
                ),
            ),
        )

        if validation_indices is not None and len(validation_indices) > 0:
            val_logits = model_predictions(
                model, features, validation_indices, batch_size, device, torch
            )
            target_source = targets if validation_targets is None else validation_targets
            metrics = prediction_metrics(
                target_source[validation_indices],
                val_logits,
                features[validation_indices, 0],
            )
            score = (-float(metrics["soft_BCE"]) if config.get("selection_metric") == "soft_BCE"
                     else float(metrics["multi_axis_utility"]))
        else:
            metrics = {"soft_BCE": float("nan"), "Spearman": float("nan")}
            score = float(epoch)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_soft_BCE": metrics["soft_BCE"],
                "validation_Spearman": metrics["Spearman"],
                "selection_score": score,
                "train_structural_mass_penalty": float(structural_mass_penalty.detach().cpu().item()),
                "train_expert_entropy_penalty": float(entropy_penalty.detach().cpu().item()),
                "train_mean_structural_mass": mean_train_structural_mass,
                "train_structural_constraint_activation": float(np.mean(retention_activations)),
                "structural_dual_used": epoch_structural_dual,
                "structural_dual_next": float(structural_dual),
            }
        )
        print(
            "{} epoch={:02d} train_loss={:.5f} val_BCE={} val_rho={} score={}".format(
                progress_prefix,
                epoch,
                float(np.mean(losses)),
                "NA" if not math.isfinite(float(metrics["soft_BCE"])) else "{:.5f}".format(metrics["soft_BCE"]),
                "NA" if not math.isfinite(float(metrics["Spearman"])) else "{:.4f}".format(metrics["Spearman"]),
                "{:.5f}".format(score),
            )
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = dict(
                (key, value.detach().cpu().clone())
                for key, value in model.state_dict().items()
            )
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and validation_indices is not None and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No OSA adapter checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    return model, history, int(best_epoch), float(best_score)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--pair-scores",
        default="results/discovery_preprocess_v2/08_pair_inference_sensitivity_v1/pair_scores.tsv.gz",
    )
    parser.add_argument(
        "--output-stage",
        default="results/discovery_preprocess_v2/06_model_training/triage_osa_patient_information_v4",
    )
    parser.add_argument(
        "--training-lock",
        default="config/triage_osa_training_lock_v4.json",
        help="Frozen discovery-only training contract.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--cross-participant-adaptation", action="store_true",
                            help="Discovery-only disjoint-participant supervision pilot; preserves production model.")
    parser.add_argument("--comparison-ranking", default=None,
                        help="Stage39 ensemble ranking directory for matched Stage36/38 exploratory re-evaluation.")
    parser.add_argument("--comparison-method", choices=("Full", "Static_constrained", "Direct_anchor"), default="Full",
                        help="Stage36 representation selected from the same Stage39 run.")
    experiment.add_argument("--external-strong-controls", action="store_true",
                            help="Discovery-only converged static refit and matched external OSA controls; CPU only.")
    parser.add_argument("--external-control-permutations", type=int, default=10000)
    parser.add_argument("--external-control-bootstrap", type=int, default=5000)
    experiment.add_argument("--experiment-suite", action="store_true",
                            help="Isolated OSA retraining ablations and fusion controls; never overwrites V4.")
    experiment.add_argument("--backbone-retention", action="store_true",
                            help="Check actual frozen backbone identity and optional locked benchmark data.")
    experiment.add_argument("--experiment-self-test", action="store_true",
                            help="Small synthetic tests only; no discovery or validation data.")
    experiment.add_argument("--audit-experiments", action="store_true",
                            help="Audit completed ablation outputs without retraining.")
    experiment.add_argument("--complete-supplements", action="store_true",
                            help="Audit, converged fusion controls, context diagnostics, L2 perturbations and benchmark inventory.")
    experiment.add_argument("--prepare-benchmarks", action="store_true",
                            help="CPU-only native benchmark inventory, historical split/label audit and missing-input report; no training.")
    experiment.add_argument("--published-osa-comparison", action="store_true",
                            help="One CPU-only, subject-aware comparison of frozen published OSA gene panels and frozen TRIAGE-OSA candidate panels.")
    parser.add_argument("--triage-root", default='resources/backbone')
    parser.add_argument("--l2-pairs", type=int, default=128,
                        help="Outcome-independent hash sample for OSA L2 sensitivity; not an accuracy benchmark.")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="20260716,20260717,20260718",
                        help="Paired training seeds; data partitions remain fixed across seeds.")
    parser.add_argument("--experiment-dir", default="results/discovery_preprocess_v2/31_osa_controlled_experiments_v1")
    parser.add_argument("--benchmark-lock", default=None,
                        help="Explicit held-out dataset manifest for residue/pair retention tests.")
    parser.add_argument("--resume-experiments", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cross_participant_adaptation:
        from osa_cross_participant_adaptation import run
        return run(args)
    if args.external_strong_controls:
        from osa_external_strong_controls import run
        return run(args)
    if args.published_osa_comparison:
        from osa_published_comparison import run
        return run(args)
    if (args.experiment_suite or args.backbone_retention or args.experiment_self_test
            or args.audit_experiments or args.complete_supplements or args.prepare_benchmarks):
        from osa_controlled_experiments import run
        return run(args)
    root = Path(args.root).expanduser().resolve()
    module_dir = Path(__file__).resolve().parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    try:
        import torch
        from triage_osa_model import TRIAGEOSAModel, INPUT_FEATURE_COLUMNS
        from osa_triage_adapter import load_frozen_checkpoint
        from osa_context_data import (
            assemble_input_matrix,
            build_pair_context,
            compute_node_evidence,
            load_discovery_inputs,
            role_for_fold,
            stable_bucket,
            stratified_sample_folds,
            subject_folds,
        )
    except ImportError as exc:
        raise RuntimeError("numpy, pandas and torch are required: {}".format(exc))

    lock_path = Path(args.training_lock)
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    base_checkpoint = (
        root / "results" / "discovery_preprocess_v2" / "06_model_training"
        / "esm_only_adaptation_v1" / "triage_tuna_esm_only_adapted.pt"
    )
    pair_scores_path = Path(args.pair_scores)
    if not pair_scores_path.is_absolute():
        pair_scores_path = root / pair_scores_path
    output_dir = Path(args.output_stage)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if args.smoke:
        output_dir = output_dir.parent / (output_dir.name + "_smoke")

    print("TRIAGE-OSA specialization environment")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("torch: {}".format(torch.__version__))
    print("CUDA: {}".format(torch.cuda.is_available()))
    print("Base checkpoint: {}".format(base_checkpoint))
    print("Pair scores: {}".format(pair_scores_path))
    print("External validation read: no")
    required = [lock_path, base_checkpoint]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing locked model inputs: {}".format(missing))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    information_fusion = bool(
        lock.get("trainable_innovation", {}).get(
            "patient_information_conditioning", False
        )
    )
    observed_base_hash = sha256_file(base_checkpoint)
    if observed_base_hash != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            "Base checkpoint hash mismatch: expected={} observed={}".format(
                EXPECTED_BASE_SHA256, observed_base_hash
            )
        )
    if args.check_env:
        synthetic = torch.zeros(
            (8, len(INPUT_FEATURE_COLUMNS)), dtype=torch.float32
        )
        synthetic[:, 0] = torch.linspace(-1.0, 1.0, 8)
        synthetic[:, 1:4] = torch.randn(8, 3)
        synthetic[:, 4:7] = 1.0 / 3.0
        smoke_model = TRIAGEOSAModel(
            input_mean=np.zeros(len(INPUT_FEATURE_COLUMNS), dtype=np.float32),
            input_std=np.ones(len(INPUT_FEATURE_COLUMNS), dtype=np.float32),
            hidden_dim=int(lock["optimization"]["hidden_dimension"]),
            dropout=float(lock["optimization"]["dropout"]),
            minimum_structural_mass=float(
                lock["optimization"]["minimum_mean_structural_mass"]
            ),
        )
        smoke_output = smoke_model(synthetic)
        smoke_loss = smoke_output["adapted_logit"].mean()
        smoke_loss.backward()
        if not bool(torch.isfinite(smoke_output["adapted_logit"]).all().item()):
            raise RuntimeError("Synthetic OSA adapter forward pass is non-finite")
        gate_error = torch.max(
            torch.abs(smoke_output["context_gates"].sum(dim=1) - 1.0)
        ).item()
        if gate_error > 1e-6:
            raise RuntimeError("Synthetic context gates do not sum to one")
        expert_error = torch.max(
            torch.abs(smoke_output["expert_weights"].sum(dim=1) - 1.0)
        ).item()
        if expert_error > 1e-6:
            raise RuntimeError("Synthetic expert weights do not sum to one")
        if bool((smoke_output["expert_weights"] < 0.0).any().item()):
            raise RuntimeError("Synthetic expert fusion contains a negative weight")
        minimum_observed_structural_mass = float(
            smoke_output["expert_weights"][:, :2].sum(dim=1).min().item()
        )
        if minimum_observed_structural_mass + 1e-7 < float(
            lock["optimization"]["minimum_mean_structural_mass"]
        ):
            raise RuntimeError(
                "Synthetic expert fusion violates the structural-mass constraint"
            )
        if not any(
            parameter.grad is not None for parameter in smoke_model.parameters()
        ):
            raise RuntimeError("Synthetic OSA adapter backward pass produced no gradients")
        backbone_model, _, backbone_metadata = load_frozen_checkpoint(
            base_checkpoint, torch.device("cpu")
        )
        del backbone_model
        print("Synthetic TRIAGE-OSA specialization forward/backward: PASS")
        print(
            "Original TRIAGE backbone strict-load: PASS ({} keys)".format(
                backbone_metadata["loaded_key_count"]
            )
        )
        print("TRIAGE-OSA specialization preflight: PASS")
        print("Formal pair scores present: {}".format(pair_scores_path.is_file()))
        return 0
    if not pair_scores_path.is_file() and args.smoke:
        fallback = (
            root / "results" / "discovery_preprocess_v2"
            / "08_pair_inference_primary_v1" / "pair_scores.tsv.gz"
        )
        if fallback.is_file():
            pair_scores_path = fallback
            print("Smoke fallback pair scores: {}".format(pair_scores_path))
    if not pair_scores_path.is_file():
        raise RuntimeError(
            "Missing frozen pair inference. Run sensitivity_500 inference first: {}".format(
                pair_scores_path
            )
        )

    optimization = dict(lock["optimization"])
    optimization["seed"] = int(optimization.get("seed", DEFAULT_SEED))
    if args.smoke:
        optimization["max_epochs"] = 5
        optimization["patience"] = 5
        optimization["batch_size"] = min(512, int(optimization["batch_size"]))
    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    set_seeds(optimization["seed"], torch)

    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force: {}".format(output_dir))
        resolved = output_dir.resolve()
        if root not in resolved.parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(pair_scores_path, sep="\t")
    pairs["protein_A"] = pairs["protein_A"].astype(str)
    pairs["protein_B"] = pairs["protein_B"].astype(str)
    if "checkpoint_sha256" not in pairs:
        raise RuntimeError("Pair inference lacks base-checkpoint provenance")
    pair_checkpoint_hashes = sorted(
        set(pairs["checkpoint_sha256"].dropna().astype(str).tolist())
    )
    if pair_checkpoint_hashes != [EXPECTED_BASE_SHA256]:
        raise RuntimeError(
            "Pair outputs were not generated by the locked base checkpoint: {}".format(
                pair_checkpoint_hashes
            )
        )
    proteins = sorted(set(pairs["protein_A"]) | set(pairs["protein_B"]))
    expected_pairs = len(proteins) * (len(proteins) - 1) // 2
    if len(pairs) != expected_pairs or pairs["pair_id"].duplicated().any():
        raise RuntimeError("Pair inference is not a complete unique unordered universe")
    if not args.smoke and (len(proteins) != 500 or len(pairs) != 124750):
        raise RuntimeError(
            "Formal training requires 500 proteins/124750 pairs; observed {}/{}".format(
                len(proteins), len(pairs)
            )
        )
    if not args.smoke:
        primary_count = int(
            pairs["primary_pair"].astype(str).str.lower().eq("yes").sum()
        )
        if primary_count != 31125:
            raise RuntimeError(
                "Sensitivity inference does not contain the exact primary prefix: {}".format(
                    primary_count
                )
            )
    expression, metadata, input_paths = load_discovery_inputs(root, proteins)
    information_contract = lock.get("patient_information_manifest", {})
    information_path = input_paths["discovery_patient_information"]
    observed_information_hash = sha256_file(information_path)
    if observed_information_hash != str(information_contract.get("sha256", "")):
        raise RuntimeError(
            "Discovery patient-information manifest hash mismatch: expected={} observed={}".format(
                information_contract.get("sha256", ""), observed_information_hash
            )
        )
    sample_fold_table = stratified_sample_folds(
        metadata["GSE75097"], folds=4, seed=optimization["seed"]
    )
    subject_fold_table = subject_folds(
        metadata["GSE71356"], seed=optimization["seed"]
    )
    protein_fold_table = pd.DataFrame(
        {
            "dataset": "protein_universe",
            "unit_id": proteins,
            "stratum": "UniProt_accession",
            "fold": [stable_bucket(value, 4, optimization["seed"]) for value in proteins],
        }
    )
    fold_assignments = pd.concat(
        [sample_fold_table, subject_fold_table, protein_fold_table], ignore_index=True
    )
    fold_assignments.to_csv(output_dir / "fold_assignments.tsv", sep="\t", index=False)

    partition_rows = []
    for outer_fold in range(4):
        for _, row in fold_assignments.iterrows():
            partition_rows.append(
                {
                    "outer_fold": int(outer_fold + 1),
                    "dataset": str(row["dataset"]),
                    "unit_id": str(row["unit_id"]),
                    "stratum": str(row["stratum"]),
                    "assignment_fold": int(row["fold"]),
                    "partition": role_for_fold(int(row["fold"]), outer_fold, 4),
                }
            )
    pd.DataFrame(partition_rows).to_csv(
        output_dir / "cross_validation_partition_assignments.tsv",
        sep="\t",
        index=False,
    )
    sample_fold_lookup = sample_fold_table.set_index("unit_id")["fold"].to_dict()
    subject_fold_lookup = subject_fold_table.set_index("unit_id")["fold"].to_dict()
    protein_fold_lookup = protein_fold_table.set_index("unit_id")["fold"].to_dict()

    # The discovery-wide target is used only as a participant-stability reference
    # and for the final fit.  It is never used to select a CV checkpoint.
    all_samples = metadata["GSE75097"]["sample_id"].astype(str).tolist()
    all_subjects = metadata["GSE71356"]["subject_id"].astype(str).unique().tolist()
    full_node = compute_node_evidence(
        expression, metadata, proteins, all_samples, all_subjects
    )
    full_context = build_pair_context(
        pairs, full_node, information_fusion=information_fusion
    )
    full_target = full_context["context_anchor"].to_numpy(dtype=np.float32)

    cv_metrics = []
    cv_predictions = []
    participant_stability = []
    best_epochs = []
    number_of_cv_folds = 1 if args.smoke else 4

    for outer_fold in range(number_of_cv_folds):
        sample_roles = dict(
            (
                unit,
                role_for_fold(fold, outer_fold, 4),
            )
            for unit, fold in sample_fold_lookup.items()
        )
        subject_roles = dict(
            (
                unit,
                role_for_fold(fold, outer_fold, 4),
            )
            for unit, fold in subject_fold_lookup.items()
        )
        protein_roles = dict(
            (
                unit,
                role_for_fold(fold, outer_fold, 4),
            )
            for unit, fold in protein_fold_lookup.items()
        )
        # Participant folds audit biological stability.  A single held-out IH
        # subject is too noisy to serve as pair-level gold truth, so every weak
        # target in this outer fold is constructed exclusively from the
        # approximately three-quarter development participants.  The protein
        # partitions below remain strictly disjoint for optimization/evaluation.
        development_samples = [
            unit for unit, value in sample_roles.items() if value != "test"
        ]
        heldout_samples = [
            unit for unit, value in sample_roles.items() if value == "test"
        ]
        development_subjects = [
            unit for unit, value in subject_roles.items() if value != "test"
        ]
        heldout_subjects = [
            unit for unit, value in subject_roles.items() if value == "test"
        ]
        development_node = compute_node_evidence(
            expression,
            metadata,
            proteins,
            development_samples,
            development_subjects,
        )
        heldout_node = compute_node_evidence(
            expression,
            metadata,
            proteins,
            heldout_samples,
            heldout_subjects,
        )
        train_context = build_pair_context(
            pairs, development_node, information_fusion=information_fusion
        )
        heldout_context = build_pair_context(
            pairs, heldout_node, information_fusion=information_fusion
        )
        features = assemble_input_matrix(pairs, train_context)
        training_target = train_context["context_anchor"].to_numpy(dtype=np.float32)
        heldout_target = heldout_context["context_anchor"].to_numpy(dtype=np.float32)
        participant_stability.append(
            {
                "outer_fold": int(outer_fold + 1),
                "development_GSE75097_units": int(len(development_samples)),
                "heldout_GSE75097_units": int(len(heldout_samples)),
                "development_GSE71356_units": int(len(development_subjects)),
                "heldout_GSE71356_units": int(len(heldout_subjects)),
                "development_vs_full_target_Spearman": spearman_correlation(
                    training_target, full_target
                ),
                "heldout_vs_full_target_Spearman": spearman_correlation(
                    heldout_target, full_target
                ),
                "development_vs_heldout_target_Spearman": spearman_correlation(
                    training_target, heldout_target
                ),
                "heldout_target_used_for_training_or_selection": False,
            }
        )
        pair_roles = []
        for first, second in zip(pairs["protein_A"], pairs["protein_B"]):
            first_role = protein_roles[first]
            second_role = protein_roles[second]
            pair_roles.append(first_role if first_role == second_role else "cross_partition")
        pair_roles = np.asarray(pair_roles, dtype=object)
        split_indices = dict(
            (role, np.where(pair_roles == role)[0])
            for role in ("train", "validation", "test")
        )
        if min(len(value) for value in split_indices.values()) == 0:
            raise RuntimeError("An OSA CV fold has an empty protein-disjoint partition")

        model, history, best_epoch, best_score = fit_adapter(
            features=features,
            targets=training_target,
            train_indices=split_indices["train"],
            validation_indices=split_indices["validation"],
            config=optimization,
            device=device,
            torch=torch,
            model_class=TRIAGEOSAModel,
            progress_prefix="[fold {}]".format(outer_fold + 1),
            validation_targets=training_target,
        )
        best_epochs.append(best_epoch)
        pd.DataFrame(history).to_csv(
            output_dir / "fold_{}_history.tsv".format(outer_fold + 1),
            sep="\t",
            index=False,
        )
        for role in ("train", "validation", "test"):
            index = split_indices[role]
            logits = model_predictions(
                model, features, index, optimization["batch_size"], device, torch
            )
            metrics = prediction_metrics(
                training_target[index], logits, features[index, 0]
            )
            expert_weights = model_expert_weights(
                model, features, index, optimization["batch_size"], device, torch
            )
            metrics["mean_expert_weight_PPI"] = float(np.mean(expert_weights[:, 0]))
            metrics["mean_expert_weight_L1_L2_L3"] = float(np.mean(expert_weights[:, 1]))
            metrics["mean_expert_weight_OSA_context"] = float(np.mean(expert_weights[:, 2]))
            metrics["mean_structural_expert_mass"] = float(
                np.mean(expert_weights[:, 0] + expert_weights[:, 1])
            )
            metrics.update(
                {
                    "outer_fold": int(outer_fold + 1),
                    "partition": role,
                    "model": "TRIAGE_OSA_specialized",
                    "best_epoch": int(best_epoch),
                    "selection_score": float(best_score),
                    "development_GSE75097_units": int(len(development_samples)),
                    "heldout_GSE75097_units": int(len(heldout_samples)),
                    "development_GSE71356_units": int(len(development_subjects)),
                    "heldout_GSE71356_units": int(len(heldout_subjects)),
                    "target_scope": "development_participants_only",
                }
            )
            cv_metrics.append(metrics)
            feature_index = dict(
                (name, position) for position, name in enumerate(INPUT_FEATURE_COLUMNS)
            )
            context_probability = np.clip(
                features[index, feature_index["context_anchor"]], 1e-6, 1.0 - 1e-6
            )
            molecular_probability = np.clip(
                train_context["molecular_context_anchor"].to_numpy(dtype=float)[index],
                1e-6, 1.0 - 1e-6,
            )
            information_probability = np.clip(
                train_context["clinical_information_anchor"].to_numpy(dtype=float)[index],
                1e-6, 1.0 - 1e-6,
            )
            comparator_logits = {
                "frozen_TRIAGE_p_PPI": features[index, feature_index["logit_PPI"]],
                "context_only_anchor": np.log(context_probability / (1.0 - context_probability)),
                "molecular_context_without_patient_information": np.log(
                    molecular_probability / (1.0 - molecular_probability)
                ),
                "patient_information_only": np.log(
                    information_probability / (1.0 - information_probability)
                ),
                "L3_only": features[index, feature_index["logit_L3"]],
                "equal_L1_L2_L3": np.mean(
                    features[index][:, [
                        feature_index["logit_L1"],
                        feature_index["logit_L2"],
                        feature_index["logit_L3"],
                    ]], axis=1
                ),
            }
            for comparator_name, comparator_logit in comparator_logits.items():
                base_metrics = prediction_metrics(
                    training_target[index], comparator_logit, features[index, 0]
                )
                base_metrics.update(
                    {
                        "outer_fold": int(outer_fold + 1),
                        "partition": role,
                        "model": comparator_name,
                        "best_epoch": 0,
                        "selection_score": float("nan"),
                        "development_GSE75097_units": int(len(development_samples)),
                        "heldout_GSE75097_units": int(len(heldout_samples)),
                        "development_GSE71356_units": int(len(development_subjects)),
                        "heldout_GSE71356_units": int(len(heldout_subjects)),
                        "target_scope": "development_participants_only",
                    }
                )
                base_metrics["mean_expert_weight_PPI"] = float("nan")
                base_metrics["mean_expert_weight_L1_L2_L3"] = float("nan")
                base_metrics["mean_expert_weight_OSA_context"] = float("nan")
                base_metrics["mean_structural_expert_mass"] = float("nan")
                cv_metrics.append(base_metrics)
            if role == "test":
                frame = pairs.iloc[index][
                    ["pair_index", "pair_id", "protein_A", "protein_B"]
                ].copy()
                frame["outer_fold"] = int(outer_fold + 1)
                frame["weak_target_development_participants"] = training_target[index]
                frame["triage_osa_logit"] = logits
                frame["frozen_TRIAGE_logit_PPI"] = features[index, 0]
                frame["triage_osa_score"] = 1.0 / (
                    1.0 + np.exp(-np.clip(logits, -30.0, 30.0))
                )
                cv_predictions.append(frame)

    metrics_frame = pd.DataFrame(cv_metrics)
    metrics_frame.to_csv(output_dir / "cross_validation_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(participant_stability).to_csv(
        output_dir / "participant_context_stability.tsv", sep="\t", index=False
    )
    pd.concat(cv_predictions, ignore_index=True).to_csv(
        output_dir / "cross_validation_test_predictions.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))

    full_features = assemble_input_matrix(pairs, full_context)
    all_indices = np.arange(len(pairs), dtype=np.int64)
    final_model, final_history, _, _ = fit_adapter(
        features=full_features,
        targets=full_target,
        train_indices=all_indices,
        validation_indices=None,
        config=optimization,
        device=device,
        torch=torch,
        model_class=TRIAGEOSAModel,
        fixed_epochs=final_epochs,
        progress_prefix="[final]",
    )
    pd.DataFrame(final_history).to_csv(
        output_dir / "final_training_history.tsv", sep="\t", index=False
    )
    full_node.to_csv(output_dir / "full_discovery_node_evidence.tsv", sep="\t", index=False)
    information_rows = []
    availability = full_node.attrs.get("patient_information_availability", {})
    for field in ("age", "sex", "BMI", "AHI", "hypertension", "EDS"):
        if field == "AHI":
            observed = int(
                pd.to_numeric(metadata["GSE75097"].get(field), errors="coerce").notna().sum()
            )
            available = int(observed >= 6)
        else:
            available = int(availability.get(field, 0))
            if field in metadata["GSE75097"]:
                observed = int(
                    metadata["GSE75097"][field].astype(str)
                    .str.strip().replace({"": np.nan, "nan": np.nan}).notna().sum()
                )
            else:
                observed = 0
        information_rows.append({
            "field": field,
            "individual_values_observed": observed,
            "included_in_v4": "yes" if available else "no",
            "role": (
                "disease_physiology" if field == "AHI"
                else "clinical_manifestation" if field in ("hypertension", "EDS")
                else "demographic_confounding_audit"
            ),
            "cohort_mean_used_as_individual_value": "no",
        })
    pd.DataFrame(information_rows).to_csv(
        output_dir / "patient_information_availability.tsv", sep="\t", index=False
    )

    # The exported all-discovery checkpoint must satisfy the same structural
    # constraint as the protein-disjoint CV models.  CV eligibility alone is
    # insufficient because the fixed-epoch final refit has its own optimizer
    # trajectory and adaptive dual state.
    final_expert_weights = model_expert_weights(
        final_model,
        full_features,
        all_indices,
        int(optimization["batch_size"]),
        device,
        torch,
    )
    final_checkpoint_expert_weights = {
        "PPI": float(np.mean(final_expert_weights[:, 0])),
        "L1_L2_L3": float(np.mean(final_expert_weights[:, 1])),
        "OSA_context": float(np.mean(final_expert_weights[:, 2])),
    }
    final_checkpoint_structural_mass = float(
        np.mean(final_expert_weights[:, 0] + final_expert_weights[:, 1])
    )
    final_checkpoint_constraint_pass = bool(
        math.isfinite(final_checkpoint_structural_mass)
        and final_checkpoint_structural_mass + 1e-7
        >= float(optimization["minimum_mean_structural_mass"])
    )

    test_metrics = metrics_frame.loc[
        (metrics_frame["partition"] == "test")
        & (metrics_frame["model"] == "TRIAGE_OSA_specialized")
    ]
    comparator_test_metrics = metrics_frame.loc[
        (metrics_frame["partition"] == "test")
        & (metrics_frame["model"] != "TRIAGE_OSA_specialized")
    ]
    base_test_metrics = comparator_test_metrics.loc[
        comparator_test_metrics["model"] == "frozen_TRIAGE_p_PPI"
    ]
    mean_test_spearman = float(test_metrics["Spearman"].mean())
    mean_test_auprc = float(test_metrics["weak_top20_AUPRC"].mean())
    mean_test_top5 = float(test_metrics["top5_enrichment_ratio"].mean())
    base_mean_test_spearman = float(base_test_metrics["Spearman"].mean())
    base_mean_test_auprc = float(base_test_metrics["weak_top20_AUPRC"].mean())
    delta_spearman = mean_test_spearman - base_mean_test_spearman
    delta_auprc = mean_test_auprc - base_mean_test_auprc
    mean_test_utility = float(test_metrics["multi_axis_utility"].mean())
    mean_test_retention = float(test_metrics["structural_retention_Spearman"].mean())
    mean_test_structural_mass = float(test_metrics["mean_structural_expert_mass"].mean())
    mean_test_expert_weights = {
        "PPI": float(test_metrics["mean_expert_weight_PPI"].mean()),
        "L1_L2_L3": float(test_metrics["mean_expert_weight_L1_L2_L3"].mean()),
        "OSA_context": float(test_metrics["mean_expert_weight_OSA_context"].mean()),
    }
    comparator_summary = comparator_test_metrics.groupby("model").agg(
        mean_test_Spearman=("Spearman", "mean"),
        mean_test_weak_top20_AUPRC=("weak_top20_AUPRC", "mean"),
        mean_test_top5_enrichment_ratio=("top5_enrichment_ratio", "mean"),
        mean_test_structural_retention=("structural_retention_Spearman", "mean"),
        mean_test_multi_axis_utility=("multi_axis_utility", "mean"),
    ).reset_index()
    comparator_summary.to_csv(
        output_dir / "internal_baseline_comparison.tsv", sep="\t", index=False
    )
    best_comparator_utility = float(
        comparator_summary["mean_test_multi_axis_utility"].max()
    )
    context_row = comparator_summary.loc[
        comparator_summary["model"] == "context_only_anchor"
    ].iloc[0]
    molecular_row = comparator_summary.loc[
        comparator_summary["model"]
        == "molecular_context_without_patient_information"
    ].iloc[0]
    delta_utility_vs_molecular_context = (
        mean_test_utility - float(molecular_row["mean_test_multi_axis_utility"])
    )
    delta_utility_vs_best = mean_test_utility - best_comparator_utility
    performance_gate = bool(
        math.isfinite(delta_spearman)
        and math.isfinite(delta_auprc)
        and math.isfinite(mean_test_top5)
        and math.isfinite(mean_test_utility)
        and delta_spearman >= 0.05
        and delta_auprc >= -0.01
        and mean_test_top5 >= 1.0
        and mean_test_spearman >= float(context_row["mean_test_Spearman"]) - 0.10
        and mean_test_retention >= float(context_row["mean_test_structural_retention"]) + 0.05
        and delta_utility_vs_best >= 0.0
        and delta_utility_vs_molecular_context >= 0.0
        and mean_test_structural_mass >= float(
            optimization["minimum_mean_structural_mass"]
        )
        and final_checkpoint_constraint_pass
    )
    if args.smoke:
        status = "PASS_SMOKE" if performance_gate else "BLOCKED_SMOKE_PERFORMANCE"
    else:
        status = (
            "PASS_FORMAL_DISCOVERY_TRAINING"
            if performance_gate
            else "BLOCKED_FORMAL_DISCOVERY_PERFORMANCE"
        )

    base_payload = torch.load(str(base_checkpoint), map_location="cpu")
    backbone_model, backbone_cfg, backbone_metadata = load_frozen_checkpoint(
        base_checkpoint, torch.device("cpu")
    )
    base_state = dict(
        (key, value.detach().cpu())
        for key, value in backbone_model.state_dict().items()
    )
    del backbone_model
    checkpoint_path = output_dir / "triage_osa_specialized.pt"
    unified_state = TRIAGEOSAModel.unified_state_dict(
        base_state,
        dict(
            (key, value.detach().cpu())
            for key, value in final_model.state_dict().items()
        ),
    )
    composite_checkpoint = {
        "checkpoint_type": "TRIAGE_Fuse_OSA_specialized_model",
        "checkpoint_version": SCRIPT_VERSION,
        "architecture_lineage": "pretrained_TRIAGE_Fuse_L1_L2_L3_plus_nonnegative_OSA_reliability_fusion_v3",
        "created_at_utc": utc_now(),
        "base_checkpoint_sha256": EXPECTED_BASE_SHA256,
        "base_checkpoint_path_at_training": str(base_checkpoint),
        "model_state_dict": unified_state,
        "state_layout": {
            "frozen_backbone_prefix": "triage_backbone.",
            "trainable_OSA_prefix": "osa_specialization.",
            "single_unified_checkpoint": True,
        },
        "base_model_config": backbone_cfg.to_dict(),
        "base_feature_spec": base_payload.get("feature_spec", {}),
        "strict_backbone_load_metadata": backbone_metadata,
        "osa_model_config": final_model.export_config(),
        "final_training_epochs": int(final_epochs),
        "cross_validation_best_epochs": [int(value) for value in best_epochs],
        "training_universe": {
            "proteins": int(len(proteins)),
            "pairs": int(len(pairs)),
            "pair_scores_sha256": sha256_file(pair_scores_path),
        },
        "training_data_sha256": dict(
            (name, sha256_file(path)) for name, path in input_paths.items()
        ),
        "training_lock_sha256": sha256_file(lock_path),
        "discovery_datasets": ["GSE75097", "GSE71356"],
        "external_validation_read": False,
        "known_PPI_or_complex_read": False,
        "score_semantics": "OSA interaction-priority score; not physical PPI probability",
        "checkpoint_selection_eligible": performance_gate,
        "internal_performance_gate": {
            "status": status,
            "delta_test_Spearman_vs_frozen_PPI": delta_spearman,
            "delta_test_weak_top20_AUPRC_vs_frozen_PPI": delta_auprc,
            "mean_test_top5_enrichment_ratio": mean_test_top5,
            "mean_test_multi_axis_utility": mean_test_utility,
            "delta_multi_axis_utility_vs_best_single_baseline": delta_utility_vs_best,
            "mean_test_structural_expert_mass": mean_test_structural_mass,
            "final_checkpoint_structural_expert_mass": final_checkpoint_structural_mass,
            "final_checkpoint_structural_constraint_pass": final_checkpoint_constraint_pass,
        },
    }
    torch.save(composite_checkpoint, str(checkpoint_path))

    contract = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "lock_file": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": EXPECTED_BASE_SHA256,
        "pair_scores": str(pair_scores_path),
        "pair_scores_sha256": sha256_file(pair_scores_path),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in final_model.parameters())
        ),
        "frozen_backbone_embedded_in_checkpoint": True,
        "original_TRIAGEPPIModel_instantiated_and_strict_loaded": True,
        "single_unified_model_state_dict": True,
        "detached_adapter_checkpoint": False,
        "L1_L2_L3_T_MEB_preserved": True,
        "context_conditioned_gate": True,
        "reliability_aware_residual": True,
        "nonnegative_convex_expert_fusion": True,
        "pair_adaptive_expert_weights": True,
        "per_pair_structural_mass_constrained_simplex": True,
        "constraint_triggered_structural_distillation": True,
        "adaptive_structural_dual": True,
        "soft_structural_mass_constraint": True,
        "minimum_contribution_for_each_L1_L2_L3_branch": True,
        "context_anchor_preserves_locked_r_OSA_semantics": True,
        "patient_information_conditioning": bool(information_fusion),
        "patient_information_fields": [
            "age", "sex", "BMI", "AHI", "hypertension", "EDS"
        ],
        "cohort_summary_values_used_as_individual_features": False,
        "clinical_outcome_prediction": False,
        "selection_compares_PPI_context_L3_and_equal_L1_L2_L3": True,
        "participant_leaveout_context_audit": True,
        "single_participant_target_used_for_model_selection": False,
        "protein_disjoint_cross_validation": True,
        "input_feature_columns": list(INPUT_FEATURE_COLUMNS),
        "external_validation_read": False,
        "forbidden_stages_read": [],
        "final_epoch_rule": "median discovery-only CV best epoch",
        "final_epochs": int(final_epochs),
        "final_checkpoint_expert_weights": final_checkpoint_expert_weights,
        "final_checkpoint_structural_expert_mass": final_checkpoint_structural_mass,
        "final_checkpoint_structural_constraint_pass": final_checkpoint_constraint_pass,
        "smoke": bool(args.smoke),
    }
    write_json(output_dir / "training_contract.json", contract)
    summary = {
        "status": status,
        "created_at_utc": utc_now(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "proteins": int(len(proteins)),
        "pairs": int(len(pairs)),
        "cross_validation_folds": int(number_of_cv_folds),
        "cross_validation_best_epochs": best_epochs,
        "final_epochs": int(final_epochs),
        "evaluation_semantics": "protein-disjoint recovery of discovery weak targets built from development participants; not external biological validation",
        "patient_information_conditioning": bool(information_fusion),
        "patient_information_availability": availability,
        "clinical_outcome_prediction": False,
        "mean_test_Spearman": mean_test_spearman,
        "mean_test_weak_top20_AUPRC": mean_test_auprc,
        "mean_test_top5_enrichment_ratio": mean_test_top5,
        "mean_test_structural_retention_Spearman": mean_test_retention,
        "mean_test_structural_expert_mass": mean_test_structural_mass,
        "mean_test_expert_weights": mean_test_expert_weights,
        "final_checkpoint_expert_weights": final_checkpoint_expert_weights,
        "final_checkpoint_structural_expert_mass": final_checkpoint_structural_mass,
        "final_checkpoint_structural_constraint_pass": final_checkpoint_constraint_pass,
        "mean_test_multi_axis_utility": mean_test_utility,
        "best_single_baseline_multi_axis_utility": best_comparator_utility,
        "delta_multi_axis_utility_vs_best_single_baseline": delta_utility_vs_best,
        "delta_multi_axis_utility_vs_molecular_context_without_patient_information": delta_utility_vs_molecular_context,
        "base_mean_test_Spearman": base_mean_test_spearman,
        "base_mean_test_weak_top20_AUPRC": base_mean_test_auprc,
        "delta_test_Spearman_vs_frozen_PPI": delta_spearman,
        "delta_test_weak_top20_AUPRC_vs_frozen_PPI": delta_auprc,
        "performance_gate": {
            "passed": performance_gate,
            "minimum_delta_Spearman": 0.05,
            "minimum_delta_weak_top20_AUPRC": -0.01,
            "minimum_top5_enrichment_ratio": 1.0,
            "minimum_context_rank_preservation": -0.10,
            "minimum_structural_retention_gain_vs_context_only": 0.05,
            "minimum_multi_axis_utility_gain_vs_best_single_baseline": 0.0,
            "minimum_multi_axis_utility_gain_vs_molecular_context_without_patient_information": 0.0,
            "minimum_mean_structural_expert_mass": float(
                optimization["minimum_mean_structural_mass"]
            ),
            "final_checkpoint_must_pass_structural_constraint": True,
        },
        "single_heldout_participant_used_as_gold_target": False,
        "external_validation_read": False,
        "known_PPI_or_complex_read": False,
        "checkpoint_selection_eligible": performance_gate,
        "next_step": (
            "Run formal four-fold discovery training; do not select it using Stages 10-19."
            if args.smoke and performance_gate
            else (
                "Load with predict_triage_osa.py, then evaluate once on locked Stages 10-19."
                if performance_gate
                else "Do not use this checkpoint; revise the discovery-only adapter before formal training or validation."
            )
        ),
    }
    write_json(output_dir / "run_summary.json", summary)
    print("TRIAGE-OSA specialization training: {}".format(summary["status"]))
    print("Checkpoint: {}".format(checkpoint_path))
    print("Checkpoint SHA256: {}".format(summary["checkpoint_sha256"]))
    print("Mean protein-disjoint test Spearman: {:.4f}".format(summary["mean_test_Spearman"]))
    print("Delta Spearman versus frozen PPI: {:.4f}".format(delta_spearman))
    print("Performance gate: {}".format("PASS" if performance_gate else "FAIL"))
    print("External validation read: no")
    return 0 if performance_gate else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
