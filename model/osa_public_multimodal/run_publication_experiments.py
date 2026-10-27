#!/usr/bin/env python3
"""TRIAGE-OSA public-data multimodal manuscript experiments.

This program does not retrain or alter the frozen V4 pair ranking.  It exposes
three manuscript-level outputs:

1. frozen pair-level L1-L2-L3 / OSA / patient-information evidence;
2. participant-leaveout molecular profiles conditional on frozen candidates;
3. exploratory phenotype-association and modality-ablation statistics.

The third output is not a clinical diagnosis or treatment-response model.
External CPAP and validation datasets are never read by this program.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROGRAM_VERSION = "1.1.0"
AXES = ("OSA_state", "severity", "AHI", "hypertension", "EDS")
VARIANTS = (
    "phenotype_expression_only",
    "molecular_plus_phenotype",
    "structural_plus_phenotype",
    "full_tri_modal",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--lock",
        default="model/osa_public_multimodal/experiment_lock.json",
    )
    parser.add_argument("--output-stage", default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--publication-supplement", action="store_true",
                        help="Add baselines, readout ablations and conditional uncertainty; do not retrain V4")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def utc_timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value):
    if isinstance(value, dict):
        return dict((str(key), json_safe(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def rank_average(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(
        method="average"
    ).to_numpy(dtype=float)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    rx = rank_average(x[valid])
    ry = rank_average(y[valid])
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def binary_auc(y, score):
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(score)
    y = y[valid].astype(int)
    score = score[valid]
    positive = y == 1
    negative = y == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    ranks = rank_average(score)
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def signed_percentile(raw):
    raw = np.asarray(raw, dtype=float)
    result = np.zeros(raw.shape, dtype=float)
    valid = np.isfinite(raw)
    if valid.sum() == 0:
        return result
    absolute = np.abs(raw[valid])
    ranks = rank_average(absolute)
    denominator = max(1.0, float(valid.sum() - 1))
    result[valid] = np.sign(raw[valid]) * ((ranks - 1.0) / denominator)
    return result


def continuous_effect(expression, target):
    expression = np.asarray(expression, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(target)
    if valid.sum() < 6 or np.nanstd(target[valid]) <= 1e-12:
        return np.full(expression.shape[0], np.nan, dtype=float)
    x = target[valid]
    x = x - np.mean(x)
    x_norm = np.sqrt(np.sum(x * x))
    y = expression[:, valid]
    y = y - np.nanmean(y, axis=1, keepdims=True)
    numerator = np.nansum(y * x.reshape(1, -1), axis=1)
    denominator = np.sqrt(np.nansum(y * y, axis=1)) * x_norm
    return np.divide(
        numerator,
        denominator,
        out=np.full(expression.shape[0], np.nan, dtype=float),
        where=denominator > 1e-12,
    )


def binary_effect(expression, target):
    expression = np.asarray(expression, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(target)
    positive = valid & (target > 0.5)
    negative = valid & (target <= 0.5)
    if positive.sum() < 2 or negative.sum() < 2:
        return np.full(expression.shape[0], np.nan, dtype=float)
    difference = np.nanmean(expression[:, positive], axis=1) - np.nanmean(
        expression[:, negative], axis=1
    )
    joined = expression[:, valid]
    scale = np.nanstd(joined, axis=1, ddof=1)
    fallback = np.nanmedian(scale[np.isfinite(scale) & (scale > 1e-8)])
    if not np.isfinite(fallback):
        fallback = 1.0
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    return difference / scale


def target_values(metadata, axis):
    if axis == "OSA_state":
        return (metadata["study_group"].astype(str) != "PS").astype(float).to_numpy()
    if axis == "severity":
        mapping = {"PS": 0.0, "MSO": 1.0, "VSO": 2.0}
        return metadata["study_group"].astype(str).map(mapping).to_numpy(dtype=float)
    if axis == "AHI":
        return pd.to_numeric(metadata["AHI_events_h"], errors="coerce").to_numpy(dtype=float)
    return pd.to_numeric(metadata[axis], errors="coerce").to_numpy(dtype=float)


def is_binary_axis(axis):
    return axis in ("OSA_state", "hypertension", "EDS")


def adjust_expression_for_age_sex(train_expression, test_expression, train_meta, test_meta):
    age = pd.to_numeric(train_meta["age_years"], errors="coerce").to_numpy(dtype=float)
    age_mean = float(np.nanmean(age))
    age_std = float(np.nanstd(age, ddof=1))
    if not np.isfinite(age_std) or age_std <= 1e-8:
        age_std = 1.0
    age = np.where(np.isfinite(age), age, age_mean)
    sex = train_meta["sex"].astype(str).str.lower().eq("male").astype(float).to_numpy()
    design = np.column_stack([np.ones(len(train_meta)), (age - age_mean) / age_std, sex])
    test_age = pd.to_numeric(test_meta["age_years"], errors="coerce").to_numpy(dtype=float)
    test_age = np.where(np.isfinite(test_age), test_age, age_mean)
    test_sex = test_meta["sex"].astype(str).str.lower().eq("male").astype(float).to_numpy()
    test_design = np.column_stack(
        [np.ones(len(test_meta)), (test_age - age_mean) / age_std, test_sex]
    )
    coefficients = np.linalg.pinv(design).dot(train_expression.T)
    train_residual = train_expression.T - design.dot(coefficients)
    test_residual = test_expression.T - test_design.dot(coefficients)
    mean = np.nanmean(train_residual, axis=0)
    std = np.nanstd(train_residual, axis=0, ddof=1)
    fallback = np.nanmedian(std[np.isfinite(std) & (std > 1e-8)])
    if not np.isfinite(fallback):
        fallback = 1.0
    std = np.where(np.isfinite(std) & (std > 1e-8), std, fallback)
    train_z = ((train_residual - mean) / std).T
    test_z = ((test_residual - mean) / std).T
    return np.nan_to_num(train_z), np.nan_to_num(test_z)


def aggregate_structural_nodes(pairs, proteins, levels=("p_L1", "p_L2", "p_L3")):
    probability = np.clip(
        pairs[list(levels)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float),
        1e-6,
        1.0,
    )
    structural = np.exp(np.mean(np.log(probability), axis=1))
    frame_a = pd.DataFrame({"protein": pairs["protein_A"].astype(str), "score": structural})
    frame_b = pd.DataFrame({"protein": pairs["protein_B"].astype(str), "score": structural})
    edges = pd.concat([frame_a, frame_b], ignore_index=True)

    def top_mean(series):
        values = np.sort(series.to_numpy(dtype=float))
        return float(np.mean(values[-min(10, len(values)) :]))

    node = edges.groupby("protein", sort=False)["score"].apply(top_mean)
    node = node.reindex(proteins).fillna(float(node.median())).to_numpy(dtype=float)
    ranks = rank_average(node)
    return (ranks - 1.0) / max(1.0, float(len(node) - 1))


def calculate_disease_strength(train_z, train_meta):
    osa = target_values(train_meta, "OSA_state")
    severity = target_values(train_meta, "severity")
    ahi = target_values(train_meta, "AHI")
    evidence = np.column_stack(
        [
            np.abs(signed_percentile(binary_effect(train_z, osa))),
            np.abs(signed_percentile(continuous_effect(train_z, severity))),
            np.abs(signed_percentile(continuous_effect(train_z, ahi))),
        ]
    )
    return np.nanmean(evidence, axis=1)


def modality_weights(target_signed, disease_strength, structural_strength, variant):
    phenotype_strength = np.abs(target_signed)
    weight = target_signed.copy()
    if variant in ("molecular_plus_phenotype", "full_tri_modal"):
        weight *= 0.25 + 0.75 * disease_strength
    if variant in ("structural_plus_phenotype", "full_tri_modal"):
        weight *= 0.25 + 0.75 * structural_strength
    # Preserve the sign but prevent a single near-zero feature family from
    # eliminating all phenotype evidence.
    weight *= 0.25 + 0.75 * phenotype_strength
    denominator = float(np.sum(np.abs(weight)))
    return weight / denominator if denominator > 1e-12 else weight


def weight_diagnostics(weight, disease_strength, structural_strength):
    absolute = np.abs(np.asarray(weight, dtype=float))
    keep = max(1, int(math.ceil(0.10 * len(absolute))))
    selected = np.argsort(-absolute, kind="mergesort")[:keep]
    probability = absolute / max(float(absolute.sum()), 1e-12)
    positive = probability[probability > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return {
        "top_weight_structural_support": float(np.mean(structural_strength[selected])),
        "top_weight_OSA_molecular_support": float(np.mean(disease_strength[selected])),
        "effective_weighted_proteins": float(np.exp(entropy)),
    }


def build_profiles(expression, metadata, structural_strength, variants=VARIANTS):
    sample_ids = metadata["sample_id"].astype(str).tolist()
    rows = []
    for heldout, sample_id in enumerate(sample_ids):
        train_index = np.asarray([i for i in range(len(sample_ids)) if i != heldout])
        test_index = np.asarray([heldout])
        train_meta = metadata.iloc[train_index].reset_index(drop=True)
        test_meta = metadata.iloc[test_index].reset_index(drop=True)
        train_z, test_z = adjust_expression_for_age_sex(
            expression[:, train_index], expression[:, test_index], train_meta, test_meta
        )
        disease_strength = calculate_disease_strength(train_z, train_meta)
        for axis in AXES:
            train_target = target_values(train_meta, axis)
            effect = (
                binary_effect(train_z, train_target)
                if is_binary_axis(axis)
                else continuous_effect(train_z, train_target)
            )
            target_signed = signed_percentile(effect)
            observed_target = float(target_values(test_meta, axis)[0])
            for variant in variants:
                weight = modality_weights(
                    target_signed, disease_strength, structural_strength, variant
                )
                score = float(np.dot(weight, test_z[:, 0]))
                row = {
                        "dataset": "GSE75097",
                        "sample_id": sample_id,
                        "subject_id": str(test_meta.iloc[0]["subject_id"]),
                        "study_group": str(test_meta.iloc[0]["study_group"]),
                        "axis": axis,
                        "variant": variant,
                        "mechanism_profile_score": score,
                        "observed_phenotype": observed_target,
                        "age_years": float(test_meta.iloc[0]["age_years"]),
                        "sex": str(test_meta.iloc[0]["sex"]),
                        "BMI_available": "no",
                        "heldout_participant_used_for_weight_construction": "no",
                        "demographic_adjustment": "age+sex",
                        "clinical_probability": "no",
                    }
                row.update(weight_diagnostics(weight, disease_strength, structural_strength))
                rows.append(row)
    return pd.DataFrame(rows)


def profile_metrics(profiles):
    rows = []
    for (axis, variant), part in profiles.groupby(["axis", "variant"], sort=False):
        target = part["observed_phenotype"].to_numpy(dtype=float)
        score = part["mechanism_profile_score"].to_numpy(dtype=float)
        metric_name = "AUROC" if is_binary_axis(axis) else "Spearman"
        metric = binary_auc(target, score) if is_binary_axis(axis) else spearman(target, score)
        structural_support = float(part["top_weight_structural_support"].mean())
        molecular_support = float(part["top_weight_OSA_molecular_support"].mean())
        association_quality = (
            max(0.0, 2.0 * (metric - 0.5))
            if metric_name == "AUROC"
            else max(0.0, metric)
        )
        multi_axis_utility = (
            0.50 * association_quality
            + 0.25 * structural_support
            + 0.25 * molecular_support
        )
        rows.append(
            {
                "axis": axis,
                "variant": variant,
                "metric": metric_name,
                "value": metric,
                "participants": int(len(part)),
                "association_quality_0_to_1": association_quality,
                "mean_top_weight_structural_support": structural_support,
                "mean_top_weight_OSA_molecular_support": molecular_support,
                "mean_effective_weighted_proteins": float(
                    part["effective_weighted_proteins"].mean()
                ),
                "multi_axis_utility": multi_axis_utility,
                "interpretation": "exploratory participant-isolated association; not clinical accuracy",
            }
        )
    result = pd.DataFrame(rows)
    result["pareto_optimal_within_axis"] = "no"
    objectives = [
        "association_quality_0_to_1",
        "mean_top_weight_structural_support",
        "mean_top_weight_OSA_molecular_support",
    ]
    for axis, part in result.groupby("axis", sort=False):
        values = part[objectives].to_numpy(dtype=float)
        pareto = np.ones(len(part), dtype=bool)
        for index in range(len(part)):
            dominated = np.all(values >= values[index], axis=1) & np.any(
                values > values[index], axis=1
            )
            if bool(dominated.any()):
                pareto[index] = False
        result.loc[part.index, "pareto_optimal_within_axis"] = np.where(
            pareto, "yes", "no"
        )
    return result


def permute_metadata(metadata, permutation):
    result = metadata.copy()
    columns = ["study_group", "AHI_events_h", "hypertension", "EDS"]
    result.loc[:, columns] = metadata.iloc[permutation][columns].to_numpy()
    return result


def permutation_test(expression, metadata, structural_strength, observed, permutations, seed):
    rng = np.random.RandomState(int(seed))
    observed_full = observed.loc[observed["variant"] == "full_tri_modal"].set_index("axis")
    exceed = dict((axis, 0) for axis in AXES)
    null_sum = dict((axis, 0.0) for axis in AXES)
    null_square = dict((axis, 0.0) for axis in AXES)
    for iteration in range(int(permutations)):
        order = rng.permutation(len(metadata))
        null_profiles = build_profiles(
            expression,
            permute_metadata(metadata, order),
            structural_strength,
            variants=("full_tri_modal",),
        )
        null_metrics = profile_metrics(null_profiles)
        null_metrics = null_metrics.loc[
            null_metrics["variant"] == "full_tri_modal"
        ].set_index("axis")
        for axis in AXES:
            value = float(null_metrics.loc[axis, "value"])
            observed_value = float(observed_full.loc[axis, "value"])
            if np.isfinite(value):
                null_sum[axis] += value
                null_square[axis] += value * value
                if value >= observed_value:
                    exceed[axis] += 1
        if (iteration + 1) % max(1, min(1000, int(permutations) // 10)) == 0:
            print("[permutation] {}/{}".format(iteration + 1, permutations))
    rows = []
    for axis in AXES:
        n = float(permutations)
        mean = null_sum[axis] / n
        variance = max(0.0, null_square[axis] / n - mean * mean)
        rows.append(
            {
                "axis": axis,
                "observed_metric": float(observed_full.loc[axis, "value"]),
                "metric": str(observed_full.loc[axis, "metric"]),
                "null_mean": mean,
                "null_sd": math.sqrt(variance),
                "one_sided_permutation_P": (1.0 + exceed[axis]) / (1.0 + permutations),
                "permutations": int(permutations),
            }
        )
    result = pd.DataFrame(rows)
    order = np.argsort(result["one_sided_permutation_P"].to_numpy(dtype=float))
    adjusted = np.empty(len(result), dtype=float)
    running = 1.0
    for reverse_rank in range(len(order) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        value = min(1.0, float(result.loc[index, "one_sided_permutation_P"]) * len(order) / rank)
        running = min(running, value)
        adjusted[index] = running
    result["BH_FDR_across_profile_axes"] = adjusted
    return result


def pair_output(pairs):
    columns = [
        "pair_id", "protein_A", "protein_B", "gene_A", "gene_B",
        "p_PPI", "p_L1", "p_L2", "p_L3", "g_L1", "g_L2", "g_L3",
        "evidence_reliability", "molecular_context_anchor",
        "clinical_information_anchor", "context_anchor", "triage_osa_score",
        "osa_expert_weight_PPI", "osa_expert_weight_L1_L2_L3",
        "osa_expert_weight_OSA_context", "triage_osa_rank_primary_250",
    ]
    missing = sorted(set(columns) - set(pairs.columns))
    if missing:
        raise RuntimeError("Frozen V4 ranking lacks: {}".format(",".join(missing)))
    result = pairs[columns].copy()
    probability = np.clip(
        result[["p_L1", "p_L2", "p_L3"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float),
        1e-6,
        1.0,
    )
    result["structural_L1_L2_L3_score"] = np.exp(np.mean(np.log(probability), axis=1))
    result["output_layer_1_semantics"] = "population-level OSA interaction priority"
    result["physical_PPI_probability"] = "no"
    result["clinical_probability"] = "no"
    return result


def load_inputs(root, lock):
    paths = dict((key, resolve(root, value)) for key, value in lock["frozen_inputs"].items())
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing frozen inputs: {}".format(missing))
    pairs = pd.read_csv(paths["ranking"], sep="\t", low_memory=False)
    if pairs["pair_id"].duplicated().any():
        raise RuntimeError("Frozen V4 ranking contains duplicate pair_id")
    proteins = sorted(set(pairs["protein_A"].astype(str)) | set(pairs["protein_B"].astype(str)))
    expression = pd.read_csv(paths["expression"], sep="\t", index_col=0)
    expression.index = expression.index.astype(str)
    metadata = pd.read_csv(paths["patient_information"], sep="\t")
    metadata = metadata.loc[metadata["dataset"].astype(str) == "GSE75097"].copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata = metadata.loc[metadata["sample_id"].isin(expression.columns.astype(str))].copy()
    metadata = metadata.sort_values("sample_id").reset_index(drop=True)
    if metadata["sample_id"].duplicated().any():
        raise RuntimeError("GSE75097 patient information contains duplicate samples")
    if metadata["subject_id"].isna().any() or metadata["subject_id"].duplicated().any():
        raise RuntimeError("LOPO requires exactly one sample per identified participant")
    if not metadata["study_group"].isin(["PS", "MSO", "VSO"]).all():
        raise RuntimeError("Unexpected study group; PS means primary snoring, not healthy control")
    if not metadata["sex"].astype(str).str.lower().isin(["male", "female"]).all():
        raise RuntimeError("Unknown sex cannot be silently encoded as female")
    if not np.isfinite(pd.to_numeric(metadata["age_years"], errors="coerce")).all():
        raise RuntimeError("Age is missing; an explicit fold-local missing-data policy is required")
    for axis in AXES:
        target = target_values(metadata, axis)
        if not np.isfinite(target).all():
            raise RuntimeError("Missing phenotype requires an axis-specific inclusion rule: " + axis)
        if is_binary_axis(axis) and not np.isin(target, [0.0, 1.0]).all():
            raise RuntimeError("Non-binary phenotype: " + axis)
    if expression.index.duplicated().any() or expression.columns.duplicated().any():
        raise RuntimeError("Duplicate expression identifiers")
    if metadata["cohort_summary_used_as_individual_value"].astype(str).str.lower().ne("no").any():
        raise RuntimeError("Cohort summaries cannot be used as individual values")
    observed_proteins = [protein for protein in proteins if protein in expression.index]
    if len(observed_proteins) < int(0.90 * len(proteins)):
        raise RuntimeError("Too few primary proteins have GSE75097 expression")
    expression = expression.loc[observed_proteins, metadata["sample_id"].tolist()]
    if not np.isfinite(expression.to_numpy(dtype=float)).all():
        raise RuntimeError("Nonfinite expression; do not silently convert it into zero")
    return paths, pairs, observed_proteins, expression.to_numpy(dtype=float), metadata


def fixed_ridge_score(train_x, test_x, target, include_intercept=True):
    """Fixed lambda=1 for mean squared loss; continuous ranking, not probability.

    Features are already train-standardized. Divide by sqrt(p) so feature
    count does not itself change kernel scale. No tuning on heldout outcomes.
    """
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    center = train_x.mean(axis=0)
    train_x = (train_x - center) / math.sqrt(train_x.shape[1])
    test_x = (test_x - center) / math.sqrt(test_x.shape[1])
    target = np.asarray(target, dtype=float)
    kernel = train_x.dot(train_x.T)
    alpha = np.linalg.solve(kernel + len(target) * np.eye(len(target)), target - target.mean())
    component = float(test_x.dot(train_x.T).dot(alpha)[0])
    return float(target.mean() + component) if include_intercept else component


def build_simple_baselines(expression, all_expression, all_proteins, metadata):
    """Use the same LOPO participants; feature-selection sensitivity is a separate estimand."""
    rows, selections = [], []
    for heldout in range(len(metadata)):
        ix = np.asarray([i for i in range(len(metadata)) if i != heldout])
        train = metadata.iloc[ix].reset_index(drop=True)
        test = metadata.iloc[[heldout]].reset_index(drop=True)
        train_z, test_z = adjust_expression_for_age_sex(expression[:, ix], expression[:, [heldout]], train, test)
        all_z, all_test_z = adjust_expression_for_age_sex(all_expression[:, ix], all_expression[:, [heldout]], train, test)
        raw_demographics = np.column_stack([
            pd.to_numeric(metadata["age_years"]).to_numpy(dtype=float),
            metadata["sex"].str.lower().eq("male").to_numpy(dtype=float),
        ])
        d_mean = raw_demographics[ix].mean(axis=0)
        d_std = raw_demographics[ix].std(axis=0, ddof=1)
        d_std = np.where(d_std > 1e-8, d_std, 1.0)
        d_train = (raw_demographics[ix] - d_mean) / d_std
        d_test = (raw_demographics[[heldout]] - d_mean) / d_std
        for axis in AXES:
            target = target_values(train, axis)
            effect = binary_effect(all_z, target) if is_binary_axis(axis) else continuous_effect(all_z, target)
            # Selection is repeated inside every leaveout fold, using all mapped
            # expression proteins rather than the original phenotype-selected 250.
            selected = np.argsort(-np.abs(effect), kind="mergesort")[:min(250, len(effect))]
            signed = signed_percentile(effect[selected])
            weights = modality_weights(signed, np.ones(len(selected)), np.ones(len(selected)), "phenotype_expression_only")
            scores = {
                "age_sex_ridge_raw_prediction": fixed_ridge_score(d_train, d_test, target),
                "frozen250_ridge_raw_prediction": fixed_ridge_score(train_z.T, test_z.T, target),
                # LOO training means depend on the omitted outcome. Keep raw
                # predictions AND intercept-free components, with an intercept-
                # only diagnostic, rather than reversing a disappointing score.
                "age_sex_ridge_signal_component": fixed_ridge_score(d_train, d_test, target, False),
                "frozen250_ridge_signal_component": fixed_ridge_score(train_z.T, test_z.T, target, False),
                "LOPO_training_mean_diagnostic": float(target.mean()),
                "fold_selected250_expression_only": float(weights.dot(all_test_z[selected, 0])),
            }
            for variant, score in scores.items():
                rows.append({"dataset": "GSE75097", "sample_id": str(test.iloc[0]["sample_id"]),
                             "subject_id": str(test.iloc[0]["subject_id"]), "axis": axis,
                             "variant": variant, "mechanism_profile_score": score,
                             "observed_phenotype": float(target_values(test, axis)[0]),
                             "clinical_probability": "no"})
            selections.extend({"heldout_subject_id": str(test.iloc[0]["subject_id"]),
                               "axis": axis, "protein": str(all_proteins[j]),
                               "selection_rank": k + 1,
                               "selection_used_heldout_phenotype": "no"}
                              for k, j in enumerate(selected))
    return pd.DataFrame(rows), pd.DataFrame(selections)


def conditional_bootstrap(profiles, iterations, seed):
    """Paired resampling of OOF score rows, NOT full-pipeline bootstrap CIs.

    Binary axes are class-stratified. These descriptive intervals condition on
    fitted LOPO scores and cannot capture overlapping-training-set uncertainty
    or global candidate-selection optimism. No p-value is derived from them.
    """
    rng = np.random.RandomState(seed)
    estimates, differences = [], []
    for axis in AXES:
        part = profiles.loc[profiles["axis"] == axis]
        wide = part.pivot(index="subject_id", columns="variant", values="mechanism_profile_score")
        target_table = part.groupby("subject_id")["observed_phenotype"].agg(["first", "nunique"])
        if not target_table["nunique"].eq(1).all() or wide.isna().any().any():
            raise RuntimeError("Unmatched participant predictions or inconsistent targets")
        y = target_table.loc[wide.index, "first"].to_numpy(dtype=float)
        names = wide.columns.tolist()
        values = wide.to_numpy(dtype=float)
        metric = binary_auc if is_binary_axis(axis) else spearman
        point = np.asarray([metric(y, values[:, j]) for j in range(len(names))])
        draws = np.full((iterations, len(names)), np.nan)
        for b in range(iterations):
            if is_binary_axis(axis):
                strata = [np.flatnonzero(y == label) for label in (0, 1)]
                idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            else:
                idx = rng.randint(0, len(y), size=len(y))
            for j in range(len(names)):
                draws[b, j] = metric(y[idx], values[idx, j])
        full = names.index("full_tri_modal")
        for j, name in enumerate(names):
            finite = draws[:, j][np.isfinite(draws[:, j])]
            low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            estimates.append({"axis": axis, "variant": name, "metric": "AUROC" if is_binary_axis(axis) else "Spearman",
                              "estimate": point[j], "conditional_interval_low": low,
                              "conditional_interval_high": high, "participants": len(y),
                              "positive_participants": int((y == 1).sum()) if is_binary_axis(axis) else np.nan,
                              "negative_participants": int((y == 0).sum()) if is_binary_axis(axis) else np.nan,
                              "bootstrap_requested": iterations, "bootstrap_valid": len(finite),
                              "interval_scope": "descriptive_fixed_OOF_scores_not_full_pipeline_CI"})
            if j != full:
                delta = draws[:, full] - draws[:, j]
                delta = delta[np.isfinite(delta)]
                lo, hi = np.percentile(delta, [2.5, 97.5]) if len(delta) else (np.nan, np.nan)
                differences.append({"axis": axis, "reference": "full_tri_modal", "comparator": name,
                                    "delta_estimate": point[full] - point[j],
                                    "conditional_interval_low": lo, "conditional_interval_high": hi,
                                    "paired_subjects": len(y), "bootstrap_valid": len(delta),
                                    "interpretation": "descriptive_paired_difference_not_confirmatory_test"})
        print("[bootstrap] {}: {} paired resamples".format(axis, iterations), flush=True)
    return pd.DataFrame(estimates), pd.DataFrame(differences)


def readout_rank_sensitivity(pairs):
    """Static score controls, not model retraining and not physical-PPI accuracy."""
    levels = pairs[["p_L1", "p_L2", "p_L3"]].to_numpy(dtype=float)
    controls = {"L1_only": levels[:, 0], "L2_only": levels[:, 1], "L3_only": levels[:, 2],
                "equal_L1_L2_L3": levels.mean(axis=1), "frozen_PPI": pairs["p_PPI"].to_numpy(dtype=float),
                "molecular_context": pairs["molecular_context_anchor"].to_numpy(dtype=float),
                "patient_information": pairs["clinical_information_anchor"].to_numpy(dtype=float)}
    reference = pairs["triage_osa_score"].to_numpy(dtype=float)
    rows = []
    for name, scores in controls.items():
        for fraction in (0.01, 0.05, 0.10):
            k = int(math.ceil(len(pairs) * fraction))
            a = set(np.argsort(-reference, kind="mergesort")[:k])
            b = set(np.argsort(-scores, kind="mergesort")[:k])
            rows.append({"comparator": name, "pairs": len(pairs), "top_fraction": fraction,
                         "top_k": k, "overlap_fraction": len(a & b) / float(k),
                         "Jaccard": len(a & b) / float(len(a | b)), "Spearman_vs_V4": spearman(reference, scores),
                         "interpretation": "ranking_sensitivity_not_accuracy_or_retrained_ablation"})
    return pd.DataFrame(rows)


def publication_inventory(root):
    """Inspect completion metadata only; endpoint values never select a model."""
    base = root / "results" / "discovery_preprocess_v2"
    stages = ["06_model_training/triage_osa_patient_information_v4",
              "09_pair_context_ranking_triage_osa_v4", "11_cpap_reversal_endpoint_triage_osa_v4",
              "12_cpap_subject_signflip_triage_osa_v4", "14_adipose_cross_tissue_endpoint_triage_osa_v4",
              "16_scrna_cell_context_endpoint_triage_osa_v4", "18_pediatric_proteome_endpoint_triage_osa_v4",
              "19_known_ppi_complex_recovery_triage_osa_v4", "21_independent_validation_endpoint_triage_osa_v4",
              "22_consolidated_evidence_triage_osa_v4", "25_cpap_subject_clinical_package_triage_osa_v4",
              "26_v4_publication_figures", "27_public_multimodal_experiments_v1"]
    rows = []
    for stage in stages:
        path = base / stage / "run_summary.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                status = json.load(handle).get("status", "STATUS_MISSING")
        else:
            status = "NOT_SYNCED_OR_NOT_RUN"
        rows.append({"stage": stage, "reported_status": status,
                     "interpretation": "file_completion_not_evidence_of_positive_biological_result"})
    return pd.DataFrame(rows)


def run_publication_supplement(root, lock, args, paths, pairs, proteins, expression, metadata):
    if args.permutations is not None:
        raise RuntimeError("Supplement uses --bootstrap, not --permutations; original permutation results remain untouched")
    iterations = 100 if args.smoke else args.bootstrap
    if iterations < 100:
        raise RuntimeError("Use at least 100 bootstrap resamples")
    default_stage = "29_publication_robustness_v1" + ("_smoke" if args.smoke else "")
    stage = args.output_stage or default_stage
    if Path(stage).name != stage or not stage.startswith("29_publication_robustness"):
        raise RuntimeError("Supplement output-stage must be a 29_publication_robustness* directory name")
    output = root / "results" / "discovery_preprocess_v2" / stage
    source = root / "results" / "discovery_preprocess_v2" / "27_public_multimodal_experiments_v1"
    if output.resolve() == source.resolve():
        raise RuntimeError("Supplement must not overwrite Stage 27")
    if output.exists() and not args.force and not args.check_env:
        raise RuntimeError("Output exists; use --force: " + str(output))
    all_frame = pd.read_csv(paths["expression"], sep="\t", index_col=0)
    all_frame = all_frame.sort_index().loc[:, metadata["sample_id"].tolist()]
    if all_frame.index.duplicated().any() or not np.isfinite(all_frame.to_numpy(dtype=float)).all():
        raise RuntimeError("All-mapped-expression sensitivity requires unique finite features")
    if not (source / "modality_ablation_metrics.tsv").is_file():
        raise RuntimeError("Original Stage 27 metrics are required for reproducibility audit")
    print("[supplement] Frozen V4; all mapped proteins={}; no external endpoint values used".format(len(all_frame)), flush=True)
    if args.check_env:
        print("TRIAGE-OSA publication supplement preflight: PASS", flush=True)
        return 0
    structural = aggregate_structural_nodes(pairs, proteins)
    profiles = build_profiles(expression, metadata, structural)
    original = pd.read_csv(source / "modality_ablation_metrics.tsv", sep="\t")
    reproduction = profile_metrics(profiles).merge(original[["axis", "variant", "value"]],
                                                  on=["axis", "variant"], suffixes=("_recomputed", "_original"),
                                                  validate="one_to_one")
    if len(reproduction) != len(AXES) * len(VARIANTS):
        raise RuntimeError("Original Stage 27 comparison grid is incomplete")
    reproduction["absolute_difference"] = abs(reproduction["value_recomputed"] - reproduction["value_original"])
    if not reproduction["absolute_difference"].le(1e-10).all():
        raise RuntimeError("Stage 27 scores do not reproduce; stop before supplementary comparisons")
    frames = [profiles]
    for label, levels in [("L1_only", ("p_L1",)), ("L2_only", ("p_L2",)), ("L3_only", ("p_L3",)),
                          ("without_L1", ("p_L2", "p_L3")), ("without_L2", ("p_L1", "p_L3")),
                          ("without_L3", ("p_L1", "p_L2"))]:
        changed = aggregate_structural_nodes(pairs, proteins, levels)
        frame = build_profiles(expression, metadata, changed, variants=("full_tri_modal",))
        frame["variant"] = "readout_" + label
        frames.append(frame)
        print("[readout sensitivity] " + label, flush=True)
    baselines, selected = build_simple_baselines(expression, all_frame.to_numpy(dtype=float),
                                               all_frame.index.astype(str).tolist(), metadata)
    frames.append(baselines)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    estimates, differences = conditional_bootstrap(combined, iterations, int(args.seed or lock["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    tables = {"stage27_reproduction.tsv": reproduction,
              "participant_predictions_all_variants.tsv": combined,
              "participant_conditional_intervals.tsv": estimates,
              "paired_modality_differences.tsv": differences,
              "fold_local_feature_selection.tsv": selected,
              "pair_readout_rank_sensitivity.tsv": readout_rank_sensitivity(pairs),
              "stage_completion_inventory.tsv": publication_inventory(root)}
    for name, table in tables.items():
        table.to_csv(output / name, sep="\t", index=False)
    limitations = [
        "Original frozen250 were selected on full GSE75097 discovery; LOPO results are selection-conditioned exploratory associations.",
        "Fold-selected expression baseline avoids that candidate-membership leakage, but is not a fully nested validation of the multimodal model.",
        "Bootstrap intervals condition on existing overlapping LOPO fits; they are descriptive, not full-pipeline or confirmatory confidence intervals.",
        "Readout branch removal reuses frozen branch predictions; it is not a retrained L1/L2/L3 backbone ablation.",
        "Composite utility rewards structural and molecular support used in constructing the scores; utility is not independent validation of benefit.",
        "Original label-permutation P values condition on selected candidates; unrestricted exchangeability after demographic adjustment is not established.",
        "Raw ridge LOPO predictions can be anti-correlated through the changing training-target mean; report the intercept-only diagnostic and centered signal components alongside raw predictions, without sign reversal.",
        "PS is primary snoring, not a healthy control cohort. No diagnostic, prognostic or CPAP benefit claim is supported.",
        "All five axes, including hypertension and EDS negative results, must be retained without sign reversal or outcome-driven model selection.",
    ]
    summary = {"status": "PASS_SUPPLEMENT_SMOKE" if args.smoke else "PASS_DESCRIPTIVE_PUBLICATION_SUPPLEMENT",
               "created_at_utc": utc_timestamp(), "program_version": PROGRAM_VERSION,
               "participants": len(metadata), "frozen_primary_proteins": len(proteins),
               "all_mapped_expression_proteins": len(all_frame), "variants": combined["variant"].nunique(),
               "bootstrap": iterations, "seed": int(args.seed or lock["seed"]),
               "stage27_reproduction_max_error": float(reproduction["absolute_difference"].max()),
               "V4_retrained": False, "V4_ranking_changed": False,
               "external_endpoint_values_read": False, "stage_status_metadata_read": True,
               "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version,
               "limitations": limitations,
               "remaining_before_submission": [
                   "Integrate author-organized original-method residue/pair benchmarks and retrained ablations; do not relabel readout controls.",
                   "Use Stage27 as exploratory association only; new nested or external participant evaluation is required for a patient-prediction claim.",
                   "Update source-data figures, legends and manuscript with conditional uncertainty, all negative endpoints and version history.",
               ]}
    with (output / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("TRIAGE-OSA publication supplement: " + summary["status"], flush=True)
    print("Output: " + str(output), flush=True)
    return 0


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    lock_path = resolve(root, args.lock)
    if not lock_path.is_file():
        raise RuntimeError("Missing experiment lock: {}".format(lock_path))
    with open(str(lock_path), "r", encoding="utf-8") as handle:
        lock = json.load(handle)
    paths, pairs, proteins, expression, metadata = load_inputs(root, lock)
    expected_pairs = int(lock["expected_primary_pairs"])
    expected_participants = int(lock["expected_participants"])
    if len(pairs) != expected_pairs or len(metadata) != expected_participants:
        raise RuntimeError(
            "Frozen dimensions drift: pairs={}/{} participants={}/{}".format(
                len(pairs), expected_pairs, len(metadata), expected_participants
            )
        )
    print("TRIAGE-OSA public multimodal environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {} pandas: {}".format(np.__version__, pd.__version__))
    print("Frozen V4 pairs: {:,}; primary proteins: {}; expression-observed: {}".format(
        len(pairs), lock["expected_primary_proteins"], len(proteins)
    ))
    print("Participants: {} (PS/MSO/VSO={}/{}/{})".format(
        len(metadata),
        int((metadata["study_group"] == "PS").sum()),
        int((metadata["study_group"] == "MSO").sum()),
        int((metadata["study_group"] == "VSO").sum()),
    ))
    print("BMI available: 0; cohort-mean imputation: no")
    print("External validation read: no")
    if args.check_env and not args.publication_supplement:
        return 0
    if args.publication_supplement:
        return run_publication_supplement(root, lock, args, paths, pairs, proteins, expression, metadata)

    output_stage = args.output_stage or (lock["output_stage"] + ("_smoke" if args.smoke else ""))
    output_dir = root / "results" / "discovery_preprocess_v2" / output_stage
    if output_dir.exists() and not args.force:
        raise RuntimeError("Output directory exists; use --force: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    permutations = (
        20 if args.smoke else int(args.permutations or lock["default_permutations"])
    )
    seed = int(args.seed or lock["seed"])

    layer1 = pair_output(pairs)
    structural_strength = aggregate_structural_nodes(
        pairs.loc[
            pairs["protein_A"].astype(str).isin(set(proteins))
            & pairs["protein_B"].astype(str).isin(set(proteins))
        ].copy(),
        proteins,
    )
    profiles = build_profiles(expression, metadata, structural_strength)
    metrics = profile_metrics(profiles)
    permutation = permutation_test(
        expression, metadata, structural_strength, metrics, permutations, seed
    )

    layer1_path = output_dir / "output_layer_1_pair_mechanism.tsv.gz"
    layer2_path = output_dir / "output_layer_2_participant_mechanism_profiles.tsv"
    layer3_path = output_dir / "output_layer_3_phenotype_associations.tsv"
    ablation_path = output_dir / "modality_ablation_metrics.tsv"
    permutation_path = output_dir / "patient_information_permutation.tsv"
    layer1.to_csv(layer1_path, sep="\t", index=False, compression="gzip")
    profiles.loc[profiles["variant"] == "full_tri_modal"].to_csv(
        layer2_path, sep="\t", index=False
    )
    full_metrics = metrics.loc[metrics["variant"] == "full_tri_modal"].merge(
        permutation, on=["axis", "metric"], how="left", validate="one_to_one"
    )
    full_metrics.to_csv(layer3_path, sep="\t", index=False)
    metrics.to_csv(ablation_path, sep="\t", index=False)
    permutation.to_csv(permutation_path, sep="\t", index=False)

    summary = {
        "status": "PASS_SMOKE" if args.smoke else "PASS_FORMAL_PUBLIC_MULTIMODAL",
        "created_at_utc": utc_timestamp(),
        "program_version": PROGRAM_VERSION,
        "scope": lock["scope"],
        "frozen_v4_ranking_changed": False,
        "participants": len(metadata),
        "primary_pairs": len(pairs),
        "primary_proteins_expected": int(lock["expected_primary_proteins"]),
        "expression_observed_primary_proteins": len(proteins),
        "profile_axes": list(AXES),
        "profile_variants": list(VARIANTS),
        "permutations": permutations,
        "participant_isolation": lock["participant_isolation"],
        "demographic_adjustment": lock["demographic_adjustment"],
        "BMI_available": False,
        "cohort_mean_imputation": False,
        "external_validation_read": False,
        "clinical_probability_output": False,
        "outputs": {
            "layer_1_pair_mechanism": str(layer1_path),
            "layer_2_participant_mechanism_profiles": str(layer2_path),
            "layer_3_phenotype_associations": str(layer3_path),
            "modality_ablation": str(ablation_path),
            "patient_information_permutation": str(permutation_path),
        },
        "full_tri_modal_metrics": full_metrics.to_dict(orient="records"),
        "mean_multi_axis_utility_by_variant": metrics.groupby("variant")[
            "multi_axis_utility"
        ].mean().to_dict(),
        "interpretation": (
            "Participant-isolated molecular mechanism profiles and exploratory "
            "phenotype associations; not diagnosis, prognosis, or CPAP response prediction."
        ),
    }
    with open(str(output_dir / "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("TRIAGE-OSA public multimodal experiment: {}".format(summary["status"]))
    print("Output: {}".format(output_dir))
    print("Frozen V4 ranking changed: no")
    print("Clinical probability output: no")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
