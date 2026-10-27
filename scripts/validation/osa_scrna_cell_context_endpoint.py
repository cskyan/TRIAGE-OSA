#!/usr/bin/env python3
"""Evaluate the frozen GSE214865 pediatric cell-context pair endpoint.

Python 3.7 compatible.  One run evaluates all three locked primary lineages,
all frozen TRIAGE-Fuse/baseline rankings, the omnibus top-1-percent endpoint,
the 0.5-to-5-percent enrichment curve, matched-node randomization, and joint
participant-label permutation.

GSE214865 is held-out validation.  This script never changes candidate or pair
membership, model parameters, ranking weights, thresholds, or lineage labels.
It does not read previous CPAP/adipose validation, known PPI, pathway, or
complex labels.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.3.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
LOCK_REL = "config/osa_scrna_validation_lock.json"
HANDOFF_REL = "config/triage_osa_external_validation_lock_v4.json"
CANDIDATE_REL = (
    "results/discovery_preprocess_v2/04_candidates_v1/"
    "candidate_proteins_primary_250.tsv"
)
RANKING_REL = (
    "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/"
    "pair_context_scores_triage_osa_primary_250.tsv.gz"
)
RANKING_SHA256 = "1e87e8ecaaf6d4a70ad89086237bcdc84633e5029d7f30efb0a0ccaadf6b03d3"
PREPROCESS_STAGE = "15_scrna_validation_preprocess_v1"
DEFAULT_STAGE = "16_scrna_cell_context_endpoint_triage_osa_v4"
LINEAGES = ("T_NK", "B", "Myeloid")
FRACTIONS = (0.005, 0.01, 0.02, 0.05)
RANKING_FAMILY = (
    ("full_primary", "OSA_interaction_priority_rank"),
    ("PPI_only", "rank_p_PPI"),
    ("context_only", "rank_r_OSA"),
    ("L3_only", "rank_baseline_L3_only"),
    ("L1_L2", "rank_baseline_L1_L2"),
    ("L1_L3", "rank_baseline_L1_L3"),
    ("L2_L3", "rank_baseline_L2_L3"),
    ("equal_L1_L2_L3", "rank_baseline_equal_L1_L2_L3"),
    ("PPI25_context75", "rank_priority_PPI25_context75"),
    ("PPI75_context25", "rank_priority_PPI75_context25"),
    (
        "bidirectional_conservative",
        "rank_priority_bidirectional_conservative",
    ),
)


def parse_rank(value):
    """Parse integer-valued ranks, including pandas strings such as 857.0."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise RuntimeError("Invalid rank value: {!r}".format(value))
    if not math.isfinite(numeric) or numeric != int(numeric):
        raise RuntimeError("Rank is not a finite integer: {!r}".format(value))
    return int(numeric)


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def fmt(value):
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return "{:.10g}".format(value)


def bh_adjust(values):
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [1.0] * len(values)
    running = 1.0
    for reverse in range(len(values) - 1, -1, -1):
        index = order[reverse]
        rank = reverse + 1
        running = min(running, values[index] * len(values) / float(rank))
        result[index] = min(1.0, running)
    return result


def midrank_percentiles(np, values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        result[order[start:end]] = (average_rank - 0.5) / len(values)
        start = end
    return result


def trapezoid_auec(np, differences):
    fractions = np.asarray(FRACTIONS, dtype=np.float64)
    increments = fractions[1:] - fractions[:-1]
    areas = 0.5 * (differences[..., 1:] + differences[..., :-1]) * increments
    return np.sum(areas, axis=-1) / float(FRACTIONS[-1] - FRACTIONS[0])


def load_candidates(path, read_tsv):
    rows, _ = read_tsv(path)
    if len(rows) != 250:
        raise RuntimeError("Frozen primary candidate count is not 250")
    if len(set(row["UniProt_accession"] for row in rows)) != 250:
        raise RuntimeError("Frozen primary candidates are not unique")
    if [int(row["discovery_rank"]) for row in rows] != list(range(1, 251)):
        raise RuntimeError("Frozen candidate discovery ranks are invalid")
    return rows


def load_ranking(np, path):
    fields = [
        "pair_index", "pair_id", "protein_A", "protein_B", "gene_A", "gene_B",
        "triage_osa_rank_primary_250",
    ] + [rank_field for _, rank_field in RANKING_FAMILY]
    rows = []
    with gzip.open(str(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("Frozen ranking lacks fields: " + ",".join(missing))
        for source in reader:
            rows.append({field: source[field] for field in fields})
    if len(rows) != 31125:
        raise RuntimeError("Frozen primary pair count is not 31,125")
    if any(
        int(float(row["triage_osa_rank_primary_250"]))
        != parse_rank(row["OSA_interaction_priority_rank"])
        for row in rows
    ):
        raise RuntimeError("TRIAGE-OSA native and compatibility ranks differ")
    for _, rank_field in RANKING_FAMILY:
        ranks = sorted(parse_rank(row[rank_field]) for row in rows)
        if ranks != list(range(1, 31126)):
            raise RuntimeError("Invalid frozen rank permutation: " + rank_field)
    return rows


def load_candidate_effects(path, candidates):
    target = set(row["Ensembl_gene"].split(".")[0] for row in candidates)
    effects = {}
    with gzip.open(str(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["Ensembl_gene"] not in target:
                continue
            if row["lineage"] not in LINEAGES:
                continue
            if row["candidate_selection_allowed"] != "no":
                raise RuntimeError("GSE214865 effect row permits candidate selection")
            key = (row["lineage"], row["Ensembl_gene"])
            if key in effects:
                raise RuntimeError("Duplicate GSE214865 candidate effect row")
            effects[key] = row
    return effects


def verify_preprocess(root, preprocess_dir, lock_path, sha256_file):
    summary_path = preprocess_dir / "run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Missing Stage 15 run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Stage 15 status is not PASS")
    if summary.get("participants_processed") != 22:
        raise RuntimeError("Stage 15 participant count is not 22")
    if summary.get("frozen_candidate_or_pair_data_read") is not False:
        raise RuntimeError("Stage 15 isolation flag failed")
    if summary.get("validation_used_for_training_or_selection") is not False:
        raise RuntimeError("Stage 15 validation isolation flag failed")
    artifacts = summary.get("artifact_sha256", {})
    required_names = {
        "analysis_lock_snapshot.json",
        "scrna_pseudobulk_effects.tsv.gz",
        "pseudobulk_primary_logcpm.npz",
        "pseudobulk_primary_logcpm_metadata.tsv",
    }
    seen_names = set()
    for relative, expected in artifacts.items():
        path = root / relative
        seen_names.add(path.name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError("Stage 15 artifact verification failed: " + relative)
    missing = required_names - seen_names
    if missing:
        raise RuntimeError("Stage 15 summary lacks artifacts: " + ",".join(sorted(missing)))
    snapshot = preprocess_dir / "analysis_lock_snapshot.json"
    if sha256_file(snapshot) != sha256_file(lock_path):
        raise RuntimeError("Stage 15 used a different single-cell validation lock")
    return summary, summary_path


def build_node_evidence(np, candidates, effects):
    index = {row["UniProt_accession"]: i for i, row in enumerate(candidates)}
    scores = np.full((len(LINEAGES), len(candidates)), np.nan, dtype=np.float64)
    opposite_scores = np.full_like(scores, np.nan)
    available = np.zeros(scores.shape, dtype=bool)
    positive = np.zeros(scores.shape, dtype=bool)
    oriented_t = np.full_like(scores, np.nan)
    oriented_beta = np.full_like(scores, np.nan)
    detection = np.full_like(scores, np.nan)
    mapping_status = {}
    evidence_rows = []
    for lineage_index, lineage in enumerate(LINEAGES):
        accessions = []
        values = []
        opposite_values = []
        for candidate in candidates:
            accession = candidate["UniProt_accession"]
            candidate_index = index[accession]
            ensembl = candidate["Ensembl_gene"].split(".")[0]
            row = effects.get((lineage, ensembl))
            key = (lineage, accession)
            if row is None:
                mapping_status[key] = "feature_absent"
                continue
            detection[lineage_index, candidate_index] = float(
                row["participant_detection_fraction_CPM_ge_1"]
            )
            if row["tested"] != "yes":
                mapping_status[key] = "mapped_but_gene_filter_failed"
                continue
            sign = 1.0 if candidate["OSA_beta_sign"] == "positive" else -1.0
            beta = sign * float(row["beta_log2CPM"])
            statistic = sign * float(row["moderated_t"])
            available[lineage_index, candidate_index] = True
            positive[lineage_index, candidate_index] = beta > 0.0
            oriented_t[lineage_index, candidate_index] = statistic
            oriented_beta[lineage_index, candidate_index] = beta
            mapping_status[key] = "exact_Ensembl_tested"
            accessions.append(accession)
            values.append(statistic)
            opposite_values.append(-statistic)
        forward = midrank_percentiles(np, values)
        reverse = midrank_percentiles(np, opposite_values)
        for accession, value, opposite in zip(accessions, forward, reverse):
            candidate_index = index[accession]
            scores[lineage_index, candidate_index] = value
            opposite_scores[lineage_index, candidate_index] = opposite
    for candidate in candidates:
        accession = candidate["UniProt_accession"]
        candidate_index = index[accession]
        for lineage_index, lineage in enumerate(LINEAGES):
            ensembl = candidate["Ensembl_gene"].split(".")[0]
            row = effects.get((lineage, ensembl))
            evidence_rows.append({
                "discovery_rank": candidate["discovery_rank"],
                "UniProt_accession": accession,
                "gene_symbol": candidate["gene_symbol_current"],
                "Ensembl_gene": ensembl,
                "lineage": lineage,
                "mapping_status": mapping_status[(lineage, accession)],
                "available": "yes" if available[lineage_index, candidate_index] else "no",
                "participant_detection_fraction_CPM_ge_1": (
                    row["participant_detection_fraction_CPM_ge_1"]
                    if row is not None else ""
                ),
                "pediatric_beta_OSA_minus_control": (
                    row["beta_log2CPM"] if row is not None and row["tested"] == "yes" else ""
                ),
                "pediatric_moderated_t": (
                    row["moderated_t"] if row is not None and row["tested"] == "yes" else ""
                ),
                "pediatric_P_value": (
                    row["P_value"] if row is not None and row["tested"] == "yes" else ""
                ),
                "pediatric_FDR_BH": (
                    row["FDR_BH"] if row is not None and row["tested"] == "yes" else ""
                ),
                "adult_discovery_OSA_sign": candidate["OSA_beta_sign"],
                "oriented_beta": fmt(oriented_beta[lineage_index, candidate_index]),
                "oriented_moderated_t": fmt(oriented_t[lineage_index, candidate_index]),
                "node_score": fmt(scores[lineage_index, candidate_index]),
                "opposite_direction_node_score": fmt(
                    opposite_scores[lineage_index, candidate_index]
                ),
                "same_direction_positive": (
                    "yes" if positive[lineage_index, candidate_index]
                    else "no" if available[lineage_index, candidate_index] else ""
                ),
                "candidate_selection_allowed": "no",
            })
    return {
        "scores": scores,
        "opposite_scores": opposite_scores,
        "available": available,
        "positive": positive,
        "oriented_t": oriented_t,
        "oriented_beta": oriented_beta,
        "detection": detection,
        "rows": evidence_rows,
    }


def pair_metric(np, scores, positive, left, right, ranks):
    eligible = np.isfinite(scores[left]) & np.isfinite(scores[right])
    support = np.full(len(left), np.nan, dtype=np.float64)
    support[eligible] = np.sqrt(scores[left[eligible]] * scores[right[eligible]])
    both_positive = positive[left] & positive[right] & eligible
    background = float(np.mean(support[eligible]))
    background_binary = float(np.mean(both_positive[eligible]))
    rows = []
    differences = []
    ratios = []
    for fraction in FRACTIONS:
        threshold = int(math.ceil(len(left) * fraction))
        selected = eligible & (ranks <= threshold)
        if int(np.sum(selected)) == 0:
            raise RuntimeError("No eligible top pairs for a locked fraction")
        top = float(np.mean(support[selected]))
        top_binary = float(np.mean(both_positive[selected]))
        difference = top - background
        ratio = top / max(background, 1e-12)
        differences.append(difference)
        ratios.append(ratio)
        rows.append({
            "fraction": fmt(fraction),
            "frozen_rank_threshold": str(threshold),
            "eligible_top_pairs": str(int(np.sum(selected))),
            "eligible_background_pairs": str(int(np.sum(eligible))),
            "top_mean_support": fmt(top),
            "background_mean_support": fmt(background),
            "support_enrichment_ratio": fmt(ratio),
            "support_enrichment_difference": fmt(difference),
            "top_both_nodes_positive_fraction": fmt(top_binary),
            "background_both_nodes_positive_fraction": fmt(background_binary),
            "binary_positive_ratio": fmt(
                top_binary / max(background_binary, 1e-12)
            ),
        })
    return {
        "rows": rows,
        "AUEC": float(trapezoid_auec(np, np.asarray(differences))),
        "top1_ratio": ratios[1],
        "top1_difference": differences[1],
        "eligible": eligible,
        "support": support,
        "both_positive": both_positive,
        "differences": differences,
        "ratios": ratios,
    }


def observed_endpoints(np, pair_rows, left, right, evidence):
    endpoint_rows = []
    model_rows = []
    pair_metrics = {}
    ranks_by_model = {
        model: np.asarray([parse_rank(row[field]) for row in pair_rows], dtype=np.int64)
        for model, field in RANKING_FAMILY
    }
    model_lineage = {}
    for model, rank_field in RANKING_FAMILY:
        components = []
        for lineage_index, lineage in enumerate(LINEAGES):
            result = pair_metric(
                np,
                evidence["scores"][lineage_index],
                evidence["positive"][lineage_index],
                left, right, ranks_by_model[model],
            )
            model_lineage[(model, lineage)] = result
            if model == "full_primary":
                pair_metrics[lineage] = result
            for row in result["rows"]:
                output = dict(row)
                output.update({
                    "endpoint": "pediatric_OSA_same_direction",
                    "scope": "lineage",
                    "lineage": lineage,
                    "model": model,
                    "rank_field": rank_field,
                    "metric_scope": "fraction",
                    "normalized_AUEC_0p5_to_5pct": "",
                    "component_lineages": "1",
                })
                endpoint_rows.append(output)
            endpoint_rows.append({
                "endpoint": "pediatric_OSA_same_direction",
                "scope": "lineage",
                "lineage": lineage,
                "model": model,
                "rank_field": rank_field,
                "metric_scope": "AUEC",
                "fraction": "0.005_to_0.05",
                "normalized_AUEC_0p5_to_5pct": fmt(result["AUEC"]),
                "component_lineages": "1",
            })
            components.append(result)
        mean_differences = np.mean(np.asarray([
            item["differences"] for item in components
        ], dtype=np.float64), axis=0)
        mean_ratios = np.mean(np.asarray([
            item["ratios"] for item in components
        ], dtype=np.float64), axis=0)
        for fraction_index, fraction in enumerate(FRACTIONS):
            endpoint_rows.append({
                "endpoint": "pediatric_OSA_same_direction",
                "scope": "omnibus_mean_three_lineages",
                "lineage": "T_NK+B+Myeloid",
                "model": model,
                "rank_field": rank_field,
                "metric_scope": "fraction",
                "fraction": fmt(fraction),
                "frozen_rank_threshold": str(int(math.ceil(len(left) * fraction))),
                "support_enrichment_ratio": fmt(mean_ratios[fraction_index]),
                "support_enrichment_difference": fmt(
                    mean_differences[fraction_index]
                ),
                "normalized_AUEC_0p5_to_5pct": "",
                "component_lineages": "3",
            })
        omnibus_auec = float(trapezoid_auec(np, mean_differences))
        endpoint_rows.append({
            "endpoint": "pediatric_OSA_same_direction",
            "scope": "omnibus_mean_three_lineages",
            "lineage": "T_NK+B+Myeloid",
            "model": model,
            "rank_field": rank_field,
            "metric_scope": "AUEC",
            "fraction": "0.005_to_0.05",
            "normalized_AUEC_0p5_to_5pct": fmt(omnibus_auec),
            "component_lineages": "3",
        })
        model_rows.append({
            "model": model,
            "rank_field": rank_field,
            "omnibus_top_1pct_mean_ratio": fmt(mean_ratios[1]),
            "omnibus_top_1pct_mean_difference": fmt(mean_differences[1]),
            "omnibus_AUEC_0p5_to_5pct": fmt(omnibus_auec),
        })
    full_difference = float(model_rows[0]["omnibus_top_1pct_mean_difference"])
    full_auec = float(model_rows[0]["omnibus_AUEC_0p5_to_5pct"])
    for row in model_rows:
        row["full_minus_model_top_1pct_difference"] = fmt(
            full_difference - float(row["omnibus_top_1pct_mean_difference"])
        )
        row["full_minus_model_AUEC"] = fmt(
            full_auec - float(row["omnibus_AUEC_0p5_to_5pct"])
        )
        row["comparison_is_descriptive"] = "yes"
    return endpoint_rows, model_rows, pair_metrics, ranks_by_model["full_primary"]


def matched_blocks(np, candidates, detection, block_size=10):
    mean_detection = np.empty(len(candidates), dtype=np.float64)
    for index in range(len(candidates)):
        values = detection[:, index]
        values = values[np.isfinite(values)]
        mean_detection[index] = float(np.mean(values)) if values.size else -0.01
    length = np.asarray([
        float(row["log2_UniProt_length"]) for row in candidates
    ], dtype=np.float64)
    values = np.stack([mean_detection, length], axis=1)
    standardized = (
        values - np.mean(values, axis=0)
    ) / np.maximum(np.std(values, axis=0, ddof=1), 1e-12)
    unassigned = set(range(len(candidates)))
    blocks = []
    while unassigned:
        anchor = min(unassigned)
        ordered = sorted(
            unassigned,
            key=lambda index: (
                float(np.sum((standardized[index] - standardized[anchor]) ** 2)),
                candidates[index]["UniProt_accession"],
            ),
        )
        block = ordered[:block_size]
        blocks.append(np.asarray(block, dtype=np.int64))
        unassigned.difference_update(block)
    rows = []
    for block_index, block in enumerate(blocks, 1):
        for index in block:
            rows.append({
                "block_id": str(block_index),
                "block_size": str(len(block)),
                "UniProt_accession": candidates[index]["UniProt_accession"],
                "gene_symbol": candidates[index]["gene_symbol_current"],
                "pediatric_detection_mean": fmt(mean_detection[index]),
                "log2_UniProt_length": candidates[index]["log2_UniProt_length"],
            })
    return blocks, rows


def draw_statistics(np, scores, left, right, ranks):
    left_score = scores[:, left]
    right_score = scores[:, right]
    valid = np.isfinite(left_score) & np.isfinite(right_score)
    support = np.zeros(left_score.shape, dtype=np.float64)
    support[valid] = np.sqrt(left_score[valid] * right_score[valid])
    background_count = np.sum(valid, axis=1)
    if np.any(background_count == 0):
        raise RuntimeError("A permutation draw has no eligible pairs")
    background = np.sum(support, axis=1) / background_count
    differences = []
    ratios = []
    for fraction in FRACTIONS:
        threshold = int(math.ceil(len(left) * fraction))
        rank_mask = ranks <= threshold
        selected = valid & rank_mask.reshape(1, -1)
        selected_count = np.sum(selected, axis=1)
        if np.any(selected_count == 0):
            raise RuntimeError("A permutation draw has no eligible top pairs")
        top = np.sum(support * rank_mask.reshape(1, -1), axis=1) / selected_count
        differences.append(top - background)
        ratios.append(top / np.maximum(background, 1e-12))
    differences = np.stack(differences, axis=1)
    ratios = np.stack(ratios, axis=1)
    return differences, ratios, trapezoid_auec(np, differences)


def model_top1_differences(np, scores, left, right, top1_masks):
    left_score = scores[:, left]
    right_score = scores[:, right]
    valid = np.isfinite(left_score) & np.isfinite(right_score)
    support = np.zeros(left_score.shape, dtype=np.float64)
    support[valid] = np.sqrt(left_score[valid] * right_score[valid])
    background = np.sum(support, axis=1) / np.maximum(
        np.sum(valid, axis=1), 1
    )
    mask_float = top1_masks.astype(np.float64, copy=False)
    top_sum = np.dot(support, mask_float)
    top_count = np.dot(valid.astype(np.float64), mask_float)
    if np.any(top_count == 0.0):
        raise RuntimeError("A permutation draw has no eligible model top-1-percent pairs")
    return top_sum / top_count - background.reshape(-1, 1)


def node_aware_randomization(
    np, rng, iterations, batch_size, base_scores, blocks, left, right, ranks,
    model_top1_masks,
):
    null_top1 = np.empty((iterations, len(LINEAGES)), dtype=np.float64)
    null_auec = np.empty_like(null_top1)
    null_model_omnibus = np.zeros(
        (iterations, model_top1_masks.shape[1]), dtype=np.float64
    )
    completed = 0
    while completed < iterations:
        size = min(batch_size, iterations - completed)
        permuted = np.repeat(base_scores.reshape(1, len(LINEAGES), -1), size, axis=0)
        for draw in range(size):
            for block in blocks:
                source = rng.permutation(block)
                permuted[draw][:, block] = base_scores[:, source]
        for lineage_index in range(len(LINEAGES)):
            differences, _, auec = draw_statistics(
                np, permuted[:, lineage_index, :], left, right, ranks
            )
            null_top1[completed:completed + size, lineage_index] = differences[:, 1]
            null_auec[completed:completed + size, lineage_index] = auec
            null_model_omnibus[completed:completed + size, :] += (
                model_top1_differences(
                    np, permuted[:, lineage_index, :], left, right,
                    model_top1_masks,
                ) / float(len(LINEAGES))
            )
        completed += size
        if completed % 1000 == 0 or completed == iterations:
            print("[node-aware] {}/{}".format(completed, iterations), flush=True)
    return {
        "top1_difference": null_top1,
        "AUEC": null_auec,
        "omnibus_top1_difference": np.mean(null_top1, axis=1),
        "omnibus_AUEC": np.mean(null_auec, axis=1),
        "model_omnibus_top1_difference": null_model_omnibus,
    }


def load_participant_expression(np, path, candidates, evidence):
    data = np.load(str(path), allow_pickle=False)
    matrix = data["log2_CPM_plus_0p5"]
    feature_ids = data["Ensembl_gene"].tolist()
    if matrix.shape[0] != len(feature_ids):
        raise RuntimeError("Stage 15 primary logCPM feature mismatch")
    feature_index = {value: index for index, value in enumerate(feature_ids)}
    subject_ids = [str(value) for value in data["subject_id"].tolist()]
    disease = [str(value) for value in data["disease_group"].tolist()]
    lineages = [str(value) for value in data["lineage"].tolist()]
    subject_group = {}
    for subject, group in zip(subject_ids, disease):
        if subject in subject_group and subject_group[subject] != group:
            raise RuntimeError("Inconsistent GSE214865 participant group")
        subject_group[subject] = group
    subjects = sorted(subject_group)
    if len(subjects) != 22:
        raise RuntimeError("Primary logCPM does not contain 22 participants")
    observed_labels = np.asarray([
        1.0 if subject_group[subject] == "OSA" else 0.0 for subject in subjects
    ], dtype=np.float64)
    if int(np.sum(observed_labels)) != 11:
        raise RuntimeError("Primary logCPM group count is not 11 versus 11")
    expressions = []
    for lineage_index, lineage in enumerate(LINEAGES):
        column_map = {
            subject_ids[index]: index for index in range(len(subject_ids))
            if lineages[index] == lineage
        }
        if set(column_map) != set(subjects):
            raise RuntimeError("Lineage participant coverage mismatch: " + lineage)
        columns = [column_map[subject] for subject in subjects]
        expression = np.full(
            (len(candidates), len(subjects)), np.nan, dtype=np.float64
        )
        for candidate_index, candidate in enumerate(candidates):
            if not evidence["available"][lineage_index, candidate_index]:
                continue
            ensembl = candidate["Ensembl_gene"].split(".")[0]
            if ensembl not in feature_index:
                raise RuntimeError("Available candidate lacks logCPM feature")
            expression[candidate_index, :] = matrix[feature_index[ensembl], columns]
        expressions.append(expression)
    signs = np.asarray([
        1.0 if row["OSA_beta_sign"] == "positive" else -1.0
        for row in candidates
    ], dtype=np.float64)
    return expressions, subjects, observed_labels, signs


def participant_score_draws(np, expression, available, signs, labels):
    indices = np.where(available)[0]
    y = expression[indices, :]
    if not np.isfinite(y).all():
        raise RuntimeError("Available candidate participant expression is non-finite")
    n_osa = np.sum(labels, axis=1)
    n_control = labels.shape[1] - n_osa
    if np.any(n_osa != 11.0) or np.any(n_control != 11.0):
        raise RuntimeError("Participant permutation does not preserve 11 versus 11")
    mean_osa = np.dot(labels, y.T) / n_osa.reshape(-1, 1)
    mean_control = np.dot(1.0 - labels, y.T) / n_control.reshape(-1, 1)
    squares = y * y
    ss_osa = np.dot(labels, squares.T) - n_osa.reshape(-1, 1) * mean_osa * mean_osa
    ss_control = (
        np.dot(1.0 - labels, squares.T)
        - n_control.reshape(-1, 1) * mean_control * mean_control
    )
    pooled = np.maximum((ss_osa + ss_control) / 20.0, 1e-15)
    se = np.sqrt(pooled * (1.0 / n_osa + 1.0 / n_control).reshape(-1, 1))
    oriented_t = (mean_osa - mean_control) / se
    oriented_t = oriented_t * signs[indices].reshape(1, -1)
    output = np.full((labels.shape[0], expression.shape[0]), np.nan, dtype=np.float64)
    for draw in range(labels.shape[0]):
        output[draw, indices] = midrank_percentiles(np, oriented_t[draw])
    return output


def participant_label_randomization(
    np, rng, iterations, batch_size, expressions, available, signs,
    observed_labels, left, right, ranks,
):
    observed_top1 = np.empty(len(LINEAGES), dtype=np.float64)
    observed_auec = np.empty(len(LINEAGES), dtype=np.float64)
    observed_scores = []
    labels = observed_labels.reshape(1, -1)
    for lineage_index in range(len(LINEAGES)):
        scores = participant_score_draws(
            np, expressions[lineage_index], available[lineage_index], signs, labels
        )
        differences, _, auec = draw_statistics(np, scores, left, right, ranks)
        observed_top1[lineage_index] = differences[0, 1]
        observed_auec[lineage_index] = auec[0]
        observed_scores.append(scores[0])
    null_top1 = np.empty((iterations, len(LINEAGES)), dtype=np.float64)
    null_auec = np.empty_like(null_top1)
    completed = 0
    participant_count = len(observed_labels)
    while completed < iterations:
        size = min(batch_size, iterations - completed)
        random_order = np.argsort(
            rng.random_sample((size, participant_count)), axis=1
        )
        permuted_labels = np.zeros((size, participant_count), dtype=np.float64)
        rows = np.arange(size).reshape(-1, 1)
        permuted_labels[rows, random_order[:, :11]] = 1.0
        for lineage_index in range(len(LINEAGES)):
            scores = participant_score_draws(
                np, expressions[lineage_index], available[lineage_index],
                signs, permuted_labels,
            )
            differences, _, auec = draw_statistics(
                np, scores, left, right, ranks
            )
            null_top1[completed:completed + size, lineage_index] = differences[:, 1]
            null_auec[completed:completed + size, lineage_index] = auec
        completed += size
        if completed % 1000 == 0 or completed == iterations:
            print("[participant-label] {}/{}".format(completed, iterations), flush=True)
    return {
        "observed_scores": np.stack(observed_scores, axis=0),
        "observed_top1_difference": observed_top1,
        "observed_AUEC": observed_auec,
        "observed_omnibus_top1_difference": float(np.mean(observed_top1)),
        "observed_omnibus_AUEC": float(np.mean(observed_auec)),
        "null_top1_difference": null_top1,
        "null_AUEC": null_auec,
        "null_omnibus_top1_difference": np.mean(null_top1, axis=1),
        "null_omnibus_AUEC": np.mean(null_auec, axis=1),
    }


def randomization_summary(np, scheme, scope, lineage, statistic, observed, null):
    null = np.asarray(null, dtype=np.float64)
    return {
        "randomization": scheme,
        "scope": scope,
        "lineage": lineage,
        "statistic": statistic,
        "observed": fmt(observed),
        "null_mean": fmt(np.mean(null)),
        "null_SD": fmt(np.std(null, ddof=1)),
        "null_q025": fmt(np.quantile(null, 0.025)),
        "null_q50": fmt(np.quantile(null, 0.50)),
        "null_q975": fmt(np.quantile(null, 0.975)),
        "one_sided_P": fmt(
            (1.0 + float(np.sum(null >= observed))) / (len(null) + 1.0)
        ),
        "iterations": str(len(null)),
        "BH_FDR_within_lineage_family": "",
        "claim_role": "primary" if scope == "omnibus_mean_three_lineages" and statistic == "top_1pct_support_difference" and scheme == "node_aware_joint_three_lineage_vector" else "secondary",
    }


def build_randomization_rows(np, node, participant, observed_lineage):
    rows = []
    node_omnibus_observed = float(np.mean([
        observed_lineage[lineage]["top1_difference"] for lineage in LINEAGES
    ]))
    node_omnibus_auec = float(np.mean([
        observed_lineage[lineage]["AUEC"] for lineage in LINEAGES
    ]))
    rows.extend([
        randomization_summary(
            np, "node_aware_joint_three_lineage_vector",
            "omnibus_mean_three_lineages", "T_NK+B+Myeloid",
            "top_1pct_support_difference", node_omnibus_observed,
            node["omnibus_top1_difference"],
        ),
        randomization_summary(
            np, "node_aware_joint_three_lineage_vector",
            "omnibus_mean_three_lineages", "T_NK+B+Myeloid",
            "AUEC_0p5_to_5pct", node_omnibus_auec,
            node["omnibus_AUEC"],
        ),
    ])
    node_lineage_top_rows = []
    participant_lineage_top_rows = []
    for lineage_index, lineage in enumerate(LINEAGES):
        node_lineage_top_rows.append(len(rows))
        rows.append(randomization_summary(
            np, "node_aware_joint_three_lineage_vector", "lineage", lineage,
            "top_1pct_support_difference",
            observed_lineage[lineage]["top1_difference"],
            node["top1_difference"][:, lineage_index],
        ))
        rows.append(randomization_summary(
            np, "node_aware_joint_three_lineage_vector", "lineage", lineage,
            "AUEC_0p5_to_5pct", observed_lineage[lineage]["AUEC"],
            node["AUEC"][:, lineage_index],
        ))
    rows.extend([
        randomization_summary(
            np, "participant_label_joint_across_lineages",
            "omnibus_mean_three_lineages", "T_NK+B+Myeloid",
            "top_1pct_support_difference_unmoderated_t",
            participant["observed_omnibus_top1_difference"],
            participant["null_omnibus_top1_difference"],
        ),
        randomization_summary(
            np, "participant_label_joint_across_lineages",
            "omnibus_mean_three_lineages", "T_NK+B+Myeloid",
            "AUEC_0p5_to_5pct_unmoderated_t",
            participant["observed_omnibus_AUEC"],
            participant["null_omnibus_AUEC"],
        ),
    ])
    for lineage_index, lineage in enumerate(LINEAGES):
        participant_lineage_top_rows.append(len(rows))
        rows.append(randomization_summary(
            np, "participant_label_joint_across_lineages", "lineage", lineage,
            "top_1pct_support_difference_unmoderated_t",
            participant["observed_top1_difference"][lineage_index],
            participant["null_top1_difference"][:, lineage_index],
        ))
        rows.append(randomization_summary(
            np, "participant_label_joint_across_lineages", "lineage", lineage,
            "AUEC_0p5_to_5pct_unmoderated_t",
            participant["observed_AUEC"][lineage_index],
            participant["null_AUEC"][:, lineage_index],
        ))
    for indices in (node_lineage_top_rows, participant_lineage_top_rows):
        adjusted = bh_adjust([float(rows[index]["one_sided_P"]) for index in indices])
        for index, fdr in zip(indices, adjusted):
            rows[index]["BH_FDR_within_lineage_family"] = fmt(fdr)
    return rows


def model_comparison_randomization(np, model_rows, node_null):
    observed = np.asarray([
        float(row["omnibus_top_1pct_mean_difference"])
        for row in model_rows
    ], dtype=np.float64)
    null = node_null["model_omnibus_top1_difference"]
    if null.shape[1] != len(model_rows):
        raise RuntimeError("Model-comparison null dimension mismatch")
    rows = []
    p_values = []
    for model_index in range(1, len(model_rows)):
        observed_delta = observed[0] - observed[model_index]
        null_delta = null[:, 0] - null[:, model_index]
        p_value = (
            1.0 + float(np.sum(null_delta >= observed_delta))
        ) / (len(null_delta) + 1.0)
        p_values.append(p_value)
        rows.append({
            "full_model": model_rows[0]["model"],
            "comparator_model": model_rows[model_index]["model"],
            "statistic": "full_minus_comparator_omnibus_top_1pct_support_difference",
            "observed_full": fmt(observed[0]),
            "observed_comparator": fmt(observed[model_index]),
            "observed_delta": fmt(observed_delta),
            "null_delta_mean": fmt(np.mean(null_delta)),
            "null_delta_SD": fmt(np.std(null_delta, ddof=1)),
            "null_delta_q025": fmt(np.quantile(null_delta, 0.025)),
            "null_delta_q50": fmt(np.quantile(null_delta, 0.50)),
            "null_delta_q975": fmt(np.quantile(null_delta, 0.975)),
            "one_sided_P_full_greater": fmt(p_value),
            "BH_FDR_across_10_comparisons": "",
            "full_superiority_claim_allowed": "no",
            "claim_role": "secondary_model_comparison",
        })
    adjusted = bh_adjust(p_values)
    for row, fdr in zip(rows, adjusted):
        row["BH_FDR_across_10_comparisons"] = fmt(fdr)
        row["full_superiority_claim_allowed"] = (
            "yes" if float(row["observed_delta"]) > 0.0 and fdr < 0.05
            else "no"
        )
    return rows


def negative_controls(np, pair_rows, evidence, left, right, primary_ranks):
    rows = []
    n_pairs = len(pair_rows)
    top_mask = primary_ranks <= int(math.ceil(n_pairs * 0.01))
    bottom_mask = primary_ranks > n_pairs - int(math.ceil(n_pairs * 0.01))
    per_lineage = []
    for lineage_index, lineage in enumerate(LINEAGES):
        primary = pair_metric(
            np, evidence["scores"][lineage_index],
            evidence["positive"][lineage_index], left, right, primary_ranks,
        )
        opposite = pair_metric(
            np, evidence["opposite_scores"][lineage_index],
            ~evidence["positive"][lineage_index], left, right, primary_ranks,
        )
        eligible = primary["eligible"]
        background = float(np.mean(primary["support"][eligible]))
        bottom_ratio = float(
            np.mean(primary["support"][eligible & bottom_mask]) / max(background, 1e-12)
        )
        opposite_background = float(np.mean(opposite["support"][opposite["eligible"]]))
        opposite_ratio = float(
            np.mean(opposite["support"][opposite["eligible"] & top_mask])
            / max(opposite_background, 1e-12)
        )
        rows.extend([
            {
                "scope": "lineage", "lineage": lineage,
                "negative_control": "bottom_1pct_same_direction_support",
                "support_ratio": fmt(bottom_ratio),
                "interpretation": "descriptive_negative_control",
            },
            {
                "scope": "lineage", "lineage": lineage,
                "negative_control": "top_1pct_opposite_direction_support",
                "support_ratio": fmt(opposite_ratio),
                "interpretation": "descriptive_negative_control",
            },
        ])
        per_lineage.append((bottom_ratio, opposite_ratio))
    rows.extend([
        {
            "scope": "omnibus_mean_three_lineages",
            "lineage": "T_NK+B+Myeloid",
            "negative_control": "bottom_1pct_same_direction_support",
            "support_ratio": fmt(np.mean([value[0] for value in per_lineage])),
            "interpretation": "descriptive_negative_control",
        },
        {
            "scope": "omnibus_mean_three_lineages",
            "lineage": "T_NK+B+Myeloid",
            "negative_control": "top_1pct_opposite_direction_support",
            "support_ratio": fmt(np.mean([value[1] for value in per_lineage])),
            "interpretation": "descriptive_negative_control",
        },
    ])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preprocess-stage", default=PREPROCESS_STAGE)
    parser.add_argument("--output-stage", default=DEFAULT_STAGE)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.permutations <= 0 or args.batch_size <= 0:
        raise RuntimeError("permutations and batch-size must be positive")
    root = args.root.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from osa_cpap_validation_preprocess import (
        atomic_json, read_tsv, relative_path, sha256_file,
        write_gzip_tsv, write_tsv,
    )
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc

    lock_path = root / LOCK_REL
    handoff_path = root / HANDOFF_REL
    candidate_path = root / CANDIDATE_REL
    ranking_path = root / RANKING_REL
    preprocess_dir = (
        root / "results" / "discovery_preprocess_v2" / args.preprocess_stage
    )
    effect_path = preprocess_dir / "scrna_pseudobulk_effects.tsv.gz"
    expression_path = preprocess_dir / "pseudobulk_primary_logcpm.npz"
    required = [
        lock_path, handoff_path, candidate_path, ranking_path, effect_path, expression_path,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff["ranking"]["primary_250"] != RANKING_REL:
        raise RuntimeError("TRIAGE-OSA handoff ranking path mismatch")
    if handoff["ranking"]["primary_250_sha256"] != RANKING_SHA256:
        raise RuntimeError("TRIAGE-OSA handoff ranking hash mismatch")
    if handoff["prespecified_endpoint_locks"]["scrna"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("TRIAGE-OSA handoff scRNA lock mismatch")
    if sha256_file(candidate_path) != lock["input_hashes"]["candidate_proteins_primary_250.tsv"]:
        raise RuntimeError("Frozen candidate hash mismatch")
    if sha256_file(ranking_path) != RANKING_SHA256:
        raise RuntimeError("Frozen ranking hash mismatch")
    preprocess_summary, preprocess_summary_path = verify_preprocess(
        root, preprocess_dir, lock_path, sha256_file
    )
    candidates = load_candidates(candidate_path, read_tsv)
    pair_rows = load_ranking(np, ranking_path)
    candidate_index = {
        row["UniProt_accession"]: index for index, row in enumerate(candidates)
    }
    if any(
        row["protein_A"] not in candidate_index
        or row["protein_B"] not in candidate_index
        for row in pair_rows
    ):
        raise RuntimeError("Frozen pair contains a protein outside primary 250")
    left = np.asarray([
        candidate_index[row["protein_A"]] for row in pair_rows
    ], dtype=np.int64)
    right = np.asarray([
        candidate_index[row["protein_B"]] for row in pair_rows
    ], dtype=np.int64)
    effects = load_candidate_effects(effect_path, candidates)
    evidence = build_node_evidence(np, candidates, effects)
    expected_coverage = [206, 215, 213]
    actual_coverage = [
        int(np.sum(evidence["available"][index]))
        for index in range(len(LINEAGES))
    ]
    if actual_coverage != expected_coverage:
        raise RuntimeError(
            "Stage 15 candidate coverage changed: {}".format(actual_coverage)
        )
    print("OSA GSE214865 cell-context endpoint environment: PASS")
    print("Python: " + sys.version.replace("\n", " "))
    print("numpy: " + np.__version__)
    print("Frozen proteins=250 pairs=31,125 ranking_models={}".format(
        len(RANKING_FAMILY)
    ))
    print("Lineage coverage T_NK=206 B=215 Myeloid=213")
    print("Previous CPAP/adipose validation, known PPI, pathway read: no")
    if args.check_env:
        return 0

    formal_iterations = int(lock["randomization"]["formal_iterations"])
    formal = args.permutations == formal_iterations
    output_base = root / "results" / "discovery_preprocess_v2" / args.output_stage
    output_dir = (
        output_base if formal
        else output_base / "development_{}".format(args.permutations)
    )
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace")
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    endpoint_rows, model_rows, observed_lineage, primary_ranks = observed_endpoints(
        np, pair_rows, left, right, evidence
    )
    model_top1_masks = np.stack([
        np.asarray([parse_rank(row[rank_field]) for row in pair_rows], dtype=np.int64)
        <= int(math.ceil(len(pair_rows) * 0.01))
        for _, rank_field in RANKING_FAMILY
    ], axis=1)
    blocks, block_rows = matched_blocks(
        np, candidates, evidence["detection"], block_size=10
    )
    seed = int(lock["randomization"]["seed"])
    node_null = node_aware_randomization(
        np, np.random.RandomState(seed + 400), args.permutations,
        args.batch_size, evidence["scores"], blocks, left, right,
        primary_ranks, model_top1_masks,
    )
    expressions, subjects, observed_labels, signs = load_participant_expression(
        np, expression_path, candidates, evidence
    )
    participant_null = participant_label_randomization(
        np, np.random.RandomState(seed + 500), args.permutations,
        args.batch_size, expressions, evidence["available"], signs,
        observed_labels, left, right, primary_ranks,
    )
    random_rows = build_randomization_rows(
        np, node_null, participant_null, observed_lineage
    )
    model_random_rows = model_comparison_randomization(
        np, model_rows, node_null
    )
    control_rows = negative_controls(
        np, pair_rows, evidence, left, right, primary_ranks
    )

    pair_output = []
    for pair_index, pair in enumerate(pair_rows):
        available_count = sum(
            int(observed_lineage[lineage]["eligible"][pair_index])
            for lineage in LINEAGES
        )
        support_values = [
            observed_lineage[lineage]["support"][pair_index]
            for lineage in LINEAGES
            if observed_lineage[lineage]["eligible"][pair_index]
        ]
        row = {
            "pair_id": pair["pair_id"],
            "protein_A": pair["protein_A"],
            "protein_B": pair["protein_B"],
            "gene_A": pair["gene_A"],
            "gene_B": pair["gene_B"],
            "OSA_interaction_priority_rank": pair["OSA_interaction_priority_rank"],
            "available_lineage_count": str(available_count),
            "mean_available_lineage_support": fmt(
                np.mean(support_values) if support_values else None
            ),
            "common_three_lineage_eligible": "yes" if available_count == 3 else "no",
        }
        for lineage in LINEAGES:
            result = observed_lineage[lineage]
            eligible = bool(result["eligible"][pair_index])
            row["{}_eligible".format(lineage)] = "yes" if eligible else "no"
            row["{}_pair_support".format(lineage)] = fmt(
                result["support"][pair_index] if eligible else None
            )
            row["{}_both_nodes_positive".format(lineage)] = (
                "yes" if result["both_positive"][pair_index]
                else "no" if eligible else ""
            )
        pair_output.append(row)

    paths = {
        "candidate": output_dir / "candidate_scrna_node_support.tsv",
        "pair": output_dir / "pair_scrna_support.tsv.gz",
        "endpoint": output_dir / "ranking_endpoint_metrics.tsv",
        "model": output_dir / "model_comparison.tsv",
        "model_random": output_dir / "model_comparison_permutation.tsv",
        "random": output_dir / "randomization_summary.tsv",
        "blocks": output_dir / "node_matching_blocks.tsv",
        "controls": output_dir / "negative_controls.tsv",
        "nulls": output_dir / "permutation_nulls.npz",
    }
    write_tsv(paths["candidate"], evidence["rows"], list(evidence["rows"][0]))
    write_gzip_tsv(paths["pair"], pair_output, list(pair_output[0]))
    endpoint_fields = [
        "endpoint", "scope", "lineage", "model", "rank_field",
        "metric_scope", "fraction", "frozen_rank_threshold",
        "eligible_top_pairs", "eligible_background_pairs",
        "top_mean_support", "background_mean_support",
        "support_enrichment_ratio", "support_enrichment_difference",
        "top_both_nodes_positive_fraction",
        "background_both_nodes_positive_fraction", "binary_positive_ratio",
        "normalized_AUEC_0p5_to_5pct", "component_lineages",
    ]
    write_tsv(paths["endpoint"], endpoint_rows, endpoint_fields)
    write_tsv(paths["model"], model_rows, list(model_rows[0]))
    write_tsv(
        paths["model_random"], model_random_rows, list(model_random_rows[0])
    )
    write_tsv(paths["random"], random_rows, list(random_rows[0]))
    write_tsv(paths["blocks"], block_rows, list(block_rows[0]))
    write_tsv(paths["controls"], control_rows, list(control_rows[0]))
    null_temp = paths["nulls"].with_name(paths["nulls"].name + ".tmp")
    with null_temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            lineage=np.asarray(LINEAGES),
            node_top1_difference=node_null["top1_difference"],
            node_AUEC=node_null["AUEC"],
            node_omnibus_top1_difference=node_null["omnibus_top1_difference"],
            node_omnibus_AUEC=node_null["omnibus_AUEC"],
            node_model_omnibus_top1_difference=node_null["model_omnibus_top1_difference"],
            participant_top1_difference=participant_null["null_top1_difference"],
            participant_AUEC=participant_null["null_AUEC"],
            participant_omnibus_top1_difference=participant_null["null_omnibus_top1_difference"],
            participant_omnibus_AUEC=participant_null["null_omnibus_AUEC"],
            participant_observed_top1_difference=participant_null["observed_top1_difference"],
            participant_observed_AUEC=participant_null["observed_AUEC"],
        )
    os.replace(str(null_temp), str(paths["nulls"]))

    primary_row = next(
        row for row in random_rows
        if row["randomization"] == "node_aware_joint_three_lineage_vector"
        and row["scope"] == "omnibus_mean_three_lineages"
        and row["statistic"] == "top_1pct_support_difference"
    )
    participant_row = next(
        row for row in random_rows
        if row["randomization"] == "participant_label_joint_across_lineages"
        and row["scope"] == "omnibus_mean_three_lineages"
        and row["statistic"] == "top_1pct_support_difference_unmoderated_t"
    )
    status = "PASS_FORMAL" if formal else "PASS_DEVELOPMENT_ONLY"
    summary = {
        "created_at_utc": utc_now(),
        "status": status,
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "numpy": np.__version__,
        "validation_lock": {
            "path": LOCK_REL, "sha256": sha256_file(lock_path),
        },
        "external_validation_handoff": {
            "path": HANDOFF_REL, "sha256": sha256_file(handoff_path),
        },
        "frozen_candidates": {
            "path": CANDIDATE_REL, "sha256": sha256_file(candidate_path),
            "count": 250,
        },
        "frozen_ranking": {
            "path": RANKING_REL, "sha256": sha256_file(ranking_path),
            "pairs": 31125, "ranking_models": len(RANKING_FAMILY),
            "model": handoff["model"]["name"],
            "primary_rank_field": "OSA_interaction_priority_rank",
        },
        "preprocess_summary_sha256": sha256_file(preprocess_summary_path),
        "permutations": args.permutations,
        "formal_iterations_required": formal_iterations,
        "batch_size": args.batch_size,
        "seed": seed,
        "lineage_candidate_coverage": dict(zip(LINEAGES, actual_coverage)),
        "lineage_eligible_pairs": {
            lineage: int(np.sum(observed_lineage[lineage]["eligible"]))
            for lineage in LINEAGES
        },
        "primary": {
            "endpoint": "mean top-1-percent support difference across T_NK, B, and Myeloid",
            "observed": float(primary_row["observed"]),
            "node_aware_one_sided_P": float(primary_row["one_sided_P"]),
            "claim_allowed": float(primary_row["one_sided_P"]) < 0.05 and formal,
        },
        "participant_label_secondary": {
            "observed": float(participant_row["observed"]),
            "one_sided_P": float(participant_row["one_sided_P"]),
        },
        "secondary_model_comparison": {
            "comparisons": len(model_random_rows),
            "full_superiority_claims_at_BH_FDR_0p05": int(sum(
                row["full_superiority_claim_allowed"] == "yes"
                for row in model_random_rows
            )),
        },
        "single_gene_FDR_significance_required": False,
        "ranking_changed": False,
        "candidate_membership_changed": False,
        "previous_CPAP_or_adipose_validation_read": False,
        "known_PPI_pathway_or_complex_read": False,
        "validation_used_for_training_or_selection": False,
        "runtime_seconds": time.time() - started,
    }
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, summary)
    output_paths = list(paths.values()) + [summary_path]
    manifest_rows = [{
        "path": relative_path(path, root),
        "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in output_paths]
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(manifest_path, manifest_rows, ["path", "bytes", "sha256"])
    print("OSA GSE214865 cell-context endpoint: " + status)
    print("Primary omnibus difference={} node-aware P={}".format(
        primary_row["observed"], primary_row["one_sided_P"]
    ))
    print("Participant-label omnibus P={}".format(
        participant_row["one_sided_P"]
    ))
    print("Output: " + str(output_dir))
    print("Frozen candidate membership and ranking unchanged")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(1)
