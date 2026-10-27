#!/usr/bin/env python3
"""Evaluate locked CPAP reversal enrichment for frozen OSA pair rankings.

Python 3.7 compatible.  Cohorts are evaluated separately.  The primary null
permutes complete node evidence vectors within outcome-independent matched
blocks, preserving the complete pair graph and node reuse exactly.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import io
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
LOCK_REL = "config/osa_cpap_validation_lock.json"
HANDOFF_REL = "config/triage_osa_external_validation_lock_v4.json"
RANKING_REL = (
    "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/"
    "pair_context_scores_triage_osa_primary_250.tsv.gz"
)
RANKING_SHA256 = "1e87e8ecaaf6d4a70ad89086237bcdc84633e5029d7f30efb0a0ccaadf6b03d3"
CANDIDATE_REL = (
    "results/discovery_preprocess_v2/04_candidates_v1/"
    "candidate_proteins_primary_250.tsv"
)
PREPROCESS_STAGE = "10_cpap_validation_preprocess_v1"
DEFAULT_STAGE = "11_cpap_reversal_endpoint_triage_osa_v4"
COHORTS = ["GSE133601", "GSE49800"]
RANKING_FAMILY = [
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
    ("bidirectional_conservative", "rank_priority_bidirectional_conservative"),
]
FRACTIONS = [0.005, 0.01, 0.02, 0.05]


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


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relative_path(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def read_tsv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_tsv(path, rows, fields):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=fields, delimiter="\t",
                    lineterminator="\n", extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(str(tmp), str(path))


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


def fmt(value):
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return "{:.10g}".format(value)


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


def normalized_auec(np, fractions, differences):
    width = float(fractions[-1] - fractions[0])
    increments = fractions[1:] - fractions[:-1]
    areas = 0.5 * (differences[1:] + differences[:-1]) * increments
    return float(np.sum(areas) / width)


def matched_blocks(np, candidates, available_accessions, block_size=10):
    candidate_map = {row["UniProt_accession"]: row for row in candidates}
    accessions = sorted(available_accessions)
    values = np.asarray([[
        float(candidate_map[accession]["protein_detection_fraction"]),
        float(candidate_map[accession]["log2_UniProt_length"]),
    ] for accession in accessions], dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    standardized = (values - mean) / np.maximum(std, 1e-12)
    position = {accession: index for index, accession in enumerate(accessions)}
    unassigned = set(accessions)
    blocks = []
    while unassigned:
        anchor = min(unassigned)
        anchor_vector = standardized[position[anchor]]
        ordered = sorted(
            unassigned,
            key=lambda accession: (
                float(np.sum((standardized[position[accession]] - anchor_vector) ** 2)),
                accession,
            ),
        )
        block = ordered[:block_size]
        blocks.append(block)
        unassigned.difference_update(block)
    return blocks


def build_node_evidence(np, candidate_rows, validation_rows):
    validation = {row["UniProt_accession"]: row for row in validation_rows}
    available = []
    oriented_t = []
    evidence = {}
    for candidate in candidate_rows:
        accession = candidate["UniProt_accession"]
        row = validation[accession]
        if row["validation_available"] != "yes":
            evidence[accession] = {
                "available": False,
                "oriented_beta": None,
                "oriented_t": None,
                "score": None,
                "same_direction_score": None,
                "direction_reversed": None,
            }
            continue
        discovery_sign = 1.0 if row["discovery_OSA_beta_sign"] == "positive" else -1.0
        oriented_beta = -discovery_sign * float(row["CPAP_beta_post_minus_pre"])
        value = -discovery_sign * float(row["moderated_t"])
        available.append(accession)
        oriented_t.append(value)
        evidence[accession] = {
            "available": True,
            "oriented_beta": oriented_beta,
            "oriented_t": value,
            "score": None,
            "same_direction_score": None,
            "direction_reversed": oriented_beta > 0.0,
        }
    reversal_scores = midrank_percentiles(np, oriented_t)
    same_direction_scores = midrank_percentiles(np, [-value for value in oriented_t])
    for accession, reversal, same in zip(
        available, reversal_scores, same_direction_scores
    ):
        evidence[accession]["score"] = float(reversal)
        evidence[accession]["same_direction_score"] = float(same)
    return evidence, available


def endpoint_metrics(np, pair_rows, support, both_reversed, eligible, rank_field):
    n_universe = len(pair_rows)
    ranks = np.asarray([parse_rank(row[rank_field]) for row in pair_rows], dtype=np.int64)
    background_mean = float(np.mean(support[eligible]))
    background_binary = float(np.mean(both_reversed[eligible]))
    rows = []
    differences = []
    ratios = []
    for fraction in FRACTIONS:
        threshold = int(math.ceil(n_universe * fraction))
        selected = eligible & (ranks <= threshold)
        top_mean = float(np.mean(support[selected]))
        top_binary = float(np.mean(both_reversed[selected]))
        difference = top_mean - background_mean
        ratio = top_mean / max(background_mean, 1e-12)
        rows.append({
            "fraction": fmt(fraction),
            "frozen_rank_threshold": str(threshold),
            "eligible_top_pairs": str(int(np.sum(selected))),
            "eligible_background_pairs": str(int(np.sum(eligible))),
            "top_mean_pair_reversal_support": fmt(top_mean),
            "background_mean_pair_reversal_support": fmt(background_mean),
            "support_enrichment_ratio": fmt(ratio),
            "support_enrichment_difference": fmt(difference),
            "top_both_nodes_reversed_fraction": fmt(top_binary),
            "background_both_nodes_reversed_fraction": fmt(background_binary),
            "binary_reversal_ratio": fmt(
                top_binary / max(background_binary, 1e-12)
            ),
        })
        differences.append(difference)
        ratios.append(ratio)
    auec = normalized_auec(
        np, np.asarray(FRACTIONS, dtype=float),
        np.asarray(differences, dtype=float),
    )
    return rows, auec, ratios[1]


def node_aware_permutation(
    np, rng, iterations, batch_size, candidate_rows, evidence, pair_rows,
    eligible, left_index, right_index, primary_ranks,
):
    candidate_order = [row["UniProt_accession"] for row in candidate_rows]
    candidate_index = {
        accession: index for index, accession in enumerate(candidate_order)
    }
    available = [accession for accession in candidate_order if evidence[accession]["available"]]
    blocks = matched_blocks(np, candidate_rows, available, 10)
    block_indices = [np.asarray([
        candidate_index[accession] for accession in block
    ], dtype=np.int64) for block in blocks]
    base = np.zeros(len(candidate_order), dtype=np.float64)
    for accession in available:
        base[candidate_index[accession]] = evidence[accession]["score"]
    masks = []
    for fraction in FRACTIONS:
        threshold = int(math.ceil(len(pair_rows) * fraction))
        masks.append(eligible & (primary_ranks <= threshold))
    null_ratio = np.empty(iterations, dtype=np.float64)
    null_auec = np.empty(iterations, dtype=np.float64)
    completed = 0
    while completed < iterations:
        size = min(batch_size, iterations - completed)
        permuted = np.repeat(base.reshape(1, -1), size, axis=0)
        for batch_row in range(size):
            for indices in block_indices:
                permuted[batch_row, indices] = base[rng.permutation(indices)]
        pair_support = np.sqrt(
            permuted[:, left_index] * permuted[:, right_index]
        )
        background = pair_support[:, eligible].mean(axis=1)
        differences = []
        for mask in masks:
            differences.append(pair_support[:, mask].mean(axis=1) - background)
        difference_matrix = np.stack(differences, axis=1)
        top_one_mean = pair_support[:, masks[1]].mean(axis=1)
        null_ratio[completed:completed + size] = top_one_mean / np.maximum(
            background, 1e-12
        )
        fraction_array = np.asarray(FRACTIONS, dtype=np.float64)
        increments = fraction_array[1:] - fraction_array[:-1]
        areas = 0.5 * (
            difference_matrix[:, 1:] + difference_matrix[:, :-1]
        ) * increments.reshape(1, -1)
        null_auec[completed:completed + size] = np.sum(areas, axis=1) / (
            FRACTIONS[-1] - FRACTIONS[0]
        )
        completed += size
        if completed % 1000 == 0 or completed == iterations:
            print("[permutation] {}/{}".format(completed, iterations), flush=True)
    block_rows = []
    candidate_map = {row["UniProt_accession"]: row for row in candidate_rows}
    for block_id, block in enumerate(blocks, 1):
        for accession in block:
            block_rows.append({
                "block_id": str(block_id),
                "block_size": str(len(block)),
                "UniProt_accession": accession,
                "protein_detection_fraction": candidate_map[accession]["protein_detection_fraction"],
                "log2_UniProt_length": candidate_map[accession]["log2_UniProt_length"],
            })
    return null_ratio, null_auec, block_rows


def null_summary(np, name, observed, null):
    null = np.asarray(null, dtype=float)
    return {
        "statistic": name,
        "observed": fmt(observed),
        "null_mean": fmt(np.mean(null)),
        "null_SD": fmt(np.std(null, ddof=1)),
        "null_q025": fmt(np.quantile(null, 0.025)),
        "null_q50": fmt(np.quantile(null, 0.50)),
        "null_q975": fmt(np.quantile(null, 0.975)),
        "one_sided_permutation_P": fmt(
            (1.0 + float(np.sum(null >= observed))) / (len(null) + 1.0)
        ),
        "iterations": str(len(null)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preprocess-stage", default=PREPROCESS_STAGE)
    parser.add_argument("--output-stage", default=DEFAULT_STAGE)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--permutation-batch", type=int, default=100)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.permutations <= 0 or args.permutation_batch <= 0:
        raise RuntimeError("permutations and permutation-batch must be positive")
    root = args.root.expanduser().resolve()
    lock_path = root / LOCK_REL
    handoff_path = root / HANDOFF_REL
    ranking_path = root / RANKING_REL
    candidate_path = root / CANDIDATE_REL
    preprocess_dir = root / "results" / "discovery_preprocess_v2" / args.preprocess_stage
    preprocess_summary_path = preprocess_dir / "run_summary.json"
    for path in [lock_path, handoff_path, ranking_path, candidate_path, preprocess_summary_path]:
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff["ranking"]["primary_250"] != RANKING_REL:
        raise RuntimeError("TRIAGE-OSA handoff ranking path mismatch")
    if handoff["ranking"]["primary_250_sha256"] != RANKING_SHA256:
        raise RuntimeError("TRIAGE-OSA handoff ranking hash mismatch")
    if handoff["prespecified_endpoint_locks"]["cpap"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("TRIAGE-OSA handoff CPAP lock mismatch")
    if sha256_file(ranking_path) != RANKING_SHA256:
        raise RuntimeError("Frozen ranking hash mismatch")
    preprocess_summary = json.loads(preprocess_summary_path.read_text(encoding="utf-8"))
    if preprocess_summary.get("status") != "PASS":
        raise RuntimeError("CPAP preprocessing has not passed")
    if preprocess_summary["validation_lock"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("Preprocessing used a different validation lock")
    ranking_rows, _ = read_tsv(ranking_path)
    candidate_rows, _ = read_tsv(candidate_path)
    if len(ranking_rows) != 31125 or len(candidate_rows) != 250:
        raise RuntimeError("Frozen universe row count mismatch")
    if any(
        int(float(row["triage_osa_rank_primary_250"]))
        != parse_rank(row["OSA_interaction_priority_rank"])
        for row in ranking_rows
    ):
        raise RuntimeError("TRIAGE-OSA native and compatibility ranks differ")
    for _, rank_field in RANKING_FAMILY:
        ranks = sorted(parse_rank(row[rank_field]) for row in ranking_rows)
        if ranks != list(range(1, 31126)):
            raise RuntimeError("Invalid frozen ranking: " + rank_field)
    print("OSA CPAP reversal endpoint environment: PASS")
    print("Frozen pairs: 31,125; ranking models: {}".format(len(RANKING_FAMILY)))
    print("Permutations: {} seed={}".format(
        args.permutations, lock["permutation"]["seed"]
    ))
    print("Known PPI/pathway data read: no")
    if args.check_env:
        return 0
    output_dir = root / "results" / "discovery_preprocess_v2" / args.output_stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace")
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed = int(lock["permutation"]["seed"])
    cohort_results = {}
    outputs = []
    pair_left = [row["protein_A"] for row in ranking_rows]
    pair_right = [row["protein_B"] for row in ranking_rows]
    candidate_order = [row["UniProt_accession"] for row in candidate_rows]
    candidate_index = {
        accession: index for index, accession in enumerate(candidate_order)
    }
    left_index = np.asarray([candidate_index[value] for value in pair_left], dtype=np.int64)
    right_index = np.asarray([candidate_index[value] for value in pair_right], dtype=np.int64)
    primary_ranks = np.asarray([
        parse_rank(row["OSA_interaction_priority_rank"]) for row in ranking_rows
    ], dtype=np.int64)
    for cohort_number, dataset in enumerate(COHORTS):
        validation_path = preprocess_dir / dataset / "candidate_validation_evidence.tsv"
        validation_rows, _ = read_tsv(validation_path)
        evidence, available = build_node_evidence(
            np, candidate_rows, validation_rows
        )
        score_vector = np.zeros(250, dtype=np.float64)
        same_vector = np.zeros(250, dtype=np.float64)
        available_vector = np.zeros(250, dtype=bool)
        reversed_vector = np.zeros(250, dtype=bool)
        node_output = []
        validation_map = {row["UniProt_accession"]: row for row in validation_rows}
        for candidate in candidate_rows:
            accession = candidate["UniProt_accession"]
            index = candidate_index[accession]
            item = evidence[accession]
            available_vector[index] = item["available"]
            if item["available"]:
                score_vector[index] = item["score"]
                same_vector[index] = item["same_direction_score"]
                reversed_vector[index] = item["direction_reversed"]
            source = validation_map[accession]
            node_output.append({
                "dataset": dataset,
                "UniProt_accession": accession,
                "gene_symbol": candidate["gene_symbol_current"],
                "Entrez": candidate["Entrez"],
                "validation_available": "yes" if item["available"] else "no",
                "discovery_OSA_beta": source["discovery_OSA_beta"],
                "CPAP_beta_post_minus_pre": source["CPAP_beta_post_minus_pre"],
                "CPAP_moderated_t": source["moderated_t"],
                "reversal_oriented_beta": fmt(item["oriented_beta"]),
                "reversal_oriented_moderated_t": fmt(item["oriented_t"]),
                "node_reversal_score": fmt(item["score"]),
                "node_same_direction_score_negative_control": fmt(
                    item["same_direction_score"]
                ),
                "directionally_reversed": (
                    "yes" if item["direction_reversed"] else "no"
                    if item["available"] else ""
                ),
            })
        eligible = available_vector[left_index] & available_vector[right_index]
        support = np.sqrt(score_vector[left_index] * score_vector[right_index])
        same_support = np.sqrt(same_vector[left_index] * same_vector[right_index])
        both_reversed = reversed_vector[left_index] & reversed_vector[right_index]
        pair_output = []
        for index, rank_row in enumerate(ranking_rows):
            pair_output.append({
                "pair_id": rank_row["pair_id"],
                "protein_A": rank_row["protein_A"],
                "protein_B": rank_row["protein_B"],
                "OSA_interaction_priority_rank": rank_row["OSA_interaction_priority_rank"],
                "validation_eligible": "yes" if eligible[index] else "no",
                "node_reversal_score_A": fmt(
                    evidence[rank_row["protein_A"]]["score"]
                ),
                "node_reversal_score_B": fmt(
                    evidence[rank_row["protein_B"]]["score"]
                ),
                "pair_reversal_support": fmt(support[index]) if eligible[index] else "",
                "both_nodes_directionally_reversed": (
                    "yes" if both_reversed[index] else "no"
                    if eligible[index] else ""
                ),
                "same_direction_pair_support_negative_control": (
                    fmt(same_support[index]) if eligible[index] else ""
                ),
            })
        metric_rows = []
        primary_auec = None
        primary_ratio = None
        for model_name, rank_field in RANKING_FAMILY:
            fraction_rows, auec, ratio_one = endpoint_metrics(
                np, ranking_rows, support, both_reversed, eligible, rank_field
            )
            for row in fraction_rows:
                row["dataset"] = dataset
                row["model"] = model_name
                row["rank_field"] = rank_field
                row["metric_scope"] = "fraction"
                row["normalized_AUEC_0p5_to_5pct"] = ""
                metric_rows.append(row)
            metric_rows.append({
                "dataset": dataset,
                "model": model_name,
                "rank_field": rank_field,
                "metric_scope": "AUEC",
                "fraction": "0.005_to_0.05",
                "normalized_AUEC_0p5_to_5pct": fmt(auec),
            })
            if model_name == "full_primary":
                primary_auec = auec
                primary_ratio = ratio_one
        bottom_mask = eligible & (primary_ranks > len(ranking_rows) - 312)
        background_mean = float(np.mean(support[eligible]))
        bottom_ratio = float(np.mean(support[bottom_mask])) / max(background_mean, 1e-12)
        same_top = eligible & (primary_ranks <= 312)
        same_ratio = float(np.mean(same_support[same_top])) / max(
            float(np.mean(same_support[eligible])), 1e-12
        )
        rng = np.random.RandomState(seed + cohort_number)
        null_ratio, null_auec, block_rows = node_aware_permutation(
            np, rng, args.permutations, args.permutation_batch,
            candidate_rows, evidence, ranking_rows, eligible,
            left_index, right_index, primary_ranks,
        )
        permutation_rows = [
            null_summary(np, "top_1pct_support_enrichment_ratio", primary_ratio, null_ratio),
            null_summary(np, "normalized_AUEC_0p5_to_5pct", primary_auec, null_auec),
        ]
        for row in permutation_rows:
            row["dataset"] = dataset
            row["permutation_type"] = "node_aware_matched_block"
        cohort_dir = output_dir / dataset
        cohort_dir.mkdir(parents=True, exist_ok=True)
        node_path = cohort_dir / "candidate_node_reversal.tsv"
        pair_path = cohort_dir / "pair_reversal_support.tsv.gz"
        metrics_path = cohort_dir / "ranking_endpoint_metrics.tsv"
        block_path = cohort_dir / "node_matching_blocks.tsv"
        permutation_path = cohort_dir / "permutation_summary.tsv"
        write_tsv(node_path, node_output, list(node_output[0]))
        write_gzip_tsv(pair_path, pair_output, list(pair_output[0]))
        metric_fields = [
            "dataset", "model", "rank_field", "metric_scope", "fraction",
            "frozen_rank_threshold", "eligible_top_pairs",
            "eligible_background_pairs", "top_mean_pair_reversal_support",
            "background_mean_pair_reversal_support", "support_enrichment_ratio",
            "support_enrichment_difference", "top_both_nodes_reversed_fraction",
            "background_both_nodes_reversed_fraction", "binary_reversal_ratio",
            "normalized_AUEC_0p5_to_5pct",
        ]
        write_tsv(metrics_path, metric_rows, metric_fields)
        write_tsv(block_path, block_rows, list(block_rows[0]))
        permutation_fields = [
            "dataset", "permutation_type", "statistic", "observed",
            "null_mean", "null_SD", "null_q025", "null_q50", "null_q975",
            "one_sided_permutation_P", "iterations",
        ]
        write_tsv(permutation_path, permutation_rows, permutation_fields)
        outputs.extend([
            node_path, pair_path, metrics_path, block_path, permutation_path,
        ])
        cohort_results[dataset] = {
            "available_nodes": len(available),
            "eligible_pairs": int(np.sum(eligible)),
            "primary_top_1pct_enrichment_ratio": primary_ratio,
            "primary_AUEC": primary_auec,
            "bottom_1pct_enrichment_ratio_negative_control": bottom_ratio,
            "same_direction_top_1pct_ratio_negative_control": same_ratio,
            "top_1pct_permutation_P": float(permutation_rows[0]["one_sided_permutation_P"]),
            "AUEC_permutation_P": float(permutation_rows[1]["one_sided_permutation_P"]),
            "null_log_ratio_SD": float(np.std(np.log(np.maximum(null_ratio, 1e-12)), ddof=1)),
            "null_ratio": null_ratio,
            "null_auec": null_auec,
        }
        print("[{}] nodes={} eligible_pairs={} top1_ratio={:.4f} P={} AUEC={:.4g} P={}".format(
            dataset, len(available), int(np.sum(eligible)), primary_ratio,
            permutation_rows[0]["one_sided_permutation_P"], primary_auec,
            permutation_rows[1]["one_sided_permutation_P"],
        ))
    effects = np.asarray([
        math.log(max(cohort_results[dataset]["primary_top_1pct_enrichment_ratio"], 1e-12))
        for dataset in COHORTS
    ])
    standard_errors = np.asarray([
        cohort_results[dataset]["null_log_ratio_SD"] for dataset in COHORTS
    ])
    weights = 1.0 / np.maximum(standard_errors ** 2, 1e-12)
    pooled = float(np.sum(weights * effects) / np.sum(weights))
    pooled_se = float(math.sqrt(1.0 / np.sum(weights)))
    z_value = pooled / pooled_se
    one_sided_p = 0.5 * math.erfc(z_value / math.sqrt(2.0))
    q_value = float(np.sum(weights * (effects - pooled) ** 2))
    i2 = max(0.0, (q_value - 1.0) / max(q_value, 1e-12))
    direction_consistent = bool(np.all(effects > 0.0) or np.all(effects < 0.0))
    meta_rows = []
    for dataset, effect, se in zip(COHORTS, effects, standard_errors):
        meta_rows.append({
            "scope": dataset,
            "log_enrichment_ratio": fmt(effect),
            "SE_from_permutation_null": fmt(se),
            "enrichment_ratio": fmt(math.exp(effect)),
            "one_sided_P": fmt(cohort_results[dataset]["top_1pct_permutation_P"]),
            "direction_consistent_across_cohorts": "yes" if direction_consistent else "no",
            "heterogeneity_Q": "",
            "I2": "",
            "pooled_claim_allowed": "",
        })
    meta_rows.append({
        "scope": "fixed_effect_meta",
        "log_enrichment_ratio": fmt(pooled),
        "SE_from_permutation_null": fmt(pooled_se),
        "enrichment_ratio": fmt(math.exp(pooled)),
        "one_sided_P": fmt(one_sided_p),
        "direction_consistent_across_cohorts": "yes" if direction_consistent else "no",
        "heterogeneity_Q": fmt(q_value),
        "I2": fmt(i2),
        "pooled_claim_allowed": (
            "yes" if direction_consistent and pooled > 0.0 and one_sided_p < 0.05
            else "no"
        ),
    })
    meta_path = output_dir / "cross_cohort_meta.tsv"
    write_tsv(meta_path, meta_rows, list(meta_rows[0]))
    outputs.append(meta_path)
    formal = args.permutations == int(lock["permutation"]["formal_iterations"])
    summary_results = {}
    for dataset in COHORTS:
        result = dict(cohort_results[dataset])
        result.pop("null_ratio")
        result.pop("null_auec")
        summary_results[dataset] = result
    summary = {
        "created_at_utc": utc_now(),
        "status": (
            "PASS_PRIMARY_FORMAL_SUBJECT_ROBUSTNESS_PENDING"
            if formal else "PASS_DEVELOPMENT_ONLY"
        ),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "validation_lock": {"path": LOCK_REL, "sha256": sha256_file(lock_path)},
        "external_validation_handoff": {
            "path": HANDOFF_REL, "sha256": sha256_file(handoff_path),
        },
        "ranking": {
            "path": RANKING_REL, "sha256": sha256_file(ranking_path),
            "model": handoff["model"]["name"],
            "primary_rank_field": "OSA_interaction_priority_rank",
        },
        "preprocess_summary_sha256": sha256_file(preprocess_summary_path),
        "permutations": args.permutations,
        "seed": seed,
        "cohorts": summary_results,
        "meta": {
            "pooled_log_enrichment_ratio": pooled,
            "pooled_enrichment_ratio": math.exp(pooled),
            "pooled_SE": pooled_se,
            "one_sided_P": one_sided_p,
            "heterogeneity_Q": q_value,
            "I2": i2,
            "direction_consistent": direction_consistent,
            "pooled_claim_allowed": (
                direction_consistent and pooled > 0.0 and one_sided_p < 0.05
            ),
        },
        "ranking_changed": False,
        "known_PPI_or_pathway_read": False,
        "raw_expression_cohorts_merged": False,
        "subject_aware_sign_flip_pending": True,
        "runtime_seconds": time.time() - started,
        "next_gate": (
            "Run subject-aware sign-flip robustness and freeze figures/tables"
            if formal else "Rerun with 10,000 permutations for formal inference"
        ),
        "python": sys.version,
        "numpy": np.__version__,
    }
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, summary)
    outputs.append(summary_path)
    manifest_rows = [{
        "path": relative_path(path, root),
        "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in outputs]
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(manifest_path, manifest_rows, ["path", "bytes", "sha256"])
    print("OSA CPAP reversal endpoint: {}".format(summary["status"]))
    print("Output: {}".format(output_dir))
    print("Frozen ranking unchanged; known PPI/pathway data read: no")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
