#!/usr/bin/env python3
"""Evaluate locked GSE135917 cross-tissue endpoints for frozen OSA pairs.

Python 3.7 compatible.  Case-control OSA-state support and paired CPAP
reversal are evaluated separately.  GSE135917 is never used to change the
candidate universe, pair universe, model, or frozen ranking.
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
LOCK_REL = "config/osa_adipose_validation_lock.json"
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
PREPROCESS_STAGE = "13_adipose_validation_preprocess_v1"
DEFAULT_STAGE = "14_adipose_cross_tissue_endpoint_triage_osa_v4"
FRACTIONS = [0.005, 0.01, 0.02, 0.05]
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


def trapezoid_auec(np, differences):
    fractions = np.asarray(FRACTIONS, dtype=np.float64)
    increments = fractions[1:] - fractions[:-1]
    areas = 0.5 * (differences[..., 1:] + differences[..., :-1]) * increments
    return np.sum(areas, axis=-1) / (FRACTIONS[-1] - FRACTIONS[0])


def endpoint_metrics(np, pair_rows, support, positive, eligible, rank_field):
    ranks = np.asarray([parse_rank(row[rank_field]) for row in pair_rows], dtype=np.int64)
    background = float(np.mean(support[eligible]))
    background_binary = float(np.mean(positive[eligible]))
    rows = []
    differences = []
    ratio_one = None
    for fraction in FRACTIONS:
        threshold = int(math.ceil(len(pair_rows) * fraction))
        selected = eligible & (ranks <= threshold)
        top_mean = float(np.mean(support[selected]))
        top_binary = float(np.mean(positive[selected]))
        difference = top_mean - background
        ratio = top_mean / max(background, 1e-12)
        if fraction == 0.01:
            ratio_one = ratio
        rows.append({
            "fraction": fmt(fraction),
            "frozen_rank_threshold": str(threshold),
            "eligible_top_pairs": str(int(np.sum(selected))),
            "eligible_background_pairs": str(int(np.sum(eligible))),
            "top_mean_support": fmt(top_mean),
            "background_mean_support": fmt(background),
            "support_enrichment_ratio": fmt(ratio),
            "support_enrichment_difference": fmt(difference),
            "top_both_nodes_positive_fraction": fmt(top_binary),
            "background_both_nodes_positive_fraction": fmt(background_binary),
            "binary_positive_ratio": fmt(top_binary / max(background_binary, 1e-12)),
        })
        differences.append(difference)
    auec = float(trapezoid_auec(np, np.asarray(differences, dtype=np.float64)))
    return rows, auec, ratio_one


def evidence_from_candidate(np, candidates, candidate_evidence, endpoint):
    source = {row["UniProt_accession"]: row for row in candidate_evidence}
    evidence = {}
    accessions = []
    oriented_t = []
    for candidate in candidates:
        accession = candidate["UniProt_accession"]
        row = source[accession]
        if row["adipose_available"] != "yes":
            evidence[accession] = {
                "available": False, "score": None, "oriented_t": None,
                "oriented_beta": None, "positive": None,
            }
            continue
        discovery_sign = 1.0 if row["discovery_OSA_beta_sign"] == "positive" else -1.0
        if endpoint == "state":
            beta = discovery_sign * float(row["state_beta"])
            value = discovery_sign * float(row["state_moderated_t"])
        elif endpoint == "state_adjusted":
            beta = discovery_sign * float(row["state_adjusted_beta"])
            value = discovery_sign * float(row["state_adjusted_moderated_t"])
        elif endpoint == "CPAP":
            beta = -discovery_sign * float(row["CPAP_beta"])
            value = -discovery_sign * float(row["CPAP_moderated_t"])
        else:
            raise RuntimeError("Unknown endpoint: " + endpoint)
        accessions.append(accession)
        oriented_t.append(value)
        evidence[accession] = {
            "available": True, "score": None, "oriented_t": value,
            "oriented_beta": beta, "positive": beta > 0.0,
        }
    from osa_cpap_reversal_endpoint import midrank_percentiles
    scores = midrank_percentiles(np, oriented_t)
    for accession, score in zip(accessions, scores):
        evidence[accession]["score"] = float(score)
    return evidence, accessions


def evidence_vectors(np, candidates, evidence):
    score = np.zeros(len(candidates), dtype=np.float64)
    available = np.zeros(len(candidates), dtype=bool)
    positive = np.zeros(len(candidates), dtype=bool)
    for index, candidate in enumerate(candidates):
        item = evidence[candidate["UniProt_accession"]]
        available[index] = item["available"]
        if item["available"]:
            score[index] = item["score"]
            positive[index] = item["positive"]
    return score, available, positive


def bh_two(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 1.0
    for reverse_position in range(len(values) - 1, -1, -1):
        index = order[reverse_position]
        rank = reverse_position + 1
        running = min(running, values[index] * len(values) / float(rank))
        adjusted[index] = min(1.0, running)
    return adjusted


def case_control_t(np, expression, labels):
    n_osa = 10.0
    n_control = 8.0
    mean_osa = np.dot(expression, labels.T) / n_osa
    mean_control = np.dot(expression, (1.0 - labels).T) / n_control
    sum_squares = expression * expression
    ss_osa = np.dot(sum_squares, labels.T) - n_osa * mean_osa * mean_osa
    ss_control = (
        np.dot(sum_squares, (1.0 - labels).T)
        - n_control * mean_control * mean_control
    )
    pooled = np.maximum((ss_osa + ss_control) / 16.0, 1e-15)
    se = np.sqrt(pooled * (1.0 / n_osa + 1.0 / n_control))
    return (mean_osa - mean_control) / se


def fill_candidate_scores(np, candidates, available_accessions, available_scores):
    candidate_index = {
        row["UniProt_accession"]: index for index, row in enumerate(candidates)
    }
    scores = np.zeros((available_scores.shape[0], len(candidates)), dtype=np.float64)
    for source_index, accession in enumerate(available_accessions):
        scores[:, candidate_index[accession]] = available_scores[:, source_index]
    return scores


def pair_draw_statistics(np, scores, left_index, right_index, eligible, masks):
    support = np.sqrt(scores[:, left_index] * scores[:, right_index])
    background = np.mean(support[:, eligible], axis=1)
    differences = []
    top_one = None
    for mask_index, mask in enumerate(masks):
        top = np.mean(support[:, mask], axis=1)
        differences.append(top - background)
        if mask_index == 1:
            top_one = top
    return (
        top_one / np.maximum(background, 1e-12),
        trapezoid_auec(np, np.stack(differences, axis=1)),
    )


def randomization_summary(np, endpoint, method, statistic, observed, null):
    return {
        "endpoint": endpoint,
        "randomization": method,
        "statistic": statistic,
        "observed": fmt(observed),
        "null_mean": fmt(np.mean(null)),
        "null_SD": fmt(np.std(null, ddof=1)),
        "null_q025": fmt(np.quantile(null, 0.025)),
        "null_q50": fmt(np.quantile(null, 0.50)),
        "null_q975": fmt(np.quantile(null, 0.975)),
        "one_sided_P": fmt((1.0 + float(np.sum(null >= observed))) / (len(null) + 1.0)),
        "iterations": str(len(null)),
    }


def subject_randomizations(
    np, rng, iterations, batch_size, candidates, candidate_evidence,
    case_rows, case_fields, paired_rows, paired_fields, sample_manifest,
    left_index, right_index, eligible, masks,
):
    from osa_cpap_subject_signflip import paired_t, score_draws
    evidence_map = {row["UniProt_accession"]: row for row in candidate_evidence}
    available = [
        row["UniProt_accession"] for row in candidates
        if evidence_map[row["UniProt_accession"]]["adipose_available"] == "yes"
    ]
    discovery_sign = np.asarray([
        1.0 if evidence_map[accession]["discovery_OSA_beta_sign"] == "positive"
        else -1.0 for accession in available
    ], dtype=np.float64).reshape(-1, 1)
    case_map = {row["UniProt_accession"]: row for row in case_rows}
    case_samples = [field[len("sample_"):] for field in case_fields if field.startswith("sample_")]
    manifest_map = {row["sample_id"]: row for row in sample_manifest}
    case_expression = np.asarray([[
        float(case_map[accession]["sample_" + sample]) for sample in case_samples
    ] for accession in available], dtype=np.float64)
    observed_labels = np.asarray([[
        1.0 if manifest_map[sample]["disease_group"] == "OSA" else 0.0
        for sample in case_samples
    ]], dtype=np.float64)
    if int(np.sum(observed_labels)) != 10:
        raise RuntimeError("Observed GSE135917 disease label count mismatch")
    observed_case_t = case_control_t(np, case_expression, observed_labels)
    observed_case_available = score_draws(np, discovery_sign * observed_case_t)
    observed_case_scores = fill_candidate_scores(
        np, candidates, available, observed_case_available
    )
    observed_case_ratio, observed_case_auec = pair_draw_statistics(
        np, observed_case_scores, left_index, right_index, eligible, masks
    )
    paired_map = {row["UniProt_accession"]: row for row in paired_rows}
    subject_fields = [field for field in paired_fields if field.startswith("subject_")]
    deltas = np.asarray([[
        float(paired_map[accession][field]) for field in subject_fields
    ] for accession in available], dtype=np.float64)
    observed_paired_t = paired_t(
        np, deltas, np.ones((1, len(subject_fields)), dtype=np.float64)
    )
    observed_paired_available = score_draws(np, -discovery_sign * observed_paired_t)
    observed_paired_scores = fill_candidate_scores(
        np, candidates, available, observed_paired_available
    )
    observed_paired_ratio, observed_paired_auec = pair_draw_statistics(
        np, observed_paired_scores, left_index, right_index, eligible, masks
    )
    null_case_ratio = np.empty(iterations, dtype=np.float64)
    null_case_auec = np.empty(iterations, dtype=np.float64)
    null_paired_ratio = np.empty(iterations, dtype=np.float64)
    null_paired_auec = np.empty(iterations, dtype=np.float64)
    completed = 0
    while completed < iterations:
        size = min(batch_size, iterations - completed)
        labels = np.zeros((size, len(case_samples)), dtype=np.float64)
        for row_index in range(size):
            labels[row_index, rng.permutation(len(case_samples))[:10]] = 1.0
        case_t = case_control_t(np, case_expression, labels)
        case_available_scores = score_draws(np, discovery_sign * case_t)
        case_scores = fill_candidate_scores(
            np, candidates, available, case_available_scores
        )
        ratio, auec = pair_draw_statistics(
            np, case_scores, left_index, right_index, eligible, masks
        )
        null_case_ratio[completed:completed + size] = ratio
        null_case_auec[completed:completed + size] = auec
        signs = rng.choice(
            np.asarray([-1.0, 1.0]), size=(size, len(subject_fields))
        )
        paired_t_values = paired_t(np, deltas, signs)
        paired_available_scores = score_draws(
            np, -discovery_sign * paired_t_values
        )
        paired_scores = fill_candidate_scores(
            np, candidates, available, paired_available_scores
        )
        ratio, auec = pair_draw_statistics(
            np, paired_scores, left_index, right_index, eligible, masks
        )
        null_paired_ratio[completed:completed + size] = ratio
        null_paired_auec[completed:completed + size] = auec
        completed += size
        if completed % 1000 == 0 or completed == iterations:
            print("[subject-randomization] {}/{}".format(completed, iterations), flush=True)
    observed = {
        "state_ratio": float(observed_case_ratio[0]),
        "state_auec": float(observed_case_auec[0]),
        "CPAP_ratio": float(observed_paired_ratio[0]),
        "CPAP_auec": float(observed_paired_auec[0]),
    }
    nulls = {
        "state_ratio": null_case_ratio, "state_auec": null_case_auec,
        "CPAP_ratio": null_paired_ratio, "CPAP_auec": null_paired_auec,
    }
    return observed, nulls


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
        atomic_json, read_tsv, relative_path, sha256_file, write_gzip_tsv,
        write_tsv,
    )
    from osa_cpap_reversal_endpoint import node_aware_permutation
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    lock_path = root / LOCK_REL
    handoff_path = root / HANDOFF_REL
    ranking_path = root / RANKING_REL
    candidate_path = root / CANDIDATE_REL
    preprocess_dir = root / "results" / "discovery_preprocess_v2" / args.preprocess_stage
    preprocess_summary_path = preprocess_dir / "run_summary.json"
    preprocess_manifest_path = preprocess_dir / "output_manifest.tsv"
    required = [
        lock_path, handoff_path, ranking_path, candidate_path, preprocess_summary_path,
        preprocess_manifest_path,
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
    if handoff["prespecified_endpoint_locks"]["adipose"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("TRIAGE-OSA handoff adipose lock mismatch")
    if sha256_file(ranking_path) != RANKING_SHA256:
        raise RuntimeError("Frozen ranking hash mismatch")
    if sha256_file(candidate_path) != lock["input_hashes"]["candidate_proteins_primary_250.tsv"]:
        raise RuntimeError("Frozen candidate hash mismatch")
    preprocess_summary = json.loads(preprocess_summary_path.read_text(encoding="utf-8"))
    if preprocess_summary.get("status") != "PASS":
        raise RuntimeError("Adipose preprocessing has not passed")
    if preprocess_summary["validation_lock"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("Adipose preprocessing used a different validation lock")
    manifest_rows, _ = read_tsv(preprocess_manifest_path)
    for row in manifest_rows:
        path = root / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError("Adipose preprocess manifest verification failed: " + row["path"])
    candidates, _ = read_tsv(candidate_path)
    pair_rows, _ = read_tsv(ranking_path)
    if any(
        int(float(row["triage_osa_rank_primary_250"]))
        != parse_rank(row["OSA_interaction_priority_rank"])
        for row in pair_rows
    ):
        raise RuntimeError("TRIAGE-OSA native and compatibility ranks differ")
    candidate_evidence_path = preprocess_dir / "candidate_adipose_evidence.tsv"
    candidate_evidence, _ = read_tsv(candidate_evidence_path)
    case_path = preprocess_dir / "candidate_case_control_expression.tsv.gz"
    paired_path = preprocess_dir / "candidate_paired_deltas.tsv.gz"
    case_rows, case_fields = read_tsv(case_path)
    paired_rows, paired_fields = read_tsv(paired_path)
    sample_manifest, _ = read_tsv(root / "metadata/sample_manifest.tsv")
    sample_manifest = [row for row in sample_manifest if row["dataset"] == "GSE135917"]
    if len(candidates) != 250 or len(pair_rows) != 31125 or len(candidate_evidence) != 250:
        raise RuntimeError("Adipose frozen universe row count mismatch")
    for _, rank_field in RANKING_FAMILY:
        ranks = sorted(parse_rank(row[rank_field]) for row in pair_rows)
        if ranks != list(range(1, 31126)):
            raise RuntimeError("Invalid frozen rank: " + rank_field)
    print("OSA adipose cross-tissue endpoint environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("Frozen pairs: 31,125; permutations: {}".format(args.permutations))
    print("Previous CPAP validation/known PPI/pathway read: no")
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
    candidate_index = {
        row["UniProt_accession"]: index for index, row in enumerate(candidates)
    }
    left_index = np.asarray([
        candidate_index[row["protein_A"]] for row in pair_rows
    ], dtype=np.int64)
    right_index = np.asarray([
        candidate_index[row["protein_B"]] for row in pair_rows
    ], dtype=np.int64)
    primary_ranks = np.asarray([
        parse_rank(row["OSA_interaction_priority_rank"]) for row in pair_rows
    ], dtype=np.int64)
    endpoint_evidence = {}
    endpoint_vectors = {}
    for endpoint in ["state", "state_adjusted", "CPAP"]:
        evidence, available = evidence_from_candidate(
            np, candidates, candidate_evidence, endpoint
        )
        endpoint_evidence[endpoint] = evidence
        endpoint_vectors[endpoint] = evidence_vectors(np, candidates, evidence)
    state_score, state_available, state_positive = endpoint_vectors["state"]
    adjusted_score, adjusted_available, adjusted_positive = endpoint_vectors["state_adjusted"]
    cpap_score, cpap_available, cpap_positive = endpoint_vectors["CPAP"]
    eligible = state_available[left_index] & state_available[right_index] & cpap_available[left_index] & cpap_available[right_index]
    state_support = np.sqrt(state_score[left_index] * state_score[right_index])
    cpap_support = np.sqrt(cpap_score[left_index] * cpap_score[right_index])
    adjusted_support = np.sqrt(adjusted_score[left_index] * adjusted_score[right_index])
    joint_node = np.sqrt(state_score * cpap_score)
    joint_support = np.sqrt(joint_node[left_index] * joint_node[right_index])
    both_state = state_positive[left_index] & state_positive[right_index]
    both_cpap = cpap_positive[left_index] & cpap_positive[right_index]
    both_adjusted = adjusted_positive[left_index] & adjusted_positive[right_index]
    both_joint = both_state & both_cpap
    metric_rows = []
    primary_observed = {}
    metric_inputs = [
        ("state", state_support, both_state),
        ("CPAP", cpap_support, both_cpap),
        ("joint_secondary", joint_support, both_joint),
    ]
    for endpoint, support, positive in metric_inputs:
        for model_name, rank_field in RANKING_FAMILY:
            fraction_rows, auec, ratio = endpoint_metrics(
                np, pair_rows, support, positive, eligible, rank_field
            )
            for row in fraction_rows:
                row.update({
                    "endpoint": endpoint, "model": model_name,
                    "rank_field": rank_field, "metric_scope": "fraction",
                    "normalized_AUEC_0p5_to_5pct": "",
                })
                metric_rows.append(row)
            metric_rows.append({
                "endpoint": endpoint, "model": model_name,
                "rank_field": rank_field, "metric_scope": "AUEC",
                "fraction": "0.005_to_0.05",
                "normalized_AUEC_0p5_to_5pct": fmt(auec),
            })
            if model_name == "full_primary" and endpoint in ["state", "CPAP"]:
                primary_observed[endpoint] = {"ratio": ratio, "auec": auec}
    adjusted_fraction, adjusted_auec, adjusted_ratio = endpoint_metrics(
        np, pair_rows, adjusted_support, both_adjusted, eligible,
        "OSA_interaction_priority_rank",
    )
    for row in adjusted_fraction:
        row.update({
            "endpoint": "state_covariate_adjusted_sensitivity",
            "model": "full_primary", "rank_field": "OSA_interaction_priority_rank",
            "metric_scope": "fraction", "normalized_AUEC_0p5_to_5pct": "",
        })
        metric_rows.append(row)
    metric_rows.append({
        "endpoint": "state_covariate_adjusted_sensitivity",
        "model": "full_primary", "rank_field": "OSA_interaction_priority_rank",
        "metric_scope": "AUEC", "fraction": "0.005_to_0.05",
        "normalized_AUEC_0p5_to_5pct": fmt(adjusted_auec),
    })
    seed = int(lock["randomization"]["seed"])
    node_rows = []
    blocks_output = None
    for endpoint_number, endpoint in enumerate(["state", "CPAP"]):
        rng = np.random.RandomState(seed + 200 + endpoint_number)
        null_ratio, null_auec, block_rows = node_aware_permutation(
            np, rng, args.permutations, args.batch_size, candidates,
            endpoint_evidence[endpoint], pair_rows, eligible,
            left_index, right_index, primary_ranks,
        )
        if blocks_output is None:
            blocks_output = block_rows
        node_rows.extend([
            randomization_summary(
                np, endpoint, "node_aware_matched_block",
                "top_1pct_support_enrichment_ratio",
                primary_observed[endpoint]["ratio"], null_ratio,
            ),
            randomization_summary(
                np, endpoint, "node_aware_matched_block",
                "normalized_AUEC_0p5_to_5pct",
                primary_observed[endpoint]["auec"], null_auec,
            ),
        ])
    masks = []
    for fraction in FRACTIONS:
        threshold = int(math.ceil(len(pair_rows) * fraction))
        masks.append(eligible & (primary_ranks <= threshold))
    subject_observed, subject_nulls = subject_randomizations(
        np, np.random.RandomState(seed + 300), args.permutations,
        args.batch_size, candidates, candidate_evidence, case_rows,
        case_fields, paired_rows, paired_fields, sample_manifest,
        left_index, right_index, eligible, masks,
    )
    subject_rows = []
    for endpoint in ["state", "CPAP"]:
        method = (
            "case_control_participant_label_permutation"
            if endpoint == "state" else "paired_participant_complete_vector_sign_flip"
        )
        subject_rows.extend([
            randomization_summary(
                np, endpoint, method,
                "top_1pct_support_enrichment_ratio_unmoderated_t",
                subject_observed[endpoint + "_ratio"],
                subject_nulls[endpoint + "_ratio"],
            ),
            randomization_summary(
                np, endpoint, method,
                "normalized_AUEC_0p5_to_5pct_unmoderated_t",
                subject_observed[endpoint + "_auec"],
                subject_nulls[endpoint + "_auec"],
            ),
        ])
    primary_p = [float(node_rows[0]["one_sided_P"]), float(node_rows[2]["one_sided_P"])]
    primary_fdr = bh_two(primary_p)
    multiplicity_rows = []
    for endpoint, p_value, fdr in zip(["state", "CPAP"], primary_p, primary_fdr):
        multiplicity_rows.append({
            "endpoint": endpoint,
            "family": "cross_tissue_primary_top_1pct",
            "one_sided_P": fmt(p_value),
            "BH_FDR_two_primary_tests": fmt(fdr),
            "claim_allowed": "yes" if fdr < 0.05 else "no",
        })
    node_output = []
    for candidate in candidates:
        accession = candidate["UniProt_accession"]
        state = endpoint_evidence["state"][accession]
        adjusted = endpoint_evidence["state_adjusted"][accession]
        cpap = endpoint_evidence["CPAP"][accession]
        node_output.append({
            "UniProt_accession": accession,
            "gene_symbol": candidate["gene_symbol_current"],
            "Entrez": candidate["Entrez"],
            "adipose_available": "yes" if state["available"] else "no",
            "state_oriented_beta": fmt(state["oriented_beta"]),
            "state_oriented_moderated_t": fmt(state["oriented_t"]),
            "state_node_score": fmt(state["score"]),
            "state_direction_positive": "yes" if state.get("positive") else "no" if state["available"] else "",
            "state_adjusted_oriented_beta": fmt(adjusted["oriented_beta"]),
            "state_adjusted_node_score": fmt(adjusted["score"]),
            "CPAP_oriented_beta": fmt(cpap["oriented_beta"]),
            "CPAP_oriented_moderated_t": fmt(cpap["oriented_t"]),
            "CPAP_node_score": fmt(cpap["score"]),
            "CPAP_direction_positive": "yes" if cpap.get("positive") else "no" if cpap["available"] else "",
            "joint_node_score": fmt(
                math.sqrt(state["score"] * cpap["score"])
                if state["available"] and cpap["available"] else None
            ),
            "persistent_state_without_reversal": (
                "yes" if state.get("positive") and not cpap.get("positive")
                else "no" if state["available"] and cpap["available"] else ""
            ),
        })
    pair_output = []
    for index, row in enumerate(pair_rows):
        pair_output.append({
            "pair_id": row["pair_id"], "protein_A": row["protein_A"],
            "protein_B": row["protein_B"],
            "OSA_interaction_priority_rank": row["OSA_interaction_priority_rank"],
            "validation_eligible": "yes" if eligible[index] else "no",
            "state_pair_support": fmt(state_support[index]) if eligible[index] else "",
            "CPAP_pair_support": fmt(cpap_support[index]) if eligible[index] else "",
            "joint_pair_support": fmt(joint_support[index]) if eligible[index] else "",
            "both_nodes_state_positive": "yes" if both_state[index] else "no" if eligible[index] else "",
            "both_nodes_CPAP_positive": "yes" if both_cpap[index] else "no" if eligible[index] else "",
            "both_nodes_joint_positive": "yes" if both_joint[index] else "no" if eligible[index] else "",
        })
    bottom = eligible & (primary_ranks > len(pair_rows) - 312)
    top = eligible & (primary_ranks <= 312)
    negative_controls = {
        "state_bottom_1pct_ratio": float(np.mean(state_support[bottom]) / np.mean(state_support[eligible])),
        "CPAP_bottom_1pct_ratio": float(np.mean(cpap_support[bottom]) / np.mean(cpap_support[eligible])),
        "state_opposite_direction_top_1pct_ratio": float(
            np.mean(np.sqrt((1.0 - state_score[left_index[top]]) * (1.0 - state_score[right_index[top]])))
            / np.mean(np.sqrt((1.0 - state_score[left_index[eligible]]) * (1.0 - state_score[right_index[eligible]])))
        ),
        "CPAP_same_direction_top_1pct_ratio": float(
            np.mean(np.sqrt((1.0 - cpap_score[left_index[top]]) * (1.0 - cpap_score[right_index[top]])))
            / np.mean(np.sqrt((1.0 - cpap_score[left_index[eligible]]) * (1.0 - cpap_score[right_index[eligible]])))
        ),
    }
    paths = {
        "node": output_dir / "candidate_adipose_node_support.tsv",
        "pair": output_dir / "pair_adipose_support.tsv.gz",
        "metrics": output_dir / "ranking_endpoint_metrics.tsv",
        "node_perm": output_dir / "node_aware_permutation_summary.tsv",
        "subject_perm": output_dir / "subject_randomization_summary.tsv",
        "multiplicity": output_dir / "primary_multiplicity.tsv",
        "blocks": output_dir / "node_matching_blocks.tsv",
    }
    write_tsv(paths["node"], node_output, list(node_output[0]))
    write_gzip_tsv(paths["pair"], pair_output, list(pair_output[0]))
    metric_fields = [
        "endpoint", "model", "rank_field", "metric_scope", "fraction",
        "frozen_rank_threshold", "eligible_top_pairs", "eligible_background_pairs",
        "top_mean_support", "background_mean_support", "support_enrichment_ratio",
        "support_enrichment_difference", "top_both_nodes_positive_fraction",
        "background_both_nodes_positive_fraction", "binary_positive_ratio",
        "normalized_AUEC_0p5_to_5pct",
    ]
    write_tsv(paths["metrics"], metric_rows, metric_fields)
    random_fields = [
        "endpoint", "randomization", "statistic", "observed", "null_mean",
        "null_SD", "null_q025", "null_q50", "null_q975", "one_sided_P",
        "iterations",
    ]
    write_tsv(paths["node_perm"], node_rows, random_fields)
    write_tsv(paths["subject_perm"], subject_rows, random_fields)
    write_tsv(paths["multiplicity"], multiplicity_rows, list(multiplicity_rows[0]))
    write_tsv(paths["blocks"], blocks_output, list(blocks_output[0]))
    output_paths = list(paths.values())
    formal = args.permutations == int(lock["randomization"]["formal_iterations"])
    summary = {
        "created_at_utc": utc_now(),
        "status": "PASS_FORMAL" if formal else "PASS_DEVELOPMENT_ONLY",
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
        "preprocess_manifest_sha256": sha256_file(preprocess_manifest_path),
        "python": sys.version, "numpy": np.__version__,
        "permutations": args.permutations, "seed": seed,
        "available_nodes": int(np.sum(state_available)),
        "eligible_pairs": int(np.sum(eligible)),
        "primary": {
            "state_top_1pct_ratio": primary_observed["state"]["ratio"],
            "state_AUEC": primary_observed["state"]["auec"],
            "state_node_aware_P": primary_p[0], "state_BH_FDR": primary_fdr[0],
            "CPAP_top_1pct_ratio": primary_observed["CPAP"]["ratio"],
            "CPAP_AUEC": primary_observed["CPAP"]["auec"],
            "CPAP_node_aware_P": primary_p[1], "CPAP_BH_FDR": primary_fdr[1],
            "any_primary_claim_allowed": bool(min(primary_fdr) < 0.05),
        },
        "subject_aware": {
            "state_top_1pct_ratio": subject_observed["state_ratio"],
            "state_P": float(subject_rows[0]["one_sided_P"]),
            "CPAP_top_1pct_ratio": subject_observed["CPAP_ratio"],
            "CPAP_P": float(subject_rows[2]["one_sided_P"]),
        },
        "state_covariate_adjusted_sensitivity": {
            "top_1pct_ratio": adjusted_ratio, "AUEC": adjusted_auec,
        },
        "negative_controls": negative_controls,
        "ranking_changed": False,
        "candidate_membership_changed": False,
        "previous_CPAP_validation_read": False,
        "known_PPI_or_pathway_read": False,
        "validation_used_for_training_or_selection": False,
        "runtime_seconds": time.time() - started,
        "next_gate": "Audit server Python 3.7 run, then proceed to participant-level scRNA pseudobulk",
    }
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, summary)
    output_paths.append(summary_path)
    manifest_out = [{
        "path": relative_path(path, root), "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in output_paths]
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(manifest_path, manifest_out, ["path", "bytes", "sha256"])
    print("OSA adipose cross-tissue endpoint: {}".format(summary["status"]))
    print("State ratio={:.4f} P={} FDR={}".format(
        primary_observed["state"]["ratio"], fmt(primary_p[0]), fmt(primary_fdr[0])
    ))
    print("CPAP ratio={:.4f} P={} FDR={}".format(
        primary_observed["CPAP"]["ratio"], fmt(primary_p[1]), fmt(primary_fdr[1])
    ))
    print("Output: {}".format(output_dir))
    print("Frozen ranking unchanged; previous validation/known PPI/pathway read: no")
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
