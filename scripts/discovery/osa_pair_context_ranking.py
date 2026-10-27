#!/usr/bin/env python3
"""Build frozen discovery-only r_OSA and OSA interaction-priority rankings.

Python 3.7 compatible.  The script reads only frozen discovery candidates,
the frozen primary pair registry, frozen disease-independent TRIAGE outputs,
and GSE75097/GSE71356 discovery expression.  It must run before any external
validation or known-PPI/pathway label is read.
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
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
LOCK_REL = "config/osa_pair_context_ranking_lock.json"
CANDIDATE_REL = (
    "results/discovery_preprocess_v2/04_candidates_v1/"
    "candidate_proteins_primary_250.tsv"
)
REGISTRY_REL = (
    "results/discovery_preprocess_v2/07_pair_registry_v1/"
    "pair_registry_primary_250.tsv.gz"
)
MODEL_REL = (
    "results/discovery_preprocess_v2/08_pair_inference_primary_v1/"
    "pair_scores.tsv.gz"
)
GSE75097_EXPR_REL = (
    "results/discovery_preprocess_v2/03_mapping/GSE75097/"
    "protein_expression_primary_samples.tsv.gz"
)
GSE71356_EXPR_REL = (
    "results/discovery_preprocess_v2/03_mapping/GSE71356/"
    "protein_expression_primary_samples.tsv.gz"
)
GSE71356_META_REL = (
    "results/discovery_preprocess_v2/04_discovery_v2/designs/"
    "GSE71356_IH_x_time__samples.tsv"
)
DEFAULT_STAGE = "09_pair_context_ranking_v1"


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
        raise RuntimeError("Attempted to serialize a non-finite value")
    return "{:.10g}".format(value)


def sigmoid(value):
    value = max(-30.0, min(30.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def midrank_percentiles(np, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("Percentile input must be a finite vector")
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


def ordinal_ranks(rows, score_key, extra_keys=None):
    extra_keys = extra_keys or []
    order = sorted(
        range(len(rows)),
        key=lambda index: tuple(
            [-float(rows[index][score_key])]
            + [-float(rows[index][key]) for key in extra_keys]
            + [rows[index]["pair_id"]]
        ),
    )
    ranks = [0] * len(rows)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    return ranks


def load_expression(np, path, wanted):
    rows, fields = read_tsv(path)
    if not fields or fields[0] != "UniProt_accession":
        raise RuntimeError("Unexpected expression schema: " + str(path))
    samples = fields[1:]
    data = {}
    for row in rows:
        accession = row["UniProt_accession"]
        if accession not in wanted:
            continue
        vector = np.asarray([float(row[sample]) for sample in samples], dtype=np.float64)
        if not np.isfinite(vector).all():
            raise RuntimeError("Non-finite expression for " + accession)
        data[accession] = vector
    return data, samples


def subject_center(np, data, samples, metadata_path):
    rows, _ = read_tsv(metadata_path)
    subject = {row["sample_id"]: row["subject_id"] for row in rows}
    if set(samples) != set(subject):
        raise RuntimeError("GSE71356 expression/metadata sample mismatch")
    groups = {}
    for index, sample in enumerate(samples):
        groups.setdefault(subject[sample], []).append(index)
    centered = {}
    for accession, vector in data.items():
        value = vector.copy()
        for indices in groups.values():
            value[indices] -= value[indices].mean()
        centered[accession] = value
    return centered, sorted(groups)


def shrinkage_correlations(np, data, accession_order, n_eff):
    available = [accession for accession in accession_order if accession in data]
    if len(available) < 2:
        raise RuntimeError("Fewer than two proteins available for correlation")
    matrix = np.stack([data[accession] for accession in available], axis=0)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.sqrt((matrix * matrix).sum(axis=1, keepdims=True))
    if (norms <= 1e-12).any():
        raise RuntimeError("Zero-variance protein encountered in correlation")
    unit = matrix / norms
    raw = np.clip(np.dot(unit, unit.T), -1.0, 1.0)
    upper = raw[np.triu_indices(len(available), 1)]
    fisher = np.arctanh(np.clip(upper, -0.999999, 0.999999))
    sampling_variance = 1.0 / (float(n_eff) - 3.0)
    observed_variance = float(np.var(fisher, ddof=1))
    prior_variance = max(0.0, observed_variance - sampling_variance)
    factor = (
        prior_variance / (prior_variance + sampling_variance)
        if prior_variance > 0.0 else 0.0
    )
    shrunk = np.tanh(np.arctanh(np.clip(raw, -0.999999, 0.999999)) * factor)
    np.fill_diagonal(shrunk, 1.0)
    index = {accession: position for position, accession in enumerate(available)}
    metadata = {
        "available_proteins": len(available),
        "expression_samples": int(matrix.shape[1]),
        "n_eff": int(n_eff),
        "off_diagonal_pairs": int(len(upper)),
        "observed_Fisher_z_variance": observed_variance,
        "sampling_variance": sampling_variance,
        "estimated_prior_variance": prior_variance,
        "posterior_shrinkage_factor": factor,
    }
    return raw, shrunk, index, metadata


def corr_value(matrix, index, left, right):
    if left not in index or right not in index:
        return None
    return float(matrix[index[left], index[right]])


def pair_direction(left, right):
    valid = {"positive", "negative"}
    if left not in valid or right not in valid:
        return ""
    return "yes" if left == right else "no"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-stage", default=DEFAULT_STAGE)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    paths = {
        "lock": root / LOCK_REL,
        "candidate": root / CANDIDATE_REL,
        "registry": root / REGISTRY_REL,
        "model": root / MODEL_REL,
        "GSE75097_expression": root / GSE75097_EXPR_REL,
        "GSE71356_expression": root / GSE71356_EXPR_REL,
        "GSE71356_metadata": root / GSE71356_META_REL,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError("Missing {}: {}".format(name, path))
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    expected_hashes = lock["input_hashes"]
    hash_map = {
        "candidate_proteins_primary_250.tsv": paths["candidate"],
        "pair_registry_primary_250.tsv.gz": paths["registry"],
        "pair_scores.tsv.gz": paths["model"],
        "GSE75097_protein_expression_primary.tsv.gz": paths["GSE75097_expression"],
        "GSE71356_protein_expression_primary.tsv.gz": paths["GSE71356_expression"],
        "GSE71356_primary_sample_metadata.tsv": paths["GSE71356_metadata"],
    }
    observed_hashes = {}
    for name, path in hash_map.items():
        observed_hashes[name] = sha256_file(path)
        if observed_hashes[name] != expected_hashes[name]:
            raise RuntimeError("Frozen input hash mismatch: " + name)
    candidates, candidate_fields = read_tsv(paths["candidate"])
    registry, registry_fields = read_tsv(paths["registry"])
    model_rows, model_fields = read_tsv(paths["model"])
    if len(candidates) != 250 or len(registry) != 31125 or len(model_rows) != 31125:
        raise RuntimeError("Unexpected candidate, registry or model row count")
    candidate_map = {row["UniProt_accession"]: row for row in candidates}
    if len(candidate_map) != 250:
        raise RuntimeError("Candidate accessions are not unique")
    for registry_row, model_row in zip(registry, model_rows):
        if any(registry_row[key] != model_row[key] for key in registry_fields):
            raise RuntimeError("Model output does not preserve frozen pair registry")
    wanted = set(candidate_map)
    expression_75097, samples_75097 = load_expression(
        np, paths["GSE75097_expression"], wanted
    )
    expression_71356, samples_71356 = load_expression(
        np, paths["GSE71356_expression"], wanted
    )
    centered_71356, subjects_71356 = subject_center(
        np, expression_71356, samples_71356, paths["GSE71356_metadata"]
    )
    accession_order = [row["UniProt_accession"] for row in candidates]
    raw_75097, shrink_75097, index_75097, corr_meta_75097 = shrinkage_correlations(
        np, expression_75097, accession_order, 46
    )
    raw_71356, shrink_71356, index_71356, corr_meta_71356 = shrinkage_correlations(
        np, centered_71356, accession_order, 12
    )
    print("OSA pair context environment: PASS")
    print("Candidates: {} pairs: {:,}".format(len(candidates), len(registry)))
    print("GSE75097 expression coverage: {}/250 samples={}".format(
        len(expression_75097), len(samples_75097)
    ))
    print("GSE71356 expression coverage: {}/250 samples={} subjects={}".format(
        len(expression_71356), len(samples_71356), len(subjects_71356)
    ))
    print("Validation/known PPI/pathway data read: no")
    if args.check_env:
        return 0

    output_dir = root / "results" / "discovery_preprocess_v2" / args.output_stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace this run")
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    node_rows = []
    for candidate in candidates:
        row = dict(candidate)
        accession = candidate["UniProt_accession"]
        row["GSE75097_expression_available"] = "yes" if accession in expression_75097 else "no"
        row["GSE71356_expression_available"] = "yes" if accession in expression_71356 else "no"
        node_rows.append(row)
    node_fields = candidate_fields + [
        "GSE75097_expression_available", "GSE71356_expression_available",
    ]

    pair_rows = []
    for model_row in model_rows:
        row = dict(model_row)
        left = candidate_map[row["protein_A"]]
        right = candidate_map[row["protein_B"]]
        score_a = float(left["discovery_score"])
        score_b = float(right["discovery_score"])
        adult_a = float(left["adult_OSA_axis"])
        adult_b = float(right["adult_OSA_axis"])
        ih_a = float(left["IH_strength"]) if left["IH_available"] == "yes" else None
        ih_b = float(right["IH_strength"]) if right["IH_available"] == "yes" else None
        row.update({
            "discovery_score_A": fmt(score_a),
            "discovery_score_B": fmt(score_b),
            "adult_OSA_axis_A": fmt(adult_a),
            "adult_OSA_axis_B": fmt(adult_b),
            "OSA_strength_A": left["OSA_strength"],
            "OSA_strength_B": right["OSA_strength"],
            "severity_strength_A": left["severe_strength"],
            "severity_strength_B": right["severe_strength"],
            "AHI_strength_A": left["AHI_strength"],
            "AHI_strength_B": right["AHI_strength"],
            "IH_strength_A": fmt(ih_a),
            "IH_strength_B": fmt(ih_b),
            "min_node_support": fmt(min(score_a, score_b)),
            "geometric_node_support": fmt(math.sqrt(score_a * score_b)),
            "r_OSA": fmt(math.sqrt(score_a * score_b)),
            "adult_OSA_pair_geometric": fmt(math.sqrt(adult_a * adult_b)),
            "IH_pair_geometric_when_both_available": (
                fmt(math.sqrt(ih_a * ih_b))
                if ih_a is not None and ih_b is not None else ""
            ),
            "detectability_A": left["protein_detection_fraction"],
            "detectability_B": right["protein_detection_fraction"],
            "OSA_pair_direction_agreement": pair_direction(
                left["OSA_beta_sign"], right["OSA_beta_sign"]
            ),
            "severity_pair_direction_agreement": pair_direction(
                left["severe_beta_sign"], right["severe_beta_sign"]
            ),
            "AHI_pair_direction_agreement": pair_direction(
                left["AHI_beta_sign"], right["AHI_beta_sign"]
            ),
            "IH_pair_direction_agreement": pair_direction(
                left["IH_beta_sign"], right["IH_beta_sign"]
            ),
        })
        direction_values = [
            row["OSA_pair_direction_agreement"],
            row["severity_pair_direction_agreement"],
            row["AHI_pair_direction_agreement"],
            row["IH_pair_direction_agreement"],
        ]
        available_direction = [value for value in direction_values if value]
        row["pair_direction_available_axes"] = str(len(available_direction))
        row["pair_direction_same_axes"] = str(sum(
            value == "yes" for value in available_direction
        ))
        row["pair_direction_coherence_fraction"] = fmt(
            sum(value == "yes" for value in available_direction)
            / float(len(available_direction))
        )
        left_accession, right_accession = row["protein_A"], row["protein_B"]
        c75097_raw = corr_value(raw_75097, index_75097, left_accession, right_accession)
        c75097_shrunk = corr_value(
            shrink_75097, index_75097, left_accession, right_accession
        )
        c71356_raw = corr_value(raw_71356, index_71356, left_accession, right_accession)
        c71356_shrunk = corr_value(
            shrink_71356, index_71356, left_accession, right_accession
        )
        row["GSE75097_Pearson_r"] = fmt(c75097_raw)
        row["GSE75097_shrunk_r"] = fmt(c75097_shrunk)
        row["GSE71356_subject_centered_Pearson_r"] = fmt(c71356_raw)
        row["GSE71356_subject_centered_shrunk_r"] = fmt(c71356_shrunk)
        if c75097_shrunk is not None and c71356_shrunk is not None:
            combined_z = 0.5 * (
                math.atanh(max(-0.999999, min(0.999999, c75097_shrunk)))
                + math.atanh(max(-0.999999, min(0.999999, c71356_shrunk)))
            )
            row["cross_dataset_shrunk_r_mean_z"] = fmt(math.tanh(combined_z))
            row["coexpression_sign_agreement"] = (
                "yes" if c75097_shrunk * c71356_shrunk >= 0.0 else "no"
            )
        else:
            row["cross_dataset_shrunk_r_mean_z"] = ""
            row["coexpression_sign_agreement"] = ""
        row["coexpression_used_in_primary_r_OSA"] = "no"
        row["known_PPI_status_at_ranking"] = "not_read"
        row["same_discovery_module"] = "not_computed_core_v1"
        gap = float(row["AB_BA_probability_gap"])
        if gap < 0.25:
            row["order_stability_stratum"] = "stable_lt_0.25"
        elif gap < 0.50:
            row["order_stability_stratum"] = "moderate_0.25_to_0.50"
        elif gap < 0.75:
            row["order_stability_stratum"] = "high_0.50_to_0.75"
        else:
            row["order_stability_stratum"] = "very_high_ge_0.75"
        row["truncation_stratum"] = (
            "both_truncated" if row["truncated_A"] == "yes" and row["truncated_B"] == "yes"
            else "one_truncated" if row["truncated_A"] == "yes" or row["truncated_B"] == "yes"
            else "none_truncated"
        )
        row["p_PPI_bidirectional_conservative"] = fmt(min(
            float(row["p_PPI_AB"]), float(row["p_PPI_BA"])
        ))
        g1, g2, g3 = float(row["g_L1"]), float(row["g_L2"]), float(row["g_L3"])
        c1 = float(row["contribution_L1"])
        c2 = float(row["contribution_L2"])
        c3 = float(row["contribution_L3"])
        row["baseline_L3_only"] = row["p_L3"]
        row["baseline_L1_L2"] = fmt(sigmoid((c1 + c2) / max(g1 + g2, 1e-12)))
        row["baseline_L1_L3"] = fmt(sigmoid((c1 + c3) / max(g1 + g3, 1e-12)))
        row["baseline_L2_L3"] = fmt(sigmoid((c2 + c3) / max(g2 + g3, 1e-12)))
        row["baseline_equal_L1_L2_L3"] = fmt(sigmoid((
            float(row["logit_L1"]) + float(row["logit_L2"])
            + float(row["logit_L3"])
        ) / 3.0))
        pair_rows.append(row)

    percentile_sources = {
        "p_PPI_percentile": "p_PPI",
        "r_OSA_percentile": "r_OSA",
        "p_PPI_bidirectional_conservative_percentile": "p_PPI_bidirectional_conservative",
        "baseline_L3_only_percentile": "baseline_L3_only",
        "baseline_L1_L2_percentile": "baseline_L1_L2",
        "baseline_L1_L3_percentile": "baseline_L1_L3",
        "baseline_L2_L3_percentile": "baseline_L2_L3",
        "baseline_equal_L1_L2_L3_percentile": "baseline_equal_L1_L2_L3",
    }
    for output_name, source_name in percentile_sources.items():
        values = [float(row[source_name]) for row in pair_rows]
        percentiles = midrank_percentiles(np, values)
        for row, value in zip(pair_rows, percentiles):
            row[output_name] = fmt(value)
    for row in pair_rows:
        ppi_pct = float(row["p_PPI_percentile"])
        context_pct = float(row["r_OSA_percentile"])
        conservative_pct = float(row["p_PPI_bidirectional_conservative_percentile"])
        row["OSA_interaction_priority_score"] = fmt(math.sqrt(ppi_pct * context_pct))
        row["priority_PPI25_context75"] = fmt(
            ppi_pct ** 0.25 * context_pct ** 0.75
        )
        row["priority_PPI75_context25"] = fmt(
            ppi_pct ** 0.75 * context_pct ** 0.25
        )
        row["priority_bidirectional_conservative"] = fmt(
            math.sqrt(conservative_pct * context_pct)
        )
        row["combined_score_is_physical_PPI_probability"] = "no"

    rank_specs = {
        "rank_p_PPI": ("p_PPI", []),
        "rank_r_OSA": ("r_OSA", []),
        "rank_priority_PPI25_context75": ("priority_PPI25_context75", []),
        "rank_priority_PPI75_context25": ("priority_PPI75_context25", []),
        "rank_priority_bidirectional_conservative": (
            "priority_bidirectional_conservative", []
        ),
        "rank_baseline_L3_only": ("baseline_L3_only", []),
        "rank_baseline_L1_L2": ("baseline_L1_L2", []),
        "rank_baseline_L1_L3": ("baseline_L1_L3", []),
        "rank_baseline_L2_L3": ("baseline_L2_L3", []),
        "rank_baseline_equal_L1_L2_L3": ("baseline_equal_L1_L2_L3", []),
    }
    primary_ranks = ordinal_ranks(
        pair_rows, "OSA_interaction_priority_score", ["p_PPI", "r_OSA"]
    )
    for row, rank in zip(pair_rows, primary_ranks):
        row["OSA_interaction_priority_rank"] = str(rank)
    for output_name, (score_name, extra) in rank_specs.items():
        ranks = ordinal_ranks(pair_rows, score_name, extra)
        for row, rank in zip(pair_rows, ranks):
            row[output_name] = str(rank)
    reported_fractions = lock["frozen_top_sets"]["reported_fractions"]
    for fraction in reported_fractions:
        count = int(math.ceil(len(pair_rows) * float(fraction)))
        label = "top_{:g}pct_primary".format(100.0 * float(fraction)).replace(".", "p")
        for row in pair_rows:
            row[label] = "yes" if int(row["OSA_interaction_priority_rank"]) <= count else "no"

    context_fields = [
        "discovery_score_A", "discovery_score_B",
        "adult_OSA_axis_A", "adult_OSA_axis_B",
        "OSA_strength_A", "OSA_strength_B",
        "severity_strength_A", "severity_strength_B",
        "AHI_strength_A", "AHI_strength_B", "IH_strength_A", "IH_strength_B",
        "min_node_support", "geometric_node_support", "r_OSA",
        "adult_OSA_pair_geometric", "IH_pair_geometric_when_both_available",
        "detectability_A", "detectability_B",
        "OSA_pair_direction_agreement", "severity_pair_direction_agreement",
        "AHI_pair_direction_agreement", "IH_pair_direction_agreement",
        "pair_direction_available_axes", "pair_direction_same_axes",
        "pair_direction_coherence_fraction",
        "GSE75097_Pearson_r", "GSE75097_shrunk_r",
        "GSE71356_subject_centered_Pearson_r",
        "GSE71356_subject_centered_shrunk_r",
        "cross_dataset_shrunk_r_mean_z", "coexpression_sign_agreement",
        "coexpression_used_in_primary_r_OSA", "same_discovery_module",
        "known_PPI_status_at_ranking", "order_stability_stratum",
        "truncation_stratum", "p_PPI_bidirectional_conservative",
        "baseline_L3_only", "baseline_L1_L2", "baseline_L1_L3",
        "baseline_L2_L3", "baseline_equal_L1_L2_L3",
    ]
    percentile_fields = list(percentile_sources.keys())
    score_fields = [
        "OSA_interaction_priority_score", "priority_PPI25_context75",
        "priority_PPI75_context25", "priority_bidirectional_conservative",
        "combined_score_is_physical_PPI_probability",
        "OSA_interaction_priority_rank",
    ] + list(rank_specs.keys())
    top_fields = [
        "top_{:g}pct_primary".format(100.0 * float(value)).replace(".", "p")
        for value in reported_fractions
    ]
    pair_fields = model_fields + context_fields + percentile_fields + score_fields + top_fields
    pair_path = output_dir / "pair_context_scores.tsv.gz"
    node_path = output_dir / "protein_node_evidence.tsv"
    write_gzip_tsv(pair_path, pair_rows, pair_fields)
    write_tsv(node_path, node_rows, node_fields)

    top_count = int(math.ceil(len(pair_rows) * 0.05))
    ranked_rows = sorted(pair_rows, key=lambda row: int(row["OSA_interaction_priority_rank"]))
    top_rows = ranked_rows[:top_count]
    top_fields_output = [
        "OSA_interaction_priority_rank", "pair_id", "protein_A", "gene_A",
        "protein_B", "gene_B", "OSA_interaction_priority_score",
        "p_PPI", "p_PPI_percentile", "r_OSA", "r_OSA_percentile",
        "discovery_score_A", "discovery_score_B",
        "order_stability_stratum", "AB_BA_probability_gap",
        "truncation_stratum", "tri_level_consensus_diagnostic",
        "T_MEB_contribution_conflict", "cross_level_coherence_diagnostic",
        "pair_direction_coherence_fraction", "GSE75097_shrunk_r",
        "GSE71356_subject_centered_shrunk_r",
    ]
    top_path = output_dir / "top_primary_5pct.tsv"
    write_tsv(top_path, top_rows, top_fields_output)

    maximum_formula_error = 0.0
    for row in pair_rows:
        expected_r = math.sqrt(
            float(row["discovery_score_A"]) * float(row["discovery_score_B"])
        )
        expected_priority = math.sqrt(
            float(row["p_PPI_percentile"]) * float(row["r_OSA_percentile"])
        )
        maximum_formula_error = max(
            maximum_formula_error,
            abs(expected_r - float(row["r_OSA"])),
            abs(expected_priority - float(row["OSA_interaction_priority_score"])),
        )
    ranks = sorted(int(row["OSA_interaction_priority_rank"]) for row in pair_rows)
    if ranks != list(range(1, len(pair_rows) + 1)):
        raise RuntimeError("Primary ranks are not contiguous and unique")
    if maximum_formula_error > 2e-9:
        raise RuntimeError("Formula reconstruction error exceeds tolerance")
    if any(row["known_PPI_status_at_ranking"] != "not_read" for row in pair_rows):
        raise RuntimeError("Known-PPI isolation failure")

    summary = {
        "created_at_utc": utc_now(),
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "ranking_lock": {
            "path": LOCK_REL,
            "sha256": sha256_file(paths["lock"]),
        },
        "frozen_input_hashes": observed_hashes,
        "proteins": len(candidates),
        "pairs": len(pair_rows),
        "primary_formula": "sqrt(percentile(p_PPI) * percentile(r_OSA))",
        "r_OSA_formula": "sqrt(discovery_score_A * discovery_score_B)",
        "maximum_formula_reconstruction_error": maximum_formula_error,
        "GSE75097_correlation": corr_meta_75097,
        "GSE71356_subject_centered_correlation": corr_meta_71356,
        "GSE71356_subjects": subjects_71356,
        "top_set_counts": {
            str(value): int(math.ceil(len(pair_rows) * float(value)))
            for value in reported_fractions
        },
        "model_stability_strata": {
            label: sum(row["order_stability_stratum"] == label for row in pair_rows)
            for label in sorted(set(row["order_stability_stratum"] for row in pair_rows))
        },
        "truncation_strata": {
            label: sum(row["truncation_stratum"] == label for row in pair_rows)
            for label in sorted(set(row["truncation_stratum"] for row in pair_rows))
        },
        "outputs": {
            "pair_context_scores": {
                "path": relative_path(pair_path, root),
                "sha256": sha256_file(pair_path),
            },
            "protein_node_evidence": {
                "path": relative_path(node_path, root),
                "sha256": sha256_file(node_path),
            },
            "top_primary_5pct": {
                "path": relative_path(top_path, root),
                "sha256": sha256_file(top_path),
                "rows": len(top_rows),
            },
        },
        "validation_data_read": False,
        "known_PPI_or_pathway_read": False,
        "candidate_or_pair_membership_changed": False,
        "correlation_used_in_primary_ranking": False,
        "combined_score_is_physical_PPI_probability": False,
        "ranking_frozen": True,
        "next_gate": "Run locked baselines/permutations, then one-time CPAP validation",
        "python": sys.version,
        "numpy": np.__version__,
    }
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, summary)
    outputs = [pair_path, node_path, top_path, summary_path]
    manifest_rows = [{
        "path": relative_path(path, root),
        "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in outputs]
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(manifest_path, manifest_rows, ["path", "bytes", "sha256"])
    print("OSA pair context and ranking: PASS")
    print("Pairs ranked: {:,}".format(len(pair_rows)))
    print("Primary top 1%: {} pairs".format(int(math.ceil(len(pair_rows) * 0.01))))
    print("Output: {}".format(output_dir))
    print("Validation/known PPI/pathway data read: no")
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
