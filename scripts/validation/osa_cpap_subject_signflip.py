#!/usr/bin/env python3
"""Subject-aware CPAP reversal robustness for the frozen OSA pair ranking.

Python 3.7 compatible.  Each Monte Carlo draw applies one sign to the entire
post-minus-pre vector of each participant.  This preserves cross-protein
dependence within a participant and never changes candidate membership, pair
membership, or the frozen discovery ranking.

This is a prespecified secondary robustness analysis.  It uses unmoderated
paired t statistics in both the observed statistic and every sign-flip draw so
that the randomization test is internally exact.  The primary moderated-t
endpoint remains in osa_cpap_reversal_endpoint.py.
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
DEFAULT_STAGE = "12_cpap_subject_signflip_triage_osa_v4"
COHORTS = ["GSE133601", "GSE49800"]
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


def trapezoid_auec(np, differences):
    fractions = np.asarray(FRACTIONS, dtype=np.float64)
    increments = fractions[1:] - fractions[:-1]
    areas = 0.5 * (differences[..., 1:] + differences[..., :-1]) * increments
    return np.sum(areas, axis=-1) / (FRACTIONS[-1] - FRACTIONS[0])


def paired_t(np, deltas, signs):
    """Return protein x draw paired t values for subject sign vectors."""
    n_subjects = deltas.shape[1]
    signed_sum = np.dot(deltas, signs.T)
    mean = signed_sum / float(n_subjects)
    sum_squares = np.sum(deltas * deltas, axis=1).reshape(-1, 1)
    variance = (sum_squares - n_subjects * mean * mean) / float(n_subjects - 1)
    variance = np.maximum(variance, 1e-15)
    return mean / np.sqrt(variance / float(n_subjects))


def score_draws(np, oriented_t):
    """Convert available-node t values to draw x node midrank percentiles."""
    draws = oriented_t.shape[1]
    result = np.empty((draws, oriented_t.shape[0]), dtype=np.float64)
    for draw in range(draws):
        result[draw, :] = midrank_percentiles(np, oriented_t[:, draw])
    return result


def pair_statistics(np, node_scores, left_index, right_index, eligible, masks):
    support = np.sqrt(node_scores[:, left_index] * node_scores[:, right_index])
    background = np.mean(support[:, eligible], axis=1)
    top_means = []
    differences = []
    for mask in masks:
        top_mean = np.mean(support[:, mask], axis=1)
        top_means.append(top_mean)
        differences.append(top_mean - background)
    difference_matrix = np.stack(differences, axis=1)
    ratios = top_means[1] / np.maximum(background, 1e-12)
    auec = trapezoid_auec(np, difference_matrix)
    return ratios, auec


def null_summary(np, dataset, statistic, observed, null, iterations):
    return {
        "dataset": dataset,
        "randomization_unit": "participant_complete_delta_vector",
        "statistic": statistic,
        "observed": fmt(observed),
        "null_mean": fmt(np.mean(null)),
        "null_SD": fmt(np.std(null, ddof=1)),
        "null_q025": fmt(np.quantile(null, 0.025)),
        "null_q50": fmt(np.quantile(null, 0.50)),
        "null_q975": fmt(np.quantile(null, 0.975)),
        "one_sided_signflip_P": fmt(
            (1.0 + float(np.sum(null >= observed))) / (iterations + 1.0)
        ),
        "iterations": str(iterations),
    }


def verify_preprocess_manifest(root, preprocess_dir):
    manifest_path = preprocess_dir / "output_manifest.tsv"
    rows, _ = read_tsv(manifest_path)
    failures = []
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            failures.append("missing:" + row["path"])
        elif sha256_file(path) != row["sha256"]:
            failures.append("hash:" + row["path"])
    if failures:
        raise RuntimeError(
            "CPAP preprocessing manifest verification failed: "
            + "; ".join(failures[:5])
        )
    return manifest_path


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
    lock_path = root / LOCK_REL
    handoff_path = root / HANDOFF_REL
    ranking_path = root / RANKING_REL
    candidate_path = root / CANDIDATE_REL
    preprocess_dir = root / "results" / "discovery_preprocess_v2" / args.preprocess_stage
    preprocess_summary_path = preprocess_dir / "run_summary.json"
    required = [lock_path, handoff_path, ranking_path, candidate_path, preprocess_summary_path]
    for path in required:
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
    if sha256_file(candidate_path) != lock["input_hashes"]["candidate_proteins_primary_250.tsv"]:
        raise RuntimeError("Frozen candidate hash mismatch")
    preprocess_summary = json.loads(preprocess_summary_path.read_text(encoding="utf-8"))
    if preprocess_summary.get("status") != "PASS":
        raise RuntimeError("CPAP preprocessing has not passed")
    if preprocess_summary["validation_lock"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("CPAP preprocessing used a different validation lock")
    preprocess_manifest_path = verify_preprocess_manifest(root, preprocess_dir)
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
    ranks = np.asarray([
        parse_rank(row["OSA_interaction_priority_rank"]) for row in ranking_rows
    ], dtype=np.int64)
    if sorted(ranks.tolist()) != list(range(1, 31126)):
        raise RuntimeError("Frozen primary rank is not a complete permutation")
    candidate_order = [row["UniProt_accession"] for row in candidate_rows]
    candidate_index = {
        accession: index for index, accession in enumerate(candidate_order)
    }
    left_index = np.asarray([
        candidate_index[row["protein_A"]] for row in ranking_rows
    ], dtype=np.int64)
    right_index = np.asarray([
        candidate_index[row["protein_B"]] for row in ranking_rows
    ], dtype=np.int64)
    print("OSA CPAP subject-aware sign-flip environment: PASS")
    print("Frozen pairs: 31,125; candidates: 250")
    print("Permutations: {} seed={}".format(
        args.permutations, lock["permutation"]["seed"]
    ))
    print("Validation data used for testing only; ranking changed: no")
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
    outputs = []
    cohort_results = {}
    summary_rows = []
    for cohort_number, dataset in enumerate(COHORTS):
        cohort_preprocess = preprocess_dir / dataset
        deltas_path = cohort_preprocess / "candidate_participant_deltas.tsv.gz"
        evidence_path = cohort_preprocess / "candidate_validation_evidence.tsv"
        delta_rows, delta_fields = read_tsv(deltas_path)
        evidence_rows, _ = read_tsv(evidence_path)
        delta_map = {row["UniProt_accession"]: row for row in delta_rows}
        evidence_map = {row["UniProt_accession"]: row for row in evidence_rows}
        subject_fields = [field for field in delta_fields if field.startswith("subject_")]
        available_accessions = [
            accession for accession in candidate_order
            if delta_map[accession]["validation_available"] == "yes"
        ]
        available_index = {
            accession: index for index, accession in enumerate(available_accessions)
        }
        deltas = np.asarray([[
            float(delta_map[accession][field]) for field in subject_fields
        ] for accession in available_accessions], dtype=np.float64)
        discovery_sign = np.asarray([
            1.0 if evidence_map[accession]["discovery_OSA_beta_sign"] == "positive"
            else -1.0
            for accession in available_accessions
        ], dtype=np.float64).reshape(-1, 1)
        available_vector = np.asarray([
            accession in available_index for accession in candidate_order
        ], dtype=bool)
        eligible = available_vector[left_index] & available_vector[right_index]
        masks = []
        for fraction in FRACTIONS:
            threshold = int(math.ceil(len(ranking_rows) * fraction))
            masks.append(eligible & (ranks <= threshold))
        observed_signs = np.ones((1, len(subject_fields)), dtype=np.float64)
        observed_t = paired_t(np, deltas, observed_signs)
        observed_scores_available = score_draws(
            np, -discovery_sign * observed_t
        )
        observed_scores = np.zeros((1, 250), dtype=np.float64)
        for accession, source_index in available_index.items():
            observed_scores[0, candidate_index[accession]] = (
                observed_scores_available[0, source_index]
            )
        observed_ratio, observed_auec = pair_statistics(
            np, observed_scores, left_index, right_index, eligible, masks
        )
        null_ratio = np.empty(args.permutations, dtype=np.float64)
        null_auec = np.empty(args.permutations, dtype=np.float64)
        rng = np.random.RandomState(seed + 100 + cohort_number)
        completed = 0
        while completed < args.permutations:
            size = min(args.batch_size, args.permutations - completed)
            signs = rng.choice(
                np.asarray([-1.0, 1.0]), size=(size, len(subject_fields))
            )
            permuted_t = paired_t(np, deltas, signs)
            available_scores = score_draws(
                np, -discovery_sign * permuted_t
            )
            scores = np.zeros((size, 250), dtype=np.float64)
            for accession, source_index in available_index.items():
                scores[:, candidate_index[accession]] = (
                    available_scores[:, source_index]
                )
            ratios, auec = pair_statistics(
                np, scores, left_index, right_index, eligible, masks
            )
            null_ratio[completed:completed + size] = ratios
            null_auec[completed:completed + size] = auec
            completed += size
            if completed % 1000 == 0 or completed == args.permutations:
                print("[{} sign-flip] {}/{}".format(
                    dataset, completed, args.permutations
                ), flush=True)
        cohort_rows = [
            null_summary(
                np, dataset, "top_1pct_support_enrichment_ratio_unmoderated_t",
                observed_ratio[0], null_ratio, args.permutations,
            ),
            null_summary(
                np, dataset, "normalized_AUEC_0p5_to_5pct_unmoderated_t",
                observed_auec[0], null_auec, args.permutations,
            ),
        ]
        summary_rows.extend(cohort_rows)
        null_rows = []
        for iteration in range(args.permutations):
            null_rows.append({
                "dataset": dataset,
                "iteration": str(iteration + 1),
                "top_1pct_support_enrichment_ratio": fmt(null_ratio[iteration]),
                "normalized_AUEC_0p5_to_5pct": fmt(null_auec[iteration]),
            })
        null_path = output_dir / (dataset + "_subject_signflip_null.tsv.gz")
        write_gzip_tsv(null_path, null_rows, list(null_rows[0]))
        outputs.append(null_path)
        cohort_results[dataset] = {
            "subjects": len(subject_fields),
            "available_nodes": len(available_accessions),
            "eligible_pairs": int(np.sum(eligible)),
            "observed_top_1pct_enrichment_ratio_unmoderated_t": float(observed_ratio[0]),
            "top_1pct_one_sided_signflip_P": float(cohort_rows[0]["one_sided_signflip_P"]),
            "observed_AUEC_unmoderated_t": float(observed_auec[0]),
            "AUEC_one_sided_signflip_P": float(cohort_rows[1]["one_sided_signflip_P"]),
            "null_log_ratio_SD": float(np.std(np.log(np.maximum(null_ratio, 1e-12)), ddof=1)),
        }
        print("[{}] subjects={} top1_ratio={:.4f} P={} AUEC={:.5f} P={}".format(
            dataset, len(subject_fields), observed_ratio[0],
            cohort_rows[0]["one_sided_signflip_P"], observed_auec[0],
            cohort_rows[1]["one_sided_signflip_P"],
        ))
    summary_path = output_dir / "subject_signflip_summary.tsv"
    write_tsv(summary_path, summary_rows, list(summary_rows[0]))
    outputs.append(summary_path)
    effects = np.asarray([
        math.log(max(
            cohort_results[dataset]["observed_top_1pct_enrichment_ratio_unmoderated_t"],
            1e-12,
        )) for dataset in COHORTS
    ])
    standard_errors = np.asarray([
        cohort_results[dataset]["null_log_ratio_SD"] for dataset in COHORTS
    ])
    weights = 1.0 / np.maximum(standard_errors * standard_errors, 1e-12)
    pooled = float(np.sum(weights * effects) / np.sum(weights))
    pooled_se = float(math.sqrt(1.0 / np.sum(weights)))
    pooled_p = 0.5 * math.erfc((pooled / pooled_se) / math.sqrt(2.0))
    q_value = float(np.sum(weights * (effects - pooled) ** 2))
    i2 = max(0.0, (q_value - 1.0) / max(q_value, 1e-12))
    directions_consistent = bool(np.all(effects > 0.0) or np.all(effects < 0.0))
    meta_row = {
        "scope": "fixed_effect_meta_subject_signflip_robustness",
        "pooled_log_enrichment_ratio": fmt(pooled),
        "pooled_enrichment_ratio": fmt(math.exp(pooled)),
        "SE_from_signflip_null": fmt(pooled_se),
        "one_sided_P_normal_approximation": fmt(pooled_p),
        "heterogeneity_Q": fmt(q_value),
        "I2": fmt(i2),
        "direction_consistent": "yes" if directions_consistent else "no",
        "pooled_claim_allowed": (
            "yes" if directions_consistent and pooled > 0.0 and pooled_p < 0.05
            else "no"
        ),
    }
    meta_path = output_dir / "subject_signflip_meta.tsv"
    write_tsv(meta_path, [meta_row], list(meta_row))
    outputs.append(meta_path)
    formal = args.permutations == int(lock["permutation"]["formal_iterations"])
    summary = {
        "created_at_utc": utc_now(),
        "status": "PASS_FORMAL_SECONDARY" if formal else "PASS_DEVELOPMENT_ONLY",
        "analysis_role": "prespecified_subject_aware_secondary_robustness",
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
        "preprocess_manifest": {
            "path": relative_path(preprocess_manifest_path, root),
            "sha256": sha256_file(preprocess_manifest_path),
        },
        "permutations": args.permutations,
        "seed_base": seed,
        "randomization": (
            "one common sign per participant applied to the complete "
            "available-candidate post-minus-pre vector"
        ),
        "node_statistic": "unmoderated paired t in observed and null draws",
        "cohorts": cohort_results,
        "meta": meta_row,
        "ranking_changed": False,
        "validation_used_for_training_or_selection": False,
        "known_PPI_or_pathway_read": False,
        "runtime_seconds": time.time() - started,
        "python": sys.version,
        "numpy": np.__version__,
    }
    run_summary_path = output_dir / "run_summary.json"
    atomic_json(run_summary_path, summary)
    outputs.append(run_summary_path)
    manifest_rows = [{
        "path": relative_path(path, root),
        "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in outputs]
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(manifest_path, manifest_rows, ["path", "bytes", "sha256"])
    print("OSA CPAP subject-aware sign-flip: {}".format(summary["status"]))
    print("Output: {}".format(output_dir))
    print("Frozen ranking unchanged; validation used for training/selection: no")
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
