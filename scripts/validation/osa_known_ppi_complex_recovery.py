#!/usr/bin/env python3
"""Stage 19: locked known-PPI and curated-complex recovery sanity checks.

Python 3.7 compatible.  The script evaluates the already frozen 31,125-pair
ranking against independently prepared IntAct and Complex Portal references.
Pairs present anywhere in the TUnA model data are excluded from both positive
and control pools for the primary endpoints.  Known labels never alter the
candidate universe, model, scores, or ranks.
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


SCRIPT_VERSION = "1.3.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
LOCK_REL = "config/osa_known_ppi_complex_lock.json"
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
CANDIDATE_SHA256 = "90fd8420b22a4dbb155d7628858701a42c33bf70b76288322747ecb86030027d"
REFERENCE_DIR_REL = "annotations/network_sanity_triage_osa_v4"
DEFAULT_STAGE = "19_known_ppi_complex_recovery_triage_osa_v4"


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


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relative_path(root, path):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def fmt(value):
    if value is None or value == "":
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return "{:.10g}".format(value)


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temp), str(path))


def write_tsv(path, rows, fields):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields,
            extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temp), str(path))


def write_gzip_tsv(path, rows, fields):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, delimiter="\t", fieldnames=fields,
                    extrasaction="ignore", lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(str(temp), str(path))


def read_tsv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def log_combination(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def hypergeom_upper_tail(observed, population, successes, draws):
    upper = min(successes, draws)
    if observed > upper:
        return 0.0
    denominator = log_combination(population, draws)
    values = []
    for k in range(observed, upper + 1):
        if draws - k > population - successes:
            continue
        values.append(
            log_combination(successes, k)
            + log_combination(population - successes, draws - k)
            - denominator
        )
    if not values:
        return 0.0
    maximum = max(values)
    return min(1.0, math.exp(maximum) * sum(math.exp(x - maximum) for x in values))


def bh_adjust(values):
    if not values:
        return []
    values = [float(x) for x in values]
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [1.0] * len(values)
    running = 1.0
    for reverse in range(len(values) - 1, -1, -1):
        index = order[reverse]
        running = min(running, values[index] * len(values) / float(reverse + 1))
        result[index] = min(1.0, running)
    return result


def verify_reference_preparation(root, reference_dir, lock_path, ranking_path):
    manifest_path = reference_dir / "reference_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Missing reference_manifest.json; run reference preparation first")
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Reference preparation status is not PASS")
    if summary.get("lock", {}).get("sha256") != sha256_file(lock_path):
        raise RuntimeError("Reference preparation used a different lock")
    if summary.get("frozen_ranking_sha256") != sha256_file(ranking_path):
        raise RuntimeError("Reference preparation used a different frozen ranking")
    if summary.get("validation_outcomes_read") is not False:
        raise RuntimeError("Reference preparation isolation flag failed")
    for relative, expected in summary.get("artifact_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError("Reference artifact verification failed: " + relative)
    return summary


def load_detectability(path):
    values = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = ["UniProt_accession", "protein_detection_fraction"]
        missing = [x for x in required if x not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("Frozen candidate table lacks: " + ",".join(missing))
        for row in reader:
            accession = row["UniProt_accession"]
            if accession in values:
                raise RuntimeError("Duplicate candidate accession: " + accession)
            value = float(row["protein_detection_fraction"])
            if not math.isfinite(value):
                raise RuntimeError("Non-finite candidate detectability: " + accession)
            values[accession] = value
    if len(values) != 250:
        raise RuntimeError("Frozen candidate detectability table is not primary-250")
    return values


def load_ranking(path, ranking_models, detectability):
    rows = []
    node = {}
    with gzip.open(str(path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = [
            "pair_id", "protein_A", "protein_B", "gene_A", "gene_B",
            "length_A", "length_B", "triage_osa_rank_primary_250",
        ] + list(ranking_models.values())
        missing = [x for x in required if x not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("Frozen ranking lacks: " + ",".join(missing))
        for source in reader:
            row = {x: source[x] for x in required}
            rows.append(row)
            for side in ("A", "B"):
                accession = source["protein_" + side]
                if accession not in detectability:
                    raise RuntimeError("Candidate detectability missing: " + accession)
                values = (
                    source["gene_" + side], int(float(source["length_" + side])),
                    float(detectability[accession]),
                )
                if accession in node and node[accession] != values:
                    raise RuntimeError("Inconsistent node covariates: " + accession)
                node[accession] = values
    if len(rows) != 31125 or len(node) != 250:
        raise RuntimeError("Frozen ranking universe is not 250/31,125")
    for field in ranking_models.values():
        ranks = sorted(parse_rank(row[field]) for row in rows)
        if ranks != list(range(1, 31126)):
            raise RuntimeError("Invalid frozen rank field: " + field)
    return rows, node


def quantile_bins(values, bin_count):
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), i))
    result = [0] * len(values)
    for position, index in enumerate(order):
        result[index] = min(bin_count - 1, int(position * bin_count / float(len(values))))
    return result


def auec(fractions, enrichments, lower, upper):
    points = sorted((f, e) for f, e in zip(fractions, enrichments) if lower <= f <= upper)
    if len(points) < 2:
        return float("nan")
    area = 0.0
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        area += (x1 - x0) * 0.5 * (y0 + y1)
    return area / float(points[-1][0] - points[0][0])


def build_pair_strata(rows, node, degrees, lock):
    accessions = sorted(node)
    degree_values = [float(degrees.get(x, 0)) for x in accessions]
    detect_values = [float(node[x][2]) for x in accessions]
    length_values = [float(node[x][1]) for x in accessions]
    cfg = lock["matched_control"]
    degree_bins = quantile_bins(degree_values, int(cfg["external_degree_quantile_bins"]))
    detect_bins = quantile_bins(detect_values, int(cfg["detectability_quantile_bins"]))
    length_bins = quantile_bins(length_values, int(cfg["length_quantile_bins"]))
    covariate = {}
    for index, accession in enumerate(accessions):
        covariate[accession] = (degree_bins[index], detect_bins[index], length_bins[index])
    keys = []
    for row in rows:
        a = covariate[row["protein_A"]]
        b = covariate[row["protein_B"]]
        degree = tuple(sorted((a[0], b[0])))
        detect = tuple(sorted((a[1], b[1])))
        length = tuple(sorted((a[2], b[2])))
        keys.append({
            "degree+detectability+length": degree + detect + length,
            "degree+detectability": degree + detect,
            "degree": degree,
            "universe": (0,),
        })
    return keys, covariate


def reference_definitions():
    return [
        ("IntAct_physical_training_excluded", "intact_physical", True),
        ("ComplexPortal_experimental_training_excluded", "complexportal_experimental", True),
        ("IntAct_all_training_excluded", "intact_all_positive", True),
        ("ComplexPortal_all_curated_training_excluded", "complexportal_all_curated", True),
        ("IntAct_physical_overlap_included", "intact_physical", False),
        ("ComplexPortal_experimental_overlap_included", "complexportal_experimental", False),
    ]


def matched_test(np, scores, positive_indices, eligible_indices, label, strata,
                 fallback_order, iterations, seed, model_names):
    positive_set = set(positive_indices)
    negative = [i for i in eligible_indices if i not in positive_set and not label[i]]
    pools = {}
    for level in fallback_order:
        current = {}
        for index in negative:
            current.setdefault(strata[index][level], []).append(index)
        pools[level] = current
    selected_pools = []
    fallback_counts = dict((level, 0) for level in fallback_order)
    for index in positive_indices:
        selected = None
        selected_level = ""
        for level in fallback_order:
            candidates = pools[level].get(strata[index][level], [])
            if candidates:
                selected = np.asarray(candidates, dtype=np.int64)
                selected_level = level
                break
        if selected is None:
            raise RuntimeError("No matched non-reference pool for pair index {}".format(index))
        selected_pools.append(selected)
        fallback_counts[selected_level] += 1
    positive = np.asarray(positive_indices, dtype=np.int64)
    reference_mean = scores[positive].mean(axis=0)
    rng = np.random.RandomState(seed)
    control_sum = np.zeros((iterations, scores.shape[1]), dtype=np.float64)
    for pool in selected_pools:
        sampled = pool[rng.randint(0, len(pool), size=iterations)]
        control_sum += scores[sampled]
    control_mean = control_sum / float(len(selected_pools))
    advantages = reference_mean - control_mean.mean(axis=0)
    p_values = (
        1.0 + (control_mean >= reference_mean.reshape(1, -1)).sum(axis=0)
    ) / float(iterations + 1)
    full_index = model_names.index("full_OSA_priority")
    comparisons = []
    for index, model in enumerate(model_names):
        if index == full_index:
            continue
        observed = reference_mean[full_index] - reference_mean[index]
        null = control_mean[:, full_index] - control_mean[:, index]
        p_value = (1.0 + float((null >= observed).sum())) / float(iterations + 1)
        comparisons.append({
            "comparator": model,
            "observed_reference_full_minus_comparator": observed,
            "mean_matched_control_full_minus_comparator": float(null.mean()),
            "one_sided_permutation_p": p_value,
        })
    return reference_mean, control_mean.mean(axis=0), advantages, p_values, comparisons, fallback_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--reference-dir", default=REFERENCE_DIR_REL)
    parser.add_argument("--run-id", default=DEFAULT_STAGE)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("numpy is required")

    root = Path(args.root).resolve()
    lock_path = root / LOCK_REL
    handoff_path = root / HANDOFF_REL
    ranking_path = root / RANKING_REL
    candidate_path = root / CANDIDATE_REL
    reference_dir = Path(args.reference_dir)
    if not reference_dir.is_absolute():
        reference_dir = root / reference_dir
    output_base = root / "results" / "discovery_preprocess_v2" / args.run_id
    if (
        not lock_path.is_file() or not handoff_path.is_file()
        or not ranking_path.is_file() or not candidate_path.is_file()
    ):
        raise RuntimeError("Missing Stage 19 lock or frozen Stage 9 ranking")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff["ranking"]["primary_250"] != RANKING_REL:
        raise RuntimeError("TRIAGE-OSA handoff ranking path mismatch")
    if handoff["ranking"]["primary_250_sha256"] != RANKING_SHA256:
        raise RuntimeError("TRIAGE-OSA handoff ranking hash mismatch")
    if handoff["prespecified_endpoint_locks"]["known_ppi_complex"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("TRIAGE-OSA handoff known-PPI lock mismatch")
    if sha256_file(ranking_path) != RANKING_SHA256:
        raise RuntimeError("Frozen ranking hash mismatch")
    if sha256_file(candidate_path) != CANDIDATE_SHA256:
        raise RuntimeError("Frozen candidate detectability table hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    ranking_models = lock["ranking_models"]
    detectability = load_detectability(candidate_path)
    rows, node = load_ranking(ranking_path, ranking_models, detectability)
    print("OSA known-PPI/complex recovery environment: PASS")
    print("Python: {}".format(sys.version))
    print("numpy: {}".format(np.__version__))
    print("Frozen universe: proteins={} pairs={}".format(len(node), len(rows)))
    print("Reference directory: {}".format(reference_dir))
    print("Validation outcomes read: no")
    if args.check_env:
        return 0
    reference_summary = verify_reference_preparation(
        root, reference_dir, lock_path, ranking_path
    )
    iterations = int(lock["permutation"]["development_iterations"]) if args.smoke else args.permutations
    if iterations <= 0:
        raise RuntimeError("permutations must be positive")
    output_dir = output_base / "smoke" if args.smoke else output_base
    if output_dir.exists() and args.force:
        shutil.rmtree(str(output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Output exists; use --force: " + str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    intact = dict((x["pair_id"], x) for x in read_tsv(
        reference_dir / "intact_human_candidate_pairs.tsv.gz"
    ))
    complexes = dict((x["pair_id"], x) for x in read_tsv(
        reference_dir / "complexportal_candidate_pairs.tsv.gz"
    ))
    overlap = set(x["pair_id"] for x in read_tsv(
        reference_dir / "tuna_pair_overlap.tsv.gz"
    ))
    degree_rows = read_tsv(reference_dir / "intact_candidate_degree.tsv")
    degrees = dict(
        (x["UniProt_accession"], int(x["intact_all_unique_human_protein_partners"]))
        for x in degree_rows
    )
    labels = dict((name, np.zeros(len(rows), dtype=bool)) for name, _, _ in reference_definitions())
    label_rows = []
    for index, row in enumerate(rows):
        pid = row["pair_id"]
        irow = intact.get(pid, {})
        crow = complexes.get(pid, {})
        training = pid in overlap
        base = {
            "pair_id": pid, "protein_A": row["protein_A"], "protein_B": row["protein_B"],
            "gene_A": row["gene_A"], "gene_B": row["gene_B"],
            "tuna_training_overlap": "yes" if training else "no",
            "intact_all_positive": irow.get("intact_all_positive", "no"),
            "intact_physical": irow.get("intact_physical", "no"),
            "complexportal_all_curated": crow.get("complexportal_all_curated", "no"),
            "complexportal_experimental": crow.get("complexportal_experimental", "no"),
            "complex_accessions": crow.get("complex_accessions", ""),
        }
        for name, field, exclude_training in reference_definitions():
            present = base[field] == "yes" and not (exclude_training and training)
            labels[name][index] = present
            base[name] = "yes" if present else "no"
        label_rows.append(base)
    write_gzip_tsv(output_dir / "pair_reference_labels.tsv.gz", label_rows, list(label_rows[0]))

    model_names = list(ranking_models.keys())
    scores = np.empty((len(rows), len(model_names)), dtype=np.float64)
    for column, model in enumerate(model_names):
        field = ranking_models[model]
        scores[:, column] = [
            1.0 - (parse_rank(row[field]) - 0.5) / float(len(rows)) for row in rows
        ]
    strata, node_covariate = build_pair_strata(rows, node, degrees, lock)
    fractions = [float(x) for x in lock["endpoints"]["reported_fractions"]]
    minimum = int(lock["endpoints"]["minimum_primary_reference_pairs_for_formal_endpoint"])
    primary_names = set(lock["endpoints"]["primary_references"])
    fallback_order = list(lock["matched_control"]["fallback_order"])
    metric_rows = []
    matched_rows = []
    comparison_rows = []
    coverage_rows = []
    seed = int(lock["permutation"]["seed"])
    for ref_index, (reference, _, exclude_training) in enumerate(reference_definitions()):
        label = labels[reference]
        eligible = np.asarray([
            index for index, row in enumerate(rows)
            if not (exclude_training and row["pair_id"] in overlap)
        ], dtype=np.int64)
        positives = np.flatnonzero(label)
        status = "EVALUABLE" if len(positives) >= minimum else "LIMITED_REFERENCE_COVERAGE"
        coverage_rows.append({
            "reference": reference, "primary_or_sensitivity": "primary" if reference in primary_names else "sensitivity",
            "eligible_pairs": len(eligible), "reference_pairs": len(positives),
            "training_overlap_pairs_excluded": len(rows) - len(eligible), "status": status,
        })
        prevalence = len(positives) / float(len(eligible)) if len(eligible) else float("nan")
        metric_start = len(metric_rows)
        for model_column, model in enumerate(model_names):
            ranks = np.asarray([parse_rank(row[ranking_models[model]]) for row in rows], dtype=np.int64)
            enrichments = []
            for fraction in fractions:
                top_count = int(math.ceil(len(rows) * fraction))
                top_mask = ranks <= top_count
                top_eligible = np.intersect1d(np.flatnonzero(top_mask), eligible, assume_unique=False)
                observed = int(label[top_eligible].sum())
                top_rate = observed / float(len(top_eligible)) if len(top_eligible) else float("nan")
                enrichment = top_rate / prevalence if prevalence > 0 else float("nan")
                enrichments.append(enrichment)
                p_value = hypergeom_upper_tail(
                    observed, len(eligible), len(positives), len(top_eligible)
                ) if len(positives) else 1.0
                metric_rows.append({
                    "reference": reference, "reference_status": status, "model": model,
                    "fraction": fmt(fraction), "top_eligible_pairs": len(top_eligible),
                    "top_reference_pairs": observed, "reference_pairs": len(positives),
                    "eligible_pairs": len(eligible), "top_enrichment_ratio": fmt(enrichment),
                    "one_sided_hypergeometric_p": fmt(p_value),
                    "BH_FDR_within_reference": "", "auec_0p5_to_5pct": "",
                })
            current_auec = auec(
                fractions, enrichments,
                float(lock["endpoints"]["auec_min_fraction"]),
                float(lock["endpoints"]["auec_max_fraction"]),
            )
            for row_out in metric_rows[-len(fractions):]:
                row_out["auec_0p5_to_5pct"] = fmt(current_auec)
        metric_subset = metric_rows[metric_start:]
        metric_fdr = bh_adjust([x["one_sided_hypergeometric_p"] for x in metric_subset])
        for index, row_out in enumerate(metric_subset):
            row_out["BH_FDR_within_reference"] = fmt(metric_fdr[index])
        if len(positives):
            ref_mean, control_mean, advantage, pvals, comparisons, fallback = matched_test(
                np, scores, positives.tolist(), eligible.tolist(), label, strata,
                fallback_order, iterations, seed + 1000 * ref_index, model_names,
            )
            adjusted = bh_adjust(pvals.tolist())
            for index, model in enumerate(model_names):
                matched_rows.append({
                    "reference": reference, "reference_status": status, "model": model,
                    "reference_pairs": len(positives), "iterations": iterations,
                    "reference_mean_rank_percentile": fmt(ref_mean[index]),
                    "matched_control_mean_rank_percentile": fmt(control_mean[index]),
                    "mean_percentile_advantage": fmt(advantage[index]),
                    "one_sided_matched_permutation_p": fmt(pvals[index]),
                    "BH_FDR_within_reference": fmt(adjusted[index]),
                    "exact_degree_detectability_length_matches": fallback.get("degree+detectability+length", 0),
                    "degree_detectability_fallbacks": fallback.get("degree+detectability", 0),
                    "degree_only_fallbacks": fallback.get("degree", 0),
                    "universe_fallbacks": fallback.get("universe", 0),
                })
            comp_p = bh_adjust([x["one_sided_permutation_p"] for x in comparisons])
            for index, item in enumerate(comparisons):
                comparison_rows.append({
                    "reference": reference, "reference_status": status,
                    "full_model": "full_OSA_priority", "comparator": item["comparator"],
                    "reference_pairs": len(positives), "iterations": iterations,
                    "observed_reference_full_minus_comparator": fmt(item["observed_reference_full_minus_comparator"]),
                    "mean_matched_control_full_minus_comparator": fmt(item["mean_matched_control_full_minus_comparator"]),
                    "one_sided_permutation_p": fmt(item["one_sided_permutation_p"]),
                    "BH_FDR_within_reference": fmt(comp_p[index]),
                })
    write_tsv(output_dir / "reference_coverage.tsv", coverage_rows, list(coverage_rows[0]))
    write_tsv(output_dir / "ranking_recovery_metrics.tsv", metric_rows, list(metric_rows[0]))
    write_tsv(output_dir / "matched_control_permutation.tsv", matched_rows, list(matched_rows[0]))
    write_tsv(output_dir / "full_vs_baseline_comparisons.tsv", comparison_rows, list(comparison_rows[0]))
    node_rows = []
    for accession in sorted(node):
        cov = node_covariate[accession]
        node_rows.append({
            "UniProt_accession": accession, "gene_symbol": node[accession][0],
            "length": node[accession][1], "detectability": fmt(node[accession][2]),
            "intact_all_degree": degrees.get(accession, 0),
            "degree_bin": cov[0], "detectability_bin": cov[1], "length_bin": cov[2],
        })
    write_tsv(output_dir / "node_matching_strata.tsv", node_rows, list(node_rows[0]))

    formal = iterations == int(lock["permutation"]["formal_iterations"]) and not args.smoke
    primary_coverage = [x for x in coverage_rows if x["primary_or_sensitivity"] == "primary"]
    all_primary_evaluable = all(x["status"] == "EVALUABLE" for x in primary_coverage)
    if formal and all_primary_evaluable:
        status = "PASS_FORMAL"
    elif formal:
        status = "PASS_FORMAL_PARTIAL_REFERENCE_COVERAGE"
    else:
        status = "PASS_DEVELOPMENT"
    headline = [
        row for row in metric_rows
        if row["model"] == "full_OSA_priority" and row["fraction"] == fmt(lock["endpoints"]["headline_fraction"])
    ]
    conclusion_rows = []
    for row in headline:
        conclusion_rows.append({
            "reference": row["reference"], "status": row["reference_status"],
            "top1pct_reference_pairs": row["top_reference_pairs"],
            "top1pct_eligible_pairs": row["top_eligible_pairs"],
            "top1pct_enrichment_ratio": row["top_enrichment_ratio"],
            "one_sided_hypergeometric_p": row["one_sided_hypergeometric_p"],
            "allowed_interpretation": "Generic interaction/complex recovery sanity check only; not OSA-specific physical validation.",
        })
    write_tsv(output_dir / "direct_conclusions.tsv", conclusion_rows, list(conclusion_rows[0]))
    summary_text = "# Stage 19 known-PPI and complex recovery\n\n- Status: **{}**\n- Frozen proteins/pairs: **250 / 31,125**.\n- Permutations: **{}**.\n- Candidate membership or ranking changed: **no**.\n- Validation outcomes used: **no**.\n\nThis is a generic physical-interaction/complex sanity check. Training-overlap-excluded results are primary; overlap-inclusive results are sensitivity analyses. It cannot establish OSA-specific binding.\n".format(status, iterations)
    (output_dir / "known_ppi_complex_summary.md").write_text(summary_text, encoding="utf-8")
    artifact_paths = [
        output_dir / "pair_reference_labels.tsv.gz",
        output_dir / "reference_coverage.tsv",
        output_dir / "ranking_recovery_metrics.tsv",
        output_dir / "matched_control_permutation.tsv",
        output_dir / "full_vs_baseline_comparisons.tsv",
        output_dir / "node_matching_strata.tsv",
        output_dir / "direct_conclusions.tsv",
        output_dir / "known_ppi_complex_summary.md",
    ]
    manifest_rows = [{
        "path": relative_path(root, path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in artifact_paths]
    write_tsv(output_dir / "output_manifest.tsv", manifest_rows, list(manifest_rows[0]))
    summary = {
        "status": status, "script_version": SCRIPT_VERSION,
        "script": relative_path(root, Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": utc_now(), "permutations": iterations,
        "lock": {"path": LOCK_REL, "sha256": sha256_file(lock_path)},
        "external_validation_handoff": {
            "path": HANDOFF_REL, "sha256": sha256_file(handoff_path),
        },
        "frozen_ranking_sha256": sha256_file(ranking_path),
        "reference_manifest_sha256": sha256_file(reference_dir / "reference_manifest.json"),
        "reference_coverage": coverage_rows,
        "candidate_membership_changed": False, "pair_ranking_changed": False,
        "known_labels_used_for_model_selection": False,
        "validation_outcomes_read": False,
        "sanity_check_not_osa_specific_validation": True,
        "artifact_sha256": {
            relative_path(root, path): sha256_file(path) for path in artifact_paths
        },
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print("OSA known-PPI/complex recovery: {}".format(status))
    for row in primary_coverage:
        print("[{}] reference_pairs={} eligible_pairs={} status={}".format(
            row["reference"], row["reference_pairs"], row["eligible_pairs"], row["status"]
        ))
    print("Output: {}".format(output_dir))
    print("Frozen candidate/ranking changed: no; validation outcomes read: no")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
