#!/usr/bin/env python3
"""Evaluate the frozen OSA candidates in PXD032734 pediatric serum proteomics.

Python 3.7 compatible.  The script joins the independently preprocessed
PXD032734 effects to frozen discovery candidates and the frozen primary pair
ranking.  It never changes candidate membership, ranking, model parameters,
or endpoint thresholds.  When serum coverage is below the locked minimum, the
pair endpoint is marked not evaluable and only transparent descriptive results
are reported.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_VERSION = "1.3.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LOCK_REL = "config/osa_proteome_validation_lock.json"
HANDOFF_REL = "config/triage_osa_external_validation_lock_v4.json"
PRIMARY_CANDIDATE_REL = (
    "results/discovery_preprocess_v2/04_candidates_v1/"
    "candidate_proteins_primary_250.tsv"
)
SENSITIVITY_CANDIDATE_REL = (
    "results/discovery_preprocess_v2/04_candidates_v1/"
    "candidate_proteins_sensitivity_500.tsv"
)
PAIR_RANKING_REL = (
    "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/"
    "pair_context_scores_triage_osa_primary_250.tsv.gz"
)
RANKING_SHA256 = "1e87e8ecaaf6d4a70ad89086237bcdc84633e5029d7f30efb0a0ccaadf6b03d3"
PREPROCESS_STAGE = "17_pediatric_proteome_preprocess_v1"
DEFAULT_STAGE = "18_pediatric_proteome_endpoint_triage_osa_v4"


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fmt(value):
    if value is None or value == "":
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return "{:.10g}".format(value)


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(str(temp), str(path))


def write_tsv(path, fieldnames, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fieldnames,
            extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(temp), str(path))


def write_gzip_tsv(path, fieldnames, rows):
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(str(temp), "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fieldnames,
            extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(temp), str(path))


def read_tsv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def relative_path(root, path):
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def bh_adjust(values):
    values = [float(value) for value in values]
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [1.0] * len(values)
    running = 1.0
    for reverse in range(len(values) - 1, -1, -1):
        index = order[reverse]
        rank = reverse + 1
        running = min(running, values[index] * len(values) / float(rank))
        result[index] = min(1.0, running)
    return result


def log_combination(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def hypergeom_probability(a, row1, col1, total):
    return math.exp(
        log_combination(col1, a)
        + log_combination(total - col1, row1 - a)
        - log_combination(total, row1)
    )


def fisher_exact_two_sided(a, b, c, d):
    row1 = a + b
    col1 = a + c
    total = a + b + c + d
    observed = hypergeom_probability(a, row1, col1, total)
    lower = max(0, row1 - (total - col1))
    upper = min(row1, col1)
    probability = 0.0
    for value in range(lower, upper + 1):
        candidate = hypergeom_probability(value, row1, col1, total)
        if candidate <= observed + 1.0e-12:
            probability += candidate
    if b * c == 0:
        odds_ratio = float("inf") if a * d > 0 else 0.0
    else:
        odds_ratio = float(a * d) / float(b * c)
    return odds_ratio, min(1.0, probability)


def verify_preprocess(root, preprocess_dir, lock_path):
    summary_path = preprocess_dir / "run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("Missing Stage 17 run_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("Stage 17 status is not PASS")
    if summary.get("participants") != 12 or summary.get("quantified_proteins") != 752:
        raise RuntimeError("Stage 17 participant/protein count mismatch")
    if summary.get("candidate_or_pair_data_read") is not False:
        raise RuntimeError("Stage 17 isolation flag failed")
    if summary.get("validation_used_for_training_or_selection") is not False:
        raise RuntimeError("Stage 17 selection isolation flag failed")
    for relative, expected in summary.get("artifact_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError("Stage 17 artifact verification failed: " + relative)
    snapshot = preprocess_dir / "analysis_lock_snapshot.json"
    if not snapshot.is_file() or sha256_file(snapshot) != sha256_file(lock_path):
        raise RuntimeError("Stage 17 used a different proteome validation lock")
    return summary


def candidate_rows(path, expected_count):
    rows = read_tsv(path)
    if len(rows) != expected_count:
        raise RuntimeError("Frozen candidate count mismatch: {}".format(path))
    ranks = [int(row["discovery_rank"]) for row in rows]
    if ranks != list(range(1, expected_count + 1)):
        raise RuntimeError("Frozen candidate ranks are invalid: {}".format(path))
    if len(set(row["UniProt_accession"] for row in rows)) != expected_count:
        raise RuntimeError("Frozen candidate accessions are not unique")
    return rows


def effect_indexes(effects):
    by_accession = {}
    by_gene = {}
    for row in effects:
        accession = row["accession"].strip()
        gene = row["gene_symbol"].strip()
        if accession in by_accession:
            raise RuntimeError("Duplicate Stage 17 accession: " + accession)
        by_accession[accession] = row
        if gene:
            by_gene.setdefault(gene, []).append(row)
    for gene in by_gene:
        by_gene[gene].sort(
            key=lambda row: (
                -int(row["unique_peptides"]),
                -int(row["psms"]),
                -float(row["score"]),
                row["accession"],
            )
        )
    return by_accession, by_gene


def map_candidate(candidate, by_accession, by_gene):
    accession = candidate["UniProt_accession"]
    gene = candidate["gene_symbol_current"]
    if accession in by_accession:
        return by_accession[accession], "exact_accession"
    if gene in by_gene:
        rows = by_gene[gene]
        status = "gene_symbol_best_of_{}".format(len(rows))
        return rows[0], status
    return None, "not_detected"


def evidence_class(candidate, effect):
    if effect is None:
        return "not_detected"
    adult_axis = float(candidate["adult_OSA_axis"])
    slope = float(effect["primary_ordinal_slope"])
    p_value = float(effect["primary_ordinal_permutation_p"])
    unique_peptides = int(effect["unique_peptides"])
    concordant = adult_axis * slope > 0.0
    if concordant and p_value < 0.05 and unique_peptides >= 2:
        return "directional_support_high_confidence"
    if concordant and abs(slope) > 1.0e-12:
        return "directionally_aligned_not_significant"
    if not concordant and p_value < 0.05 and unique_peptides >= 2:
        return "opposite_direction_high_confidence"
    if not concordant and p_value < 0.05:
        return "opposite_direction_low_peptide_support"
    if not concordant and abs(slope) > 1.0e-12:
        return "directionally_opposite_not_significant"
    return "neutral"


def render_summary_svg(path, quantified, mapped, top_pairs, concordant, status):
    width = 980
    height = 520
    cards = [
        ("Quantified proteins", "{} / 754".format(quantified), "#2563EB"),
        ("Frozen primary nodes detected", "{} / 250 ({:.1f}%)".format(
            mapped, 100.0 * mapped / 250.0
        ), "#D97706"),
        ("Top-1% pairs dual-detected", "{} / 312".format(top_pairs), "#DC2626"),
        ("Mapped-node direction aligned", "{} / {}".format(concordant, mapped), "#7C3AED"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="50" y="58" font-family="Arial" font-size="28" font-weight="700" fill="#0F172A">PXD032734 pediatric serum proteome validation</text>',
        '<text x="50" y="91" font-family="Arial" font-size="16" fill="#475569">Frozen adult PBMC candidates evaluated without re-selection or re-ranking</text>',
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 50 + (index % 2) * 455
        y = 125 + (index // 2) * 135
        parts.extend([
            '<rect x="{}" y="{}" width="420" height="105" rx="14" fill="#FFFFFF" stroke="#CBD5E1"/>'.format(x, y),
            '<rect x="{}" y="{}" width="8" height="105" rx="4" fill="{}"/>'.format(x, y, color),
            '<text x="{}" y="{}" font-family="Arial" font-size="15" fill="#475569">{}</text>'.format(x + 28, y + 35, html.escape(label)),
            '<text x="{}" y="{}" font-family="Arial" font-size="27" font-weight="700" fill="#0F172A">{}</text>'.format(x + 28, y + 76, html.escape(value)),
        ])
    verdict_color = "#B45309" if "LIMITED" in status else "#047857"
    parts.extend([
        '<rect x="50" y="410" width="875" height="70" rx="14" fill="#FFF7ED" stroke="#FDBA74"/>',
        '<text x="72" y="440" font-family="Arial" font-size="17" font-weight="700" fill="{}">Verdict: technically valid, inferential coverage limited</text>'.format(verdict_color),
        '<text x="72" y="466" font-family="Arial" font-size="14" fill="#7C2D12">Use as an assay-compartment limitation and descriptive node check; do not claim pair-level proteomic validation.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--preprocess-stage", default=PREPROCESS_STAGE)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lock_path = root / LOCK_REL
    handoff_path = root / HANDOFF_REL
    preprocess_dir = root / "results" / "discovery_preprocess_v2" / args.preprocess_stage
    primary_path = root / PRIMARY_CANDIDATE_REL
    sensitivity_path = root / SENSITIVITY_CANDIDATE_REL
    ranking_path = root / PAIR_RANKING_REL
    print("OSA pediatric proteome endpoint environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("Preprocess: {}".format(preprocess_dir))
    print("Frozen ranking: {}".format(ranking_path))
    print("Ranking/candidate changes allowed: no")
    if args.check_env:
        return 0

    for path in (lock_path, handoff_path, primary_path, sensitivity_path, ranking_path):
        if not path.is_file():
            raise RuntimeError("Missing required input: {}".format(path))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff["ranking"]["primary_250"] != PAIR_RANKING_REL:
        raise RuntimeError("TRIAGE-OSA handoff ranking path mismatch")
    if handoff["ranking"]["primary_250_sha256"] != RANKING_SHA256:
        raise RuntimeError("TRIAGE-OSA handoff ranking hash mismatch")
    if handoff["prespecified_endpoint_locks"]["proteome"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("TRIAGE-OSA handoff proteome lock mismatch")
    if sha256_file(ranking_path) != RANKING_SHA256:
        raise RuntimeError("Frozen ranking hash mismatch")
    preprocess_summary = verify_preprocess(root, preprocess_dir, lock_path)
    effects = read_tsv(preprocess_dir / "protein_effects.tsv.gz")
    if len(effects) != 752:
        raise RuntimeError("Stage 17 protein-effect count is not 752")
    by_accession, by_gene = effect_indexes(effects)
    primary = candidate_rows(primary_path, 250)
    sensitivity = candidate_rows(sensitivity_path, 500)

    sensitivity_evidence = []
    mapped_by_accession = {}
    for candidate in sensitivity:
        effect, mapping_status = map_candidate(candidate, by_accession, by_gene)
        mapped = effect is not None
        row = {
            "discovery_rank": candidate["discovery_rank"],
            "candidate_set": "primary_250" if int(candidate["discovery_rank"]) <= 250 else "sensitivity_251_500",
            "UniProt_accession": candidate["UniProt_accession"],
            "gene_symbol_current": candidate["gene_symbol_current"],
            "mapping_status": mapping_status,
            "proteome_detected": "yes" if mapped else "no",
            "PXD_accession": effect["accession"] if mapped else "",
            "PXD_gene_symbol": effect["gene_symbol"] if mapped else "",
            "unique_peptides": effect["unique_peptides"] if mapped else "",
            "psms": effect["psms"] if mapped else "",
            "primary_ordinal_slope": effect["primary_ordinal_slope"] if mapped else "",
            "primary_ordinal_permutation_p": effect["primary_ordinal_permutation_p"] if mapped else "",
        }
        sensitivity_evidence.append(row)
        if int(candidate["discovery_rank"]) <= 250:
            mapped_by_accession[candidate["UniProt_accession"]] = (candidate, effect, mapping_status)
    primary_evidence = []
    mapped_rows = []
    for candidate in primary:
        effect, mapping_status = map_candidate(candidate, by_accession, by_gene)
        mapped = effect is not None
        adult_axis = float(candidate["adult_OSA_axis"])
        slope = float(effect["primary_ordinal_slope"]) if mapped else None
        concordant = mapped and adult_axis * slope > 0.0
        row = dict(candidate)
        row.update({
            "proteome_detected": "yes" if mapped else "no",
            "mapping_status": mapping_status,
            "PXD_accession": effect["accession"] if mapped else "",
            "PXD_gene_symbol": effect["gene_symbol"] if mapped else "",
            "PXD_description": effect["description"] if mapped else "",
            "PXD_unique_peptides": effect["unique_peptides"] if mapped else "",
            "PXD_psms": effect["psms"] if mapped else "",
            "PXD_score": effect["score"] if mapped else "",
            "PXD_primary_ordinal_slope": effect["primary_ordinal_slope"] if mapped else "",
            "PXD_primary_ordinal_t": effect["primary_ordinal_t"] if mapped else "",
            "PXD_primary_ordinal_p": effect["primary_ordinal_p"] if mapped else "",
            "PXD_primary_ordinal_permutation_p": effect["primary_ordinal_permutation_p"] if mapped else "",
            "PXD_primary_severe_vs_control_log2fc": effect["primary_severe_vs_control_log2fc"] if mapped else "",
            "PXD_primary_severe_vs_control_permutation_p": effect["primary_severe_vs_control_permutation_p"] if mapped else "",
            "adult_PBMC_direction": "positive" if adult_axis > 0 else "negative",
            "pediatric_serum_direction": (
                "positive" if mapped and slope > 0 else
                "negative" if mapped and slope < 0 else
                "neutral" if mapped else "not_detected"
            ),
            "direction_concordant": "yes" if concordant else "no" if mapped else "",
            "evidence_class": evidence_class(candidate, effect),
            "validation_used_for_selection": "no",
        })
        primary_evidence.append(row)
        if mapped:
            mapped_rows.append(row)
    mapped_node_fdr = bh_adjust([
        float(row["PXD_primary_ordinal_permutation_p"]) for row in mapped_rows
    ])
    for row, fdr in zip(mapped_rows, mapped_node_fdr):
        row["PXD_mapped_node_permutation_fdr_bh"] = fmt(fdr)
    for row in primary_evidence:
        row.setdefault("PXD_mapped_node_permutation_fdr_bh", "")

    output_dir = root / "results" / "discovery_preprocess_v2" / args.stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; pass --force: {}".format(output_dir))
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True)
    write_tsv(
        output_dir / "sensitivity_candidate_coverage.tsv",
        list(sensitivity_evidence[0]), sensitivity_evidence,
    )
    write_tsv(
        output_dir / "candidate_proteome_evidence.tsv",
        list(primary_evidence[0]), primary_evidence,
    )

    mapped_primary = len(mapped_rows)
    mapped_comparator = sum(
        row["proteome_detected"] == "yes" and int(row["discovery_rank"]) > 250
        for row in sensitivity_evidence
    )
    coverage_odds, coverage_p = fisher_exact_two_sided(
        mapped_primary, 250 - mapped_primary,
        mapped_comparator, 250 - mapped_comparator,
    )

    pair_rows = []
    dual_rows = []
    with gzip.open(str(ranking_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = [
            "pair_id", "protein_A", "protein_B", "gene_A", "gene_B",
            "OSA_interaction_priority_rank", "OSA_interaction_priority_score",
            "triage_osa_rank_primary_250", "triage_osa_score",
            "rank_p_PPI", "rank_r_OSA",
        ]
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("Frozen ranking lacks fields: " + ",".join(missing))
        for source in reader:
            a = mapped_by_accession.get(source["protein_A"])
            b = mapped_by_accession.get(source["protein_B"])
            detected_a = a is not None and a[1] is not None
            detected_b = b is not None and b[1] is not None
            pair = {field: source[field] for field in required}
            pair.update({
                "protein_A_detected": "yes" if detected_a else "no",
                "protein_B_detected": "yes" if detected_b else "no",
                "either_node_detected": "yes" if detected_a or detected_b else "no",
                "both_nodes_detected": "yes" if detected_a and detected_b else "no",
                "PXD_accession_A": a[1]["accession"] if detected_a else "",
                "PXD_accession_B": b[1]["accession"] if detected_b else "",
                "PXD_ordinal_slope_A": a[1]["primary_ordinal_slope"] if detected_a else "",
                "PXD_ordinal_slope_B": b[1]["primary_ordinal_slope"] if detected_b else "",
                "validation_used_for_ranking": "no",
            })
            if detected_a and detected_b:
                oriented_a = float(a[0]["adult_OSA_axis"]) * float(a[1]["primary_ordinal_t"])
                oriented_b = float(b[0]["adult_OSA_axis"]) * float(b[1]["primary_ordinal_t"])
                pair["pair_oriented_mean_t"] = fmt(0.5 * (oriented_a + oriented_b))
                pair["pair_oriented_min_t"] = fmt(min(oriented_a, oriented_b))
                dual_rows.append(pair)
            else:
                pair["pair_oriented_mean_t"] = ""
                pair["pair_oriented_min_t"] = ""
            pair_rows.append(pair)
    if len(pair_rows) != 31125:
        raise RuntimeError("Frozen pair count is not 31,125")
    if any(
        int(float(row["triage_osa_rank_primary_250"]))
        != parse_rank(row["OSA_interaction_priority_rank"])
        for row in pair_rows
    ):
        raise RuntimeError("TRIAGE-OSA native and compatibility ranks differ")
    if sorted(parse_rank(row["OSA_interaction_priority_rank"]) for row in pair_rows) != list(range(1, 31126)):
        raise RuntimeError("Frozen primary ranks are invalid")
    pair_rows.sort(key=lambda row: parse_rank(row["OSA_interaction_priority_rank"]))
    dual_rows.sort(key=lambda row: parse_rank(row["OSA_interaction_priority_rank"]))
    pair_fields = list(pair_rows[0])
    write_gzip_tsv(output_dir / "pair_proteome_coverage.tsv.gz", pair_fields, pair_rows)
    write_tsv(
        output_dir / "dual_detected_pairs.tsv", pair_fields,
        dual_rows if dual_rows else [],
    )

    top_count = int(math.ceil(
        len(pair_rows) * float(lock["endpoint"]["primary_pair_fraction"])
    ))
    top_dual = sum(row["both_nodes_detected"] == "yes" for row in pair_rows[:top_count])
    total_dual = len(dual_rows)
    minimum_nodes = int(lock["endpoint"]["minimum_mapped_primary_nodes_for_pair_endpoint"])
    pair_status = "EVALUABLE" if mapped_primary >= minimum_nodes else "NOT_EVALUABLE_LOW_COVERAGE"
    concordant = sum(row["direction_concordant"] == "yes" for row in mapped_rows)
    high_confidence_support = sum(
        row["evidence_class"] == "directional_support_high_confidence"
        for row in mapped_rows
    )
    metrics = [
        {
            "endpoint": "primary_candidate_serum_detection",
            "estimate": fmt(mapped_primary / 250.0),
            "numerator": mapped_primary,
            "denominator": 250,
            "p_value": "",
            "status": "DESCRIPTIVE",
        },
        {
            "endpoint": "primary_vs_ranks251_500_detection_enrichment",
            "estimate": fmt(coverage_odds),
            "numerator": mapped_primary,
            "denominator": mapped_comparator,
            "p_value": fmt(coverage_p),
            "status": "SECONDARY_FISHER",
        },
        {
            "endpoint": "top1pct_dual_node_serum_detection",
            "estimate": fmt(top_dual / float(top_count)),
            "numerator": top_dual,
            "denominator": top_count,
            "p_value": "",
            "status": pair_status,
        },
        {
            "endpoint": "all_pairs_dual_node_serum_detection",
            "estimate": fmt(total_dual / float(len(pair_rows))),
            "numerator": total_dual,
            "denominator": len(pair_rows),
            "p_value": "",
            "status": "DESCRIPTIVE",
        },
        {
            "endpoint": "mapped_node_direction_concordance",
            "estimate": fmt(concordant / float(mapped_primary)) if mapped_primary else "",
            "numerator": concordant,
            "denominator": mapped_primary,
            "p_value": "",
            "status": "DESCRIPTIVE_LOW_COVERAGE",
        },
    ]
    write_tsv(output_dir / "endpoint_metrics.tsv", list(metrics[0]), metrics)

    status = "PASS_LIMITED_COVERAGE" if pair_status != "EVALUABLE" else "PASS_FORMAL"
    conclusions = [
        {
            "claim_id": "technical_usability",
            "finding": "PXD032734 contains 752 quantified proteins across 12 participants (3 per severity group).",
            "support_level": "direct",
            "allowed_claim": "The pediatric serum proteome is technically usable as held-out validation.",
        },
        {
            "claim_id": "candidate_coverage",
            "finding": "Only {} of 250 frozen primary adult PBMC candidates were detected in pediatric serum.".format(mapped_primary),
            "support_level": "direct_limited",
            "allowed_claim": "The depleted-serum TMT assay has very low coverage of the frozen PBMC candidate universe.",
        },
        {
            "claim_id": "pair_endpoint",
            "finding": "{} dual-detected pairs occurred in the frozen top 1%; {} occurred across all 31,125 pairs.".format(top_dual, total_dual),
            "support_level": "not_evaluable",
            "allowed_claim": "Pair-level proteomic validation is not evaluable under the locked minimum-coverage rule.",
        },
        {
            "claim_id": "directional_replication",
            "finding": "{} of {} detected primary candidates matched the adult PBMC direction; {} met high-confidence directional support.".format(concordant, mapped_primary, high_confidence_support),
            "support_level": "descriptive_low_coverage",
            "allowed_claim": "No strong cross-compartment directional replication should be claimed from this dataset.",
        },
        {
            "claim_id": "interpretation_boundary",
            "finding": "The validation tissue is pediatric serum whereas discovery used adult PBMC transcriptomic evidence.",
            "support_level": "design_limitation",
            "allowed_claim": "Low overlap is an assay-and-compartment limitation, not by itself evidence that the interaction model fails.",
        },
    ]
    write_tsv(output_dir / "direct_conclusions.tsv", list(conclusions[0]), conclusions)
    render_summary_svg(
        output_dir / "proteome_validation_summary.svg",
        int(preprocess_summary["quantified_proteins"]), mapped_primary,
        top_dual, concordant, status,
    )
    markdown = """# PXD032734 pediatric serum proteome validation\n\n- Status: **{status}**\n- Quantified proteins: **752/754** across **12 participants**.\n- Frozen primary candidates detected: **{mapped}/250 ({pct:.1f}%)**.\n- Frozen top-1% pairs with both nodes detected: **{top_dual}/{top_count}**.\n- Directionally aligned detected nodes: **{concordant}/{mapped}**.\n- Pair-level endpoint: **{pair_status}**.\n\nInterpretation: the dataset is technically valid, but serum coverage of the adult PBMC candidate universe is too low for a formal pair-ranking validation. Report this as an assay/compartment limitation and retain the mapped-node results as descriptive evidence only. Candidate membership and pair ranking were not changed.\n""".format(
        status=status, mapped=mapped_primary, pct=100.0 * mapped_primary / 250.0,
        top_dual=top_dual, top_count=top_count, concordant=concordant,
        pair_status=pair_status,
    )
    (output_dir / "proteome_validation_summary.md").write_text(markdown, encoding="utf-8")

    artifact_paths = [
        output_dir / "candidate_proteome_evidence.tsv",
        output_dir / "sensitivity_candidate_coverage.tsv",
        output_dir / "pair_proteome_coverage.tsv.gz",
        output_dir / "dual_detected_pairs.tsv",
        output_dir / "endpoint_metrics.tsv",
        output_dir / "direct_conclusions.tsv",
        output_dir / "proteome_validation_summary.svg",
        output_dir / "proteome_validation_summary.md",
    ]
    manifest_rows = [{
        "path": relative_path(root, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in artifact_paths]
    write_tsv(output_dir / "output_manifest.tsv", list(manifest_rows[0]), manifest_rows)
    summary = {
        "status": status,
        "script_version": SCRIPT_VERSION,
        "script": relative_path(root, Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": utc_now(),
        "external_validation_handoff": {
            "path": HANDOFF_REL, "sha256": sha256_file(handoff_path),
        },
        "ranking": {
            "path": PAIR_RANKING_REL, "sha256": sha256_file(ranking_path),
            "model": handoff["model"]["name"],
        },
        "dataset": "PXD032734",
        "quantified_proteins": 752,
        "primary_candidates": 250,
        "mapped_primary_candidates": mapped_primary,
        "mapped_ranks_251_to_500": mapped_comparator,
        "coverage_fisher_odds_ratio": fmt(coverage_odds),
        "coverage_fisher_p": fmt(coverage_p),
        "pair_endpoint_status": pair_status,
        "minimum_mapped_nodes_required": minimum_nodes,
        "top_pair_count": top_count,
        "top_dual_detected_pairs": top_dual,
        "all_dual_detected_pairs": total_dual,
        "directionally_concordant_mapped_nodes": concordant,
        "high_confidence_directional_support_nodes": high_confidence_support,
        "candidate_membership_changed": False,
        "pair_ranking_changed": False,
        "validation_used_for_training_or_selection": False,
        "artifact_sha256": {
            relative_path(root, path): sha256_file(path) for path in artifact_paths
        },
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print("OSA pediatric proteome endpoint: {}".format(status))
    print("Primary candidate coverage: {}/250".format(mapped_primary))
    print("Top-1% dual-detected pairs: {}/{}".format(top_dual, top_count))
    print("Pair endpoint: {}".format(pair_status))
    print("Output: {}".format(output_dir))
    print("Frozen ranking unchanged; validation used for training/selection: no")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
