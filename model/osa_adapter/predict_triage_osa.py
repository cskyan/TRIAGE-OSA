#!/usr/bin/env python3
"""Apply the trained OSA composite checkpoint to the frozen pair universe."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "4.1.1"
FROZEN_PRIMARY_RANKING_SHA256 = (
    "1e87e8ecaaf6d4a70ad89086237bcdc84633e5029d7f30efb0a0ccaadf6b03d3"
)


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


def json_safe(payload):
    """Return an RFC 8259-compatible copy with non-finite values as null."""
    if isinstance(payload, dict):
        return {str(key): json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(value) for value in payload]
    if isinstance(payload, np.generic):
        payload = payload.item()
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(
            json_safe(payload), ensure_ascii=True, indent=2, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--checkpoint",
        default="results/discovery_preprocess_v2/06_model_training/triage_osa_patient_information_v4/triage_osa_specialized.pt",
    )
    parser.add_argument(
        "--pair-scores",
        default="results/discovery_preprocess_v2/08_pair_inference_sensitivity_v1/pair_scores.tsv.gz",
    )
    parser.add_argument(
        "--output-stage",
        default="results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4",
    )
    parser.add_argument(
        "--baseline-ranking",
        default="results/discovery_preprocess_v2/09_pair_context_ranking_v1/pair_context_scores.tsv.gz",
        help="Discovery-only frozen baseline ranks retained as model comparators.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--build-evidence-package",
        action="store_true",
        help="Build the frozen Stage 22 evidence/Pareto/mechanism package without model inference.",
    )
    parser.add_argument(
        "--evidence-output-stage",
        default="results/discovery_preprocess_v2/22_consolidated_evidence_triage_osa_v4",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a deterministic 500-pair subset for Stage 22 integration testing.",
    )
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_from_root(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def evidence_input_paths(root):
    base = root / "results" / "discovery_preprocess_v2"
    return {
        "ranking": base / "09_pair_context_ranking_triage_osa_v4" / "pair_context_scores_triage_osa_primary_250.tsv.gz",
        "cpap_133_pair": base / "11_cpap_reversal_endpoint_triage_osa_v4" / "GSE133601" / "pair_reversal_support.tsv.gz",
        "cpap_498_pair": base / "11_cpap_reversal_endpoint_triage_osa_v4" / "GSE49800" / "pair_reversal_support.tsv.gz",
        "cpap_133_node": base / "11_cpap_reversal_endpoint_triage_osa_v4" / "GSE133601" / "candidate_node_reversal.tsv",
        "cpap_498_node": base / "11_cpap_reversal_endpoint_triage_osa_v4" / "GSE49800" / "candidate_node_reversal.tsv",
        "adipose_pair": base / "14_adipose_cross_tissue_endpoint_triage_osa_v4" / "pair_adipose_support.tsv.gz",
        "adipose_node": base / "14_adipose_cross_tissue_endpoint_triage_osa_v4" / "candidate_adipose_node_support.tsv",
        "scrna_pair": base / "16_scrna_cell_context_endpoint_triage_osa_v4" / "pair_scrna_support.tsv.gz",
        "scrna_node": base / "16_scrna_cell_context_endpoint_triage_osa_v4" / "candidate_scrna_node_support.tsv",
        "proteome_pair": base / "18_pediatric_proteome_endpoint_triage_osa_v4" / "pair_proteome_coverage.tsv.gz",
        "proteome_node": base / "18_pediatric_proteome_endpoint_triage_osa_v4" / "candidate_proteome_evidence.tsv",
        "network_pair": base / "19_known_ppi_complex_recovery_triage_osa_v4" / "pair_reference_labels.tsv.gz",
        "independent_node": base / "21_independent_validation_endpoint_triage_osa_v4" / "candidate_node_evidence_primary_250.tsv",
        "pathway_axis": base / "21_independent_validation_endpoint_triage_osa_v4" / "reactome_pathway_axis_results.tsv",
        "reactome_gmt": root / "annotations" / "independent_validation" / "ReactomePathways.gmt",
        "v4_handoff": root / "config" / "triage_osa_external_validation_lock_v4.json",
        "v3_2_ranking": base / "09_pair_context_ranking_triage_osa_v3_2" / "pair_context_scores_triage_osa_primary_250.tsv.gz",
        "v3_2_pareto": base / "22_consolidated_evidence_triage_osa_v3_2" / "pareto_front_candidates.tsv",
        "v3_2_shortlist": base / "22_consolidated_evidence_triage_osa_v3_2" / "assay_shortlist.tsv",
        "v3_2_modules": base / "22_consolidated_evidence_triage_osa_v3_2" / "mechanism_modules.tsv",
    }


def evidence_stage_directories(root):
    base = root / "results" / "discovery_preprocess_v2"
    return [
        base / "11_cpap_reversal_endpoint_triage_osa_v4",
        base / "12_cpap_subject_signflip_triage_osa_v4",
        base / "14_adipose_cross_tissue_endpoint_triage_osa_v4",
        base / "16_scrna_cell_context_endpoint_triage_osa_v4",
        base / "18_pediatric_proteome_endpoint_triage_osa_v4",
        base / "19_known_ppi_complex_recovery_triage_osa_v4",
        base / "21_independent_validation_endpoint_triage_osa_v4",
    ]


def verify_output_manifest(root, stage_dir):
    manifest_path = stage_dir / "output_manifest.tsv"
    if not manifest_path.is_file():
        raise RuntimeError("Missing output manifest: {}".format(manifest_path))
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    required = {"path", "bytes", "sha256"}
    if not required.issubset(set(manifest.columns)):
        raise RuntimeError("Malformed output manifest: {}".format(manifest_path))
    for row in manifest.itertuples(index=False):
        path = resolve_from_root(root, getattr(row, "path"))
        if not path.is_file():
            raise RuntimeError("Manifest artifact is missing: {}".format(path))
        if int(path.stat().st_size) != int(getattr(row, "bytes")):
            raise RuntimeError("Manifest byte mismatch: {}".format(path))
        if sha256_file(path) != str(getattr(row, "sha256")).lower():
            raise RuntimeError("Manifest SHA256 mismatch: {}".format(path))
    return int(len(manifest))


def validate_evidence_inputs(root):
    paths = evidence_input_paths(root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Missing Stage 22 inputs: {}".format(missing))
    observed_ranking_hash = sha256_file(paths["ranking"])
    if observed_ranking_hash != FROZEN_PRIMARY_RANKING_SHA256:
        raise RuntimeError(
            "Frozen primary ranking SHA256 mismatch: expected={} observed={}".format(
                FROZEN_PRIMARY_RANKING_SHA256, observed_ranking_hash
            )
        )
    handoff = json.loads(paths["v4_handoff"].read_text(encoding="utf-8"))
    if handoff.get("status") != "V4_FROZEN_FOR_EXTERNAL_VALIDATION":
        raise RuntimeError("V4 external-validation handoff is not frozen")
    if handoff["ranking"]["primary_250_sha256"] != observed_ranking_hash:
        raise RuntimeError("V4 handoff and Stage 22 ranking hashes differ")
    manifest_counts = {}
    for stage_dir in evidence_stage_directories(root):
        summary_path = stage_dir / "run_summary.json"
        if not summary_path.is_file():
            raise RuntimeError("Missing run summary: {}".format(summary_path))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not str(summary.get("status", "")).startswith("PASS"):
            raise RuntimeError(
                "Stage is not eligible for evidence integration: {} {}".format(
                    stage_dir.name, summary.get("status")
                )
            )
        manifest_counts[stage_dir.name] = verify_output_manifest(root, stage_dir)
    return paths, manifest_counts


def read_tsv(path, usecols=None, nrows=None):
    return pd.read_csv(
        path, sep="\t", usecols=usecols, nrows=nrows, low_memory=False
    )


def merge_pair_table(frame, path, columns, rename):
    table = read_tsv(path, usecols=["pair_id"] + list(columns))
    if table["pair_id"].duplicated().any():
        raise RuntimeError("Duplicate pair_id in {}".format(path))
    table = table.rename(columns=rename)
    return frame.merge(table, on="pair_id", how="left", validate="one_to_one")


def yes_mask(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["yes", "true", "1", "positive"]
    )


def add_independent_pair_support(frame, node_path):
    node = read_tsv(node_path)
    for dataset in ("GSE38792", "GSE293433", "joint"):
        accession = "UniProt_accession"
        available = "{}_available".format(dataset)
        score = "{}_node_score".format(dataset)
        required = [accession, available, score]
        if any(column not in node.columns for column in required):
            raise RuntimeError("Independent node table lacks {} fields".format(dataset))
        left = node[required].rename(
            columns={
                accession: "protein_A",
                available: "independent_{}_available_A".format(dataset),
                score: "independent_{}_score_A".format(dataset),
            }
        )
        right = node[required].rename(
            columns={
                accession: "protein_B",
                available: "independent_{}_available_B".format(dataset),
                score: "independent_{}_score_B".format(dataset),
            }
        )
        frame = frame.merge(left, on="protein_A", how="left", validate="many_to_one")
        frame = frame.merge(right, on="protein_B", how="left", validate="many_to_one")
        available_pair = yes_mask(
            frame["independent_{}_available_A".format(dataset)]
        ) & yes_mask(frame["independent_{}_available_B".format(dataset)])
        score_a = pd.to_numeric(
            frame["independent_{}_score_A".format(dataset)], errors="coerce"
        )
        score_b = pd.to_numeric(
            frame["independent_{}_score_B".format(dataset)], errors="coerce"
        )
        pair_score = np.sqrt(np.clip(score_a, 0.0, 1.0) * np.clip(score_b, 0.0, 1.0))
        frame["independent_{}_pair_available".format(dataset)] = np.where(
            available_pair, "yes", "no"
        )
        frame["independent_{}_pair_support".format(dataset)] = np.where(
            available_pair, pair_score, np.nan
        )
    return frame, node


def load_axis_gene_map(pathway_axis_path, gmt_path):
    axis_table = read_tsv(pathway_axis_path)
    axis_rows = axis_table[axis_table["level"].astype(str).eq("axis")]
    stable_to_axes = {}
    for row in axis_rows.itertuples(index=False):
        axis = str(getattr(row, "axis"))
        for stable_id in str(getattr(row, "Reactome_stable_ID")).split("|"):
            stable_to_axes.setdefault(stable_id, set()).add(axis)
    stable_to_genes = {}
    with Path(gmt_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                stable_to_genes[fields[1]] = set(fields[2:])
    gene_axes = {}
    for stable_id, axes in stable_to_axes.items():
        for gene in stable_to_genes.get(stable_id, set()):
            gene_axes.setdefault(gene, set()).update(axes)
    return gene_axes


def pareto_front_mask(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError("Pareto input must have exactly three dimensions")
    if not np.isfinite(values).all():
        raise RuntimeError("Pareto dimensions contain non-finite values")
    order = np.argsort(-values[:, 0], kind="mergesort")
    front = []
    for index in order:
        value = values[index]
        if front:
            current = values[np.asarray(front, dtype=int)]
            dominated = np.any(
                np.all(current >= value, axis=1)
                & np.any(current > value, axis=1)
            )
            if dominated:
                continue
            retained = ~(
                np.all(value >= current, axis=1)
                & np.any(value > current, axis=1)
            )
            front = list(np.asarray(front, dtype=int)[retained])
        front.append(int(index))
    mask = np.zeros(values.shape[0], dtype=bool)
    mask[np.asarray(front, dtype=int)] = True
    return mask


def add_evidence_axes(frame):
    definitions = [
        (
            "cpap_GSE133601",
            yes_mask(frame["cpap_GSE133601_eligible"]),
            pd.to_numeric(frame["cpap_GSE133601_support"], errors="coerce") > 0.5,
        ),
        (
            "cpap_GSE49800",
            yes_mask(frame["cpap_GSE49800_eligible"]),
            pd.to_numeric(frame["cpap_GSE49800_support"], errors="coerce") > 0.5,
        ),
        (
            "adipose_state",
            yes_mask(frame["adipose_eligible"]),
            pd.to_numeric(frame["adipose_state_support"], errors="coerce") > 0.5,
        ),
        (
            "adipose_CPAP",
            yes_mask(frame["adipose_eligible"]),
            pd.to_numeric(frame["adipose_CPAP_support"], errors="coerce") > 0.5,
        ),
        (
            "pediatric_scRNA",
            pd.to_numeric(frame["scrna_available_lineage_count"], errors="coerce") > 0,
            pd.to_numeric(frame["scrna_mean_support"], errors="coerce") > 0.5,
        ),
        (
            "pediatric_proteome",
            yes_mask(frame["proteome_both_nodes_detected"]),
            pd.to_numeric(frame["proteome_pair_oriented_min_t"], errors="coerce") > 0.0,
        ),
        (
            "known_network",
            pd.Series(True, index=frame.index),
            pd.concat(
                [
                    yes_mask(frame["intact_physical"]),
                    yes_mask(frame["complexportal_experimental"]),
                    yes_mask(frame["intact_all_positive"]),
                    yes_mask(frame["complexportal_all_curated"]),
                ],
                axis=1,
            ).any(axis=1),
        ),
        (
            "independent_GSE38792",
            yes_mask(frame["independent_GSE38792_pair_available"]),
            pd.to_numeric(
                frame["independent_GSE38792_pair_support"], errors="coerce"
            ) > 0.5,
        ),
        (
            "independent_GSE293433",
            yes_mask(frame["independent_GSE293433_pair_available"]),
            pd.to_numeric(
                frame["independent_GSE293433_pair_support"], errors="coerce"
            ) > 0.5,
        ),
    ]
    evaluable_columns = []
    supportive_columns = []
    for name, evaluable, supportive in definitions:
        eval_column = "{}_evaluable".format(name)
        support_column = "{}_supportive".format(name)
        frame[eval_column] = np.asarray(evaluable, dtype=bool)
        frame[support_column] = np.asarray(evaluable & supportive, dtype=bool)
        evaluable_columns.append(eval_column)
        supportive_columns.append(support_column)
    frame["evidence_evaluable_axis_count"] = frame[evaluable_columns].sum(axis=1)
    frame["evidence_supportive_axis_count"] = frame[supportive_columns].sum(axis=1)
    frame["evidence_missing_axis_count"] = len(definitions) - frame[
        "evidence_evaluable_axis_count"
    ]
    frame["evidence_breadth_score"] = (
        frame["evidence_supportive_axis_count"].astype(float) / float(len(definitions))
    )
    cpap_conflict = (
        frame["cpap_GSE133601_evaluable"]
        & frame["cpap_GSE49800_evaluable"]
        & (
            frame["cpap_GSE133601_supportive"]
            != frame["cpap_GSE49800_supportive"]
        )
    )
    independent_conflict = (
        frame["independent_GSE38792_evaluable"]
        & frame["independent_GSE293433_evaluable"]
        & (
            frame["independent_GSE38792_supportive"]
            != frame["independent_GSE293433_supportive"]
        )
    )
    lineage_values = frame[
        ["scrna_T_NK_support", "scrna_B_support", "scrna_Myeloid_support"]
    ].apply(pd.to_numeric, errors="coerce")
    lineage_conflict = (
        (lineage_values.notna().sum(axis=1) >= 2)
        & (lineage_values.min(axis=1) <= 0.5)
        & (lineage_values.max(axis=1) > 0.5)
    )
    frame["conflict_CPAP_cohort"] = cpap_conflict
    frame["conflict_independent_cohort"] = independent_conflict
    frame["conflict_scRNA_lineage"] = lineage_conflict
    frame["evidence_conflict_count"] = frame[
        [
            "conflict_CPAP_cohort",
            "conflict_independent_cohort",
            "conflict_scRNA_lineage",
        ]
    ].sum(axis=1)
    return frame


def build_model_axis_comparison(root):
    base = root / "results" / "discovery_preprocess_v2"
    records = []

    def add_rank_table(stage, endpoint, path, endpoint_column=None):
        table = read_tsv(path)
        if endpoint_column is not None:
            table = table[table[endpoint_column].astype(str).eq(endpoint)]
        top = table[
            table["metric_scope"].astype(str).eq("fraction")
            & np.isclose(pd.to_numeric(table["fraction"], errors="coerce"), 0.01)
        ]
        auec = table[table["metric_scope"].astype(str).eq("AUEC")]
        auec_map = dict(
            zip(auec["model"].astype(str), pd.to_numeric(auec["normalized_AUEC_0p5_to_5pct"], errors="coerce"))
        )
        for row in top.itertuples(index=False):
            records.append(
                {
                    "stage": stage,
                    "endpoint": endpoint,
                    "model": str(getattr(row, "model")),
                    "top_1pct_ratio": float(getattr(row, "support_enrichment_ratio")),
                    "top_1pct_difference": float(getattr(row, "support_enrichment_difference")),
                    "AUEC_0p5_to_5pct": float(auec_map.get(str(getattr(row, "model")), np.nan)),
                    "nominal_P": np.nan,
                    "BH_FDR": np.nan,
                    "analysis_role": "post_development_descriptive",
                }
            )

    add_rank_table(
        "11", "CPAP_GSE133601",
        base / "11_cpap_reversal_endpoint_triage_osa_v4" / "GSE133601" / "ranking_endpoint_metrics.tsv",
    )
    add_rank_table(
        "11", "CPAP_GSE49800",
        base / "11_cpap_reversal_endpoint_triage_osa_v4" / "GSE49800" / "ranking_endpoint_metrics.tsv",
    )
    add_rank_table(
        "14", "state",
        base / "14_adipose_cross_tissue_endpoint_triage_osa_v4" / "ranking_endpoint_metrics.tsv",
        "endpoint",
    )
    add_rank_table(
        "14", "CPAP",
        base / "14_adipose_cross_tissue_endpoint_triage_osa_v4" / "ranking_endpoint_metrics.tsv",
        "endpoint",
    )
    scrna = read_tsv(
        base / "16_scrna_cell_context_endpoint_triage_osa_v4" / "model_comparison.tsv"
    )
    for row in scrna.itertuples(index=False):
        records.append(
            {
                "stage": "16",
                "endpoint": "pediatric_scRNA_omnibus",
                "model": str(getattr(row, "model")),
                "top_1pct_ratio": float(getattr(row, "omnibus_top_1pct_mean_ratio")),
                "top_1pct_difference": float(getattr(row, "omnibus_top_1pct_mean_difference")),
                "AUEC_0p5_to_5pct": float(getattr(row, "omnibus_AUEC_0p5_to_5pct")),
                "nominal_P": np.nan,
                "BH_FDR": np.nan,
                "analysis_role": "post_development_descriptive",
            }
        )
    network = read_tsv(
        base / "19_known_ppi_complex_recovery_triage_osa_v4" / "ranking_recovery_metrics.tsv"
    )
    network = network[np.isclose(pd.to_numeric(network["fraction"], errors="coerce"), 0.01)]
    for row in network.itertuples(index=False):
        records.append(
            {
                "stage": "19",
                "endpoint": str(getattr(row, "reference")),
                "model": str(getattr(row, "model")),
                "top_1pct_ratio": float(getattr(row, "top_enrichment_ratio")),
                "top_1pct_difference": np.nan,
                "AUEC_0p5_to_5pct": float(getattr(row, "auec_0p5_to_5pct")),
                "nominal_P": float(getattr(row, "one_sided_hypergeometric_p")),
                "BH_FDR": float(getattr(row, "BH_FDR_within_reference")),
                "analysis_role": "frozen_structural_sanity",
            }
        )
    independent = read_tsv(
        base / "21_independent_validation_endpoint_triage_osa_v4" / "pair_support_metrics.tsv"
    )
    top = independent[
        independent["metric_scope"].astype(str).eq("fraction")
        & np.isclose(pd.to_numeric(independent["fraction"], errors="coerce"), 0.01)
    ]
    auec = independent[independent["metric_scope"].astype(str).eq("AUEC")]
    auec_map = dict(
        ((str(row.endpoint), str(row.model)), float(row.normalized_AUEC))
        for row in auec.itertuples(index=False)
    )
    for row in top.itertuples(index=False):
        records.append(
            {
                "stage": "21",
                "endpoint": str(getattr(row, "endpoint")),
                "model": str(getattr(row, "model")),
                "top_1pct_ratio": float(getattr(row, "support_enrichment_ratio")),
                "top_1pct_difference": float(getattr(row, "support_enrichment_difference")),
                "AUEC_0p5_to_5pct": float(
                    auec_map.get((str(getattr(row, "endpoint")), str(getattr(row, "model"))), np.nan)
                ),
                "nominal_P": np.nan,
                "BH_FDR": np.nan,
                "analysis_role": "untouched_independent_formal_descriptive_model_table",
            }
        )
    return pd.DataFrame.from_records(records)


def connected_components(edge_frame):
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row in edge_frame[["protein_A", "protein_B"]].itertuples(index=False):
        union(str(row.protein_A), str(row.protein_B))
    groups = {}
    for protein in list(parent):
        groups.setdefault(find(protein), set()).add(protein)
    return sorted(groups.values(), key=lambda values: (-len(values), sorted(values)))


def build_protein_and_module_tables(pair_frame, paths, gene_axes):
    left = pair_frame[
        ["protein_A", "gene_A", "triage_osa_rank_primary_250", "pareto_front", "evidence_supportive_axis_count"]
    ].rename(columns={"protein_A": "UniProt", "gene_A": "gene_symbol"})
    right = pair_frame[
        ["protein_B", "gene_B", "triage_osa_rank_primary_250", "pareto_front", "evidence_supportive_axis_count"]
    ].rename(columns={"protein_B": "UniProt", "gene_B": "gene_symbol"})
    long_pairs = pd.concat([left, right], ignore_index=True)
    protein = long_pairs.groupby(["UniProt", "gene_symbol"], as_index=False).agg(
        best_pair_rank=("triage_osa_rank_primary_250", "min"),
        pareto_pair_count=("pareto_front", "sum"),
        max_pair_supportive_axes=("evidence_supportive_axis_count", "max"),
    )
    protein["top_1pct_pair_count"] = protein["UniProt"].map(
        pd.concat(
            [
                pair_frame.loc[pair_frame["triage_osa_rank_primary_250"] <= 312, "protein_A"],
                pair_frame.loc[pair_frame["triage_osa_rank_primary_250"] <= 312, "protein_B"],
            ]
        ).value_counts()
    ).fillna(0).astype(int)

    proteome = read_tsv(paths["proteome_node"])
    proteome_columns = [
        "UniProt_accession", "OSA_strength", "severe_strength", "AHI_strength",
        "IH_strength", "proteome_detected", "PXD_accession",
        "PXD_primary_ordinal_t", "direction_concordant",
    ]
    protein = protein.merge(
        proteome[proteome_columns].rename(columns={"UniProt_accession": "UniProt"}),
        on="UniProt", how="left", validate="one_to_one",
    )
    for dataset, path in (("GSE133601", paths["cpap_133_node"]), ("GSE49800", paths["cpap_498_node"])):
        node = read_tsv(path)
        selected = node[
            ["UniProt_accession", "validation_available", "node_reversal_score", "directionally_reversed"]
        ].rename(
            columns={
                "UniProt_accession": "UniProt",
                "validation_available": "{}_available".format(dataset),
                "node_reversal_score": "{}_CPAP_reversal_score".format(dataset),
                "directionally_reversed": "{}_directionally_reversed".format(dataset),
            }
        )
        protein = protein.merge(selected, on="UniProt", how="left", validate="one_to_one")
    adipose = read_tsv(paths["adipose_node"])
    protein = protein.merge(
        adipose[
            ["UniProt_accession", "adipose_available", "state_node_score", "CPAP_node_score", "persistent_state_without_reversal"]
        ].rename(columns={"UniProt_accession": "UniProt"}),
        on="UniProt", how="left", validate="one_to_one",
    )
    scrna = read_tsv(paths["scrna_node"])
    scrna_scores = scrna.pivot_table(
        index="UniProt_accession", columns="lineage", values="node_score", aggfunc="mean"
    ).reset_index()
    scrna_scores.columns = [
        "UniProt" if column == "UniProt_accession" else "scrna_{}_node_score".format(column)
        for column in scrna_scores.columns
    ]
    protein = protein.merge(scrna_scores, on="UniProt", how="left", validate="one_to_one")
    independent = read_tsv(paths["independent_node"])
    independent_columns = [
        "UniProt_accession", "GSE38792_available", "GSE38792_node_score",
        "GSE293433_available", "GSE293433_node_score", "joint_available", "joint_node_score",
    ]
    protein = protein.merge(
        independent[independent_columns].rename(columns={"UniProt_accession": "UniProt"}),
        on="UniProt", how="left", validate="one_to_one",
    )
    protein["pathway_axes"] = protein["gene_symbol"].map(
        lambda gene: "|".join(sorted(gene_axes.get(str(gene), set())))
    )
    protein["pathway_axis_count"] = protein["pathway_axes"].map(
        lambda value: 0 if not value else len(value.split("|"))
    )
    protein["assay_tier"] = "not_shortlisted"
    detected = yes_mask(protein["proteome_detected"])
    tier1 = (
        (protein["pareto_pair_count"] > 0)
        & (protein["max_pair_supportive_axes"] >= 2)
        & detected
    )
    tier2 = (
        (protein["pareto_pair_count"] > 0)
        & (protein["max_pair_supportive_axes"] >= 2)
        & ~tier1
    )
    tier3 = (~tier1 & ~tier2) & (
        (protein["pareto_pair_count"] > 0) | (protein["best_pair_rank"] <= 312)
    )
    protein.loc[tier3, "assay_tier"] = "Tier3_exploratory"
    protein.loc[tier2, "assay_tier"] = "Tier2_mechanistic_low_assay_coverage"
    protein.loc[tier1, "assay_tier"] = "Tier1_detected_multi_axis"
    tier_order = {
        "Tier1_detected_multi_axis": 1,
        "Tier2_mechanistic_low_assay_coverage": 2,
        "Tier3_exploratory": 3,
        "not_shortlisted": 4,
    }
    protein["assay_tier_order"] = protein["assay_tier"].map(tier_order).astype(int)
    shortlist = protein[protein["assay_tier_order"] <= 3].sort_values(
        ["assay_tier_order", "max_pair_supportive_axes", "pareto_pair_count", "best_pair_rank"],
        ascending=[True, False, False, True], kind="mergesort",
    ).head(20).copy()
    shortlist["shortlist_rank"] = np.arange(1, len(shortlist) + 1)

    module_edges = pair_frame[
        pair_frame["pareto_front"]
        & (pd.to_numeric(pair_frame["triage_osa_rank_primary_250"], errors="coerce") <= 1557)
    ].copy()
    if module_edges.empty:
        module_edges = pair_frame.nsmallest(100, "triage_osa_rank_primary_250").copy()
    components = connected_components(module_edges)
    protein_to_gene = dict(zip(protein["UniProt"], protein["gene_symbol"]))
    protein_to_axes = dict(zip(protein["UniProt"], protein["pathway_axes"]))
    module_rows = []
    for module_index, proteins in enumerate(components, 1):
        mask = module_edges["protein_A"].isin(proteins) & module_edges["protein_B"].isin(proteins)
        edges = module_edges.loc[mask]
        axes = set()
        for accession in proteins:
            axes.update(filter(None, str(protein_to_axes.get(accession, "")).split("|")))
        module_rows.append(
            {
                "module_id": "M{:03d}".format(module_index),
                "protein_count": int(len(proteins)),
                "pair_count": int(len(edges)),
                "proteins": "|".join(sorted(proteins)),
                "genes": "|".join(sorted(set(str(protein_to_gene.get(value, "")) for value in proteins))),
                "pathway_axes": "|".join(sorted(axes)),
                "median_structural_score": float(edges["structural_score"].median()),
                "median_OSA_context_score": float(edges["osa_context_score"].median()),
                "median_evidence_breadth_score": float(edges["evidence_breadth_score"].median()),
                "conflicting_pair_fraction": float((edges["evidence_conflict_count"] > 0).mean()),
            }
        )
    modules = pd.DataFrame.from_records(module_rows)
    return protein.sort_values("best_pair_rank", kind="mergesort"), shortlist, modules


def build_version_stability(paths, v4_pair_frame, v4_shortlist, v4_modules):
    v3_ranking = read_tsv(
        paths["v3_2_ranking"],
        usecols=["pair_id", "triage_osa_rank_primary_250"],
    ).rename(columns={"triage_osa_rank_primary_250": "v3_2_rank"})
    v4_ranking = v4_pair_frame[
        ["pair_id", "triage_osa_rank_primary_250"]
    ].rename(columns={"triage_osa_rank_primary_250": "v4_rank"})
    ranks = v3_ranking.merge(
        v4_ranking, on="pair_id", how="outer", validate="one_to_one", indicator=True
    )
    if len(ranks) != 31125 or not ranks["_merge"].eq("both").all():
        raise RuntimeError("V3.2 and V4 pair universes differ")
    v3_rank_values = pd.to_numeric(ranks["v3_2_rank"], errors="raise").to_numpy(float)
    v4_rank_values = pd.to_numeric(ranks["v4_rank"], errors="raise").to_numpy(float)

    v3_pareto = set(read_tsv(paths["v3_2_pareto"], usecols=["pair_id"])["pair_id"].astype(str))
    v4_pareto = set(
        v4_pair_frame.loc[v4_pair_frame["pareto_front"], "pair_id"].astype(str)
    )
    v3_shortlist = set(
        read_tsv(paths["v3_2_shortlist"], usecols=["UniProt"])["UniProt"].astype(str)
    )
    v4_shortlist_set = set(v4_shortlist["UniProt"].astype(str))

    def module_proteins(frame):
        values = set()
        if "proteins" not in frame.columns:
            return values
        for item in frame["proteins"].fillna("").astype(str):
            values.update(value for value in item.split("|") if value)
        return values

    v3_module_proteins = module_proteins(read_tsv(paths["v3_2_modules"]))
    v4_module_proteins = module_proteins(v4_modules)
    v3_top1 = set(ranks.loc[v3_rank_values <= 312, "pair_id"].astype(str))
    v4_top1 = set(ranks.loc[v4_rank_values <= 312, "pair_id"].astype(str))
    v3_top5 = set(ranks.loc[v3_rank_values <= 1557, "pair_id"].astype(str))
    v4_top5 = set(ranks.loc[v4_rank_values <= 1557, "pair_id"].astype(str))

    groups = [
        ("top_1pct_pairs", v3_top1, v4_top1),
        ("top_5pct_pairs", v3_top5, v4_top5),
        ("pareto_pairs", v3_pareto, v4_pareto),
        ("assay_shortlist_proteins", v3_shortlist, v4_shortlist_set),
        ("mechanism_module_proteins", v3_module_proteins, v4_module_proteins),
    ]
    metric_rows = [{
        "comparison": "all_pair_rank_Spearman",
        "v3_2_count": 31125,
        "v4_count": 31125,
        "intersection": 31125,
        "union": 31125,
        "retention_fraction_of_v3_2": 1.0,
        "jaccard": 1.0,
        "value": float(np.corrcoef(v3_rank_values, v4_rank_values)[0, 1]),
    }]
    membership_rows = []
    for name, old_set, new_set in groups:
        intersection = old_set & new_set
        union = old_set | new_set
        metric_rows.append({
            "comparison": name,
            "v3_2_count": int(len(old_set)),
            "v4_count": int(len(new_set)),
            "intersection": int(len(intersection)),
            "union": int(len(union)),
            "retention_fraction_of_v3_2": (
                float(len(intersection)) / float(len(old_set)) if old_set else np.nan
            ),
            "jaccard": float(len(intersection)) / float(len(union)) if union else np.nan,
            "value": np.nan,
        })
        for identifier in sorted(union):
            in_old = identifier in old_set
            in_new = identifier in new_set
            membership_rows.append({
                "entity_set": name,
                "identifier": identifier,
                "V3_2": "yes" if in_old else "no",
                "V4": "yes" if in_new else "no",
                "status": "retained" if in_old and in_new else "gained" if in_new else "lost",
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(membership_rows)


def write_stage22_outputs(
    root, output_dir, pair_frame, protein, shortlist, modules,
    model_comparison, stability_metrics, stability_membership, summary,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "pair_evidence_matrix.tsv.gz"
    protein_path = output_dir / "protein_evidence_matrix.tsv"
    pareto_path = output_dir / "pareto_front_candidates.tsv"
    model_path = output_dir / "model_axis_comparison.tsv"
    module_path = output_dir / "mechanism_modules.tsv"
    shortlist_path = output_dir / "assay_shortlist.tsv"
    conflict_path = output_dir / "evidence_conflict_matrix.tsv"
    stability_metrics_path = output_dir / "version_stability_metrics.tsv"
    stability_membership_path = output_dir / "version_stability_membership.tsv"
    pair_frame.to_csv(pair_path, sep="\t", index=False, compression="gzip")
    protein.to_csv(protein_path, sep="\t", index=False)
    pair_frame[pair_frame["pareto_front"]].sort_values(
        ["evidence_supportive_axis_count", "triage_osa_rank_primary_250"],
        ascending=[False, True], kind="mergesort",
    ).to_csv(pareto_path, sep="\t", index=False)
    model_comparison.to_csv(model_path, sep="\t", index=False)
    modules.to_csv(module_path, sep="\t", index=False)
    shortlist.to_csv(shortlist_path, sep="\t", index=False)
    pair_frame[
        [
            "pair_id", "protein_A", "protein_B", "gene_A", "gene_B",
            "triage_osa_rank_primary_250", "conflict_CPAP_cohort",
            "conflict_independent_cohort", "conflict_scRNA_lineage",
            "evidence_conflict_count", "evidence_missing_axis_count",
        ]
    ].to_csv(conflict_path, sep="\t", index=False)
    if not stability_metrics.empty:
        stability_metrics.to_csv(stability_metrics_path, sep="\t", index=False)
        stability_membership.to_csv(stability_membership_path, sep="\t", index=False)
    summary_path = output_dir / "run_summary.json"
    write_json(summary_path, summary)
    artifacts = [
        pair_path, protein_path, pareto_path, model_path, module_path,
        shortlist_path, conflict_path, summary_path,
    ]
    if not stability_metrics.empty:
        artifacts.extend([stability_metrics_path, stability_membership_path])
    manifest_rows = []
    for path in artifacts:
        manifest_rows.append(
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "output_manifest.tsv", sep="\t", index=False
    )
    return pair_path


def build_evidence_package(args, root):
    paths, manifest_counts = validate_evidence_inputs(root)
    output_base = resolve_from_root(root, args.evidence_output_stage)
    output_dir = output_base / "smoke" if args.smoke else output_base
    print("TRIAGE-OSA Stage 22 evidence integration environment")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("pandas: {}".format(pd.__version__))
    print("Frozen ranking: {}".format(paths["ranking"]))
    print("Frozen ranking SHA256: {}".format(FROZEN_PRIMARY_RANKING_SHA256))
    print("Model inference/training: no")
    print("Validation changes ranking: no")
    if args.check_env:
        print("Stage manifests: {}".format(manifest_counts))
        print("TRIAGE-OSA Stage 22 evidence integration preflight: PASS")
        return 0

    nrows = 500 if args.smoke else None
    ranking = read_tsv(paths["ranking"], nrows=nrows)
    required = {
        "pair_id", "protein_A", "protein_B", "gene_A", "gene_B",
        "p_L1", "p_L2", "p_L3", "rank_r_OSA", "triage_osa_score",
        "triage_osa_rank_primary_250", "OSA_interaction_priority_rank",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise RuntimeError("Frozen ranking lacks Stage 22 fields: {}".format(missing))
    if ranking["pair_id"].duplicated().any():
        raise RuntimeError("Frozen ranking contains duplicate pair_id")
    if not args.smoke:
        if len(ranking) != 31125:
            raise RuntimeError("Primary pair universe must contain 31,125 rows")
        observed_ranks = sorted(
            pd.to_numeric(ranking["triage_osa_rank_primary_250"], errors="raise").astype(int).tolist()
        )
        if observed_ranks != list(range(1, 31126)):
            raise RuntimeError("Primary rank range is incomplete or duplicated")
    original_pair_ids = ranking["pair_id"].astype(str).tolist()
    frame = ranking.copy()
    frame = merge_pair_table(
        frame, paths["cpap_133_pair"],
        ["validation_eligible", "pair_reversal_support", "both_nodes_directionally_reversed"],
        {
            "validation_eligible": "cpap_GSE133601_eligible",
            "pair_reversal_support": "cpap_GSE133601_support",
            "both_nodes_directionally_reversed": "cpap_GSE133601_both_reversed",
        },
    )
    frame = merge_pair_table(
        frame, paths["cpap_498_pair"],
        ["validation_eligible", "pair_reversal_support", "both_nodes_directionally_reversed"],
        {
            "validation_eligible": "cpap_GSE49800_eligible",
            "pair_reversal_support": "cpap_GSE49800_support",
            "both_nodes_directionally_reversed": "cpap_GSE49800_both_reversed",
        },
    )
    frame = merge_pair_table(
        frame, paths["adipose_pair"],
        ["validation_eligible", "state_pair_support", "CPAP_pair_support", "joint_pair_support"],
        {
            "validation_eligible": "adipose_eligible",
            "state_pair_support": "adipose_state_support",
            "CPAP_pair_support": "adipose_CPAP_support",
            "joint_pair_support": "adipose_joint_support",
        },
    )
    frame = merge_pair_table(
        frame, paths["scrna_pair"],
        [
            "available_lineage_count", "mean_available_lineage_support",
            "T_NK_pair_support", "B_pair_support", "Myeloid_pair_support",
        ],
        {
            "available_lineage_count": "scrna_available_lineage_count",
            "mean_available_lineage_support": "scrna_mean_support",
            "T_NK_pair_support": "scrna_T_NK_support",
            "B_pair_support": "scrna_B_support",
            "Myeloid_pair_support": "scrna_Myeloid_support",
        },
    )
    frame = merge_pair_table(
        frame, paths["proteome_pair"],
        ["both_nodes_detected", "pair_oriented_mean_t", "pair_oriented_min_t"],
        {
            "both_nodes_detected": "proteome_both_nodes_detected",
            "pair_oriented_mean_t": "proteome_pair_oriented_mean_t",
            "pair_oriented_min_t": "proteome_pair_oriented_min_t",
        },
    )
    frame = merge_pair_table(
        frame, paths["network_pair"],
        [
            "intact_all_positive", "intact_physical", "complexportal_all_curated",
            "complexportal_experimental", "complex_accessions",
        ],
        {},
    )
    frame, independent_node = add_independent_pair_support(frame, paths["independent_node"])
    if frame["pair_id"].astype(str).tolist() != original_pair_ids:
        raise RuntimeError("Evidence joins changed the frozen pair order or membership")
    frame["structural_score"] = frame[["p_L1", "p_L2", "p_L3"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)
    frame["structural_percentile"] = frame["structural_score"].rank(pct=True, method="average")
    context_rank = pd.to_numeric(frame["rank_r_OSA"], errors="coerce")
    context_denominator = 31124.0 if not args.smoke else max(float(len(frame) - 1), 1.0)
    frame["osa_context_score"] = 1.0 - np.clip((context_rank - 1.0) / context_denominator, 0.0, 1.0)
    frame["osa_context_percentile"] = frame["osa_context_score"].rank(pct=True, method="average")
    frame = add_evidence_axes(frame)
    frame["pareto_front"] = pareto_front_mask(
        frame[["structural_percentile", "osa_context_percentile", "evidence_breadth_score"]].to_numpy()
    )
    frame["pareto_front_level"] = np.where(frame["pareto_front"], 1, np.nan)
    gene_axes = load_axis_gene_map(paths["pathway_axis"], paths["reactome_gmt"])
    frame["mechanism_axes_A"] = frame["gene_A"].map(
        lambda gene: "|".join(sorted(gene_axes.get(str(gene), set())))
    )
    frame["mechanism_axes_B"] = frame["gene_B"].map(
        lambda gene: "|".join(sorted(gene_axes.get(str(gene), set())))
    )
    frame["mechanism_axes_union"] = [
        "|".join(sorted(set(filter(None, str(left).split("|"))) | set(filter(None, str(right).split("|")))))
        for left, right in zip(frame["mechanism_axes_A"], frame["mechanism_axes_B"])
    ]
    protein, shortlist, modules = build_protein_and_module_tables(frame, paths, gene_axes)
    model_comparison = build_model_axis_comparison(root)
    if args.smoke:
        stability_metrics = pd.DataFrame()
        stability_membership = pd.DataFrame()
    else:
        stability_metrics, stability_membership = build_version_stability(
            paths, frame, shortlist, modules
        )
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force: {}".format(output_dir))
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace Stage 22 output outside project root")
        shutil.rmtree(str(output_dir))
    status = "PASS_SMOKE_EVIDENCE_PACKAGE" if args.smoke else "PASS_CONSOLIDATED_EVIDENCE_PACKAGE"
    summary = {
        "status": status,
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_ranking": str(paths["ranking"]),
        "frozen_ranking_sha256": FROZEN_PRIMARY_RANKING_SHA256,
        "pairs": int(len(frame)),
        "proteins": int(len(set(frame["protein_A"]) | set(frame["protein_B"]))),
        "pareto_front_pairs": int(frame["pareto_front"].sum()),
        "mechanism_modules": int(len(modules)),
        "assay_shortlist_proteins": int(len(shortlist)),
        "mean_supportive_axes": float(frame["evidence_supportive_axis_count"].mean()),
        "mean_evaluable_axes": float(frame["evidence_evaluable_axis_count"].mean()),
        "pairs_with_conflict": int((frame["evidence_conflict_count"] > 0).sum()),
        "stage_manifest_counts": manifest_counts,
        "model_inference_or_training_performed": False,
        "ranking_changed": False,
        "candidate_membership_changed": False,
        "validation_used_for_training_or_selection": False,
        "post_development_exploratory_integration": True,
        "model": "TRIAGE_Fuse_OSA_patient_information_v4",
        "external_validation_handoff": "config/triage_osa_external_validation_lock_v4.json",
        "version_stability_evaluated": not args.smoke,
        "version_stability": (
            stability_metrics.to_dict(orient="records") if not args.smoke else []
        ),
        "score_semantics": "Evidence integration and Pareto triage; not physical PPI probability or clinical risk prediction.",
    }
    pair_path = write_stage22_outputs(
        root, output_dir, frame, protein, shortlist, modules, model_comparison,
        stability_metrics, stability_membership, summary,
    )
    print("TRIAGE-OSA Stage 22 evidence integration: {}".format(status))
    print("Pairs: {} proteins: {}".format(summary["pairs"], summary["proteins"]))
    print("Pareto pairs: {} modules: {} shortlist: {}".format(
        summary["pareto_front_pairs"], summary["mechanism_modules"], summary["assay_shortlist_proteins"]
    ))
    print("Output: {}".format(output_dir))
    print("Pair evidence: {}".format(pair_path))
    print("Ranking changed: no; model training/inference: no")
    return 0


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if args.build_evidence_package:
        return build_evidence_package(args, root)
    checkpoint_path = resolve_from_root(root, args.checkpoint)
    pair_scores_path = resolve_from_root(root, args.pair_scores)
    output_dir = resolve_from_root(root, args.output_stage)
    baseline_ranking_path = resolve_from_root(root, args.baseline_ranking)
    module_dir = Path(__file__).resolve().parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    try:
        import torch
        from triage_osa_model import TRIAGEOSAModel
        from osa_context_data import (
            assemble_input_matrix,
            build_pair_context,
            compute_node_evidence,
            load_discovery_inputs,
        )
    except ImportError as exc:
        raise RuntimeError("numpy, pandas and torch are required: {}".format(exc))

    print("TRIAGE-OSA specialization prediction environment")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("torch: {}".format(torch.__version__))
    print("Checkpoint: {}".format(checkpoint_path))
    print("Pair scores: {}".format(pair_scores_path))
    print("External validation read: no")
    missing = [str(path) for path in (
        checkpoint_path, pair_scores_path, baseline_ranking_path
    ) if not path.is_file()]
    if missing:
        raise RuntimeError("Missing prediction inputs: {}".format(missing))
    device_name = str(args.device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("checkpoint_type") != "TRIAGE_Fuse_OSA_specialized_model":
        raise RuntimeError("Checkpoint is not a unified TRIAGE-OSA model")
    if not bool(checkpoint.get("checkpoint_selection_eligible", False)):
        raise RuntimeError(
            "OSA checkpoint failed or predates the locked internal performance gate"
        )
    if bool(checkpoint.get("external_validation_read", True)):
        raise RuntimeError("Checkpoint provenance does not prove validation isolation")
    observed_pair_hash = sha256_file(pair_scores_path)
    expected_pair_hash = checkpoint["training_universe"]["pair_scores_sha256"]
    if observed_pair_hash != expected_pair_hash:
        raise RuntimeError(
            "Pair inference hash differs from OSA training input: expected={} observed={}".format(
                expected_pair_hash, observed_pair_hash
            )
        )
    if args.check_env:
        print("TRIAGE-OSA checkpoint eligibility: PASS")
        print("TRIAGE-OSA prediction preflight: PASS")
        return 0

    pairs = pd.read_csv(pair_scores_path, sep="\t")
    pairs["protein_A"] = pairs["protein_A"].astype(str)
    pairs["protein_B"] = pairs["protein_B"].astype(str)
    proteins = sorted(set(pairs["protein_A"]) | set(pairs["protein_B"]))
    if int(checkpoint["training_universe"]["proteins"]) != len(proteins):
        raise RuntimeError("Checkpoint/prediction protein universe mismatch")
    if int(checkpoint["training_universe"]["pairs"]) != len(pairs):
        raise RuntimeError("Checkpoint/prediction pair universe mismatch")

    expression, metadata, input_paths = load_discovery_inputs(root, proteins)
    observed_data_hashes = dict(
        (name, sha256_file(path)) for name, path in input_paths.items()
    )
    if observed_data_hashes != checkpoint["training_data_sha256"]:
        raise RuntimeError("Discovery input hashes differ from OSA training checkpoint")
    all_samples = metadata["GSE75097"]["sample_id"].astype(str).tolist()
    all_subjects = metadata["GSE71356"]["subject_id"].astype(str).unique().tolist()
    node = compute_node_evidence(
        expression, metadata, proteins, all_samples, all_subjects
    )
    checkpoint_features = list(
        checkpoint.get("osa_model_config", {}).get("input_feature_columns", [])
    )
    information_fusion = "clinical_information_anchor" in checkpoint_features
    context = build_pair_context(
        pairs, node, information_fusion=information_fusion
    )
    features = assemble_input_matrix(
        pairs, context, input_feature_columns=checkpoint_features
    )

    _, osa_state = TRIAGEOSAModel.split_unified_state_dict(
        checkpoint["model_state_dict"]
    )
    model = TRIAGEOSAModel.from_config(checkpoint["osa_model_config"])
    model.load_state_dict(osa_state, strict=True)
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()
    collected = dict(
        (name, [])
        for name in (
            "adapted_logit",
            "adapted_probability",
            "tri_level_logit",
            "context_residual",
            "reliability_blend",
            "context_gates",
            "gate_delta",
            "expert_weights",
        )
    )
    with torch.no_grad():
        for start in range(0, len(features), int(args.batch_size)):
            stop = min(len(features), start + int(args.batch_size))
            batch = torch.from_numpy(features[start:stop]).to(device=device)
            output = model(batch)
            for name in collected:
                collected[name].append(output[name].detach().cpu().numpy())
            print("[predict] pairs {}/{}".format(stop, len(features)))
    output_values = dict(
        (name, np.concatenate(chunks, axis=0)) for name, chunks in collected.items()
    )

    result = pairs.copy()
    for column in context.columns:
        result[column] = context[column].to_numpy()
    result["triage_osa_logit"] = output_values["adapted_logit"]
    result["triage_osa_score"] = output_values["adapted_probability"]
    result["osa_context_tri_level_logit"] = output_values["tri_level_logit"]
    result["osa_context_residual"] = output_values["context_residual"]
    result["osa_reliability_blend"] = output_values["reliability_blend"]
    for index, level in enumerate(("L1", "L2", "L3")):
        result["osa_gate_{}".format(level)] = output_values["context_gates"][:, index]
        result["osa_gate_delta_{}".format(level)] = output_values["gate_delta"][:, index]
    for index, expert in enumerate(("PPI", "L1_L2_L3", "OSA_context")):
        result["osa_expert_weight_{}".format(expert)] = output_values[
            "expert_weights"
        ][:, index]

    result["triage_osa_rank_500"] = result["triage_osa_score"].rank(
        method="first", ascending=False
    ).astype(int)
    result["triage_osa_percentile_500"] = result["triage_osa_score"].rank(
        method="average", pct=True, ascending=True
    )
    primary_mask = result["primary_pair"].astype(str).str.lower().eq("yes")
    result["triage_osa_rank_primary_250"] = np.nan
    result.loc[primary_mask, "triage_osa_rank_primary_250"] = result.loc[
        primary_mask, "triage_osa_score"
    ].rank(method="first", ascending=False)

    comparator_columns = [
        "pair_id", "rank_p_PPI", "rank_r_OSA", "rank_baseline_L3_only",
        "rank_baseline_L1_L2", "rank_baseline_L1_L3",
        "rank_baseline_L2_L3", "rank_baseline_equal_L1_L2_L3",
        "rank_priority_PPI25_context75", "rank_priority_PPI75_context25",
        "rank_priority_bidirectional_conservative",
    ]
    baseline = pd.read_csv(baseline_ranking_path, sep="\t")
    missing_comparators = [column for column in comparator_columns if column not in baseline]
    if missing_comparators or len(baseline) != 31125:
        raise RuntimeError(
            "Baseline comparator ranking is incomplete: {}".format(missing_comparators)
        )
    if baseline["pair_id"].duplicated().any():
        raise RuntimeError("Baseline comparator ranking contains duplicate pairs")
    result = result.merge(
        baseline[comparator_columns], on="pair_id", how="left", validate="one_to_one"
    )
    if result.loc[primary_mask, "rank_p_PPI"].isna().any():
        raise RuntimeError("Primary OSA pairs do not match the frozen comparator ranking")

    # Compatibility aliases allow the validation scripts to consume this new
    # learned ranking after their path is explicitly changed.  Semantics are
    # recorded as priority, never physical PPI probability.
    result["OSA_interaction_priority_score"] = result["triage_osa_score"]
    result["OSA_interaction_priority_rank"] = np.where(
        primary_mask.to_numpy(),
        result["triage_osa_rank_primary_250"].fillna(
            result["triage_osa_rank_500"]
        ).to_numpy(),
        result["triage_osa_rank_500"].to_numpy(),
    ).astype(int)
    result["combined_score_is_physical_PPI_probability"] = "no"
    result["ranking_checkpoint_sha256"] = sha256_file(checkpoint_path)
    result["ranking_model"] = (
        "TRIAGE_Fuse_OSA_patient_information_v4"
        if information_fusion else "TRIAGE_Fuse_OSA_specialized_v3_2"
    )

    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force: {}".format(output_dir))
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "pair_context_scores_triage_osa_500.tsv.gz"
    primary_path = output_dir / "pair_context_scores_triage_osa_primary_250.tsv.gz"
    result.to_csv(all_path, sep="\t", index=False, compression="gzip")
    result.loc[primary_mask].sort_values(
        "triage_osa_rank_primary_250", kind="mergesort"
    ).to_csv(primary_path, sep="\t", index=False, compression="gzip")
    node.to_csv(output_dir / "full_discovery_node_evidence.tsv", sep="\t", index=False)
    result.sort_values("triage_osa_rank_500", kind="mergesort").head(1000).to_csv(
        output_dir / "top_1000_triage_osa_pairs.tsv", sep="\t", index=False
    )

    summary = {
        "status": "PASS_OSA_CONTEXT_ADAPTER_PREDICTION",
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "base_checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
        "mean_expert_weights": {
            expert: float(np.mean(output_values["expert_weights"][:, index]))
            for index, expert in enumerate(("PPI", "L1_L2_L3", "OSA_context"))
        },
        "pair_scores": str(pair_scores_path),
        "pair_scores_sha256": observed_pair_hash,
        "baseline_comparator_ranking": str(baseline_ranking_path),
        "baseline_comparator_ranking_sha256": sha256_file(baseline_ranking_path),
        "proteins": int(len(proteins)),
        "pairs": int(len(pairs)),
        "primary_pairs": int(primary_mask.sum()),
        "output_all": str(all_path),
        "output_all_sha256": sha256_file(all_path),
        "output_primary": str(primary_path),
        "output_primary_sha256": sha256_file(primary_path),
        "score_semantics": "OSA interaction-priority score; not physical PPI probability",
        "patient_information_conditioning": bool(information_fusion),
        "patient_information_fields": [
            "age", "sex", "BMI", "AHI", "hypertension", "EDS"
        ],
        "patient_information_availability": node.attrs.get(
            "patient_information_availability", {}
        ),
        "clinical_outcome_prediction": False,
        "external_validation_read": False,
        "ranking_changed_by_validation": False,
    }
    write_json(output_dir / "run_summary.json", summary)
    print("TRIAGE-OSA specialization prediction: PASS")
    print("Pairs: {}".format(len(result)))
    print("Ranking: {}".format(all_path))
    print("External validation read: no")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
