#!/usr/bin/env python3
"""Preprocess and fit the two locked paired CPAP validation cohorts.

Python 3.7 compatible.  The script implements the frozen validation contract
and does not read known-PPI/pathway labels or alter the frozen pair ranking.
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
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.2.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LOCK_REL = "config/osa_cpap_validation_lock.json"
SAMPLE_MANIFEST_REL = "metadata/sample_manifest.tsv"
CANDIDATE_REL = (
    "results/discovery_preprocess_v2/04_candidates_v1/"
    "candidate_proteins_primary_250.tsv"
)
DEFAULT_STAGE = "10_cpap_validation_preprocess_v1"
DEFAULT_CLINICAL_STAGE = "25_cpap_subject_clinical_package_v1"
DEFAULT_PREPROCESS_STAGE = "10_cpap_validation_preprocess_v1"
DEFAULT_PATIENT_RANKING = (
    "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/"
    "pair_context_scores_triage_osa_primary_250.tsv.gz"
)
COHORTS = {
    "GSE133601": {
        "matrix": "geo/GSE133601_series_matrix.txt.gz",
        "annotation": None,
        "subjects": 15,
        "transform": "deposited_log2_Brainarray_RMA",
    },
    "GSE49800": {
        "matrix": "geo/GSE49800_series_matrix.txt.gz",
        "annotation": "annotations/GPL6244.annot.gz",
        "subjects": 18,
        "transform": "log2_deposited_positive_RMA",
    },
}


CLINICAL_FIELDS = [
    "age_years", "sex", "race_ethnicity", "BMI_kg_m2", "obesity",
    "smoking_status", "baseline_AHI_events_h", "baseline_avg_SpO2_percent",
    "baseline_min_SpO2_percent", "baseline_ODI_events_h",
    "baseline_T90_percent", "baseline_ESS_score", "baseline_SBP_mmHg",
    "baseline_DBP_mmHg", "hypertension", "diabetes", "dyslipidemia",
    "cardiovascular_disease", "CPAP_duration_days",
    "CPAP_adherence_hours_night", "CPAP_percent_nights_ge4h",
    "CPAP_pressure_cmH2O", "post_AHI_events_h", "post_avg_SpO2_percent",
    "post_min_SpO2_percent", "post_ODI_events_h", "post_T90_percent",
    "post_ESS_score", "post_SBP_mmHg", "post_DBP_mmHg",
]


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


def parse_series_matrix(np, path):
    samples = None
    identifiers = []
    values = []
    in_table = False
    with gzip.open(str(path), "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table:
                continue
            fields = next(csv.reader([line], delimiter="\t"))
            if samples is None:
                if fields[0] != "ID_REF":
                    raise RuntimeError("Unexpected series matrix table header")
                samples = fields[1:]
                continue
            identifiers.append(fields[0])
            values.append([float(value) for value in fields[1:]])
    if samples is None or not values:
        raise RuntimeError("Series matrix table is absent: " + str(path))
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (len(identifiers), len(samples)):
        raise RuntimeError("Series matrix shape mismatch")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Series matrix contains non-finite values")
    return identifiers, samples, matrix


def parse_gpl6244(path):
    mapping = {}
    in_table = False
    fields = None
    with gzip.open(str(path), "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if not in_table:
                continue
            row = next(csv.reader([line], delimiter="\t"))
            if fields is None:
                fields = row
                continue
            record = dict(zip(fields, row))
            gene_ids = sorted(set(
                value.strip() for value in record.get("Gene ID", "").split("///")
                if value.strip().isdigit()
            ))
            if len(gene_ids) == 1:
                mapping[record["ID"]] = gene_ids[0]
    if not mapping:
        raise RuntimeError("No single-Entrez GPL6244 mappings were parsed")
    return mapping


def map_and_aggregate(np, dataset, identifiers, matrix, annotation):
    feature_to_entrez = {}
    excluded_zero = 0
    excluded_multi = 0
    if dataset == "GSE133601":
        for feature in identifiers:
            match = re.match(r"^(\d+)_at$", feature)
            if match:
                feature_to_entrez[feature] = match.group(1)
            else:
                excluded_zero += 1
    else:
        single_mapping = parse_gpl6244(annotation)
        for feature in identifiers:
            if feature in single_mapping:
                feature_to_entrez[feature] = single_mapping[feature]
            else:
                excluded_multi += 1
    grouped = {}
    for index, feature in enumerate(identifiers):
        entrez = feature_to_entrez.get(feature)
        if entrez is not None:
            grouped.setdefault(entrez, []).append(matrix[index, :])
    ordered = sorted(grouped, key=lambda value: (int(value), value))
    aggregated = np.stack([
        np.median(np.stack(grouped[entrez], axis=0), axis=0)
        for entrez in ordered
    ], axis=0)
    aggregation_count = {entrez: len(grouped[entrez]) for entrez in ordered}
    audit = {
        "input_features": len(identifiers),
        "single_Entrez_features": len(feature_to_entrez),
        "excluded_unmapped_or_non_Entrez": excluded_zero,
        "excluded_non_single_GPL_mapping": excluded_multi,
        "aggregated_Entrez_features": len(ordered),
        "Entrez_with_multiple_features": sum(value > 1 for value in aggregation_count.values()),
    }
    return ordered, aggregated, aggregation_count, audit


def paired_metadata(sample_manifest, dataset, matrix_samples, expected_subjects):
    rows = [row for row in sample_manifest if row["dataset"] == dataset]
    if len(rows) != 2 * expected_subjects:
        raise RuntimeError(dataset + " manifest assay count mismatch")
    by_sample = {row["sample_id"]: row for row in rows}
    if set(by_sample) != set(matrix_samples):
        raise RuntimeError(dataset + " matrix/manifest sample identity mismatch")
    by_subject = {}
    for row in rows:
        by_subject.setdefault(row["subject_id"], {})[row["treatment"]] = row["sample_id"]
        if row["candidate_selection_allowed"] != "no":
            raise RuntimeError(dataset + " validation isolation flag failure")
    if len(by_subject) != expected_subjects:
        raise RuntimeError(dataset + " subject count mismatch")
    for subject, pair in by_subject.items():
        if set(pair) != {"pre_CPAP", "post_CPAP"}:
            raise RuntimeError(dataset + " incomplete pair: " + subject)
    subjects = sorted(
        by_subject,
        key=lambda value: (int(value) if value.isdigit() else 10 ** 20, value),
    )
    return by_subject, subjects


def sample_qc(np, matrix, samples, metadata):
    correlation = np.corrcoef(matrix.T)
    if not np.isfinite(correlation).all():
        raise RuntimeError("Non-finite sample correlation matrix")
    medians = np.median(matrix, axis=0)
    center = float(np.median(medians))
    mad = float(np.median(np.abs(medians - center)))
    rows = []
    for index, sample in enumerate(samples):
        robust_z = (medians[index] - center) / max(1.4826 * mad, 1e-12)
        rows.append({
            "sample_id": sample,
            "subject_id": metadata[sample]["subject_id"],
            "treatment": metadata[sample]["treatment"],
            "median_expression": fmt(medians[index]),
            "MAD_expression": fmt(np.median(np.abs(matrix[:, index] - medians[index]))),
            "median_correlation_to_other_samples": fmt(np.median(np.delete(correlation[index], index))),
            "median_expression_robust_z": fmt(robust_z),
            "technical_outlier_flag_abs_z_gt_5": "yes" if abs(robust_z) > 5.0 else "no",
            "excluded_from_primary": "no",
        })
    return rows


def fit_paired(np, entrez_ids, expression, samples, sample_meta, by_subject, subjects):
    sample_index = {sample: index for index, sample in enumerate(samples)}
    deltas = np.stack([
        expression[:, sample_index[by_subject[subject]["post_CPAP"]]]
        - expression[:, sample_index[by_subject[subject]["pre_CPAP"]]]
        for subject in subjects
    ], axis=1)
    n = len(subjects)
    residual_df = n - 1
    beta = np.mean(deltas, axis=1)
    s2 = np.var(deltas, axis=1, ddof=1)
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from osa_discovery_differential import (
        bh_adjust, estimate_prior_variance, student_t_two_sided,
    )
    prior_s2, prior_df, observed_log_variance, sampling_component, boundary = (
        estimate_prior_variance(s2, residual_df, np)
    )
    posterior_s2 = (prior_df * prior_s2 + residual_df * s2) / (prior_df + residual_df)
    unmoderated_se = np.sqrt(s2 / float(n))
    moderated_se = np.sqrt(posterior_s2 / float(n))
    unmoderated_t = beta / unmoderated_se
    moderated_t = beta / moderated_se
    unmoderated_p = np.asarray([
        student_t_two_sided(float(value), float(residual_df))
        for value in unmoderated_t
    ])
    moderated_df = residual_df + prior_df
    moderated_p = np.asarray([
        student_t_two_sided(float(value), float(moderated_df))
        for value in moderated_t
    ])
    unmoderated_fdr = bh_adjust(unmoderated_p, np)
    moderated_fdr = bh_adjust(moderated_p, np)
    rows = []
    for index, entrez in enumerate(entrez_ids):
        rows.append({
            "Entrez": entrez,
            "mean_expression": fmt(np.mean(expression[index, :])),
            "CPAP_beta_post_minus_pre": fmt(beta[index]),
            "unmoderated_SE": fmt(unmoderated_se[index]),
            "unmoderated_t": fmt(unmoderated_t[index]),
            "unmoderated_P_value": fmt(unmoderated_p[index]),
            "unmoderated_FDR_BH": fmt(unmoderated_fdr[index]),
            "moderated_SE": fmt(moderated_se[index]),
            "moderated_t": fmt(moderated_t[index]),
            "P_value": fmt(moderated_p[index]),
            "FDR_BH": fmt(moderated_fdr[index]),
            "residual_variance": fmt(s2[index]),
            "posterior_variance": fmt(posterior_s2[index]),
            "n_subjects": str(n),
            "n_assays": str(2 * n),
            "design": "paired_post_minus_pre_equivalent_to_subject_plus_timepoint",
            "candidate_selection_allowed": "no",
        })
    fit = {
        "n_subjects": n,
        "residual_df": residual_df,
        "prior_variance": prior_s2,
        "prior_df": prior_df,
        "observed_log_residual_variance": observed_log_variance,
        "sampling_log_variance_component": sampling_component,
        "prior_df_boundary": boundary,
        "moderated_df": moderated_df,
    }
    return rows, deltas, fit


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def median(values):
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def pearson(x_values, y_values):
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return None
    x_mean = sum(x_values) / float(len(x_values))
    y_mean = sum(y_values) / float(len(y_values))
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_scale <= 0.0 or y_scale <= 0.0:
        return None
    return numerator / (x_scale * y_scale)


def clinical_dictionary_rows():
    definitions = {
        "age_years": ("demographic", "Age at baseline", "years", "baseline predictor"),
        "sex": ("demographic", "Sex", "categorical", "baseline predictor"),
        "race_ethnicity": ("demographic", "Race or ethnicity", "categorical", "baseline predictor/audit"),
        "BMI_kg_m2": ("anthropometric", "Body mass index", "kg/m2", "baseline predictor"),
        "obesity": ("anthropometric", "BMI-defined obesity", "yes/no", "derived baseline predictor"),
        "smoking_status": ("demographic", "Smoking status", "categorical", "baseline predictor"),
        "baseline_AHI_events_h": ("OSA", "Baseline apnea-hypopnea index", "events/hour", "baseline predictor"),
        "baseline_avg_SpO2_percent": ("OSA", "Mean sleep oxygen saturation", "percent", "baseline predictor"),
        "baseline_min_SpO2_percent": ("OSA", "Minimum sleep oxygen saturation", "percent", "baseline predictor"),
        "baseline_ODI_events_h": ("OSA", "Oxygen desaturation index", "events/hour", "baseline predictor"),
        "baseline_T90_percent": ("OSA", "Sleep time below 90 percent saturation", "percent", "baseline predictor"),
        "baseline_ESS_score": ("symptom", "Epworth Sleepiness Scale", "0-24", "baseline predictor"),
        "baseline_SBP_mmHg": ("cardiovascular", "Baseline systolic blood pressure", "mmHg", "baseline predictor"),
        "baseline_DBP_mmHg": ("cardiovascular", "Baseline diastolic blood pressure", "mmHg", "baseline predictor"),
        "hypertension": ("comorbidity", "Hypertension", "yes/no", "baseline predictor"),
        "diabetes": ("comorbidity", "Diabetes", "yes/no", "baseline predictor"),
        "dyslipidemia": ("comorbidity", "Dyslipidemia", "yes/no", "baseline predictor"),
        "cardiovascular_disease": ("comorbidity", "Cardiovascular disease", "yes/no", "baseline predictor"),
        "CPAP_duration_days": ("treatment", "Baseline-to-follow-up duration", "days", "treatment exposure"),
        "CPAP_adherence_hours_night": ("treatment", "Average nightly CPAP use", "hours/night", "treatment exposure"),
        "CPAP_percent_nights_ge4h": ("treatment", "Nights with at least 4 hours use", "percent", "treatment exposure"),
        "CPAP_pressure_cmH2O": ("treatment", "Therapeutic CPAP pressure", "cmH2O", "treatment exposure"),
        "post_AHI_events_h": ("outcome", "Post-treatment AHI", "events/hour", "outcome only"),
        "post_avg_SpO2_percent": ("outcome", "Post-treatment mean oxygen saturation", "percent", "outcome only"),
        "post_min_SpO2_percent": ("outcome", "Post-treatment minimum oxygen saturation", "percent", "outcome only"),
        "post_ODI_events_h": ("outcome", "Post-treatment oxygen desaturation index", "events/hour", "outcome only"),
        "post_T90_percent": ("outcome", "Post-treatment time below 90 percent saturation", "percent", "outcome only"),
        "post_ESS_score": ("outcome", "Post-treatment Epworth Sleepiness Scale", "0-24", "outcome only"),
        "post_SBP_mmHg": ("outcome", "Post-treatment systolic blood pressure", "mmHg", "outcome only"),
        "post_DBP_mmHg": ("outcome", "Post-treatment diastolic blood pressure", "mmHg", "outcome only"),
    }
    return [{
        "field": field,
        "domain": definitions[field][0],
        "definition": definitions[field][1],
        "unit_or_level": definitions[field][2],
        "model_role": definitions[field][3],
        "missing_value": "blank",
        "individual_value_required": "yes",
    } for field in CLINICAL_FIELDS]


def cohort_public_summary(dataset):
    if dataset == "GSE133601":
        return (
            "HeartBEAT transcriptomic subset: n=15; age 65.9 (SD 7.7) years; "
            "male 14/15 (93.3%); BMI 31.5 (SD 3.3) kg/m2; AHI 36.9 "
            "(SD 9.4) events/hour; mean SpO2 93.3% (SD 1.7); minimum SpO2 "
            "77.7% (SD 7.2); CPAP adherence 5.7 (SD 1.9) hours/night."
        )
    return (
        "Published GSE49800 summary: n=18; mean age 48.8 years; 50% male; "
        "61% African American; severe OSA; CPAP use 6.9 (SD 1.0) hours/night "
        "during the 14 days before follow-up; pressure 12.2 (SD 2.5) cmH2O; "
        "median follow-up 33 days (IQR 27-52)."
    )


def build_clinical_package(
    root, preprocess_stage, output_stage, clinical_input, ranking_path, force
):
    preprocess_dir = root / "results" / "discovery_preprocess_v2" / preprocess_stage
    sample_manifest_path = root / SAMPLE_MANIFEST_REL
    required = [sample_manifest_path, preprocess_dir / "run_summary.json", ranking_path]
    for dataset in COHORTS:
        required += [
            preprocess_dir / dataset / "sample_technical_qc.tsv",
            preprocess_dir / dataset / "candidate_validation_evidence.tsv",
            preprocess_dir / dataset / "candidate_participant_deltas.tsv.gz",
        ]
    for path in required:
        if not path.is_file():
            raise RuntimeError("Missing clinical-package input: " + str(path))

    sample_manifest, _ = read_tsv(sample_manifest_path)
    ranking_rows, ranking_fields = read_tsv(ranking_path)
    required_ranking = {
        "protein_A", "protein_B", "triage_osa_rank_primary_250", "pair_id"
    }
    if not required_ranking.issubset(set(ranking_fields)) or len(ranking_rows) != 31125:
        raise RuntimeError("V4 patient ranking is incomplete")
    incident = {}
    denominator = max(1.0, float(len(ranking_rows) - 1))
    for row in ranking_rows:
        rank = float(row["triage_osa_rank_primary_250"])
        percentile = 1.0 - (rank - 1.0) / denominator
        for protein in (row["protein_A"], row["protein_B"]):
            incident.setdefault(protein, []).append(percentile)
    node_priority = {}
    for protein, values in incident.items():
        values = sorted(values, reverse=True)
        node_priority[protein] = sum(values[:10]) / float(min(10, len(values)))
    priority_values = sorted(node_priority.values())
    priority_cutoff = priority_values[int(0.80 * (len(priority_values) - 1))]
    supplied = {}
    if clinical_input is not None:
        clinical_input = clinical_input.expanduser().resolve()
        clinical_rows, fields = read_tsv(clinical_input)
        for key in ["dataset", "subject_id"]:
            if key not in fields:
                raise RuntimeError("Clinical input lacks key field: " + key)
        for row in clinical_rows:
            key = (row["dataset"], row["subject_id"])
            if key in supplied:
                raise RuntimeError("Duplicate clinical subject: {} {}".format(*key))
            supplied[key] = row

    output_dir = root / "results" / "discovery_preprocess_v2" / output_stage
    if output_dir.exists():
        if not force:
            raise RuntimeError("Clinical package exists; use --force to replace")
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    reports_dir = output_dir / "subject_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    clinical_manifest_rows = []
    subject_summary_rows = []
    report_index_rows = []
    outputs = []

    for dataset in COHORTS:
        manifest_rows = [row for row in sample_manifest if row["dataset"] == dataset]
        by_subject = {}
        for row in manifest_rows:
            by_subject.setdefault(row["subject_id"], {})[row["treatment"]] = row
        qc_rows, _ = read_tsv(preprocess_dir / dataset / "sample_technical_qc.tsv")
        qc_by_sample = {row["sample_id"]: row for row in qc_rows}
        evidence_rows, _ = read_tsv(preprocess_dir / dataset / "candidate_validation_evidence.tsv")
        evidence = {row["UniProt_accession"]: row for row in evidence_rows
                    if row.get("validation_available") == "yes"}
        delta_rows, _ = read_tsv(preprocess_dir / dataset / "candidate_participant_deltas.tsv.gz")

        subjects = sorted(by_subject, key=lambda v: (int(v) if v.isdigit() else 10 ** 20, v))
        for subject in subjects:
            pair = by_subject[subject]
            if set(pair) != {"pre_CPAP", "post_CPAP"}:
                raise RuntimeError("Incomplete pair: {} {}".format(dataset, subject))
            pre, post = pair["pre_CPAP"], pair["post_CPAP"]
            clinical = supplied.get((dataset, subject), {})
            available_count = sum(str(clinical.get(field, "")).strip() != ""
                                  for field in CLINICAL_FIELDS)
            clinical_row = {
                "dataset": dataset, "subject_id": subject,
                "pre_sample_id": pre["sample_id"], "post_sample_id": post["sample_id"],
                "tissue": pre["tissue"], "pair_complete": "yes",
                "individual_clinical_fields_available": str(available_count),
                "individual_clinical_fields_expected": str(len(CLINICAL_FIELDS)),
                "clinical_data_status": ("individual_values_supplied" if available_count
                                         else "not_public_in_GEO_crosswalk_required"),
                "clinical_value_source": str(clinical.get("clinical_value_source", "")),
                "source_record_id": str(clinical.get("source_record_id", "")),
                "source_version": str(clinical.get("source_version", "")),
                "linkage_qc_status": str(clinical.get("linkage_qc_status", "not_run")),
                "public_cohort_summary_not_individual": cohort_public_summary(dataset),
            }
            for field in CLINICAL_FIELDS:
                clinical_row[field] = str(clinical.get(field, ""))
            clinical_manifest_rows.append(clinical_row)

            oriented, subject_delta, cohort_delta, ranked = [], [], [], []
            weighted_oriented, weighted_values = [], []
            subject_field = "subject_" + subject
            for row in delta_rows:
                value = safe_float(row.get(subject_field))
                item = evidence.get(row["UniProt_accession"])
                if value is None or item is None:
                    continue
                sign = 1.0 if item["discovery_OSA_beta_sign"] == "positive" else -1.0
                oriented_value = -sign * value
                cohort_value = safe_float(item.get("CPAP_beta_post_minus_pre"))
                if cohort_value is None:
                    continue
                oriented.append(oriented_value)
                priority = float(node_priority.get(row["UniProt_accession"], 0.0))
                weighted_oriented.append(oriented_value * priority)
                weighted_values.append(priority)
                subject_delta.append(value)
                cohort_delta.append(cohort_value)
                ranked.append((oriented_value, row["gene_symbol"], row["UniProt_accession"], value))
            ranked.sort(reverse=True)
            reversal_count = sum(value > 0.0 for value in oriented)
            top_priority = [
                item for item in ranked
                if node_priority.get(item[2], 0.0) >= priority_cutoff
            ]
            weighted_mean = (
                sum(weighted_oriented) / sum(weighted_values)
                if sum(weighted_values) > 0.0 else float("nan")
            )
            summary_row = {
                "dataset": dataset, "subject_id": subject,
                "pre_sample_id": pre["sample_id"], "post_sample_id": post["sample_id"],
                "candidate_proteins_measured": str(len(oriented)),
                "directionally_reversed_candidates": str(reversal_count),
                "directional_reversal_fraction": fmt(reversal_count / float(len(oriented))),
                "mean_oriented_post_minus_pre": fmt(sum(oriented) / float(len(oriented))),
                "median_oriented_post_minus_pre": fmt(median(oriented)),
                "V4_priority_weighted_mean_oriented_post_minus_pre": fmt(weighted_mean),
                "V4_top20_priority_proteins_measured": str(len(top_priority)),
                "V4_top20_priority_reversal_fraction": fmt(
                    sum(item[0] > 0.0 for item in top_priority) / float(len(top_priority))
                    if top_priority else float("nan")
                ),
                "correlation_with_cohort_mean_CPAP_delta": fmt(pearson(subject_delta, cohort_delta)),
                "pre_median_sample_correlation": qc_by_sample[pre["sample_id"]]["median_correlation_to_other_samples"],
                "post_median_sample_correlation": qc_by_sample[post["sample_id"]]["median_correlation_to_other_samples"],
                "pre_technical_outlier": qc_by_sample[pre["sample_id"]]["technical_outlier_flag_abs_z_gt_5"],
                "post_technical_outlier": qc_by_sample[post["sample_id"]]["technical_outlier_flag_abs_z_gt_5"],
                "individual_clinical_fields_available": str(available_count),
                "individual_clinical_information_fraction": fmt(
                    available_count / float(len(CLINICAL_FIELDS))
                ),
                "interpretation_scope": "research_only_not_a_clinical_diagnosis",
            }
            subject_summary_rows.append(summary_row)

            report_path = reports_dir / "{}_subject_{}.md".format(dataset, subject)
            lines = [
                "# CPAP paired molecular response report", "",
                "- Dataset: {}".format(dataset), "- Subject ID: {}".format(subject),
                "- Pre-CPAP sample: {}".format(pre["sample_id"]),
                "- Post-CPAP sample: {}".format(post["sample_id"]),
                "- Tissue: {}".format(pre["tissue"]), "", "## Data integrity", "",
                "- Complete pre/post pair: yes",
                "- Median sample correlation (pre/post): {}/{}".format(
                    summary_row["pre_median_sample_correlation"], summary_row["post_median_sample_correlation"]),
                "- Technical outlier flags (pre/post): {}/{}".format(
                    summary_row["pre_technical_outlier"], summary_row["post_technical_outlier"]),
                "", "## Molecular response summary", "",
                "- Measured frozen candidate proteins: {}".format(len(oriented)),
                "- Directionally reversed candidates: {} ({:.1f}%)".format(
                    reversal_count, 100.0 * reversal_count / float(len(oriented))),
                "- Mean oriented post-minus-pre effect: {}".format(summary_row["mean_oriented_post_minus_pre"]),
                "- Median oriented post-minus-pre effect: {}".format(summary_row["median_oriented_post_minus_pre"]),
                "- V4-priority-weighted mean oriented effect: {}".format(
                    summary_row["V4_priority_weighted_mean_oriented_post_minus_pre"]),
                "- V4 top-20% priority reversal fraction: {}".format(
                    summary_row["V4_top20_priority_reversal_fraction"]),
                "- Correlation with cohort-average CPAP delta: {}".format(
                    summary_row["correlation_with_cohort_mean_CPAP_delta"]), "",
                "A positive oriented effect means the participant's CPAP-associated change is opposite to the frozen discovery OSA direction. This is descriptive molecular evidence, not proof of clinical benefit.",
                "", "## Strongest candidate-level reversals", "",
                "| Gene | UniProt | Raw post-pre delta | Oriented reversal effect |",
                "|---|---|---:|---:|",
            ]
            for effect, gene, uniprot, raw in ranked[:10]:
                lines.append("| {} | {} | {} | {} |".format(gene, uniprot, fmt(raw), fmt(effect)))
            lines += ["", "## Strongest same-direction changes", "",
                      "| Gene | UniProt | Raw post-pre delta | Oriented reversal effect |",
                      "|---|---|---:|---:|"]
            for effect, gene, uniprot, raw in sorted(ranked)[:10]:
                lines.append("| {} | {} | {} | {} |".format(gene, uniprot, fmt(raw), fmt(effect)))
            lines += [
                "", "## Individual clinical variables", "",
                "Available individual clinical fields: {}/{}.".format(available_count, len(CLINICAL_FIELDS)), "",
                "GEO does not publish age, sex, BMI, AHI, oxygenation, ESS, blood pressure, or CPAP exposure for this individual. Cohort-level paper values must not be copied into the individual record. A verified external subject-ID crosswalk is required.",
                "", "## Scope", "",
                "Research quality-control and hypothesis-generation report only; not a diagnosis or treatment recommendation.", "",
            ]
            report_path.write_text("\n".join(lines), encoding="utf-8")
            outputs.append(report_path)
            report_index_rows.append({"dataset": dataset, "subject_id": subject,
                                      "report_path": relative_path(report_path, root),
                                      "clinical_data_status": clinical_row["clinical_data_status"]})

    variable_path = output_dir / "clinical_variable_dictionary.tsv"
    clinical_path = output_dir / "clinical_subject_manifest.tsv"
    subject_path = output_dir / "subject_molecular_response_summary.tsv"
    report_index_path = output_dir / "subject_report_index.tsv"
    variable_rows = clinical_dictionary_rows()
    write_tsv(variable_path, variable_rows, list(variable_rows[0]))
    write_tsv(clinical_path, clinical_manifest_rows, list(clinical_manifest_rows[0]))
    write_tsv(subject_path, subject_summary_rows, list(subject_summary_rows[0]))
    write_tsv(report_index_path, report_index_rows, list(report_index_rows[0]))
    outputs += [variable_path, clinical_path, subject_path, report_index_path]

    complete = sum(int(row["individual_clinical_fields_available"]) == len(CLINICAL_FIELDS)
                   for row in clinical_manifest_rows)
    status = ("PASS_CLINICAL_LINKAGE_COMPLETE" if complete == len(clinical_manifest_rows)
              else "PASS_PUBLIC_MOLECULAR_REPORTS_CLINICAL_CROSSWALK_REQUIRED")
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, {
        "created_at_utc": utc_now(), "status": status,
        "script_version": SCRIPT_VERSION, "subjects": len(clinical_manifest_rows),
        "GSE133601_subjects": 15, "GSE49800_subjects": 18,
        "individual_reports": len(report_index_rows),
        "complete_individual_clinical_records": complete,
        "public_GEO_individual_clinical_data_available": False,
        "cohort_summary_values_assigned_to_individuals": False,
        "V4_ranking": str(ranking_path),
        "V4_ranking_sha256": sha256_file(ranking_path),
        "clinical_outcome_prediction_performed": False,
        "validation_ranking_changed": False,
        "validation_data_used_for_model_training": False,
        "next_gate": "Obtain a verified clinical-to-GEO subject crosswalk and rerun with --clinical-input.",
    })
    outputs.append(summary_path)
    output_manifest_rows = [{"path": relative_path(path, root), "bytes": str(path.stat().st_size),
                             "sha256": sha256_file(path)} for path in outputs]
    output_manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(output_manifest_path, output_manifest_rows, ["path", "bytes", "sha256"])
    print("OSA CPAP subject clinical package: " + status)
    print("Subjects: {} reports: {}".format(len(clinical_manifest_rows), len(report_index_rows)))
    print("Output: {}".format(output_dir))
    print("Validation ranking changed: no; validation used for model training: no")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-stage", default=DEFAULT_STAGE)
    parser.add_argument("--build-clinical-package", action="store_true")
    parser.add_argument("--preprocess-stage", default=DEFAULT_PREPROCESS_STAGE)
    parser.add_argument("--clinical-output-stage", default=DEFAULT_CLINICAL_STAGE)
    parser.add_argument("--clinical-input", type=Path)
    parser.add_argument("--patient-ranking", default=DEFAULT_PATIENT_RANKING)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.build_clinical_package:
        ranking_path = Path(args.patient_ranking)
        if not ranking_path.is_absolute():
            ranking_path = root / ranking_path
        if args.check_env:
            clinical_dir = root / "results" / "discovery_preprocess_v2" / args.preprocess_stage
            required = [
                root / SAMPLE_MANIFEST_REL, clinical_dir / "run_summary.json",
                ranking_path,
            ]
            for dataset in COHORTS:
                required += [
                    clinical_dir / dataset / "sample_technical_qc.tsv",
                    clinical_dir / dataset / "candidate_validation_evidence.tsv",
                    clinical_dir / dataset / "candidate_participant_deltas.tsv.gz",
                ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError("Missing clinical-package inputs: " + ", ".join(missing))
            print("OSA CPAP subject clinical package environment: PASS")
            print("Public GEO individual clinical fields: unavailable")
            print("Expected subjects: GSE133601=15 GSE49800=18")
            print("Validation data training allowed: no")
            return 0
        return build_clinical_package(
            root, args.preprocess_stage, args.clinical_output_stage,
            args.clinical_input, ranking_path, args.force,
        )
    lock_path = root / LOCK_REL
    sample_manifest_path = root / SAMPLE_MANIFEST_REL
    candidate_path = root / CANDIDATE_REL
    required = [lock_path, sample_manifest_path, candidate_path]
    for config in COHORTS.values():
        required.append(root / config["matrix"])
        if config["annotation"]:
            required.append(root / config["annotation"])
    for path in required:
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    hash_paths = {
        "GSE133601_series_matrix.txt.gz": root / COHORTS["GSE133601"]["matrix"],
        "GSE49800_series_matrix.txt.gz": root / COHORTS["GSE49800"]["matrix"],
        "GPL6244.annot.gz": root / COHORTS["GSE49800"]["annotation"],
        "sample_manifest.tsv": sample_manifest_path,
        "candidate_proteins_primary_250.tsv": candidate_path,
    }
    observed_hashes = {}
    for name, path in hash_paths.items():
        observed_hashes[name] = sha256_file(path)
        if observed_hashes[name] != lock["input_hashes"][name]:
            raise RuntimeError("Frozen validation input hash mismatch: " + name)
    candidates, _ = read_tsv(candidate_path)
    sample_manifest, _ = read_tsv(sample_manifest_path)
    if len(candidates) != 250:
        raise RuntimeError("Primary candidate file is not 250 proteins")
    candidate_by_entrez = {row["Entrez"]: row for row in candidates}
    print("OSA CPAP validation preprocessing environment: PASS")
    print("Ranking frozen: yes")
    print("Validation cohorts: GSE133601 (15 pairs), GSE49800 (18 pairs)")
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
    summary = {
        "created_at_utc": utc_now(),
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "validation_lock": {"path": LOCK_REL, "sha256": sha256_file(lock_path)},
        "input_hashes": observed_hashes,
        "ranking_read": False,
        "known_PPI_or_pathway_read": False,
        "candidate_membership_changed": False,
        "cohorts": {},
    }
    outputs = []
    for dataset, config in COHORTS.items():
        identifiers, samples, matrix = parse_series_matrix(
            np, root / config["matrix"]
        )
        raw_quantiles = [float(value) for value in np.quantile(
            matrix, [0.0, 0.5, 0.99, 1.0]
        )]
        if dataset == "GSE49800":
            if (matrix <= 0.0).any():
                raise RuntimeError("GSE49800 contains non-positive values before log2")
            matrix = np.log2(matrix)
        entrez_ids, expression, aggregation_count, mapping_audit = map_and_aggregate(
            np, dataset, identifiers, matrix,
            root / config["annotation"] if config["annotation"] else None,
        )
        by_subject, subjects = paired_metadata(
            sample_manifest, dataset, samples, config["subjects"]
        )
        sample_meta = {
            row["sample_id"]: row for row in sample_manifest
            if row["dataset"] == dataset
        }
        qc_rows = sample_qc(np, expression, samples, sample_meta)
        stats_rows, deltas, fit_summary = fit_paired(
            np, entrez_ids, expression, samples, sample_meta,
            by_subject, subjects,
        )
        stats_by_entrez = {row["Entrez"]: row for row in stats_rows}
        index_by_entrez = {entrez: index for index, entrez in enumerate(entrez_ids)}
        candidate_rows = []
        delta_rows = []
        for candidate in candidates:
            entrez = candidate["Entrez"]
            stat = stats_by_entrez.get(entrez)
            row = {
                "dataset": dataset,
                "UniProt_accession": candidate["UniProt_accession"],
                "gene_symbol": candidate["gene_symbol_current"],
                "Entrez": entrez,
                "discovery_rank": candidate["discovery_rank"],
                "discovery_OSA_beta": candidate["OSA_beta"],
                "discovery_OSA_beta_sign": candidate["OSA_beta_sign"],
                "validation_available": "yes" if stat is not None else "no",
            }
            stat_fields = [
                "mean_expression", "CPAP_beta_post_minus_pre",
                "unmoderated_SE", "unmoderated_t", "unmoderated_P_value",
                "unmoderated_FDR_BH", "moderated_SE", "moderated_t",
                "P_value", "FDR_BH", "residual_variance",
                "posterior_variance", "n_subjects", "n_assays", "design",
                "candidate_selection_allowed",
            ]
            for field in stat_fields:
                row[field] = stat[field] if stat is not None else ""
            candidate_rows.append(row)
            delta_row = {
                "dataset": dataset,
                "UniProt_accession": candidate["UniProt_accession"],
                "gene_symbol": candidate["gene_symbol_current"],
                "Entrez": entrez,
                "validation_available": "yes" if stat is not None else "no",
            }
            if stat is not None:
                values = deltas[index_by_entrez[entrez], :]
                for subject, value in zip(subjects, values):
                    delta_row["subject_" + subject] = fmt(value)
            else:
                for subject in subjects:
                    delta_row["subject_" + subject] = ""
            delta_rows.append(delta_row)
        cohort_dir = output_dir / dataset
        cohort_dir.mkdir(parents=True, exist_ok=True)
        stats_path = cohort_dir / "paired_differential_all_entrez.tsv.gz"
        candidate_stats_path = cohort_dir / "candidate_validation_evidence.tsv"
        deltas_path = cohort_dir / "candidate_participant_deltas.tsv.gz"
        qc_path = cohort_dir / "sample_technical_qc.tsv"
        stats_fields = list(stats_rows[0])
        candidate_fields = list(candidate_rows[0])
        delta_fields = list(delta_rows[0])
        write_gzip_tsv(stats_path, stats_rows, stats_fields)
        write_tsv(candidate_stats_path, candidate_rows, candidate_fields)
        write_gzip_tsv(deltas_path, delta_rows, delta_fields)
        write_tsv(qc_path, qc_rows, list(qc_rows[0]))
        outputs.extend([stats_path, candidate_stats_path, deltas_path, qc_path])
        available_candidates = sum(
            row["validation_available"] == "yes" for row in candidate_rows
        )
        summary["cohorts"][dataset] = {
            "subjects": len(subjects),
            "assays": len(samples),
            "matrix_features": len(identifiers),
            "mapped_Entrez_features": len(entrez_ids),
            "candidate_coverage": available_candidates,
            "candidate_total": 250,
            "transform": config["transform"],
            "raw_value_quantiles_0_50_99_100": raw_quantiles,
            "transformed_value_quantiles_0_50_99_100": [
                float(value) for value in np.quantile(
                    expression, [0.0, 0.5, 0.99, 1.0]
                )
            ],
            "mapping_audit": mapping_audit,
            "fit": fit_summary,
            "technical_outlier_flags": sum(
                row["technical_outlier_flag_abs_z_gt_5"] == "yes"
                for row in qc_rows
            ),
            "primary_sample_exclusions": 0,
        }
        print("[{}] subjects={} mapped_Entrez={} candidates={}/250 outlier_flags={}".format(
            dataset, len(subjects), len(entrez_ids), available_candidates,
            summary["cohorts"][dataset]["technical_outlier_flags"],
        ))
    summary["next_gate"] = (
        "Construct locked node and pair reversal endpoints without changing ranking"
    )
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
    print("OSA CPAP validation preprocessing: PASS")
    print("Output: {}".format(output_dir))
    print("Frozen ranking was not read; known PPI/pathway data was not read.")
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
