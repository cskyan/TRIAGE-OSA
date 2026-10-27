#!/usr/bin/env python3
"""Preprocess and fit locked GSE135917 adipose validation contrasts.

Python 3.7 compatible.  The 18-sample case-control component and the 24-pair
CPAP component are fit separately.  This script does not read the frozen pair
ranking, known PPI labels, pathway labels, or previous validation outcomes.
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
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.0.1"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LOCK_REL = "config/osa_adipose_validation_lock.json"
MATRIX_REL = "geo/GSE135917_series_matrix.txt.gz"
ANNOTATION_REL = "annotations/GPL6244.annot.gz"
SAMPLE_MANIFEST_REL = "metadata/sample_manifest.tsv"
CANDIDATE_REL = (
    "results/discovery_preprocess_v2/04_candidates_v1/"
    "candidate_proteins_primary_250.tsv"
)
DEFAULT_STAGE = "13_adipose_validation_preprocess_v1"


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def moderated_statistics(np, beta, s2, residual_df, variance_factor):
    from osa_discovery_differential import (
        bh_adjust, estimate_prior_variance, student_t_two_sided,
    )
    prior_s2, prior_df, observed_log_variance, sampling_component, boundary = (
        estimate_prior_variance(s2, residual_df, np)
    )
    posterior_s2 = (
        prior_df * prior_s2 + residual_df * s2
    ) / (prior_df + residual_df)
    unmoderated_se = np.sqrt(np.maximum(s2 * variance_factor, 1e-30))
    moderated_se = np.sqrt(np.maximum(posterior_s2 * variance_factor, 1e-30))
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
    return {
        "beta": beta,
        "s2": s2,
        "posterior_s2": posterior_s2,
        "unmoderated_se": unmoderated_se,
        "unmoderated_t": unmoderated_t,
        "unmoderated_p": unmoderated_p,
        "unmoderated_fdr": bh_adjust(unmoderated_p, np),
        "moderated_se": moderated_se,
        "moderated_t": moderated_t,
        "moderated_p": moderated_p,
        "moderated_fdr": bh_adjust(moderated_p, np),
        "fit": {
            "residual_df": int(residual_df),
            "prior_variance": float(prior_s2),
            "prior_df": float(prior_df),
            "observed_log_residual_variance": float(observed_log_variance),
            "sampling_log_variance_component": float(sampling_component),
            "prior_df_boundary": bool(boundary),
            "moderated_df": float(moderated_df),
        },
    }


def fit_case_control(np, expression, sample_indices, metadata, adjusted):
    rows = [metadata[index] for index in sample_indices]
    disease = np.asarray([
        1.0 if row["disease_group"] == "OSA" else 0.0 for row in rows
    ])
    if int(np.sum(disease == 0.0)) != 8 or int(np.sum(disease == 1.0)) != 10:
        raise RuntimeError("GSE135917 case-control group count mismatch")
    columns = [np.ones(len(rows)), disease]
    design_terms = ["intercept", "OSA"]
    if adjusted:
        columns.extend([
            np.asarray([float(row["age"]) / 10.0 for row in rows]),
            np.asarray([1.0 if row["sex"] == "M" else 0.0 for row in rows]),
            np.asarray([float(row["BMI"]) / 5.0 for row in rows]),
        ])
        design_terms.extend(["age_per_10y", "male", "BMI_per_5"])
    design = np.stack(columns, axis=1)
    if int(np.linalg.matrix_rank(design)) != design.shape[1]:
        raise RuntimeError("GSE135917 case-control design is rank deficient")
    response = expression[:, sample_indices].T
    inverse = np.linalg.inv(np.dot(design.T, design))
    coefficients = np.dot(np.dot(inverse, design.T), response)
    residuals = response - np.dot(design, coefficients)
    residual_df = design.shape[0] - design.shape[1]
    s2 = np.sum(residuals * residuals, axis=0) / float(residual_df)
    result = moderated_statistics(
        np, coefficients[1, :], s2, residual_df, float(inverse[1, 1])
    )
    result["design_terms"] = design_terms
    result["design_rank"] = int(np.linalg.matrix_rank(design))
    result["n_control"] = int(np.sum(disease == 0.0))
    result["n_OSA"] = int(np.sum(disease == 1.0))
    result["sample_indices"] = sample_indices
    return result


def fit_paired(np, expression, samples, metadata):
    pair_rows = [
        row for row in metadata.values()
        if row["study_group"] == "study_group_2_paired_CPAP"
    ]
    by_subject = {}
    for row in pair_rows:
        by_subject.setdefault(row["subject_id"], {})[row["treatment"]] = row["sample_id"]
    if len(by_subject) != 24:
        raise RuntimeError("GSE135917 paired subject count mismatch")
    for subject, pair in by_subject.items():
        if set(pair) != {"pre_CPAP", "post_CPAP"}:
            raise RuntimeError("Incomplete GSE135917 CPAP pair: " + subject)
    subjects = sorted(
        by_subject,
        key=lambda value: (
            int(value.split("_")[-1]) if value.split("_")[-1].isdigit() else 10 ** 20,
            value,
        ),
    )
    sample_index = {sample: index for index, sample in enumerate(samples)}
    deltas = np.stack([
        expression[:, sample_index[by_subject[subject]["post_CPAP"]]]
        - expression[:, sample_index[by_subject[subject]["pre_CPAP"]]]
        for subject in subjects
    ], axis=1)
    n = len(subjects)
    beta = np.mean(deltas, axis=1)
    s2 = np.var(deltas, axis=1, ddof=1)
    result = moderated_statistics(np, beta, s2, n - 1, 1.0 / float(n))
    result["subjects"] = subjects
    result["deltas"] = deltas
    result["n_subjects"] = n
    return result


def statistics_rows(fmt, entrez_ids, expression, result, design, sample_count):
    rows = []
    for index, entrez in enumerate(entrez_ids):
        rows.append({
            "Entrez": entrez,
            "mean_expression": fmt(float(np_mean(expression[index, :]))),
            "beta": fmt(result["beta"][index]),
            "unmoderated_SE": fmt(result["unmoderated_se"][index]),
            "unmoderated_t": fmt(result["unmoderated_t"][index]),
            "unmoderated_P_value": fmt(result["unmoderated_p"][index]),
            "unmoderated_FDR_BH": fmt(result["unmoderated_fdr"][index]),
            "moderated_SE": fmt(result["moderated_se"][index]),
            "moderated_t": fmt(result["moderated_t"][index]),
            "P_value": fmt(result["moderated_p"][index]),
            "FDR_BH": fmt(result["moderated_fdr"][index]),
            "residual_variance": fmt(result["s2"][index]),
            "posterior_variance": fmt(result["posterior_s2"][index]),
            "residual_df": str(result["fit"]["residual_df"]),
            "n_assays": str(sample_count),
            "design": design,
            "candidate_selection_allowed": "no",
        })
    return rows


def np_mean(values):
    return sum(float(value) for value in values) / float(len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-stage", default=DEFAULT_STAGE)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from osa_cpap_validation_preprocess import (
        atomic_json, fmt, map_and_aggregate, parse_series_matrix, read_tsv,
        sample_qc, sha256_file, relative_path, write_gzip_tsv, write_tsv,
    )
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    required = {
        "lock": root / LOCK_REL,
        "matrix": root / MATRIX_REL,
        "annotation": root / ANNOTATION_REL,
        "manifest": root / SAMPLE_MANIFEST_REL,
        "candidates": root / CANDIDATE_REL,
        "raw": root / "geo/GSE135917_RAW.tar",
    }
    for path in required.values():
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    lock = json.loads(required["lock"].read_text(encoding="utf-8"))
    hash_paths = {
        "GSE135917_series_matrix.txt.gz": required["matrix"],
        "GSE135917_RAW.tar": required["raw"],
        "GPL6244.annot.gz": required["annotation"],
        "sample_manifest.tsv": required["manifest"],
        "candidate_proteins_primary_250.tsv": required["candidates"],
    }
    observed_hashes = {}
    for name, path in hash_paths.items():
        observed_hashes[name] = sha256_file(path)
        if observed_hashes[name] != lock["input_hashes"][name]:
            raise RuntimeError("Frozen adipose input hash mismatch: " + name)
    manifest_rows, _ = read_tsv(required["manifest"])
    manifest_rows = [row for row in manifest_rows if row["dataset"] == "GSE135917"]
    candidates, _ = read_tsv(required["candidates"])
    if len(manifest_rows) != 66 or len(candidates) != 250:
        raise RuntimeError("GSE135917 manifest or candidate universe mismatch")
    for row in manifest_rows:
        if row["candidate_selection_allowed"] != "no" or row["primary_include"] != "yes":
            raise RuntimeError("GSE135917 validation isolation/sample freeze failure")
    print("OSA adipose validation preprocessing environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("Case-control: 8 control + 10 OSA; paired CPAP: 24 subjects")
    print("Ranking/known PPI/pathway/previous validation read: no")
    if args.check_env:
        return 0
    identifiers, samples, matrix = parse_series_matrix(np, required["matrix"])
    if len(identifiers) != 33297 or len(samples) != 66:
        raise RuntimeError("Unexpected GSE135917 series matrix dimensions")
    if float(np.min(matrix)) < 1.0 or float(np.max(matrix)) > 30.0:
        raise RuntimeError("GSE135917 matrix does not satisfy frozen log2-scale gate")
    metadata = {row["sample_id"]: row for row in manifest_rows}
    if set(samples) != set(metadata):
        raise RuntimeError("GSE135917 matrix/manifest sample identity mismatch")
    entrez_ids, expression, aggregation_count, mapping_audit = map_and_aggregate(
        np, "GSE135917", identifiers, matrix, required["annotation"]
    )
    entrez_index = {entrez: index for index, entrez in enumerate(entrez_ids)}
    sample_index = {sample: index for index, sample in enumerate(samples)}
    case_samples = [
        sample for sample in samples
        if metadata[sample]["study_group"] == "study_group_1_case_control"
    ]
    case_indices = [sample_index[sample] for sample in case_samples]
    case_metadata_by_index = {sample_index[sample]: metadata[sample] for sample in case_samples}
    primary_case = fit_case_control(np, expression, case_indices, case_metadata_by_index, False)
    adjusted_case = fit_case_control(np, expression, case_indices, case_metadata_by_index, True)
    paired = fit_paired(np, expression, samples, metadata)
    primary_rows = statistics_rows(
        fmt, entrez_ids, expression[:, case_indices], primary_case,
        "intercept_plus_OSA; beta=OSA_minus_control", len(case_indices),
    )
    adjusted_rows = statistics_rows(
        fmt, entrez_ids, expression[:, case_indices], adjusted_case,
        "intercept_plus_OSA_plus_age_per_10y_plus_male_plus_BMI_per_5; beta=OSA",
        len(case_indices),
    )
    paired_rows = statistics_rows(
        fmt, entrez_ids, expression, paired,
        "paired_post_minus_pre_equivalent_to_subject_plus_timepoint", 48,
    )
    primary_by_entrez = {row["Entrez"]: row for row in primary_rows}
    adjusted_by_entrez = {row["Entrez"]: row for row in adjusted_rows}
    paired_by_entrez = {row["Entrez"]: row for row in paired_rows}
    candidate_evidence = []
    case_expression_rows = []
    paired_delta_rows = []
    for candidate in candidates:
        accession = candidate["UniProt_accession"]
        entrez = candidate["Entrez"]
        state = primary_by_entrez.get(entrez)
        adjusted = adjusted_by_entrez.get(entrez)
        cpap = paired_by_entrez.get(entrez)
        out = {
            "dataset": "GSE135917",
            "UniProt_accession": accession,
            "gene_symbol": candidate["gene_symbol_current"],
            "Entrez": entrez,
            "discovery_rank": candidate["discovery_rank"],
            "discovery_OSA_beta": candidate["OSA_beta"],
            "discovery_OSA_beta_sign": candidate["OSA_beta_sign"],
            "adipose_available": "yes" if state is not None and cpap is not None else "no",
        }
        for prefix, source in [
            ("state", state), ("state_adjusted", adjusted), ("CPAP", cpap),
        ]:
            for field in [
                "beta", "unmoderated_SE", "unmoderated_t", "unmoderated_P_value",
                "unmoderated_FDR_BH", "moderated_SE", "moderated_t", "P_value",
                "FDR_BH",
            ]:
                out[prefix + "_" + field] = source[field] if source is not None else ""
        if state is not None and adjusted is not None:
            out["state_primary_adjusted_sign_agreement"] = (
                "yes" if float(state["beta"]) * float(adjusted["beta"]) > 0.0
                else "no"
            )
        else:
            out["state_primary_adjusted_sign_agreement"] = ""
        candidate_evidence.append(out)
        case_row = {
            "dataset": "GSE135917",
            "UniProt_accession": accession,
            "gene_symbol": candidate["gene_symbol_current"],
            "Entrez": entrez,
            "validation_available": "yes" if state is not None else "no",
        }
        delta_row = dict(case_row)
        if entrez in entrez_index:
            feature_index = entrez_index[entrez]
            for sample in case_samples:
                case_row["sample_" + sample] = fmt(expression[feature_index, sample_index[sample]])
            for subject_index, subject in enumerate(paired["subjects"]):
                delta_row["subject_" + subject] = fmt(paired["deltas"][feature_index, subject_index])
        else:
            for sample in case_samples:
                case_row["sample_" + sample] = ""
            for subject in paired["subjects"]:
                delta_row["subject_" + subject] = ""
        case_expression_rows.append(case_row)
        paired_delta_rows.append(delta_row)
    qc_rows = []
    for qc_group in [
        "study_group_1_case_control", "study_group_2_paired_CPAP",
    ]:
        group_samples = [
            sample for sample in samples
            if metadata[sample]["study_group"] == qc_group
        ]
        group_indices = [sample_index[sample] for sample in group_samples]
        group_qc = sample_qc(
            np, expression[:, group_indices], group_samples, metadata
        )
        for row in group_qc:
            source = metadata[row["sample_id"]]
            row["qc_reference_group"] = qc_group
            row["study_group"] = source["study_group"]
            row["disease_group"] = source["disease_group"]
            row["timepoint"] = source["timepoint"]
        qc_rows.extend(group_qc)
    original_order = {sample: index for index, sample in enumerate(samples)}
    qc_rows.sort(key=lambda row: original_order[row["sample_id"]])
    output_dir = root / "results" / "discovery_preprocess_v2" / args.output_stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace")
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "case": output_dir / "case_control_differential_all_entrez.tsv.gz",
        "adjusted": output_dir / "case_control_covariate_sensitivity_all_entrez.tsv.gz",
        "paired": output_dir / "paired_CPAP_differential_all_entrez.tsv.gz",
        "candidate": output_dir / "candidate_adipose_evidence.tsv",
        "case_matrix": output_dir / "candidate_case_control_expression.tsv.gz",
        "paired_delta": output_dir / "candidate_paired_deltas.tsv.gz",
        "qc": output_dir / "sample_technical_qc.tsv",
    }
    write_gzip_tsv(paths["case"], primary_rows, list(primary_rows[0]))
    write_gzip_tsv(paths["adjusted"], adjusted_rows, list(adjusted_rows[0]))
    write_gzip_tsv(paths["paired"], paired_rows, list(paired_rows[0]))
    write_tsv(paths["candidate"], candidate_evidence, list(candidate_evidence[0]))
    write_gzip_tsv(paths["case_matrix"], case_expression_rows, list(case_expression_rows[0]))
    write_gzip_tsv(paths["paired_delta"], paired_delta_rows, list(paired_delta_rows[0]))
    write_tsv(paths["qc"], qc_rows, list(qc_rows[0]))
    output_paths = list(paths.values())
    available = sum(row["adipose_available"] == "yes" for row in candidate_evidence)
    summary = {
        "created_at_utc": utc_now(),
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "helper_script": {
            "path": "scripts/osa_cpap_validation_preprocess.py",
            "sha256": sha256_file(root / "scripts/osa_cpap_validation_preprocess.py"),
        },
        "validation_lock": {"path": LOCK_REL, "sha256": sha256_file(required["lock"])},
        "input_hashes": observed_hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "matrix": {
            "features": len(identifiers), "samples": len(samples),
            "transform": "none_already_log2_normalized",
            "quantiles_0_50_99_100": [
                float(value) for value in np.quantile(matrix, [0.0, 0.5, 0.99, 1.0])
            ],
            "mapping_audit": mapping_audit,
        },
        "case_control": {
            "controls": primary_case["n_control"], "OSA": primary_case["n_OSA"],
            "primary_design_terms": primary_case["design_terms"],
            "primary_fit": primary_case["fit"],
            "adjusted_design_terms": adjusted_case["design_terms"],
            "adjusted_fit": adjusted_case["fit"],
        },
        "paired_CPAP": {
            "subjects": paired["n_subjects"], "assays": 48,
            "fit": paired["fit"],
        },
        "candidate_coverage": available,
        "candidate_total": 250,
        "technical_outlier_flags": sum(
            row["technical_outlier_flag_abs_z_gt_5"] == "yes" for row in qc_rows
        ),
        "primary_sample_exclusions": 0,
        "ranking_read": False,
        "known_PPI_or_pathway_read": False,
        "previous_validation_outcomes_read": False,
        "candidate_membership_changed": False,
        "next_gate": "Evaluate frozen cross-tissue pair endpoints without reranking",
    }
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, summary)
    output_paths.append(summary_path)
    manifest_rows_out = [{
        "path": relative_path(path, root),
        "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in output_paths]
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(manifest_path, manifest_rows_out, ["path", "bytes", "sha256"])
    print("OSA adipose validation preprocessing: PASS")
    print("Candidate coverage: {}/250; outlier flags: {}".format(
        available, summary["technical_outlier_flags"]
    ))
    print("Output: {}".format(output_dir))
    print("Ranking/known PPI/pathway/previous validation read: no")
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
