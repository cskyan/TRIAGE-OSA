#!/usr/bin/env python3
"""Preprocess GSE214865 and fit participant-level pseudobulk contrasts.

Python 3.7 compatible.  The script reads the 22 CellRanger-filtered 10x
triplets directly from the GEO TAR archive, applies a locked technical QC and
marker-module annotation, aggregates raw counts by participant and cell
lineage, and fits the locked pediatric OSA-minus-control pseudobulk models.

The frozen adult candidate list, frozen pair ranking, known PPI labels, and
all previous validation outcomes are deliberately not read by this script.
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
import tarfile
import time
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
LOCK_REL = "config/osa_scrna_validation_lock.json"
RAW_REL = "geo/GSE214865_RAW.tar"
SAMPLE_MANIFEST_REL = "metadata/sample_manifest.tsv"
DATASET_REGISTRY_REL = "metadata/dataset_registry.tsv"
DEFAULT_STAGE = "15_scrna_validation_preprocess_v1"
PRIMARY_LINEAGES = ("T_NK", "B", "Myeloid")
FINE_TYPES = ("T", "NK", "B", "Monocyte", "Dendritic", "Platelet")


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def fmt(value):
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return "{:.10g}".format(value)


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def relative_path(root, path):
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def marker_definitions(lock):
    marker_modules = lock["annotation"]["marker_modules"]
    return [(name, list(marker_modules[name])) for name in FINE_TYPES]


def expected_samples(manifest_path):
    rows = [
        row for row in read_tsv(manifest_path)
        if row.get("dataset") == "GSE214865"
        and row.get("primary_include") == "yes"
    ]
    if len(rows) != 22:
        raise RuntimeError(
            "Expected 22 primary GSE214865 samples, found {}".format(len(rows))
        )
    seen_sample = set()
    seen_subject = set()
    groups = {"control": 0, "OSA": 0}
    for row in rows:
        sample = row["sample_id"]
        subject = row["subject_id"]
        group = row["disease_group"]
        if sample in seen_sample or subject in seen_subject:
            raise RuntimeError("Duplicate GSE214865 sample or participant")
        if group not in groups:
            raise RuntimeError("Unexpected GSE214865 disease group: " + group)
        if row.get("candidate_selection_allowed") != "no":
            raise RuntimeError("GSE214865 selection flag is not frozen to no")
        seen_sample.add(sample)
        seen_subject.add(subject)
        groups[group] += 1
    if groups != {"control": 11, "OSA": 11}:
        raise RuntimeError("GSE214865 group count mismatch: " + repr(groups))
    return sorted(rows, key=lambda row: row["sample_id"])


def archive_index(tar):
    members = [member for member in tar.getmembers() if member.isfile()]
    if len(members) != 66:
        raise RuntimeError(
            "Expected 66 regular files in GSE214865 TAR, found {}".format(
                len(members)
            )
        )
    index = {Path(member.name).name: member for member in members}
    if len(index) != len(members):
        raise RuntimeError("Duplicate basenames in GSE214865 TAR")
    return index


def sample_member(index, sample_id, subject_id, suffix):
    name = "{}_{}_{}".format(sample_id, subject_id, suffix)
    if name not in index:
        raise RuntimeError("Missing GSE214865 TAR member: " + name)
    return index[name]


def read_gzip_member_bytes(tar, member):
    extracted = tar.extractfile(member)
    if extracted is None:
        raise RuntimeError("Could not extract TAR member: " + member.name)
    with extracted:
        with gzip.GzipFile(fileobj=extracted, mode="rb") as handle:
            return handle.read()


def parse_features(raw_bytes):
    rows = []
    for line in raw_bytes.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            raise RuntimeError("Malformed features.tsv row")
        rows.append((fields[0].split(".")[0], fields[1], fields[2] if len(fields) > 2 else ""))
    if not rows:
        raise RuntimeError("Empty features.tsv")
    if any(row[2] not in ("", "Gene Expression") for row in rows):
        raise RuntimeError("Non-gene-expression feature found")
    if len(set(row[0] for row in rows)) != len(rows):
        raise RuntimeError("Duplicate version-stripped Ensembl identifiers")
    return rows


def parse_barcodes(raw_bytes):
    rows = [line for line in raw_bytes.decode("utf-8").splitlines() if line]
    if len(set(rows)) != len(rows):
        raise RuntimeError("Duplicate barcodes within one GSE214865 sample")
    return rows


def load_matrix(tar, member, scipy_io):
    extracted = tar.extractfile(member)
    if extracted is None:
        raise RuntimeError("Could not extract TAR member: " + member.name)
    with extracted:
        with gzip.GzipFile(fileobj=extracted, mode="rb") as handle:
            matrix = scipy_io.mmread(handle)
    if not hasattr(matrix, "tocsc"):
        raise RuntimeError("10x Matrix Market input was unexpectedly dense")
    matrix = matrix.tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.nnz and float(matrix.data.min()) < 0.0:
        raise RuntimeError("Negative counts found in 10x matrix")
    return matrix


def classify_cells(np, matrix, counts, symbols, lock):
    modules = marker_definitions(lock)
    symbol_rows = {}
    for index, symbol in enumerate(symbols):
        symbol_rows.setdefault(symbol, []).append(index)
    scores = []
    detected = []
    marker_presence = []
    for module_name, markers in modules:
        rows = []
        found = []
        for marker in markers:
            if marker in symbol_rows:
                rows.extend(symbol_rows[marker])
                found.append(marker)
        if len(found) != len(markers):
            raise RuntimeError(
                "Missing locked markers for {}: {}".format(
                    module_name,
                    ",".join(sorted(set(markers) - set(found))),
                )
            )
        values = matrix[rows, :].toarray().astype(np.float64, copy=False)
        present = np.sum(values > 0.0, axis=0)
        scale = 10000.0 / np.maximum(counts, 1.0)
        normalized = np.log1p(values * scale.reshape(1, -1))
        scores.append(np.mean(normalized, axis=0))
        detected.append(present)
        marker_presence.append({
            "module": module_name,
            "markers": ",".join(markers),
            "rows": len(rows),
        })
    scores = np.stack(scores, axis=0)
    detected = np.stack(detected, axis=0)
    minimum_markers = int(lock["annotation"]["minimum_detected_markers"])
    minimum_score = float(lock["annotation"]["minimum_top_module_score"])
    minimum_margin = float(lock["annotation"]["minimum_top_minus_second_score"])
    eligible = (detected >= minimum_markers) & (scores >= minimum_score)
    eligible_scores = np.where(eligible, scores, -np.inf)
    best_index = np.argmax(eligible_scores, axis=0)
    best_score = eligible_scores[best_index, np.arange(matrix.shape[1])]
    if scores.shape[0] > 1:
        second_score = np.partition(eligible_scores, -2, axis=0)[-2, :]
    else:
        second_score = np.full(matrix.shape[1], -np.inf)
    second_for_margin = np.where(np.isfinite(second_score), second_score, 0.0)
    margin = best_score - second_for_margin
    assigned = np.isfinite(best_score) & (margin >= minimum_margin)
    names = np.asarray([item[0] for item in modules], dtype=object)
    fine = np.full(matrix.shape[1], "Unassigned", dtype=object)
    fine[assigned] = names[best_index[assigned]]
    primary = np.full(matrix.shape[1], "", dtype=object)
    primary[(fine == "T") | (fine == "NK")] = "T_NK"
    primary[fine == "B"] = "B"
    primary[(fine == "Monocyte") | (fine == "Dendritic")] = "Myeloid"
    return {
        "scores": scores,
        "detected": detected,
        "fine": fine,
        "primary": primary,
        "margin": margin,
        "best_score": best_score,
        "marker_presence": marker_presence,
        "module_names": [item[0] for item in modules],
    }


def technical_qc(np, matrix, symbols, lock):
    counts = np.asarray(matrix.sum(axis=0)).reshape(-1).astype(np.float64)
    genes = np.diff(matrix.indptr).astype(np.int64)
    mito_rows = np.asarray([
        index for index, symbol in enumerate(symbols)
        if symbol.upper().startswith("MT-")
    ], dtype=np.int64)
    if mito_rows.size == 0:
        raise RuntimeError("No mitochondrial genes were found")
    mito = np.asarray(matrix[mito_rows, :].sum(axis=0)).reshape(-1)
    mito_fraction = mito / np.maximum(counts, 1.0)
    rule = lock["technical_QC"]
    pass_genes_low = genes >= int(rule["lower_detected_genes"])
    pass_counts_low = counts >= float(rule["lower_total_UMI"])
    pass_genes_high = genes <= int(rule["upper_detected_genes"])
    pass_counts_high = counts <= float(rule["upper_total_UMI"])
    pass_mito = mito_fraction <= float(rule["upper_mitochondrial_fraction"])
    passed = (
        pass_genes_low & pass_counts_low & pass_genes_high
        & pass_counts_high & pass_mito
    )
    return {
        "counts": counts,
        "genes": genes,
        "mito_fraction": mito_fraction,
        "pass": passed,
        "pass_genes_low": pass_genes_low,
        "pass_counts_low": pass_counts_low,
        "pass_genes_high": pass_genes_high,
        "pass_counts_high": pass_counts_high,
        "pass_mito": pass_mito,
        "mito_gene_count": int(mito_rows.size),
    }


def qc_reason(qc, index):
    reasons = []
    if not qc["pass_genes_low"][index]:
        reasons.append("genes_below_minimum")
    if not qc["pass_counts_low"][index]:
        reasons.append("UMI_below_minimum")
    if not qc["pass_genes_high"][index]:
        reasons.append("genes_above_maximum")
    if not qc["pass_counts_high"][index]:
        reasons.append("UMI_above_maximum")
    if not qc["pass_mito"][index]:
        reasons.append("mitochondrial_fraction_above_maximum")
    return "PASS" if not reasons else ";".join(reasons)


def aggregate_pseudobulk(np, scipy_sparse, matrix, tech_pass, fine, primary):
    group_keys = (
        [("primary", value) for value in PRIMARY_LINEAGES]
        + [("secondary", value) for value in FINE_TYPES]
    )
    row_parts = []
    col_parts = []
    for group_index, (level, group) in enumerate(group_keys):
        labels = primary if level == "primary" else fine
        cell_indices = np.where(tech_pass & (labels == group))[0]
        if cell_indices.size:
            row_parts.append(cell_indices)
            col_parts.append(np.full(cell_indices.size, group_index, dtype=np.int64))
    if row_parts:
        design_rows = np.concatenate(row_parts)
        design_cols = np.concatenate(col_parts)
    else:
        design_rows = np.asarray([], dtype=np.int64)
        design_cols = np.asarray([], dtype=np.int64)
    design = scipy_sparse.coo_matrix(
        (
            np.ones(design_rows.size, dtype=np.int32),
            (design_rows, design_cols),
        ),
        shape=(matrix.shape[1], len(group_keys)),
    ).tocsr()
    counts = matrix.dot(design).toarray()
    cell_counts = np.asarray(design.sum(axis=0)).reshape(-1).astype(np.int64)
    return group_keys, counts, cell_counts


def fit_effects(np, feature_ids, symbols, counts, columns, lock):
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from osa_adipose_validation_preprocess import moderated_statistics

    minimum_cells = int(
        lock["pseudobulk"]["minimum_cells_per_participant_lineage"]
    )
    minimum_group = int(lock["pseudobulk"]["minimum_participants_per_group"])
    results = []
    expression_columns = []
    expression_blocks = []
    lineage_summaries = []
    for lineage in PRIMARY_LINEAGES:
        indices = [
            index for index, row in enumerate(columns)
            if row["analysis_level"] == "primary"
            and row["cell_type"] == lineage
            and int(row["cell_count"]) >= minimum_cells
        ]
        control = [index for index in indices if columns[index]["disease_group"] == "control"]
        osa = [index for index in indices if columns[index]["disease_group"] == "OSA"]
        if len(control) < minimum_group or len(osa) < minimum_group:
            lineage_summaries.append({
                "lineage": lineage,
                "n_control": len(control),
                "n_OSA": len(osa),
                "tested_genes": 0,
                "status": "BLOCKED_INSUFFICIENT_PARTICIPANTS",
            })
            continue
        ordered = control + osa
        raw = counts[:, ordered].astype(np.float64, copy=False)
        library = np.sum(raw, axis=0)
        if np.any(library <= 0.0):
            raise RuntimeError("Zero primary-lineage pseudobulk library")
        cpm = raw / library.reshape(1, -1) * 1000000.0
        logcpm = np.log2(cpm + 0.5)
        tested = np.sum(cpm >= 1.0, axis=1) >= 5
        tested_indices = np.where(tested)[0]
        y = logcpm[tested, :]
        n_control = len(control)
        n_osa = len(osa)
        mean_control = np.mean(y[:, :n_control], axis=1)
        mean_osa = np.mean(y[:, n_control:], axis=1)
        beta = mean_osa - mean_control
        residuals_control = y[:, :n_control] - mean_control.reshape(-1, 1)
        residuals_osa = y[:, n_control:] - mean_osa.reshape(-1, 1)
        residual_df = n_control + n_osa - 2
        s2 = (
            np.sum(residuals_control * residuals_control, axis=1)
            + np.sum(residuals_osa * residuals_osa, axis=1)
        ) / float(residual_df)
        variance_factor = 1.0 / float(n_control) + 1.0 / float(n_osa)
        fit = moderated_statistics(
            np, beta, s2, residual_df, variance_factor
        )
        fit_index = {feature_index: result_index for result_index, feature_index in enumerate(tested_indices)}
        detection = np.mean(cpm >= 1.0, axis=1)
        mean_cpm = np.mean(cpm, axis=1)
        for feature_index in range(len(feature_ids)):
            result_index = fit_index.get(feature_index)
            row = {
                "dataset": "GSE214865",
                "contrast": "OSA_minus_control",
                "lineage": lineage,
                "Ensembl_gene": feature_ids[feature_index],
                "gene_symbol": symbols[feature_index],
                "tested": "yes" if result_index is not None else "no",
                "mean_CPM": fmt(mean_cpm[feature_index]),
                "participant_detection_fraction_CPM_ge_1": fmt(detection[feature_index]),
                "beta_log2CPM": "",
                "unmoderated_SE": "",
                "unmoderated_t": "",
                "unmoderated_P_value": "",
                "unmoderated_FDR_BH": "",
                "moderated_SE": "",
                "moderated_t": "",
                "P_value": "",
                "FDR_BH": "",
                "residual_variance": "",
                "posterior_variance": "",
                "residual_df": str(residual_df),
                "moderated_df": fmt(fit["fit"]["moderated_df"]),
                "n_control": str(n_control),
                "n_OSA": str(n_osa),
                "design": "participant_level_intercept_plus_OSA",
                "candidate_selection_allowed": "no",
            }
            if result_index is not None:
                row.update({
                    "beta_log2CPM": fmt(fit["beta"][result_index]),
                    "unmoderated_SE": fmt(fit["unmoderated_se"][result_index]),
                    "unmoderated_t": fmt(fit["unmoderated_t"][result_index]),
                    "unmoderated_P_value": fmt(fit["unmoderated_p"][result_index]),
                    "unmoderated_FDR_BH": fmt(fit["unmoderated_fdr"][result_index]),
                    "moderated_SE": fmt(fit["moderated_se"][result_index]),
                    "moderated_t": fmt(fit["moderated_t"][result_index]),
                    "P_value": fmt(fit["moderated_p"][result_index]),
                    "FDR_BH": fmt(fit["moderated_fdr"][result_index]),
                    "residual_variance": fmt(fit["s2"][result_index]),
                    "posterior_variance": fmt(fit["posterior_s2"][result_index]),
                })
            results.append(row)
        for local_index, source_index in enumerate(ordered):
            source = columns[source_index]
            expression_columns.append({
                "column_id": source["column_id"],
                "sample_id": source["sample_id"],
                "subject_id": source["subject_id"],
                "disease_group": source["disease_group"],
                "lineage": lineage,
                "cell_count": source["cell_count"],
                "library_size": source["library_size"],
                "expression_column_index": len(expression_columns),
            })
        expression_blocks.append(logcpm)
        lineage_summaries.append({
            "lineage": lineage,
            "n_control": n_control,
            "n_OSA": n_osa,
            "tested_genes": int(np.sum(tested)),
            "prior_variance": fmt(fit["fit"]["prior_variance"]),
            "prior_df": fmt(fit["fit"]["prior_df"]),
            "moderated_df": fmt(fit["fit"]["moderated_df"]),
            "status": "PASS",
        })
    expression = (
        np.concatenate(expression_blocks, axis=1)
        if expression_blocks else np.empty((len(feature_ids), 0), dtype=np.float64)
    )
    return results, expression, expression_columns, lineage_summaries


def check_hashes(root, lock, keys):
    path_map = {
        "GSE214865_RAW.tar": root / RAW_REL,
        "sample_manifest.tsv": root / SAMPLE_MANIFEST_REL,
        "dataset_registry.tsv": root / DATASET_REGISTRY_REL,
    }
    rows = []
    for key in keys:
        path = path_map[key]
        expected = lock["input_hashes"][key]
        actual = sha256_file(path) if path.is_file() else ""
        rows.append({
            "artifact": key,
            "path": relative_path(root, path),
            "exists": "yes" if path.is_file() else "no",
            "expected_sha256": expected,
            "actual_sha256": actual,
            "hash_pass": "yes" if actual == expected else "no",
        })
    return rows


def run(args):
    root = args.root.expanduser().resolve()
    lock_path = root / LOCK_REL
    raw_path = root / RAW_REL
    manifest_path = root / SAMPLE_MANIFEST_REL
    registry_path = root / DATASET_REGISTRY_REL
    for path in (lock_path, raw_path, manifest_path, registry_path):
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    try:
        import numpy as np
        import scipy
        import scipy.io as scipy_io
        import scipy.sparse as scipy_sparse
    except ImportError as exc:
        raise RuntimeError("numpy and scipy are required: " + str(exc))

    print("OSA GSE214865 pseudobulk environment: PASS")
    print("Python: " + sys.version.replace("\n", " "))
    print("numpy: " + np.__version__)
    print("scipy: " + scipy.__version__)
    print("Raw archive: " + str(raw_path))
    print("Training/candidate/ranking data read: no")

    hash_rows = check_hashes(
        root, lock,
        ("GSE214865_RAW.tar", "sample_manifest.tsv", "dataset_registry.tsv"),
    )
    failed_hashes = [row["artifact"] for row in hash_rows if row["hash_pass"] != "yes"]
    if failed_hashes:
        raise RuntimeError("Input hash mismatch: " + ", ".join(failed_hashes))
    samples = expected_samples(manifest_path)
    if args.check_env:
        print("Input hashes: PASS")
        print("Manifest: 22 participants (11 control, 11 OSA)")
        return 0

    smoke = bool(args.smoke)
    if smoke:
        control = next(row for row in samples if row["disease_group"] == "control")
        osa = next(row for row in samples if row["disease_group"] == "OSA")
        samples = [control, osa]
    output_base = root / "results" / "discovery_preprocess_v2" / args.output_stage
    output_dir = output_base / "smoke" if smoke else output_base
    if output_dir.exists():
        if not args.force:
            raise RuntimeError(
                "Output exists; pass --force to replace: " + str(output_dir)
            )
        resolved = output_dir.resolve()
        required_parent = (root / "results" / "discovery_preprocess_v2").resolve()
        if required_parent not in resolved.parents:
            raise RuntimeError("Refusing to remove output outside results tree")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(str(lock_path), str(output_dir / "analysis_lock_snapshot.json"))

    started = utc_now()
    start_time = time.time()
    sample_qc_rows = []
    cell_type_rows = []
    pseudobulk_columns = []
    pseudobulk_blocks = []
    feature_reference = None
    feature_reference_sha = None
    total_nnz = 0
    total_input_cells = 0
    total_tech_pass = 0
    total_assigned = 0
    annotation_temp = output_dir / "cell_annotations.tsv.gz.tmp"
    annotation_final = output_dir / "cell_annotations.tsv.gz"
    annotation_fields = [
        "dataset", "sample_id", "subject_id", "disease_group", "barcode",
        "total_UMI", "detected_genes", "mitochondrial_fraction",
        "technical_QC_pass", "technical_QC_reason", "fine_cell_type",
        "primary_lineage", "annotation_top_score", "annotation_margin",
    ] + ["{}_module_score".format(value) for value in FINE_TYPES]

    try:
        with gzip.open(
            str(annotation_temp), "wt", encoding="utf-8", newline=""
        ) as annotation_handle:
            annotation_writer = csv.DictWriter(
                annotation_handle, delimiter="\t", fieldnames=annotation_fields,
                lineterminator="\n",
            )
            annotation_writer.writeheader()
            with tarfile.open(str(raw_path), "r") as tar:
                member_index = archive_index(tar)
                for sample_number, sample in enumerate(samples, 1):
                    sample_start = time.time()
                    sample_id = sample["sample_id"]
                    subject_id = sample["subject_id"]
                    group = sample["disease_group"]
                    print(
                        "[GSE214865] sample {}/{} {} {}".format(
                            sample_number, len(samples), sample_id, subject_id
                        )
                    )
                    feature_member = sample_member(
                        member_index, sample_id, subject_id, "features.tsv.gz"
                    )
                    barcode_member = sample_member(
                        member_index, sample_id, subject_id, "barcodes.tsv.gz"
                    )
                    matrix_member = sample_member(
                        member_index, sample_id, subject_id, "matrix.mtx.gz"
                    )
                    feature_bytes = read_gzip_member_bytes(tar, feature_member)
                    feature_sha = hashlib.sha256(feature_bytes).hexdigest()
                    features = parse_features(feature_bytes)
                    barcodes = parse_barcodes(
                        read_gzip_member_bytes(tar, barcode_member)
                    )
                    if feature_reference is None:
                        feature_reference = features
                        feature_reference_sha = feature_sha
                    elif features != feature_reference or feature_sha != feature_reference_sha:
                        raise RuntimeError(
                            "Feature table differs for sample " + sample_id
                        )
                    matrix = load_matrix(tar, matrix_member, scipy_io)
                    if matrix.shape != (len(features), len(barcodes)):
                        raise RuntimeError(
                            "Matrix dimensions do not match features/barcodes for "
                            + sample_id
                        )
                    symbols = [row[1] for row in features]
                    qc = technical_qc(np, matrix, symbols, lock)
                    annotation = classify_cells(
                        np, matrix, qc["counts"], symbols, lock
                    )
                    annotation["fine"][~qc["pass"]] = "QC_fail"
                    annotation["primary"][~qc["pass"]] = ""
                    group_keys, pseudobulk, cell_counts = aggregate_pseudobulk(
                        np, scipy_sparse, matrix, qc["pass"],
                        annotation["fine"], annotation["primary"],
                    )
                    pseudobulk_blocks.append(pseudobulk)
                    library_sizes = np.sum(pseudobulk, axis=0)
                    for group_index, (level, cell_type) in enumerate(group_keys):
                        column_id = "{}__{}__{}".format(
                            sample_id, level, cell_type
                        )
                        column = {
                            "column_id": column_id,
                            "dataset": "GSE214865",
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "disease_group": group,
                            "analysis_level": level,
                            "cell_type": cell_type,
                            "cell_count": str(int(cell_counts[group_index])),
                            "library_size": str(int(library_sizes[group_index])),
                            "pseudobulk_column_index": str(len(pseudobulk_columns)),
                            "candidate_selection_allowed": "no",
                        }
                        pseudobulk_columns.append(column)
                        cell_type_rows.append(column.copy())

                    tech_count = int(np.sum(qc["pass"]))
                    assigned_mask = qc["pass"] & (annotation["fine"] != "Unassigned")
                    assigned_count = int(np.sum(assigned_mask))
                    sample_qc_rows.append({
                        "dataset": "GSE214865",
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "disease_group": group,
                        "input_filtered_cells": str(matrix.shape[1]),
                        "matrix_features": str(matrix.shape[0]),
                        "matrix_nnz": str(matrix.nnz),
                        "total_UMI": str(int(np.sum(qc["counts"]))),
                        "median_UMI": fmt(np.median(qc["counts"])),
                        "median_detected_genes": fmt(np.median(qc["genes"])),
                        "median_mitochondrial_fraction": fmt(np.median(qc["mito_fraction"])),
                        "mitochondrial_gene_count": str(qc["mito_gene_count"]),
                        "technical_QC_pass_cells": str(tech_count),
                        "technical_QC_pass_fraction": fmt(tech_count / float(matrix.shape[1])),
                        "assigned_cells": str(assigned_count),
                        "assigned_fraction_of_technical_pass": fmt(
                            assigned_count / float(max(tech_count, 1))
                        ),
                        "feature_table_sha256_decompressed": feature_sha,
                        "seconds": fmt(time.time() - sample_start),
                        "status": "PASS",
                    })

                    for cell_index, barcode in enumerate(barcodes):
                        row = {
                            "dataset": "GSE214865",
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "disease_group": group,
                            "barcode": barcode,
                            "total_UMI": str(int(qc["counts"][cell_index])),
                            "detected_genes": str(int(qc["genes"][cell_index])),
                            "mitochondrial_fraction": fmt(qc["mito_fraction"][cell_index]),
                            "technical_QC_pass": "yes" if qc["pass"][cell_index] else "no",
                            "technical_QC_reason": qc_reason(qc, cell_index),
                            "fine_cell_type": annotation["fine"][cell_index],
                            "primary_lineage": annotation["primary"][cell_index],
                            "annotation_top_score": fmt(annotation["best_score"][cell_index]),
                            "annotation_margin": fmt(annotation["margin"][cell_index]),
                        }
                        for module_index, module_name in enumerate(annotation["module_names"]):
                            row["{}_module_score".format(module_name)] = fmt(
                                annotation["scores"][module_index, cell_index]
                            )
                        annotation_writer.writerow(row)
                    total_nnz += int(matrix.nnz)
                    total_input_cells += int(matrix.shape[1])
                    total_tech_pass += tech_count
                    total_assigned += assigned_count
                    print(
                        "  cells={} tech_pass={} assigned={} nnz={} seconds={:.1f}".format(
                            matrix.shape[1], tech_count, assigned_count,
                            matrix.nnz, time.time() - sample_start,
                        )
                    )
                    del matrix
        os.replace(str(annotation_temp), str(annotation_final))
    except Exception:
        if annotation_temp.exists():
            annotation_temp.unlink()
        raise

    if feature_reference is None:
        raise RuntimeError("No GSE214865 samples were processed")
    feature_ids = [row[0] for row in feature_reference]
    symbols = [row[1] for row in feature_reference]
    feature_types = [row[2] for row in feature_reference]
    pseudobulk_counts = np.concatenate(pseudobulk_blocks, axis=1)
    if pseudobulk_counts.shape[1] != len(pseudobulk_columns):
        raise RuntimeError("Pseudobulk column metadata mismatch")

    minimum_cells = int(
        lock["pseudobulk"]["minimum_cells_per_participant_lineage"]
    )
    gate_failures = []
    if total_tech_pass / float(max(total_input_cells, 1)) < float(
        lock["technical_QC"]["required_overall_technical_pass_fraction"]
    ):
        gate_failures.append("overall technical-QC pass fraction below lock")
    if total_assigned / float(max(total_tech_pass, 1)) < float(
        lock["annotation"]["required_assigned_fraction_among_technical_pass"]
    ):
        gate_failures.append("overall assigned fraction below lock")
    for row in sample_qc_rows:
        if int(row["technical_QC_pass_cells"]) < int(
            lock["technical_QC"]["required_technical_pass_cells_per_participant"]
        ):
            gate_failures.append(
                "technical-QC cells below lock for " + row["sample_id"]
            )
    for lineage in PRIMARY_LINEAGES:
        lineage_rows = [
            row for row in pseudobulk_columns
            if row["analysis_level"] == "primary" and row["cell_type"] == lineage
        ]
        if len(lineage_rows) != len(samples):
            gate_failures.append("missing primary pseudobulk columns for " + lineage)
        below = [row["sample_id"] for row in lineage_rows if int(row["cell_count"]) < minimum_cells]
        if below:
            gate_failures.append(
                "primary lineage {} below {} cells: {}".format(
                    lineage, minimum_cells, ",".join(below)
                )
            )

    feature_rows = ({
        "feature_index": str(index),
        "Ensembl_gene": feature_ids[index],
        "gene_symbol": symbols[index],
        "feature_type": feature_types[index],
        "feature_table_sha256_decompressed": feature_reference_sha,
    } for index in range(len(feature_ids)))
    write_gzip_tsv(
        output_dir / "feature_manifest.tsv.gz",
        [
            "feature_index", "Ensembl_gene", "gene_symbol", "feature_type",
            "feature_table_sha256_decompressed",
        ],
        feature_rows,
    )
    write_tsv(
        output_dir / "sample_qc.tsv", list(sample_qc_rows[0].keys()), sample_qc_rows
    )
    write_tsv(
        output_dir / "pseudobulk_metadata.tsv",
        list(pseudobulk_columns[0].keys()), pseudobulk_columns,
    )
    write_tsv(
        output_dir / "cell_type_counts.tsv",
        list(cell_type_rows[0].keys()), cell_type_rows,
    )
    write_tsv(
        output_dir / "input_hash_audit.tsv", list(hash_rows[0].keys()), hash_rows
    )

    count_npz = output_dir / "pseudobulk_counts.npz"
    count_temp = count_npz.with_name(count_npz.name + ".tmp")
    with count_temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            counts=pseudobulk_counts,
            Ensembl_gene=np.asarray(feature_ids),
            gene_symbol=np.asarray(symbols),
            column_id=np.asarray([row["column_id"] for row in pseudobulk_columns]),
        )
    os.replace(str(count_temp), str(count_npz))

    effect_rows = []
    primary_expression = np.empty((len(feature_ids), 0), dtype=np.float64)
    expression_columns = []
    lineage_summaries = []
    if not smoke and not gate_failures:
        (
            effect_rows, primary_expression, expression_columns,
            lineage_summaries,
        ) = fit_effects(
            np, feature_ids, symbols, pseudobulk_counts,
            pseudobulk_columns, lock,
        )
        if any(row["status"] != "PASS" for row in lineage_summaries):
            gate_failures.append("one or more primary lineage models failed")
    elif smoke:
        lineage_summaries = [{
            "lineage": lineage,
            "n_control": 1,
            "n_OSA": 1,
            "tested_genes": 0,
            "status": "NOT_FIT_SMOKE",
        } for lineage in PRIMARY_LINEAGES]

    if effect_rows:
        write_gzip_tsv(
            output_dir / "scrna_pseudobulk_effects.tsv.gz",
            list(effect_rows[0].keys()), effect_rows,
        )
        expression_npz = output_dir / "pseudobulk_primary_logcpm.npz"
        expression_temp = expression_npz.with_name(expression_npz.name + ".tmp")
        with expression_temp.open("wb") as handle:
            np.savez_compressed(
                handle,
                log2_CPM_plus_0p5=primary_expression,
                Ensembl_gene=np.asarray(feature_ids),
                gene_symbol=np.asarray(symbols),
                column_id=np.asarray([row["column_id"] for row in expression_columns]),
                subject_id=np.asarray([row["subject_id"] for row in expression_columns]),
                disease_group=np.asarray([row["disease_group"] for row in expression_columns]),
                lineage=np.asarray([row["lineage"] for row in expression_columns]),
            )
        os.replace(str(expression_temp), str(expression_npz))
        write_tsv(
            output_dir / "pseudobulk_primary_logcpm_metadata.tsv",
            list(expression_columns[0].keys()), expression_columns,
        )
    write_tsv(
        output_dir / "lineage_model_qc.tsv",
        list(lineage_summaries[0].keys()), lineage_summaries,
    )

    status = (
        "PASS_SMOKE" if smoke and not gate_failures
        else "PASS" if not gate_failures
        else "BLOCKED_QC"
    )
    artifact_paths = [
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "run_summary.json"
    ]
    summary = {
        "status": status,
        "script": relative_path(root, Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "seconds": time.time() - start_time,
        "root": str(root),
        "output": relative_path(root, output_dir),
        "smoke": smoke,
        "participants_processed": len(samples),
        "input_filtered_cells": total_input_cells,
        "technical_QC_pass_cells": total_tech_pass,
        "technical_QC_pass_fraction": total_tech_pass / float(max(total_input_cells, 1)),
        "assigned_cells": total_assigned,
        "assigned_fraction_of_technical_pass": total_assigned / float(max(total_tech_pass, 1)),
        "features": len(feature_ids),
        "matrix_nnz": total_nnz,
        "primary_lineages": list(PRIMARY_LINEAGES),
        "fine_types": list(FINE_TYPES) + ["Unassigned"],
        "gate_failures": gate_failures,
        "lineage_models": lineage_summaries,
        "frozen_candidate_or_pair_data_read": False,
        "validation_used_for_training_or_selection": False,
        "artifact_sha256": {
            relative_path(root, path): sha256_file(path)
            for path in sorted(artifact_paths, key=lambda value: value.name)
        },
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print("OSA GSE214865 participant-level pseudobulk: " + status)
    print("Output: " + str(output_dir))
    if gate_failures:
        print("Gate failures:")
        for failure in gate_failures:
            print("- " + failure)
    print("Frozen candidates/ranking read: no")
    return 0 if status.startswith("PASS") else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-stage", default=DEFAULT_STAGE)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
