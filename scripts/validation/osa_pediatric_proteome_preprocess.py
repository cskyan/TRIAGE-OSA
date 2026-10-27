#!/usr/bin/env python3
"""Preprocess the held-out PXD032734 pediatric serum TMT proteome.

Python 3.7 compatible.  The script parses the official Frontiers XLSX with
the Python standard library, verifies the locked 12-participant design, and
fits protein-level severity effects with participant-label permutation.

Discovery candidates, pair rankings, model checkpoints, and all earlier
validation outcomes are deliberately not read by this preprocessing stage.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
LOCK_REL = "config/osa_proteome_validation_lock.json"
DEFAULT_STAGE = "17_pediatric_proteome_preprocess_v1"


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
    if value is None:
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


def relative_path(root, path):
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def column_index(cell_reference):
    letters = "".join(ch for ch in cell_reference if ch.isalpha())
    value = 0
    for ch in letters.upper():
        value = value * 26 + ord(ch) - ord("A") + 1
    return value - 1


def shared_strings(archive):
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(name))
    values = []
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1] != "si":
            continue
        values.append("".join(
            child.text or "" for child in item.iter()
            if child.tag.rsplit("}", 1)[-1] == "t"
        ))
    return values


def xlsx_sheet_rows(path, wanted_sheet):
    with zipfile.ZipFile(str(path), "r") as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relation_id = None
        for node in workbook.iter():
            if node.tag.rsplit("}", 1)[-1] == "sheet" and node.get("name") == wanted_sheet:
                relation_id = node.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                break
        if relation_id is None:
            raise RuntimeError("Locked XLSX sheet is missing: " + wanted_sheet)
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for node in relationships:
            if node.get("Id") == relation_id:
                target = node.get("Target")
                break
        if not target:
            raise RuntimeError("Could not resolve locked XLSX sheet relationship")
        target = target.replace("\\", "/")
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = "xl/" + target
        strings = shared_strings(archive)
        sheet = ET.fromstring(archive.read(target))
        rows = []
        for row_node in sheet.iter():
            if row_node.tag.rsplit("}", 1)[-1] != "row":
                continue
            values = {}
            for cell in row_node:
                if cell.tag.rsplit("}", 1)[-1] != "c":
                    continue
                reference = cell.get("r", "")
                index = column_index(reference)
                kind = cell.get("t", "")
                raw = None
                if kind == "inlineStr":
                    raw = "".join(
                        node.text or "" for node in cell.iter()
                        if node.tag.rsplit("}", 1)[-1] == "t"
                    )
                else:
                    for node in cell:
                        if node.tag.rsplit("}", 1)[-1] == "v":
                            raw = node.text
                            break
                    if raw is not None and kind == "s":
                        raw = strings[int(raw)]
                values[index] = "" if raw is None else raw
            if values:
                width = max(values) + 1
                rows.append([values.get(index, "") for index in range(width)])
    return rows


def bh_adjust(values):
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [1.0] * len(values)
    running = 1.0
    for reverse in range(len(values) - 1, -1, -1):
        index = order[reverse]
        rank = reverse + 1
        running = min(running, values[index] * len(values) / float(rank))
        result[index] = min(1.0, running)
    return result


def beta_continued_fraction(a, b, x):
    maximum_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def t_two_sided_p(value, degrees_freedom):
    if not math.isfinite(value):
        return 0.0 if math.isinf(value) else 1.0
    x = float(degrees_freedom) / (degrees_freedom + value * value)
    return min(1.0, max(0.0, regularized_beta(
        x, 0.5 * degrees_freedom, 0.5
    )))


def fit_effects(np, matrix, severity):
    severity = np.asarray(severity, dtype=np.float64)
    centered_x = severity - np.mean(severity)
    ssx = float(np.dot(centered_x, centered_x))
    means = np.mean(matrix, axis=1)
    slopes = np.dot(matrix, centered_x) / ssx
    fitted = means.reshape(-1, 1) + slopes.reshape(-1, 1) * centered_x.reshape(1, -1)
    residual = matrix - fitted
    slope_df = matrix.shape[1] - 2
    slope_mse = np.sum(residual * residual, axis=1) / float(slope_df)
    slope_se = np.sqrt(slope_mse / ssx)
    slope_t = np.divide(slopes, slope_se, out=np.zeros_like(slopes), where=slope_se > 0)
    slope_p = np.asarray([
        t_two_sided_p(value, slope_df) for value in slope_t
    ], dtype=np.float64)

    group_means = np.stack([
        np.mean(matrix[:, severity == group], axis=1) for group in range(4)
    ], axis=1)
    within = matrix - group_means[:, severity.astype(int)]
    contrast_df = matrix.shape[1] - 4
    contrast_mse = np.sum(within * within, axis=1) / float(contrast_df)
    severe_difference = group_means[:, 3] - group_means[:, 0]
    contrast_se = np.sqrt(contrast_mse * (1.0 / 3.0 + 1.0 / 3.0))
    contrast_t = np.divide(
        severe_difference, contrast_se,
        out=np.zeros_like(severe_difference), where=contrast_se > 0,
    )
    contrast_p = np.asarray([
        t_two_sided_p(value, contrast_df) for value in contrast_t
    ], dtype=np.float64)
    return {
        "slope": slopes,
        "slope_se": slope_se,
        "slope_t": slope_t,
        "slope_p": slope_p,
        "severe_difference": severe_difference,
        "severe_se": contrast_se,
        "severe_t": contrast_t,
        "severe_p": contrast_p,
    }


def permutation_p_values(np, matrix, severity, permutations, seed):
    rng = np.random.RandomState(seed)
    severity = np.asarray(severity, dtype=np.int64)
    centered = severity.astype(np.float64) - float(np.mean(severity))
    denominator = float(np.dot(centered, centered))
    observed_slope = np.abs(np.dot(matrix, centered) / denominator)
    observed_contrast = np.abs(
        np.mean(matrix[:, severity == 3], axis=1)
        - np.mean(matrix[:, severity == 0], axis=1)
    )
    slope_exceed = np.zeros(matrix.shape[0], dtype=np.int64)
    contrast_exceed = np.zeros(matrix.shape[0], dtype=np.int64)
    chunk = 500
    for start in range(0, permutations, chunk):
        count = min(chunk, permutations - start)
        labels = np.stack([rng.permutation(severity) for _ in range(count)], axis=0)
        centered_labels = labels.astype(np.float64) - np.mean(labels, axis=1).reshape(-1, 1)
        permuted_slopes = np.abs(np.dot(centered_labels, matrix.T) / denominator)
        slope_exceed += np.sum(
            permuted_slopes >= observed_slope.reshape(1, -1) - 1.0e-15,
            axis=0,
        )
        weights = np.zeros_like(labels, dtype=np.float64)
        weights[labels == 3] = 1.0 / 3.0
        weights[labels == 0] = -1.0 / 3.0
        permuted_contrasts = np.abs(np.dot(weights, matrix.T))
        contrast_exceed += np.sum(
            permuted_contrasts >= observed_contrast.reshape(1, -1) - 1.0e-15,
            axis=0,
        )
    return (
        (slope_exceed.astype(np.float64) + 1.0) / float(permutations + 1),
        (contrast_exceed.astype(np.float64) + 1.0) / float(permutations + 1),
    )


def parse_number(value, field, accession):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise RuntimeError("Invalid {} for {}: {!r}".format(field, accession, value))
    if not math.isfinite(result):
        raise RuntimeError("Non-finite {} for {}".format(field, accession))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError as error:
        print("ERROR: numpy is required: {}".format(error), file=sys.stderr)
        return 2

    root = Path(args.root).resolve()
    lock_path = root / LOCK_REL
    if not lock_path.is_file():
        print("ERROR: missing lock: {}".format(lock_path), file=sys.stderr)
        return 2
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    workbook_path = root / lock["source"]["workbook"]
    clinical_path = root / lock["source"]["clinical_table"]
    print("OSA pediatric proteome preprocessing environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("Workbook: {}".format(workbook_path))
    print("Candidate/ranking/validation data read: no")
    if args.check_env:
        return 0

    for path in (workbook_path, clinical_path):
        if not path.is_file():
            print("ERROR: missing locked source: {}".format(path), file=sys.stderr)
            return 2
    if workbook_path.stat().st_size != int(lock["source"]["workbook_size"]):
        print("ERROR: workbook size does not match lock", file=sys.stderr)
        return 2
    if sha256_file(workbook_path) != lock["source"]["workbook_sha256"]:
        print("ERROR: workbook SHA256 does not match lock", file=sys.stderr)
        return 2
    if sha256_file(clinical_path) != lock["source"]["clinical_table_sha256"]:
        print("ERROR: clinical-table SHA256 does not match lock", file=sys.stderr)
        return 2

    output_dir = root / "results" / "discovery_preprocess_v2" / args.stage
    if output_dir.exists():
        if not args.force:
            print("ERROR: output exists; pass --force: {}".format(output_dir), file=sys.stderr)
            return 2
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True)

    raw_rows = xlsx_sheet_rows(workbook_path, lock["source"]["sheet"])
    expected_columns = lock["source"]["expected_columns"]
    header = raw_rows[0] + [""] * (len(expected_columns) - len(raw_rows[0]))
    header = header[:len(expected_columns)]
    if header != expected_columns:
        raise RuntimeError("Official workbook header does not match frozen lock")
    data_rows = []
    for raw in raw_rows[1:]:
        row = raw + [""] * (len(expected_columns) - len(raw))
        if any(str(value).strip() for value in row):
            data_rows.append(row[:len(expected_columns)])
    if len(data_rows) != int(lock["source"]["expected_rows"]):
        raise RuntimeError("Expected {} protein rows, found {}".format(
            lock["source"]["expected_rows"], len(data_rows)
        ))

    sample_ids = list(lock["sample_groups"].keys())
    if sample_ids != expected_columns[14:26]:
        raise RuntimeError("Locked sample order does not match workbook")
    severity = np.asarray([
        lock["sample_groups"][sample]["severity_ordinal"] for sample in sample_ids
    ], dtype=np.int64)
    if list(np.bincount(severity, minlength=4)) != [3, 3, 3, 3]:
        raise RuntimeError("Locked PXD032734 group counts are not 3/3/3/3")

    quantified = []
    excluded = []
    accessions = set()
    for values in data_rows:
        accession = str(values[0]).strip()
        if not accession or accession in accessions:
            raise RuntimeError("Blank or duplicate protein accession: " + accession)
        accessions.add(accession)
        unique_peptides = int(round(parse_number(values[6], "unique peptides", accession)))
        intensities = [
            parse_number(values[14 + index], sample_ids[index], accession)
            for index in range(len(sample_ids))
        ]
        # Proteome Discoverer reports one quantified protein group supported by
        # razor/shared evidence but zero unique peptides.  The article's 752
        # quantified proteins are reproduced by the reporter-intensity rule;
        # unique-peptide count is retained as an evidence-quality attribute.
        valid = all(value > 0.0 for value in intensities)
        record = {
            "accession": accession,
            "gene_symbol": str(values[1]).strip(),
            "description": str(values[2]).strip(),
            "coverage_percent": parse_number(values[3], "coverage", accession),
            "peptides": int(round(parse_number(values[4], "peptides", accession))),
            "psms": int(round(parse_number(values[5], "PSMs", accession))),
            "unique_peptides": unique_peptides,
            "amino_acids": int(round(parse_number(values[7], "amino acids", accession))),
            "molecular_weight_kda": parse_number(values[8], "molecular weight", accession),
            "score": parse_number(values[10], "score", accession),
            "intensities": intensities,
        }
        if valid:
            quantified.append(record)
        else:
            excluded.append(record)
    if len(quantified) != int(lock["source"]["expected_quantified_rows"]):
        raise RuntimeError("Expected {} quantified rows, found {}".format(
            lock["source"]["expected_quantified_rows"], len(quantified)
        ))

    intensity = np.asarray([row["intensities"] for row in quantified], dtype=np.float64)
    log2_raw = np.log2(intensity)
    sample_medians = np.median(log2_raw, axis=0)
    grand_median = float(np.median(sample_medians))
    log2_centered = log2_raw - sample_medians.reshape(1, -1) + grand_median
    primary = fit_effects(np, log2_raw, severity)
    sensitivity = fit_effects(np, log2_centered, severity)
    permutations = int(lock["statistics"]["participant_label_permutations"])
    primary_perm_slope, primary_perm_severe = permutation_p_values(
        np, log2_raw, severity, permutations,
        int(lock["statistics"]["random_seed"]),
    )
    slope_fdr = bh_adjust(primary["slope_p"])
    slope_perm_fdr = bh_adjust(primary_perm_slope)
    severe_fdr = bh_adjust(primary["severe_p"])
    severe_perm_fdr = bh_adjust(primary_perm_severe)

    sample_rows = []
    for index, sample in enumerate(sample_ids):
        sample_rows.append({
            "dataset": "PXD032734",
            "sample_id": sample,
            "subject_id": sample,
            "group": lock["sample_groups"][sample]["group"],
            "severity_ordinal": int(severity[index]),
            "quantified_proteins": len(quantified),
            "log2_median_before_centering": fmt(sample_medians[index]),
            "log2_median_after_centering": fmt(np.median(log2_centered[:, index])),
            "candidate_selection_allowed": "no",
        })
    write_tsv(output_dir / "sample_manifest.tsv", list(sample_rows[0]), sample_rows)

    matrix_fields = [
        "accession", "gene_symbol", "description", "coverage_percent",
        "peptides", "psms", "unique_peptides", "amino_acids",
        "molecular_weight_kda", "score",
    ] + ["intensity_" + sample for sample in sample_ids] + [
        "log2_" + sample for sample in sample_ids
    ] + ["log2_median_centered_" + sample for sample in sample_ids]
    matrix_rows = []
    effect_rows = []
    for index, record in enumerate(quantified):
        matrix_row = {key: record[key] for key in matrix_fields if key in record}
        for sample_index, sample in enumerate(sample_ids):
            matrix_row["intensity_" + sample] = fmt(intensity[index, sample_index])
            matrix_row["log2_" + sample] = fmt(log2_raw[index, sample_index])
            matrix_row["log2_median_centered_" + sample] = fmt(
                log2_centered[index, sample_index]
            )
        matrix_rows.append(matrix_row)
        effect_rows.append({
            "accession": record["accession"],
            "gene_symbol": record["gene_symbol"],
            "description": record["description"],
            "coverage_percent": fmt(record["coverage_percent"]),
            "peptides": record["peptides"],
            "psms": record["psms"],
            "unique_peptides": record["unique_peptides"],
            "score": fmt(record["score"]),
            "primary_ordinal_slope": fmt(primary["slope"][index]),
            "primary_ordinal_se": fmt(primary["slope_se"][index]),
            "primary_ordinal_t": fmt(primary["slope_t"][index]),
            "primary_ordinal_p": fmt(primary["slope_p"][index]),
            "primary_ordinal_fdr_bh": fmt(slope_fdr[index]),
            "primary_ordinal_permutation_p": fmt(primary_perm_slope[index]),
            "primary_ordinal_permutation_fdr_bh": fmt(slope_perm_fdr[index]),
            "primary_severe_vs_control_log2fc": fmt(primary["severe_difference"][index]),
            "primary_severe_vs_control_se": fmt(primary["severe_se"][index]),
            "primary_severe_vs_control_t": fmt(primary["severe_t"][index]),
            "primary_severe_vs_control_p": fmt(primary["severe_p"][index]),
            "primary_severe_vs_control_fdr_bh": fmt(severe_fdr[index]),
            "primary_severe_vs_control_permutation_p": fmt(primary_perm_severe[index]),
            "primary_severe_vs_control_permutation_fdr_bh": fmt(severe_perm_fdr[index]),
            "sensitivity_ordinal_slope": fmt(sensitivity["slope"][index]),
            "sensitivity_ordinal_p": fmt(sensitivity["slope_p"][index]),
            "sensitivity_severe_vs_control_log2fc": fmt(
                sensitivity["severe_difference"][index]
            ),
            "sensitivity_severe_vs_control_p": fmt(sensitivity["severe_p"][index]),
            "candidate_selection_allowed": "no",
        })
    write_gzip_tsv(output_dir / "protein_quantification.tsv.gz", matrix_fields, matrix_rows)
    write_gzip_tsv(
        output_dir / "protein_effects.tsv.gz", list(effect_rows[0]), effect_rows
    )

    excluded_rows = [{
        "accession": row["accession"],
        "gene_symbol": row["gene_symbol"],
        "unique_peptides": row["unique_peptides"],
        "reason": "not_quantified_by_locked_rule",
    } for row in excluded]
    write_tsv(
        output_dir / "excluded_protein_groups.tsv",
        ["accession", "gene_symbol", "unique_peptides", "reason"],
        excluded_rows,
    )

    source_rows = []
    for role, path, expected in (
        ("quantification_workbook", workbook_path, lock["source"]["workbook_sha256"]),
        ("clinical_group_table", clinical_path, lock["source"]["clinical_table_sha256"]),
        ("analysis_lock", lock_path, sha256_file(lock_path)),
    ):
        source_rows.append({
            "role": role,
            "path": relative_path(root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "expected_sha256": expected,
            "match": "yes" if sha256_file(path) == expected else "no",
        })
    write_tsv(output_dir / "input_hash_audit.tsv", list(source_rows[0]), source_rows)
    shutil.copy2(str(lock_path), str(output_dir / "analysis_lock_snapshot.json"))

    artifact_paths = [
        output_dir / "sample_manifest.tsv",
        output_dir / "protein_quantification.tsv.gz",
        output_dir / "protein_effects.tsv.gz",
        output_dir / "excluded_protein_groups.tsv",
        output_dir / "input_hash_audit.tsv",
        output_dir / "analysis_lock_snapshot.json",
    ]
    manifest_rows = [{
        "path": relative_path(root, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in artifact_paths]
    write_tsv(output_dir / "output_manifest.tsv", list(manifest_rows[0]), manifest_rows)

    summary = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "script": relative_path(root, Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": utc_now(),
        "dataset": "PXD032734",
        "participants": 12,
        "groups": {"non_OSA": 3, "mild_OSA": 3, "moderate_OSA": 3, "severe_OSA": 3},
        "reported_protein_groups": len(data_rows),
        "quantified_proteins": len(quantified),
        "excluded_protein_groups": len(excluded),
        "sample_log2_median_range": fmt(float(np.max(sample_medians) - np.min(sample_medians))),
        "participant_label_permutations": permutations,
        "candidate_or_pair_data_read": False,
        "validation_used_for_training_or_selection": False,
        "artifact_sha256": {
            relative_path(root, path): sha256_file(path) for path in artifact_paths
        },
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print("OSA pediatric proteome preprocessing: PASS")
    print("Participants: 12 (3 per severity group)")
    print("Quantified proteins: {} / {}".format(len(quantified), len(data_rows)))
    print("Output: {}".format(output_dir))
    print("No discovery candidate, pair ranking, or previous validation data was read.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
