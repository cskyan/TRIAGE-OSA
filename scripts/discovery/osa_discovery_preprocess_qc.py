#!/usr/bin/env python3
'Preprocess and technically QC the locked OSA discovery microarrays.\n\nThis is a Python-only Phase-2 pipeline for GSE75097 and GSE71356.  It is\ncompatible with Python 3.7+ and does not require R/Bioconductor.  Raw inputs\nare never modified.\n\nPrimary preprocessing is deliberately outcome-blind and dataset-specific:\n\n1. read exported non-normalized intensity and Detection P-value pairs;\n2. explicitly map array labels to GEO GSM accessions;\n3. audit the signal scale and apply log2 only when the data are linear-scale;\n4. apply a locked dataset-specific detection rule: P < 0.01 in at least six\n   arrays for GSE75097, and P < 0.05 in at least two arrays for GSE71356;\n5. quantile-normalize each dataset independently;\n6. calculate technical sample correlation from all normalized input probes;\n7. audit deposited series-matrix column identity before concordance testing;\n8. write an explicit sample-freeze manifest and GSE71356 detection-threshold\n   sensitivity analysis.\n\nThe pipeline does not perform differential expression and never reads a\nvalidation outcome.  GSE133601, GSE49800, GSE135917, GSE214865 and PXD032734\nare outside the executable dataset catalog in this script.\n\nRequired Python packages\n------------------------\nnumpy, pandas, matplotlib\n\nPython 3.7-compatible installation, if needed::\n\n    python -m pip install numpy==1.21.6 pandas==1.3.5 matplotlib==3.5.3\n\nExamples\n--------\nEnvironment and input check::\n\n    python scripts/discovery/osa_discovery_preprocess_qc.py         --root .         --check-env\n\nFull preprocessing::\n\n    python scripts/discovery/osa_discovery_preprocess_qc.py         --root .         --run-id discovery_preprocess_v1\n'

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.1.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


DATASETS = {
    "GSE75097": {
        "raw": "geo/GSE75097_non-normalized.txt.gz",
        "series": "geo/GSE75097_series_matrix.txt.gz",
        "annotation": "annotations/GPL10558.annot.gz",
        "detection_p": 0.01,
        "minimum_detected_samples": 6,
        "comparison": "gene_symbol_median",
        "deposited_processing": "quantile normalization with IlluminaGUI in R",
    },
    "GSE71356": {
        "raw": "geo/GSE71356_non-normalized_data.txt.gz",
        "series": "geo/GSE71356_series_matrix.txt.gz",
        "annotation": "annotations/GPL2700.annot.gz",
        "detection_p": 0.05,
        "minimum_detected_samples": 2,
        "comparison": "probe_id",
        "deposited_processing": "BeadStudio detection filtering plus background/brightness adjustment",
        "detection_sensitivity_minimum_samples": [2, 3, 4, 6, 8],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def require_scientific_python() -> Tuple[Any, Any, Any]:
    missing = []  # type: List[str]
    modules = {}  # type: Dict[str, Any]
    for name in ["numpy", "pandas", "matplotlib"]:
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing Python packages: {}. For Python 3.7 run: "
            "python -m pip install numpy==1.21.6 pandas==1.3.5 "
            "matplotlib==3.5.3".format(", ".join(missing))
        )
    matplotlib = modules["matplotlib"]
    matplotlib.use("Agg")
    return modules["numpy"], modules["pandas"], matplotlib


def read_geo_metadata(path: Path) -> Dict[str, List[List[str]]]:
    rows = defaultdict(list)  # type: Dict[str, List[List[str]]]
    with gzip.open(str(path), "rt", encoding="utf-8-sig", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                break
            if line.startswith("!"):
                parts = next(csv.reader([line], delimiter="\t", quotechar='"'))
                rows[parts[0]].append(parts[1:])
    return dict(rows)


def series_sample_mapping(dataset: str, series_path: Path) -> Tuple[List[str], Dict[str, str]]:
    metadata = read_geo_metadata(series_path)
    ids = metadata.get("!Sample_geo_accession", [[]])[0]
    titles = metadata.get("!Sample_title", [[]])[0]
    if not ids or len(ids) != len(titles):
        raise RuntimeError("Malformed GEO sample metadata: " + str(series_path))

    raw_to_gsm = {}  # type: Dict[str, str]
    if dataset == "GSE75097":
        for gsm, title in zip(ids, titles):
            match = re.search(r"\[([^\]]+)\]\s*$", title)
            if not match:
                raise RuntimeError("Cannot recover GSE75097 array label from title: " + title)
            raw_to_gsm[match.group(1).strip()] = gsm
    elif dataset == "GSE71356":
        description_rows = metadata.get("!Sample_description", [])
        if not description_rows or len(description_rows[0]) != len(ids):
            raise RuntimeError("GSE71356 first Sample_description row is unavailable")
        for gsm, description in zip(ids, description_rows[0]):
            raw_to_gsm[description.strip()] = gsm
    else:
        raise RuntimeError("Unsupported discovery dataset: " + dataset)
    if len(raw_to_gsm) != len(ids):
        raise RuntimeError("Raw-label mapping is not one-to-one for " + dataset)
    return list(ids), raw_to_gsm


def load_sample_manifest(root: Path, pd: Any) -> Any:
    path = root / "metadata" / "sample_manifest.tsv"
    if not path.is_file():
        raise RuntimeError("Missing metadata/sample_manifest.tsv")
    frame = pd.read_csv(str(path), sep="\t", dtype=str, keep_default_na=False)
    required = {
        "dataset", "sample_id", "subject_id", "study_group", "condition",
        "timepoint", "candidate_selection_allowed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("sample_manifest.tsv lacks columns: " + ", ".join(missing))
    duplicated = frame.duplicated(subset=["dataset", "sample_id"], keep=False)
    if bool(duplicated.any()):
        raise RuntimeError("sample_manifest.tsv has duplicate dataset/sample_id keys")
    return frame


def validate_analysis_lock(root: Path, explicit: Optional[Path] = None) -> Path:
    path = explicit.expanduser() if explicit is not None else (root / "config" / "analysis_lock.yaml")
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(
            "Missing config/analysis_lock.yaml. Copy the file generated on the server "
            "before preprocessing."
        )
    text = path.read_text(encoding="utf-8")
    required_tokens = [
        'allowed_datasets: ["GSE75097", "GSE71356"]',
        'validation_may_select_candidates: false',
        'cross_platform_expression_merge: false',
        'raw_data_immutable: true',
    ]
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise RuntimeError("analysis_lock.yaml is missing locked safeguards: " + "; ".join(missing))
    return path


def load_sample_freeze_policy(
    root: Path, explicit: Optional[Path] = None
) -> Tuple[Path, Dict[str, Any]]:
    path = (
        explicit.expanduser()
        if explicit is not None
        else (root / "config" / "discovery_sample_freeze_policy.json")
    )
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("Missing discovery sample-freeze policy: " + str(path))
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError("Cannot read sample-freeze policy {}: {}".format(path, exc))
    datasets = policy.get("datasets", {})
    for dataset in DATASETS:
        dataset_policy = datasets.get(dataset)
        if not isinstance(dataset_policy, dict):
            raise RuntimeError("Sample-freeze policy lacks dataset " + dataset)
        primary_set = dataset_policy.get("primary_set")
        analysis_sets = dataset_policy.get("analysis_sets", {})
        if primary_set not in analysis_sets:
            raise RuntimeError(
                "Sample-freeze policy primary_set is missing for " + dataset
            )
        for set_name, set_policy in analysis_sets.items():
            if not re.match(r"^[A-Za-z0-9_]+$", str(set_name)):
                raise RuntimeError("Invalid sample-freeze analysis-set name: " + str(set_name))
            for field in ["exclude_sample_ids", "exclude_subject_ids"]:
                if not isinstance(set_policy.get(field, []), list):
                    raise RuntimeError(
                        "Sample-freeze {} {} must be a list".format(set_name, field)
                    )
    return path, policy


def read_raw_export(
    dataset: str,
    raw_path: Path,
    series_path: Path,
    sample_manifest: Any,
    np: Any,
    pd: Any,
) -> Tuple[Any, Any, Any]:
    frame = pd.read_csv(
        str(raw_path), sep="\t", compression="gzip", dtype=str,
        keep_default_na=False, low_memory=False,
    )
    if frame.shape[1] < 3 or (frame.shape[1] - 1) % 2 != 0:
        raise RuntimeError("Expected alternating intensity/Detection-P columns in " + str(raw_path))
    probe_col = frame.columns[0]
    intensity_columns = list(frame.columns[1::2])
    detection_columns = list(frame.columns[2::2])
    if len(intensity_columns) != len(detection_columns):
        raise RuntimeError("Intensity and Detection-P column counts differ")

    expected_gsms, raw_to_gsm = series_sample_mapping(dataset, series_path)
    missing_labels = [label for label in intensity_columns if label not in raw_to_gsm]
    extra_labels = [label for label in raw_to_gsm if label not in intensity_columns]
    if missing_labels or extra_labels:
        raise RuntimeError(
            "Raw/GEO sample mapping mismatch for {}. Unmapped raw labels: {}; "
            "missing raw columns: {}".format(dataset, missing_labels, extra_labels)
        )
    gsm_columns = [raw_to_gsm[label] for label in intensity_columns]
    if set(gsm_columns) != set(expected_gsms):
        raise RuntimeError("Raw columns do not map to all deposited GSM samples for " + dataset)

    manifest = sample_manifest[sample_manifest["dataset"] == dataset].copy()
    if manifest.shape[0] != len(expected_gsms):
        raise RuntimeError("Manifest sample count mismatch for " + dataset)
    if set(manifest["sample_id"]) != set(expected_gsms):
        raise RuntimeError("Manifest GSM set mismatch for " + dataset)
    if set(manifest["candidate_selection_allowed"]) != {"yes"}:
        raise RuntimeError(dataset + " is not uniformly allowed for discovery selection")

    probe_ids = frame[probe_col].astype(str)
    if bool(probe_ids.duplicated().any()):
        duplicates = int(probe_ids.duplicated().sum())
        raise RuntimeError("{} duplicate probe identifiers in {}".format(duplicates, dataset))

    intensity = frame[intensity_columns].apply(pd.to_numeric, errors="coerce")
    detection = frame[detection_columns].apply(pd.to_numeric, errors="coerce")
    intensity.columns = gsm_columns
    detection.columns = gsm_columns
    intensity.index = probe_ids
    detection.index = probe_ids
    intensity.index.name = "probe_id"
    detection.index.name = "probe_id"
    intensity = intensity.loc[:, expected_gsms]
    detection = detection.loc[:, expected_gsms]

    if bool(intensity.isna().any().any()) or bool(detection.isna().any().any()):
        raise RuntimeError("Non-numeric or missing intensity/Detection-P values in " + dataset)
    values = intensity.to_numpy(dtype=float)
    pvalues = detection.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()) or not bool(np.isfinite(pvalues).all()):
        raise RuntimeError("Non-finite raw values in " + dataset)
    if float(pvalues.min()) < 0.0 or float(pvalues.max()) > 1.0:
        raise RuntimeError("Detection P values outside [0,1] in " + dataset)

    sample_map = manifest.set_index("sample_id").loc[expected_gsms].reset_index()
    sample_map.insert(1, "raw_array_label", [label for label in intensity_columns])
    return intensity, detection, sample_map


def scale_and_log2(intensity: Any, np: Any, pd: Any) -> Tuple[Any, Dict[str, Any]]:
    values = intensity.to_numpy(dtype=float)
    quantiles = np.nanpercentile(values, [0, 1, 25, 50, 75, 99, 100])
    q99 = float(quantiles[5])
    maximum = float(quantiles[6])
    if q99 > 100.0 or maximum > 1000.0:
        scale = "linear"
        floor = 1.0
        floored = int((values < floor).sum())
        transformed_values = np.log2(np.maximum(values, floor))
        transformed = pd.DataFrame(
            transformed_values, index=intensity.index.copy(), columns=intensity.columns.copy()
        )
        action = "log2(max(intensity,1.0))"
    elif q99 < 50.0 and maximum < 100.0:
        scale = "log2"
        floor = None
        floored = 0
        transformed = intensity.astype(float).copy()
        action = "none_already_log2"
    else:
        raise RuntimeError(
            "Ambiguous signal scale: q99={:.4g}, max={:.4g}; do not transform automatically".format(
                q99, maximum
            )
        )
    summary = {
        "detected_input_scale": scale,
        "transform": action,
        "floor": floor,
        "floored_value_count": floored,
        "raw_quantiles": {
            "min": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "p25": float(quantiles[2]),
            "median": float(quantiles[3]),
            "p75": float(quantiles[4]),
            "p99": q99,
            "max": maximum,
        },
    }
    return transformed, summary


def quantile_normalize(frame: Any, np: Any, pd: Any) -> Any:
    values = frame.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise RuntimeError("Quantile normalization requires a complete finite matrix")
    order = np.argsort(values, axis=0, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=0)
    rank_means = sorted_values.mean(axis=1)
    normalized = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        normalized[order[:, column], column] = rank_means
    return pd.DataFrame(normalized, index=frame.index.copy(), columns=frame.columns.copy())


def robust_bounds(values: Any, np: Any, multiplier: float = 4.0) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad == 0.0:
        return -float("inf"), float("inf")
    scaled_mad = 1.4826 * mad
    return median - multiplier * scaled_mad, median + multiplier * scaled_mad


def sample_and_probe_qc(
    log2_matrix: Any,
    normalized: Any,
    detection: Any,
    detection_p: float,
    minimum_detected_samples: int,
    np: Any,
    pd: Any,
) -> Tuple[Any, Any, Any, Any]:
    detected = detection < detection_p
    detected_count = detected.sum(axis=1).astype(int)
    keep = detected_count >= int(minimum_detected_samples)
    if int(keep.sum()) < 100:
        raise RuntimeError("Detection filtering retained fewer than 100 probes")
    filtered = normalized.loc[keep].copy()
    # Technical sample correlation must use the complete normalized platform
    # probe universe.  A detection-filtered subset can be dominated by
    # sample-specific low-detection probes (notably in GSE71356), producing an
    # artificially alarming correlation heatmap that is unsuitable for array
    # QC.  The filtered correlation is retained as a clearly named secondary
    # output for biological/discovery structure only.
    technical_correlation = normalized.corr(method="pearson")
    filtered_correlation = filtered.corr(method="pearson")

    sample_rows = []  # type: List[Dict[str, Any]]
    for sample in normalized.columns:
        raw_values = log2_matrix[sample].to_numpy(dtype=float)
        norm_values = normalized[sample].to_numpy(dtype=float)
        sample_rows.append({
            "sample_id": sample,
            "log2_min": float(np.min(raw_values)),
            "log2_q25": float(np.percentile(raw_values, 25)),
            "log2_median": float(np.median(raw_values)),
            "log2_q75": float(np.percentile(raw_values, 75)),
            "log2_max": float(np.max(raw_values)),
            "detection_rate_p_lt_{}".format(detection_p): float(detected[sample].mean()),
            "normalized_median": float(np.median(norm_values)),
            "normalized_iqr": float(np.percentile(norm_values, 75) - np.percentile(norm_values, 25)),
            "median_inter_sample_correlation": float(
                technical_correlation[sample].drop(sample).median()
            ),
        })
    sample_qc = pd.DataFrame(sample_rows)
    detect_field = "detection_rate_p_lt_{}".format(detection_p)
    median_low, median_high = robust_bounds(sample_qc["log2_median"], np)
    detect_low, _ = robust_bounds(sample_qc[detect_field], np)
    corr_low, _ = robust_bounds(sample_qc["median_inter_sample_correlation"], np)

    flags = []  # type: List[str]
    statuses = []  # type: List[str]
    for _, row in sample_qc.iterrows():
        sample_flags = []  # type: List[str]
        if float(row["log2_median"]) < median_low or float(row["log2_median"]) > median_high:
            sample_flags.append("raw_median_outlier")
        if float(row[detect_field]) < detect_low:
            sample_flags.append("low_detection_rate")
        if float(row["median_inter_sample_correlation"]) < corr_low:
            sample_flags.append("low_inter_sample_correlation")
        flags.append(";".join(sample_flags))
        statuses.append("REVIEW" if sample_flags else "PASS")
    sample_qc["technical_flags"] = flags
    sample_qc["qc_status"] = statuses
    sample_qc["automatic_exclusion"] = "no"

    probe_qc = pd.DataFrame({
        "probe_id": normalized.index.astype(str),
        "detected_count": detected_count.to_numpy(dtype=int),
        "detected_fraction": detected.mean(axis=1).to_numpy(dtype=float),
        "median_detection_p": detection.median(axis=1).to_numpy(dtype=float),
        "mean_log2_before_normalization": log2_matrix.mean(axis=1).to_numpy(dtype=float),
        "sd_log2_before_normalization": log2_matrix.std(axis=1, ddof=1).to_numpy(dtype=float),
        "sd_after_quantile_normalization": normalized.std(axis=1, ddof=1).to_numpy(dtype=float),
        "keep_primary": np.where(keep.to_numpy(dtype=bool), "yes", "no"),
    })
    return sample_qc, probe_qc, filtered, technical_correlation, filtered_correlation


def pca_scores(filtered: Any, sample_map: Any, top_n: int, np: Any, pd: Any) -> Tuple[Any, List[float]]:
    variances = filtered.var(axis=1, ddof=1).sort_values(ascending=False)
    chosen = variances.index[: min(int(top_n), len(variances))]
    matrix = filtered.loc[chosen].T.to_numpy(dtype=float)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    scores = u * singular
    variance = singular ** 2
    explained = variance / variance.sum() if float(variance.sum()) > 0 else variance
    columns = ["PC{}".format(i + 1) for i in range(min(3, scores.shape[1]))]
    result = pd.DataFrame(scores[:, : len(columns)], columns=columns)
    result.insert(0, "sample_id", list(filtered.columns))
    metadata_columns = [
        column for column in ["sample_id", "subject_id", "study_group", "condition", "timepoint"]
        if column in sample_map.columns
    ]
    result = result.merge(sample_map[metadata_columns], on="sample_id", how="left", validate="one_to_one")
    return result, [float(value) for value in explained[: len(columns)]]


def add_pca_review_flags(sample_qc: Any, scores: Any, np: Any, pd: Any) -> Any:
    pc_columns = [column for column in ["PC1", "PC2", "PC3"] if column in scores.columns]
    standardized = []  # type: List[Any]
    for column in pc_columns:
        values = scores[column].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad == 0.0:
            standardized.append(np.zeros_like(values))
        else:
            standardized.append((values - median) / (1.4826 * mad))
    if standardized:
        distance = np.sqrt(np.sum(np.vstack(standardized) ** 2, axis=0))
    else:
        distance = np.zeros(scores.shape[0], dtype=float)
    _, upper = robust_bounds(distance, np, multiplier=4.0)
    pca_table = pd.DataFrame({
        "sample_id": scores["sample_id"].tolist(),
        "pca_robust_distance": distance,
        "pca_review_threshold": upper,
    })
    result = sample_qc.merge(pca_table, on="sample_id", how="left", validate="one_to_one")
    for index, row in result.iterrows():
        if float(row["pca_robust_distance"]) > float(row["pca_review_threshold"]):
            existing = str(row["technical_flags"]).strip(";")
            flags = [value for value in existing.split(";") if value]
            if "pca_outlier" not in flags:
                flags.append("pca_outlier")
            result.at[index, "technical_flags"] = ";".join(flags)
            result.at[index, "qc_status"] = "REVIEW"
    return result


def read_series_matrix(path: Path, pd: Any) -> Any:
    begin_line = None  # type: Optional[int]
    with gzip.open(str(path), "rt", encoding="utf-8-sig", errors="replace") as handle:
        for number, line in enumerate(handle):
            if line.strip() == "!series_matrix_table_begin":
                begin_line = number
                break
    if begin_line is None:
        raise RuntimeError("Series matrix table marker not found: " + str(path))
    frame = pd.read_csv(
        str(path), sep="\t", compression="gzip", skiprows=begin_line + 1,
        comment="!", dtype=str, keep_default_na=False, low_memory=False,
    )
    if frame.empty or frame.shape[1] < 2:
        raise RuntimeError("Empty series matrix table: " + str(path))
    feature_column = frame.columns[0]
    feature_ids = frame[feature_column].astype(str)
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values.index = feature_ids
    values.index.name = "feature_id"
    if bool(values.isna().any().any()):
        raise RuntimeError("Non-numeric deposited series matrix values: " + str(path))
    if bool(values.index.duplicated().any()):
        values = values.groupby(level=0, sort=False).median()
    return values


def read_annotation(path: Path, pd: Any) -> Any:
    begin_line = None  # type: Optional[int]
    with gzip.open(str(path), "rt", encoding="utf-8-sig", errors="replace") as handle:
        for number, line in enumerate(handle):
            if line.strip() == "!platform_table_begin":
                begin_line = number
                break
    if begin_line is None:
        raise RuntimeError("Platform annotation table marker not found: " + str(path))
    frame = pd.read_csv(
        str(path), sep="\t", compression="gzip", skiprows=begin_line + 1,
        comment="!", dtype=str, keep_default_na=False, low_memory=False,
    )
    required = {"ID", "Gene symbol"}
    if not required.issubset(set(frame.columns)):
        raise RuntimeError("Annotation lacks ID/Gene symbol: " + str(path))
    return frame


def deposited_concordance_table(
    current: Any,
    deposited: Any,
    common_samples: Sequence[str],
    comparison_method: str,
    identity_hypothesis: str,
    pd: Any,
    np: Any,
) -> Any:
    common_features = current.index.intersection(deposited.index)
    if len(common_features) < 1000:
        raise RuntimeError(
            "Too few common features for deposited comparison: {}".format(len(common_features))
        )
    left_matrix = current.loc[common_features, list(common_samples)]
    right_matrix = deposited.loc[common_features, list(common_samples)]
    rows = []  # type: List[Dict[str, Any]]
    for sample in common_samples:
        left_series = left_matrix[sample]
        right_series = right_matrix[sample]
        correlation = float(np.corrcoef(
            left_series.to_numpy(dtype=float), right_series.to_numpy(dtype=float)
        )[0, 1])
        # Ranking first and then using Pearson is Spearman correlation without
        # adding a scipy dependency (important for the Python 3.7 server).
        rank_correlation = float(left_series.rank(method="average").corr(
            right_series.rank(method="average"), method="pearson"
        ))
        rows.append({
            "sample_id": sample,
            "comparison_method": comparison_method,
            "column_identity_hypothesis": identity_hypothesis,
            "common_features": int(len(common_features)),
            "pearson_r": correlation,
            "rank_correlation": rank_correlation,
        })
    return pd.DataFrame(rows)


def concordance_hypothesis_row(name: str, table: Any, selected: bool) -> Dict[str, Any]:
    return {
        "hypothesis": name,
        "median_pearson_r": float(table["pearson_r"].median()),
        "minimum_pearson_r": float(table["pearson_r"].min()),
        "median_rank_correlation": float(table["rank_correlation"].median()),
        "minimum_rank_correlation": float(table["rank_correlation"].min()),
        "selected": "yes" if selected else "no",
    }


def compare_with_deposited(
    dataset: str,
    normalized_filtered: Any,
    sample_map: Any,
    series_path: Path,
    annotation_path: Path,
    pd: Any,
    np: Any,
) -> Tuple[Any, Dict[str, Any], Any, Any, Any]:
    deposited_direct = read_series_matrix(series_path, pd)
    common_samples = [
        sample for sample in normalized_filtered.columns if sample in deposited_direct.columns
    ]
    if len(common_samples) != normalized_filtered.shape[1]:
        raise RuntimeError("Deposited series matrix lacks one or more discovery samples")

    if dataset == "GSE71356":
        current = normalized_filtered.copy()
        comparison_method = "probe_id"
    elif dataset == "GSE75097":
        annotation = read_annotation(annotation_path, pd)
        mapping = annotation[["ID", "Gene symbol"]].copy()
        mapping["Gene symbol"] = mapping["Gene symbol"].astype(str).str.strip()
        mapping = mapping[
            (mapping["Gene symbol"] != "")
            & (~mapping["Gene symbol"].str.contains("///", regex=False))
        ]
        mapping = mapping.drop_duplicates(subset=["ID"], keep="first")
        shared_probes = normalized_filtered.index.intersection(mapping["ID"])
        mapped = normalized_filtered.loc[shared_probes].copy()
        symbol_map = mapping.set_index("ID").loc[shared_probes, "Gene symbol"]
        mapped.insert(0, "__gene_symbol", symbol_map.to_numpy())
        current = mapped.groupby("__gene_symbol", sort=False).median()
        comparison_method = "GPL10558_gene_symbol_median_for_QC_only"
    else:
        raise RuntimeError("Unsupported deposited comparison dataset")

    direct_table = deposited_concordance_table(
        current, deposited_direct, common_samples, comparison_method,
        "geo_header_direct", pd, np,
    )
    audit_rows = []  # type: List[Dict[str, Any]]
    selected_table = direct_table
    selected_hypothesis = "geo_header_direct"
    correction_applied = False
    hypothesis_rows = []  # type: List[Dict[str, Any]]

    sample_lookup = sample_map.set_index("sample_id")
    if dataset == "GSE75097":
        # The deposited GSE75097 payload follows lexicographically sorted
        # Illumina array IDs, while its table header follows GEO submission/GSM
        # order.  Evaluate both hypotheses and only accept the correction when
        # the full per-sample concordance distribution provides strong support.
        sorted_map = sample_map.sort_values(
            by="raw_array_label", ascending=True, kind="mergesort"
        )
        payload_samples = sorted_map["sample_id"].tolist()
        payload_labels = sorted_map["raw_array_label"].tolist()
        header_samples = list(deposited_direct.columns)
        if len(payload_samples) != len(header_samples):
            raise RuntimeError("GSE75097 column-identity audit sample-count mismatch")
        corrected = deposited_direct.copy()
        corrected.columns = payload_samples
        if bool(corrected.columns.duplicated().any()):
            raise RuntimeError("Corrected GSE75097 deposited columns are not unique")
        corrected = corrected.loc[:, common_samples]
        corrected_table = deposited_concordance_table(
            current, corrected, common_samples, comparison_method,
            "sorted_array_id_payload", pd, np,
        )
        direct_median = float(direct_table["rank_correlation"].median())
        corrected_median = float(corrected_table["rank_correlation"].median())
        corrected_minimum = float(corrected_table["rank_correlation"].min())
        improvement = corrected_median - direct_median
        correction_supported = (
            improvement >= 0.02
            and corrected_median >= 0.95
            and corrected_minimum >= 0.90
        )
        if correction_supported:
            selected_table = corrected_table
            selected_hypothesis = "sorted_array_id_payload"
            correction_applied = True
        elif corrected_median > direct_median:
            raise RuntimeError(
                "GSE75097 deposited column identity is ambiguous: corrected median rank "
                "{:.4f}, direct {:.4f}, corrected minimum {:.4f}".format(
                    corrected_median, direct_median, corrected_minimum
                )
            )
        hypothesis_rows.append(concordance_hypothesis_row(
            "geo_header_direct", direct_table, not correction_applied
        ))
        hypothesis_rows.append(concordance_hypothesis_row(
            "sorted_array_id_payload", corrected_table, correction_applied
        ))
        hypothesis_rows[1]["median_rank_improvement_over_direct"] = improvement
        direct_by_sample = direct_table.set_index("sample_id")
        corrected_by_sample = corrected_table.set_index("sample_id")
        for position, (header_sample, payload_sample, payload_label) in enumerate(
            zip(header_samples, payload_samples, payload_labels), start=1
        ):
            audit_rows.append({
                "series_column_position": position,
                "geo_header_sample_id": header_sample,
                "geo_header_raw_array_label": sample_lookup.loc[
                    header_sample, "raw_array_label"
                ],
                "inferred_payload_sample_id": payload_sample,
                "inferred_payload_raw_array_label": payload_label,
                "identity_match": "yes" if header_sample == payload_sample else "no",
                "selected_column_name": (
                    payload_sample if correction_applied else header_sample
                ),
                "correction_applied": "yes" if correction_applied else "no",
                "direct_rank_correlation_for_header_sample": float(
                    direct_by_sample.loc[header_sample, "rank_correlation"]
                ),
                "corrected_rank_correlation_for_payload_sample": float(
                    corrected_by_sample.loc[payload_sample, "rank_correlation"]
                ),
            })
    else:
        hypothesis_rows.append(concordance_hypothesis_row(
            "geo_header_direct", direct_table, True
        ))
        for position, sample in enumerate(list(deposited_direct.columns), start=1):
            audit_rows.append({
                "series_column_position": position,
                "geo_header_sample_id": sample,
                "geo_header_raw_array_label": sample_lookup.loc[sample, "raw_array_label"],
                "inferred_payload_sample_id": sample,
                "inferred_payload_raw_array_label": sample_lookup.loc[
                    sample, "raw_array_label"
                ],
                "identity_match": "yes",
                "selected_column_name": sample,
                "correction_applied": "no",
                "direct_rank_correlation_for_header_sample": float(
                    direct_table.set_index("sample_id").loc[sample, "rank_correlation"]
                ),
                "corrected_rank_correlation_for_payload_sample": float("nan"),
            })

    hypothesis_table = pd.DataFrame(hypothesis_rows)
    column_audit = pd.DataFrame(audit_rows)
    median_pearson = float(selected_table["pearson_r"].median())
    median_rank = float(selected_table["rank_correlation"].median())
    primary_metric = "rank_correlation" if dataset == "GSE75097" else "pearson_r"
    primary_median = median_rank if primary_metric == "rank_correlation" else median_pearson
    status = "PASS" if primary_median >= 0.85 else "WARN_LOW_CONCORDANCE"
    summary = {
        "comparison_method": comparison_method,
        "common_features": int(selected_table["common_features"].iloc[0]),
        "selected_column_identity_hypothesis": selected_hypothesis,
        "deposited_column_correction_applied": correction_applied,
        "column_identity_mismatches": int((column_audit["identity_match"] == "no").sum()),
        "direct_header_median_rank_correlation": float(
            direct_table["rank_correlation"].median()
        ),
        "direct_header_minimum_rank_correlation": float(
            direct_table["rank_correlation"].min()
        ),
        "median_sample_pearson_r": median_pearson,
        "minimum_sample_pearson_r": float(selected_table["pearson_r"].min()),
        "median_sample_rank_correlation": median_rank,
        "minimum_sample_rank_correlation": float(selected_table["rank_correlation"].min()),
        "primary_concordance_metric": primary_metric,
        "primary_median_correlation": primary_median,
        "status_threshold": 0.85,
        "status": status,
        "interpretation": "Sensitivity/QC only; deposited processing is not mixed with primary matrix",
    }
    return selected_table, summary, column_audit, hypothesis_table, direct_table


def build_sample_freeze_manifest(
    dataset: str,
    sample_map: Any,
    sample_qc: Any,
    policy: Dict[str, Any],
    pd: Any,
) -> Tuple[Any, Dict[str, Any]]:
    dataset_policy = policy["datasets"][dataset]
    analysis_sets = dataset_policy["analysis_sets"]
    primary_set = str(dataset_policy["primary_set"])
    known_samples = set(sample_map["sample_id"])
    known_subjects = set(sample_map["subject_id"])
    base_columns = [
        column for column in [
            "sample_id", "raw_array_label", "subject_id", "study_group",
            "condition", "timepoint",
        ] if column in sample_map.columns
    ]
    table = sample_map[base_columns].copy()
    set_counts = {}  # type: Dict[str, Any]
    for set_name, set_policy in analysis_sets.items():
        excluded_samples = set(str(value) for value in set_policy.get("exclude_sample_ids", []))
        excluded_subjects = set(str(value) for value in set_policy.get("exclude_subject_ids", []))
        unknown_samples = sorted(excluded_samples - known_samples)
        unknown_subjects = sorted(excluded_subjects - known_subjects)
        if unknown_samples or unknown_subjects:
            raise RuntimeError(
                "Sample-freeze {} {} has unknown sample/subject IDs: {}; {}".format(
                    dataset, set_name, unknown_samples, unknown_subjects
                )
            )
        column = "include_{}".format(set_name)
        table[column] = [
            "no" if (
                str(row["sample_id"]) in excluded_samples
                or str(row["subject_id"]) in excluded_subjects
            ) else "yes"
            for _, row in table.iterrows()
        ]
        set_counts[set_name] = {
            "included": int((table[column] == "yes").sum()),
            "excluded": int((table[column] == "no").sum()),
            "description": str(set_policy.get("description", "")),
        }

    primary_policy = analysis_sets[primary_set]
    primary_column = "include_{}".format(primary_set)
    reasons = primary_policy.get("exclusion_reasons", {})
    primary_reasons = []  # type: List[str]
    for _, row in table.iterrows():
        if row[primary_column] == "yes":
            primary_reasons.append("Retained by locked primary sample-freeze policy.")
        else:
            sample = str(row["sample_id"])
            subject = str(row["subject_id"])
            primary_reasons.append(str(
                reasons.get(sample, reasons.get(subject, "Excluded by locked technical-QC policy."))
            ))
    table.insert(
        len(base_columns), "primary_analysis_include", table[primary_column].tolist()
    )
    table.insert(len(base_columns) + 1, "primary_freeze_reason", primary_reasons)

    primary_excluded = set(
        table.loc[table["primary_analysis_include"] == "no", "sample_id"].astype(str)
    )
    qc_review = set(
        sample_qc.loc[sample_qc["qc_status"] == "REVIEW", "sample_id"].astype(str)
    )
    if not primary_excluded.issubset(qc_review):
        raise RuntimeError(
            "Primary sample exclusions lack a technical QC REVIEW flag: "
            + ", ".join(sorted(primary_excluded - qc_review))
        )
    summary = {
        "primary_set": primary_set,
        "primary_included_samples": int((table["primary_analysis_include"] == "yes").sum()),
        "primary_excluded_samples": sorted(primary_excluded),
        "analysis_sets": set_counts,
        "single_assay_exclusion_in_paired_GSE71356": False,
    }
    return table, summary


def gse71356_detection_sensitivity(
    normalized: Any,
    detection: Any,
    sample_map: Any,
    detection_p: float,
    cutoffs: Sequence[int],
    primary_cutoff: int,
    top_variable: int,
    np: Any,
    pd: Any,
) -> Tuple[Any, Any]:
    detected_count = (detection < detection_p).sum(axis=1).astype(int)
    membership = pd.DataFrame({
        "probe_id": normalized.index.astype(str),
        "detected_count": detected_count.to_numpy(dtype=int),
    })
    rows = []  # type: List[Dict[str, Any]]
    for cutoff in cutoffs:
        cutoff_value = int(cutoff)
        keep = detected_count >= cutoff_value
        membership["keep_detected_ge_{}".format(cutoff_value)] = np.where(
            keep.to_numpy(dtype=bool), "yes", "no"
        )
        retained = normalized.loc[keep].copy()
        if retained.shape[0] < 2:
            raise RuntimeError(
                "GSE71356 sensitivity cutoff {} retained fewer than two probes".format(
                    cutoff_value
                )
            )
        scores, explained = pca_scores(retained, sample_map, top_variable, np, pd)
        empty_qc = pd.DataFrame({
            "sample_id": list(normalized.columns),
            "technical_flags": [""] * normalized.shape[1],
            "qc_status": ["PASS"] * normalized.shape[1],
            "automatic_exclusion": ["no"] * normalized.shape[1],
        })
        flagged = add_pca_review_flags(empty_qc, scores, np, pd)
        flagged_samples = flagged.loc[
            flagged["qc_status"] == "REVIEW", "sample_id"
        ].astype(str).tolist()
        rows.append({
            "detection_p_threshold": float(detection_p),
            "minimum_detected_samples": cutoff_value,
            "threshold_role": "locked_primary" if cutoff_value == primary_cutoff else "sensitivity",
            "retained_probes": int(retained.shape[0]),
            "retained_fraction": float(retained.shape[0] / normalized.shape[0]),
            "pca_probe_count": int(min(top_variable, retained.shape[0])),
            "pca_pc1_explained": float(explained[0]) if len(explained) > 0 else float("nan"),
            "pca_pc2_explained": float(explained[1]) if len(explained) > 1 else float("nan"),
            "pca_pc3_explained": float(explained[2]) if len(explained) > 2 else float("nan"),
            "pca_review_sample_count": len(flagged_samples),
            "pca_review_samples": ";".join(flagged_samples),
        })
    return pd.DataFrame(rows), membership


def write_tsv(frame: Any, path: Path, index: bool = False, index_label: Optional[str] = None) -> None:
    compression = "gzip" if path.name.endswith(".gz") else None
    kwargs = {
        "sep": "\t",
        "index": index,
        "index_label": index_label,
        "compression": compression,
        "float_format": "%.8g",
    }
    # pandas <=1.5 used ``line_terminator`` while newer pandas uses
    # ``lineterminator``.  Detect the installed API so the same script runs on
    # the server's Python 3.7 stack and on modern local validation runtimes.
    parameters = inspect.signature(frame.to_csv).parameters
    if "lineterminator" in parameters:
        kwargs["lineterminator"] = "\n"
    elif "line_terminator" in parameters:
        kwargs["line_terminator"] = "\n"
    frame.to_csv(str(path), **kwargs)


def plot_distributions(log2_matrix: Any, normalized: Any, path: Path, plt: Any) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(max(10, log2_matrix.shape[1] * 0.22), 8), sharex=True)
    axes[0].boxplot([log2_matrix[column].to_numpy() for column in log2_matrix.columns], showfliers=False)
    axes[0].set_ylabel("log2 intensity")
    axes[0].set_title("Before quantile normalization")
    axes[1].boxplot([normalized[column].to_numpy() for column in normalized.columns], showfliers=False)
    axes[1].set_ylabel("normalized log2 intensity")
    axes[1].set_title("After quantile normalization")
    axes[1].set_xticks(range(1, len(normalized.columns) + 1))
    axes[1].set_xticklabels(list(normalized.columns), rotation=90, fontsize=6)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def plot_correlation(correlation: Any, path: Path, plt: Any, title: str) -> None:
    import numpy as np
    size = max(8, correlation.shape[0] * 0.22)
    values = correlation.to_numpy(dtype=float)
    off_diagonal = values[~np.eye(values.shape[0], dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    dynamic_min = float(np.percentile(finite, 2)) if finite.size else 0.0
    dynamic_min = max(-1.0, min(0.95, dynamic_min))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(values, vmin=dynamic_min, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(range(correlation.shape[1]))
    axis.set_yticks(range(correlation.shape[0]))
    axis.set_xticklabels(list(correlation.columns), rotation=90, fontsize=6)
    axis.set_yticklabels(list(correlation.index), fontsize=6)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def plot_pca(scores: Any, explained: Sequence[float], path: Path, plt: Any) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    color_field = "condition" if scores["study_group"].nunique() == 1 else "study_group"
    groups = list(dict.fromkeys(scores[color_field].tolist()))
    color_map = plt.get_cmap("tab10")
    for index, group in enumerate(groups):
        subset = scores[scores[color_field] == group]
        axis.scatter(subset["PC1"], subset["PC2"], s=45, alpha=0.85, label=group, color=color_map(index % 10))
    axis.axhline(0, color="#cccccc", linewidth=0.7)
    axis.axvline(0, color="#cccccc", linewidth=0.7)
    axis.set_xlabel("PC1 ({:.1f}%)".format(100.0 * explained[0] if explained else 0.0))
    axis.set_ylabel("PC2 ({:.1f}%)".format(100.0 * explained[1] if len(explained) > 1 else 0.0))
    axis.set_title("PCA on top variable filtered probes")
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def plot_sample_qc(sample_qc: Any, detection_field: str, path: Path, plt: Any) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(max(10, sample_qc.shape[0] * 0.24), 7), sharex=True)
    x = list(range(sample_qc.shape[0]))
    axes[0].bar(x, sample_qc[detection_field], color="#4c78a8")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_ylim(0, 1)
    axes[1].bar(x, sample_qc["median_inter_sample_correlation"], color="#59a14f")
    axes[1].set_ylabel("Median correlation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sample_qc["sample_id"], rotation=90, fontsize=6)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def output_inventory(directory: Path, root: Path) -> List[Dict[str, Any]]:
    rows = []  # type: List[Dict[str, Any]]
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "output_manifest.tsv":
            continue
        rows.append({
            "relative_path": relative_path(path, root),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def prepare_output_directory(root: Path, output: Path, force: bool) -> None:
    root_results = (root / "results").resolve()
    resolved = output.resolve()
    prefix = str(root_results) + os.sep
    if not str(resolved).startswith(prefix):
        raise RuntimeError("Output directory must remain under ROOT/results: " + str(resolved))
    if resolved.exists():
        if any(resolved.iterdir()):
            if not force:
                raise RuntimeError("Output exists and is not empty; use a new --run-id or --force: " + str(resolved))
            shutil.rmtree(str(resolved))
    resolved.mkdir(parents=True, exist_ok=True)


def process_dataset(
    dataset: str,
    root: Path,
    output_dir: Path,
    sample_manifest: Any,
    sample_freeze_policy: Dict[str, Any],
    detection_p_override: Optional[float],
    top_variable: int,
    np: Any,
    pd: Any,
    plt: Any,
) -> Dict[str, Any]:
    config = DATASETS[dataset]
    raw_path = root / config["raw"]
    series_path = root / config["series"]
    annotation_path = root / config["annotation"]
    for path in [raw_path, series_path, annotation_path]:
        if not path.is_file():
            raise RuntimeError("Missing input for {}: {}".format(dataset, path))

    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    intensity, detection, sample_map = read_raw_export(
        dataset, raw_path, series_path, sample_manifest, np, pd
    )
    log2_matrix, scale_summary = scale_and_log2(intensity, np, pd)
    normalized = quantile_normalize(log2_matrix, np, pd)
    detection_p = (
        float(detection_p_override)
        if detection_p_override is not None
        else float(config["detection_p"])
    )
    minimum_detected = int(config["minimum_detected_samples"])
    sample_qc, probe_qc, filtered, technical_correlation, filtered_correlation = sample_and_probe_qc(
        log2_matrix, normalized, detection, detection_p, minimum_detected, np, pd
    )
    sample_qc = sample_qc.merge(
        sample_map[["sample_id", "subject_id", "study_group", "condition", "timepoint"]],
        on="sample_id", how="left", validate="one_to_one",
    )
    scores, explained = pca_scores(filtered, sample_map, top_variable, np, pd)
    sample_qc = add_pca_review_flags(sample_qc, scores, np, pd)
    sample_freeze, sample_freeze_summary = build_sample_freeze_manifest(
        dataset, sample_map, sample_qc, sample_freeze_policy, pd
    )
    sample_qc = sample_qc.merge(
        sample_freeze[["sample_id", "primary_analysis_include", "primary_freeze_reason"]],
        on="sample_id", how="left", validate="one_to_one",
    )
    (
        concordance,
        concordance_summary,
        column_identity_audit,
        identity_hypotheses,
        direct_concordance,
    ) = compare_with_deposited(
        dataset, filtered, sample_map, series_path, annotation_path, pd, np
    )

    detection_sensitivity = None
    detection_membership = None
    if dataset == "GSE71356":
        sensitivity_cutoffs = config.get(
            "detection_sensitivity_minimum_samples", [minimum_detected]
        )
        detection_sensitivity, detection_membership = gse71356_detection_sensitivity(
            normalized=normalized,
            detection=detection,
            sample_map=sample_map,
            detection_p=detection_p,
            cutoffs=sensitivity_cutoffs,
            primary_cutoff=minimum_detected,
            top_variable=top_variable,
            np=np,
            pd=pd,
        )

    write_tsv(sample_map, dataset_dir / "sample_mapping.tsv")
    write_tsv(log2_matrix, dataset_dir / "log2_unnormalized_probe_matrix.tsv.gz", index=True, index_label="probe_id")
    write_tsv(detection, dataset_dir / "detection_p_matrix.tsv.gz", index=True, index_label="probe_id")
    write_tsv(normalized, dataset_dir / "quantile_normalized_probe_matrix.tsv.gz", index=True, index_label="probe_id")
    write_tsv(filtered, dataset_dir / "filtered_normalized_probe_matrix.tsv.gz", index=True, index_label="probe_id")
    write_tsv(sample_qc, dataset_dir / "sample_qc.tsv")
    write_tsv(sample_freeze, dataset_dir / "sample_freeze.tsv")
    write_tsv(probe_qc, dataset_dir / "probe_qc.tsv.gz")
    write_tsv(
        technical_correlation, dataset_dir / "sample_correlation.tsv",
        index=True, index_label="sample_id",
    )
    write_tsv(
        filtered_correlation, dataset_dir / "filtered_probe_sample_correlation.tsv",
        index=True, index_label="sample_id",
    )
    write_tsv(scores, dataset_dir / "pca_scores.tsv")
    write_tsv(concordance, dataset_dir / "series_matrix_concordance.tsv")
    write_tsv(
        direct_concordance,
        dataset_dir / "series_matrix_concordance_geo_header_direct.tsv",
    )
    write_tsv(
        column_identity_audit,
        dataset_dir / "series_matrix_column_identity_audit.tsv",
    )
    write_tsv(
        identity_hypotheses,
        dataset_dir / "series_matrix_identity_hypotheses.tsv",
    )
    if detection_sensitivity is not None and detection_membership is not None:
        write_tsv(
            detection_sensitivity,
            dataset_dir / "detection_threshold_sensitivity.tsv",
        )
        write_tsv(
            detection_membership,
            dataset_dir / "detection_threshold_probe_membership.tsv.gz",
        )

    plot_distributions(log2_matrix, normalized, dataset_dir / "distribution_before_after.png", plt)
    plot_correlation(
        technical_correlation,
        dataset_dir / "sample_correlation_heatmap.png",
        plt,
        "Technical Pearson correlation: all normalized input probes",
    )
    plot_correlation(
        filtered_correlation,
        dataset_dir / "filtered_probe_sample_correlation_heatmap.png",
        plt,
        "Secondary correlation: detection-filtered normalized probes",
    )
    plot_pca(scores, explained, dataset_dir / "pca_top_variable.png", plt)
    detection_field = "detection_rate_p_lt_{}".format(detection_p)
    plot_sample_qc(sample_qc, detection_field, dataset_dir / "sample_qc_metrics.png", plt)

    summary = {
        "dataset": dataset,
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "inputs": {
            "raw": {"path": config["raw"], "sha256": sha256_file(raw_path)},
            "series": {"path": config["series"], "sha256": sha256_file(series_path)},
            "annotation": {"path": config["annotation"], "sha256": sha256_file(annotation_path)},
        },
        "samples": int(intensity.shape[1]),
        "input_probes": int(intensity.shape[0]),
        "detection_p_threshold": float(detection_p),
        "minimum_detected_samples": minimum_detected,
        "retained_probes": int(filtered.shape[0]),
        "retained_fraction": float(filtered.shape[0] / intensity.shape[0]),
        "normalization": "outcome-blind log2 scale audit followed by independent dataset quantile normalization",
        "background_correction": "not re-estimated; supplied export lacks negative-control probe classes",
        "deposited_processing": config["deposited_processing"],
        "scale_audit": scale_summary,
        "pca_top_variable_probes": int(min(top_variable, filtered.shape[0])),
        "pca_explained_variance": explained,
        "sample_qc_status_counts": {
            str(key): int(value) for key, value in sample_qc["qc_status"].value_counts().to_dict().items()
        },
        "technical_correlation_probe_universe": "all_quantile_normalized_input_probes",
        "secondary_filtered_probe_correlation_reported": True,
        "automatic_sample_exclusion": False,
        "sample_freeze": sample_freeze_summary,
        "series_matrix_concordance": concordance_summary,
        "detection_threshold_sensitivity": (
            detection_sensitivity.to_dict(orient="records")
            if detection_sensitivity is not None else []
        ),
        "candidate_selection_allowed": True,
        "differential_expression_performed": False,
    }
    summary_path = dataset_dir / "normalization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    inventory = output_inventory(dataset_dir, root)
    write_tsv(pd.DataFrame(inventory), dataset_dir / "output_manifest.tsv")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Python-only preprocessing/QC for locked OSA discovery arrays (Python 3.7+)."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="OSA project root")
    parser.add_argument(
        "--analysis-lock", type=Path, default=None,
        help="analysis_lock.yaml path (default: ROOT/config/analysis_lock.yaml)",
    )
    parser.add_argument(
        "--sample-freeze-policy", type=Path, default=None,
        help=(
            "Discovery sample-freeze JSON path "
            "(default: ROOT/config/discovery_sample_freeze_policy.json)"
        ),
    )
    parser.add_argument(
        "--run-id", default="discovery_preprocess_v1",
        help="Stable results run identifier (default: discovery_preprocess_v1)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS),
        help="Discovery datasets to process",
    )
    parser.add_argument(
        "--detection-p", type=float, default=None,
        help="Optional override applied to both datasets; default uses locked dataset-specific thresholds",
    )
    parser.add_argument("--top-variable", type=int, default=5000, help="Top variable probes for PCA")
    parser.add_argument("--check-env", action="store_true", help="Check dependencies and input/lock presence only")
    parser.add_argument("--force", action="store_true", help="Replace this run-id output directory")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if sys.version_info < (3, 7):
        parser.error("Python 3.7 or newer is required")
    if args.detection_p is not None and not (0.0 < args.detection_p < 1.0):
        parser.error("--detection-p must be between 0 and 1")
    if args.top_variable < 100:
        parser.error("--top-variable must be at least 100")
    if not re.match(r"^[A-Za-z0-9._-]+$", args.run_id):
        parser.error("--run-id may contain only letters, numbers, dot, underscore and hyphen")

    np, pd, matplotlib = require_scientific_python()
    import matplotlib.pyplot as plt

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error("OSA root does not exist: " + str(root))
    lock_path = validate_analysis_lock(root, args.analysis_lock)
    freeze_policy_path, freeze_policy = load_sample_freeze_policy(
        root, args.sample_freeze_policy
    )
    sample_manifest = load_sample_manifest(root, pd)
    for dataset in args.datasets:
        for key in ["raw", "series", "annotation"]:
            path = root / DATASETS[dataset][key]
            if not path.is_file():
                raise RuntimeError("Missing {} input: {}".format(dataset, path))

    if args.check_env:
        print("OSA discovery Python environment: PASS")
        print("Python: {}".format(sys.version.replace("\n", " ")))
        print("numpy: {}".format(np.__version__))
        print("pandas: {}".format(pd.__version__))
        print("matplotlib: {}".format(matplotlib.__version__))
        print("analysis_lock: {}".format(lock_path))
        print("sample_freeze_policy: {}".format(freeze_policy_path))
        print("datasets: {}".format(", ".join(args.datasets)))
        return 0

    output_dir = root / "results" / args.run_id / "02_qc" / "discovery"
    prepare_output_directory(root, output_dir, args.force)
    summaries = []  # type: List[Dict[str, Any]]
    for dataset in args.datasets:
        print("[{}] preprocessing and technical QC".format(dataset), flush=True)
        summary = process_dataset(
            dataset=dataset,
            root=root,
            output_dir=output_dir,
            sample_manifest=sample_manifest,
            sample_freeze_policy=freeze_policy,
            detection_p_override=args.detection_p,
            top_variable=int(args.top_variable),
            np=np,
            pd=pd,
            plt=plt,
        )
        summaries.append(summary)
        print(
            "  PASS samples={} probes={} retained={} series_r_median={:.4f}".format(
                summary["samples"], summary["input_probes"], summary["retained_probes"],
                summary["series_matrix_concordance"]["primary_median_correlation"],
            ),
            flush=True,
        )

    run_summary = {
        "run_id": args.run_id,
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "root": str(root),
        "analysis_lock": {
            "path": relative_path(lock_path, root),
            "sha256": sha256_file(lock_path),
        },
        "sample_manifest": {
            "path": "metadata/sample_manifest.tsv",
            "sha256": sha256_file(root / "metadata" / "sample_manifest.tsv"),
        },
        "sample_freeze_policy": {
            "path": relative_path(freeze_policy_path, root),
            "sha256": sha256_file(freeze_policy_path),
        },
        "datasets": summaries,
        "cross_platform_merge_performed": False,
        "validation_data_read": False,
        "differential_expression_performed": False,
        "next_gate": (
            "Confirm corrected deposited-column identity, locked sample sets, and "
            "GSE71356 detection sensitivity before probe-gene-UniProt mapping"
        ),
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "platform": platform.platform(),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    run_inventory = output_inventory(output_dir, root)
    write_tsv(pd.DataFrame(run_inventory), output_dir / "output_manifest.tsv")
    print("OSA discovery preprocessing/QC: PASS")
    print("Output: {}".format(output_dir))
    print("No validation data or differential-expression outcome was read.")
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
