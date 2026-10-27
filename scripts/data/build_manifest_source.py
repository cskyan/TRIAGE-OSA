#!/usr/bin/env python3
'Build the locked OSA Phase-1 metadata manifests.\n\nThe script is intentionally dependency-free and compatible with Python 3.7+.\nIt reads the GEO series matrices, the fully validated GSE214865 archive metadata,\nthe PXD032734 local file inventory, and a successful audit JSON produced by\n``osa_geo_server_pipeline.py``.\n\nIt never edits raw data.  With ``--write`` it creates:\n\n* metadata/dataset_registry.tsv\n* metadata/sample_manifest.tsv\n* metadata/assay_manifest.tsv\n* metadata/file_manifest.tsv\n* metadata/contrast_registry.tsv\n* metadata/metadata_qc.tsv\n* metadata/metadata_codebook.tsv\n* metadata/manifest_summary.json\n* config/analysis_lock.yaml\n\nExamples\n--------\nServer run with automatic selection of the newest successful audit::\n\n    python scripts/data/build_manifest_source.py         --root .         --write\n\nUse a specific audit and overwrite a previous generated manifest::\n\n    python scripts/data/build_manifest_source.py         --root .         --audit-json audit/osa_geo_audit_20260714-004604.json         --write --force\n\nBackward compatibility: when no action flag is supplied, the source payload is\nprinted as escaped JSON for ``workbook_build/build_manifest.mjs``.\n'

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_VERSION = "2.0.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


DATASET_CONFIG = {
    "GSE75097": {
        "role": "discovery_OSA_state_severity",
        "selection": "yes",
        "expected_subjects": 48,
        "expected_assays": 48,
        "age_group": "adult",
        "tissue": "PBMC",
        "platform": "GPL10904",
        "analysis_annotation": "GPL10558_raw_probe_mapping; GPL10904_deposited_matrix_audit",
    },
    "GSE71356": {
        "role": "discovery_IH_anchor",
        "selection": "yes",
        "expected_subjects": 4,
        "expected_assays": 16,
        "age_group": "adult",
        "tissue": "PBMC",
        "platform": "GPL2700",
        "analysis_annotation": "GPL2700",
    },
    "GSE133601": {
        "role": "validation_CPAP_1",
        "selection": "no",
        "expected_subjects": 15,
        "expected_assays": 30,
        "age_group": "adult",
        "tissue": "PBMC",
        "platform": "GPL17244",
        "analysis_annotation": "GPL17244_Brainarray_v16_Entrez; exact_custom_CDF_required",
    },
    "GSE49800": {
        "role": "validation_CPAP_2",
        "selection": "no",
        "expected_subjects": 18,
        "expected_assays": 36,
        "age_group": "adult",
        "tissue": "PBL",
        "platform": "GPL6244",
        "analysis_annotation": "GPL6244",
    },
    "GSE135917": {
        "role": "cross_tissue_validation",
        "selection": "no",
        "expected_subjects": 42,
        "expected_assays": 66,
        "age_group": "adult",
        "tissue": "subcutaneous_adipose",
        "platform": "GPL6244",
        "analysis_annotation": "GPL6244",
    },
    "GSE214865": {
        "role": "external_single_cell",
        "selection": "no",
        "expected_subjects": 22,
        "expected_assays": 22,
        "age_group": "pediatric",
        "tissue": "PBMC",
        "platform": "GPL24676",
        "analysis_annotation": "10x_features.tsv_gene_ids",
    },
    "PXD032734": {
        "role": "external_proteome",
        "selection": "no",
        "expected_subjects": 12,
        "expected_assays": 10,
        "age_group": "pediatric",
        "tissue": "serum",
        "platform": "Q_Exactive_HF_X_TMTpro",
        "analysis_annotation": "reviewed_canonical_UniProt_mapping_after_search",
    },
}


SAMPLE_FIELDS = [
    "dataset", "sample_id", "subject_id", "paired_id", "study_group",
    "tissue", "age_group", "disease_group", "severity_group", "treatment",
    "condition", "exposure", "timepoint", "platform", "analysis_annotation",
    "batch", "sex", "age", "BMI", "AHI", "hypertension", "EDS",
    "study_role", "candidate_selection_allowed", "primary_include",
    "include", "local_available", "data_available", "qc_status",
    "exclusion_reason", "title", "source_file", "metadata_source", "notes",
]


DATASET_FIELDS = [
    "dataset", "study_role", "candidate_selection_allowed",
    "expected_subjects", "manifest_subjects", "expected_assays",
    "manifest_rows", "observed_assays", "locally_available_samples",
    "age_group", "tissue", "platform", "analysis_annotation",
    "data_integrity_status", "status", "analysis_gate", "notes",
]


ASSAY_FIELDS = [
    "dataset", "assay_id", "biological_sample_id", "subject_id",
    "assay_type", "platform", "data_file", "fraction", "TMT_channel",
    "local_available", "mapping_status",
]


FILE_FIELDS = [
    "dataset", "relative_path", "file_type", "size_bytes", "exists",
    "sha256", "checksum", "integrity_status", "audit_action",
    "audit_key", "notes",
]


CONTRAST_FIELDS = [
    "contrast_id", "dataset", "analysis_set", "design", "contrast",
    "estimand", "candidate_selection", "priority", "status", "notes",
]


QC_FIELDS = [
    "dataset", "check", "expected", "observed", "status", "required_action",
]


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


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def geo_fields(line: str) -> List[str]:
    return next(csv.reader([line], delimiter="\t", quotechar='"'))


def read_geo_metadata(path: Path) -> Dict[str, List[List[str]]]:
    rows = defaultdict(list)  # type: Dict[str, List[List[str]]]
    with gzip.open(str(path), "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                break
            if line.startswith("!"):
                parts = geo_fields(line)
                rows[parts[0]].append(parts[1:])
    return dict(rows)


def value_at(
    rows: Dict[str, List[List[str]]], key: str, idx: int, default: str = ""
) -> str:
    values = rows.get(key, [])
    if not values or idx >= len(values[0]):
        return default
    return values[0][idx]


def characteristics_at(
    rows: Dict[str, List[List[str]]], idx: int
) -> Dict[str, str]:
    out = {}  # type: Dict[str, str]
    for values in rows.get("!Sample_characteristics_ch1", []):
        if idx >= len(values):
            continue
        value = values[idx]
        if ":" in value:
            key, val = value.split(":", 1)
            out[key.strip().lower()] = val.strip()
    return out


def base_sample(
    dataset: str, sample_id: str, title: str, config: Dict[str, Any]
) -> Dict[str, str]:
    row = {field: "" for field in SAMPLE_FIELDS}
    row.update({
        "dataset": dataset,
        "sample_id": sample_id,
        "tissue": config["tissue"],
        "age_group": config["age_group"],
        "platform": config["platform"],
        "analysis_annotation": config["analysis_annotation"],
        "study_role": config["role"],
        "candidate_selection_allowed": config["selection"],
        "primary_include": "yes",
        "include": "yes",
        "local_available": "yes",
        "data_available": "yes",
        "qc_status": "not_run",
        "title": title,
        "metadata_source": "GEO_series_matrix",
    })
    return row


def transform_geo_sample(
    dataset: str,
    sample: Dict[str, str],
    characteristics: Dict[str, str],
    source: str,
) -> None:
    title = sample["title"]
    if dataset == "GSE75097":
        if "Primary snoring" in source:
            group = "PS"
        elif "(MSO)" in source:
            group = "MSO"
        elif "(VSO)" in source:
            group = "VSO"
        else:
            group = "VSOC"
        sample.update({
            "subject_id": sample["sample_id"],
            "study_group": group,
            "disease_group": "primary_snoring" if group == "PS" else "OSA",
            "severity_group": {
                "PS": "reference",
                "MSO": "moderate_to_severe",
                "VSO": "very_severe",
                "VSOC": "very_severe_treated",
            }[group],
            "treatment": "long_term_CPAP" if group == "VSOC" else "untreated",
            "condition": (
                "CPAP_associated_cross_sectional"
                if group == "VSOC"
                else ("control" if group == "PS" else "OSA")
            ),
            "timepoint": "cross_sectional",
            "age": characteristics.get("age", ""),
            "sex": characteristics.get("sex", ""),
            "AHI": characteristics.get("apnea hyponea index", ""),
            "hypertension": characteristics.get("hypertension", ""),
            "EDS": characteristics.get("eds", ""),
            "notes": "VSOC is unpaired and cannot be called CPAP reversal" if group == "VSOC" else "",
        })
    elif dataset == "GSE71356":
        subject = characteristics.get("individual", "")
        exposure_text = characteristics.get("exposure", "")
        exposure = "IH" if "IH" in exposure_text else "control_exposure"
        timepoint = "post_5h" if exposure_text.startswith("after") else "baseline_0h"
        sample.update({
            "subject_id": subject,
            "paired_id": subject,
            "study_group": "randomized_crossover",
            "disease_group": "healthy",
            "treatment": exposure,
            "condition": exposure,
            "exposure": exposure,
            "timepoint": timepoint,
            "notes": "Primary estimand is exposure-by-time interaction",
        })
    elif dataset == "GSE133601":
        subject = characteristics.get("subject", title.split()[0])
        timepoint_text = characteristics.get("timepoint", "")
        timepoint = "post" if timepoint_text.lower().startswith("post") else "pre"
        sample.update({
            "subject_id": subject,
            "paired_id": subject,
            "study_group": "HeartBEAT_CPAP",
            "disease_group": "OSA",
            "treatment": "post_CPAP" if timepoint == "post" else "pre_CPAP",
            "condition": "CPAP",
            "timepoint": timepoint,
            "notes": "Validation only; exact Brainarray v16 CDF required for raw-CEL RMA",
        })
    elif dataset == "GSE49800":
        subject = characteristics.get("subject", "")
        if not subject:
            match = re.search(r"(\d+)$", title)
            subject = match.group(1) if match else sample["sample_id"]
        is_post = "Post" in title or characteristics.get("treatment", "") == "CPAP"
        sample.update({
            "subject_id": subject,
            "paired_id": subject,
            "study_group": "paired_CPAP",
            "disease_group": "severe_OSA",
            "severity_group": "severe",
            "treatment": "post_CPAP" if is_post else "pre_CPAP",
            "condition": "CPAP",
            "timepoint": "post" if is_post else "pre",
        })
    elif dataset == "GSE135917":
        paired = "before CPAP" in title or "after CPAP" in title
        if paired:
            match = re.search(r"(\d+)[bc]$", title)
            subject = match.group(1) if match else sample["sample_id"]
            is_post = "after CPAP" in title
            sample.update({
                "subject_id": "SG2_" + subject,
                "paired_id": "SG2_" + subject,
                "study_group": "study_group_2_paired_CPAP",
                "disease_group": "OSA",
                "treatment": "post_CPAP" if is_post else "pre_CPAP",
                "condition": "CPAP",
                "timepoint": "post" if is_post else "pre",
            })
        else:
            is_control = title.startswith("Normal Subject")
            match = re.search(r"Subject\s+(\d+)", title)
            subject = match.group(1) if match else sample["sample_id"]
            sample.update({
                "subject_id": "SG1_" + subject,
                "study_group": "study_group_1_case_control",
                "disease_group": "control" if is_control else "OSA",
                "treatment": "none",
                "condition": "control" if is_control else "OSA",
                "timepoint": "cross_sectional",
            })
        sample.update({
            "age": characteristics.get("age", ""),
            "sex": characteristics.get("sex", ""),
            "BMI": characteristics.get("bmi", ""),
        })


def geo_samples(root: Path) -> List[Dict[str, str]]:
    samples = []  # type: List[Dict[str, str]]
    paths = sorted((root / "geo").glob("*series_matrix.txt.gz"))
    expected_datasets = {"GSE75097", "GSE71356", "GSE133601", "GSE49800", "GSE135917"}
    observed_datasets = {path.name.split("_")[0] for path in paths}
    missing = sorted(expected_datasets - observed_datasets)
    if missing:
        raise RuntimeError("Missing GEO series matrices: " + ", ".join(missing))

    for path in paths:
        dataset = path.name.split("_")[0]
        if dataset not in expected_datasets:
            continue
        config = DATASET_CONFIG[dataset]
        rows = read_geo_metadata(path)
        ids = rows.get("!Sample_geo_accession", [[]])[0]
        titles = rows.get("!Sample_title", [[]])[0]
        sources = rows.get("!Sample_source_name_ch1", [[]])[0]
        if not ids or len(ids) != len(titles) or len(ids) != len(sources):
            raise RuntimeError("Malformed series metadata in " + str(path))
        for idx, sample_id in enumerate(ids):
            sample = base_sample(dataset, sample_id, titles[idx], config)
            sample["source_file"] = relpath(path, root)
            transform_geo_sample(dataset, sample, characteristics_at(rows, idx), sources[idx])
            samples.append(sample)
    return samples


def read_audit(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or "summary" not in report or "results" not in report:
        raise RuntimeError("Audit JSON is not a server-pipeline report: " + str(path))
    summary = report.get("summary", {})
    results = report.get("results", [])
    if summary.get("selected") != 16 or summary.get("passed") != 16 or summary.get("failed") != 0:
        raise RuntimeError("Audit is not a complete 16/16 pass: " + str(path))
    failed = [item.get("key", "unknown") for item in results if item.get("status") != "pass"]
    if failed:
        raise RuntimeError("Audit contains failed artifacts: " + ", ".join(failed))
    return report


def find_successful_audit(root: Path, explicit: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve(), read_audit(path.resolve())

    candidates = sorted((root / "audit").glob("osa_geo_audit_*.json"), reverse=True)
    errors = []  # type: List[str]
    for path in candidates:
        try:
            return path.resolve(), read_audit(path.resolve())
        except Exception as exc:  # keep searching older successful audits
            errors.append("{}: {}".format(path.name, exc))
    detail = "; ".join(errors) if errors else "no osa_geo_audit_*.json files found"
    raise RuntimeError("No successful 16/16 audit found under audit/: " + detail)


def audit_result_by_key(audit: Dict[str, Any], key: str) -> Dict[str, Any]:
    for item in audit.get("results", []):
        if item.get("key") == key:
            return item
    raise RuntimeError("Audit is missing required result: " + key)


def scrna_samples(root: Path, audit: Dict[str, Any]) -> List[Dict[str, str]]:
    result = audit_result_by_key(audit, "gse214865_raw")
    validation = result.get("validation", {})
    sample_checks = validation.get("sample_checks", [])
    names = [str(item.get("sample", "")) for item in sample_checks if item.get("sample")]

    if len(names) != 22:
        archive_path = root / "geo" / "GSE214865_RAW.tar"
        names_set = set()  # type: Set[str]
        with tarfile.open(str(archive_path), "r:*", ignore_zeros=True) as archive:
            for member in archive:
                match = re.match(r"(GSM\d+_(?:Control|OSA)\d+)_", Path(member.name).name)
                if match:
                    names_set.add(match.group(1))
        names = sorted(names_set, key=lambda value: int(re.search(r"GSM(\d+)", value).group(1)))

    if len(names) != 22:
        raise RuntimeError("GSE214865 sample inventory is not 22 after successful audit")

    samples = []  # type: List[Dict[str, str]]
    config = DATASET_CONFIG["GSE214865"]
    for name in sorted(names, key=lambda value: int(re.search(r"GSM(\d+)", value).group(1))):
        match = re.match(r"(GSM\d+)_((?:Control|OSA)\d+)$", name)
        if not match:
            raise RuntimeError("Unexpected GSE214865 sample label: " + name)
        sample_id, label = match.groups()
        disease = "control" if label.startswith("Control") else "OSA"
        sample = base_sample("GSE214865", sample_id, label, config)
        sample.update({
            "subject_id": label,
            "study_group": "pediatric_case_control",
            "disease_group": disease,
            "treatment": "none",
            "condition": disease,
            "timepoint": "cross_sectional",
            "qc_status": "not_run",
            "source_file": "geo/GSE214865_RAW.tar",
            "metadata_source": "validated_10x_archive_and_GEO_label",
            "notes": "External participant-level pseudobulk only",
        })
        samples.append(sample)
    return samples


def pxd_raw_files(root: Path) -> List[Path]:
    files = list((root / "PXD032734").glob("F*.raw"))
    def fraction_number(path: Path) -> int:
        match = re.search(r"F(\d+)", path.name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 10 ** 9
    return sorted(files, key=fraction_number)


def assay_rows(root: Path, samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows = []  # type: List[Dict[str, Any]]
    matrix_datasets = {"GSE75097", "GSE71356", "GSE133601", "GSE49800", "GSE135917"}
    for sample in samples:
        dataset = sample["dataset"]
        if dataset in matrix_datasets:
            rows.append({
                "dataset": dataset,
                "assay_id": sample["sample_id"],
                "biological_sample_id": sample["sample_id"],
                "subject_id": sample["subject_id"],
                "assay_type": "expression_array",
                "platform": sample["platform"],
                "data_file": "geo/{}_series_matrix.txt.gz".format(dataset),
                "fraction": "",
                "TMT_channel": "",
                "local_available": "yes",
                "mapping_status": "linked",
            })
        elif dataset == "GSE214865":
            stem = "{}_{}".format(sample["sample_id"], sample["title"])
            rows.append({
                "dataset": dataset,
                "assay_id": sample["sample_id"] + "_10x",
                "biological_sample_id": sample["sample_id"],
                "subject_id": sample["subject_id"],
                "assay_type": "10x_3prime_scRNA",
                "platform": sample["platform"],
                "data_file": "geo/GSE214865_RAW.tar::{}_[barcodes|features|matrix].tsv.gz".format(stem),
                "fraction": "",
                "TMT_channel": "",
                "local_available": "yes",
                "mapping_status": "linked",
            })

    for raw in pxd_raw_files(root):
        match = re.search(r"F(\d+)", raw.name, flags=re.IGNORECASE)
        fraction = int(match.group(1)) if match else ""
        rows.append({
            "dataset": "PXD032734",
            "assay_id": "TMT_fraction_{:02d}".format(fraction) if fraction != "" else raw.stem,
            "biological_sample_id": "pooled_TMT_multiplex_subjects_unresolved",
            "subject_id": "",
            "assay_type": "LCMS_fraction",
            "platform": DATASET_CONFIG["PXD032734"]["platform"],
            "data_file": relpath(raw, root),
            "fraction": fraction,
            "TMT_channel": "unresolved",
            "local_available": "yes",
            "mapping_status": "BLOCKED_channel_to_subject",
        })
    return rows


def dataset_from_relative_path(relative_path: str) -> str:
    name = Path(relative_path).name
    if name.startswith("GSE"):
        return name.split("_")[0]
    return ""


def file_type_for_path(path_text: str) -> str:
    lower = path_text.lower()
    if "series_matrix" in lower:
        return "processed_expression_matrix"
    if "non-normalized" in lower:
        return "non_normalized_expression"
    if lower.endswith("_raw.tar"):
        return "raw_archive"
    if lower.endswith(".annot.gz"):
        return "platform_annotation"
    if lower.endswith("platform.soft.txt"):
        return "platform_SOFT"
    if lower.endswith(".raw"):
        return "LCMS_fraction_RAW"
    if lower.endswith(".msfview"):
        return "ProteomeDiscoverer_view_index"
    if lower.endswith(".listing"):
        return "repository_listing"
    return "other"


def file_rows(root: Path, audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []  # type: List[Dict[str, Any]]
    audited_paths = set()  # type: Set[str]
    for item in audit.get("results", []):
        relative = str(item.get("relative_path", "")).replace("\\", "/")
        audited_paths.add(relative)
        path = root / Path(relative)
        size = path.stat().st_size if path.exists() else ""
        expected_size = item.get("validation", {}).get("size_bytes", "")
        integrity = "PASS" if item.get("status") == "pass" and size == expected_size else "FAIL"
        dataset = str(item.get("dataset", ""))
        rows.append({
            "dataset": dataset,
            "relative_path": relative,
            "file_type": file_type_for_path(relative),
            "size_bytes": size,
            "exists": "yes" if path.exists() else "no",
            "sha256": item.get("sha256", ""),
            "checksum": item.get("sha256", ""),
            "integrity_status": integrity,
            "audit_action": item.get("action", ""),
            "audit_key": item.get("key", ""),
            "notes": "Fully validated by osa_geo_server_pipeline.py",
        })

    pxd_dir = root / "PXD032734"
    if pxd_dir.exists():
        for path in sorted(item for item in pxd_dir.rglob("*") if item.is_file()):
            relative = relpath(path, root)
            if relative in audited_paths:
                continue
            rows.append({
                "dataset": "PXD032734",
                "relative_path": relative,
                "file_type": file_type_for_path(relative),
                "size_bytes": path.stat().st_size,
                "exists": "yes",
                "sha256": "not_computed",
                "checksum": "not_computed",
                "integrity_status": "PRESENT_NOT_CONTENT_VALIDATED",
                "audit_action": "inventory_only",
                "audit_key": "",
                "notes": "TMT channel-to-subject metadata remains unresolved",
            })
    return rows


def contrasts() -> List[Dict[str, str]]:
    rows = [
        ("GSE75097_OSA_state", "GSE75097", "PS+MSO+VSO untreated", "~ group", "(MSO+VSO) - PS", "adult OSA state", "yes", "primary", "planned", "Age/sex-adjusted model is sensitivity only if metadata completeness permits"),
        ("GSE75097_severe", "GSE75097", "PS+VSO untreated", "~ group", "VSO - PS", "very severe OSA state", "yes", "primary", "planned", "Cross-sectional"),
        ("GSE75097_AHI", "GSE75097", "PS+MSO+VSO untreated", "~ AHI", "AHI continuous beta", "continuous severity", "yes", "primary", "planned", "Run only after units and missingness are audited"),
        ("GSE75097_ordinal", "GSE75097", "PS+MSO+VSO untreated", "~ ordinal_group", "PS=0, MSO=1, VSO=2", "ordinal severity trend", "yes", "sensitivity", "planned", "Predeclared ordinal sensitivity"),
        ("GSE75097_VSOC", "GSE75097", "VSO+VSOC", "~ treatment_group", "VSOC - VSO", "long-term CPAP-associated cross-sectional difference", "no", "secondary", "planned_no_selection", "Not paired; never call reversal"),
        ("GSE71356_IH_x_time", "GSE71356", "all crossover samples", "~ subject + exposure + time + exposure:time", "(IH_post-IH_pre)-(control_post-control_pre)", "intermittent hypoxia response", "yes", "primary", "planned", "Subject-blocked crossover design"),
        ("GSE133601_CPAP", "GSE133601", "15 complete pairs", "~ subject + timepoint", "post - pre", "CPAP paired response", "no", "validation_primary", "locked_blind", "Never train, tune, select candidates or thresholds"),
        ("GSE49800_CPAP", "GSE49800", "18 complete pairs", "~ subject + timepoint", "post - pre", "CPAP paired response", "no", "validation_primary", "locked_blind", "Never train, tune, select candidates or thresholds"),
        ("GSE135917_OSA", "GSE135917", "study group 1 only", "~ disease_group", "OSA - control", "adipose OSA state", "no", "cross_tissue", "locked_blind", "Case-control component analyzed separately"),
        ("GSE135917_CPAP", "GSE135917", "study group 2 only; 24 pairs", "~ subject + timepoint", "post - pre", "adipose CPAP response or persistence", "no", "cross_tissue", "locked_blind", "Paired component analyzed separately"),
        ("GSE214865_pseudobulk", "GSE214865", "participant x cell type pseudobulk", "~ disease_group", "OSA - control", "pediatric cell-type localization", "no", "external", "planned_after_model_freeze", "Participant is the statistical unit; covariate sensitivity only if complete"),
        ("PXD032734_severity", "PXD032734", "12 TMT subjects after verified mapping", "subject-level model with plex/batch", "severity groups vs non-OSA", "pediatric serum protein support", "no", "external", "BLOCKED_METADATA", "Do not run until reporter channel-to-subject map is verified"),
    ]
    return [dict(zip(CONTRAST_FIELDS, row)) for row in rows]


def pair_summary(samples: List[Dict[str, str]], dataset: str) -> Tuple[int, int]:
    grouped = defaultdict(list)  # type: Dict[str, List[str]]
    for sample in samples:
        if sample["dataset"] == dataset and sample["paired_id"]:
            grouped[sample["paired_id"]].append(sample["timepoint"])
    complete = sum(1 for values in grouped.values() if sorted(values) == ["post", "pre"])
    return len(grouped), complete


def qc_rows(
    samples: List[Dict[str, str]], assays: List[Dict[str, Any]], audit: Dict[str, Any]
) -> List[Dict[str, str]]:
    def count(dataset: str, **conditions: str) -> int:
        return sum(
            1 for sample in samples
            if sample["dataset"] == dataset
            and all(sample.get(key) == value for key, value in conditions.items())
        )

    gse133_pairs = pair_summary(samples, "GSE133601")[1]
    gse498_pairs = pair_summary(samples, "GSE49800")[1]
    gse135_pairs = pair_summary(samples, "GSE135917")[1]
    scrna = audit_result_by_key(audit, "gse214865_raw").get("validation", {})
    pxd_fractions = sum(1 for row in assays if row["dataset"] == "PXD032734")
    rows = [
        ("ALL_GEO", "server_integrity_audit", "16 pass / 0 fail", "{} pass / {} fail".format(audit["summary"]["passed"], audit["summary"]["failed"]), "PASS", "Freeze audit provenance"),
        ("GSE75097", "group_counts", "PS=6; MSO=16; VSO=12; VSOC=14", "PS={}; MSO={}; VSO={}; VSOC={}".format(count("GSE75097", study_group="PS"), count("GSE75097", study_group="MSO"), count("GSE75097", study_group="VSO"), count("GSE75097", study_group="VSOC")), "PASS", "VSOC remains unpaired and selection=no"),
        ("GSE71356", "crossover_structure", "4 subjects x 4 assays", "{} subjects x 4 assays".format(len({s["subject_id"] for s in samples if s["dataset"] == "GSE71356"})), "PASS_WITH_NOTE", "Verify exposure order if recoverable; use exposure-by-time interaction"),
        ("GSE133601", "paired_completeness", "15 complete pairs", "{} complete pairs".format(gse133_pairs), "PASS", "Keep validation blinded"),
        ("GSE133601", "Brainarray_CDF", "exact v16 Entrez custom CDF", "GPL17244 platform table present; executable CDF package not verified", "BLOCKED_CDF", "Install and record the exact v16 CDF before raw-CEL RMA"),
        ("GSE49800", "paired_completeness_and_raw", "18 complete pairs; 36 CEL", "{} complete pairs; 36 CEL audit passed".format(gse498_pairs), "PASS", "Keep validation blinded"),
        ("GSE135917", "split_design", "18 case-control + 48 paired assays; 24 pairs", "{} case-control + {} paired assays; {} complete pairs".format(count("GSE135917", study_group="study_group_1_case_control"), count("GSE135917", study_group="study_group_2_paired_CPAP"), gse135_pairs), "PASS", "Analyze the two study groups separately"),
        ("GSE214865", "archive_and_groups", "22 samples / 66 members / 11+11", "{} samples / {} members / control={} OSA={}".format(scrna.get("sample_count", ""), scrna.get("member_count", ""), scrna.get("group_counts", {}).get("control", ""), scrna.get("group_counts", {}).get("osa", "")), "PASS", "Proceed to participant-level scRNA QC after model freeze"),
        ("PXD032734", "TMT_mapping", "12 subjects mapped to reporter channels", "Channel-to-subject map unresolved", "BLOCKED_METADATA", "Recover SDRF or verified experimental design; do not infer groups"),
        ("PXD032734", "fraction_identity", "10 LC fractions", "{} LC fractions".format(pxd_fractions), "PASS", "F1-F10 are fractions, not subjects"),
    ]
    return [dict(zip(QC_FIELDS, row)) for row in rows]


def dataset_rows(
    samples: List[Dict[str, str]],
    assays: List[Dict[str, Any]],
    audit: Dict[str, Any],
) -> List[Dict[str, Any]]:
    notes = {
        "GSE75097": "PS=6, MSO=16, VSO=12, VSOC=14; VSOC is cross-sectional",
        "GSE71356": "16 assays from 4 participants; paired crossover IH/control",
        "GSE133601": "15 complete pre/post pairs; raw CEL complete; exact Brainarray v16 CDF gate remains",
        "GSE49800": "18 complete pre/post pairs; 36 raw CEL files passed audit",
        "GSE135917": "Study group 1: 8 control + 10 OSA; study group 2: 24 CPAP pairs",
        "GSE214865": "22 participants; control 11 and OSA 11; 66 10x members passed audit",
        "PXD032734": "Expected 12 TMT subjects; F1-F10 are LC fractions; channel map unresolved",
    }
    audited_datasets = set()
    for item in audit.get("results", []):
        for dataset in str(item.get("dataset", "")).split(";"):
            if dataset:
                audited_datasets.add(dataset)

    rows = []  # type: List[Dict[str, Any]]
    for dataset, config in DATASET_CONFIG.items():
        dataset_samples = [s for s in samples if s["dataset"] == dataset]
        subject_ids = {s["subject_id"] for s in dataset_samples if s["subject_id"]}
        dataset_assays = [a for a in assays if a["dataset"] == dataset]
        local_samples = sum(1 for s in dataset_samples if s["local_available"] == "yes")
        if dataset == "PXD032734":
            integrity = "PRESENT_NOT_CONTENT_VALIDATED"
            status = "BLOCKED_METADATA"
            gate = "TMT channel-to-subject/condition/plex/reference mapping"
        elif dataset == "GSE133601":
            integrity = "PASS" if dataset in audited_datasets else "FAIL"
            status = "READY_FOR_METADATA_QC_WITH_CDF_GATE"
            gate = "Exact Brainarray v16 Entrez custom CDF before raw-CEL RMA"
        else:
            integrity = "PASS" if dataset in audited_datasets else "FAIL"
            status = "READY_FOR_METADATA_QC"
            gate = "none"
        rows.append({
            "dataset": dataset,
            "study_role": config["role"],
            "candidate_selection_allowed": config["selection"],
            "expected_subjects": config["expected_subjects"],
            "manifest_subjects": len(subject_ids),
            "expected_assays": config["expected_assays"],
            "manifest_rows": len(dataset_samples),
            "observed_assays": len(dataset_assays),
            "locally_available_samples": local_samples,
            "age_group": config["age_group"],
            "tissue": config["tissue"],
            "platform": config["platform"],
            "analysis_annotation": config["analysis_annotation"],
            "data_integrity_status": integrity,
            "status": status,
            "analysis_gate": gate,
            "notes": notes[dataset],
        })
    return rows


def codebook() -> List[Dict[str, str]]:
    definitions = {
        "dataset": "GEO or ProteomeXchange accession",
        "sample_id": "Deposited assay/sample identifier; not automatically a biological subject",
        "subject_id": "Biological participant identifier",
        "paired_id": "Identifier linking repeated measurements",
        "study_role": "Locked discovery, validation, cross-tissue or external role",
        "candidate_selection_allowed": "Whether this row may influence candidate selection or model tuning",
        "primary_include": "Current eligibility before technical QC; not permission for candidate selection",
        "local_available": "Whether required files for this assay are locally present",
        "qc_status": "Technical QC status; not a disease-result filter",
        "analysis_annotation": "Platform or mapping resource to use in analysis",
        "contrast_id": "Stable identifier for a predeclared statistical contrast",
        "estimand": "Scientific quantity estimated by the contrast",
        "BLOCKED_CDF": "Raw-CEL reprocessing waits for the exact Brainarray CDF package",
        "BLOCKED_METADATA": "Analysis waits for verified experimental-design metadata",
    }
    return [{"field": key, "definition": value} for key, value in definitions.items()]


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    errors = []  # type: List[str]
    samples = payload["samples"]
    datasets = {row["dataset"]: row for row in payload["datasets"]}

    expected_rows = {
        "GSE75097": 48,
        "GSE71356": 16,
        "GSE133601": 30,
        "GSE49800": 36,
        "GSE135917": 66,
        "GSE214865": 22,
        "PXD032734": 0,
    }
    for dataset, expected in expected_rows.items():
        actual = sum(1 for row in samples if row["dataset"] == dataset)
        if actual != expected:
            errors.append("{} sample rows: expected {}, observed {}".format(dataset, expected, actual))

    allowed = sorted({row["dataset"] for row in samples if row["candidate_selection_allowed"] == "yes"})
    if allowed != ["GSE71356", "GSE75097"]:
        errors.append("Candidate-selection datasets are not exactly GSE71356 and GSE75097: " + str(allowed))

    for dataset in ["GSE133601", "GSE49800", "GSE135917", "GSE214865"]:
        if any(row["candidate_selection_allowed"] != "no" for row in samples if row["dataset"] == dataset):
            errors.append(dataset + " contains candidate-selection leakage")

    group_counts = Counter(
        row["study_group"] for row in samples if row["dataset"] == "GSE75097"
    )
    if group_counts != Counter({"PS": 6, "MSO": 16, "VSO": 12, "VSOC": 14}):
        errors.append("GSE75097 group counts are incorrect: " + str(dict(group_counts)))

    for dataset, expected_pairs in [("GSE133601", 15), ("GSE49800", 18), ("GSE135917", 24)]:
        _, complete = pair_summary(samples, dataset)
        if complete != expected_pairs:
            errors.append("{} complete pairs: expected {}, observed {}".format(dataset, expected_pairs, complete))

    sc_counts = Counter(
        row["disease_group"] for row in samples if row["dataset"] == "GSE214865"
    )
    if sc_counts != Counter({"control": 11, "OSA": 11}):
        errors.append("GSE214865 group counts are incorrect: " + str(dict(sc_counts)))

    for dataset, row in datasets.items():
        if dataset != "PXD032734" and row["data_integrity_status"] != "PASS":
            errors.append(dataset + " does not have PASS integrity")

    if any(row["integrity_status"] == "FAIL" for row in payload["files"]):
        errors.append("file_manifest contains integrity FAIL")
    return errors


def analysis_lock_yaml(
    root: Path,
    audit_path: Path,
    audit: Dict[str, Any],
    audit_sha256: str,
    created_at: str,
) -> str:
    # Forward slashes keep double-quoted YAML valid on Windows as well as Linux;
    # backslashes such as ``\U`` or non-ASCII path components can be parsed as
    # YAML escape sequences.
    yaml_root = str(root).replace("\\", "/").replace('"', '\\"')
    yaml_audit = relpath(audit_path, root).replace("\\", "/").replace('"', '\\"')
    lines = [
        'schema_version: "1.0"',
        'project: "OSA_TRIAGE_Fuse_L1_L2_L3"',
        'created_at_utc: "{}"'.format(created_at),
        'script_version: "{}"'.format(SCRIPT_VERSION),
        'minimum_python: "3.7"',
        'project_root: "{}"'.format(yaml_root),
        'source_audit:',
        '  path: "{}"'.format(yaml_audit),
        '  sha256: "{}"'.format(audit_sha256),
        '  selected: {}'.format(audit["summary"]["selected"]),
        '  passed: {}'.format(audit["summary"]["passed"]),
        '  failed: {}'.format(audit["summary"]["failed"]),
        'dataset_roles:',
        '  GSE75097: "discovery_OSA_state_severity"',
        '  GSE71356: "discovery_IH_anchor"',
        '  GSE133601: "validation_CPAP_1"',
        '  GSE49800: "validation_CPAP_2"',
        '  GSE135917: "cross_tissue_validation"',
        '  GSE214865: "external_single_cell"',
        '  PXD032734: "external_proteome"',
        'candidate_selection:',
        '  allowed_datasets: ["GSE75097", "GSE71356"]',
        '  forbidden_datasets: ["GSE133601", "GSE49800", "GSE135917", "GSE214865", "PXD032734"]',
        '  primary_protein_universe: 250',
        '  sensitivity_protein_universe: 500',
        'validation_blinding:',
        '  cpap_reversal_is_training_loss: false',
        '  validation_may_select_candidates: false',
        '  validation_may_tune_hyperparameters: false',
        '  validation_may_select_thresholds: false',
        'primary_endpoint:',
        '  name: "CPAP-reversal enrichment among frozen top-ranked protein pairs"',
        '  headline_top_fraction: 0.01',
        '  auec_top_fraction_range: [0.005, 0.05]',
        '  cohorts_separate_before_meta_analysis: true',
        'identifier_mapping:',
        '  final_pair_identifier: "reviewed canonical UniProt accession"',
        '  multi_gene_probe_primary: "exclude"',
        '  multiple_probes_per_gene_primary: "median_after_normalization"',
        '  isoforms_primary: "collapse_to_canonical"',
        'model_boundaries:',
        '  p_PPI_and_r_OSA_reported_separately: true',
        '  disease_context_may_modify_L1_or_L2: false',
        '  combined_score_name: "OSA interaction-priority score"',
        '  combined_score_is_physical_interaction_probability: false',
        'known_gates:',
        '  GSE133601: "exact Brainarray v16 Entrez custom CDF required before raw-CEL RMA"',
        '  PXD032734: "verified TMT channel-to-subject/condition/plex/reference mapping required"',
        'raw_data_immutable: true',
        'cross_platform_expression_merge: false',
        'scRNA_statistical_unit: "participant"',
        '',
    ]
    return "\n".join(lines)


def build_payload(root: Path, audit_path: Path, audit: Dict[str, Any]) -> Dict[str, Any]:
    samples = geo_samples(root) + scrna_samples(root, audit)
    assays = assay_rows(root, samples)
    files = file_rows(root, audit)
    datasets = dataset_rows(samples, assays, audit)
    payload = {
        "schema_version": "2.0",
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "root": str(root),
        "audit_path": str(audit_path),
        "datasets": datasets,
        "samples": samples,
        "assays": assays,
        "contrasts": contrasts(),
        "files": files,
        "qc_checks": qc_rows(samples, assays, audit),
        "codebook": codebook(),
    }
    errors = validate_payload(payload)
    if errors:
        raise RuntimeError("Manifest validation failed:\n- " + "\n- ".join(errors))
    return payload


def atomic_text_write(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError("Output exists; use --force to replace generated files: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(str(temp), str(path))


def tsv_text(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), delimiter="\t", extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue()


def write_outputs(
    root: Path,
    metadata_dir: Path,
    config_dir: Path,
    audit_path: Path,
    audit: Dict[str, Any],
    payload: Dict[str, Any],
    force: bool,
) -> List[Path]:
    audit_hash = sha256_file(audit_path)
    outputs = [
        (metadata_dir / "dataset_registry.tsv", tsv_text(payload["datasets"], DATASET_FIELDS)),
        (metadata_dir / "sample_manifest.tsv", tsv_text(payload["samples"], SAMPLE_FIELDS)),
        (metadata_dir / "assay_manifest.tsv", tsv_text(payload["assays"], ASSAY_FIELDS)),
        (metadata_dir / "file_manifest.tsv", tsv_text(payload["files"], FILE_FIELDS)),
        (metadata_dir / "contrast_registry.tsv", tsv_text(payload["contrasts"], CONTRAST_FIELDS)),
        (metadata_dir / "metadata_qc.tsv", tsv_text(payload["qc_checks"], QC_FIELDS)),
        (metadata_dir / "metadata_codebook.tsv", tsv_text(payload["codebook"], ["field", "definition"])),
    ]

    summary = {
        "schema_version": payload["schema_version"],
        "script_version": SCRIPT_VERSION,
        "created_at_utc": payload["created_at_utc"],
        "root": str(root),
        "audit_path": relpath(audit_path, root),
        "audit_sha256": audit_hash,
        "audit_summary": audit["summary"],
        "dataset_rows": len(payload["datasets"]),
        "sample_rows": len(payload["samples"]),
        "assay_rows": len(payload["assays"]),
        "file_rows": len(payload["files"]),
        "contrast_rows": len(payload["contrasts"]),
        "qc_rows": len(payload["qc_checks"]),
        "candidate_selection_datasets": ["GSE75097", "GSE71356"],
        "known_gates": {
            "GSE133601": "BLOCKED_CDF for raw-CEL RMA only",
            "PXD032734": "BLOCKED_METADATA for subject-level proteomics",
        },
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    outputs.append((metadata_dir / "manifest_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"))
    outputs.append((
        config_dir / "analysis_lock.yaml",
        analysis_lock_yaml(root, audit_path, audit, audit_hash, payload["created_at_utc"]),
    ))

    paths = []  # type: List[Path]
    for path, text in outputs:
        atomic_text_write(path, text, force=force)
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build locked OSA Phase-1 metadata manifests (Python 3.7+; standard library only)."
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="OSA project root (default: parent of this script directory)",
    )
    parser.add_argument(
        "--audit-json", default=None,
        help="Successful osa_geo_audit JSON; relative paths are resolved under --root",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="Output metadata directory (default: ROOT/metadata)",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Output config directory (default: ROOT/config)",
    )
    parser.add_argument("--write", action="store_true", help="Write TSV/JSON/YAML outputs")
    parser.add_argument("--force", action="store_true", help="Replace existing generated outputs")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate in memory without writing")
    parser.add_argument("--json-stdout", action="store_true", help="Print source payload JSON to stdout")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.force and not args.write:
        parser.error("--force requires --write")
    if args.write and args.json_stdout:
        parser.error("Choose --write or --json-stdout, not both")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error("OSA root does not exist: " + str(root))
    audit_path, audit = find_successful_audit(root, args.audit_json)
    payload = build_payload(root, audit_path, audit)

    metadata_dir = (args.metadata_dir or (root / "metadata")).expanduser()
    if not metadata_dir.is_absolute():
        metadata_dir = root / metadata_dir
    config_dir = (args.config_dir or (root / "config")).expanduser()
    if not config_dir.is_absolute():
        config_dir = root / config_dir

    if args.write:
        paths = write_outputs(
            root=root,
            metadata_dir=metadata_dir.resolve(),
            config_dir=config_dir.resolve(),
            audit_path=audit_path,
            audit=audit,
            payload=payload,
            force=args.force,
        )
        print("OSA Phase-1 manifest build: PASS")
        print("Audit: {} (16/16 pass)".format(audit_path))
        print("Rows: datasets={} samples={} assays={} files={} contrasts={} qc={}".format(
            len(payload["datasets"]), len(payload["samples"]), len(payload["assays"]),
            len(payload["files"]), len(payload["contrasts"]), len(payload["qc_checks"]),
        ))
        for path in paths:
            print("WROTE " + str(path))
        print("KNOWN GATE GSE133601: exact Brainarray v16 CDF before raw-CEL RMA")
        print("KNOWN GATE PXD032734: verified TMT channel-to-subject mapping")
    elif args.dry_run:
        print("OSA Phase-1 manifest dry-run: PASS")
        print("Audit: {} (16/16 pass)".format(audit_path))
        print("Rows: datasets={} samples={} assays={} files={} contrasts={} qc={}".format(
            len(payload["datasets"]), len(payload["samples"]), len(payload["assays"]),
            len(payload["files"]), len(payload["contrasts"]), len(payload["qc_checks"]),
        ))
    else:
        # ASCII-escaped JSON avoids Windows legacy-console encoding failures and
        # remains fully reversible by JSON.parse/ConvertFrom-Json.
        print(json.dumps(payload, ensure_ascii=True, indent=2))
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
