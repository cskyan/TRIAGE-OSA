#!/usr/bin/env python3
"""Freeze discovery probe -> HGNC gene -> reviewed canonical UniProt mapping.

Python 3.7+; requires numpy and pandas only.  This script never downloads a
database and never reads validation data or differential-expression results.
It consumes the frozen HGNC/UniProt snapshots declared in
config/identifier_mapping_lock.json and the completed discovery QC run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "GSE75097": {
        "platform": "GPL10558",
        "annotation": "annotations/GPL10558.annot.gz",
    },
    "GSE71356": {
        "platform": "GPL2700",
        "annotation": "annotations/GPL2700.annot.gz",
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


def require_python() -> Tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Missing numpy/pandas. For Python 3.7 use: "
            "python -m pip install numpy==1.21.6 pandas==1.3.5"
        ) from exc
    return np, pd


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot read JSON {}: {}".format(path, exc))


def json_default(value: Any) -> Any:
    """Convert NumPy/pandas scalar values for Python 3.7 json encoding."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError("Object of type {} is not JSON serializable".format(
        type(value).__name__
    ))


def verify_mapping_lock(root: Path, explicit: Optional[Path]) -> Tuple[Path, Dict[str, Any]]:
    path = explicit.expanduser() if explicit is not None else root / "config" / "identifier_mapping_lock.json"
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("Missing identifier mapping lock: " + str(path))
    lock = load_json(path)
    for source in ["hgnc", "uniprot"]:
        item = lock.get(source, {})
        source_path = root / str(item.get("path", ""))
        if not source_path.is_file():
            raise RuntimeError("Missing locked {} snapshot: {}".format(source, source_path))
        actual = sha256_file(source_path)
        if actual != item.get("sha256"):
            raise RuntimeError(
                "{} snapshot SHA-256 mismatch: expected {}, observed {}".format(
                    source, item.get("sha256"), actual
                )
            )
    return path, lock


def read_gpl_annotation(path: Path, pd: Any) -> Any:
    begin = None
    with gzip.open(str(path), "rt", encoding="utf-8-sig", errors="replace") as handle:
        for number, line in enumerate(handle):
            if line.strip() == "!platform_table_begin":
                begin = number
                break
    if begin is None:
        raise RuntimeError("GPL platform table marker not found: " + str(path))
    frame = pd.read_csv(
        str(path), sep="\t", compression="gzip", skiprows=begin + 1,
        comment="!", dtype=str, keep_default_na=False, low_memory=False,
    )
    required = {"ID", "Gene title", "Gene symbol", "Gene ID"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError("GPL annotation lacks columns: " + ", ".join(missing))
    if bool(frame["ID"].duplicated().any()):
        raise RuntimeError("GPL annotation contains duplicate feature IDs: " + str(path))
    return frame


def clean_token(value: Any) -> str:
    text = str(value).strip()
    return "" if text in {"", "---", "NA", "N/A", "nan", "None"} else text


def split_pipe(value: Any) -> List[str]:
    text = clean_token(value)
    return [item.strip() for item in text.split("|") if item.strip()] if text else []


def split_semicolon(value: Any) -> List[str]:
    text = clean_token(value)
    return [item.strip() for item in text.split(";") if item.strip()] if text else []


def add_unique(index: Dict[str, List[int]], key: str, row_number: int) -> None:
    if key:
        index.setdefault(key, []).append(row_number)


def build_hgnc_indices(hgnc: Any) -> Dict[str, Any]:
    by_entrez = {}  # type: Dict[str, int]
    by_symbol = {}  # type: Dict[str, int]
    alias = {}  # type: Dict[str, List[int]]
    previous = {}  # type: Dict[str, List[int]]
    for number, row in hgnc.iterrows():
        entrez = clean_token(row.get("entrez_id", ""))
        symbol = clean_token(row.get("symbol", ""))
        if entrez:
            if entrez in by_entrez:
                raise RuntimeError("HGNC snapshot has duplicate Entrez ID: " + entrez)
            by_entrez[entrez] = number
        if symbol:
            if symbol in by_symbol:
                raise RuntimeError("HGNC snapshot has duplicate approved symbol: " + symbol)
            by_symbol[symbol] = number
        for token in split_pipe(row.get("alias_symbol", "")):
            add_unique(alias, token, number)
        for token in split_pipe(row.get("prev_symbol", "")):
            add_unique(previous, token, number)
    return {
        "by_entrez": by_entrez,
        "by_symbol": by_symbol,
        "alias": alias,
        "previous": previous,
    }


def build_uniprot_indices(uniprot: Any) -> Dict[str, Any]:
    by_accession = {}  # type: Dict[str, int]
    by_geneid = defaultdict(set)  # type: Dict[str, Set[str]]
    for number, row in uniprot.iterrows():
        accession = clean_token(row.get("Entry", ""))
        if not accession or clean_token(row.get("Reviewed", "")) != "reviewed":
            continue
        if accession in by_accession:
            raise RuntimeError("UniProt snapshot has duplicate accession: " + accession)
        by_accession[accession] = number
        for gene_id in split_semicolon(row.get("GeneID", "")):
            by_geneid[gene_id].add(accession)
    return {"by_accession": by_accession, "by_geneid": by_geneid}


def resolve_hgnc(
    gene_id_original: str,
    symbol_original: str,
    hgnc_indices: Dict[str, Any],
) -> Tuple[Optional[int], str, str]:
    if gene_id_original and gene_id_original in hgnc_indices["by_entrez"]:
        return hgnc_indices["by_entrez"][gene_id_original], "entrez_exact", ""
    if symbol_original and symbol_original in hgnc_indices["by_symbol"]:
        return hgnc_indices["by_symbol"][symbol_original], "approved_symbol_exact", ""
    alias_rows = hgnc_indices["alias"].get(symbol_original, []) if symbol_original else []
    if len(alias_rows) == 1:
        return alias_rows[0], "alias_symbol_unique", ""
    if len(alias_rows) > 1:
        return None, "", "ambiguous_hgnc_alias"
    previous_rows = hgnc_indices["previous"].get(symbol_original, []) if symbol_original else []
    if len(previous_rows) == 1:
        return previous_rows[0], "previous_symbol_unique", ""
    if len(previous_rows) > 1:
        return None, "", "ambiguous_hgnc_previous_symbol"
    return None, "", "unmapped_hgnc"


def resolve_uniprot(
    hgnc_row: Any,
    uniprot: Any,
    uniprot_indices: Dict[str, Any],
) -> Dict[str, str]:
    accession_index = uniprot_indices["by_accession"]
    hgnc_candidates = set(
        item for item in split_pipe(hgnc_row.get("uniprot_ids", ""))
        if item in accession_index and "-" not in item
    )
    entrez = clean_token(hgnc_row.get("entrez_id", ""))
    entrez_candidates = set(uniprot_indices["by_geneid"].get(entrez, set()))
    if hgnc_candidates and entrez_candidates:
        intersection = hgnc_candidates.intersection(entrez_candidates)
        if not intersection:
            return {
                "mapping_status": "uniprot_crossref_conflict",
                "exclusion_reason": "HGNC and UniProt GeneID reviewed-accession sets do not intersect",
                "candidate_accessions": ";".join(sorted(hgnc_candidates.union(entrez_candidates))),
            }
        candidates = intersection
        source = "HGNC_uniprot_ids_and_UniProt_GeneID"
    elif hgnc_candidates:
        candidates = hgnc_candidates
        source = "HGNC_uniprot_ids"
    elif entrez_candidates:
        candidates = set(item for item in entrez_candidates if "-" not in item)
        source = "UniProt_GeneID"
    else:
        candidates = set()
        source = ""

    symbol = clean_token(hgnc_row.get("symbol", ""))
    resolution = ""
    if len(candidates) > 1:
        symbol_matches = []  # type: List[str]
        for accession in candidates:
            row = uniprot.loc[accession_index[accession]]
            if clean_token(row.get("Gene Names (primary)", "")) == symbol:
                symbol_matches.append(accession)
        if len(symbol_matches) == 1:
            candidates = {symbol_matches[0]}
            resolution = "resolved_by_unique_primary_gene_symbol"
        else:
            return {
                "mapping_status": "ambiguous_multi_uniprot",
                "exclusion_reason": "Multiple reviewed canonical UniProt accessions remain for one HGNC gene",
                "candidate_accessions": ";".join(sorted(candidates)),
            }
    if not candidates:
        return {
            "mapping_status": "no_reviewed_uniprot",
            "exclusion_reason": "No reviewed canonical human UniProt accession",
            "candidate_accessions": "",
        }
    accession = sorted(candidates)[0]
    row = uniprot.loc[accession_index[accession]]
    return {
        "mapping_status": "mapped_reviewed_canonical",
        "exclusion_reason": "",
        "candidate_accessions": accession,
        "UniProt_accession": accession,
        "UniProt_entry_name": clean_token(row.get("Entry Name", "")),
        "UniProt_reviewed": "yes",
        "canonical_flag": "yes",
        "UniProt_gene_primary": clean_token(row.get("Gene Names (primary)", "")),
        "UniProt_protein_name": clean_token(row.get("Protein names", "")),
        "UniProt_length": clean_token(row.get("Length", "")),
        "uniprot_mapping_source": source,
        "uniprot_resolution": resolution,
    }


def map_dataset_features(
    dataset: str,
    platform: str,
    annotation: Any,
    probe_qc: Any,
    sensitivity_membership: Optional[Any],
    hgnc: Any,
    hgnc_indices: Dict[str, Any],
    uniprot: Any,
    uniprot_indices: Dict[str, Any],
    mapping_version: str,
    pd: Any,
) -> Any:
    annotation = annotation.set_index("ID")
    if set(probe_qc["probe_id"]) - set(annotation.index):
        raise RuntimeError(dataset + " has probes absent from GPL annotation")
    sensitivity = None
    sensitivity_columns = []  # type: List[str]
    if sensitivity_membership is not None:
        sensitivity = sensitivity_membership.set_index("probe_id")
        sensitivity_columns = [
            column for column in sensitivity.columns if column.startswith("keep_detected_ge_")
        ]
    rows = []  # type: List[Dict[str, Any]]
    for _, qc in probe_qc.iterrows():
        feature = str(qc["probe_id"])
        ann = annotation.loc[feature]
        symbol_original = clean_token(ann.get("Gene symbol", ""))
        gene_id_original = clean_token(ann.get("Gene ID", ""))
        row = {
            "dataset": dataset,
            "platform": platform,
            "original_feature": feature,
            "original_annotation": "Gene title={}; Gene symbol={}; Gene ID={}".format(
                clean_token(ann.get("Gene title", "")), symbol_original, gene_id_original
            ),
            "gene_symbol_original": symbol_original,
            "Entrez_original": gene_id_original,
            "gene_symbol_current": "",
            "HGNC_ID": "",
            "Entrez": "",
            "Ensembl_gene": "",
            "UniProt_accession": "",
            "UniProt_entry_name": "",
            "UniProt_reviewed": "no",
            "canonical_flag": "no",
            "UniProt_gene_primary": "",
            "UniProt_protein_name": "",
            "UniProt_length": "",
            "candidate_UniProt_accessions": "",
            "mapping_source": "",
            "mapping_version": mapping_version,
            "mapping_status": "",
            "exclusion_reason": "",
            "retained_primary_probe": str(qc.get("keep_primary", "")),
            "detected_count": str(qc.get("detected_count", "")),
            "detected_fraction": str(qc.get("detected_fraction", "")),
        }
        for column in sensitivity_columns:
            row[column] = str(sensitivity.loc[feature, column])
        if "///" in symbol_original or "///" in gene_id_original:
            row["mapping_status"] = "ambiguous_multi_gene"
            row["exclusion_reason"] = "GPL probe maps explicitly to multiple genes"
            rows.append(row)
            continue
        if not symbol_original and not gene_id_original:
            row["mapping_status"] = "unmapped_no_gene_annotation"
            row["exclusion_reason"] = "GPL has no Gene symbol or Gene ID"
            rows.append(row)
            continue
        hgnc_number, method, failure = resolve_hgnc(
            gene_id_original, symbol_original, hgnc_indices
        )
        if hgnc_number is None:
            row["mapping_status"] = failure
            row["exclusion_reason"] = "No unique approved HGNC gene could be resolved"
            rows.append(row)
            continue
        hrow = hgnc.loc[hgnc_number]
        row.update({
            "gene_symbol_current": clean_token(hrow.get("symbol", "")),
            "HGNC_ID": clean_token(hrow.get("hgnc_id", "")),
            "Entrez": clean_token(hrow.get("entrez_id", "")),
            "Ensembl_gene": clean_token(hrow.get("ensembl_gene_id", "")),
            "mapping_source": method,
        })
        protein = resolve_uniprot(hrow, uniprot, uniprot_indices)
        row["mapping_status"] = protein.get("mapping_status", "")
        row["exclusion_reason"] = protein.get("exclusion_reason", "")
        row["candidate_UniProt_accessions"] = protein.get("candidate_accessions", "")
        for key in [
            "UniProt_accession", "UniProt_entry_name", "UniProt_reviewed",
            "canonical_flag", "UniProt_gene_primary", "UniProt_protein_name",
            "UniProt_length",
        ]:
            if key in protein:
                row[key] = protein[key]
        if protein.get("uniprot_mapping_source"):
            row["mapping_source"] += "+" + protein["uniprot_mapping_source"]
        if protein.get("uniprot_resolution"):
            row["mapping_source"] += "+" + protein["uniprot_resolution"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_tsv(frame: Any, path: Path, index: bool = False, index_label: Optional[str] = None) -> None:
    compression = "gzip" if path.name.endswith(".gz") else None
    kwargs = {
        "sep": "\t", "index": index, "index_label": index_label,
        # Preserve medians of already serialized 8-significant-digit probe
        # values without introducing an additional half-unit rounding step.
        "compression": compression, "float_format": "%.10g",
    }
    try:
        frame.to_csv(str(path), lineterminator="\n", **kwargs)
    except TypeError:
        frame.to_csv(str(path), line_terminator="\n", **kwargs)


def aggregate_dataset(
    dataset: str,
    mapping: Any,
    qc_dir: Path,
    output_dir: Path,
    pd: Any,
) -> Dict[str, Any]:
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filtered = pd.read_csv(
        str(qc_dir / dataset / "filtered_normalized_probe_matrix.tsv.gz"),
        sep="\t", compression="gzip", index_col=0,
    )
    freeze = pd.read_csv(
        str(qc_dir / dataset / "sample_freeze.tsv"), sep="\t",
        dtype=str, keep_default_na=False,
    )
    primary_samples = freeze.loc[
        freeze["primary_analysis_include"] == "yes", "sample_id"
    ].tolist()
    if list(filtered.columns) != freeze["sample_id"].tolist():
        raise RuntimeError(dataset + " expression/freeze sample order mismatch")

    mapped_gene = mapping[(mapping["HGNC_ID"] != "")].copy()
    feature_to_gene = mapped_gene.set_index("original_feature")["HGNC_ID"]
    features = [feature for feature in filtered.index if feature in feature_to_gene.index]
    expression = filtered.loc[features].copy()
    expression.insert(0, "HGNC_ID", feature_to_gene.loc[features].to_numpy())
    gene_expression = expression.groupby("HGNC_ID", sort=True).median()

    gene_rows = []  # type: List[Dict[str, Any]]
    for hgnc_id, group in mapped_gene.groupby("HGNC_ID", sort=True):
        retained_group = group[group["original_feature"].isin(features)]
        if retained_group.empty:
            continue
        accessions = sorted(set(
            value for value in retained_group["UniProt_accession"] if value
        ))
        statuses = sorted(set(retained_group["mapping_status"]))
        gene_rows.append({
            "dataset": dataset,
            "HGNC_ID": hgnc_id,
            "gene_symbol_current": retained_group["gene_symbol_current"].iloc[0],
            "Entrez": retained_group["Entrez"].iloc[0],
            "Ensembl_gene": retained_group["Ensembl_gene"].iloc[0],
            "platform_probe_count": int(group.shape[0]),
            "retained_probe_count": int(retained_group.shape[0]),
            "probe_aggregation": "median_after_normalization",
            "UniProt_accession": accessions[0] if len(accessions) == 1 else "",
            "protein_mapping_status": (
                "mapped_reviewed_canonical" if len(accessions) == 1
                else ("ambiguous_multi_uniprot" if len(accessions) > 1 else "no_reviewed_uniprot")
            ),
            "feature_mapping_statuses": ";".join(statuses),
        })
    gene_metadata = pd.DataFrame(gene_rows).sort_values("HGNC_ID").reset_index(drop=True)
    if list(gene_expression.index) != gene_metadata["HGNC_ID"].tolist():
        raise RuntimeError(dataset + " gene expression/metadata order mismatch")

    protein_ready = gene_metadata[
        gene_metadata["protein_mapping_status"] == "mapped_reviewed_canonical"
    ].copy()
    accession_counts = protein_ready["UniProt_accession"].value_counts()
    ambiguous_accessions = set(accession_counts[accession_counts > 1].index)
    conflict_rows = []  # type: List[Dict[str, Any]]
    for accession in sorted(ambiguous_accessions):
        conflict = protein_ready[protein_ready["UniProt_accession"] == accession]
        conflict_rows.append({
            "dataset": dataset,
            "UniProt_accession": accession,
            "HGNC_gene_count": int(conflict.shape[0]),
            "HGNC_IDs": ";".join(conflict["HGNC_ID"].astype(str).tolist()),
            "gene_symbols": ";".join(
                conflict["gene_symbol_current"].astype(str).tolist()
            ),
            "mapping_status": "ambiguous_multi_gene_per_uniprot",
            "exclusion_reason": (
                "One reviewed canonical UniProt accession is linked to multiple "
                "HGNC genes in the retained platform mapping"
            ),
        })
    if ambiguous_accessions:
        gene_metadata.loc[
            gene_metadata["UniProt_accession"].isin(ambiguous_accessions),
            "protein_mapping_status",
        ] = "ambiguous_multi_gene_per_uniprot"
        protein_ready = protein_ready[
            ~protein_ready["UniProt_accession"].isin(ambiguous_accessions)
        ].copy()
    protein_ready = protein_ready.sort_values("UniProt_accession").reset_index(drop=True)
    protein_expression = gene_expression.loc[protein_ready["HGNC_ID"]].copy()
    protein_expression.index = protein_ready["UniProt_accession"].tolist()
    protein_expression.index.name = "UniProt_accession"
    protein_metadata = protein_ready[[
        "dataset", "UniProt_accession", "HGNC_ID", "gene_symbol_current",
        "Entrez", "Ensembl_gene", "retained_probe_count", "probe_aggregation",
    ]].copy()
    protein_metadata["UniProt_reviewed"] = "yes"
    protein_metadata["canonical_flag"] = "yes"

    write_tsv(gene_metadata, dataset_dir / "gene_aggregation.tsv")
    write_tsv(
        pd.DataFrame(
            conflict_rows,
            columns=[
                "dataset", "UniProt_accession", "HGNC_gene_count", "HGNC_IDs",
                "gene_symbols", "mapping_status", "exclusion_reason",
            ],
        ),
        dataset_dir / "protein_multi_gene_conflicts.tsv",
    )
    write_tsv(
        gene_expression, dataset_dir / "gene_expression_all_samples.tsv.gz",
        index=True, index_label="HGNC_ID",
    )
    write_tsv(
        gene_expression.loc[:, primary_samples],
        dataset_dir / "gene_expression_primary_samples.tsv.gz",
        index=True, index_label="HGNC_ID",
    )
    write_tsv(protein_metadata, dataset_dir / "protein_aggregation.tsv")
    write_tsv(
        protein_expression, dataset_dir / "protein_expression_all_samples.tsv.gz",
        index=True, index_label="UniProt_accession",
    )
    write_tsv(
        protein_expression.loc[:, primary_samples],
        dataset_dir / "protein_expression_primary_samples.tsv.gz",
        index=True, index_label="UniProt_accession",
    )
    return {
        "dataset": dataset,
        "all_samples": int(filtered.shape[1]),
        "primary_samples": len(primary_samples),
        "filtered_probes": int(filtered.shape[0]),
        "aggregated_hgnc_genes": int(gene_expression.shape[0]),
        "reviewed_canonical_proteins": int(protein_expression.shape[0]),
        "excluded_multi_gene_per_uniprot": len(ambiguous_accessions),
    }


def prepare_output(root: Path, output_dir: Path, metadata_map: Path, force: bool) -> None:
    result_root = (root / "results").resolve()
    resolved = output_dir.resolve()
    if not str(resolved).startswith(str(result_root) + os.sep):
        raise RuntimeError("Mapping output must remain under ROOT/results")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise RuntimeError("Mapping output exists; use --force or a new preprocessing run")
        shutil.rmtree(str(output_dir))
    if metadata_map.exists() and not force:
        raise RuntimeError("Canonical metadata map exists; use --force to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_map.parent.mkdir(parents=True, exist_ok=True)


def output_manifest(paths: Sequence[Path], root: Path, pd: Any) -> Any:
    rows = []  # type: List[Dict[str, Any]]
    for path in sorted(set(item.resolve() for item in paths if item.is_file())):
        if path.name == "output_manifest.tsv":
            continue
        rows.append({
            "relative_path": relative_path(path, root),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze discovery probe-gene-reviewed canonical UniProt mapping (Python 3.7+)."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preprocess-run-id", default="discovery_preprocess_v2")
    parser.add_argument("--mapping-lock", type=Path, default=None)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if sys.version_info < (3, 7):
        raise RuntimeError("Python 3.7 or newer is required")
    if not re.match(r"^[A-Za-z0-9._-]+$", args.preprocess_run_id):
        raise RuntimeError("Invalid preprocessing run ID")
    np, pd = require_python()
    root = args.root.expanduser().resolve()
    qc_dir = root / "results" / args.preprocess_run_id / "02_qc" / "discovery"
    if not (qc_dir / "run_summary.json").is_file():
        raise RuntimeError("Completed discovery QC run is missing: " + str(qc_dir))
    qc_summary = load_json(qc_dir / "run_summary.json")
    safeguards = [
        qc_summary.get("validation_data_read") is False,
        qc_summary.get("differential_expression_performed") is False,
        qc_summary.get("cross_platform_merge_performed") is False,
    ]
    if not all(safeguards):
        raise RuntimeError("Discovery QC run violates mapping-stage isolation safeguards")
    lock_path, lock = verify_mapping_lock(root, args.mapping_lock)
    hgnc_path = root / lock["hgnc"]["path"]
    uniprot_path = root / lock["uniprot"]["path"]
    if args.check_env:
        print("OSA identifier mapping environment: PASS")
        print("Python: {}".format(sys.version.replace("\n", " ")))
        print("numpy: {}".format(np.__version__))
        print("pandas: {}".format(pd.__version__))
        print("QC run: {}".format(qc_dir))
        print("HGNC: {}".format(hgnc_path))
        print("UniProt: {}".format(uniprot_path))
        return 0

    output_dir = root / "results" / args.preprocess_run_id / "03_mapping"
    metadata_map = root / "metadata" / "feature_gene_protein_map.tsv"
    prepare_output(root, output_dir, metadata_map, args.force)

    hgnc = pd.read_csv(
        str(hgnc_path), sep="\t", dtype=str, keep_default_na=False, low_memory=False
    )
    uniprot = pd.read_csv(
        str(uniprot_path), sep="\t", compression="gzip", dtype=str,
        keep_default_na=False, low_memory=False,
    )
    hgnc_indices = build_hgnc_indices(hgnc)
    uniprot_indices = build_uniprot_indices(uniprot)
    mapping_version = "HGNC_{}+UniProt_{}".format(
        lock["hgnc"]["snapshot_date"], lock["uniprot"]["release"]
    )

    mapped_frames = []  # type: List[Any]
    for dataset in ["GSE71356", "GSE75097"]:
        print("[{}] feature identifier mapping".format(dataset), flush=True)
        config = DATASETS[dataset]
        annotation = read_gpl_annotation(root / config["annotation"], pd)
        probe_qc = pd.read_csv(
            str(qc_dir / dataset / "probe_qc.tsv.gz"), sep="\t",
            compression="gzip", dtype=str, keep_default_na=False,
        )
        sensitivity = None
        sensitivity_path = qc_dir / dataset / "detection_threshold_probe_membership.tsv.gz"
        if sensitivity_path.is_file():
            sensitivity = pd.read_csv(
                str(sensitivity_path), sep="\t", compression="gzip",
                dtype=str, keep_default_na=False,
            )
        mapped_frames.append(map_dataset_features(
            dataset=dataset,
            platform=config["platform"],
            annotation=annotation,
            probe_qc=probe_qc,
            sensitivity_membership=sensitivity,
            hgnc=hgnc,
            hgnc_indices=hgnc_indices,
            uniprot=uniprot,
            uniprot_indices=uniprot_indices,
            mapping_version=mapping_version,
            pd=pd,
        ))
    feature_map = pd.concat(mapped_frames, ignore_index=True, sort=False)
    write_tsv(feature_map, metadata_map)
    write_tsv(feature_map, output_dir / "feature_gene_protein_map.tsv.gz")

    summaries = []  # type: List[Dict[str, Any]]
    status_rows = []  # type: List[Dict[str, Any]]
    for dataset in ["GSE71356", "GSE75097"]:
        dataset_map = feature_map[feature_map["dataset"] == dataset].copy()
        aggregate_summary = aggregate_dataset(
            dataset, dataset_map, qc_dir, output_dir, pd
        )
        summaries.append(aggregate_summary)
        for scope, scoped in [
            ("all_platform_probes", dataset_map),
            ("primary_filtered_probes", dataset_map[dataset_map["retained_primary_probe"] == "yes"]),
        ]:
            for status, count in scoped["mapping_status"].value_counts().items():
                status_rows.append({
                    "dataset": dataset, "scope": scope,
                    "mapping_status": status, "feature_count": int(count),
                    "scope_total": int(scoped.shape[0]),
                    "fraction": float(count / scoped.shape[0]) if scoped.shape[0] else float("nan"),
                })
    status_table = pd.DataFrame(status_rows)
    write_tsv(status_table, output_dir / "mapping_status_summary.tsv")

    protein_sets = {}  # type: Dict[str, Set[str]]
    gene_sets = {}  # type: Dict[str, Set[str]]
    for dataset in DATASETS:
        protein = pd.read_csv(
            str(output_dir / dataset / "protein_aggregation.tsv"), sep="\t", dtype=str
        )
        gene = pd.read_csv(
            str(output_dir / dataset / "gene_aggregation.tsv"), sep="\t", dtype=str
        )
        protein_sets[dataset] = set(protein["UniProt_accession"])
        gene_sets[dataset] = set(gene["HGNC_ID"])
    overlap = pd.DataFrame([{
        "dataset_A": "GSE71356", "dataset_B": "GSE75097",
        "shared_HGNC_genes": len(gene_sets["GSE71356"].intersection(gene_sets["GSE75097"])),
        "shared_reviewed_canonical_UniProt": len(
            protein_sets["GSE71356"].intersection(protein_sets["GSE75097"])
        ),
    }])
    write_tsv(overlap, output_dir / "cross_dataset_mapping_overlap.tsv")

    run_summary = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preprocess_run_id": args.preprocess_run_id,
        "preprocess_run_summary_sha256": sha256_file(qc_dir / "run_summary.json"),
        "mapping_lock": {
            "path": relative_path(lock_path, root), "sha256": sha256_file(lock_path),
        },
        "hgnc": lock["hgnc"],
        "uniprot": lock["uniprot"],
        "feature_map": {
            "path": relative_path(metadata_map, root),
            "sha256": sha256_file(metadata_map),
            "rows": int(feature_map.shape[0]),
        },
        "datasets": summaries,
        "cross_dataset_overlap": overlap.iloc[0].to_dict(),
        "validation_data_read": False,
        "differential_expression_read": False,
        "candidate_selection_performed": False,
        "next_gate": "Audit mapping coverage and high-weight candidates before discovery differential expression",
        "python": sys.version,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    summary_path = output_dir / "mapping_run_summary.json"
    summary_path.write_text(
        json.dumps(
            run_summary, ensure_ascii=True, indent=2, default=json_default
        ) + "\n",
        encoding="utf-8",
    )
    outputs = [metadata_map]
    outputs.extend(path for path in output_dir.rglob("*") if path.is_file())
    write_tsv(output_manifest(outputs, root, pd), output_dir / "output_manifest.tsv")
    print("OSA discovery identifier mapping: PASS")
    print("Feature map: {}".format(metadata_map))
    print("Output: {}".format(output_dir))
    print("No validation data, differential-expression result, or candidate ranking was read.")
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
