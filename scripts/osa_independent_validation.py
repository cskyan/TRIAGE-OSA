#!/usr/bin/env python3
"""Locked preprocessing and endpoint testing for independent OSA validation.

Python 3.7 compatible. This single driver validates frozen inputs, aligns
sample identities, performs cohort-specific preprocessing/QC, fits locked
OSA-minus-control gene models, and writes candidate-level evidence. The
default mode performs Stage 20 preprocessing without reading ranking values
or pathway outcomes. ``--endpoint`` performs the previously locked Stage 21
pair-support, comparator and Reactome tests against the frozen V4 ranking; it
never retrains or reranks the model and never reads an earlier validation
endpoint during execution. Because V3.2 outcomes were known before V4, this
V4 rerun is explicitly an exploratory locked model-update comparison.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET


SCRIPT_VERSION = "1.3.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LOCK_REL = "config/osa_independent_validation_preprocess_lock.json"
ENDPOINT_LOCK_REL = "config/osa_independent_validation_endpoint_lock.json"
IMPLEMENTATION_LOCK_REL = "config/osa_independent_validation_implementation_lock.json"
V4_IMPLEMENTATION_LOCK_REL = "config/osa_independent_validation_implementation_lock_v4.json"
V4_HANDOFF_REL = "config/triage_osa_external_validation_lock_v4.json"
V4_RANKING_REL = (
    "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/"
    "pair_context_scores_triage_osa_primary_250.tsv.gz"
)
DEFAULT_STAGE = "20_independent_validation_preprocess_v1"
DEFAULT_ENDPOINT_STAGE = "21_independent_validation_endpoint_triage_osa_v4"
LONG_STAGE = "33_longitudinal_validation_design_v2"
LONG_RESULT = "34_longitudinal_validation_endpoint_v1"
LONG_RECONSTRUCTION_RESULT = "35_longitudinal_network_reconstruction_v1"
LONG_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE283nnn/GSE283703/"
FRACTIONS = (0.005, 0.01, 0.02, 0.05)
RANKING_FAMILY = (
    ("full_TRIAGE_OSA", "triage_osa_rank_primary_250"),
    ("context_only", "rank_r_OSA"),
    ("PPI_only", "rank_p_PPI"),
    ("equal_L1_L2_L3", "rank_baseline_equal_L1_L2_L3"),
    ("L3_only", "rank_baseline_L3_only"),
    ("L1_L2", "rank_baseline_L1_L2"),
    ("L1_L3", "rank_baseline_L1_L3"),
    ("L2_L3", "rank_baseline_L2_L3"),
    ("PPI25_context75", "rank_priority_PPI25_context75"),
    ("PPI75_context25", "rank_priority_PPI75_context25"),
    ("bidirectional_conservative", "rank_priority_bidirectional_conservative"),
)
REACTOME_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def prepare_reactome_reference(root, force):
    output_dir = root / "annotations" / "independent_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "ReactomePathways.gmt.zip"
    gmt_path = output_dir / "ReactomePathways.gmt"
    manifest_path = output_dir / "reference_manifest.json"
    if manifest_path.exists() and not force:
        raise RuntimeError("Reference manifest exists; use --force to replace")
    part = archive_path.with_name(archive_path.name + ".part")
    if part.exists():
        part.unlink()
    request = urllib.request.Request(
        REACTOME_URL, headers={"User-Agent": "TRIAGE-OSA-validation/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if not zipfile.is_zipfile(str(part)):
        raise RuntimeError("Downloaded Reactome reference is not a valid ZIP")
    with zipfile.ZipFile(str(part), "r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError("Corrupt Reactome ZIP member: " + corrupt)
        members = [name for name in archive.namelist() if name.lower().endswith(".gmt")]
        if len(members) != 1:
            raise RuntimeError("Expected exactly one GMT in Reactome archive")
        payload = archive.read(members[0])
    lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
    malformed = []
    for line_number, line in enumerate(lines, 1):
        fields = line.split("\t")
        # GMT requires a set name, description/stable-ID field, and at least
        # one member. Reactome legitimately includes one-gene sets (3 fields).
        if len(fields) < 3 or any(not value.strip() for value in fields[:3]):
            malformed.append(line_number)
    if len(lines) < 1000 or malformed:
        raise RuntimeError(
            "Unexpected Reactome GMT structure/pathway count: pathways={} malformed_lines={}".format(
                len(lines), len(malformed)
            )
        )
    os.replace(str(part), str(archive_path))
    temporary_gmt = gmt_path.with_name(gmt_path.name + ".tmp")
    temporary_gmt.write_bytes(payload)
    os.replace(str(temporary_gmt), str(gmt_path))
    manifest = {
        "created_at_utc": utc_now(), "status": "PASS",
        "source_url": REACTOME_URL,
        "downloaded_before_independent_outcome_analysis": True,
        "archive": {
            "path": "annotations/independent_validation/ReactomePathways.gmt.zip",
            "bytes": archive_path.stat().st_size,
            "sha256": file_sha256(archive_path),
        },
        "gmt": {
            "path": "annotations/independent_validation/ReactomePathways.gmt",
            "bytes": gmt_path.stat().st_size,
            "sha256": file_sha256(gmt_path),
            "pathways": len(lines),
            "minimum_fields_per_pathway": 3,
            "malformed_lines": 0,
        },
        "validation_or_model_outputs_read": False,
        "next_gate": "Freeze exact pathway-axis membership and endpoint formulas before preprocessing",
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary_manifest), str(manifest_path))
    print("OSA independent Reactome reference preparation: PASS")
    print("Pathways: {}".format(len(lines)))
    print("GMT SHA256: {}".format(manifest["gmt"]["sha256"]))
    print("Output: {}".format(output_dir))
    print("Validation/model outputs read: no")
    return 0


def parse_sample_metadata(path):
    selected = {}
    wanted = {
        "!Sample_title", "!Sample_geo_accession", "!Sample_description",
        "!Sample_characteristics_ch1", "!Sample_platform_id",
        "!Sample_data_processing",
    }
    with gzip.open(str(path), "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            key = line.split("\t", 1)[0]
            if key not in wanted:
                continue
            fields = next(csv.reader([line.rstrip("\r\n")], delimiter="\t"))
            selected.setdefault(key, []).append(fields[1:])
    for required in ["!Sample_title", "!Sample_geo_accession", "!Sample_platform_id"]:
        if required not in selected or len(selected[required]) != 1:
            raise RuntimeError("Missing or repeated series metadata field: " + required)
    return selected


def excel_column_index(reference):
    match = re.match(r"^([A-Z]+)", reference or "")
    if not match:
        raise RuntimeError("Invalid XLSX cell reference: " + str(reference))
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def xlsx_cell_text(cell, shared, namespace):
    kind = cell.attrib.get("t", "")
    if kind == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(namespace + "t")
        )
    value = cell.find(namespace + "v")
    if value is None or value.text is None:
        return ""
    if kind == "s":
        index = int(value.text)
        if index < 0 or index >= len(shared):
            raise RuntimeError("XLSX shared-string index out of range")
        return shared[index]
    return value.text


def parse_tpm_xlsx(np, path):
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(str(path), "r") as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(namespace + "si"):
                shared.append("".join(
                    node.text or "" for node in item.iter(namespace + "t")
                ))
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in archive.namelist():
            raise RuntimeError("GSE293433 workbook lacks sheet1.xml")
        headers = None
        identifiers = []
        values = []
        malformed = []
        workbook_rows = 0
        with archive.open(sheet_name, "r") as source:
            for _, element in ET.iterparse(source, events=("end",)):
                if element.tag != namespace + "row":
                    continue
                cells = {}
                for cell in element.findall(namespace + "c"):
                    index = excel_column_index(cell.attrib.get("r", ""))
                    cells[index] = xlsx_cell_text(cell, shared, namespace)
                width = max(cells) + 1 if cells else 0
                row = [cells.get(index, "") for index in range(width)]
                element.clear()
                if headers is None:
                    headers = row
                    continue
                workbook_rows += 1
                row = row + [""] * max(0, len(headers) - len(row))
                row = row[:len(headers)]
                identifier = str(row[0]).strip()
                strict = re.match(r"^ENSG\d+(?:\.\d+)?$", identifier) is not None
                numeric = []
                numeric_ok = True
                for value in row[1:]:
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        numeric_ok = False
                        break
                    if not math.isfinite(parsed):
                        numeric_ok = False
                        break
                    numeric.append(parsed)
                if not strict or not numeric_ok or len(numeric) != 10:
                    malformed.append({
                        "workbook_data_row": workbook_rows,
                        "identifier_preview": identifier[:120],
                        "strict_Ensembl": "yes" if strict else "no",
                        "ten_finite_numeric_values": "yes" if numeric_ok and len(numeric) == 10 else "no",
                        "exclusion_reason": "non_Ensembl_or_non_numeric_footer_row",
                    })
                    continue
                identifiers.append(identifier.split(".", 1)[0])
                values.append(numeric)
    expected = ["name", "S1", "S2", "S3", "S4", "S5", "Z1", "Z2", "Z3", "Z4", "Z5"]
    if headers != expected:
        raise RuntimeError("Unexpected GSE293433 workbook columns: " + repr(headers))
    if workbook_rows != 27301 or len(identifiers) != 27292 or len(malformed) != 9:
        raise RuntimeError(
            "GSE293433 structural audit mismatch rows={} valid={} malformed={}".format(
                workbook_rows, len(identifiers), len(malformed)
            )
        )
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Duplicate valid GSE293433 Ensembl identifiers")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (27292, 10) or not np.isfinite(matrix).all():
        raise RuntimeError("Invalid GSE293433 numeric matrix")
    if float(np.min(matrix)) < 0.0:
        raise RuntimeError("GSE293433 TPM matrix contains negative values")
    return identifiers, headers[1:], matrix, malformed, workbook_rows


def fit_two_group(np, expression, groups):
    from osa_adipose_validation_preprocess import moderated_statistics
    disease = np.asarray([1.0 if value == "OSA" else 0.0 for value in groups])
    n_control = int(np.sum(disease == 0.0))
    n_osa = int(np.sum(disease == 1.0))
    design = np.stack([np.ones(len(groups)), disease], axis=1)
    if int(np.linalg.matrix_rank(design)) != 2:
        raise RuntimeError("Two-group design is rank deficient")
    response = expression.T
    inverse = np.linalg.inv(np.dot(design.T, design))
    coefficients = np.dot(np.dot(inverse, design.T), response)
    residuals = response - np.dot(design, coefficients)
    residual_df = design.shape[0] - design.shape[1]
    s2 = np.sum(residuals * residuals, axis=0) / float(residual_df)
    result = moderated_statistics(
        np, coefficients[1, :], s2, residual_df, float(inverse[1, 1])
    )
    result["n_control"] = n_control
    result["n_OSA"] = n_osa
    return result


def statistics_rows(fmt, identifiers, identifier_name, expression, result):
    rows = []
    for index, identifier in enumerate(identifiers):
        rows.append({
            identifier_name: identifier,
            "mean_expression_model_scale": fmt(float(np_mean(expression[index, :]))),
            "beta_OSA_minus_control": fmt(result["beta"][index]),
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
            "design": "intercept_plus_OSA; beta=OSA_minus_control",
            "candidate_selection_allowed": "no",
        })
    return rows


def np_mean(values):
    return sum(float(value) for value in values) / float(len(values))


def parse_rank(value):
    parsed = float(value)
    if not math.isfinite(parsed) or abs(parsed - round(parsed)) > 1e-8:
        raise RuntimeError("Invalid frozen rank: " + str(value))
    return int(round(parsed))


def bh_adjust(values):
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [1.0] * len(values)
    running = 1.0
    for reverse_position in range(len(values) - 1, -1, -1):
        index = order[reverse_position]
        rank = reverse_position + 1
        running = min(running, float(values[index]) * len(values) / float(rank))
        result[index] = min(1.0, running)
    return result


def sign_test_one_sided(successes, total):
    if total <= 0:
        return float("nan")
    def combination(n_value, k_value):
        k_value = min(k_value, n_value - k_value)
        numerator = 1
        denominator = 1
        for offset in range(1, k_value + 1):
            numerator *= n_value - k_value + offset
            denominator *= offset
        return numerator // denominator
    return min(1.0, sum(
        combination(total, value) for value in range(successes, total + 1)
    ) / float(2 ** total))


def verify_stage20(root, read_tsv, sha256_file):
    stage = root / "results/discovery_preprocess_v2/20_independent_validation_preprocess_v1"
    summary_path = stage / "run_summary.json"
    manifest_path = stage / "output_manifest.tsv"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Missing Stage 20 preprocessing outputs")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS_INDEPENDENT_PREPROCESS":
        raise RuntimeError("Stage 20 preprocessing has not passed")
    manifest, _ = read_tsv(manifest_path)
    if len(manifest) != 7:
        raise RuntimeError("Unexpected Stage 20 output manifest length")
    for row in manifest:
        path = root / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise RuntimeError("Stage 20 manifest verification failed: " + row["path"])
    return stage, summary, sha256_file(summary_path), sha256_file(manifest_path)


def build_independent_node_evidence(np, candidates, candidate_evidence, dataset):
    from osa_cpap_reversal_endpoint import midrank_percentiles
    source = {row["UniProt_accession"]: row for row in candidate_evidence}
    available_field = dataset + (
        "_primary_filter_available" if dataset == "GSE293433" else "_available"
    )
    beta_field = dataset + "_beta_OSA_minus_control"
    t_field = dataset + "_moderated_t"
    available = []
    oriented_t = []
    evidence = {}
    for candidate in candidates:
        accession = candidate["UniProt_accession"]
        row = source[accession]
        if row[available_field] != "yes":
            evidence[accession] = {
                "available": False, "oriented_beta": None,
                "oriented_t": None, "score": None,
                "same_direction": None,
            }
            continue
        discovery_sign = 1.0 if float(candidate["OSA_beta"]) > 0.0 else -1.0
        oriented_beta = discovery_sign * float(row[beta_field])
        value = discovery_sign * float(row[t_field])
        available.append(accession)
        oriented_t.append(value)
        evidence[accession] = {
            "available": True, "oriented_beta": oriented_beta,
            "oriented_t": value, "score": None,
            "same_direction": oriented_beta > 0.0,
        }
    scores = midrank_percentiles(np, oriented_t)
    for accession, score in zip(available, scores):
        evidence[accession]["score"] = float(score)
    return evidence, available


def evidence_arrays(np, candidates, evidence):
    score = np.zeros(len(candidates), dtype=np.float64)
    available = np.zeros(len(candidates), dtype=bool)
    positive = np.zeros(len(candidates), dtype=bool)
    oriented_t = np.full(len(candidates), np.nan, dtype=np.float64)
    for index, candidate in enumerate(candidates):
        item = evidence[candidate["UniProt_accession"]]
        available[index] = item["available"]
        if item["available"]:
            score[index] = item["score"]
            positive[index] = item["same_direction"]
            oriented_t[index] = item["oriented_t"]
    return score, available, positive, oriented_t


def observed_pair_metrics(np, pair_rows, support, positive, eligible, rank_field):
    from osa_cpap_reversal_endpoint import normalized_auec
    ranks = np.asarray([parse_rank(row[rank_field]) for row in pair_rows], dtype=np.int64)
    background = float(np.mean(support[eligible]))
    binary_background = float(np.mean(positive[eligible]))
    rows = []
    differences = []
    ratios = []
    for fraction in FRACTIONS:
        threshold = int(math.ceil(len(pair_rows) * fraction))
        selected = eligible & (ranks <= threshold)
        if int(np.sum(selected)) == 0:
            raise RuntimeError("Frozen top set has zero eligible pairs")
        top = float(np.mean(support[selected]))
        top_binary = float(np.mean(positive[selected]))
        difference = top - background
        ratio = top / max(background, 1e-12)
        differences.append(difference)
        ratios.append(ratio)
        rows.append({
            "fraction": fraction, "rank_threshold": threshold,
            "eligible_top_pairs": int(np.sum(selected)),
            "eligible_background_pairs": int(np.sum(eligible)),
            "top_mean_pair_support": top,
            "background_mean_pair_support": background,
            "support_enrichment_ratio": ratio,
            "support_enrichment_difference": difference,
            "top_both_nodes_same_direction_fraction": top_binary,
            "background_both_nodes_same_direction_fraction": binary_background,
        })
    auec = normalized_auec(
        np, np.asarray(FRACTIONS, dtype=np.float64),
        np.asarray(differences, dtype=np.float64),
    )
    return rows, float(auec), float(ratios[1])


def family_node_permutations(
    np, rng, iterations, batch_size, label, candidates, evidence,
    pair_rows, eligible, left_index, right_index,
):
    from osa_cpap_reversal_endpoint import matched_blocks, normalized_auec
    candidate_order = [row["UniProt_accession"] for row in candidates]
    candidate_index = {value: index for index, value in enumerate(candidate_order)}
    available = [value for value in candidate_order if evidence[value]["available"]]
    blocks = matched_blocks(np, candidates, available, 10)
    block_indices = [np.asarray(
        [candidate_index[value] for value in block], dtype=np.int64
    ) for block in blocks]
    base = np.zeros(len(candidates), dtype=np.float64)
    for accession in available:
        base[candidate_index[accession]] = evidence[accession]["score"]
    masks = {}
    for model, field in RANKING_FAMILY:
        ranks = np.asarray([parse_rank(row[field]) for row in pair_rows], dtype=np.int64)
        masks[model] = [
            eligible & (ranks <= int(math.ceil(len(pair_rows) * fraction)))
            for fraction in FRACTIONS
        ]
        if any(int(np.sum(mask)) == 0 for mask in masks[model]):
            raise RuntimeError("No eligible top pairs for " + model)
    null_ratio = {
        model: np.empty(iterations, dtype=np.float64) for model, _ in RANKING_FAMILY
    }
    null_auec = {
        model: np.empty(iterations, dtype=np.float64) for model, _ in RANKING_FAMILY
    }
    completed = 0
    fraction_array = np.asarray(FRACTIONS, dtype=np.float64)
    while completed < iterations:
        size = min(batch_size, iterations - completed)
        permuted = np.repeat(base.reshape(1, -1), size, axis=0)
        for batch_row in range(size):
            for indices in block_indices:
                permuted[batch_row, indices] = base[rng.permutation(indices)]
        support = np.sqrt(
            permuted[:, left_index] * permuted[:, right_index]
        )
        background = support[:, eligible].mean(axis=1)
        for model, _ in RANKING_FAMILY:
            top_means = np.stack([
                support[:, mask].mean(axis=1) for mask in masks[model]
            ], axis=1)
            differences = top_means - background.reshape(-1, 1)
            null_ratio[model][completed:completed + size] = (
                top_means[:, 1] / np.maximum(background, 1e-12)
            )
            null_auec[model][completed:completed + size] = np.asarray([
                normalized_auec(np, fraction_array, row) for row in differences
            ], dtype=np.float64)
        completed += size
        if completed % 1000 == 0 or completed == iterations:
            print("[{} node-permutation] {}/{}".format(
                label, completed, iterations
            ), flush=True)
    block_rows = []
    candidate_map = {row["UniProt_accession"]: row for row in candidates}
    for block_id, block in enumerate(blocks, 1):
        for accession in block:
            block_rows.append({
                "endpoint": label, "block_id": block_id,
                "block_size": len(block), "UniProt_accession": accession,
                "protein_detection_fraction": candidate_map[accession]["protein_detection_fraction"],
                "log2_UniProt_length": candidate_map[accession]["log2_UniProt_length"],
            })
    return null_ratio, null_auec, block_rows


def permutation_row(np, endpoint, model, statistic, observed, null, iterations):
    null = np.asarray(null, dtype=np.float64)
    return {
        "endpoint": endpoint, "model": model, "statistic": statistic,
        "observed": float(observed), "null_mean": float(np.mean(null)),
        "null_SD": float(np.std(null, ddof=1)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q50": float(np.quantile(null, 0.50)),
        "null_q975": float(np.quantile(null, 0.975)),
        "one_sided_P": (1.0 + float(np.sum(null >= observed))) / (iterations + 1.0),
        "iterations": iterations,
    }


def hgnc_identifier_maps(path):
    entrez_to_symbol = {}
    ensembl_to_symbol = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            symbol = row.get("symbol", "").strip()
            entrez = row.get("entrez_id", "").strip()
            ensembl = row.get("ensembl_gene_id", "").strip().split(".", 1)[0]
            if symbol and entrez:
                entrez_to_symbol[entrez] = symbol
            if symbol and ensembl:
                ensembl_to_symbol[ensembl] = symbol
    return entrez_to_symbol, ensembl_to_symbol


def reactome_sets(path):
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 3:
                result[fields[1]] = {
                    "name": fields[0], "genes": set(fields[2:])
                }
    return result


def pathway_gene_nulls(np, rng, values, set_sizes, iterations, label):
    sizes = sorted(set(int(value) for value in set_sizes))
    total = float(np.sum(values))
    count = len(values)
    nulls = {size: np.empty(iterations, dtype=np.float64) for size in sizes}
    for iteration in range(iterations):
        order = rng.permutation(count)
        cumulative = np.cumsum(values[order])
        for size in sizes:
            selected_sum = float(cumulative[size - 1])
            selected_mean = selected_sum / float(size)
            other_mean = (total - selected_sum) / float(count - size)
            nulls[size][iteration] = selected_mean - other_mean
        done = iteration + 1
        if done % 1000 == 0 or done == iterations:
            print("[{} gene-label-permutation] {}/{}".format(
                label, done, iterations
            ), flush=True)
    return nulls


def pathway_endpoint(
    np, rng, dataset, stat_rows, identifier_field, identifier_to_symbol,
    reactome, axis_lock, iterations,
):
    symbol_values = {}
    for row in stat_rows:
        symbol = identifier_to_symbol.get(row[identifier_field], "")
        if symbol:
            symbol_values.setdefault(symbol, []).append(float(row["moderated_t"]))
    symbols = sorted(symbol_values)
    values = np.asarray([
        np_mean(symbol_values[symbol]) for symbol in symbols
    ], dtype=np.float64)
    symbol_set = set(symbols)
    definitions = []
    for axis, config in axis_lock.items():
        union = set()
        for stable_id in config["stable_ids"]:
            if stable_id not in reactome:
                raise RuntimeError("Locked Reactome stable ID is absent: " + stable_id)
            genes = reactome[stable_id]["genes"] & symbol_set
            definitions.append({
                "level": "pathway", "axis": axis, "stable_id": stable_id,
                "name": reactome[stable_id]["name"], "genes": genes,
                "expected_direction": "two_sided",
                "minimum": 5,
            })
            union.update(reactome[stable_id]["genes"])
        expected = config["expected_direction"]
        if expected == "up_in_GSE38792_two_sided_in_GSE293433":
            expected = "up" if dataset == "GSE38792" else "two_sided"
        elif expected == "down_in_GSE38792_two_sided_in_GSE293433":
            expected = "down" if dataset == "GSE38792" else "two_sided"
        definitions.append({
            "level": "axis", "axis": axis, "stable_id": "|".join(config["stable_ids"]),
            "name": axis, "genes": union & symbol_set,
            "expected_direction": expected, "minimum": 15,
        })
    sizes = [len(item["genes"]) for item in definitions if (
        len(item["genes"]) >= item["minimum"] and len(item["genes"]) < len(values)
    )]
    nulls = pathway_gene_nulls(np, rng, values, sizes, iterations, dataset)
    value_map = dict(zip(symbols, values))
    total = float(np.sum(values))
    rows = []
    axis_positions = []
    axis_p = []
    for item in definitions:
        genes = sorted(item["genes"])
        size = len(genes)
        eligible = size >= item["minimum"] and size < len(values)
        row = {
            "dataset": dataset, "level": item["level"], "axis": item["axis"],
            "Reactome_stable_ID": item["stable_id"], "Reactome_name": item["name"],
            "expected_direction": item["expected_direction"],
            "tested_genes": len(values), "measurable_set_genes": size,
            "minimum_required": item["minimum"], "eligible": "yes" if eligible else "no",
            "competitive_effect": "", "nominal_P": "", "BH_FDR_axis_family": "",
        }
        if eligible:
            selected_sum = sum(value_map[gene] for gene in genes)
            selected_mean = selected_sum / float(size)
            other_mean = (total - selected_sum) / float(len(values) - size)
            observed = selected_mean - other_mean
            null = nulls[size]
            direction = item["expected_direction"]
            if direction == "up":
                p_value = (1.0 + float(np.sum(null >= observed))) / (iterations + 1.0)
            elif direction == "down":
                p_value = (1.0 + float(np.sum(null <= observed))) / (iterations + 1.0)
            else:
                p_value = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (iterations + 1.0)
            row["competitive_effect"] = observed
            row["nominal_P"] = p_value
            if item["level"] == "axis":
                axis_positions.append(len(rows))
                axis_p.append(p_value)
        rows.append(row)
    adjusted = bh_adjust(axis_p)
    for position, value in zip(axis_positions, adjusted):
        rows[position]["BH_FDR_axis_family"] = value
    return rows, len(values)


def exact_label_direction_test(
    np, dataset, expression, sample_groups, candidate_rows, identifiers,
    identifier_field, observed_successes, batch_size=250,
):
    identifier_index = {value: index for index, value in enumerate(identifiers)}
    selected = [row for row in candidate_rows if row[identifier_field] in identifier_index]
    matrix = np.asarray([
        expression[identifier_index[row[identifier_field]], :] for row in selected
    ], dtype=np.float64)
    discovery_sign = np.asarray([
        1.0 if float(row["OSA_beta"]) > 0.0 else -1.0 for row in selected
    ], dtype=np.float64).reshape(-1, 1)
    n_samples = matrix.shape[1]
    n_osa = sum(value == "OSA" for value in sample_groups)
    observed_labels = np.asarray([
        1.0 if value == "OSA" else 0.0 for value in sample_groups
    ], dtype=np.float64).reshape(1, -1)
    observed_beta = (
        np.dot(matrix, observed_labels.T) / float(n_osa)
        - np.dot(matrix, (1.0 - observed_labels).T) / float(n_samples - n_osa)
    )
    reproduced_successes = int(np.sum(discovery_sign * observed_beta > 0.0))
    if reproduced_successes != int(observed_successes):
        raise RuntimeError("Exact-label observed statistic does not reproduce Stage 20: " + dataset)
    combinations = list(itertools.combinations(range(n_samples), n_osa))
    counts = np.empty(len(combinations), dtype=np.int64)
    completed = 0
    while completed < len(combinations):
        current = combinations[completed:completed + batch_size]
        labels = np.zeros((len(current), n_samples), dtype=np.float64)
        for row_index, indices in enumerate(current):
            labels[row_index, list(indices)] = 1.0
        mean_osa = np.dot(matrix, labels.T) / float(n_osa)
        mean_control = np.dot(matrix, (1.0 - labels).T) / float(n_samples - n_osa)
        oriented_beta = discovery_sign * (mean_osa - mean_control)
        counts[completed:completed + len(current)] = np.sum(
            oriented_beta > 0.0, axis=0
        )
        completed += len(current)
    return {
        "dataset": dataset,
        "statistic": "candidate_same_direction_count",
        "measurable_primary_candidates": len(selected),
        "observed": int(observed_successes),
        "assignments": len(combinations),
        "null_mean": float(np.mean(counts)),
        "null_SD": float(np.std(counts, ddof=1)),
        "null_q025": float(np.quantile(counts, 0.025)),
        "null_q50": float(np.quantile(counts, 0.50)),
        "null_q975": float(np.quantile(counts, 0.975)),
        "one_sided_exact_P": float(np.sum(counts >= observed_successes)) / float(len(counts)),
    }


def run_exact_label_sensitivities(np, root, candidates, directional_rows):
    from osa_cpap_validation_preprocess import map_and_aggregate, parse_series_matrix
    matrix_path = root / "geo/GSE38792_series_matrix.txt.gz"
    gpl_path = root / "annotations/GPL6244.annot.gz"
    ids387, samples387, matrix387 = parse_series_matrix(np, matrix_path)
    entrez387, expression387, _, _ = map_and_aggregate(
        np, "GSE38792", ids387, matrix387, gpl_path
    )
    metadata387 = parse_sample_metadata(matrix_path)
    titles387 = metadata387["!Sample_title"][0]
    groups387 = ["OSA" if "_OSA_" in value else "control" for value in titles387]
    xlsx_path = root / "geo/GSE293433_processed_data_mRNA.xlsx"
    metadata293 = parse_sample_metadata(root / "geo/GSE293433_series_matrix.txt.gz")
    descriptions = metadata293.get("!Sample_description", [[]])[0]
    libraries = [value.split(":", 1)[1].strip() for value in descriptions]
    ids293_all, workbook_samples, tpm_all, _, _ = parse_tpm_xlsx(np, xlsx_path)
    reorder = [workbook_samples.index(value) for value in libraries]
    tpm_all = tpm_all[:, reorder]
    primary_filter = np.sum(tpm_all >= 1.0, axis=1) >= 2
    ids293 = [value for value, keep in zip(ids293_all, primary_filter) if bool(keep)]
    expression293 = np.log2(tpm_all[primary_filter, :] + 1.0)
    groups293 = ["OSA" if value.startswith("S") else "control" for value in libraries]
    success_map = {
        row["dataset"]: int(row["same_direction_candidates"])
        for row in directional_rows
    }
    rows = [
        exact_label_direction_test(
            np, "GSE38792", expression387, groups387, candidates, entrez387,
            "Entrez", success_map["GSE38792"], 250,
        ),
        exact_label_direction_test(
            np, "GSE293433", expression293, groups293, candidates, ids293,
            "Ensembl_gene", success_map["GSE293433"], 252,
        ),
    ]
    if rows[0]["assignments"] != 43758 or rows[1]["assignments"] != 252:
        raise RuntimeError("Exact label-assignment count mismatch")
    if rows[0]["measurable_primary_candidates"] != 236 or rows[1]["measurable_primary_candidates"] != 204:
        raise RuntimeError("Exact label-sensitivity candidate coverage mismatch")
    return rows


def write_formatted_tsv(path, rows, fields, write_tsv, fmt):
    formatted = []
    for source in rows:
        row = {}
        for field in fields:
            value = source.get(field, "")
            if isinstance(value, (float,)):
                row[field] = fmt(value)
            else:
                row[field] = str(value)
        formatted.append(row)
    write_tsv(path, formatted, fields)


def run_independent_endpoint(args, root, np):
    from osa_cpap_validation_preprocess import (
        atomic_json, fmt, read_tsv, relative_path, sha256_file, write_tsv,
    )
    from osa_cpap_reversal_endpoint import midrank_percentiles
    stage20, stage20_summary, stage20_summary_sha, stage20_manifest_sha = verify_stage20(
        root, read_tsv, sha256_file
    )
    lock_path = root / ENDPOINT_LOCK_REL
    implementation_lock_path = root / V4_IMPLEMENTATION_LOCK_REL
    handoff_path = root / V4_HANDOFF_REL
    ranking_path = root / V4_RANKING_REL
    candidate_path = root / "results/discovery_preprocess_v2/04_candidates_v1/candidate_proteins_primary_250.tsv"
    hgnc_path = root / "annotations/hgnc_complete_set_20260714.tsv"
    reactome_path = root / "annotations/independent_validation/ReactomePathways.gmt"
    reference_manifest_path = root / "annotations/independent_validation/reference_manifest.json"
    required = [
        lock_path, implementation_lock_path, handoff_path, ranking_path, candidate_path,
        hgnc_path, reactome_path, reference_manifest_path,
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError("Missing endpoint input: " + str(path))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "PRE_RESULT_ENDPOINT_LOCK":
        raise RuntimeError("Independent endpoint is not frozen")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff.get("status") != "V4_FROZEN_FOR_EXTERNAL_VALIDATION":
        raise RuntimeError("V4 external-validation handoff is not frozen")
    if handoff["ranking"]["primary_250"] != V4_RANKING_REL:
        raise RuntimeError("V4 handoff ranking path mismatch")
    if handoff["separation_policy"].get("v4_reanalysis_is_confirmatory") is not False:
        raise RuntimeError("V4 independent rerun must remain non-confirmatory")
    implementation_lock = json.loads(
        implementation_lock_path.read_text(encoding="utf-8")
    )
    if implementation_lock.get("status") != "V4_POST_ADAPTATION_PRE_FORMAL_LOCK":
        raise RuntimeError("V4 independent implementation is not frozen")
    if implementation_lock["endpoint_lock_sha256"] != sha256_file(lock_path):
        raise RuntimeError("Implementation lock refers to a different endpoint lock")
    if implementation_lock["v4_external_handoff_sha256"] != sha256_file(handoff_path):
        raise RuntimeError("Implementation lock refers to a different V4 handoff")
    if not longitudinal_legacy_source_matches(implementation_lock["script_sha256"]):
        raise RuntimeError("V4 Stage 21 script changed after implementation freeze")
    frozen = lock["frozen_inputs"]
    checks = {
        "preprocess_lock": (root / frozen["preprocess_lock"]["path"], frozen["preprocess_lock"]["sha256"]),
        "reactome_gmt": (reactome_path, frozen["reactome_gmt"]["sha256"]),
        "reactome_manifest": (reference_manifest_path, frozen["reactome_manifest"]["sha256"]),
        "primary_ranking": (ranking_path, handoff["ranking"]["primary_250_sha256"]),
        "candidate_universe": (candidate_path, frozen["candidate_universe"]["sha256"]),
    }
    input_hashes = {}
    for name, (path, expected) in checks.items():
        observed = sha256_file(path)
        input_hashes[name] = observed
        if observed != expected:
            raise RuntimeError("Frozen endpoint hash mismatch: " + name)
    stage20_raw_paths = {
        "GSE38792_series_matrix.txt.gz": root / "geo/GSE38792_series_matrix.txt.gz",
        "GSE293433_processed_data_mRNA.xlsx": root / "geo/GSE293433_processed_data_mRNA.xlsx",
        "GSE293433_series_matrix.txt.gz": root / "geo/GSE293433_series_matrix.txt.gz",
        "GPL6244.annot.gz": root / "annotations/GPL6244.annot.gz",
        "hgnc_complete_set_20260714.tsv": hgnc_path,
    }
    for name, path in stage20_raw_paths.items():
        observed = sha256_file(path)
        expected = stage20_summary["input_hashes"][name]
        input_hashes["stage20_raw_" + name] = observed
        if observed != expected:
            raise RuntimeError("Stage 20 raw input changed before exact sensitivity: " + name)
    candidates, _ = read_tsv(candidate_path)
    pairs, pair_fields = read_tsv(ranking_path)
    candidate_evidence, _ = read_tsv(stage20 / "candidate_independent_evidence_500.tsv")
    candidate_evidence = [row for row in candidate_evidence if row["primary_250"] == "yes"]
    if len(candidates) != 250 or len(candidate_evidence) != 250 or len(pairs) != 31125:
        raise RuntimeError("Independent endpoint universe mismatch")
    if len({row["UniProt_accession"] for row in candidates}) != 250:
        raise RuntimeError("Duplicate primary candidate accession")
    if len({row["pair_id"] for row in pairs}) != 31125:
        raise RuntimeError("Duplicate frozen pair")
    for _, field in RANKING_FAMILY:
        if field not in pair_fields:
            raise RuntimeError("Frozen ranking lacks: " + field)
        ranks = sorted(parse_rank(row[field]) for row in pairs)
        if ranks != list(range(1, 31126)):
            raise RuntimeError("Rank is not a complete permutation: " + field)
    iterations = 50 if args.smoke else int(args.permutations)
    if iterations <= 0 or args.batch_size <= 0:
        raise RuntimeError("permutations and batch-size must be positive")
    formal_iterations = int(lock["randomization"]["iterations"])
    if not args.smoke and iterations != formal_iterations:
        raise RuntimeError("Formal Stage 21 requires exactly {} permutations".format(formal_iterations))
    print("OSA independent validation endpoint environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("Frozen proteins=250 pairs=31,125 ranking_models={}".format(len(RANKING_FAMILY)))
    print("Permutations: {}{}".format(iterations, " (smoke)" if args.smoke else ""))
    print("Model: {}".format(handoff["model"]["name"]))
    print("V4 reanalysis role: locked exploratory model-update comparison")
    print("Training/reranking/previous validation read: no")
    if args.check_env:
        return 0
    output_dir = root / "results/discovery_preprocess_v2" / args.endpoint_output_stage
    if args.smoke:
        output_dir = output_dir / "smoke"
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Endpoint output exists; use --force to replace")
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace endpoint output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    candidate_index = {row["UniProt_accession"]: index for index, row in enumerate(candidates)}
    left = np.asarray([candidate_index[row["protein_A"]] for row in pairs], dtype=np.int64)
    right = np.asarray([candidate_index[row["protein_B"]] for row in pairs], dtype=np.int64)
    evidence = {}
    arrays = {}
    for dataset in ["GSE38792", "GSE293433"]:
        evidence[dataset], _ = build_independent_node_evidence(
            np, candidates, candidate_evidence, dataset
        )
        arrays[dataset] = evidence_arrays(np, candidates, evidence[dataset])
    joint_evidence = {}
    score387, available387, positive387, oriented387 = arrays["GSE38792"]
    score293, available293, positive293, oriented293 = arrays["GSE293433"]
    for index, candidate in enumerate(candidates):
        accession = candidate["UniProt_accession"]
        available = bool(available387[index] and available293[index])
        joint_evidence[accession] = {
            "available": available,
            "score": float(math.sqrt(score387[index] * score293[index])) if available else None,
            "oriented_t": float(math.sqrt(max(oriented387[index], 0.0) * max(oriented293[index], 0.0))) if available else None,
            "oriented_beta": None,
            "same_direction": bool(positive387[index] and positive293[index]) if available else None,
        }
    evidence["joint"] = joint_evidence
    arrays["joint"] = evidence_arrays(np, candidates, joint_evidence)
    observed = {}
    metric_rows = []
    node_rows = []
    block_rows = []
    comparator_rows = []
    seed = int(lock["randomization"]["seed"])
    for endpoint_index, endpoint in enumerate(["GSE38792", "GSE293433", "joint"]):
        score, available, positive, _ = arrays[endpoint]
        eligible = available[left] & available[right]
        support = np.sqrt(score[left] * score[right])
        both_positive = positive[left] & positive[right]
        observed[endpoint] = {}
        for model, field in RANKING_FAMILY:
            fraction_rows, auec, ratio = observed_pair_metrics(
                np, pairs, support, both_positive, eligible, field
            )
            observed[endpoint][model] = {"ratio": ratio, "auec": auec}
            for row in fraction_rows:
                row.update({
                    "endpoint": endpoint, "model": model, "rank_field": field,
                    "metric_scope": "fraction", "normalized_AUEC": "",
                })
                metric_rows.append(row)
            metric_rows.append({
                "endpoint": endpoint, "model": model, "rank_field": field,
                "metric_scope": "AUEC", "fraction": "0.005_to_0.05",
                "normalized_AUEC": auec,
                "eligible_background_pairs": int(np.sum(eligible)),
            })
        null_ratio, null_auec, blocks = family_node_permutations(
            np, np.random.RandomState(seed + endpoint_index), iterations,
            args.batch_size, endpoint, candidates, evidence[endpoint], pairs,
            eligible, left, right,
        )
        block_rows.extend(blocks)
        for model, _ in RANKING_FAMILY:
            node_rows.append(permutation_row(
                np, endpoint, model, "top_1pct_support_enrichment_ratio",
                observed[endpoint][model]["ratio"], null_ratio[model], iterations,
            ))
            node_rows.append(permutation_row(
                np, endpoint, model, "normalized_AUEC_0p5_to_5pct",
                observed[endpoint][model]["auec"], null_auec[model], iterations,
            ))
        for comparator in ["context_only", "PPI_only", "equal_L1_L2_L3"]:
            observed_delta = (
                observed[endpoint]["full_TRIAGE_OSA"]["ratio"]
                - observed[endpoint][comparator]["ratio"]
            )
            null_delta = null_ratio["full_TRIAGE_OSA"] - null_ratio[comparator]
            comparator_rows.append({
                "endpoint": endpoint, "comparator": comparator,
                "statistic": "full_minus_comparator_top_1pct_ratio",
                "observed_delta": observed_delta,
                "null_mean_delta": float(np.mean(null_delta)),
                "one_sided_P": (1.0 + float(np.sum(null_delta >= observed_delta))) / (iterations + 1.0),
                "BH_FDR_primary_comparator_family": "",
                "positive_gain": "yes" if observed_delta > 0.0 else "no",
            })
    primary_comparator_indices = [
        index for index, row in enumerate(comparator_rows)
        if row["endpoint"] == "GSE38792"
    ]
    adjusted = bh_adjust([
        comparator_rows[index]["one_sided_P"] for index in primary_comparator_indices
    ])
    for index, value in zip(primary_comparator_indices, adjusted):
        comparator_rows[index]["BH_FDR_primary_comparator_family"] = value
    node_output = []
    for index, candidate in enumerate(candidates):
        row = {
            "UniProt_accession": candidate["UniProt_accession"],
            "gene_symbol": candidate["gene_symbol_current"],
            "discovery_rank": candidate["discovery_rank"],
            "discovery_OSA_beta": candidate["OSA_beta"],
        }
        for endpoint in ["GSE38792", "GSE293433", "joint"]:
            item = evidence[endpoint][candidate["UniProt_accession"]]
            row[endpoint + "_available"] = "yes" if item["available"] else "no"
            row[endpoint + "_oriented_beta"] = item["oriented_beta"] if item["oriented_beta"] is not None else ""
            row[endpoint + "_oriented_t"] = item["oriented_t"] if item["oriented_t"] is not None else ""
            row[endpoint + "_node_score"] = item["score"] if item["score"] is not None else ""
            row[endpoint + "_same_direction"] = (
                "yes" if item["same_direction"] else "no"
            ) if item["same_direction"] is not None else ""
        node_output.append(row)
    directional_rows = []
    for dataset in ["GSE38792", "GSE293433"]:
        items = [evidence[dataset][row["UniProt_accession"]] for row in candidates]
        items = [item for item in items if item["available"]]
        successes = sum(bool(item["same_direction"]) for item in items)
        directional_rows.append({
            "dataset": dataset, "measurable_primary_candidates": len(items),
            "same_direction_candidates": successes,
            "same_direction_fraction": successes / float(len(items)),
            "one_sided_exact_sign_P": sign_test_one_sided(successes, len(items)),
        })
    label_rows = run_exact_label_sensitivities(
        np, root, candidates, directional_rows
    )
    entrez_map, ensembl_map = hgnc_identifier_maps(hgnc_path)
    reactome = reactome_sets(reactome_path)
    stats387, _ = read_tsv(stage20 / "GSE38792_differential_all_entrez.tsv.gz")
    stats293, _ = read_tsv(stage20 / "GSE293433_differential_filtered_ensembl.tsv.gz")
    pathway_rows = []
    rows, tested387 = pathway_endpoint(
        np, np.random.RandomState(seed + 100), "GSE38792", stats387,
        "Entrez", entrez_map, reactome, lock["reactome_axes"], iterations,
    )
    pathway_rows.extend(rows)
    rows, tested293 = pathway_endpoint(
        np, np.random.RandomState(seed + 101), "GSE293433", stats293,
        "Ensembl_gene", ensembl_map, reactome, lock["reactome_axes"], iterations,
    )
    pathway_rows.extend(rows)
    primary_node = next(
        row for row in node_rows
        if row["endpoint"] == "GSE38792"
        and row["model"] == "full_TRIAGE_OSA"
        and row["statistic"] == "top_1pct_support_enrichment_ratio"
    )
    context_delta = next(
        row for row in comparator_rows
        if row["endpoint"] == "GSE38792" and row["comparator"] == "context_only"
    )
    formal = (not args.smoke and iterations == formal_iterations)
    paths = {
        "node": output_dir / "candidate_node_evidence_primary_250.tsv",
        "direction": output_dir / "candidate_directional_concordance.tsv",
        "label": output_dir / "candidate_exact_label_sensitivity.tsv",
        "metrics": output_dir / "pair_support_metrics.tsv",
        "permutation": output_dir / "node_aware_permutation_summary.tsv",
        "comparator": output_dir / "comparator_delta_permutation.tsv",
        "blocks": output_dir / "matched_node_blocks.tsv",
        "pathway": output_dir / "reactome_pathway_axis_results.tsv",
    }
    write_formatted_tsv(paths["node"], node_output, list(node_output[0]), write_tsv, fmt)
    write_formatted_tsv(paths["direction"], directional_rows, list(directional_rows[0]), write_tsv, fmt)
    write_formatted_tsv(paths["label"], label_rows, list(label_rows[0]), write_tsv, fmt)
    metric_fields = [
        "endpoint", "model", "rank_field", "metric_scope", "fraction",
        "rank_threshold", "eligible_top_pairs", "eligible_background_pairs",
        "top_mean_pair_support", "background_mean_pair_support",
        "support_enrichment_ratio", "support_enrichment_difference",
        "top_both_nodes_same_direction_fraction",
        "background_both_nodes_same_direction_fraction", "normalized_AUEC",
    ]
    write_formatted_tsv(paths["metrics"], metric_rows, metric_fields, write_tsv, fmt)
    write_formatted_tsv(paths["permutation"], node_rows, list(node_rows[0]), write_tsv, fmt)
    write_formatted_tsv(paths["comparator"], comparator_rows, list(comparator_rows[0]), write_tsv, fmt)
    write_formatted_tsv(paths["blocks"], block_rows, list(block_rows[0]), write_tsv, fmt)
    write_formatted_tsv(paths["pathway"], pathway_rows, list(pathway_rows[0]), write_tsv, fmt)
    summary = {
        "created_at_utc": utc_now(),
        "status": "PASS_FORMAL_INDEPENDENT_VALIDATION" if formal else "PASS_SMOKE",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "endpoint_lock": {"path": ENDPOINT_LOCK_REL, "sha256": sha256_file(lock_path)},
        "implementation_lock": {
            "path": V4_IMPLEMENTATION_LOCK_REL,
            "sha256": sha256_file(implementation_lock_path),
        },
        "external_validation_handoff": {
            "path": V4_HANDOFF_REL,
            "sha256": sha256_file(handoff_path),
        },
        "model": handoff["model"]["name"],
        "ranking": {
            "path": V4_RANKING_REL,
            "sha256": input_hashes["primary_ranking"],
            "rank_field": "triage_osa_rank_primary_250",
        },
        "stage20": {
            "status": stage20_summary["status"], "run_summary_sha256": stage20_summary_sha,
            "output_manifest_sha256": stage20_manifest_sha,
        },
        "frozen_input_hashes": input_hashes,
        "proteins": 250, "pairs": 31125, "ranking_models": len(RANKING_FAMILY),
        "permutations": iterations, "seed": seed,
        "candidate_coverage": {
            "GSE38792": int(np.sum(available387)),
            "GSE293433": int(np.sum(available293)),
            "joint": int(np.sum(available387 & available293)),
        },
        "tested_pathway_symbols": {"GSE38792": tested387, "GSE293433": tested293},
        "exact_label_assignments": {"GSE38792": 43758, "GSE293433": 252},
        "primary_endpoint": primary_node,
        "GSE38792_full_minus_context_top1pct": context_delta,
        "disease_support_component_of_fusion_gate": (
            "PASS_POSITIVE_GAIN" if context_delta["observed_delta"] > 0.0
            else "NO_POSITIVE_GAIN"
        ),
        "model_retrained_or_reranked": False,
        "candidate_membership_changed": False,
        "previous_validation_outcomes_read_by_this_execution": False,
        "previous_v3_2_validation_outcomes_known_before_v4": True,
        "v4_reanalysis_is_confirmatory": False,
        "v4_reanalysis_role": "locked exploratory model-update comparison using unchanged endpoint definitions",
        "known_PPI_or_complex_read": False,
        "elapsed_seconds": time.time() - started,
        "next_gate": "Build the V4 consolidated evidence package without retraining or reranking",
    }
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, summary)
    output_paths = list(paths.values()) + [summary_path]
    manifest = [{
        "path": relative_path(path, root), "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in output_paths]
    write_tsv(output_dir / "output_manifest.tsv", manifest, ["path", "bytes", "sha256"])
    print("OSA independent validation endpoint: {}".format(summary["status"]))
    print("Primary GSE38792 top1 ratio={:.4f} P={:.10f}".format(
        float(primary_node["observed"]), float(primary_node["one_sided_P"])
    ))
    print("Full-minus-context top1 delta={:.4f} P={:.10f}".format(
        float(context_delta["observed_delta"]), float(context_delta["one_sided_P"])
    ))
    print("Output: {}".format(output_dir))
    print("Training/reranking/previous validation read by this execution: no")
    return 0


def longitudinal_write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def longitudinal_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(str(temporary), str(path))


def longitudinal_legacy_source_matches(expected, source_path=None):
    """Keep Stage21's original source lock without re-freezing old analyses.

    Remove only the explicitly added longitudinal constants/functions/CLI branch,
    restore this guard's previous spelling, and require the exact old source digest.
    Any modification to the legacy scientific implementation still fails closed.
    """
    path = Path(source_path) if source_path else Path(__file__).resolve()
    if file_sha256(path) == expected:
        return True
    source = path.read_text(encoding='utf-8')
    try:
        source = source[:source.index('\ndef longitudinal_write_tsv') + 1] + source[source.index('\ndef main():\n') + 1:]
        source = '\n'.join(line for line in source.split('\n') if not line.startswith((
            'LONG_STAGE =', 'LONG_RESULT =', 'LONG_RECONSTRUCTION_RESULT =', 'LONG_BASE_URL =',
            '    longitudinal_mode =', '    longitudinal_mode.add_argument')))
        reconstruction_start = source.index('    if args.longitudinal_network_reconstruction:')
        reconstruction_end = source.index('    if args.longitudinal_prepare or args.longitudinal_endpoint:', reconstruction_start)
        source = source[:reconstruction_start] + source[reconstruction_end:]
        start = source.index('    if args.longitudinal_prepare or args.longitudinal_endpoint:')
        end = source.index('    if args.prepare_references:', start)
        source = source[:start] + source[end:]
        source = source.replace(
            '    if not longitudinal_legacy_source_matches(implementation_lock["script_sha256"]):',
            '    if implementation_lock["script_sha256"] != sha256_file(Path(__file__).resolve()):', 1)
    except ValueError:
        return False
    return hashlib.sha256(source.encode('utf-8')).hexdigest() == expected


def longitudinal_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def longitudinal_fetch(url, path):
    """Cache public files atomically; never replace an existing source file."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TRIAGE-OSA-research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1024 * 1024)
    with gzip.open(str(partial), "rb") as handle:
        while handle.read(1024 * 1024):
            pass
    os.replace(str(partial), str(path))


def parse_longitudinal_soft(path):
    """Use GEO title AM/PM, not an invented meaning for library suffix t1/t3."""
    samples, current = [], None
    series = []
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("^SAMPLE = "):
                current = {"sample_id": line.split(" = ", 1)[1]}
                samples.append(current)
            elif line.startswith("^"):
                current = None
            elif " = " in line:
                key, value = line.split(" = ", 1)
                if current is not None:
                    current.setdefault(key, []).append(value)
                elif key in ("!Series_summary", "!Series_overall_design"):
                    series.append(value)
    result = []
    seen = set()
    for sample in samples:
        title = sample["!Sample_title"][0]
        match = re.fullmatch(r"(Healthy\d+|OSA\d+)_(?:(Short|Long)_Treat_)?(AM|PM)_(.+)", title)
        if not match:
            raise RuntimeError("Unrecognized longitudinal title: " + title)
        subject, treatment, clock, suffix = match.groups()
        visit = {None: "baseline", "Short": "month4", "Long": "month24"}[treatment]
        libraries = [x.split(":", 1)[1].strip() for x in sample.get("!Sample_description", [])
                     if x.startswith("Library name:")]
        if libraries != [suffix]:
            raise RuntimeError("Title/library mismatch: " + title)
        characteristics = dict(x.split(": ", 1) for x in sample.get("!Sample_characteristics_ch1", []) if ": " in x)
        disease = "OSA" if subject.startswith("OSA") else "Healthy"
        expected_treatment = {"baseline": "none", "month4": "short-term CPAP", "month24": "long-term CPAP"}[visit]
        if characteristics.get("disease") != disease or characteristics.get("treatment") != expected_treatment:
            raise RuntimeError("Conflicting GEO disease/treatment: " + title)
        key = (subject, visit, clock)
        if key in seen:
            raise RuntimeError("Duplicate subject/visit/clock: " + repr(key))
        seen.add(key)
        result.append({
            "dataset": "GSE283703", "sample_id": sample["sample_id"], "subject_id": subject,
            "disease": disease, "visit": visit, "clock": clock, "library_name": suffix,
            "title": title, "tissue": characteristics.get("tissue", ""),
            "cell_type": characteristics.get("cell type", ""), "treatment": expected_treatment,
            "age": characteristics.get("age", ""), "sex": characteristics.get("sex", ""),
            "BMI": characteristics.get("bmi", ""), "AHI": characteristics.get("ahi", ""),
            "ESS": characteristics.get("ess", ""), "adherence_hours": "",
            "clock_source": "explicit_GEO_title_not_library_suffix",
            "clinical_values_imputed": "no", "training_allowed": "no",
        })
    libraries = [r["library_name"] for r in result]
    if len(libraries) != len(set(libraries)) or len(result) != 71:
        raise RuntimeError("Unexpected longitudinal sample/library count")
    return result, series


def longitudinal_contrasts(samples):
    subjects = sorted({r["subject_id"] for r in samples if r["disease"] == "OSA"})
    index = {(r["subject_id"], r["visit"], r["clock"]): r["library_name"] for r in samples}
    rows = []
    for subject in subjects:
        for visit in ("month4", "month24"):
            for clock in ("AM", "PM", "DAY_MEAN"):
                clocks = ("AM", "PM") if clock == "DAY_MEAN" else (clock,)
                pre = [index.get((subject, "baseline", c), "") for c in clocks]
                post = [index.get((subject, visit, c), "") for c in clocks]
                rows.append({"contrast": visit + "_" + clock, "subject_id": subject,
                             "pre_libraries": "|".join(pre), "post_libraries": "|".join(post),
                             "eligible": "yes" if all(pre + post) else "no",
                             "reason": "" if all(pre + post) else "missing_same_clock_baseline_or_followup"})
    return rows


def longitudinal_rewire(edges, rng, attempts_per_edge=30):
    """Double-edge swaps: exact node identities and degrees, no loops/multiedges."""
    original = sorted({tuple(sorted(e)) for e in edges})
    work, edge_set = list(original), set(original)
    swaps = 0
    if len(work) < 2:
        return work, swaps
    for _ in range(attempts_per_edge * len(work)):
        i, j = rng.sample(range(len(work)), 2)
        a, b = work[i]
        c, d = work[j]
        if len({a, b, c, d}) != 4:
            continue
        if rng.random() < 0.5:
            c, d = d, c
        left, right = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if left in edge_set or right in edge_set:
            continue
        edge_set.remove(work[i]); edge_set.remove(work[j])
        edge_set.add(left); edge_set.add(right)
        work[i], work[j] = left, right
        swaps += 1
    return sorted(work), swaps


def prepare_longitudinal(args, root):
    """Freeze ranking/graphs/analysis before any new expression value is inspected."""
    import random
    out = root / "results/discovery_preprocess_v2" / LONG_STAGE
    result = root / "results/discovery_preprocess_v2" / LONG_RESULT
    if (out / "analysis_contract.json").exists():
        print("Longitudinal design already frozen; verifying source files")
        contract = json.loads((out / "analysis_contract.json").read_text(encoding="utf-8"))
        for relative, expected in contract["frozen_files"].items():
            if file_sha256(root / relative) != expected:
                raise RuntimeError("Frozen longitudinal input changed: " + relative)
        print("PASS_FROZEN_DESIGN_REUSED; --force does not overwrite a scientific lock")
        return 0
    if (result / "run_summary.json").exists():
        raise RuntimeError("Cannot create a pre-outcome design after endpoint output exists")
    soft = root / "geo/GSE283703_family.soft.gz"
    if not soft.exists():
        if args.check_env:
            print("PREPARE_REQUIRED: metadata has not been downloaded; no expression values needed")
            return 0
        longitudinal_fetch(LONG_BASE_URL + "soft/" + soft.name, soft)
    samples, protocol = parse_longitudinal_soft(soft)
    candidate_path = root / "results/discovery_preprocess_v2/04_candidates_v1/candidate_proteins_primary_250.tsv"
    ranking_path = root / V4_RANKING_REL
    axis_path = root / ENDPOINT_LOCK_REL
    gmt_path = root / "annotations/independent_validation/ReactomePathways.gmt"
    hgnc_path = root / "annotations/hgnc_complete_set_20260714.tsv"
    required = (candidate_path, ranking_path, axis_path, gmt_path, hgnc_path)
    for path in required:
        if not path.exists():
            raise RuntimeError("Missing design input: " + str(path))
    if args.check_env:
        print("Longitudinal design preflight: PASS; samples={} subjects={}".format(
            len(samples), len({r['subject_id'] for r in samples})))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    candidates = longitudinal_rows(candidate_path)
    ranks = longitudinal_rows(ranking_path)
    if len(candidates) != 250 or len(ranks) != 31125 or len({r['pair_id'] for r in ranks}) != 31125:
        raise RuntimeError("Unexpected frozen primary universe")
    node = {r["UniProt_accession"]: r for r in candidates}
    if len(node) != 250:
        raise RuntimeError("Duplicate candidate proteins")
    axes = json.loads(axis_path.read_text(encoding="utf-8"))["reactome_axes"]
    gmt = {}
    with gmt_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) >= 3:
                gmt[fields[1]] = set(fields[2:])
    axis_genes = {axis: set().union(*(gmt[x] for x in definition['stable_ids'])) for axis, definition in axes.items()}
    panels = {"ALL": set(node)}
    panels.update({axis: {p for p, row in node.items() if row['gene_symbol_current'] in genes}
                   for axis, genes in axis_genes.items()})
    methods = dict(RANKING_FAMILY[:5])
    graph_rows, graph_qc, annotation_rows = [], [], []
    rng = random.Random(20260904)
    for panel, members in panels.items():
        # Pathway-anchored candidate neighborhoods: at least one endpoint is a
        # directly annotated member. This does not assign its partner to the pathway.
        eligible = [r for r in ranks if r['protein_A'] in members or r['protein_B'] in members]
        # Fixed 1% within each panel, min 1; tiny panels remain explicitly not evaluable.
        count = max(1, int(math.ceil(len(eligible) * .01)))
        for method, rank_field in methods.items():
            ordered = sorted(eligible, key=lambda r: (parse_rank(r[rank_field]), r['pair_id']))
            chosen = ordered[:count]
            edges = [tuple(sorted((r['protein_A'], r['protein_B']))) for r in chosen]
            graph_qc.append({"panel": panel, "method": method, "panel_nodes": len(members),
                             "eligible_edges": len(eligible), "selected_edges": len(edges),
                             "selected_nodes": len({p for e in edges for p in e}),
                             "graph_role": "frozen_model_selection"})
            graph_rows.extend({"panel": panel, "method": method, "replicate": 0,
                               "protein_A": a, "protein_B": b} for a, b in edges)
            if method == "full_TRIAGE_OSA" and len(edges) >= 2:
                seen = set()
                for rep in range(1, 101):
                    rewired, swaps = longitudinal_rewire(edges, rng)
                    if not swaps or rewired == sorted(edges) or tuple(rewired) in seen:
                        continue
                    seen.add(tuple(rewired))
                    graph_rows.extend({"panel": panel, "method": "full_degree_preserving_null",
                                       "replicate": rep, "protein_A": a, "protein_B": b} for a, b in rewired)
                graph_qc.append({"panel": panel, "method": "full_degree_preserving_null",
                                 "selected_edges": len(edges), "unique_null_graphs": len(seen),
                                 "graph_role": "same_nodes_same_degrees_no_expression_read"})
    for protein, row in node.items():
        memberships = sorted(axis for axis, genes in axis_genes.items() if row['gene_symbol_current'] in genes)
        annotation_rows.append({"UniProt": protein, "symbol": row['gene_symbol_current'],
                                "Ensembl_gene": row['Ensembl_gene'], "Entrez": row['Entrez'],
                                "OSA_beta": row['OSA_beta'], "pathway_axes": '|'.join(memberships),
                                "annotation_status": "annotated" if memberships else "no_locked_axis_annotation"})
    output_files = []
    def save(name, rows, fields):
        path = out / name
        longitudinal_write_tsv(path, rows, fields)
        output_files.append(path)
    save("sample_manifest.tsv", samples, list(samples[0]))
    contrasts = longitudinal_contrasts(samples)
    save("paired_contrasts.tsv", contrasts, list(contrasts[0]))
    save("candidate_axis_annotation.tsv", annotation_rows, list(annotation_rows[0]))
    gene_sets = [{'axis':axis,'symbol':gene} for axis,genes in sorted(axis_genes.items()) for gene in sorted(genes)]
    save("pathway_gene_sets.tsv", gene_sets, ['axis','symbol'])
    save("frozen_graphs.tsv", graph_rows, ["panel", "method", "replicate", "protein_A", "protein_B"])
    save("frozen_graph_coverage.tsv", graph_qc, ["panel", "method", "panel_nodes", "eligible_edges", "selected_edges", "selected_nodes", "unique_null_graphs", "graph_role"])
    family_rows = [
        {"dataset": "GSE38792", "subset": "case_control", "cohort_family": "Gharib_hernia_18", "tissue": "visceral_adipose", "participants":18, "status":"shared_participants_confirmed", "source":"10.1093/sleep/zsz314; PMID23288968"},
        {"dataset": "GSE135917", "subset": "Study_Group_1", "cohort_family": "Gharib_hernia_18", "tissue": "subcutaneous_adipose", "participants":18, "status":"shared_participants_confirmed", "source":"10.1093/sleep/zsz314"},
        {"dataset": "GSE135917", "subset": "Study_Group_2", "cohort_family": "GSE135917_Study2_unresolved", "participants":24, "status":"overlap_with_GSE49800_not_established", "source":"10.1093/sleep/zsz314"},
        {"dataset": "GSE49800", "subset": "CPAP", "cohort_family": "GSE49800_unresolved", "participants":18, "status":"overlap_with_GSE135917_not_established", "source":"GEO"},
        {"dataset": "GSE283703", "subset": "all", "cohort_family": "Koritala_GSE283703", "participants":19, "status":"external_by_public_provenance_not_person_identity_verified", "source":"GEO_SOFT"},
    ]
    save("cohort_family_correction.tsv", family_rows, ["dataset", "subset", "cohort_family", "tissue", "participants", "status", "source"])
    # A reusable metadata overlay, not a rewrite of frozen historical manifests.
    longitudinal_write_tsv(root / "metadata/cohort_family_correction.tsv", family_rows,
                           ["dataset", "subset", "cohort_family", "tissue", "participants", "status", "source"])
    summary = {"status": "PASS_METADATA_AND_PRE_OUTCOME_DESIGN", "created_at_utc": utc_now(),
               "dataset": "GSE283703", "samples": len(samples),
               "OSA_subjects": len({r['subject_id'] for r in samples if r['disease']=='OSA'}),
               "healthy_subjects": len({r['subject_id'] for r in samples if r['disease']=='Healthy'}),
               "paired_n": {c: sum(r['contrast']==c and r['eligible']=='yes' for r in contrasts) for c in sorted({r['contrast'] for r in contrasts})},
               "expression_values_read": False, "clinical_fields_complete": False,
               "protocol": protocol, "model_retrained": False,
               "warning": "AM/PM follows explicit titles; t1/t3 suffix meanings are not inferred. No individual age/sex/BMI/ESS/adherence are imputed."}
    longitudinal_json(out / "run_summary.json", summary)
    contract = {
        "schema": "OSA_longitudinal_2.0", "created_at_utc": utc_now(),
        "design_revision_reason": "Metadata/annotation-only feasibility audit: pathway-internal top1% graphs were too small. Use pathway-anchored neighborhoods and separately test full pathway expression. No new expression values inspected.",
        "outcomes_inspected_before_lock": False, "training_allowed": False,
        "primary": "ALL_month4_AM_full_minus_context_signed_concordant_pair_reversal",
        "node_score": "-sign(frozen_discovery_OSA_beta)*[log2(normalized_count_post+1)-log2(normalized_count_pre+1)]",
        "edge_score": "sign(a)*min(abs(a),abs(b)) if a*b>0 else 0; equal edge weights",
        "graph_score": "mean edge score within each subject; nodes are not independent replicates",
        "axis_graph_definition": "top1pct of frozen candidate edges with at least one directly annotated axis member; partner membership not inferred",
        "pathway_score": "mean log2(count+1) change of full mapped axis genes minus other mapped HGNC genes, not restricted to model candidates; paired subject sign-flips preserve gene correlation",
        "minimum_pathway_genes": 15, "minimum_pathway_coverage": 0.5,
        "feature_mapping": "exact HGNC approved symbol/Ensembl gene ID/Entrez, strip Ensembl version only, exclude ambiguous; average log2(count+1) for multiple features per gene, report multiplicity",
        "null": "exact subject sign-flips, two-sided; assumes symmetric paired differences under null, not randomized causal CPAP assignment",
        "secondary": "all six locked axes and ALL, month4/month24 AM/PM/DAY_MEAN; full-vs-context/PPI/equal/L3, same-node degree-rewire contrast, unique-node response",
        "multiplicity": "primary single test; BH across all evaluable non-primary tests, including time interaction",
        "minimum_subjects": 6, "minimum_nodes": 5, "minimum_edges": 3,
        "minimum_edge_coverage": 0.8, "minimum_node_coverage": 0.8, "minimum_null_graphs": 20,
        "seed": 20260904, "bootstrap_replicates": 5000,
        "interpretation": "prospective frozen molecular-response validation; not physical PPI or clinical benefit prediction",
        "limitations": ["small cohort", "no untreated longitudinal OSA arm", "time/batch/treatment confounding cannot be eliminated", "same-node rewiring tests response organization, not binding"],
        "frozen_files": {str(p.relative_to(root)).replace('\\','/'): file_sha256(p) for p in list(required)+[soft]+output_files},
    }
    longitudinal_json(out / "analysis_contract.json", contract)
    print("Longitudinal metadata/design: PASS; 19 subjects, 71 samples; no expression values read")
    print("Paired counts: " + repr(summary['paired_n']))
    print("Output: " + str(out))
    return 0


def longitudinal_signflip(np, values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or not len(values) or len(values) > 18:
        raise RuntimeError("Invalid exact sign-flip input")
    patterns = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = patterns.dot(values) / len(values)
    observed = float(np.mean(values))
    return observed, float(np.mean(np.abs(null) >= abs(observed) - 1e-12))


def longitudinal_rank_correlation(np, first, second):
    """Tie-aware Spearman correlation without a scipy dependency."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(valid)) < 5:
        return float("nan")

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        position = 0
        while position < len(values):
            end = position + 1
            while end < len(values) and values[order[end]] == values[order[position]]:
                end += 1
            result[order[position:end]] = 0.5 * (position + end - 1)
            position = end
        return result

    left, right = ranks(first[valid]), ranks(second[valid])
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def longitudinal_neighbor_reconstruction(np, edges, values):
    """Leave-one-node-out reconstruction from graph neighbours."""
    neighbors = {}
    for first, second in edges:
        if first not in values or second not in values:
            continue
        neighbors.setdefault(first, []).append(second)
        neighbors.setdefault(second, []).append(first)
    nodes = sorted(node for node in neighbors if neighbors[node])
    observed = np.asarray([values[node] for node in nodes], dtype=float)
    predicted = np.asarray([
        np.mean([values[other] for other in neighbors[node]]) for node in nodes
    ], dtype=float)
    return {
        "nodes": int(len(nodes)),
        "Spearman": longitudinal_rank_correlation(np, observed, predicted),
        "MAE": float(np.mean(np.abs(observed - predicted))),
        "sign_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
    }


def run_longitudinal_network_reconstruction(args, root):
    """Post-hoc diagnostic of edge organization; never trains or reranks."""
    import numpy as np
    design = root / "results/discovery_preprocess_v2" / LONG_STAGE
    endpoint = root / "results/discovery_preprocess_v2" / LONG_RESULT
    output = root / "results/discovery_preprocess_v2" / LONG_RECONSTRUCTION_RESULT
    required = [
        design / "analysis_contract.json", design / "frozen_graphs.tsv",
        endpoint / "run_summary.json", endpoint / "subject_node_responses.tsv",
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError("Missing longitudinal reconstruction input: " + str(path))
    endpoint_summary = json.loads(required[2].read_text(encoding="utf-8"))
    if endpoint_summary.get("status") != "PASS_FROZEN_LONGITUDINAL_EXECUTION":
        raise RuntimeError("Stage 34 is not a completed frozen longitudinal execution")
    if args.check_env:
        print("Longitudinal network reconstruction environment: PASS")
        print("Input: Stage 34 subject-node responses; model training/reranking: no")
        print("Primary: subject-mean leave-one-node-out Spearman, full graph minus same-node same-degree null")
        return 0
    if output.exists() and not args.force:
        raise RuntimeError("Reconstruction output exists; use --force to replace this post-hoc diagnostic")
    output.mkdir(parents=True, exist_ok=True)

    values = {}
    for row in longitudinal_rows(required[3]):
        values.setdefault((row["contrast"], row["subject_id"]), {})[row["protein"]] = float(row["signed_reversal"])
    graphs = {}
    for row in longitudinal_rows(required[1]):
        if row["panel"] != "ALL" or row["method"] not in (
                "full_TRIAGE_OSA", "full_degree_preserving_null"):
            continue
        graphs.setdefault((row["method"], int(row["replicate"])), []).append(
            (row["protein_A"], row["protein_B"])
        )
    full_edges = graphs.get(("full_TRIAGE_OSA", 0), [])
    null_keys = sorted(key for key in graphs if key[0] == "full_degree_preserving_null")
    if len(full_edges) < 3 or len(null_keys) < 20:
        raise RuntimeError("Insufficient full/null graph definitions for reconstruction")

    detail = []
    subject_deltas = {}
    for key in sorted(values):
        contrast, subject = key
        full = longitudinal_neighbor_reconstruction(np, full_edges, values[key])
        null = [longitudinal_neighbor_reconstruction(np, graphs[item], values[key]) for item in null_keys]
        null_rho = float(np.mean([item["Spearman"] for item in null]))
        null_mae = float(np.mean([item["MAE"] for item in null]))
        null_sign = float(np.mean([item["sign_accuracy"] for item in null]))
        row = {
            "contrast": contrast, "subject_id": subject, "nodes": full["nodes"],
            "full_Spearman": full["Spearman"], "null_mean_Spearman": null_rho,
            "delta_Spearman": full["Spearman"] - null_rho,
            "full_MAE": full["MAE"], "null_mean_MAE": null_mae,
            "delta_MAE_improvement": null_mae - full["MAE"],
            "full_sign_accuracy": full["sign_accuracy"],
            "null_mean_sign_accuracy": null_sign,
            "delta_sign_accuracy": full["sign_accuracy"] - null_sign,
        }
        detail.append(row)
        subject_deltas.setdefault(subject, []).append(row)

    tests = []
    rng = np.random.RandomState(20260904)

    def add_test(name, metric, selected, primary=False):
        subjects = sorted(selected)
        vector = np.asarray([selected[x] for x in subjects], dtype=float)
        effect, p_value = longitudinal_signflip(np, vector)
        sampled = vector[rng.randint(len(vector), size=(5000, len(vector)))]
        low, high = np.percentile(np.mean(sampled, axis=1), [2.5, 97.5])
        tests.append({
            "comparison": name, "metric": metric, "n_subjects": len(subjects),
            "analysis_role": "primary" if primary else "secondary",
            "effect": effect, "CI95_low": float(low), "CI95_high": float(high),
            "exact_signflip_P": p_value, "BH_FDR_secondary": "",
        })

    expected_contrasts = set(row["contrast"] for row in detail)
    complete_subjects = dict(
        (subject, rows) for subject, rows in subject_deltas.items()
        if set(row["contrast"] for row in rows) == expected_contrasts
    )
    add_test(
        "full_minus_degree_null_all_contrasts_subject_mean", "Spearman",
        dict((subject, float(np.mean([x["delta_Spearman"] for x in rows])))
             for subject, rows in complete_subjects.items()), True,
    )
    for metric, column in (("MAE_improvement", "delta_MAE_improvement"),
                           ("sign_accuracy", "delta_sign_accuracy")):
        add_test(
            "full_minus_degree_null_all_contrasts_subject_mean", metric,
            dict((subject, float(np.mean([x[column] for x in rows])))
                 for subject, rows in complete_subjects.items()),
        )
    for contrast in sorted(set(row["contrast"] for row in detail)):
        selected = dict((row["subject_id"], row["delta_Spearman"])
                        for row in detail if row["contrast"] == contrast)
        add_test("full_minus_degree_null_" + contrast, "Spearman", selected)

    secondary = [index for index, row in enumerate(tests) if row["analysis_role"] == "secondary"]
    order = sorted(secondary, key=lambda index: tests[index]["exact_signflip_P"])
    running = 1.0
    for position in range(len(order) - 1, -1, -1):
        index = order[position]
        running = min(running, tests[index]["exact_signflip_P"] * len(order) / float(position + 1))
        tests[index]["BH_FDR_secondary"] = running
    primary = tests[0]
    supportive = primary["effect"] > 0.0 and primary["exact_signflip_P"] <= 0.05
    longitudinal_write_tsv(output / "subject_reconstruction_metrics.tsv", detail, list(detail[0]))
    longitudinal_write_tsv(output / "reconstruction_tests.tsv", tests, list(tests[0]))
    summary = {
        "status": "PASS_POSTHOC_NETWORK_RECONSTRUCTION",
        "created_at_utc": utc_now(), "primary": primary,
        "full_edges": len(full_edges), "null_graphs": len(null_keys),
        "subjects_any_contrast": len(subject_deltas),
        "subjects_complete_all_contrasts": len(complete_subjects),
        "contrasts": len(expected_contrasts),
        "exploratory_structural_signal": bool(supportive),
        "model_retrained": False, "ranking_changed": False,
        "validation_outcomes_read": True, "checkpoint_selection_eligible": False,
        "next_decision": (
            "Design a discovery-only topology-contrastive adapter and require a new untouched validation cohort"
            if supportive else
            "Do not increase structural weight; retain structure as prior and narrow the manuscript claim"
        ),
        "claim_boundary": "Tests molecular response organization, not physical binding, symptoms, adherence, or clinical benefit",
    }
    longitudinal_json(output / "run_summary.json", summary)
    print("Longitudinal network reconstruction: " + summary["status"])
    print("Primary: " + repr(primary))
    print("Exploratory structural signal: " + ("yes" if supportive else "no"))
    print("Output: " + str(output))
    return 0


def longitudinal_graph_score(np, edges, values):
    a = np.asarray([values[x] for x, y in edges])
    b = np.asarray([values[y] for x, y in edges])
    score = np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)
    return float(np.mean(score))


def run_longitudinal(args, root):
    import numpy as np
    design = root / "results/discovery_preprocess_v2" / LONG_STAGE
    contract_path = design / "analysis_contract.json"
    if not contract_path.exists():
        raise RuntimeError("Run --longitudinal-prepare first; no endpoint is allowed before design freeze")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for relative, expected in contract['frozen_files'].items():
        if file_sha256(root / relative) != expected:
            raise RuntimeError("Frozen longitudinal input changed: " + relative)
    if args.check_env:
        print("Longitudinal endpoint preflight: PASS; Python {} numpy {}".format(sys.version.split()[0], np.__version__))
        print("Frozen design present; expression will be downloaded only in the actual endpoint run")
        return 0
    if args.smoke:
        raise RuntimeError("Do not preview real held-out outcomes with --smoke; use synthetic unit tests")
    out = root / "results/discovery_preprocess_v2" / LONG_RESULT
    if out.exists() and not args.force:
        raise RuntimeError("Endpoint output exists; --force only reruns the SAME frozen analysis")
    out.mkdir(parents=True, exist_ok=True)
    longitudinal_json(out / 'run_summary.json', {'status':'RUNNING', 'created_at_utc':utc_now(), 'model_retrained':False})
    path = root / "geo/GSE283703_Normalized_counts.csv.gz"
    longitudinal_fetch(LONG_BASE_URL + "suppl/" + path.name, path)
    samples = longitudinal_rows(design / 'sample_manifest.tsv')
    annotations = longitudinal_rows(design / 'candidate_axis_annotation.tsv')
    identifiers = {}
    for row in annotations:
        for field in ('symbol','Ensembl_gene','Entrez'):
            for identifier in re.split(r'[|;, ]+',row[field]):
                if identifier:
                    identifiers.setdefault(identifier, set()).add(row['UniProt'])
    hgnc = longitudinal_rows(root/'annotations/hgnc_complete_set_20260714.tsv')
    symbol_map = {}
    for row in hgnc:
        if row['status'] != 'Approved':
            continue
        for field in ('symbol','ensembl_gene_id','entrez_id'):
            for identifier in re.split(r'[|;, ]+',row[field]):
                if identifier:
                    symbol_map.setdefault(identifier,set()).add(row['symbol'])
    expression, feature_audit, symbol_expression = {}, [], {}
    with gzip.open(str(path),'rt',encoding='utf-8-sig',newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        libraries = header[1:]
        if len(set(libraries)) != len(libraries) or set(libraries) != {r['library_name'] for r in samples}:
            raise RuntimeError('Expression columns do not uniquely match all GEO library names')
        for row in reader:
            if len(row) != len(header):
                raise RuntimeError('Ragged expression CSV')
            original_id = row[0]
            identifier = re.sub(r'\.\d+$','',original_id) if original_id.startswith('ENSG') else original_id
            symbols = symbol_map.get(identifier,set())
            if not symbols:
                feature_audit.append({'original_id':original_id,'protein':'','status':'unmapped_identifier'})
                continue
            if len(symbols) != 1:
                feature_audit.append({'original_id':original_id,'protein':'','status':'ambiguous_excluded'})
                continue
            values = np.asarray([float(x) for x in row[1:]], dtype=float)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise RuntimeError('Normalized counts must be finite and nonnegative')
            transformed = np.log2(values + 1.0)
            symbol = next(iter(symbols))
            symbol_expression.setdefault(symbol, []).append(transformed)
            mapped = identifiers.get(identifier, set()) or identifiers.get(symbol,set())
            protein = next(iter(mapped)) if len(mapped)==1 else ''
            if protein:
                expression.setdefault(protein, []).append(transformed)
            feature_audit.append({'original_id':original_id,'protein':protein,'symbol':symbol,'status':'mapped'})
    expression = {p: np.mean(v,axis=0) for p,v in expression.items()}
    symbol_multiplicity = {p:len(v) for p,v in symbol_expression.items()}
    symbol_expression = {p:np.mean(v,axis=0) for p,v in symbol_expression.items()}
    longitudinal_write_tsv(out/'feature_mapping_audit.tsv',feature_audit,['original_id','protein','symbol','status'])
    longitudinal_write_tsv(out/'gene_feature_multiplicity.tsv',[{'symbol':p,'features':n} for p,n in sorted(symbol_multiplicity.items())],['symbol','features'])
    qc_rows = []
    if symbol_expression:
        matrix = np.asarray([symbol_expression[p] for p in sorted(symbol_expression)])
        metadata_by_library = {r['library_name']:r for r in samples}
        for i,library in enumerate(libraries):
            row = metadata_by_library[library]
            qc_rows.append({'library_name':library,'sample_id':row['sample_id'],'subject_id':row['subject_id'],
                            'mapped_genes':len(matrix),'nonzero_genes':int(np.sum(matrix[:,i]>0)),
                            'median_log2_count_plus1':float(np.median(matrix[:,i])),
                            'q95_log2_count_plus1':float(np.percentile(matrix[:,i],95)),
                            'sample_excluded':'no'})
        longitudinal_write_tsv(out/'sample_technical_qc.tsv',qc_rows,list(qc_rows[0]))
    if len(expression) < 200:
        longitudinal_json(out/'run_summary.json',{'status':'BLOCKED_FEATURE_MAPPING_OR_COVERAGE','mapped_candidates':len(expression),'required':200,'expression_values_read':True})
        raise RuntimeError('Candidate coverage below 200/250; inspect gene.ID mapping, do not guess transcript identifiers')
    column = {name:i for i,name in enumerate(libraries)}
    direction = {r['UniProt']:float(np.sign(float(r['OSA_beta']))) for r in annotations}
    contrasts = longitudinal_rows(design/'paired_contrasts.tsv')
    reversal = {}
    subject_node_rows = []
    for row in contrasts:
        if row['eligible'] != 'yes':
            continue
        pre = [column[x] for x in row['pre_libraries'].split('|')]
        post = [column[x] for x in row['post_libraries'].split('|')]
        values = {p:-direction[p]*float(np.mean(x[post])-np.mean(x[pre])) for p,x in expression.items()}
        reversal[(row['contrast'],row['subject_id'])] = values
        subject_node_rows.extend({'contrast':row['contrast'],'subject_id':row['subject_id'],'protein':p,'signed_reversal':value} for p,value in values.items())
    graph_rows = longitudinal_rows(design/'frozen_graphs.tsv')
    graphs = {}
    for row in graph_rows:
        graphs.setdefault((row['panel'],row['method'],int(row['replicate'])),[]).append((row['protein_A'],row['protein_B']))
    graph_coverage, scores, node_scores = [], {}, {}
    for key, all_edges in sorted(graphs.items()):
        edges = [(a,b) for a,b in all_edges if a in expression and b in expression]
        nodes = {p for e in edges for p in e}
        all_nodes = {p for e in all_edges for p in e}
        edge_cov = len(edges)/float(len(all_edges))
        node_cov = len(nodes)/float(len(all_nodes))
        eligible = (len(edges)>=contract['minimum_edges'] and len(nodes)>=contract['minimum_nodes']
                    and edge_cov>=contract['minimum_edge_coverage'] and node_cov>=contract['minimum_node_coverage'])
        # A degree-preserving null remains exact only with complete graph-node coverage.
        if key[1] == 'full_degree_preserving_null' and node_cov < 1.0:
            eligible = False
        graph_coverage.append({'panel':key[0],'method':key[1],'replicate':key[2],'frozen_edges':len(all_edges),
                               'mapped_edges':len(edges),'mapped_nodes':len(nodes),'edge_coverage':edge_cov,
                               'node_coverage':node_cov,'eligible':'yes' if eligible else 'no'})
        if not eligible:
            continue
        for (contrast,subject), values in reversal.items():
            scores[(key[0],key[1],key[2],contrast,subject)] = longitudinal_graph_score(np,edges,values)
            if key[2] == 0:
                node_scores[(key[0],key[1],contrast,subject)] = float(np.mean([values[p] for p in sorted(nodes)]))
    groups = {}
    for (panel,method,rep,contrast,subject), value in scores.items():
        if method == 'full_degree_preserving_null':
            groups.setdefault((panel,contrast,subject), []).append(value)
    for (panel,contrast,subject), values in groups.items():
        if len(values)>=contract['minimum_null_graphs']:
            scores[(panel,'degree_null_mean',0,contrast,subject)] = float(np.mean(values))
    rng = np.random.RandomState(contract['seed'])
    tests = []
    def add_test(panel,contrast,name,values,primary=False):
        subjects = sorted(values)
        test = {'panel':panel,'contrast':contrast,'comparison':name,'n_subjects':len(subjects),
                'analysis_role':'primary' if primary else 'secondary','status':'NOT_EVALUABLE',
                'effect':'','CI95_low':'','CI95_high':'','exact_signflip_P':'','BH_FDR_secondary':''}
        if len(subjects)>=contract['minimum_subjects']:
            vector = np.asarray([values[s] for s in subjects],dtype=float)
            effect,p = longitudinal_signflip(np,vector)
            means = np.mean(vector[rng.randint(len(vector),size=(contract['bootstrap_replicates'],len(vector)))],axis=1)
            lo,hi = np.percentile(means,[2.5,97.5])
            test.update(status='EVALUATED',effect=effect,CI95_low=float(lo),CI95_high=float(hi),exact_signflip_P=p)
        tests.append(test)
    panels = sorted({r['panel'] for r in longitudinal_rows(design/'frozen_graph_coverage.tsv')})
    contrast_names = sorted({r['contrast'] for r in contrasts})
    osa_subjects = sorted({r['subject_id'] for r in samples if r['disease']=='OSA'})
    comparison_methods = ['context_only','PPI_only','equal_L1_L2_L3','L3_only','degree_null_mean']
    difference_vectors = {}
    for panel in panels:
        for contrast in contrast_names:
            full = {s:scores[(panel,'full_TRIAGE_OSA',0,contrast,s)] for s in osa_subjects if (panel,'full_TRIAGE_OSA',0,contrast,s) in scores}
            add_test(panel,contrast,'full_molecular_response',full)
            for method in comparison_methods:
                delta = {s:full[s]-scores[(panel,method,0,contrast,s)] for s in full if (panel,method,0,contrast,s) in scores}
                difference_vectors[(panel,contrast,method)] = delta
                primary = panel=='ALL' and contrast=='month4_AM' and method=='context_only'
                add_test(panel,contrast,'full_minus_'+method,delta,primary)
    # Direct within-subject time interaction; significance differences are not interaction tests.
    for panel in panels:
        for clock in ('AM','PM','DAY_MEAN'):
            for method in ('context_only','degree_null_mean'):
                short = difference_vectors[(panel,'month4_'+clock,method)]
                long = difference_vectors[(panel,'month24_'+clock,method)]
                add_test(panel,'month24_minus_month4_'+clock,'full_minus_'+method+'_time_interaction',
                         {s:long[s]-short[s] for s in set(short)&set(long)})
    # All genes rather than just model candidates support this independent
    # pathway-level analysis. The subject, not the gene, is the testing unit.
    gene_sets = {}
    for row in longitudinal_rows(design/'pathway_gene_sets.tsv'):
        gene_sets.setdefault(row['axis'],set()).add(row['symbol'])
    gene_universe = set(symbol_expression)
    pathway_coverage, pathway_subject_rows = [], []
    for axis, genes in sorted(gene_sets.items()):
        selected, background = genes & gene_universe, gene_universe - genes
        fraction = len(selected)/float(len(genes))
        eligible = len(selected)>=contract['minimum_pathway_genes'] and fraction>=contract['minimum_pathway_coverage'] and bool(background)
        pathway_coverage.append({'axis':axis,'frozen_genes':len(genes),'mapped_genes':len(selected),
                                 'coverage':fraction,'eligible':'yes' if eligible else 'no'})
        activity = None
        if eligible:
            activity = np.mean([symbol_expression[x] for x in sorted(selected)],axis=0) - np.mean([symbol_expression[x] for x in sorted(background)],axis=0)
        axis_values = {}
        for contrast in contrast_names:
            values = {}
            if eligible:
                for row in contrasts:
                    if row['contrast']!=contrast or row['eligible']!='yes':
                        continue
                    pre = [column[x] for x in row['pre_libraries'].split('|')]
                    post = [column[x] for x in row['post_libraries'].split('|')]
                    value = float(np.mean(activity[post])-np.mean(activity[pre]))
                    values[row['subject_id']] = value
                    pathway_subject_rows.append({'axis':axis,'contrast':contrast,'subject_id':row['subject_id'],'competitive_log_change':value})
            axis_values[contrast] = values
            add_test(axis,contrast,'full_pathway_competitive_expression_change',values)
        for clock in ('AM','PM','DAY_MEAN'):
            short,long = axis_values['month4_'+clock],axis_values['month24_'+clock]
            add_test(axis,'month24_minus_month4_'+clock,'full_pathway_time_interaction',
                     {s:long[s]-short[s] for s in set(short)&set(long)})
    positions = [i for i,r in enumerate(tests) if r['status']=='EVALUATED' and r['analysis_role']=='secondary']
    order = sorted(positions,key=lambda i:tests[i]['exact_signflip_P'])
    running = 1.0
    for rank in range(len(order)-1,-1,-1):
        i = order[rank]
        running = min(running,tests[i]['exact_signflip_P']*len(order)/(rank+1.0))
        tests[i]['BH_FDR_secondary'] = running
    longitudinal_write_tsv(out/'endpoint_tests.tsv',tests,list(tests[0]))
    longitudinal_write_tsv(out/'pathway_mapping_coverage.tsv',pathway_coverage,list(pathway_coverage[0]))
    longitudinal_write_tsv(out/'subject_pathway_responses.tsv',pathway_subject_rows,['axis','contrast','subject_id','competitive_log_change'])
    longitudinal_write_tsv(out/'graph_mapping_coverage.tsv',graph_coverage,list(graph_coverage[0]))
    longitudinal_write_tsv(out/'subject_node_responses.tsv',subject_node_rows,['contrast','subject_id','protein','signed_reversal'])
    score_rows = [{'panel':k[0],'method':k[1],'replicate':k[2],'contrast':k[3],'subject_id':k[4],'edge_response':v,
                   'unique_node_mean_response':node_scores.get((k[0],k[1],k[3],k[4]),'')} for k,v in sorted(scores.items())]
    longitudinal_write_tsv(out/'subject_graph_responses.tsv',score_rows,['panel','method','replicate','contrast','subject_id','edge_response','unique_node_mean_response'])
    primary = next(r for r in tests if r['analysis_role']=='primary')
    summary = {'status':'PASS_FROZEN_LONGITUDINAL_EXECUTION' if primary['status']=='EVALUATED' else 'COMPLETE_PRIMARY_NOT_EVALUABLE',
               'created_at_utc':utc_now(),'mapped_candidates':len(expression),'primary':primary,
               'evaluated_tests':sum(r['status']=='EVALUATED' for r in tests),
               'not_evaluable_tests':sum(r['status']!='EVALUATED' for r in tests),
               'contract_sha256':file_sha256(contract_path),'expression_sha256':file_sha256(path),
               'python':sys.version,'numpy':np.__version__,'model_retrained':False,'ranking_changed':False,
               'clinical_prediction_claim':False,'physical_PPI_validation_claim':False,
               'caveats':contract['limitations'], 'bootstrap_CI':'subject percentile bootstrap, 5000 repeats; exploratory precision at small n'}
    longitudinal_json(out/'run_summary.json',summary)
    print('Longitudinal endpoint: '+summary['status'])
    print('Primary: '+repr(primary))
    print('Output: '+str(out))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-stage", default=DEFAULT_STAGE)
    parser.add_argument("--endpoint-output-stage", default=DEFAULT_ENDPOINT_STAGE)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--prepare-references", action="store_true")
    parser.add_argument("--endpoint", action="store_true")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    longitudinal_mode = parser.add_mutually_exclusive_group()
    longitudinal_mode.add_argument("--longitudinal-prepare", action="store_true", help="Download metadata and freeze new OSA graphs/paired design; no expression values")
    longitudinal_mode.add_argument("--longitudinal-endpoint", action="store_true", help="Run frozen GSE283703 paired module response tests; no training")
    longitudinal_mode.add_argument("--longitudinal-network-reconstruction", action="store_true", help="Post-hoc leave-one-node-out topology diagnostic using Stage 34; no training/reranking")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.longitudinal_network_reconstruction:
        if args.endpoint or args.prepare_references:
            parser.error("Longitudinal reconstruction cannot be mixed with legacy endpoint/reference modes")
        return run_longitudinal_network_reconstruction(args, root)
    if args.longitudinal_prepare or args.longitudinal_endpoint:
        if args.endpoint or args.prepare_references:
            parser.error("Longitudinal modes cannot be mixed with legacy endpoint/reference modes")
        if args.longitudinal_prepare:
            return prepare_longitudinal(args, root)
        try:
            return run_longitudinal(args, root)
        except Exception as exc:
            summary_path = root / 'results/discovery_preprocess_v2' / LONG_RESULT / 'run_summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                if summary.get('status') == 'RUNNING':
                    summary.update(status='FAILED_ENDPOINT_EXECUTION',error=str(exc))
                    longitudinal_json(summary_path,summary)
            raise
    if args.prepare_references:
        return prepare_reactome_reference(root, args.force)
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
    if args.endpoint:
        return run_independent_endpoint(args, root, np)
    required = {
        "lock": root / LOCK_REL,
        "endpoint_lock": root / ENDPOINT_LOCK_REL,
        "gse38792_raw": root / "geo/GSE38792_RAW.tar",
        "gse38792_matrix": root / "geo/GSE38792_series_matrix.txt.gz",
        "gse293433_xlsx": root / "geo/GSE293433_processed_data_mRNA.xlsx",
        "gse293433_matrix": root / "geo/GSE293433_series_matrix.txt.gz",
        "gpl6244": root / "annotations/GPL6244.annot.gz",
        "hgnc": root / "annotations/hgnc_complete_set_20260714.tsv",
        "primary": root / "results/discovery_preprocess_v2/04_candidates_v1/candidate_proteins_primary_250.tsv",
        "sensitivity": root / "results/discovery_preprocess_v2/04_candidates_v1/candidate_proteins_sensitivity_500.tsv",
        "geo_audit": root / "audit/osa_geo_audit_20260808-233455.json",
        "reactome_gmt": root / "annotations/independent_validation/ReactomePathways.gmt",
        "reference_manifest": root / "annotations/independent_validation/reference_manifest.json",
        "v4_handoff": root / V4_HANDOFF_REL,
        "primary_ranking": root / V4_RANKING_REL,
    }
    for path in required.values():
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    lock = json.loads(required["lock"].read_text(encoding="utf-8"))
    endpoint_lock = json.loads(required["endpoint_lock"].read_text(encoding="utf-8"))
    if endpoint_lock.get("status") != "PRE_RESULT_ENDPOINT_LOCK":
        raise RuntimeError("Independent endpoint lock is not pre-result frozen")
    hash_names = {
        "GSE38792_RAW.tar": "gse38792_raw",
        "GSE38792_series_matrix.txt.gz": "gse38792_matrix",
        "GSE293433_processed_data_mRNA.xlsx": "gse293433_xlsx",
        "GSE293433_series_matrix.txt.gz": "gse293433_matrix",
        "GPL6244.annot.gz": "gpl6244",
        "hgnc_complete_set_20260714.tsv": "hgnc",
        "candidate_proteins_primary_250.tsv": "primary",
        "candidate_proteins_sensitivity_500.tsv": "sensitivity",
        "osa_geo_audit_20260808-233455.json": "geo_audit",
    }
    observed_hashes = {}
    for name, key in hash_names.items():
        observed_hashes[name] = sha256_file(required[key])
        if observed_hashes[name] != lock["input_hashes"][name]:
            raise RuntimeError("Frozen independent-validation hash mismatch: " + name)
    v4_handoff = json.loads(required["v4_handoff"].read_text(encoding="utf-8"))
    if v4_handoff.get("status") != "V4_FROZEN_FOR_EXTERNAL_VALIDATION":
        raise RuntimeError("V4 external-validation handoff is not frozen")
    endpoint_expected = {
        "preprocess_lock": endpoint_lock["frozen_inputs"]["preprocess_lock"]["sha256"],
        "reactome_gmt": endpoint_lock["frozen_inputs"]["reactome_gmt"]["sha256"],
        "reference_manifest": endpoint_lock["frozen_inputs"]["reactome_manifest"]["sha256"],
        "primary_ranking": v4_handoff["ranking"]["primary_250_sha256"],
    }
    endpoint_observed = {
        "preprocess_lock": sha256_file(required["lock"]),
        "reactome_gmt": sha256_file(required["reactome_gmt"]),
        "reference_manifest": sha256_file(required["reference_manifest"]),
        "primary_ranking": sha256_file(required["primary_ranking"]),
    }
    for name in sorted(endpoint_expected):
        if endpoint_observed[name] != endpoint_expected[name]:
            raise RuntimeError("Frozen endpoint input hash mismatch: " + name)
    audit = json.loads(required["geo_audit"].read_text(encoding="utf-8"))
    if audit.get("summary") != {"selected": 4, "passed": 4, "failed": 0}:
        raise RuntimeError("Independent GEO audit is not a consolidated four-file PASS")
    candidates_250, _ = read_tsv(required["primary"])
    candidates_500, _ = read_tsv(required["sensitivity"])
    if len(candidates_250) != 250 or len(candidates_500) != 500:
        raise RuntimeError("Frozen candidate universe mismatch")
    print("OSA independent validation environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("Datasets: GSE38792 8 control + 10 OSA; GSE293433 5 control + 5 OSA")
    print("Endpoint/reference/ranking hash gate: PASS")
    print("Model: {}".format(v4_handoff["model"]["name"]))
    print("V4 reanalysis role: locked exploratory model-update comparison")
    print("Ranking values/checkpoint/pathway outcomes/previous validation read: no")
    if args.check_env:
        return 0

    meta387 = parse_sample_metadata(required["gse38792_matrix"])
    ids387, samples387, matrix387 = parse_series_matrix(np, required["gse38792_matrix"])
    titles387 = meta387["!Sample_title"][0]
    accessions387 = meta387["!Sample_geo_accession"][0]
    if samples387 != accessions387 or len(samples387) != 18 or len(ids387) != 33297:
        raise RuntimeError("GSE38792 matrix/sample identity mismatch")
    groups387 = ["OSA" if "_OSA_" in title else "control" for title in titles387]
    if groups387.count("control") != 8 or groups387.count("OSA") != 10:
        raise RuntimeError("GSE38792 group count mismatch")
    entrez387, expression387, _, mapping387 = map_and_aggregate(
        np, "GSE38792", ids387, matrix387, required["gpl6244"]
    )
    fit387 = fit_two_group(np, expression387, groups387)
    stats387 = statistics_rows(fmt, entrez387, "Entrez", expression387, fit387)
    by_entrez387 = {row["Entrez"]: row for row in stats387}

    meta293 = parse_sample_metadata(required["gse293433_matrix"])
    titles293 = meta293["!Sample_title"][0]
    accessions293 = meta293["!Sample_geo_accession"][0]
    descriptions293 = meta293.get("!Sample_description", [[]])[0]
    libraries293 = [value.split(":", 1)[1].strip() for value in descriptions293]
    if len(libraries293) != 10 or len(set(libraries293)) != 10:
        raise RuntimeError("GSE293433 library-name metadata mismatch")
    ids293_all, workbook_samples, tpm_all, malformed, workbook_rows = parse_tpm_xlsx(
        np, required["gse293433_xlsx"]
    )
    if set(workbook_samples) != set(libraries293):
        raise RuntimeError("GSE293433 workbook/metadata library mismatch")
    reorder = [workbook_samples.index(value) for value in libraries293]
    tpm_all = tpm_all[:, reorder]
    groups293 = ["OSA" if value.startswith("S") else "control" for value in libraries293]
    if groups293.count("control") != 5 or groups293.count("OSA") != 5:
        raise RuntimeError("GSE293433 group count mismatch")
    primary_filter = np.sum(tpm_all >= 1.0, axis=1) >= 2
    sensitivity_low = np.sum(tpm_all >= 0.1, axis=1) >= 2
    sensitivity_strict = np.sum(tpm_all >= 1.0, axis=1) >= 3
    ids293 = [value for value, keep in zip(ids293_all, primary_filter) if bool(keep)]
    tpm293 = tpm_all[primary_filter, :]
    expression293 = np.log2(tpm293 + 1.0)
    fit293 = fit_two_group(np, expression293, groups293)
    stats293 = statistics_rows(fmt, ids293, "Ensembl_gene", expression293, fit293)
    by_ensembl293 = {row["Ensembl_gene"]: row for row in stats293}
    unfiltered293 = set(ids293_all)

    manifest = []
    meta_qc387 = {}
    for index, sample in enumerate(samples387):
        subject = titles387[index].replace("Visceral fat_", "")
        row = {
            "dataset": "GSE38792", "sample_id": sample,
            "library_name": sample, "subject_id": subject,
            "tissue": "visceral_adipose", "disease_group": groups387[index],
            "platform": "GPL6244", "assay": "microarray_RMA_log2",
            "role": "untouched_independent_primary_validation",
            "candidate_selection_allowed": "no", "primary_include": "yes",
            "source_title": titles387[index],
        }
        manifest.append(row)
        meta_qc387[sample] = {"subject_id": subject, "treatment": "none"}
    meta_qc293 = {}
    for index, library in enumerate(libraries293):
        row = {
            "dataset": "GSE293433", "sample_id": accessions293[index],
            "library_name": library, "subject_id": library,
            "tissue": "PBMC", "disease_group": groups293[index],
            "platform": "GPL24676", "assay": "RNAseq_TPM_log2_plus_1",
            "role": "untouched_independent_secondary_validation",
            "candidate_selection_allowed": "no", "primary_include": "yes",
            "source_title": titles293[index],
        }
        manifest.append(row)
        meta_qc293[library] = {"subject_id": library, "treatment": "none"}
    qc387 = sample_qc(np, expression387, samples387, meta_qc387)
    qc293 = sample_qc(np, expression293, libraries293, meta_qc293)
    for row, group in zip(qc387, groups387):
        row["dataset"] = "GSE38792"
        row["disease_group"] = group
    for row, group in zip(qc293, groups293):
        row["dataset"] = "GSE293433"
        row["disease_group"] = group

    candidate_rows = []
    primary_accessions = set(row["UniProt_accession"] for row in candidates_250)
    for candidate in candidates_500:
        source387 = by_entrez387.get(candidate["Entrez"])
        ensembl = candidate.get("Ensembl_gene", "")
        source293 = by_ensembl293.get(ensembl)
        row = {
            "UniProt_accession": candidate["UniProt_accession"],
            "gene_symbol": candidate["gene_symbol_current"],
            "Entrez": candidate["Entrez"], "Ensembl_gene": ensembl,
            "discovery_rank": candidate["discovery_rank"],
            "primary_250": "yes" if candidate["UniProt_accession"] in primary_accessions else "no",
            "GSE38792_available": "yes" if source387 is not None else "no",
            "GSE293433_unfiltered_available": "yes" if ensembl in unfiltered293 else "no",
            "GSE293433_primary_filter_available": "yes" if source293 is not None else "no",
        }
        for prefix, source in [("GSE38792", source387), ("GSE293433", source293)]:
            for field in [
                "beta_OSA_minus_control", "unmoderated_SE", "unmoderated_t",
                "unmoderated_P_value", "unmoderated_FDR_BH", "moderated_SE",
                "moderated_t", "P_value", "FDR_BH",
            ]:
                row[prefix + "_" + field] = source[field] if source is not None else ""
        candidate_rows.append(row)

    output_dir = root / "results" / "discovery_preprocess_v2" / args.output_stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace")
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "independent_sample_manifest.tsv",
        "qc": output_dir / "sample_technical_qc.tsv",
        "malformed": output_dir / "GSE293433_excluded_workbook_rows.tsv",
        "gse38792": output_dir / "GSE38792_differential_all_entrez.tsv.gz",
        "gse293433": output_dir / "GSE293433_differential_filtered_ensembl.tsv.gz",
        "candidate": output_dir / "candidate_independent_evidence_500.tsv",
    }
    write_tsv(paths["manifest"], manifest, list(manifest[0]))
    write_tsv(paths["qc"], qc387 + qc293, list((qc387 + qc293)[0]))
    write_tsv(paths["malformed"], malformed, list(malformed[0]))
    write_gzip_tsv(paths["gse38792"], stats387, list(stats387[0]))
    write_gzip_tsv(paths["gse293433"], stats293, list(stats293[0]))
    write_tsv(paths["candidate"], candidate_rows, list(candidate_rows[0]))
    coverage = {}
    for scope, selected in [
        ("primary_250", [r for r in candidate_rows if r["primary_250"] == "yes"]),
        ("sensitivity_500", candidate_rows),
    ]:
        coverage[scope] = {
            "total": len(selected),
            "GSE38792": sum(r["GSE38792_available"] == "yes" for r in selected),
            "GSE293433_unfiltered": sum(r["GSE293433_unfiltered_available"] == "yes" for r in selected),
            "GSE293433_primary_filter": sum(r["GSE293433_primary_filter_available"] == "yes" for r in selected),
        }
    summary = {
        "created_at_utc": utc_now(), "status": "PASS_INDEPENDENT_PREPROCESS",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "preprocess_lock": {"path": LOCK_REL, "sha256": sha256_file(required["lock"])},
        "endpoint_lock": {"path": ENDPOINT_LOCK_REL, "sha256": sha256_file(required["endpoint_lock"])},
        "endpoint_input_hashes": endpoint_observed,
        "input_hashes": observed_hashes, "python": sys.version, "numpy": np.__version__,
        "GSE38792": {
            "input_features": len(ids387), "samples": len(samples387),
            "controls": fit387["n_control"], "OSA": fit387["n_OSA"],
            "matrix_range": [float(np.min(matrix387)), float(np.max(matrix387))],
            "mapped_genes": len(entrez387), "mapping_audit": mapping387,
            "fit": fit387["fit"],
        },
        "GSE293433": {
            "workbook_rows": workbook_rows, "valid_numeric_Ensembl_rows": len(ids293_all),
            "excluded_malformed_footer_rows": len(malformed),
            "workbook_order": workbook_samples, "metadata_order": libraries293,
            "alignment": "library_name", "samples": len(libraries293),
            "controls": fit293["n_control"], "OSA": fit293["n_OSA"],
            "TPM_range": [float(np.min(tpm_all)), float(np.max(tpm_all))],
            "primary_filter_genes": int(np.sum(primary_filter)),
            "sensitivity_TPM_0.1_in_2_genes": int(np.sum(sensitivity_low)),
            "sensitivity_TPM_1_in_3_genes": int(np.sum(sensitivity_strict)),
            "fit": fit293["fit"],
        },
        "candidate_coverage": coverage,
        "technical_outlier_flags": sum(
            row["technical_outlier_flag_abs_z_gt_5"] == "yes" for row in qc387 + qc293
        ),
        "primary_sample_exclusions": 0,
        "ranking_content_read": False, "ranking_hash_verified": True,
        "checkpoint_read": False,
        "known_PPI_or_pathway_read": False,
        "previous_validation_outcomes_read": False,
        "candidate_membership_changed": False,
        "next_gate": "Run the already frozen Stage 21 independent endpoint",
    }
    summary_path = output_dir / "run_summary.json"
    atomic_json(summary_path, summary)
    output_paths = list(paths.values()) + [summary_path]
    output_manifest = [{
        "path": relative_path(path, root), "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in output_paths]
    write_tsv(
        output_dir / "output_manifest.tsv", output_manifest,
        ["path", "bytes", "sha256"],
    )
    print("OSA independent validation preprocessing: PASS")
    print("GSE38792 mapped genes: {}; candidate coverage primary={}/250 sensitivity={}/500".format(
        len(entrez387), coverage["primary_250"]["GSE38792"],
        coverage["sensitivity_500"]["GSE38792"],
    ))
    print("GSE293433 retained genes: {}; candidate coverage primary={}/250 sensitivity={}/500".format(
        int(np.sum(primary_filter)), coverage["primary_250"]["GSE293433_primary_filter"],
        coverage["sensitivity_500"]["GSE293433_primary_filter"],
    ))
    print("Output: {}".format(output_dir))
    print("Ranking values/checkpoint/pathway outcomes/previous validation read: no")
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
