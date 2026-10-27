#!/usr/bin/env python3
"""Build frozen unordered OSA protein-pair registries.

Python 3.7+.  The registries are derived only from the already frozen
discovery candidate tables.  No model output, known PPI, pathway or validation
cohort is read.  The sensitivity registry begins with the exact primary block,
which permits checkpoint-identical primary results to be reused later.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.0.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_N = 250
SENSITIVITY_N = 500
PRIMARY_PAIRS = PRIMARY_N * (PRIMARY_N - 1) // 2
SENSITIVITY_PAIRS = SENSITIVITY_N * (SENSITIVITY_N - 1) // 2
FIELDS = [
    "pair_index", "pair_id", "protein_A", "protein_B", "gene_A", "gene_B",
    "discovery_rank_A", "discovery_rank_B", "length_A", "length_B",
    "primary_pair", "sensitivity_pair", "registry_block",
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


def sha256_lines(lines):
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_candidates(path, expected):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "discovery_rank", "UniProt_accession", "gene_symbol_current",
        "UniProt_length",
    }
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise RuntimeError("Missing candidate columns {} in {}".format(missing, path))
    if len(rows) != expected:
        raise RuntimeError(
            "Expected {} candidates in {}, observed {}".format(expected, path, len(rows))
        )
    normalized = []
    seen = set()
    for row in rows:
        accession = str(row["UniProt_accession"]).strip()
        if not accession or accession in seen:
            raise RuntimeError("Blank or duplicate UniProt accession: " + accession)
        seen.add(accession)
        normalized.append({
            "accession": accession,
            "gene": str(row["gene_symbol_current"]).strip(),
            "rank": int(float(row["discovery_rank"])),
            "length": int(float(row["UniProt_length"])),
        })
    normalized.sort(key=lambda item: (item["rank"], item["accession"]))
    ranks = [item["rank"] for item in normalized]
    if len(set(ranks)) != expected:
        raise RuntimeError("Discovery ranks are not unique in " + str(path))
    return normalized


def oriented_pair(first, second):
    return (first, second) if first["accession"] < second["accession"] else (second, first)


def pair_row(first, second, index, block, primary_ids):
    a, b = oriented_pair(first, second)
    pair_id = a["accession"] + "__" + b["accession"]
    return {
        "pair_index": str(index),
        "pair_id": pair_id,
        "protein_A": a["accession"],
        "protein_B": b["accession"],
        "gene_A": a["gene"],
        "gene_B": b["gene"],
        "discovery_rank_A": str(a["rank"]),
        "discovery_rank_B": str(b["rank"]),
        "length_A": str(a["length"]),
        "length_B": str(b["length"]),
        "primary_pair": "yes" if a["accession"] in primary_ids and b["accession"] in primary_ids else "no",
        "sensitivity_pair": "yes",
        "registry_block": block,
    }


def primary_rows(candidates):
    ids = {item["accession"] for item in candidates}
    index = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            index += 1
            yield pair_row(candidates[i], candidates[j], index, "primary_250", ids)


def sensitivity_rows(candidates, primary_count=PRIMARY_N):
    primary = candidates[:primary_count]
    primary_ids = {item["accession"] for item in primary}
    index = 0
    for row in primary_rows(primary):
        index += 1
        row["pair_index"] = str(index)
        yield row
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if i < primary_count and j < primary_count:
                continue
            index += 1
            yield pair_row(
                candidates[i], candidates[j], index,
                "sensitivity_extension_251_500", primary_ids,
            )


def write_deterministic_gzip_tsv(path, rows):
    pair_ids = []
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                    pair_ids.append(row["pair_id"])
                    count += 1
    return count, sha256_lines(pair_ids), sha256_lines(sorted(pair_ids))


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative_path(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--candidate-stage", default="04_candidates_v1")
    parser.add_argument("--output-stage", default="07_pair_registry_v1")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    candidate_dir = root / "results" / "discovery_preprocess_v2" / args.candidate_stage
    primary_path = candidate_dir / "candidate_proteins_primary_250.tsv"
    sensitivity_path = candidate_dir / "candidate_proteins_sensitivity_500.tsv"
    audit_path = root / "results" / "discovery_preprocess_v2" / "05_model_assets_v1" / "model_asset_audit.json"
    for path in (primary_path, sensitivity_path, audit_path):
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_READY_FOR_PAIR_INFERENCE":
        raise RuntimeError("Model asset audit has not released pair inference")
    primary = read_candidates(primary_path, PRIMARY_N)
    sensitivity = read_candidates(sensitivity_path, SENSITIVITY_N)
    if [x["accession"] for x in primary] != [x["accession"] for x in sensitivity[:PRIMARY_N]]:
        raise RuntimeError("Primary candidates are not the first frozen 250 sensitivity candidates")
    print("OSA pair registry environment: PASS")
    print("Primary candidates: {}".format(len(primary)))
    print("Sensitivity candidates: {}".format(len(sensitivity)))
    print("Validation/model output/known PPI read: no")
    if args.check_env:
        return 0
    output_dir = root / "results" / "discovery_preprocess_v2" / args.output_stage
    if output_dir.exists():
        if not args.force:
            raise RuntimeError("Output exists; use --force to replace: " + str(output_dir))
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_registry = output_dir / "pair_registry_primary_250.tsv.gz"
    sensitivity_registry = output_dir / "pair_registry_sensitivity_500.tsv.gz"
    pc, po, pm = write_deterministic_gzip_tsv(primary_registry, primary_rows(primary))
    sc, so, sm = write_deterministic_gzip_tsv(sensitivity_registry, sensitivity_rows(sensitivity))
    if pc != PRIMARY_PAIRS or sc != SENSITIVITY_PAIRS:
        raise RuntimeError("Pair counts are incorrect: primary={} sensitivity={}".format(pc, sc))
    summary = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "candidate_stage": args.candidate_stage,
        "candidate_inputs": {
            "primary": {"path": relative_path(primary_path, root), "sha256": sha256_file(primary_path)},
            "sensitivity": {"path": relative_path(sensitivity_path, root), "sha256": sha256_file(sensitivity_path)},
        },
        "selection_inputs_read": "frozen discovery candidates only",
        "validation_data_read": False,
        "model_output_read": False,
        "known_PPI_or_pathway_read": False,
        "unordered": True,
        "self_pairs": False,
        "orientation": "lexicographic UniProt accession; inference is additionally A-B/B-A symmetric",
        "sensitivity_order": "exact primary block first, then pairs with at least one rank 251-500 protein",
        "registries": {
            "primary_250": {
                "path": relative_path(primary_registry, root), "pairs": pc,
                "ordered_pair_id_sha256": po, "membership_pair_id_sha256": pm,
                "file_sha256": sha256_file(primary_registry),
            },
            "sensitivity_500": {
                "path": relative_path(sensitivity_registry, root), "pairs": sc,
                "ordered_pair_id_sha256": so, "membership_pair_id_sha256": sm,
                "file_sha256": sha256_file(sensitivity_registry),
                "primary_prefix_pairs": PRIMARY_PAIRS,
            },
        },
    }
    summary_path = output_dir / "pair_registry_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    outputs = [primary_registry, sensitivity_registry, summary_path]
    manifest_rows = [
        {"path": path.relative_to(root).as_posix(), "bytes": str(path.stat().st_size), "sha256": sha256_file(path)}
        for path in outputs
    ]
    write_tsv(output_dir / "output_manifest.tsv", manifest_rows, ["path", "bytes", "sha256"])
    print("OSA pair registry: PASS")
    print("Primary pairs: {:,}".format(pc))
    print("Sensitivity pairs: {:,}".format(sc))
    print("Output: {}".format(output_dir))
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
