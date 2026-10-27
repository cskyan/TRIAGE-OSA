#!/usr/bin/env python3
"""Prepare independent IntAct and Complex Portal references for OSA Stage 19.

Python 3.7 compatible.  IntAct is queried only for the 250 frozen OSA
proteins through the official PSICQUIC service, avoiding a >800 MB whole
database download.  Complex Portal human ComplexTab is frozen byte-for-byte.
All TUnA pairs are conservatively audited by exact sequence SHA-256 so pairs
seen anywhere in the model data can be excluded from the primary endpoint.

This script prepares labels only.  It never reads validation outcomes and it
does not change candidates, scores, or ranks.
"""

from __future__ import print_function

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.3.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LOCK_REL = "config/osa_known_ppi_complex_lock.json"
HANDOFF_REL = "config/triage_osa_external_validation_lock_v4.json"
RANKING_REL = (
    "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4/"
    "pair_context_scores_triage_osa_primary_250.tsv.gz"
)
RANKING_SHA256 = "1e87e8ecaaf6d4a70ad89086237bcdc84633e5029d7f30efb0a0ccaadf6b03d3"
DEFAULT_ANNOTATION_DIR = "annotations/network_sanity_triage_osa_v4"


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.upper().encode("ascii")).hexdigest()


def relative_path(root, path):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temp), str(path))


def write_tsv(path, rows, fields):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fields,
            extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temp), str(path))


def write_gzip_tsv(path, rows, fields):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, delimiter="\t", fieldnames=fields,
                    extrasaction="ignore", lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(str(temp), str(path))


def open_text(path):
    return gzip.open(str(path), "rt", encoding="utf-8-sig", newline="") \
        if path.suffix == ".gz" else path.open("r", encoding="utf-8-sig", newline="")


def read_tsv(path):
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_fasta(path):
    records = {}
    current = None
    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    sequence = "".join(chunks).upper()
                    if not sequence:
                        raise RuntimeError("Empty FASTA sequence: " + current)
                    records[current] = sequence
                current = line[1:].split()[0]
                chunks = []
            else:
                if current is None:
                    raise RuntimeError("FASTA sequence before header: " + str(path))
                chunks.append(line)
    if current is not None:
        sequence = "".join(chunks).upper()
        if not sequence:
            raise RuntimeError("Empty FASTA sequence: " + current)
        records[current] = sequence
    return records


def canonical_uniprot(value):
    value = value.strip()
    if value.startswith("uniprotkb:"):
        value = value.split(":", 1)[1]
    value = value.split("(", 1)[0].split("-", 1)[0].strip().upper()
    if re.match(r"^[A-Z0-9]{6,10}$", value) and not value.startswith("CHEBI"):
        return value
    return ""


def canonical_from_identifier_field(field):
    for item in field.split("|"):
        if item.startswith("uniprotkb:"):
            accession = canonical_uniprot(item)
            if accession:
                return accession
    return ""


def pair_id(a, b):
    return "__".join(sorted((a, b)))


def frozen_universe(ranking_path):
    candidates = set()
    count = 0
    ranks = set()
    with gzip.open(str(ranking_path), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = [
            "pair_id", "protein_A", "protein_B",
            "triage_osa_rank_primary_250",
        ]
        missing = [x for x in required if x not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("Frozen ranking lacks: " + ",".join(missing))
        for row in reader:
            count += 1
            a, b = row["protein_A"], row["protein_B"]
            if row["pair_id"] != pair_id(a, b):
                raise RuntimeError("Noncanonical pair_id in frozen ranking")
            candidates.update((a, b))
            ranks.add(int(float(row["triage_osa_rank_primary_250"])))
    if count != 31125 or len(candidates) != 250 or ranks != set(range(1, 31126)):
        raise RuntimeError("Frozen 250-protein/31,125-pair universe check failed")
    return sorted(candidates)


def download_file(url, destination, timeout, retries):
    temp = destination.with_name(destination.name + ".part")
    if temp.exists():
        temp.unlink()
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "OSA-TRIAGE-Fuse/1.0 research-audit"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                headers = dict(response.headers.items())
                final_url = response.geturl()
                with temp.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            os.replace(str(temp), str(destination))
            return {"requested_url": url, "final_url": final_url, "headers": headers}
        except Exception as error:
            last_error = error
            if temp.exists():
                temp.unlink()
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("Download failed after {} attempts: {} ({})".format(
        retries, url, last_error
    ))


def fetch_bytes(url, timeout, retries):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "OSA-TRIAGE-Fuse/1.0 research-audit"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), dict(response.headers.items()), response.geturl()
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("IntAct query failed: {} ({})".format(url, last_error))


def header_value(headers, name, default=""):
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return default


def intact_queries(candidates, lock, query_dir, download, force, timeout, retries):
    query_dir.mkdir(parents=True, exist_ok=True)
    template = lock["references"]["intact"]["query_url_template"]
    page_size = int(lock["references"]["intact"]["page_size"])
    audits = []
    for index, accession in enumerate(candidates, 1):
        destination = query_dir / (accession + ".mitab25.gz")
        if force and destination.exists():
            destination.unlink()
        if not destination.exists():
            if not download:
                raise RuntimeError(
                    "Missing IntAct query cache {}; rerun with --download".format(destination)
                )
            rows = []
            first = 0
            expected = None
            final_url = ""
            release_headers = {}
            while expected is None or first < expected:
                url = template.format(
                    accession=urllib.parse.quote(accession), first=first,
                    page_size=page_size,
                )
                payload, headers, final_url = fetch_bytes(url, timeout, retries)
                text = payload.decode("utf-8", errors="replace")
                page_rows = [line for line in text.splitlines() if line.strip()]
                if expected is None:
                    expected = int(header_value(
                        headers, "X-PSICQUIC-Count", str(len(page_rows))
                    ))
                    release_headers = headers
                rows.extend(page_rows)
                if not page_rows:
                    break
                first += len(page_rows)
            temp = destination.with_name(destination.name + ".tmp")
            with gzip.open(str(temp), "wt", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(row + "\n")
            os.replace(str(temp), str(destination))
            query_meta = {
                "accession": accession, "reported_count": expected,
                "downloaded_rows": len(rows), "final_url": final_url,
                "x_psicquic_impl_version": header_value(
                    release_headers, "X-PSICQUIC-Impl-Version", ""
                ),
            }
        else:
            with gzip.open(str(destination), "rt", encoding="utf-8") as handle:
                row_count = sum(1 for line in handle if line.strip())
            query_meta = {
                "accession": accession, "reported_count": row_count,
                "downloaded_rows": row_count, "final_url": "cached",
                "x_psicquic_impl_version": "cached",
            }
        query_meta.update({
            "path": str(destination), "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })
        audits.append(query_meta)
        if index % 10 == 0 or index == len(candidates):
            print("[IntAct] candidates {}/{}".format(index, len(candidates)), flush=True)
        if download:
            time.sleep(0.05)
    return audits


def parse_intact(candidates, query_dir, physical_types):
    candidate_set = set(candidates)
    edge = {}
    degree_all = {x: set() for x in candidates}
    degree_physical = {x: set() for x in candidates}
    parsed_rows = 0
    eligible_rows = 0
    for accession in candidates:
        path = query_dir / (accession + ".mitab25.gz")
        with gzip.open(str(path), "rt", encoding="utf-8", newline="") as handle:
            for raw in handle:
                fields = raw.rstrip("\n\r").split("\t")
                parsed_rows += 1
                if len(fields) < 15:
                    continue
                a = canonical_from_identifier_field(fields[0])
                b = canonical_from_identifier_field(fields[1])
                if not a or not b or a == b:
                    continue
                if "taxid:9606" not in fields[9] or "taxid:9606" not in fields[10]:
                    continue
                eligible_rows += 1
                interaction_types = sorted(set(re.findall(r"MI:\d+", fields[11])))
                is_physical = bool(set(interaction_types) & physical_types)
                if a in candidate_set:
                    degree_all[a].add(b)
                    if is_physical:
                        degree_physical[a].add(b)
                if b in candidate_set:
                    degree_all[b].add(a)
                    if is_physical:
                        degree_physical[b].add(a)
                if a not in candidate_set or b not in candidate_set:
                    continue
                pid = pair_id(a, b)
                item = edge.setdefault(pid, {
                    "pair_id": pid, "protein_A": min(a, b), "protein_B": max(a, b),
                    "intact_all_positive": "yes", "intact_physical": "no",
                    "interaction_type_mi": set(), "source_record_count": 0,
                })
                item["source_record_count"] += 1
                item["interaction_type_mi"].update(interaction_types)
                if is_physical:
                    item["intact_physical"] = "yes"
    rows = []
    for pid in sorted(edge):
        item = edge[pid]
        item["interaction_type_mi"] = "|".join(sorted(item["interaction_type_mi"]))
        rows.append(item)
    degrees = [{
        "UniProt_accession": accession,
        "intact_all_unique_human_protein_partners": len(degree_all[accession]),
        "intact_physical_unique_human_protein_partners": len(degree_physical[accession]),
    } for accession in candidates]
    return rows, degrees, parsed_rows, eligible_rows


def parse_complex_portal(path, candidates, primary_codes):
    candidate_set = set(candidates)
    pair_map = {}
    complex_rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = [
            "#Complex ac", "Recommended name", "Taxonomy identifier",
            "Identifiers (and stoichiometry) of molecules in complex", "Evidence Code",
        ]
        missing = [x for x in required if x not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("Complex Portal lacks columns: " + ",".join(missing))
        for row in reader:
            if row["Taxonomy identifier"].strip() != "9606":
                continue
            components = []
            for token in row["Identifiers (and stoichiometry) of molecules in complex"].split("|"):
                accession = canonical_uniprot(token)
                if accession:
                    components.append(accession)
            components = sorted(set(components))
            mapped = sorted(candidate_set.intersection(components))
            evidence_codes = sorted(set(re.findall(r"ECO:\d+", row["Evidence Code"])))
            experimental = bool(set(evidence_codes) & primary_codes)
            complex_rows.append({
                "complex_accession": row["#Complex ac"],
                "recommended_name": row["Recommended name"],
                "evidence_codes": "|".join(evidence_codes),
                "experimental_primary": "yes" if experimental else "no",
                "protein_components": len(components),
                "mapped_candidate_components": len(mapped),
                "mapped_candidate_accessions": "|".join(mapped),
            })
            for i in range(len(mapped)):
                for j in range(i + 1, len(mapped)):
                    a, b = mapped[i], mapped[j]
                    pid = pair_id(a, b)
                    item = pair_map.setdefault(pid, {
                        "pair_id": pid, "protein_A": a, "protein_B": b,
                        "complexportal_all_curated": "yes",
                        "complexportal_experimental": "no",
                        "complex_accessions": set(), "complex_names": set(),
                    })
                    item["complex_accessions"].add(row["#Complex ac"])
                    item["complex_names"].add(row["Recommended name"])
                    if experimental:
                        item["complexportal_experimental"] = "yes"
    pairs = []
    for pid in sorted(pair_map):
        item = pair_map[pid]
        item["complex_accessions"] = "|".join(sorted(item["complex_accessions"]))
        item["complex_names"] = "|".join(sorted(item["complex_names"]))
        pairs.append(item)
    return pairs, complex_rows


def first_column(fieldnames, candidates):
    lower = {x.lower(): x for x in fieldnames}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return ""


def training_overlap(candidates, osa_fasta_path, tuna_fourpack):
    tuna_fasta_path = tuna_fourpack / "tuna.seq.fasta"
    shapes_path = tuna_fourpack / "tuna.shapes.tsv"
    for path in (osa_fasta_path, tuna_fasta_path, shapes_path):
        if not path.is_file():
            raise RuntimeError("Missing training-overlap input: " + str(path))
    osa = parse_fasta(osa_fasta_path)
    tuna = parse_fasta(tuna_fasta_path)
    missing_osa = [x for x in candidates if x not in osa]
    if missing_osa:
        raise RuntimeError("OSA FASTA lacks frozen accessions: " + ",".join(missing_osa[:10]))
    tuna_by_hash = {}
    for uid, sequence in tuna.items():
        tuna_by_hash.setdefault(sequence_sha256(sequence), set()).add(uid)
    osa_to_tuna = {}
    tuna_to_osa = {}
    mapping_rows = []
    for accession in candidates:
        digest = sequence_sha256(osa[accession])
        reasons = {}
        for uid in tuna_by_hash.get(digest, set()):
            reasons.setdefault(uid, set()).add("exact_sequence_sha256")
        for uid in tuna:
            identifier_tokens = set(re.split(r"[|:_]", uid.upper()))
            if accession.upper() == uid.upper() or accession.upper() in identifier_tokens:
                reasons.setdefault(uid, set()).add("direct_fasta_identifier")
        candidate_sequence = osa[accession]
        for uid, tuna_sequence in tuna.items():
            if min(len(candidate_sequence), len(tuna_sequence)) < 40:
                continue
            if tuna_sequence in candidate_sequence or candidate_sequence in tuna_sequence:
                reasons.setdefault(uid, set()).add("exact_sequence_containment_ge40aa")
        matches = sorted(reasons)
        osa_to_tuna[accession] = set(matches)
        for uid in matches:
            tuna_to_osa.setdefault(uid, set()).add(accession)
        mapping_rows.append({
            "UniProt_accession": accession, "sequence_sha256": digest,
            "tuna_exact_match_count": len(matches),
            "tuna_exact_match_uids": "|".join(matches),
            "tuna_match_reasons": "|".join(
                "{}:{}".format(uid, ",".join(sorted(reasons[uid]))) for uid in matches
            ),
            "ambiguous_tuna_match": "yes" if len(matches) > 1 else "no",
        })
    overlaps = {}
    with shapes_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        a_col = first_column(fields, ["A_uid", "protein_A", "uidA", "a", "protA"])
        b_col = first_column(fields, ["B_uid", "protein_B", "uidB", "b", "protB"])
        id_col = first_column(fields, ["pair_id", "id", "pair"])
        if not a_col or not b_col:
            raise RuntimeError("Cannot identify TUnA pair columns: " + ",".join(fields))
        for index, row in enumerate(reader, 1):
            osa_a = tuna_to_osa.get(row[a_col], set())
            osa_b = tuna_to_osa.get(row[b_col], set())
            for a in osa_a:
                for b in osa_b:
                    if a == b:
                        continue
                    pid = pair_id(a, b)
                    item = overlaps.setdefault(pid, {
                        "pair_id": pid, "protein_A": min(a, b), "protein_B": max(a, b),
                        "tuna_source_pair_count": 0, "tuna_source_pair_ids": set(),
                    })
                    item["tuna_source_pair_count"] += 1
                    item["tuna_source_pair_ids"].add(row[id_col] if id_col else str(index))
    overlap_rows = []
    for pid in sorted(overlaps):
        item = overlaps[pid]
        item["tuna_source_pair_ids"] = "|".join(sorted(item["tuna_source_pair_ids"]))
        overlap_rows.append(item)
    return mapping_rows, overlap_rows, len(tuna)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--triage-root", default='resources/backbone')
    parser.add_argument("--tuna-fourpack", default="")
    parser.add_argument("--osa-fasta", default="")
    parser.add_argument("--annotation-dir", default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    lock_path = root / LOCK_REL
    handoff_path = root / HANDOFF_REL
    ranking_path = root / RANKING_REL
    annotation_dir = Path(args.annotation_dir)
    if not annotation_dir.is_absolute():
        annotation_dir = root / annotation_dir
    triage_root = Path(args.triage_root).resolve()
    tuna_fourpack = Path(args.tuna_fourpack).resolve() if args.tuna_fourpack else (
        triage_root / "pair_data" / "TUnA_Intra0" / "fourpack"
    )
    osa_fasta = Path(args.osa_fasta).resolve() if args.osa_fasta else (
        root / "model_assets" / "osa_frozen_750_full.fasta"
    )
    if not lock_path.is_file() or not handoff_path.is_file() or not ranking_path.is_file():
        raise RuntimeError("Missing lock or frozen Stage 9 ranking")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff["ranking"]["primary_250"] != RANKING_REL:
        raise RuntimeError("TRIAGE-OSA handoff ranking path mismatch")
    if handoff["ranking"]["primary_250_sha256"] != RANKING_SHA256:
        raise RuntimeError("TRIAGE-OSA handoff ranking hash mismatch")
    if handoff["prespecified_endpoint_locks"]["known_ppi_complex"]["sha256"] != sha256_file(lock_path):
        raise RuntimeError("TRIAGE-OSA handoff known-PPI lock mismatch")
    if sha256_file(ranking_path) != RANKING_SHA256:
        raise RuntimeError("Frozen ranking hash mismatch")
    candidates = frozen_universe(ranking_path)
    print("OSA known-PPI reference preparation environment: PASS")
    print("Python: {}".format(sys.version))
    print("Frozen candidates: {}".format(len(candidates)))
    print("Annotation directory: {}".format(annotation_dir))
    print("TUnA fourpack: {}".format(tuna_fourpack))
    print("OSA FASTA: {}".format(osa_fasta))
    print("Validation outcomes read: no")
    if args.check_env:
        return 0
    if args.timeout <= 0 or args.retries <= 0:
        raise RuntimeError("timeout and retries must be positive")
    if annotation_dir.exists() and args.force:
        shutil.rmtree(str(annotation_dir))
    annotation_dir.mkdir(parents=True, exist_ok=True)

    complex_path = annotation_dir / "complexportal_9606.tsv"
    complex_download = {"requested_url": "cached", "final_url": "cached", "headers": {}}
    if not complex_path.exists():
        if not args.download:
            raise RuntimeError("Missing Complex Portal file; rerun with --download")
        complex_download = download_file(
            lock["references"]["complex_portal"]["url"], complex_path,
            args.timeout, args.retries,
        )
    query_dir = annotation_dir / "intact_queries"
    audits = intact_queries(
        candidates, lock, query_dir, args.download, False,
        args.timeout, args.retries,
    )
    physical_types = set(lock["references"]["intact"]["primary_interaction_type_mi"])
    intact_pairs, degrees, parsed_rows, eligible_rows = parse_intact(
        candidates, query_dir, physical_types
    )
    complex_pairs, complex_rows = parse_complex_portal(
        complex_path, candidates,
        set(lock["references"]["complex_portal"]["primary_evidence_codes"]),
    )
    sequence_map, overlap_rows, tuna_proteins = training_overlap(
        candidates, osa_fasta, tuna_fourpack
    )

    outputs = {
        "intact_pairs": annotation_dir / "intact_human_candidate_pairs.tsv.gz",
        "intact_degree": annotation_dir / "intact_candidate_degree.tsv",
        "complex_pairs": annotation_dir / "complexportal_candidate_pairs.tsv.gz",
        "complexes": annotation_dir / "complexportal_candidate_complexes.tsv",
        "sequence_map": annotation_dir / "tuna_sequence_overlap_map.tsv",
        "training_overlap": annotation_dir / "tuna_pair_overlap.tsv.gz",
        "query_audit": annotation_dir / "intact_query_audit.tsv",
    }
    write_gzip_tsv(outputs["intact_pairs"], intact_pairs, [
        "pair_id", "protein_A", "protein_B", "intact_all_positive",
        "intact_physical", "interaction_type_mi", "source_record_count",
    ])
    write_tsv(outputs["intact_degree"], degrees, list(degrees[0]))
    write_gzip_tsv(outputs["complex_pairs"], complex_pairs, [
        "pair_id", "protein_A", "protein_B", "complexportal_all_curated",
        "complexportal_experimental", "complex_accessions", "complex_names",
    ])
    write_tsv(outputs["complexes"], complex_rows, list(complex_rows[0]))
    write_tsv(outputs["sequence_map"], sequence_map, list(sequence_map[0]))
    write_gzip_tsv(outputs["training_overlap"], overlap_rows, [
        "pair_id", "protein_A", "protein_B", "tuna_source_pair_count",
        "tuna_source_pair_ids",
    ])
    audit_rows = []
    for item in audits:
        audit_rows.append({
            "accession": item["accession"],
            "reported_count": item["reported_count"],
            "downloaded_rows": item["downloaded_rows"],
            "path": relative_path(root, Path(item["path"])),
            "bytes": item["bytes"], "sha256": item["sha256"],
            "final_url": item["final_url"],
            "x_psicquic_impl_version": item["x_psicquic_impl_version"],
        })
    write_tsv(outputs["query_audit"], audit_rows, list(audit_rows[0]))

    artifact_paths = [complex_path] + list(outputs.values())
    manifest_rows = [{
        "path": relative_path(root, path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in artifact_paths]
    write_tsv(annotation_dir / "output_manifest.tsv", manifest_rows, list(manifest_rows[0]))
    summary = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "script": relative_path(root, Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": utc_now(),
        "lock": {"path": LOCK_REL, "sha256": sha256_file(lock_path)},
        "external_validation_handoff": {
            "path": HANDOFF_REL, "sha256": sha256_file(handoff_path),
        },
        "frozen_ranking_sha256": sha256_file(ranking_path),
        "candidates": len(candidates),
        "intact_query_rows": parsed_rows,
        "intact_eligible_human_protein_rows": eligible_rows,
        "intact_candidate_pairs_all": len(intact_pairs),
        "intact_candidate_pairs_physical": sum(x["intact_physical"] == "yes" for x in intact_pairs),
        "complexportal_complexes": len(complex_rows),
        "complexportal_candidate_pairs_all": len(complex_pairs),
        "complexportal_candidate_pairs_experimental": sum(x["complexportal_experimental"] == "yes" for x in complex_pairs),
        "tuna_proteins": tuna_proteins,
        "candidate_sequences_matching_tuna": sum(int(x["tuna_exact_match_count"]) > 0 for x in sequence_map),
        "candidate_pairs_overlapping_tuna": len(overlap_rows),
        "complex_portal_download": complex_download,
        "candidate_or_pair_membership_changed": False,
        "validation_outcomes_read": False,
        "reference_labels_used_for_ranking_or_selection": False,
        "artifact_sha256": {
            relative_path(root, path): sha256_file(path) for path in artifact_paths
        },
    }
    atomic_json(annotation_dir / "reference_manifest.json", summary)
    print("OSA known-PPI reference preparation: PASS")
    print("IntAct candidate pairs: all={} physical={}".format(
        len(intact_pairs), sum(x["intact_physical"] == "yes" for x in intact_pairs)
    ))
    print("Complex Portal candidate pairs: all={} experimental={}".format(
        len(complex_pairs), sum(x["complexportal_experimental"] == "yes" for x in complex_pairs)
    ))
    print("TUnA-overlap candidate pairs: {}".format(len(overlap_rows)))
    print("Output: {}".format(annotation_dir))
    print("Validation outcomes read: no; frozen ranking changed: no")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
