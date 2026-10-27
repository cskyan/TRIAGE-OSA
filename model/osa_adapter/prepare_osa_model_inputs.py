#!/usr/bin/env python3
"""Prepare frozen OSA protein sequences for TRIAGE inference.

This script reads only the already-frozen discovery candidates and preliminary
controls.  It retrieves canonical reviewed human UniProt sequences (or accepts
an offline FASTA), verifies accession and length, then writes both full-length
and checkpoint-native N-terminal input FASTA files.  It never reads validation
data and it does not change candidate membership.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
try:
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
except ImportError:  # pragma: no cover - Python 2 is not supported
    raise


SCRIPT_VERSION = "1.0.0"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
VALID_AA = re.compile(r"^[A-Z]+$")


def utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def dataframe_tsv_text(frame):
    """Support both pandas 1.1 (Python 3.7 server) and current pandas."""
    try:
        return frame.to_csv(sep="\t", index=False, lineterminator="\n")
    except TypeError:
        return frame.to_csv(sep="\t", index=False, line_terminator="\n")


def parse_accession(header):
    token = header.split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return token


def parse_fasta_text(text):
    sequences = {}
    accession = None
    chunks = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if accession is not None:
                sequences[accession] = "".join(chunks).upper()
            accession = parse_accession(line[1:])
            chunks = []
        else:
            chunks.append(re.sub(r"\s+", "", line))
    if accession is not None:
        sequences[accession] = "".join(chunks).upper()
    return sequences


def read_fasta(path):
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        return parse_fasta_text(handle.read())


def fasta_text(records, width=60):
    lines = []
    for accession in sorted(records):
        sequence = records[accession]
        lines.append(">{}".format(accession))
        lines.extend(sequence[i:i + width] for i in range(0, len(sequence), width))
    return "\n".join(lines) + "\n"


def chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def fetch_batch(accessions, timeout, retries):
    terms = ["accession:{}".format(accession) for accession in accessions]
    query = "({}) AND reviewed:true AND organism_id:9606".format(" OR ".join(terms))
    url = UNIPROT_STREAM_URL + "?" + urlencode({"format": "fasta", "query": query})
    request = Request(url, headers={
        "Accept": "text/plain",
        "User-Agent": "OSA-TRIAGE-input-preparation/{}".format(SCRIPT_VERSION),
    })
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            return parse_fasta_text(payload.decode("utf-8"))
        except (HTTPError, URLError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("UniProt request failed after {} attempts: {}".format(retries, last_error))


def load_entities(root, pd):
    candidate_dir = root / "results" / "discovery_preprocess_v2" / "04_candidates_v1"
    primary_path = candidate_dir / "candidate_proteins_primary_250.tsv"
    sensitivity_path = candidate_dir / "candidate_proteins_sensitivity_500.tsv"
    controls_path = candidate_dir / "matched_control_proteins_preliminary.tsv"
    ranking_path = candidate_dir / "protein_ranking_all.tsv.gz"
    for path in [primary_path, sensitivity_path, controls_path, ranking_path]:
        if not path.is_file():
            raise RuntimeError("Missing frozen input: " + str(path))
    primary = pd.read_csv(str(primary_path), sep="\t")
    sensitivity = pd.read_csv(str(sensitivity_path), sep="\t")
    controls = pd.read_csv(str(controls_path), sep="\t")
    ranking = pd.read_csv(str(ranking_path), sep="\t", compression="gzip")
    primary_ids = set(primary["UniProt_accession"].astype(str))
    sensitivity_ids = set(sensitivity["UniProt_accession"].astype(str))
    control_ids = set(controls["control_UniProt"].astype(str))
    if len(primary_ids) != 250 or len(sensitivity_ids) != 500 or len(control_ids) != 250:
        raise RuntimeError("Frozen candidate/control counts are not 250/500/250")
    if primary_ids - sensitivity_ids:
        raise RuntimeError("Primary 250 is not nested within sensitivity 500")
    if sensitivity_ids & control_ids:
        raise RuntimeError("Preliminary controls overlap frozen candidates")
    subset = ranking[ranking["UniProt_accession"].astype(str).isin(sensitivity_ids | control_ids)].copy()
    subset["UniProt_accession"] = subset["UniProt_accession"].astype(str)
    if subset["UniProt_accession"].nunique() != 750:
        raise RuntimeError("Could not recover exactly 750 unique proteins from frozen ranking")
    subset = subset.drop_duplicates("UniProt_accession").copy()
    subset["in_primary_250"] = subset["UniProt_accession"].isin(primary_ids)
    subset["in_sensitivity_500"] = subset["UniProt_accession"].isin(sensitivity_ids)
    subset["in_preliminary_control"] = subset["UniProt_accession"].isin(control_ids)
    return subset.sort_values("UniProt_accession").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare frozen OSA TRIAGE sequence inputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--offline-fasta", default="", help="Use a local FASTA instead of UniProt REST.")
    parser.add_argument("--native-max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required", file=sys.stderr)
        return 1
    root = Path(args.root).expanduser().resolve()
    entities = load_entities(root, pd)
    print("OSA TRIAGE sequence preparation environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("pandas: {}".format(pd.__version__))
    print("Frozen proteins: {}".format(entities.shape[0]))
    if args.check_env:
        return 0
    if args.native_max_len <= 0 or args.native_max_len > 1022:
        raise RuntimeError("--native-max-len must be in 1..1022")

    asset_dir = root / "model_assets"
    full_path = asset_dir / "osa_frozen_750_full.fasta"
    native_path = asset_dir / "osa_frozen_750_native{}.fasta".format(args.native_max_len)
    manifest_path = asset_dir / "osa_sequence_manifest.tsv"
    audit_path = asset_dir / "osa_sequence_preparation.json"
    outputs = [full_path, native_path, manifest_path, audit_path]
    if any(path.exists() for path in outputs) and not args.force:
        raise RuntimeError("Sequence outputs already exist; use --force to replace")

    wanted = entities["UniProt_accession"].tolist()
    if args.offline_fasta:
        source_path = Path(args.offline_fasta).expanduser().resolve()
        if not source_path.is_file():
            raise RuntimeError("Offline FASTA does not exist: " + str(source_path))
        all_sequences = read_fasta(source_path)
        source = str(source_path)
        source_sha256 = sha256_file(source_path)
    else:
        all_sequences = {}
        batches = list(chunks(wanted, int(args.batch_size)))
        for number, batch in enumerate(batches, 1):
            result = fetch_batch(batch, int(args.timeout), int(args.retries))
            duplicate = set(all_sequences) & set(result)
            if duplicate:
                raise RuntimeError("Duplicate accessions returned across UniProt batches: {}".format(sorted(duplicate)[:5]))
            all_sequences.update(result)
            print("[UniProt] batch {}/{} returned {} sequences".format(number, len(batches), len(result)), flush=True)
        source = UNIPROT_STREAM_URL
        source_sha256 = ""

    sequences = {}
    sequence_errors = []
    wanted_set = set(wanted)
    for accession in wanted:
        sequence = all_sequences.get(accession, "").upper()
        if not sequence:
            sequence_errors.append("missing:{}".format(accession))
            continue
        if not VALID_AA.match(sequence):
            sequence_errors.append("invalid_residue:{}".format(accession))
            continue
        sequences[accession] = sequence
    extra = sorted(set(all_sequences) - wanted_set)
    if sequence_errors:
        raise RuntimeError("Sequence retrieval failed: count={} examples={}".format(len(sequence_errors), sequence_errors[:10]))

    native_sequences = dict((key, value[:int(args.native_max_len)]) for key, value in sequences.items())
    full_text = fasta_text(sequences)
    native_text = fasta_text(native_sequences)
    atomic_write_text(full_path, full_text)
    atomic_write_text(native_path, native_text)

    entity_lookup = entities.set_index("UniProt_accession")
    rows = []
    for accession in wanted:
        sequence = sequences[accession]
        record = entity_lookup.loc[accession]
        expected_length = pd.to_numeric(record.get("UniProt_length"), errors="coerce")
        expected_value = int(expected_length) if pd.notna(expected_length) else 0
        observed = len(sequence)
        rows.append({
            "UniProt_accession": accession,
            "gene_symbol_current": record.get("gene_symbol_current", ""),
            "in_primary_250": "yes" if bool(record["in_primary_250"]) else "no",
            "in_sensitivity_500": "yes" if bool(record["in_sensitivity_500"]) else "no",
            "in_preliminary_control": "yes" if bool(record["in_preliminary_control"]) else "no",
            "expected_UniProt_length": expected_value if expected_value else "",
            "observed_sequence_length": observed,
            "length_matches_locked_annotation": "yes" if expected_value == observed else "no",
            "native_input_length": min(observed, int(args.native_max_len)),
            "explicitly_N_terminal_truncated": "yes" if observed > int(args.native_max_len) else "no",
            "sequence_sha256": sha256_bytes(sequence.encode("ascii")),
        })
    manifest = pd.DataFrame(rows)
    length_mismatch = int(manifest["length_matches_locked_annotation"].eq("no").sum())
    manifest_text = dataframe_tsv_text(manifest)
    atomic_write_text(manifest_path, manifest_text)

    audit = {
        "created_at_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "status": "PASS" if length_mismatch == 0 else "BLOCKED_LENGTH_DRIFT",
        "sequence_source": source,
        "sequence_source_sha256": source_sha256,
        "requested_proteins": len(wanted),
        "retrieved_proteins": len(sequences),
        "extra_source_records_ignored": len(extra),
        "length_mismatch_vs_locked_annotation": length_mismatch,
        "native_max_len": int(args.native_max_len),
        "explicitly_truncated_proteins": int(manifest["explicitly_N_terminal_truncated"].eq("yes").sum()),
        "full_fasta": {"path": str(full_path), "sha256": sha256_file(full_path)},
        "native_fasta": {"path": str(native_path), "sha256": sha256_file(native_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "validation_data_read": False,
        "candidate_membership_changed": False,
        "inference_performed": False,
        "policy_note": "Native primary input reproduces the checkpoint training convention using an explicit, audited N-terminal crop; multi-window inference remains a separately frozen sensitivity analysis.",
    }
    atomic_write_text(audit_path, json.dumps(audit, indent=2, sort_keys=False) + "\n")
    print("OSA TRIAGE sequence preparation: {}".format(audit["status"]))
    print("Full FASTA: {}".format(full_path))
    print("Native FASTA: {}".format(native_path))
    print("Manifest: {}".format(manifest_path))
    print("Length mismatches: {}".format(length_mismatch))
    print("No validation data was read and candidate membership was unchanged.")
    return 0 if length_mismatch == 0 else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
