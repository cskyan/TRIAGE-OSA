#!/usr/bin/env python3
"""Audit frozen TRIAGE checkpoints and OSA candidate feature availability.

Python 3.7+; numpy/pandas required, torch optional for strict checkpoint
inspection.  The script is read-only with respect to TRIAGE/new3 assets and
does not download sequences or generate embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_VERSION = "1.2.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]
SERVER_RUNTIME = MODEL_DIR / "triage_server_candidate"
BASELINE = MODEL_DIR / "triage_base"
RUNTIME = (
    SERVER_RUNTIME
    if (SERVER_RUNTIME / "source_manifest.json").is_file()
    else BASELINE
)


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


def json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError("Object of type {} is not JSON serializable".format(
        type(value).__name__
    ))


def require_packages() -> Tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("This audit requires numpy and pandas") from exc
    return np, pd


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "sep": "\t", "index": False, "na_rep": "NA",
        "float_format": "%.10g",
    }
    try:
        frame.to_csv(str(path), lineterminator="\n", **kwargs)
    except TypeError:
        frame.to_csv(str(path), line_terminator="\n", **kwargs)


def normalize_fasta_id(header: str) -> List[str]:
    first = header.strip().split()[0]
    values = [first]
    if "|" in first:
        values.extend(part for part in first.split("|") if part)
    return list(dict.fromkeys(values))


def read_fasta_ids(paths: Iterable[Path]) -> Set[str]:
    identifiers = set()  # type: Set[str]
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(">"):
                    identifiers.update(normalize_fasta_id(line[1:]))
    return identifiers


def table_identifier_set(path: Path, pd: Any) -> Set[str]:
    if not path.is_file():
        return set()
    header = pd.read_csv(str(path), sep="\t", nrows=0)
    lookup = {column.lower(): column for column in header.columns}
    column = None
    for name in ["uid", "chain_uid", "protein", "protein_id", "id"]:
        if name in lookup:
            column = lookup[name]
            break
    if column is None:
        column = header.columns[0]
    values = pd.read_csv(
        str(path), sep="\t", usecols=[column], dtype=str,
        keep_default_na=False,
    )[column]
    return set(values.astype(str))


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path.resolve()
    return None


def embedding_record(
    accession: str, directories: Sequence[Path], np: Any,
    check_finite: bool = True,
) -> Dict[str, Any]:
    for directory in directories:
        for stem in [accession, accession.upper(), accession.lower()]:
            path = directory / (stem + ".npy")
            if not path.is_file():
                continue
            try:
                array = np.load(str(path), mmap_mode="r")
                shape = tuple(int(value) for value in array.shape)
                finite = bool(np.isfinite(array).all()) if check_finite else True
            except Exception as exc:
                return {
                    "ESM_available": "invalid", "ESM_path": str(path),
                    "ESM_rows": 0, "ESM_dimension": 0,
                    "ESM_finite": "no", "ESM_error": str(exc),
                }
            return {
                "ESM_available": "yes" if len(shape) == 2 and finite else "invalid",
                "ESM_path": str(path.resolve()),
                "ESM_rows": shape[0] if len(shape) >= 1 else 0,
                "ESM_dimension": shape[1] if len(shape) >= 2 else 0,
                "ESM_finite": "yes" if finite else "no",
                "ESM_error": "" if finite else "non-finite values",
            }
    return {
        "ESM_available": "no", "ESM_path": "", "ESM_rows": 0,
        "ESM_dimension": 0, "ESM_finite": "no", "ESM_error": "",
    }


def coord_available(accession: str, directories: Sequence[Path]) -> Tuple[str, str]:
    for directory in directories:
        for stem in [accession, accession.upper(), accession.lower()]:
            path = directory / (stem + ".npy")
            if path.is_file():
                return "yes", str(path.resolve())
    return "no", ""


def source_baseline_audit(triage_root: Path) -> List[Dict[str, Any]]:
    manifest = load_json(RUNTIME / "source_manifest.json")
    rows = []
    source_paths = {
        "compute_esm.py": triage_root / "scripts" / "compute_esm.py",
    }
    for name, expected in manifest["files"].items():
        local_path = RUNTIME / name
        source_path = source_paths.get(name, triage_root / name)
        local_hash = sha256_file(local_path) if local_path.is_file() else ""
        source_hash = sha256_file(source_path) if source_path.is_file() else ""
        rows.append({
            "file": name, "expected_sha256": expected,
            "baseline_path": str(local_path.resolve()),
            "baseline_sha256": local_hash,
            "baseline_match": "yes" if local_hash == expected else "no",
            "source_path": str(source_path.resolve()),
            "source_exists": "yes" if source_path.is_file() else "no",
            "source_sha256": source_hash,
            "source_match": "yes" if source_hash == expected else "no",
        })
    return rows


def inspect_checkpoint(path: Optional[Path], device: str = "cpu") -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "MISSING", "path": "" if path is None else str(path)}
    result = {
        "status": "PRESENT_NOT_STRICTLY_INSPECTED",
        "path": str(path.resolve()), "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }  # type: Dict[str, Any]
    try:
        import torch
    except ImportError:
        result["torch_status"] = "not_installed"
        return result
    try:
        try:
            checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(path), map_location=device)
        if not isinstance(checkpoint, dict):
            result["status"] = "INVALID_NOT_DICTIONARY"
            return result
        state = checkpoint.get(
            "model_state_dict", checkpoint.get("model_state", checkpoint)
        )
        config = checkpoint.get("config", {})
        result.update({
            "status": "SCHEMA_READ",
            "stage": checkpoint.get("stage", ""),
            "epoch": checkpoint.get("epoch", ""),
            "best_metric": checkpoint.get("best_metric", ""),
            "state_key_count": len(state) if isinstance(state, dict) else 0,
            "config_field_count": len(config) if isinstance(config, dict) else 0,
            "d_res_in": config.get("d_res_in", "") if isinstance(config, dict) else "",
            "d_model": config.get("d_model", "") if isinstance(config, dict) else "",
            "max_pair_len": config.get("max_pair_len", "") if isinstance(config, dict) else "",
            "feature_spec": checkpoint.get("feature_spec", {}),
        })
        adapter_dir = Path(__file__).resolve().parent
        if str(adapter_dir) not in sys.path:
            sys.path.insert(0, str(adapter_dir))
        from osa_triage_adapter import (
            ADAPTER_VERSION, SymmetricTRIAGEAdapter, load_frozen_checkpoint,
        )
        model, cfg, strict_metadata = load_frozen_checkpoint(
            path, torch.device(device)
        )
        adapter = SymmetricTRIAGEAdapter(model, strict_metadata)
        smoke_a = torch.zeros(
            (1, 8, int(cfg.d_res_in)), dtype=torch.float32, device=device
        )
        smoke_b = torch.zeros(
            (1, 7, int(cfg.d_res_in)), dtype=torch.float32, device=device
        )
        smoke_out = adapter(
            resA=smoke_a,
            maskA=torch.ones((1, 8), dtype=torch.bool, device=device),
            coordsA=torch.full((1, 8, 3), float("nan"), device=device),
            resB=smoke_b,
            maskB=torch.ones((1, 7), dtype=torch.bool, device=device),
            coordsB=torch.full((1, 7, 3), float("nan"), device=device),
        )
        required_outputs = [
            "p_PPI", "p_L1", "p_L2", "p_L3", "S_interface",
            "topm_interface_indices", "g_L1", "g_L2", "g_L3",
            "T_MEB_reconstruction_error", "r_rel_diagnostic",
            "tri_level_consensus_diagnostic",
            "T_MEB_gate_entropy_normalized",
            "T_MEB_contribution_conflict",
            "order_stability_diagnostic",
            "L1_L2_projection_correlation_mean",
            "cross_level_coherence_diagnostic",
        ]
        missing_outputs = [key for key in required_outputs if key not in smoke_out]
        if missing_outputs:
            raise RuntimeError(
                "Core smoke test missing outputs: {}".format(missing_outputs)
            )
        gate_sum = smoke_out["g_L1"] + smoke_out["g_L2"] + smoke_out["g_L3"]
        gate_error = float(torch.abs(gate_sum - 1.0).max().detach().cpu())
        reconstruction_error = float(
            smoke_out["T_MEB_reconstruction_error"].max().detach().cpu()
        )
        finite_probability = bool(
            torch.isfinite(smoke_out["p_PPI"]).all().detach().cpu()
        )
        diagnostic_names = [
            "tri_level_consensus_diagnostic",
            "T_MEB_gate_entropy_normalized",
            "T_MEB_contribution_conflict",
            "order_stability_diagnostic",
            "cross_level_coherence_diagnostic",
        ]
        diagnostic_finite = all(
            bool(torch.isfinite(smoke_out[key]).all().detach().cpu())
            for key in diagnostic_names
        )
        diagnostic_in_range = all(
            bool(
                (
                    (smoke_out[key] >= 0.0) & (smoke_out[key] <= 1.0)
                ).all().detach().cpu()
            )
            for key in diagnostic_names
        )
        ppi_identity_error = float(
            torch.abs(
                smoke_out["p_PPI"] - torch.sigmoid(smoke_out["logit_PPI"])
            ).max().detach().cpu()
        )
        interface_shape = list(smoke_out["S_interface"].shape)
        if gate_error > 1e-5 or reconstruction_error > 1e-5:
            raise RuntimeError(
                "Core smoke test failed: gate_error={} reconstruction_error={}".format(
                    gate_error, reconstruction_error
                )
            )
        if (
            not finite_probability or not diagnostic_finite
            or not diagnostic_in_range or ppi_identity_error > 1e-7
            or interface_shape != [1, 8, 7]
        ):
            raise RuntimeError(
                "Core smoke test invalid probability/diagnostic/map: "
                "finite={} diagnostic_finite={} diagnostic_range={} "
                "ppi_identity_error={} shape={}".format(
                    finite_probability, diagnostic_finite,
                    diagnostic_in_range, ppi_identity_error, interface_shape
                )
            )
        result["status"] = "STRICT_COMPATIBILITY_PASS"
        result["strict_metadata"] = strict_metadata
        result["adapter_version"] = ADAPTER_VERSION
        result["adapter_sha256"] = sha256_file(
            adapter_dir / "osa_triage_adapter.py"
        )
        result["max_pair_len"] = int(cfg.max_pair_len)
        result["core_smoke_test"] = {
            "status": "PASS",
            "required_output_count": len(required_outputs),
            "gate_simplex_max_error": gate_error,
            "T_MEB_reconstruction_max_error": reconstruction_error,
            "finite_p_PPI": finite_probability,
            "diagnostics_finite": diagnostic_finite,
            "diagnostics_in_unit_interval": diagnostic_in_range,
            "p_PPI_sigmoid_logit_identity_max_error": ppi_identity_error,
            "symmetric_interface_shape": interface_shape,
            "native_evidence_schema_version": smoke_out[
                "native_evidence_schema_version"
            ],
        }
    except Exception as exc:
        result["status"] = "STRICT_COMPATIBILITY_FAIL"
        result["error"] = repr(exc)
    return result


def output_manifest(paths: Sequence[Path], root: Path, pd: Any) -> Any:
    rows = []
    for path in sorted(set(item.resolve() for item in paths if item.is_file())):
        rows.append({
            "path": relative_path(path, root), "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return pd.DataFrame(rows, columns=["path", "bytes", "sha256"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--triage-root", type=Path,
        default=Path('resources/backbone'),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--candidate-stage", default="04_candidates_v1")
    parser.add_argument("--output-stage", default="05_model_assets_v1")
    parser.add_argument("--esm-dir", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-embedding-finite-scan", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    triage_root = args.triage_root.expanduser().resolve()
    np, pd = require_packages()
    candidate_dir = root / "results" / "discovery_preprocess_v2" / args.candidate_stage
    primary_path = candidate_dir / "candidate_proteins_primary_250.tsv"
    sensitivity_path = candidate_dir / "candidate_proteins_sensitivity_500.tsv"
    controls_path = candidate_dir / "matched_control_proteins_preliminary.tsv"
    ranking_path = candidate_dir / "protein_ranking_all.tsv.gz"
    for path in [
        primary_path, sensitivity_path, controls_path, ranking_path,
        RUNTIME / "source_manifest.json",
    ]:
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    checkpoint_candidates = [
        triage_root / "triage_runs" / "tuna_from_engineering" / "triage_tuna_pair_finetuned.pt",
        triage_root / "triage_runs" / "triage_topk_final.pt",
        triage_root / "triage_runs" / "triage_final.pt",
    ]
    checkpoint = args.checkpoint
    explicit_checkpoint = checkpoint is not None
    if checkpoint is not None and not checkpoint.is_absolute():
        # User-facing result paths are project-root relative.  TRIAGE runtime
        # checkpoints discovered automatically remain triage-root relative.
        checkpoint = root / checkpoint
    if checkpoint is None:
        checkpoint = first_existing(checkpoint_candidates)
    print("OSA TRIAGE asset audit environment: PASS")
    print("Python: {}".format(sys.version.replace("\n", " ")))
    print("numpy: {}".format(np.__version__))
    print("pandas: {}".format(pd.__version__))
    print("TRIAGE root: {}".format(triage_root))
    print("Checkpoint: {}".format(checkpoint or "MISSING"))
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

    primary = pd.read_csv(str(primary_path), sep="\t")
    sensitivity = pd.read_csv(str(sensitivity_path), sep="\t")
    controls = pd.read_csv(str(controls_path), sep="\t")
    ranking = pd.read_csv(str(ranking_path), sep="\t", compression="gzip")
    primary_ids = set(primary["UniProt_accession"].astype(str))
    sensitivity_ids = set(sensitivity["UniProt_accession"].astype(str))
    control_ids = set(controls["control_UniProt"].astype(str))
    if len(primary_ids) != 250 or len(sensitivity_ids) != 500:
        raise RuntimeError("Frozen candidate counts are not 250/500")
    protein_meta = sensitivity[[
        "UniProt_accession", "gene_symbol_current", "UniProt_length",
        "discovery_rank",
    ]].copy()
    control_meta = ranking[
        ranking["UniProt_accession"].astype(str).isin(control_ids)
    ][[
        "UniProt_accession", "gene_symbol_current", "UniProt_length",
        "discovery_rank",
    ]].copy()
    if control_meta.shape[0] != len(control_ids):
        raise RuntimeError("Cannot recover all preliminary controls from full ranking")
    entity = pd.concat([protein_meta, control_meta], ignore_index=True, sort=False)
    entity = entity.drop_duplicates("UniProt_accession", keep="first")
    entity["role"] = "preliminary_control"
    entity.loc[entity["UniProt_accession"].isin(sensitivity_ids), "role"] = "sensitivity_only_251_500"
    entity.loc[entity["UniProt_accession"].isin(primary_ids), "role"] = "primary_250"
    entity["in_primary_250"] = entity["UniProt_accession"].isin(primary_ids)
    entity["in_sensitivity_500"] = entity["UniProt_accession"].isin(sensitivity_ids)
    entity["in_preliminary_control"] = entity["UniProt_accession"].isin(control_ids)

    extra_esm = [path.expanduser().resolve() for path in args.esm_dir]
    esm_dirs = extra_esm + [
        root / "model_assets" / "esm2",
        triage_root / "pair_data" / "TUnA_Intra0" / "fourpack" / "emb" / "esm2",
        triage_root / "esm_embeddings" / "pair",
        triage_root / "esm_embeddings" / "rbp400",
        triage_root / "esm_embeddings" / "dest",
        triage_root / "esm_embeddings" / "site",
    ]
    fasta_paths = [
        root / "model_assets" / "osa_frozen_750_full.fasta",
        root / "model_assets" / "osa_frozen_750_native512.fasta",
        root / "model_assets" / "osa_candidates.fasta",
        triage_root / "pair_data" / "TUnA_Intra0" / "fourpack" / "tuna.seq.fasta",
    ]
    fasta_ids = read_fasta_ids(fasta_paths)
    pssm_path = triage_root / "pair_data" / "TUnA_Intra0" / "fourpack" / "tuna.all.pssm.tsv"
    dssp_path = triage_root / "pair_data" / "TUnA_Intra0" / "fourpack" / "tuna.all.dssp.tsv"
    pssm_ids = table_identifier_set(pssm_path, pd)
    dssp_ids = table_identifier_set(dssp_path, pd)
    coord_dirs = [
        root / "model_assets" / "coords",
        triage_root / "pair_data" / "TUnA_Intra0" / "coords",
    ]
    checkpoint_attempts = []
    if explicit_checkpoint:
        checkpoint_audit = inspect_checkpoint(checkpoint, args.device)
        checkpoint_attempts.append(checkpoint_audit)
    else:
        checkpoint_audit = {"status": "MISSING", "path": ""}
        for candidate in checkpoint_candidates:
            if not candidate.is_file():
                continue
            attempt = inspect_checkpoint(candidate.resolve(), args.device)
            checkpoint_attempts.append(attempt)
            checkpoint_audit = attempt
            if attempt.get("status") == "STRICT_COMPATIBILITY_PASS":
                checkpoint = candidate.resolve()
                break
        else:
            checkpoint = first_existing(checkpoint_candidates)
    max_pair_len = int(checkpoint_audit.get("max_pair_len") or 512)

    rows = []
    sorted_entity = entity.sort_values(["role", "UniProt_accession"])
    for row_number, (_, protein) in enumerate(sorted_entity.iterrows(), 1):
        accession = str(protein["UniProt_accession"])
        emb = embedding_record(
            accession, esm_dirs, np,
            check_finite=not args.skip_embedding_finite_scan,
        )
        coord_status, coord_path = coord_available(accession, coord_dirs)
        length = pd.to_numeric(pd.Series([protein.get("UniProt_length")]), errors="coerce").iloc[0]
        length_value = int(length) if pd.notna(length) else 0
        sequence_available = "yes" if accession in fasta_ids else "no"
        pssm_available = "yes" if accession in pssm_ids else "no"
        dssp_available = "yes" if accession in dssp_ids else "no"
        if emb["ESM_available"] != "yes":
            tier = "unavailable"
        elif pssm_available == "yes" and dssp_available == "yes" and coord_status == "yes":
            tier = "ESM+PSSM+DSSP+coords"
        elif pssm_available == "yes" and dssp_available == "yes":
            tier = "ESM+PSSM+DSSP"
        elif pssm_available == "yes":
            tier = "ESM+PSSM"
        else:
            tier = "ESM_only"
        length_within = bool(length_value > 0 and length_value <= max_pair_len)
        rows.append({
            "UniProt_accession": accession,
            "gene_symbol_current": protein.get("gene_symbol_current", ""),
            "role": protein["role"],
            "in_primary_250": "yes" if bool(protein["in_primary_250"]) else "no",
            "in_sensitivity_500": "yes" if bool(protein["in_sensitivity_500"]) else "no",
            "in_preliminary_control": "yes" if bool(protein["in_preliminary_control"]) else "no",
            "discovery_rank": protein.get("discovery_rank", np.nan),
            "UniProt_length": length_value if length_value else np.nan,
            "sequence_available": sequence_available,
            "ESM_available": emb["ESM_available"],
            "ESM_path": emb["ESM_path"],
            "ESM_rows": emb["ESM_rows"],
            "ESM_dimension": emb["ESM_dimension"],
            "ESM_finite": emb["ESM_finite"],
            "ESM_error": emb["ESM_error"],
            "PSSM_available": pssm_available,
            "DSSP_available": dssp_available,
            "coords_available": coord_status,
            "coords_path": coord_path,
            "feature_availability_tier": tier,
            "length_within_checkpoint_max": "yes" if length_within else "no",
            "ESM_covers_checkpoint_input": "yes" if (
                emb["ESM_available"] == "yes"
                and int(emb["ESM_rows"]) >= min(max(length_value, 1), max_pair_len)
            ) else "no",
            "frozen_inference_ready_primary_policy": "yes" if (
                emb["ESM_available"] == "yes"
                and int(emb["ESM_dimension"]) == 1280
                and emb["ESM_finite"] == "yes"
                and length_value > 0
                and int(emb["ESM_rows"]) >= min(length_value, max_pair_len)
            ) else "no",
        })
        if row_number % 50 == 0 or row_number == sorted_entity.shape[0]:
            print("[asset-audit] proteins {}/{}".format(
                row_number, sorted_entity.shape[0]
            ), flush=True)
    availability = pd.DataFrame(rows)
    baseline_rows = source_baseline_audit(triage_root)
    baseline = pd.DataFrame(baseline_rows)
    coverage_rows = []
    coverage_definitions = [
        ("primary_250", "in_primary_250"),
        ("sensitivity_500", "in_sensitivity_500"),
        ("preliminary_control", "in_preliminary_control"),
    ]
    for role, membership_column in coverage_definitions:
        subset = availability[availability[membership_column] == "yes"]
        coverage_rows.append({
            "role": role, "proteins": int(subset.shape[0]),
            "sequence_available": int(subset["sequence_available"].eq("yes").sum()),
            "ESM_available": int(subset["ESM_available"].eq("yes").sum()),
            "ESM_dimension_1280": int(subset["ESM_dimension"].eq(1280).sum()),
            "ESM_finite": int(subset["ESM_finite"].eq("yes").sum()),
            "PSSM_available": int(subset["PSSM_available"].eq("yes").sum()),
            "DSSP_available": int(subset["DSSP_available"].eq("yes").sum()),
            "coords_available": int(subset["coords_available"].eq("yes").sum()),
            "length_within_checkpoint_max": int(
                subset["length_within_checkpoint_max"].eq("yes").sum()
            ),
            "frozen_inference_ready_primary_policy": int(
                subset["frozen_inference_ready_primary_policy"].eq("yes").sum()
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    blockers = []
    if checkpoint_audit.get("status") != "STRICT_COMPATIBILITY_PASS":
        blockers.append("No checkpoint passed strict architecture/state compatibility")
    primary_cov = coverage[coverage["role"] == "primary_250"].iloc[0]
    if int(primary_cov["ESM_available"]) < 250:
        blockers.append("Primary 250 lacks complete ESM2-650M embeddings")
    if int(primary_cov["frozen_inference_ready_primary_policy"]) < 250:
        blockers.append("Primary 250 lacks complete checkpoint-native sequence/embedding inputs")
    pair_feature_spec = checkpoint_audit.get("feature_spec", {}).get("pair", {})
    if bool(pair_feature_spec.get("pssm")) and int(primary_cov["PSSM_available"]) < 250:
        blockers.append(
            "Selected checkpoint was trained with pair PSSM, but primary 250 PSSM coverage is incomplete; generate matched PSSM or use a frozen ESM-only/feature-dropout adapted checkpoint"
        )
    if (
        bool(pair_feature_spec.get("dssp_ss"))
        or bool(pair_feature_spec.get("dssp_rsa"))
    ) and int(primary_cov["DSSP_available"]) < 250:
        blockers.append(
            "Selected checkpoint was trained with pair DSSP features, but primary 250 DSSP coverage is incomplete; generate matched DSSP or use a frozen ESM-only/feature-dropout adapted checkpoint"
        )
    if not bool(baseline["baseline_match"].eq("yes").all()):
        blockers.append("Immutable local baseline hash mismatch")
    present_source = baseline[baseline["source_exists"] == "yes"]
    if not present_source.empty and not bool(present_source["source_match"].eq("yes").all()):
        blockers.append("Server TRIAGE source differs from the immutable audited baseline")
    status = "PASS_READY_FOR_PAIR_INFERENCE" if not blockers else "BLOCKED_ASSET_PREPARATION"

    availability_path = output_dir / "protein_model_availability.tsv"
    coverage_path = output_dir / "availability_summary.tsv"
    baseline_path = output_dir / "triage_source_audit.tsv"
    write_tsv(availability, availability_path)
    write_tsv(coverage, coverage_path)
    write_tsv(baseline, baseline_path)
    outputs = [availability_path, coverage_path, baseline_path]
    summary = {
        "created_at_utc": utc_now(), "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "status": status, "blockers": blockers,
        "triage_root": str(triage_root),
        "checkpoint": checkpoint_audit,
        "checkpoint_candidates_checked": [str(path) for path in checkpoint_candidates],
        "checkpoint_attempts": checkpoint_attempts,
        "runtime_source_snapshot": str(RUNTIME),
        "max_pair_len": max_pair_len,
        "ESM_directories": [str(path) for path in esm_dirs],
        "FASTA_paths": [str(path) for path in fasta_paths],
        "coverage": coverage.to_dict(orient="records"),
        "baseline_files_match": bool(baseline["baseline_match"].eq("yes").all()),
        "source_files_match_when_present": bool(
            baseline.loc[baseline["source_exists"].eq("yes"), "source_match"].eq("yes").all()
        ),
        "validation_data_read": False,
        "candidate_membership_changed": False,
        "pair_inference_performed": False,
        "embedding_finite_scan_performed": not args.skip_embedding_finite_scan,
        "next_gate": (
            "Run frozen symmetric A-B/B-A pair inference"
            if status == "PASS_READY_FOR_PAIR_INFERENCE"
            else "Resolve all listed blockers before pair inference"
        ),
        "python": sys.version,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    summary_path = output_dir / "model_asset_audit.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    outputs.append(summary_path)
    manifest_path = output_dir / "output_manifest.tsv"
    write_tsv(output_manifest(outputs, root, pd), manifest_path)
    print("OSA TRIAGE model asset audit: {}".format(status))
    print(coverage.to_string(index=False))
    if blockers:
        print("Blockers:")
        for blocker in blockers:
            print("- " + blocker)
    print("Output: {}".format(output_dir))
    print("No validation data was read and no candidate membership was changed.")
    return 0 if status == "PASS_READY_FOR_PAIR_INFERENCE" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
