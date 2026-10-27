#!/usr/bin/env python3
"""Run frozen symmetric TRIAGE-Fuse inference for OSA protein pairs.

The script is Python 3.7 compatible, resumable by deterministic registry
chunks, and never reads OSA expression effects or external validation data.
It exports scalar L1/L2/L3/T-MEB results for every pair and compact native
evidence shards (residue profiles, L2 projections, Top-K/Top-M support and
evidence vectors).  Full dense interface maps remain reproducible on demand
from the frozen checkpoint and pair registry rather than being materialized
for every pair.
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
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


SCRIPT_VERSION = "1.1.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_REL = (
    "results/discovery_preprocess_v2/06_model_training/esm_only_adaptation_v1/"
    "triage_tuna_esm_only_adapted.pt"
)
REGISTRY_STAGE = "07_pair_registry_v1"
AUDIT_REL = "results/discovery_preprocess_v2/05_model_assets_v1/model_asset_audit.json"

MODEL_SCALAR_FIELDS = [
    "p_PPI", "logit_PPI", "p_L1", "p_L2", "p_L3",
    "logit_L1", "logit_L2", "logit_L3",
    "g_L1", "g_L2", "g_L3",
    "contribution_L1", "contribution_L2", "contribution_L3",
    "T_MEB_reconstruction_error", "evidence_reliability",
    "p_PPI_AB", "p_PPI_BA", "AB_BA_probability_gap", "AB_BA_logit_gap",
    "AB_BA_L1_probability_gap", "AB_BA_L2_probability_gap",
    "AB_BA_L3_probability_gap", "L2_MIL_probability",
    "tri_level_probability_std", "tri_level_consensus_diagnostic",
    "T_MEB_gate_entropy_normalized", "T_MEB_contribution_conflict",
    "branch_direction_agreement", "order_stability_diagnostic",
    "cross_level_coherence_diagnostic", "fusion_margin_diagnostic",
    "L1_L2_projection_correlation_A", "L1_L2_projection_correlation_B",
    "L1_L2_projection_correlation_mean",
]
DERIVED_SCALAR_FIELDS = [
    "topk_residue_A_count", "topk_residue_B_count",
    "topk_residue_A_probability_mean", "topk_residue_B_probability_mean",
    "topm_interface_count", "topm_interface_probability_mean",
    "topm_interface_probability_max", "topm_interface_entropy_normalized",
]
PROVENANCE_FIELDS = [
    "used_length_A", "used_length_B", "truncated_A", "truncated_B",
    "feature_tier_A", "feature_tier_B", "checkpoint_sha256",
    "adapter_version", "native_evidence_schema_version",
    "evidence_shard", "evidence_row",
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


def relative_path(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Not JSON serializable: {}".format(type(value).__name__))


def atomic_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def deterministic_gzip_writer(path, fieldnames, rows):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=fieldnames, delimiter="\t",
                    lineterminator="\n", extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(str(tmp), str(path))


def read_registry(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "pair_index", "pair_id", "protein_A", "protein_B",
        "length_A", "length_B",
    }
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise RuntimeError("Registry is missing columns: {}".format(missing))
    indices = [int(row["pair_index"]) for row in rows]
    if indices != list(range(1, len(rows) + 1)):
        raise RuntimeError("Registry pair_index is not contiguous from 1")
    pair_ids = [row["pair_id"] for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise RuntimeError("Registry contains duplicate pair_id values")
    return rows


class ESMFeatureCache(object):
    def __init__(self, np_module, esm_dir, max_length, max_items):
        self.np = np_module
        self.esm_dir = esm_dir
        self.max_length = int(max_length)
        self.max_items = max(1, int(max_items))
        self.cache = OrderedDict()

    def get(self, accession):
        if accession in self.cache:
            value = self.cache.pop(accession)
            self.cache[accession] = value
            return value
        path = self.esm_dir / (accession + ".npy")
        if not path.is_file():
            raise RuntimeError("Missing ESM embedding: " + str(path))
        array = self.np.load(str(path), allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 1280 or array.shape[0] <= 0:
            raise RuntimeError("Invalid ESM shape {} for {}".format(array.shape, accession))
        if not self.np.isfinite(array).all():
            raise RuntimeError("Non-finite ESM embedding for " + accession)
        used_length = min(int(array.shape[0]), self.max_length)
        value = (
            self.np.asarray(array[:self.max_length], dtype=self.np.float32),
            used_length,
        )
        self.cache[accession] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


def make_batch(torch, np_module, cache, rows, d_res_in, device):
    cached_a = [cache.get(row["protein_A"]) for row in rows]
    cached_b = [cache.get(row["protein_B"]) for row in rows]
    for row, cached in zip(rows, cached_a):
        expected = min(int(row["length_A"]), cache.max_length)
        if expected != cached[1]:
            raise RuntimeError("Registry/ESM length mismatch for " + row["protein_A"])
    for row, cached in zip(rows, cached_b):
        expected = min(int(row["length_B"]), cache.max_length)
        if expected != cached[1]:
            raise RuntimeError("Registry/ESM length mismatch for " + row["protein_B"])
    esm_a = [item[0] for item in cached_a]
    esm_b = [item[0] for item in cached_b]
    len_a = [int(arr.shape[0]) for arr in esm_a]
    len_b = [int(arr.shape[0]) for arr in esm_b]
    max_a, max_b = max(len_a), max(len_b)
    batch = len(rows)
    res_a = torch.zeros(batch, max_a, d_res_in, dtype=torch.float32)
    res_b = torch.zeros(batch, max_b, d_res_in, dtype=torch.float32)
    mask_a = torch.zeros(batch, max_a, dtype=torch.float32)
    mask_b = torch.zeros(batch, max_b, dtype=torch.float32)
    for index, array in enumerate(esm_a):
        length = len_a[index]
        res_a[index, :length, :1280] = torch.from_numpy(
            np_module.ascontiguousarray(array)
        )
        mask_a[index, :length] = 1.0
    for index, array in enumerate(esm_b):
        length = len_b[index]
        res_b[index, :length, :1280] = torch.from_numpy(
            np_module.ascontiguousarray(array)
        )
        mask_b[index, :length] = 1.0
    coords_a = torch.full((batch, max_a, 3), float("nan"), dtype=torch.float32)
    coords_b = torch.full((batch, max_b, 3), float("nan"), dtype=torch.float32)
    inputs = {
        "resA": res_a.to(device), "resB": res_b.to(device),
        "maskA": mask_a.to(device), "maskB": mask_b.to(device),
        "coordsA": coords_a.to(device), "coordsB": coords_b.to(device),
    }
    return inputs, len_a, len_b


def scalar_at(torch, output, key, index):
    value = output[key]
    if not torch.is_tensor(value):
        return value
    value = value.detach().float().cpu().reshape(-1)[index].item()
    if not math.isfinite(float(value)):
        raise RuntimeError("Non-finite model scalar {} at batch row {}".format(key, index))
    return float(value)


def normalized_entropy(np_module, logits):
    if len(logits) <= 1:
        return 0.0
    shifted = logits - logits.max()
    weight = np_module.exp(shifted)
    weight = weight / max(float(weight.sum()), 1e-12)
    entropy = -float((weight * np_module.log(np_module.maximum(weight, 1e-12))).sum())
    return entropy / math.log(len(logits))


def stable_sigmoid(np_module, logits):
    clipped = np_module.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np_module.exp(-clipped))


def append_ragged(target_values, target_offsets, values):
    target_values.append(values)
    target_offsets.append(target_offsets[-1] + int(values.shape[0]))


def process_chunk(torch, np_module, adapter, cache, chunk_rows, batch_size, device, checkpoint_hash, adapter_version):
    scalar_rows = []
    evidence = {
        "pair_id": [],
        "p_residue_A_values": [], "p_residue_A_offsets": [0],
        "p_residue_B_values": [], "p_residue_B_offsets": [0],
        "L2_projection_A_values": [], "L2_projection_A_offsets": [0],
        "L2_projection_B_values": [], "L2_projection_B_offsets": [0],
        "topk_A_indices": [], "topk_A_logits": [], "topk_A_offsets": [0],
        "topk_B_indices": [], "topk_B_logits": [], "topk_B_offsets": [0],
        "topm_indices": [], "topm_logits": [], "topm_probabilities": [], "topm_offsets": [0],
        "z_L1": [], "z_L2": [], "z_L3": [],
    }
    schema_version = None
    for start in range(0, len(chunk_rows), batch_size):
        batch_rows = chunk_rows[start:start + batch_size]
        inputs, lengths_a, lengths_b = make_batch(
            torch, np_module, cache, batch_rows, adapter.model.cfg.d_res_in, device
        )
        try:
            output = adapter(**inputs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "CUDA out of memory at batch_size={}; rerun with a smaller --batch-size".format(batch_size)
                )
            raise
        schema_version = str(output["native_evidence_schema_version"])
        for local_index, registry_row in enumerate(batch_rows):
            length_a, length_b = lengths_a[local_index], lengths_b[local_index]
            scalar = dict(registry_row)
            for key in MODEL_SCALAR_FIELDS:
                scalar[key] = "{:.10g}".format(scalar_at(torch, output, key, local_index))
            p_res_a = output["p_residue_A"][local_index, :length_a].detach().float().cpu().numpy()
            p_res_b = output["p_residue_B"][local_index, :length_b].detach().float().cpu().numpy()
            proj_a = output["L2_to_L1_projection_A"][local_index, :length_a].detach().float().cpu().numpy()
            proj_b = output["L2_to_L1_projection_B"][local_index, :length_b].detach().float().cpu().numpy()
            topk_a_count = min(int(adapter.model.cfg.topk), length_a)
            topk_b_count = min(int(adapter.model.cfg.topk), length_b)
            topm_count = min(int(adapter.model.cfg.topm), length_a * length_b)
            topk_a_idx = output["topk_residue_A_indices"][local_index, :topk_a_count].detach().cpu().numpy().astype(np_module.int32)
            topk_b_idx = output["topk_residue_B_indices"][local_index, :topk_b_count].detach().cpu().numpy().astype(np_module.int32)
            topk_a_log = output["topk_residue_A_logits"][local_index, :topk_a_count].detach().float().cpu().numpy()
            topk_b_log = output["topk_residue_B_logits"][local_index, :topk_b_count].detach().float().cpu().numpy()
            topm_idx = output["topm_interface_indices"][local_index, :topm_count].detach().cpu().numpy().astype(np_module.int32)
            topm_log = output["topm_interface_logits"][local_index, :topm_count].detach().float().cpu().numpy()
            topm_prob = output["topm_interface_probabilities"][local_index, :topm_count].detach().float().cpu().numpy()
            scalar.update({
                "topk_residue_A_count": str(topk_a_count),
                "topk_residue_B_count": str(topk_b_count),
                "topk_residue_A_probability_mean": "{:.10g}".format(float(stable_sigmoid(np_module, topk_a_log).mean())),
                "topk_residue_B_probability_mean": "{:.10g}".format(float(stable_sigmoid(np_module, topk_b_log).mean())),
                "topm_interface_count": str(topm_count),
                "topm_interface_probability_mean": "{:.10g}".format(float(topm_prob.mean())),
                "topm_interface_probability_max": "{:.10g}".format(float(topm_prob.max())),
                "topm_interface_entropy_normalized": "{:.10g}".format(normalized_entropy(np_module, topm_log)),
                "used_length_A": str(length_a), "used_length_B": str(length_b),
                "truncated_A": "yes" if int(registry_row["length_A"]) > length_a else "no",
                "truncated_B": "yes" if int(registry_row["length_B"]) > length_b else "no",
                "feature_tier_A": "ESM_only", "feature_tier_B": "ESM_only",
                "checkpoint_sha256": checkpoint_hash,
                "adapter_version": adapter_version,
                "native_evidence_schema_version": schema_version,
                "evidence_row": str(len(scalar_rows)),
            })
            scalar_rows.append(scalar)
            evidence["pair_id"].append(registry_row["pair_id"])
            append_ragged(evidence["p_residue_A_values"], evidence["p_residue_A_offsets"], p_res_a.astype(np_module.float16))
            append_ragged(evidence["p_residue_B_values"], evidence["p_residue_B_offsets"], p_res_b.astype(np_module.float16))
            append_ragged(evidence["L2_projection_A_values"], evidence["L2_projection_A_offsets"], proj_a.astype(np_module.float16))
            append_ragged(evidence["L2_projection_B_values"], evidence["L2_projection_B_offsets"], proj_b.astype(np_module.float16))
            append_ragged(evidence["topk_A_indices"], evidence["topk_A_offsets"], topk_a_idx)
            evidence["topk_A_logits"].append(topk_a_log.astype(np_module.float16))
            append_ragged(evidence["topk_B_indices"], evidence["topk_B_offsets"], topk_b_idx)
            evidence["topk_B_logits"].append(topk_b_log.astype(np_module.float16))
            append_ragged(evidence["topm_indices"], evidence["topm_offsets"], topm_idx)
            evidence["topm_logits"].append(topm_log.astype(np_module.float16))
            evidence["topm_probabilities"].append(topm_prob.astype(np_module.float16))
            for key in ("z_L1", "z_L2", "z_L3"):
                evidence[key].append(output[key][local_index].detach().float().cpu().numpy().astype(np_module.float16))
        del output, inputs
    return scalar_rows, evidence, schema_version


def concatenate(np_module, values, dtype, shape_tail=()):
    if not values:
        return np_module.empty((0,) + tuple(shape_tail), dtype=dtype)
    return np_module.concatenate(values, axis=0).astype(dtype, copy=False)


def save_evidence(np_module, path, evidence):
    payload = {
        "pair_id": np_module.asarray(evidence["pair_id"], dtype="U64"),
        "p_residue_A_values": concatenate(np_module, evidence["p_residue_A_values"], np_module.float16),
        "p_residue_A_offsets": np_module.asarray(evidence["p_residue_A_offsets"], dtype=np_module.int64),
        "p_residue_B_values": concatenate(np_module, evidence["p_residue_B_values"], np_module.float16),
        "p_residue_B_offsets": np_module.asarray(evidence["p_residue_B_offsets"], dtype=np_module.int64),
        "L2_projection_A_values": concatenate(np_module, evidence["L2_projection_A_values"], np_module.float16),
        "L2_projection_A_offsets": np_module.asarray(evidence["L2_projection_A_offsets"], dtype=np_module.int64),
        "L2_projection_B_values": concatenate(np_module, evidence["L2_projection_B_values"], np_module.float16),
        "L2_projection_B_offsets": np_module.asarray(evidence["L2_projection_B_offsets"], dtype=np_module.int64),
        "topk_A_indices": concatenate(np_module, evidence["topk_A_indices"], np_module.int32),
        "topk_A_logits": concatenate(np_module, evidence["topk_A_logits"], np_module.float16),
        "topk_A_offsets": np_module.asarray(evidence["topk_A_offsets"], dtype=np_module.int64),
        "topk_B_indices": concatenate(np_module, evidence["topk_B_indices"], np_module.int32),
        "topk_B_logits": concatenate(np_module, evidence["topk_B_logits"], np_module.float16),
        "topk_B_offsets": np_module.asarray(evidence["topk_B_offsets"], dtype=np_module.int64),
        "topm_indices": concatenate(np_module, evidence["topm_indices"], np_module.int32, (2,)),
        "topm_logits": concatenate(np_module, evidence["topm_logits"], np_module.float16),
        "topm_probabilities": concatenate(np_module, evidence["topm_probabilities"], np_module.float16),
        "topm_offsets": np_module.asarray(evidence["topm_offsets"], dtype=np_module.int64),
        "z_L1": np_module.stack(evidence["z_L1"]).astype(np_module.float16),
        "z_L2": np_module.stack(evidence["z_L2"]).astype(np_module.float16),
        "z_L3": np_module.stack(evidence["z_L3"]).astype(np_module.float16),
    }
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np_module.savez_compressed(handle, **payload)
    os.replace(str(tmp), str(path))


def merge_scalar_chunks(output_dir, chunks, fieldnames):
    rows = []
    for item in chunks:
        with gzip.open(str(item["scalar_path"]), "rt", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    merged = output_dir / "pair_scores.tsv.gz"
    deterministic_gzip_writer(merged, fieldnames, rows)
    return merged, len(rows)


def write_plain_tsv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--universe", choices=["primary_250", "sensitivity_500"], default="primary_250")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT_REL))
    parser.add_argument("--esm-dir", type=Path, default=Path("model_assets/esm2"))
    parser.add_argument("--output-stage", default="08_pair_inference_primary_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--cache-proteins", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-pairs", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise RuntimeError("batch-size and chunk-size must be positive")
    root = args.root.expanduser().resolve()
    registry = args.registry
    if registry is None:
        name = "pair_registry_primary_250.tsv.gz" if args.universe == "primary_250" else "pair_registry_sensitivity_500.tsv.gz"
        registry = root / "results" / "discovery_preprocess_v2" / REGISTRY_STAGE / name
    elif not registry.is_absolute():
        registry = root / registry
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    esm_dir = args.esm_dir
    if not esm_dir.is_absolute():
        esm_dir = root / esm_dir
    audit_path = root / AUDIT_REL
    for path in (registry, checkpoint, audit_path):
        if not path.is_file():
            raise RuntimeError("Missing required input: " + str(path))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_READY_FOR_PAIR_INFERENCE":
        raise RuntimeError("Asset audit has not released pair inference")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != audit.get("checkpoint", {}).get("sha256"):
        raise RuntimeError("Checkpoint hash differs from released asset audit")
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("numpy and torch are required") from exc
    adapter_dir = Path(__file__).resolve().parent
    if str(adapter_dir) not in sys.path:
        sys.path.insert(0, str(adapter_dir))
    from osa_triage_adapter import ADAPTER_VERSION, SymmetricTRIAGEAdapter, load_frozen_checkpoint
    adapter_hash = sha256_file(adapter_dir / "osa_triage_adapter.py")
    released_adapter_hash = audit.get("checkpoint", {}).get("adapter_sha256")
    if released_adapter_hash != adapter_hash:
        raise RuntimeError(
            "Adapter differs from the released asset audit; rerun "
            "audit_triage_assets.py before pair inference"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, cfg, metadata = load_frozen_checkpoint(checkpoint, device)
    pair_spec = metadata.get("feature_spec", {}).get("pair", {})
    if bool(pair_spec.get("pssm")) or bool(pair_spec.get("dssp_ss")) or bool(pair_spec.get("dssp_rsa")):
        raise RuntimeError("Released OSA inference checkpoint is not ESM-only")
    rows = read_registry(registry)
    expected = 31125 if args.universe == "primary_250" else 124750
    if len(rows) != expected:
        raise RuntimeError("Unexpected {} registry size: {}".format(args.universe, len(rows)))
    selected_count = len(rows)
    if args.smoke:
        selected_count = min(selected_count, int(args.smoke_pairs))
    elif args.max_pairs > 0:
        selected_count = min(selected_count, int(args.max_pairs))
    rows = rows[:selected_count]
    unique_proteins = sorted({row["protein_A"] for row in rows} | {row["protein_B"] for row in rows})
    for accession in unique_proteins:
        if not (esm_dir / (accession + ".npy")).is_file():
            raise RuntimeError("Missing ESM input for " + accession)
    print("OSA symmetric inference environment: PASS")
    print("Universe: {} pairs={:,}".format(args.universe, selected_count))
    print("Checkpoint: {}".format(checkpoint))
    print("Checkpoint SHA256: {}".format(checkpoint_hash))
    print("Adapter: {}".format(ADAPTER_VERSION))
    print("Device: {} batch_size={} chunk_size={}".format(device, args.batch_size, args.chunk_size))
    print("OSA effects/validation/known PPI read: no")
    if args.check_env:
        return 0
    base_output = root / "results" / "discovery_preprocess_v2" / args.output_stage
    output_dir = base_output / "smoke" if args.smoke else base_output
    if output_dir.exists() and args.force:
        if root not in output_dir.resolve().parents:
            raise RuntimeError("Refusing to replace output outside project root")
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "created_at_utc": utc_now(), "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "adapter_version": ADAPTER_VERSION,
        "adapter_sha256": adapter_hash,
        "universe": args.universe, "registry": relative_path(registry, root),
        "registry_sha256": sha256_file(registry), "selected_pairs": selected_count,
        "checkpoint": relative_path(checkpoint, root), "checkpoint_sha256": checkpoint_hash,
        "ESM_directory": relative_path(esm_dir, root),
        "feature_policy": (
            "ESM-only; PSSM/DSSP zero channels; coordinates NaN missing state; "
            "sequences longer than checkpoint max_pair_len are deterministically "
            "right-truncated and flagged"
        ),
        "symmetric_policy": "A-B and B-A; average branch/fused logits before sigmoid",
        "batch_size": args.batch_size, "chunk_size": args.chunk_size,
        "smoke": bool(args.smoke), "OSA_effects_read": False,
        "validation_data_read": False, "known_PPI_read": False,
        "p_PPI_modified_by_diagnostics": False,
        "diagnostics_are_calibrated_probabilities": False,
        "dense_interface_policy": "not materialized for every pair; exact full map is reproducible on demand; sparse Top-M retained for every pair",
        "innovation_diagnostics": [
            "tri-level probability consensus", "normalized T-MEB gate entropy",
            "T-MEB contribution conflict", "A-B/B-A order stability",
            "L1-to-L2 projection correlation", "cross-level coherence diagnostic",
        ],
    }
    contract_path = output_dir / "inference_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        stable_keys = [
            "script_sha256", "adapter_sha256", "universe", "registry_sha256",
            "selected_pairs", "checkpoint_sha256", "feature_policy", "symmetric_policy",
            "batch_size", "chunk_size", "smoke",
        ]
        if any(existing.get(key) != contract.get(key) for key in stable_keys):
            raise RuntimeError("Existing output contract differs; use --force for a new run")
        contract = existing
    else:
        atomic_text(contract_path, json.dumps(contract, ensure_ascii=True, indent=2) + "\n")
    adapter = SymmetricTRIAGEAdapter(model, metadata).to(device)
    cache = ESMFeatureCache(np, esm_dir, cfg.max_pair_len, args.cache_proteins)
    registry_fields = list(rows[0].keys())
    scalar_fields = registry_fields + MODEL_SCALAR_FIELDS + DERIVED_SCALAR_FIELDS + PROVENANCE_FIELDS
    completed_chunks = []
    for chunk_start in range(0, selected_count, args.chunk_size):
        chunk_end = min(selected_count, chunk_start + args.chunk_size)
        label = "{:08d}_{:08d}".format(chunk_start + 1, chunk_end)
        scalar_path = chunks_dir / ("scalar_" + label + ".tsv.gz")
        evidence_path = chunks_dir / ("evidence_" + label + ".npz")
        marker_path = chunks_dir / ("complete_" + label + ".json")
        if marker_path.is_file() and scalar_path.is_file() and evidence_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("scalar_sha256") == sha256_file(scalar_path) and marker.get("evidence_sha256") == sha256_file(evidence_path):
                completed_chunks.append({"start": chunk_start + 1, "end": chunk_end, "scalar_path": scalar_path, "evidence_path": evidence_path, "marker": marker})
                print("[resume] chunk {}".format(label), flush=True)
                continue
            raise RuntimeError("Incomplete or hash-mismatched existing chunk " + label)
        chunk_rows = rows[chunk_start:chunk_end]
        if device.type == "cuda":
            torch.cuda.synchronize()
        chunk_started = time.time()
        scalar_rows, evidence, schema_version = process_chunk(
            torch, np, adapter, cache, chunk_rows, args.batch_size, device,
            checkpoint_hash, ADAPTER_VERSION,
        )
        evidence_rel = relative_path(evidence_path, root)
        for index, scalar in enumerate(scalar_rows):
            scalar["evidence_shard"] = evidence_rel
            scalar["evidence_row"] = str(index)
        deterministic_gzip_writer(scalar_path, scalar_fields, scalar_rows)
        save_evidence(np, evidence_path, evidence)
        if device.type == "cuda":
            torch.cuda.synchronize()
        chunk_seconds = time.time() - chunk_started
        marker = {
            "start": chunk_start + 1, "end": chunk_end,
            "pairs": len(chunk_rows), "first_pair_id": chunk_rows[0]["pair_id"],
            "last_pair_id": chunk_rows[-1]["pair_id"],
            "native_evidence_schema_version": schema_version,
            "runtime_seconds": chunk_seconds,
            "pairs_per_second": len(chunk_rows) / max(chunk_seconds, 1e-12),
            "scalar_sha256": sha256_file(scalar_path),
            "evidence_sha256": sha256_file(evidence_path),
        }
        atomic_text(marker_path, json.dumps(marker, ensure_ascii=True, indent=2) + "\n")
        completed_chunks.append({"start": chunk_start + 1, "end": chunk_end, "scalar_path": scalar_path, "evidence_path": evidence_path, "marker": marker})
        print(
            "[inference] chunk {} pairs={} seconds={:.1f} pairs/s={:.3f} complete".format(
                label, len(chunk_rows), chunk_seconds,
                len(chunk_rows) / max(chunk_seconds, 1e-12),
            ),
            flush=True,
        )
    completed_chunks.sort(key=lambda item: item["start"])
    pair_scores, merged_count = merge_scalar_chunks(output_dir, completed_chunks, scalar_fields)
    if merged_count != selected_count:
        raise RuntimeError("Merged score count mismatch: {} != {}".format(merged_count, selected_count))
    evidence_manifest = output_dir / "evidence_manifest.tsv"
    evidence_rows = [{
        "start": str(item["start"]), "end": str(item["end"]),
        "pairs": str(item["marker"]["pairs"]),
        "path": relative_path(item["evidence_path"], root),
        "sha256": item["marker"]["evidence_sha256"],
        "native_evidence_schema_version": item["marker"]["native_evidence_schema_version"],
    } for item in completed_chunks]
    write_plain_tsv(evidence_manifest, evidence_rows, ["start", "end", "pairs", "path", "sha256", "native_evidence_schema_version"])
    compute_seconds = sum(
        float(item["marker"].get("runtime_seconds", 0.0))
        for item in completed_chunks
    )
    summary = {
        "created_at_utc": utc_now(), "status": "PASS",
        "pairs": selected_count, "chunks": len(completed_chunks),
        "universe": args.universe, "checkpoint_sha256": checkpoint_hash,
        "registry_sha256": sha256_file(registry),
        "recorded_compute_seconds": compute_seconds,
        "recorded_pairs_per_second": (
            selected_count / compute_seconds if compute_seconds > 0.0 else None
        ),
        "pair_scores": {"path": relative_path(pair_scores, root), "sha256": sha256_file(pair_scores)},
        "evidence_manifest": {"path": relative_path(evidence_manifest, root), "sha256": sha256_file(evidence_manifest)},
        "validation_data_read": False, "OSA_effects_read": False,
        "pair_ranking_performed": False,
        "next_gate": "Audit primary frozen p_PPI and trace diagnostics before constructing r_OSA",
    }
    summary_path = output_dir / "run_summary.json"
    atomic_text(summary_path, json.dumps(summary, ensure_ascii=True, indent=2, default=json_default) + "\n")
    outputs = [contract_path, pair_scores, evidence_manifest, summary_path]
    manifest_rows = [{
        "path": relative_path(path, root), "bytes": str(path.stat().st_size),
        "sha256": sha256_file(path),
    } for path in outputs]
    output_manifest = output_dir / "output_manifest.tsv"
    write_plain_tsv(output_manifest, manifest_rows, ["path", "bytes", "sha256"])
    print("OSA symmetric pair inference: PASS")
    print("Pairs: {:,}".format(selected_count))
    print("Scores: {}".format(pair_scores))
    print("No OSA context or validation data was read; no ranking was performed.")
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
