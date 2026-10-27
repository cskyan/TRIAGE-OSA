#!/usr/bin/env python3
"""Strict, symmetric OSA adapter for the frozen TRIAGE L1-L2-L3 model.

The adapter does not train on OSA data and does not inject OSA evidence into
the residue or interface branches.  It exposes disease-independent p_PPI and
branch diagnostics; pair-level OSA context is joined downstream.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_DIR = Path(__file__).resolve().parents[1]
SERVER_RUNTIME_DIR = MODEL_DIR / "triage_server_candidate"
BASELINE_DIR = MODEL_DIR / "triage_base"
RUNTIME_DIR = (
    SERVER_RUNTIME_DIR
    if (SERVER_RUNTIME_DIR / "source_manifest.json").is_file()
    else BASELINE_DIR
)
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

import torch
import torch.nn as nn

from triage_model import (
    TRIAGEConfig,
    TRIAGEPPIModel,
    project_interface_to_residue,
    select_topk_residues,
    select_topm_interface,
)


ADAPTER_VERSION = "1.3.0"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def config_from_checkpoint(checkpoint: Dict[str, Any]) -> TRIAGEConfig:
    cfg = TRIAGEConfig()
    saved = checkpoint.get("config", {})
    if not isinstance(saved, dict):
        raise RuntimeError("Checkpoint config is absent or not a dictionary")
    loaded = 0
    for key, value in saved.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            loaded += 1
    if loaded == 0:
        raise RuntimeError("Checkpoint config did not match TRIAGEConfig")
    return cfg


def legacy_targets(key: str) -> List[str]:
    if ".na." in key:
        return [key.replace(".na.", ".nqA."), key.replace(".na.", ".nkA.")]
    if ".nb." in key:
        return [key.replace(".nb.", ".nqB."), key.replace(".nb.", ".nkB.")]
    return [key]


def load_frozen_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[TRIAGEPPIModel, TRIAGEConfig, Dict[str, Any]]:
    """Load a checkpoint with no missing, unexpected or shape-skipped keys."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError("Missing TRIAGE checkpoint: " + str(checkpoint_path))
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("TRIAGE checkpoint must be a dictionary")
    cfg = config_from_checkpoint(checkpoint)
    model = TRIAGEPPIModel(cfg).to(device)
    state = checkpoint.get(
        "model_state_dict", checkpoint.get("model_state", checkpoint)
    )
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint state_dict is absent")
    current = model.state_dict()
    compatible = {}  # type: Dict[str, Any]
    invalid = []  # type: List[str]
    for key, value in state.items():
        loaded = False
        for target in legacy_targets(str(key)):
            if target in current and tuple(current[target].shape) == tuple(value.shape):
                compatible[target] = value
                loaded = True
            else:
                invalid.append("{}->{}".format(key, target))
        if not loaded and not legacy_targets(str(key)):
            invalid.append(str(key))
    missing = sorted(set(current) - set(compatible))
    unexpected = sorted(set(compatible) - set(current))
    if invalid or missing or unexpected:
        raise RuntimeError(
            "Strict checkpoint compatibility failed: invalid_or_shape={} missing={} "
            "unexpected={}; examples invalid={} missing={}".format(
                len(invalid), len(missing), len(unexpected), invalid[:8], missing[:8]
            )
        )
    model.load_state_dict(compatible, strict=True)
    model.eval()
    metadata = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "runtime_source_dir": str(RUNTIME_DIR),
        "runtime_source_manifest_sha256": sha256_file(
            RUNTIME_DIR / "source_manifest.json"
        ),
        "stage": checkpoint.get("stage", ""),
        "epoch": checkpoint.get("epoch", ""),
        "best_metric": checkpoint.get("best_metric", ""),
        "feature_spec": checkpoint.get("feature_spec", {}),
        "state_key_count": len(state),
        "loaded_key_count": len(compatible),
        "config_fields_loaded": len(checkpoint.get("config", {})),
        "d_res_in": int(cfg.d_res_in),
        "max_pair_len": int(cfg.max_pair_len),
    }
    return model, cfg, metadata


def swapped_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    swapped = dict(inputs)
    pairs = [
        ("resA", "resB"), ("maskA", "maskB"), ("chainA", "chainB"),
        ("coordsA", "coordsB"),
    ]
    for left, right in pairs:
        swapped[left], swapped[right] = inputs.get(right), inputs.get(left)
    return swapped


class SymmetricTRIAGEAdapter(nn.Module):
    """Order-invariant TRIAGE-Fuse scoring with native tri-level readouts.

    The frozen checkpoint contains chain-specific towers.  To preserve its
    learned parameters while making an unordered OSA protein pair invariant to
    chain assignment, the adapter evaluates A-B and B-A.  Scalar decisions are
    averaged in logit space.  Residue and interface maps are returned in the
    original A-by-B coordinate system after transposing the B-A interface map.
    """

    def __init__(self, model: TRIAGEPPIModel, checkpoint_metadata: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.checkpoint_metadata = dict(checkpoint_metadata)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    @staticmethod
    def _mean_logit(first: Dict[str, Any], second: Dict[str, Any], key: str):
        return 0.5 * (first[key] + second[key])

    @staticmethod
    def _input_mask(value: Optional[torch.Tensor], residues: torch.Tensor):
        if residues.dim() == 2:
            residues = residues.unsqueeze(0)
        if value is None:
            return torch.ones(
                residues.shape[:2], dtype=torch.bool, device=residues.device
            )
        if value.dim() == 1:
            value = value.unsqueeze(0)
        return value.to(device=residues.device) > 0.5

    @staticmethod
    def _masked_pearson(
        first: torch.Tensor,
        second: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-pair correlation with a neutral zero for degenerate vectors."""
        weight = mask.to(dtype=first.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean_first = (first * weight).sum(dim=1) / count
        mean_second = (second * weight).sum(dim=1) / count
        centered_first = (first - mean_first.unsqueeze(1)) * weight
        centered_second = (second - mean_second.unsqueeze(1)) * weight
        numerator = (centered_first * centered_second).sum(dim=1)
        denominator = torch.sqrt(
            centered_first.square().sum(dim=1)
            * centered_second.square().sum(dim=1)
        )
        correlation = numerator / denominator.clamp_min(1e-12)
        valid = (count >= 2.0) & (denominator > 1e-12)
        return torch.where(valid, correlation, torch.zeros_like(correlation))

    def train(self, mode: bool = True):
        # Frozen inference remains in eval mode even if a parent module calls train().
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, **inputs) -> Dict[str, Any]:
        self.model.eval()
        out_ab = self.model(**inputs)
        out_ba = self.model(**swapped_inputs(inputs))

        # Scalar branch evidence and final fused decision (paper Eqs. 17, 22-23).
        logit_l1 = self._mean_logit(out_ab, out_ba, "logit_res")
        logit_l2 = self._mean_logit(out_ab, out_ba, "logit_interface")
        logit_l3 = self._mean_logit(out_ab, out_ba, "logit_pair_raw")
        logit_ppi = self._mean_logit(out_ab, out_ba, "logit_triage")

        # Native evidence maps in the caller's original A/B residue indexing.
        logit_res_a = 0.5 * (
            out_ab["logit_res_A"] + out_ba["logit_res_B"]
        )
        logit_res_b = 0.5 * (
            out_ab["logit_res_B"] + out_ba["logit_res_A"]
        )
        interface_logits = 0.5 * (
            out_ab["S_interface"]
            + out_ba["S_interface"].transpose(1, 2)
        )
        mask_a = self._input_mask(inputs.get("maskA"), inputs["resA"])
        mask_b = self._input_mask(inputs.get("maskB"), inputs["resB"])
        topk_a_values, topk_a_indices = select_topk_residues(
            logit_res_a, mask_a, self.model.cfg.topk,
            self.model.cfg.block_size, self.model.cfg.block_topr,
        )
        topk_b_values, topk_b_indices = select_topk_residues(
            logit_res_b, mask_b, self.model.cfg.topk,
            self.model.cfg.block_size, self.model.cfg.block_topr,
        )
        topm_indices, _ = select_topm_interface(
            interface_logits, mask_a, mask_b, self.model.cfg.topm,
            self.model.cfg.block_size,
        )
        batch_indices = torch.arange(
            interface_logits.size(0), device=interface_logits.device
        ).view(-1, 1)
        topm_logits = interface_logits[
            batch_indices, topm_indices[..., 0], topm_indices[..., 1]
        ]
        mil_logit = torch.logsumexp(topm_logits, dim=1) - math.log(
            max(1, topm_logits.size(1))
        )
        l2_projection_a, l2_projection_b = project_interface_to_residue(
            interface_logits, mask_a, mask_b
        )

        gates = {}
        for output_name, base_name in [
            ("g_L1", "g_res"), ("g_L2", "g_interface"), ("g_L3", "g_pair")
        ]:
            gates[output_name] = 0.5 * (
                out_ab["fusion_weights"][base_name]
                + out_ba["fusion_weights"][base_name]
            )

        # Averaged gate-logit contributions preserve the exact T-MEB identity:
        # logit_PPI = contribution_L1 + contribution_L2 + contribution_L3.
        contribution_l1 = 0.5 * (
            out_ab["fusion_weights"]["g_res"] * out_ab["logit_res"]
            + out_ba["fusion_weights"]["g_res"] * out_ba["logit_res"]
        )
        contribution_l2 = 0.5 * (
            out_ab["fusion_weights"]["g_interface"] * out_ab["logit_interface"]
            + out_ba["fusion_weights"]["g_interface"] * out_ba["logit_interface"]
        )
        contribution_l3 = 0.5 * (
            out_ab["fusion_weights"]["g_pair"] * out_ab["logit_pair_raw"]
            + out_ba["fusion_weights"]["g_pair"] * out_ba["logit_pair_raw"]
        )
        p_ab = out_ab["p_triage"]
        p_ba = out_ba["p_triage"]
        p_l1 = torch.sigmoid(logit_l1)
        p_l2 = torch.sigmoid(logit_l2)
        p_l3 = torch.sigmoid(logit_l3)
        p_ppi = torch.sigmoid(logit_ppi)

        # OSA-safe model diagnostics.  They contain no disease evidence and do
        # not alter p_PPI.  They quantify whether the three native evidence
        # levels agree, whether T-MEB contributions cancel, and whether the
        # unordered A-B/B-A evaluations are stable.
        branch_probabilities = torch.stack([p_l1, p_l2, p_l3], dim=1)
        branch_std = branch_probabilities.std(dim=1, unbiased=False)
        max_three_probability_std = math.sqrt(2.0) / 3.0
        tri_level_consensus = (
            1.0 - branch_std / max_three_probability_std
        ).clamp(0.0, 1.0)
        gate_stack = torch.stack(
            [gates["g_L1"], gates["g_L2"], gates["g_L3"]], dim=1
        ).clamp_min(1e-12)
        gate_entropy = (
            -(gate_stack * torch.log(gate_stack)).sum(dim=1) / math.log(3.0)
        ).clamp(0.0, 1.0)
        contribution_stack = torch.stack(
            [contribution_l1, contribution_l2, contribution_l3], dim=1
        )
        contribution_abs_sum = contribution_stack.abs().sum(dim=1)
        contribution_conflict = (
            1.0
            - contribution_stack.sum(dim=1).abs()
            / contribution_abs_sum.clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        order_gap = torch.abs(p_ab - p_ba)
        order_stability = (1.0 - order_gap).clamp(0.0, 1.0)
        cross_level_coherence = (
            tri_level_consensus
            * (1.0 - contribution_conflict)
            * order_stability
        ).clamp(0.0, 1.0)
        branch_direction_agreement = (
            torch.stack([logit_l1, logit_l2, logit_l3], dim=1)
            * logit_ppi.unsqueeze(1)
            >= 0.0
        ).to(dtype=logit_ppi.dtype).mean(dim=1)
        l1_l2_corr_a = self._masked_pearson(
            torch.sigmoid(logit_res_a), l2_projection_a, mask_a
        )
        l1_l2_corr_b = self._masked_pearson(
            torch.sigmoid(logit_res_b), l2_projection_b, mask_b
        )
        return {
            "native_evidence_schema_version": "1.3",
            "p_PPI": p_ppi,
            "logit_PPI": logit_ppi,
            "p_L1": p_l1,
            "p_L2": p_l2,
            "p_L3": p_l3,
            "logit_L1": logit_l1,
            "logit_L2": logit_l2,
            "logit_L3": logit_l3,
            "g_L1": gates["g_L1"],
            "g_L2": gates["g_L2"],
            "g_L3": gates["g_L3"],
            "contribution_L1": contribution_l1,
            "contribution_L2": contribution_l2,
            "contribution_L3": contribution_l3,
            "T_MEB_reconstruction_error": torch.abs(
                logit_ppi - contribution_l1 - contribution_l2 - contribution_l3
            ),
            "r_rel_diagnostic": 0.5 * (
                out_ab["evidence_reliability"]
                + out_ba["evidence_reliability"]
            ),
            "evidence_reliability": 0.5 * (
                out_ab["evidence_reliability"]
                + out_ba["evidence_reliability"]
            ),
            "logit_residue_A": logit_res_a,
            "logit_residue_B": logit_res_b,
            "p_residue_A": torch.sigmoid(logit_res_a),
            "p_residue_B": torch.sigmoid(logit_res_b),
            "topk_residue_A_indices": topk_a_indices,
            "topk_residue_B_indices": topk_b_indices,
            "topk_residue_A_logits": topk_a_values,
            "topk_residue_B_logits": topk_b_values,
            "S_interface": interface_logits,
            "p_interface_map": torch.sigmoid(interface_logits),
            "topm_interface_indices": topm_indices,
            "topm_interface_logits": topm_logits,
            "topm_interface_probabilities": torch.sigmoid(topm_logits),
            "L2_to_L1_projection_A": l2_projection_a,
            "L2_to_L1_projection_B": l2_projection_b,
            "L2_MIL_logit": mil_logit,
            "L2_MIL_probability": torch.sigmoid(mil_logit),
            "z_L1": 0.5 * (out_ab["z_res"] + out_ba["z_res"]),
            "z_L2": 0.5 * (out_ab["z_interface"] + out_ba["z_interface"]),
            "z_L3": 0.5 * (out_ab["z_pair"] + out_ba["z_pair"]),
            "p_PPI_AB": p_ab,
            "p_PPI_BA": p_ba,
            "AB_BA_probability_gap": order_gap,
            "AB_BA_logit_gap": torch.abs(
                out_ab["logit_triage"] - out_ba["logit_triage"]
            ),
            "AB_BA_L1_probability_gap": torch.abs(
                torch.sigmoid(out_ab["logit_res"])
                - torch.sigmoid(out_ba["logit_res"])
            ),
            "AB_BA_L2_probability_gap": torch.abs(
                torch.sigmoid(out_ab["logit_interface"])
                - torch.sigmoid(out_ba["logit_interface"])
            ),
            "AB_BA_L3_probability_gap": torch.abs(
                out_ab["p_pair_raw"] - out_ba["p_pair_raw"]
            ),
            "tri_level_probability_std": branch_std,
            "tri_level_consensus_diagnostic": tri_level_consensus,
            "T_MEB_gate_entropy_normalized": gate_entropy,
            "T_MEB_contribution_conflict": contribution_conflict,
            "branch_direction_agreement": branch_direction_agreement,
            "order_stability_diagnostic": order_stability,
            "cross_level_coherence_diagnostic": cross_level_coherence,
            "fusion_margin_diagnostic": (2.0 * torch.abs(p_ppi - 0.5)).clamp(0.0, 1.0),
            "L1_L2_projection_correlation_A": l1_l2_corr_a,
            "L1_L2_projection_correlation_B": l1_l2_corr_b,
            "L1_L2_projection_correlation_mean": 0.5 * (
                l1_l2_corr_a + l1_l2_corr_b
            ),
        }


def geometric_percentile_priority(
    ppi_percentile: float,
    osa_percentile: float,
    ppi_weight: float = 0.5,
) -> float:
    """Combine already-frozen percentiles without calling the result p_PPI."""
    if not 0.0 <= ppi_weight <= 1.0:
        raise ValueError("ppi_weight must be in [0,1]")
    ppi = min(1.0, max(1e-12, float(ppi_percentile)))
    osa = min(1.0, max(1e-12, float(osa_percentile)))
    return math.exp(ppi_weight * math.log(ppi) + (1.0 - ppi_weight) * math.log(osa))


def checkpoint_metadata_json(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=True, sort_keys=True)
