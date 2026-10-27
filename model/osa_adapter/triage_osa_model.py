#!/usr/bin/env python3
"""OSA-specialized TRIAGE-Fuse model components.

This is the disease-specific continuation of the original TRIAGE L1-L2-L3
model.  The pretrained backbone remains immutable and is stored together with
the trainable OSA T-MEB specialization in one checkpoint state dictionary.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_VERSION = "4.0.0"
EXPERIMENT_VARIANTS = (
    "full", "no_patient", "no_context", "no_gate_delta", "no_residual",
    "equal_fusion", "static_fusion",
)
BACKBONE_PREFIX = "triage_backbone."
OSA_PREFIX = "osa_specialization."

BASE_FEATURE_COLUMNS = [
    "logit_PPI", "logit_L1", "logit_L2", "logit_L3",
    "g_L1", "g_L2", "g_L3", "evidence_reliability",
    "order_stability_diagnostic", "cross_level_coherence_diagnostic",
    "fusion_margin_diagnostic", "T_MEB_gate_entropy_normalized",
    "tri_level_probability_std", "L2_MIL_probability",
]

CONTEXT_FEATURE_COLUMNS_V3_2 = [
    "OSA_strength_min", "OSA_strength_geomean", "OSA_strength_mean",
    "OSA_strength_absdiff", "severity_strength_min",
    "severity_strength_geomean", "AHI_strength_min",
    "AHI_strength_geomean", "IH_strength_min", "IH_strength_geomean",
    "adult_signed_product", "adult_direction_agreement",
    "IH_signed_product", "IH_direction_agreement",
    "cross_axis_direction_agreement", "context_coverage_min",
    "context_coverage_mean", "context_anchor",
]

PATIENT_INFORMATION_FEATURE_COLUMNS = [
    "hypertension_strength_min", "hypertension_strength_geomean",
    "EDS_strength_min", "EDS_strength_geomean",
    "clinical_manifestation_strength_min",
    "clinical_manifestation_strength_geomean",
    "clinical_direction_agreement",
    "demographic_confounding_strength_max",
    "demographic_confounding_strength_geomean",
    "patient_information_coverage_min",
    "patient_information_coverage_mean",
    "clinical_information_anchor",
]

CONTEXT_FEATURE_COLUMNS = (
    CONTEXT_FEATURE_COLUMNS_V3_2 + PATIENT_INFORMATION_FEATURE_COLUMNS
)
INPUT_FEATURE_COLUMNS_V3_2 = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS_V3_2
INPUT_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS


class TRIAGEOSAModel(nn.Module):
    """OSA-conditioned T-MEB specialization over frozen TRIAGE evidence.

    L1, L2 and L3 branch logits and the original gates are inputs produced by
    the strictly loaded backbone.  Only this module is trainable on OSA
    discovery data.  The output is an interaction-priority score rather than
    a physical-PPI probability.
    """

    def __init__(
        self,
        input_mean,
        input_std,
        hidden_dim=32,
        dropout=0.10,
        gate_delta_limit=0.75,
        residual_limit=0.25,
        branch_floor=0.10,
        expert_floors=(0.01, 0.01, 0.01),
        minimum_structural_mass=0.10,
        residual_scale=0.10,
        input_feature_columns=None,
        experiment_variant="full",
    ):
        super(TRIAGEOSAModel, self).__init__()
        if experiment_variant not in EXPERIMENT_VARIANTS:
            raise ValueError("Unknown experiment variant: " + str(experiment_variant))
        self.experiment_variant = str(experiment_variant)
        input_mean = torch.as_tensor(input_mean, dtype=torch.float32)
        input_std = torch.as_tensor(input_std, dtype=torch.float32)
        if input_mean.ndim != 1 or input_std.ndim != 1:
            raise ValueError("input_mean and input_std must be one-dimensional")
        self.input_feature_columns = list(
            input_feature_columns or INPUT_FEATURE_COLUMNS
        )
        if input_mean.numel() != len(self.input_feature_columns):
            raise ValueError(
                "Expected {} normalization features, observed {}".format(
                    len(self.input_feature_columns), input_mean.numel()
                )
            )
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std.clamp_min(1e-6))
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.gate_delta_limit = float(gate_delta_limit)
        self.residual_limit = float(residual_limit)
        self.branch_floor = float(branch_floor)
        self.residual_scale = float(residual_scale)
        self.minimum_structural_mass = float(minimum_structural_mass)
        expert_floors = torch.as_tensor(expert_floors, dtype=torch.float32)
        if expert_floors.numel() != 3 or float(expert_floors.sum()) >= 1.0:
            raise ValueError("expert_floors must contain three nonnegative values summing below one")
        if bool((expert_floors < 0.0).any().item()):
            raise ValueError("expert_floors must be nonnegative")
        if self.minimum_structural_mass < float(expert_floors[:2].sum()):
            raise ValueError(
                "minimum_structural_mass cannot be below the two structural expert floors"
            )
        if self.minimum_structural_mass + float(expert_floors[2]) >= 1.0:
            raise ValueError(
                "minimum_structural_mass plus context floor must be below one"
            )
        self.register_buffer("expert_floors", expert_floors)

        self.context_encoder = nn.Sequential(
            nn.Linear(len(self.input_feature_columns), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.osa_gate_delta = nn.Linear(self.hidden_dim, 3)
        self.expert_gate = nn.Linear(self.hidden_dim, 3)
        self.osa_residual = nn.Linear(self.hidden_dim, 1)
        self.osa_reliability = nn.Linear(self.hidden_dim, 1)

        # Neutral initialization: expert proportions are learned per pair from
        # discovery-only reliability features rather than seeded with a fixed
        # OSA/PPI mixture.
        nn.init.zeros_(self.osa_gate_delta.weight)
        nn.init.zeros_(self.osa_gate_delta.bias)
        nn.init.zeros_(self.osa_residual.weight)
        nn.init.zeros_(self.osa_residual.bias)
        nn.init.zeros_(self.expert_gate.weight)
        nn.init.zeros_(self.expert_gate.bias)
        nn.init.zeros_(self.osa_reliability.weight)
        nn.init.constant_(self.osa_reliability.bias, -1.0)
        self.feature_index = dict(
            (name, index) for index, name in enumerate(self.input_feature_columns)
        )
        if self.experiment_variant == "no_gate_delta":
            for parameter in self.osa_gate_delta.parameters():
                parameter.requires_grad = False
        if self.experiment_variant == "no_residual":
            for module in (self.osa_residual, self.osa_reliability):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        # No new parameters for the full V4 model: old checkpoints strict-load.
        if self.experiment_variant in ("equal_fusion", "static_fusion"):
            for parameter in self.parameters():
                parameter.requires_grad = False
            if self.experiment_variant == "static_fusion":
                self.expert_gate.bias.requires_grad = True

    def export_config(self):
        return {
            "model_version": MODEL_VERSION,
            "experiment_variant": self.experiment_variant,
            "input_feature_columns": list(self.input_feature_columns),
            "hidden_dim": int(self.hidden_dim),
            "dropout": float(self.dropout_rate),
            "gate_delta_limit": float(self.gate_delta_limit),
            "residual_limit": float(self.residual_limit),
            "branch_floor": float(self.branch_floor),
            "expert_floors": self.expert_floors.detach().cpu().tolist(),
            "minimum_structural_mass": float(self.minimum_structural_mass),
            "residual_scale": float(self.residual_scale),
            "input_mean": self.input_mean.detach().cpu().tolist(),
            "input_std": self.input_std.detach().cpu().tolist(),
        }

    @classmethod
    def from_config(cls, config):
        expected = list(config.get("input_feature_columns", []))
        supported = (list(INPUT_FEATURE_COLUMNS_V3_2), list(INPUT_FEATURE_COLUMNS))
        if expected not in supported:
            raise RuntimeError("OSA model feature schema does not match checkpoint")
        return cls(
            input_mean=config["input_mean"],
            input_std=config["input_std"],
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
            gate_delta_limit=float(config.get("gate_delta_limit", 0.75)),
            residual_limit=float(config.get("residual_limit", 0.25)),
            branch_floor=float(config.get("branch_floor", 0.10)),
            expert_floors=tuple(config.get("expert_floors", (0.01, 0.01, 0.01))),
            minimum_structural_mass=float(
                config.get("minimum_structural_mass", 0.10)
            ),
            residual_scale=float(config.get("residual_scale", 0.10)),
            input_feature_columns=expected,
            experiment_variant=config.get("experiment_variant", "full"),
        )

    @staticmethod
    def unified_state_dict(backbone_state, osa_state):
        state = {}
        for key, value in backbone_state.items():
            state[BACKBONE_PREFIX + str(key)] = value
        for key, value in osa_state.items():
            state[OSA_PREFIX + str(key)] = value
        return state

    @staticmethod
    def split_unified_state_dict(state):
        backbone = {}
        osa = {}
        for key, value in state.items():
            key = str(key)
            if key.startswith(BACKBONE_PREFIX):
                backbone[key[len(BACKBONE_PREFIX):]] = value
            elif key.startswith(OSA_PREFIX):
                osa[key[len(OSA_PREFIX):]] = value
        if not backbone or not osa:
            raise RuntimeError("Unified TRIAGE-OSA checkpoint is incomplete")
        return backbone, osa

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.input_feature_columns):
            raise ValueError(
                "Expected feature matrix [N, {}], observed {}".format(
                    len(self.input_feature_columns), tuple(features.shape)
                )
            )
        normalized = torch.clamp(
            (features - self.input_mean) / self.input_std, min=-8.0, max=8.0
        )
        hidden = self.context_encoder(normalized)
        idx = self.feature_index
        base_logit = features[:, idx["logit_PPI"]]
        branch_logits = torch.stack(
            [
                features[:, idx["logit_L1"]],
                features[:, idx["logit_L2"]],
                features[:, idx["logit_L3"]],
            ],
            dim=1,
        )
        base_gates = torch.stack(
            [
                features[:, idx["g_L1"]],
                features[:, idx["g_L2"]],
                features[:, idx["g_L3"]],
            ],
            dim=1,
        ).clamp_min(1e-6)
        base_gates = base_gates / base_gates.sum(dim=1, keepdim=True)

        gate_delta = self.gate_delta_limit * torch.tanh(
            self.osa_gate_delta(hidden)
        )
        if self.experiment_variant in ("no_gate_delta", "equal_fusion", "static_fusion"):
            gate_delta = gate_delta * 0.0
        raw_osa_gates = F.softmax(torch.log(base_gates) + gate_delta, dim=1)
        osa_gates = self.branch_floor + (1.0 - 3.0 * self.branch_floor) * raw_osa_gates
        tri_level_logit = (osa_gates * branch_logits).sum(dim=1)
        context_residual = self.residual_limit * torch.tanh(
            self.osa_residual(hidden).squeeze(1)
        )
        if self.experiment_variant in ("no_residual", "equal_fusion", "static_fusion"):
            context_residual = context_residual * 0.0
        reliability_blend = 0.10 + 0.80 * torch.sigmoid(
            self.osa_reliability(hidden).squeeze(1)
        )
        context_anchor = features[:, idx["context_anchor"]].clamp(1e-5, 1.0 - 1e-5)
        context_anchor_logit = torch.logit(context_anchor)
        raw_expert_weights = F.softmax(self.expert_gate(hidden), dim=1)
        # V3.2.2 constrained adaptive simplex.  Context may vary freely up to
        # 1 - minimum_structural_mass, while the structural share is split
        # pair-by-pair between frozen PPI and adaptive L1-L2-L3.  This is a
        # lower-bound constraint, not a fixed expert mixture.
        context_floor = self.expert_floors[2]
        context_weight = context_floor + (
            1.0 - self.minimum_structural_mass - context_floor
        ) * raw_expert_weights[:, 2]
        structural_mass = 1.0 - context_weight
        structural_split = raw_expert_weights[:, :2]
        structural_split = structural_split / structural_split.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        structural_free_mass = (
            structural_mass.unsqueeze(1) - self.expert_floors[:2].sum()
        )
        structural_weights = self.expert_floors[:2] + (
            structural_free_mass * structural_split
        )
        expert_weights = torch.cat(
            [structural_weights, context_weight.unsqueeze(1)], dim=1
        )
        if self.experiment_variant == "equal_fusion":
            expert_weights = torch.ones_like(expert_weights) / 3.0
        elif self.experiment_variant == "static_fusion":
            expert_weights = F.softmax(self.expert_gate.bias, dim=0).unsqueeze(0).expand_as(expert_weights)
        structural_mass = expert_weights[:, :2].sum(dim=1)
        expert_logits = torch.stack(
            [base_logit, tri_level_logit, context_anchor_logit], dim=1
        )
        # Convex expert fusion prevents the global sign inversion observed in
        # V2.  The small residual is reliability bounded and cannot replace
        # any of the three locked experts.
        osa_logit = (expert_weights * expert_logits).sum(dim=1)
        osa_logit = osa_logit + self.residual_scale * reliability_blend * context_residual
        return {
            "adapted_logit": osa_logit,
            "adapted_probability": torch.sigmoid(osa_logit),
            "base_logit": base_logit,
            "base_gates": base_gates,
            "context_gates": osa_gates,
            "gate_delta": gate_delta,
            "tri_level_logit": tri_level_logit,
            "context_residual": context_residual,
            "reliability_blend": reliability_blend,
            "context_anchor_logit": context_anchor_logit,
            "expert_weights": expert_weights,
            "structural_expert_mass": structural_mass,
        }
