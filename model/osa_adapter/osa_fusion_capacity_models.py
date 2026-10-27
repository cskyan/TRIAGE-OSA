"""Generic score models for controlled OSA fusion comparisons.

These comparators reuse the same scalar inputs as the specialization model.
They do not expose learned branch weights or molecular-mass quantities.
"""
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from triage_osa_model import INPUT_FEATURE_COLUMNS


MODEL_TYPE = "generic_fusion_mlp"
MODEL_VERSION = "1.0.0"


def count_trainable_parameters(model):
    """Count parameters participating in optimization, excluding buffers."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


class GenericFusionMLP(nn.Module):
    """Two-layer scalar predictor with an optional deterministic starting score.

    Width 34 gives 2,823 trainable parameters for the 44-variable schema.
    The optional skip equals the neutral, zero-head TRIAGEOSAModel score.
    It is fixed, adds no trainable parameters, and does not constrain the
    learned correction or provide a molecular-mass interpretation.
    """

    def __init__(self, input_mean, input_std, hidden_dim=34, dropout=0.10,
                 residual_skip=False, input_feature_columns=None,
                 branch_floor=0.10, minimum_structural_mass=0.10,
                 component_floors=(0.01, 0.01, 0.01)):
        super(GenericFusionMLP, self).__init__()
        columns = list(INPUT_FEATURE_COLUMNS if input_feature_columns is None
                       else input_feature_columns)
        if columns != list(INPUT_FEATURE_COLUMNS):
            raise ValueError("Generic fusion requires the ordered 44-variable input schema")
        mean = torch.as_tensor(input_mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(input_std, dtype=torch.float32).detach().clone()
        if (mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape
                or mean.numel() != len(columns)):
            raise ValueError("Normalization vectors must match the 44-variable schema")
        if not bool(torch.isfinite(mean).all().item()) or not bool(torch.isfinite(std).all().item()):
            raise ValueError("Normalization vectors must be finite")
        if bool((std < 0).any().item()):
            raise ValueError("Normalization standard deviations cannot be negative")
        if isinstance(hidden_dim, bool) or int(hidden_dim) != hidden_dim or int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(residual_skip, bool):
            raise ValueError("residual_skip must be a boolean")
        if not math.isfinite(float(branch_floor)) or not 0.0 <= float(branch_floor) < 1.0 / 3.0:
            raise ValueError("branch_floor must be finite and in [0, 1/3)")
        floors = torch.as_tensor(component_floors, dtype=torch.float32).detach().clone()
        if (floors.ndim != 1 or floors.numel() != 3
                or not bool(torch.isfinite(floors).all().item())
                or bool((floors < 0.0).any().item())
                or float(floors.sum()) >= 1.0):
            raise ValueError("component_floors must be three finite, nonnegative values summing below one")
        if (not math.isfinite(float(minimum_structural_mass))
                or float(minimum_structural_mass) < float(floors[:2].sum())
                or float(minimum_structural_mass) + float(floors[2]) >= 1.0):
            raise ValueError("minimum_structural_mass is incompatible with the component floors")

        self.input_feature_columns = columns
        self.feature_index = dict((name, index) for index, name in enumerate(columns))
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.residual_skip = residual_skip
        self.branch_floor = float(branch_floor)
        self.minimum_structural_mass = float(minimum_structural_mass)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std.clamp_min(1e-6))
        self.register_buffer("component_floors", floors)
        self.encoder = nn.Sequential(
            nn.Linear(len(columns), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.score_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def _validate_features(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.input_feature_columns):
            raise ValueError("Expected feature matrix [N, 44]")
        if not torch.is_floating_point(features):
            raise ValueError("Feature matrix must be floating point")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("Feature matrix must be finite")

    def normalized_input(self, features):
        """Apply the native training-standardization and clipping operation."""
        self._validate_features(features)
        return torch.clamp((features - self.input_mean) / self.input_std,
                           min=-8.0, max=8.0)

    def neutral_full_logit(self, features):
        """Return the native model's deterministic zero-head starting score."""
        self._validate_features(features)
        index = self.feature_index
        branches = torch.stack([features[:, index[name]] for name in
                                ("logit_L1", "logit_L2", "logit_L3")], dim=1)
        gates = torch.stack([features[:, index[name]] for name in
                             ("g_L1", "g_L2", "g_L3")], dim=1).clamp_min(1e-6)
        gates = gates / gates.sum(dim=1, keepdim=True)
        gates = self.branch_floor + (1.0 - 3.0 * self.branch_floor) * F.softmax(
            torch.log(gates), dim=1)
        tri_logit = (gates * branches).sum(dim=1)
        anchor = features[:, index["context_anchor"]].clamp(1e-5, 1.0 - 1e-5)
        anchor_logit = torch.logit(anchor)
        raw_weights = F.softmax(features.new_zeros((features.shape[0], 3)), dim=1)
        context_weight = self.component_floors[2] + (
            1.0 - self.minimum_structural_mass - self.component_floors[2]
        ) * raw_weights[:, 2]
        molecular_mass = 1.0 - context_weight
        molecular_split = raw_weights[:, :2]
        molecular_split = molecular_split / molecular_split.sum(dim=1, keepdim=True)
        molecular_weights = self.component_floors[:2] + (
            molecular_mass.unsqueeze(1) - self.component_floors[:2].sum()
        ) * molecular_split
        weights = torch.cat([molecular_weights, context_weight.unsqueeze(1)], dim=1)
        components = torch.stack([features[:, index["logit_PPI"]], tri_logit,
                                  anchor_logit], dim=1)
        return (weights * components).sum(dim=1)

    def forward(self, features):
        logit = self.score_head(self.encoder(self.normalized_input(features))).squeeze(1)
        if self.residual_skip:
            logit = logit + self.neutral_full_logit(features)
        return {"adapted_logit": logit, "adapted_probability": torch.sigmoid(logit)}

    def export_config(self):
        return {
            "model_type": MODEL_TYPE,
            "model_version": MODEL_VERSION,
            "input_feature_columns": list(self.input_feature_columns),
            "input_mean": self.input_mean.detach().cpu().tolist(),
            "input_std": self.input_std.detach().cpu().tolist(),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "residual_skip": self.residual_skip,
            "branch_floor": self.branch_floor,
            "minimum_structural_mass": self.minimum_structural_mass,
            "component_floors": self.component_floors.detach().cpu().tolist(),
        }

    @classmethod
    def from_config(cls, config):
        required = {"model_type", "model_version", "input_feature_columns", "input_mean",
                    "input_std", "hidden_dim", "dropout", "residual_skip", "branch_floor",
                    "minimum_structural_mass", "component_floors"}
        if set(config) != required:
            raise ValueError("Generic fusion checkpoint configuration is incomplete or contains unknown fields")
        if config["model_type"] != MODEL_TYPE or config["model_version"] != MODEL_VERSION:
            raise ValueError("Unsupported generic fusion checkpoint type or version")
        options = dict((key, value) for key, value in config.items()
                       if key not in ("model_type", "model_version"))
        return cls(**options)
