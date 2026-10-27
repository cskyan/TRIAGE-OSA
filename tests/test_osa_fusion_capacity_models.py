"""Model contract tests; generated fixtures are not study results."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model/osa_adapter"))
from osa_fusion_capacity_models import GenericFusionMLP, count_trainable_parameters
from triage_osa_model import INPUT_FEATURE_COLUMNS, TRIAGEOSAModel


class GenericFusionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        rng = np.random.RandomState(1729)
        self.mean = np.zeros(44, dtype=np.float32)
        self.std = np.ones(44, dtype=np.float32)
        values = rng.normal(size=(37, 44)).astype(np.float32)
        values[:, 4:7] = rng.uniform(0.0, 1.0, size=(37, 3))
        values[:, INPUT_FEATURE_COLUMNS.index("context_anchor")] = rng.uniform(0.05, 0.95, 37)
        self.features = torch.from_numpy(values)

    def test_parameter_counts_no_dummy_parameters(self):
        full = TRIAGEOSAModel(self.mean, self.std)
        self.assertEqual(count_trainable_parameters(full), 2824)
        for skip in (False, True):
            matched = GenericFusionMLP(self.mean, self.std, hidden_dim=34, residual_skip=skip)
            wide = GenericFusionMLP(self.mean, self.std, hidden_dim=55, residual_skip=skip)
            self.assertEqual(count_trainable_parameters(matched), 2823)
            self.assertEqual(count_trainable_parameters(wide), 5721)
            self.assertTrue(all(parameter.requires_grad for parameter in matched.parameters()))

    def test_neutral_skip_reproduces_full_initial_score(self):
        for branch_floor, mass, floors in ((0.1, 0.1, (0.01, 0.01, 0.01)),
                                           (0.05, 0.2, (0.02, 0.03, 0.04))):
            full = TRIAGEOSAModel(self.mean, self.std, branch_floor=branch_floor,
                                 minimum_structural_mass=mass, expert_floors=floors).eval()
            generic = GenericFusionMLP(self.mean, self.std, residual_skip=True,
                                       branch_floor=branch_floor, minimum_structural_mass=mass,
                                       component_floors=floors).eval()
            with torch.no_grad():
                actual = generic(self.features)
                expected = full(self.features)["adapted_logit"]
            torch.testing.assert_close(actual["adapted_logit"], expected, atol=1e-7, rtol=1e-6)
            self.assertEqual(set(actual), {"adapted_logit", "adapted_probability"})

    def test_all_trainable_parameters_receive_active_gradients(self):
        target = torch.linspace(0.05, 0.95, self.features.shape[0])
        for skip in (False, True):
            model = GenericFusionMLP(self.mean, self.std, dropout=0.0, residual_skip=skip)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
            active = dict((name, False) for name, _ in model.named_parameters())
            for _ in range(5):
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(self.features)["adapted_logit"], target)
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                    active[name] = active[name] or bool((parameter.grad.abs() > 0).any())
                optimizer.step()
            self.assertTrue(all(active.values()), active)

    def test_config_and_state_roundtrip(self):
        for skip in (False, True):
            model = GenericFusionMLP(self.mean, self.std, residual_skip=skip).eval()
            with torch.no_grad():
                model.score_head.weight.normal_(0.0, 0.1)
                model.score_head.bias.fill_(0.2)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = GenericFusionMLP.from_config(model.export_config()).eval()
            restored.load_state_dict(torch.load(stream, map_location="cpu"), strict=True)
            self.assertEqual(model.export_config(), restored.export_config())
            self.assertTrue(torch.equal(model(self.features)["adapted_logit"],
                                        restored(self.features)["adapted_logit"]))

    def test_normalization_and_finite_outputs(self):
        mean = np.linspace(-1.0, 1.0, 44, dtype=np.float32)
        std = np.linspace(0.0, 2.0, 44, dtype=np.float32)
        model = GenericFusionMLP(mean, std).eval()
        expected = torch.clamp((self.features - torch.from_numpy(mean)) /
                               torch.from_numpy(std).clamp_min(1e-6), -8.0, 8.0)
        self.assertTrue(torch.equal(expected, model.normalized_input(self.features)))
        self.assertEqual(tuple(model(self.features)["adapted_logit"].shape), (37,))
        self.assertTrue(bool(torch.isfinite(model(self.features)["adapted_probability"]).all()))
        self.assertTrue(torch.equal(model(self.features)["adapted_logit"], torch.zeros(37)))

    def test_reject_invalid_schema_and_values(self):
        invalid_columns = list(INPUT_FEATURE_COLUMNS)
        invalid_columns[0], invalid_columns[1] = invalid_columns[1], invalid_columns[0]
        with self.assertRaises(ValueError):
            GenericFusionMLP(self.mean, self.std, input_feature_columns=invalid_columns)
        for kwargs in ({"hidden_dim": 0}, {"hidden_dim": 2.5}, {"dropout": 1.0},
                       {"residual_skip": "false"}, {"branch_floor": 0.4},
                       {"minimum_structural_mass": 0.0},
                       {"component_floors": (0.01, 0.01)},
                       {"component_floors": (0.01, float("nan"), 0.01)}):
            with self.assertRaises(ValueError):
                GenericFusionMLP(self.mean, self.std, **kwargs)
        with self.assertRaises(ValueError):
            GenericFusionMLP(self.mean[:-1], self.std)
        bad_mean = self.mean.copy()
        bad_mean[0] = np.nan
        with self.assertRaises(ValueError):
            GenericFusionMLP(bad_mean, self.std)
        model = GenericFusionMLP(self.mean, self.std)
        for values in (torch.ones(5, 43), torch.ones(44), torch.ones(2, 44, dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(values)
        bad = self.features.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(bad)
        config = model.export_config()
        config["model_type"] = "incorrect"
        with self.assertRaises(ValueError):
            GenericFusionMLP.from_config(config)
        config = model.export_config()
        config["unexpected"] = True
        with self.assertRaises(ValueError):
            GenericFusionMLP.from_config(config)


if __name__ == "__main__":
    unittest.main()
