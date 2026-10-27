"""Synthetic protocol tests for fusion controls, not experimental results."""
import itertools
import json
import math
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model/osa_adapter"))
import osa_fusion_capacity_experiments as experiment
from osa_fusion_capacity_models import GenericFusionMLP
from triage_osa_model import INPUT_FEATURE_COLUMNS

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "fusion_capacity_unit_tests"


def workspace_temporary_directory():
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    # Keep temporary test artifacts in the ignored workspace directory.
    return tempfile.TemporaryDirectory(dir=str(TEST_TEMP_ROOT.resolve()))


class FusionCapacityExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.RandomState(381)
        self.features = rng.normal(size=(48, 44)).astype(np.float32)
        self.features[:, 4:7] = rng.uniform(0.05, 0.9, (48, 3))
        self.target = rng.uniform(0.05, 0.95, 48).astype(np.float32)
        self.features[:, INPUT_FEATURE_COLUMNS.index("context_anchor")] = self.target
        self.train = np.arange(32)
        self.validation = np.arange(32, 40)
        self.optimization = dict(weight_decay=0.0001, batch_size=8, max_epochs=3,
                                 patience=3, gradient_clip=5.0, minimum_improvement=1e-6)
        self.objective = dict(bce_weight=1.0, ranking_weight=0.25,
                              minimum_ranking_target_gap=0.1)
        self.device = torch.device("cpu")

    def fit(self, spec, fixed_epochs=None):
        return experiment.fit_common(
            self.features, self.target, self.train,
            None if fixed_epochs is not None else self.validation,
            spec, self.optimization, self.objective, 73, 0.001, self.device,
            torch, fixed_epochs=fixed_epochs, prefix="[synthetic-test]")

    def test_common_fit_and_train_only_normalization(self):
        # Held-out values must not change the fitted normalization vectors.
        self.features[32:, :] += 3.0
        self.features[32:, INPUT_FEATURE_COLUMNS.index("context_anchor")] = self.target[32:]
        for spec in (dict(architecture="structured", hidden_dim=32),
                     dict(architecture="generic", hidden_dim=34, residual_skip=False),
                     dict(architecture="generic", hidden_dim=34, residual_skip=True)):
            model, history, epoch, score = self.fit(spec)
            self.assertGreaterEqual(epoch, 1)
            self.assertLessEqual(epoch, len(history))
            self.assertTrue(math.isfinite(score))
            np.testing.assert_array_equal(model.input_mean.detach().numpy(),
                                          self.features[self.train].mean(axis=0))
            expected_std = self.features[self.train].std(axis=0)
            expected_std = np.where(expected_std > 1e-6, expected_std, 1.0)
            np.testing.assert_array_equal(model.input_std.detach().numpy(), expected_std)
            model.eval()
            self.assertTrue(bool(torch.isfinite(
                model(torch.from_numpy(self.features))["adapted_logit"]).all()))

    def test_common_fit_reproducible_and_checkpoint_roundtrip(self):
        spec = dict(architecture="generic", hidden_dim=34, residual_skip=True)
        first, h1, epoch1, score1 = self.fit(spec)
        second, h2, epoch2, score2 = self.fit(spec)
        self.assertEqual(epoch1, epoch2)
        self.assertEqual(score1, score2)
        self.assertEqual([r["train_loss"] for r in h1], [r["train_loss"] for r in h2])
        for key in first.state_dict():
            self.assertTrue(torch.equal(first.state_dict()[key], second.state_dict()[key]), key)
        with workspace_temporary_directory() as directory:
            path = Path(directory) / "adapter.pt"
            experiment.save_model(path, first, {"synthetic_test": True}, torch)
            restored = experiment.restore_model(path, self.device, torch)
            payload = experiment.load_checkpoint(path, torch)
            self.assertFalse(payload["eligible_for_production"])
            self.assertTrue(payload["metadata"]["synthetic_test"])
            first.eval()
            restored.eval()
            self.assertTrue(torch.equal(first(torch.from_numpy(self.features))["adapted_logit"],
                                        restored(torch.from_numpy(self.features))["adapted_logit"]))

    def test_fixed_refit_uses_requested_epochs_without_validation(self):
        spec = dict(architecture="generic", hidden_dim=34, residual_skip=False)
        model, history, epoch, score = self.fit(spec, fixed_epochs=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(epoch, 2)
        self.assertEqual(score, 2.0)
        self.assertTrue(all(math.isnan(r["validation_soft_BCE"]) for r in history))
        self.assertTrue(all(math.isnan(r["validation_Spearman"]) for r in history))

    def test_ranking_loss_handles_empty_mask_and_direction(self):
        z = torch.tensor([-1.0, 1.0], requires_grad=True)
        permutation = torch.tensor([1, 0])
        target = torch.tensor([0.1, 0.9])
        loss = experiment.ranking_loss(z, target, permutation, 0.1, torch)
        expected = torch.nn.functional.softplus(torch.tensor(-2.0))
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=7)
        loss.backward()
        self.assertTrue(bool((z.grad.abs() > 0).all()))
        z = torch.tensor([-1.0, 1.0], requires_grad=True)
        loss = experiment.ranking_loss(z, torch.full((2,), 0.5), permutation, 0.1, torch)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.equal(z.grad, torch.zeros(2)))

    def test_safe_output_and_prediction_scopes(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory).resolve()
            allowed = root / experiment.BASE / "50_synthetic_test"
            self.assertEqual(experiment.safe_output(root, allowed), allowed)
            for path in (root, root / experiment.BASE / "31_old_result",
                         root / experiment.BASE / "50_x" / ".." / ".." / "elsewhere"):
                with self.assertRaises(RuntimeError):
                    experiment.safe_output(root, path)
        pairs = pd.DataFrame(dict(pair_id=["p%d" % i for i in range(48)],
                                  protein_A=["a%d" % i for i in range(48)],
                                  protein_B=["b%d" % i for i in range(48)]))
        splits = {"train": self.train, "validation": self.validation,
                  "test": np.arange(40, 48)}
        model = GenericFusionMLP(np.zeros(44), np.ones(44))
        primary = np.arange(48) % 2 == 0
        rows, predictions = experiment.prediction_tables(
            model, self.features, self.target, 1.0 - self.target, pairs, splits,
            primary, {"method": "synthetic"}, 8, self.device, torch)
        self.assertEqual(len(rows), 6)
        self.assertEqual(set(rows.scope), {"all_500", "primary_250"})
        self.assertEqual(len(predictions), 48)
        self.assertFalse(predictions.pair_id.duplicated().any())
        np.testing.assert_array_equal(predictions.primary_pair, primary)

    def test_final_refit_selects_lr_and_epochs_from_validation_only(self):
        # The real 31,125-row cardinality is generated locally; no study inputs
        # are opened and fit/prediction are mocked to inspect protocol choices.
        proteins = ["P%03d" % i for i in range(250)]
        combinations = list(itertools.combinations(proteins, 2))
        pairs = pd.DataFrame(dict(pair_id=["p%05d" % i for i in range(len(combinations))],
                                  protein_A=[a for a, _ in combinations],
                                  protein_B=[b for _, b in combinations]))
        features = np.zeros((len(pairs), 44), dtype=np.float32)
        context = pd.DataFrame({"context_anchor": np.full(len(pairs), 0.4, dtype=np.float32)})
        metadata = {"GSE75097": pd.DataFrame({"sample_id": ["toy_adult"]}),
                    "GSE71356": pd.DataFrame({"subject_id": ["toy_ih"]})}
        models = {"Full": dict(architecture="structured", hidden_dim=32, objective="common"),
                  "Concat_matched": dict(architecture="generic", hidden_dim=34,
                                         residual_skip=False, objective="common")}
        cv_rows = []
        for name in models:
            for rate in (0.001, 0.01):
                for fold, epoch in enumerate((2, 4, 6, 8), 1):
                    value = (0.8 if rate == 0.01 else 0.7) if name == "Full" else 0.6
                    cv_rows.append(dict(method=name, learning_rate=rate, fold=fold, seed=17,
                                        best_epoch=epoch, validation_utility=value,
                                        test_utility=1000 if rate == 0.001 else -1000))
        cv = pd.DataFrame(cv_rows)
        calls = []

        def fake_fit(*args, **kwargs):
            calls.append(dict(architecture=args[4]["architecture"], rate=float(args[8]),
                              seed=int(args[7]), epochs=int(kwargs["fixed_epochs"])))
            model = GenericFusionMLP(np.zeros(44), np.ones(44), hidden_dim=2)
            epochs = int(kwargs["fixed_epochs"])
            return model, [{"epoch": i} for i in range(1, epochs + 1)], epochs, float(epochs)

        def fake_predict(model, values, indices, *unused):
            return np.asarray(indices, dtype=np.float32) / max(1.0, float(len(values)))

        with workspace_temporary_directory() as directory:
            root = Path(directory)
            out = root / "out"
            out.mkdir()
            source = root / "source_suite"
            source.mkdir()
            (source / "experiment_contract.json").write_text(json.dumps({"optimization": {}}), encoding="utf-8")
            pairs.to_csv(root / "pairs.tsv", sep="\t", index=False)
            order = root / experiment.SOURCE_ORDER
            order.parent.mkdir(parents=True)
            pairs.to_csv(order, sep="\t", index=False)
            config = dict(source_suite="source_suite", pair_scores="pairs.tsv", models=models,
                          seeds=[17, 19], common_optimization=self.optimization,
                          common_objective=self.objective)
            with patch("osa_context_data.load_discovery_inputs", return_value=({}, metadata, {})), \
                 patch("osa_context_data.compute_node_evidence", return_value=None), \
                 patch("osa_context_data.build_pair_context", return_value=context), \
                 patch("osa_context_data.assemble_input_matrix", return_value=features), \
                 patch.object(experiment, "fit_common", side_effect=fake_fit), \
                 patch("train_triage_osa.model_predictions", side_effect=fake_predict):
                digest = experiment.final_refit(root, out, config, cv, set(proteins),
                                                self.device, torch, resume=False)
            self.assertEqual(digest, experiment.sha(out / "rankings_primary_250.tsv.gz"))
            choices = pd.read_csv(out / "final_refit_choices.tsv", sep="\t").set_index("method")
            self.assertEqual(choices.loc["Full", "learning_rate"], 0.01)
            self.assertEqual(choices.loc["Concat_matched", "learning_rate"], 0.001)
            self.assertEqual(set(choices.fixed_epochs), {5})
            self.assertEqual(len(calls), 4)
            self.assertEqual({c["seed"] for c in calls}, {17, 19})
            self.assertTrue(all(c["epochs"] == 5 for c in calls))
            rankings = pd.read_csv(out / "rankings_primary_250.tsv.gz", sep="\t")
            self.assertEqual(len(rankings), 31125)
            self.assertEqual(set(rankings.rank_Full), set(range(1, 31126)))
            self.assertEqual(rankings.pair_id.tolist(), pairs.pair_id.tolist())
            np.testing.assert_allclose(rankings.Direct_anchor, 0.4, atol=1e-7)

    def test_completed_fit_resume_retains_bytes_and_checks_artifacts(self):
        config = {"synthetic": True}
        capacities = [{"method": "toy", "trainable_parameters": 2}]
        fingerprints = {"synthetic_input": "test_digest"}
        args = SimpleNamespace(smoke=False, check_env=False, resume=True, device="cpu")
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            primary = root / experiment.BASE / "04_candidates_v1/candidate_proteins_primary_250.tsv"
            primary.parent.mkdir(parents=True)
            pd.DataFrame({"UniProt_accession": ["P%03d" % i for i in range(250)]}).to_csv(
                primary, sep="\t", index=False)
            out = root / "out"
            out.mkdir()
            contract = dict(version=experiment.VERSION, smoke=False, configuration=config,
                inputs=fingerprints, active_parameters=capacities,
                external_outcomes_read_during_fit=False,
                software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                              torch=torch.__version__, platform=platform.platform()),
                analysis_type="post_hoc_common_objective_capacity_control")
            experiment.write_json(out / "analysis_contract.json", contract)
            artifact = out / "rankings_primary_250.tsv.gz"
            artifact.write_bytes(b"synthetic immutable ranking bytes")
            experiment.write_json(out / "run_summary.json", {
                "status": "PASS_FUSION_CAPACITY_FIT",
                "artifacts": experiment.output_artifacts(out)})
            original = artifact.read_bytes()
            with patch.object(experiment, "source_contract", return_value=({}, fingerprints)), \
                 patch.object(experiment, "external_source_files", return_value=[]), \
                 patch.object(experiment, "check_capacity", return_value=capacities), \
                 patch.object(experiment, "data_folds") as folds, \
                 patch.object(experiment, "final_refit") as refit:
                self.assertEqual(experiment.run_fit(args, root, config, root / "config.json", out, torch), 0)
                folds.assert_not_called()
                refit.assert_not_called()
                self.assertEqual(artifact.read_bytes(), original)
                artifact.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "Completed output changed"):
                    experiment.run_fit(args, root, config, root / "config.json", out, torch)
                folds.assert_not_called()
                refit.assert_not_called()

    def test_external_sources_resolve_python37_relative_script_path(self):
        with workspace_temporary_directory() as directory:
            root = Path(directory).resolve()
            relative_files = [
                "config/osa_cpap_validation_lock.json",
                experiment.BASE + "/04_candidates_v1/candidate_proteins_primary_250.tsv",
                experiment.BASE + "/20_independent_validation_preprocess_v1/candidate_independent_evidence_500.tsv",
                experiment.BASE + "/13_adipose_validation_preprocess_v1/candidate_adipose_evidence.tsv",
                "model/osa_adapter/osa_fusion_capacity_experiments.py",
                "model/osa_adapter/osa_external_strong_controls.py",
                "scripts/osa_cpap_reversal_endpoint.py",
                "scripts/osa_cpap_subject_signflip.py",
                "scripts/osa_independent_validation.py",
                "scripts/osa_adipose_cross_tissue_endpoint.py"]
            for cohort in ("GSE133601", "GSE49800"):
                relative_files += [experiment.BASE + "/10_cpap_validation_preprocess_v1/" + cohort + "/" + name
                    for name in ("candidate_validation_evidence.tsv", "candidate_participant_deltas.tsv.gz")]
            for relative in relative_files:
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic path fixture\n", encoding="utf-8")
            script = root/"model/osa_adapter/osa_fusion_capacity_experiments.py"
            # Reproduce Python 3.7's __file__, relative to the launch directory.
            relative_script = script.relative_to(Path.cwd().resolve()).as_posix()
            with patch.object(experiment, "__file__", relative_script):
                required = experiment.external_source_files(root)
                fingerprints = dict((p.relative_to(root).as_posix(), experiment.sha(p)) for p in required)
            self.assertTrue(all(p.is_absolute() for p in required))
            self.assertIn("model/osa_adapter/osa_fusion_capacity_experiments.py", fingerprints)
            self.assertEqual(len(fingerprints), len(relative_files))

    def test_external_resume_verifies_contract_and_output_artifacts(self):
        args = SimpleNamespace(smoke=False, resume=False)
        config = {"external": {"permutations": 10, "participant_bootstrap": 10}}
        with workspace_temporary_directory() as directory:
            root = Path(directory)
            out = root / "out"
            out.mkdir()
            ranking = out / "rankings_primary_250.tsv.gz"
            pd.DataFrame({"pair_id": ["synthetic_pair"]}).to_csv(ranking, sep="\t", index=False)
            external_input = root / "synthetic_external.tsv"
            external_input.write_text("synthetic fixed input\n", encoding="utf-8")
            experiment.write_json(out / "run_summary.json", dict(
                status="PASS_FUSION_CAPACITY_FIT", external_outcomes_read_during_fit=False,
                ranking_sha256=experiment.sha(ranking), methods=["Full", "Concat_matched"],
                primary_reference="synthetic common Full", native_reference="synthetic native"))

            def fake_endpoint(*unused_args, **unused_kwargs):
                (out / "external_evaluation" / "synthetic_result.tsv").write_text(
                    "synthetic completed result\n", encoding="utf-8")

            with patch.object(experiment, "external_source_files", return_value=[external_input]), \
                 patch("osa_external_strong_controls.cpap", side_effect=fake_endpoint) as cpap, \
                 patch("osa_external_strong_controls.tissue_controls") as tissues:
                self.assertEqual(experiment.run_external(args, root, config, out), 0)
                result = out / "external_evaluation" / "synthetic_result.tsv"
                original = result.read_bytes()
                args.resume = True
                self.assertEqual(experiment.run_external(args, root, config, out), 0)
                self.assertEqual(cpap.call_count, 1)
                self.assertEqual(tissues.call_count, 1)
                self.assertEqual(result.read_bytes(), original)
                external_input.write_text("changed external input\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "External resume contract differs"):
                    experiment.run_external(args, root, config, out)
                external_input.write_text("synthetic fixed input\n", encoding="utf-8")
                result.write_text("tampered result\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Completed output changed"):
                    experiment.run_external(args, root, config, out)
                self.assertEqual(cpap.call_count, 1)


if __name__ == "__main__":
    unittest.main()
