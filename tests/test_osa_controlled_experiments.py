"""Synthetic regression fixtures only. Never train on local real OSA data."""
import copy
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "model" / "osa_adapter"))
import osa_controlled_experiments as exp
import train_triage_osa as trainer
import osa_context_data as context_data
from triage_osa_model import BASE_FEATURE_COLUMNS, CONTEXT_FEATURE_COLUMNS
from osa_triage_adapter import TRIAGEConfig, TRIAGEPPIModel


class ControlledExperimentsTest(unittest.TestCase):
    def test_historical_benchmark_preparation_never_certifies_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            shapes = pd.DataFrame(dict(pair_id=["p1", "p2", "p3"],
                A_uid=["A", "B", "A"], B_uid=["B", "C", "C"]))
            shapes.to_csv(p / "shapes.tsv", sep="\t", index=False)
            labels = pd.DataFrame([["B", "A", 1], ["B", "C", 0], ["A", "C", 1]])
            labels.to_csv(p / "labels.tsv", sep="\t", index=False, header=False)
            spec = dict(seed=42, split_mode="protein_component", dataset_fingerprint=dict(n_rows=3),
                        splits={})
            for i, role in enumerate(("train", "val", "test")):
                positive = [1, 0, 1][i]
                spec["splits"][role] = dict(indices=[i], pair_ids=["p%d" % (i + 1)], n=1,
                    positive=positive, negative=1-positive)
            exp.write_json(p / "split.json", spec)
            r = exp.audit_historical_tuna_split(p / "shapes.tsv", p / "split.json", p / "labels.tsv", p / "audit")
            self.assertEqual(r["labels_status"], "PASS_EXPLICIT_BINARY_LABELS")
            self.assertEqual(r["test_pairs_both_proteins_unseen"], 0)
            self.assertEqual(r["train_test_protein_overlap"], 1)
            self.assertFalse(r["independent_holdout_certified"])
            r2 = exp.audit_historical_tuna_split(p / "shapes.tsv", p / "split.json", p / "missing.tsv", p / "missing_labels")
            self.assertEqual(r2["labels_status"], "MISSING_EXPLICIT_LABEL_SOURCE")
            self.assertFalse(r2["independent_holdout_certified"])
            spec["splits"]["test"]["pair_ids"] = ["wrong"]
            exp.write_json(p / "bad.json", spec)
            with self.assertRaisesRegex(RuntimeError, "identities/order"):
                exp.audit_historical_tuna_split(p / "shapes.tsv", p / "bad.json", p / "labels.tsv", p / "bad")
            labels.loc[len(labels)] = ["A", "B", 0]
            labels.to_csv(p / "labels.tsv", sep="\t", index=False, header=False)
            with self.assertRaisesRegex(RuntimeError, "Conflicting symmetric"):
                exp.audit_historical_tuna_split(p / "shapes.tsv", p / "split.json", p / "labels.tsv", p / "conflict")

    def test_suite_roundtrip_resume_and_target_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((PROJECT / "config/triage_osa_training_lock_v4.json").read_text())
            config["patient_information_manifest"]["sha256"] = trainer.EXPECTED_BASE_SHA256
            exp.write_json(root / "lock.json", config)
            proteins = ["SYNTH_%02d" % i for i in range(48)]
            pairs = pd.DataFrame(itertools.combinations(proteins, 2), columns=["protein_A", "protein_B"])
            pairs["pair_id"] = pairs.protein_A + "__" + pairs.protein_B
            pairs["checkpoint_sha256"] = trainer.EXPECTED_BASE_SHA256
            rng = np.random.RandomState(12)
            for name in BASE_FEATURE_COLUMNS:
                pairs[name] = rng.normal(size=len(pairs))
            pairs[["g_L1", "g_L2", "g_L3"]] = 1 / 3
            pairs.to_csv(root / "pairs.tsv", sep="\t", index=False)
            md = {"GSE75097": pd.DataFrame(dict(sample_id=["P%d" % i for i in range(16)], study_group=["PS", "OSA"] * 8)),
                  "GSE71356": pd.DataFrame(dict(subject_id=["IH%d" % i for i in range(4)]))}
            def fake_context(base, node, information_fusion=True):
                rng2 = np.random.RandomState(24)
                c = pd.DataFrame(rng2.uniform(.1, .9, (len(base), len(CONTEXT_FEATURE_COLUMNS))), columns=CONTEXT_FEATURE_COLUMNS)
                c["molecular_context_anchor"] = rng2.uniform(.1, .9, len(base))
                return c
            args = SimpleNamespace(variants="all", seeds="20260716", smoke=True,
                training_lock="lock.json", pair_scores="pairs.tsv", device="cpu",
                force=False, resume_experiments=False, check_env=False)
            out = root / "results/discovery_preprocess_v2/31_test/osa_ablation_suite/smoke"
            with patch.object(trainer, "sha256_file", return_value=trainer.EXPECTED_BASE_SHA256), \
                 patch.object(context_data, "load_discovery_inputs", return_value=({}, md, {"discovery_patient_information": root / "patient.tsv"})), \
                 patch.object(context_data, "compute_node_evidence", return_value=None), \
                 patch.object(context_data, "build_pair_context", side_effect=fake_context):
                exp.run_suite(args, root, out, torch)
                summary = exp.read_json(out / "run_summary.json")
                self.assertEqual(summary["completed_units"], 7)
                before = (out / "all_metrics.tsv").read_bytes()
                args.resume_experiments = True
                exp.run_suite(args, root, out, torch)
                self.assertEqual(len(pd.read_csv(out / "all_metrics.tsv", sep="\t")), 21)
                all_targets = []
                for path in out.glob("seed_*/*/fold_1/predictions.tsv.gz"):
                    p = pd.read_csv(path, sep="\t").sort_values("pair_id")
                    all_targets.append(p.weak_target.to_numpy())
                self.assertTrue(all(np.array_equal(all_targets[0], y) for y in all_targets))
                table = pd.read_csv(out / "paired_differences_vs_full.tsv", sep="\t")
                self.assertEqual(len(table), 7)
                self.assertTrue((table.loc[table.variant == "full", "delta_Spearman_vs_full"] == 0).all())
                # A one-fold synthetic formal-shaped fixture exercises the audit.
                completed = exp.read_json(out / "run_summary.json")
                completed["status"] = "PASS_CONTROLLED_EXPERIMENT_EXECUTION"
                exp.write_json(out / "run_summary.json", completed)
                audited = exp.audit_suite(out, root / "audit")
                self.assertEqual(audited["units"], 7)
                self.assertTrue(audited["identical_targets_across_variants_and_seeds"])
                # Full strong-control export on the same synthetic saved targets.
                context = fake_context(pairs, None)
                features = context_data.assemble_input_matrix(pairs, context)
                roles = pd.read_csv(out / "fold_1_pair_partitions.tsv.gz", sep="\t").set_index("pair_id")
                assignments = pairs.pair_id.map(roles.partition).to_numpy()
                splits = dict((r, np.where(assignments == r)[0]) for r in ("train", "validation", "test"))
                axes = ["OSA_raw", "severity_raw", "AHI_raw", "IH_raw", "hypertension_signed", "EDS_signed", "discovery_anchor"]
                nodes = pd.DataFrame(dict((k, np.linspace(.1, .9, len(proteins))) for k in axes))
                yielded = (1, pairs, {"development": nodes, "heldout": nodes},
                    {"development": context, "heldout": context}, features, splits, md,
                    lambda dataset, held: md[dataset]["sample_id" if dataset == "GSE75097" else "subject_id"].tolist())
                with patch.object(exp, "reconstructed_folds", return_value=iter([yielded])):
                    result = exp.run_fusion_and_context_supplements(args, root, out, root / "strong_controls")
                self.assertEqual(result["status"], "PASS_CONVERGED_CONTROLS")
                self.assertEqual(result["fits"], 2)
                self.assertEqual(len(pd.read_csv(root / "strong_controls/comparator_metrics.tsv", sep="\t")), 21)
                # Resume fails closed when the protocol changes.
                args.seeds = "42"
                with self.assertRaisesRegex(RuntimeError, "contract differs"):
                    exp.run_suite(args, root, out, torch)
            with self.assertRaises(RuntimeError):
                exp.safe_directory(root, "results/discovery_preprocess_v2/09_pair_context_ranking_triage_osa_v4")

    def test_convex_controls_and_certificate(self):
        rng = np.random.RandomState(41)
        x = rng.normal(size=(120, 3))
        y = 1 / (1 + np.exp(-x.dot([.1, .2, .7])))
        for constrained in (False, True):
            fit = exp.convex_static_fit(x, y, 41, constrained)
            self.assertTrue(fit["converged"])
            self.assertLessEqual(fit["duality_gap"], 1e-6)
            self.assertAlmostEqual(sum(fit["weights"]), 1, places=7)
            self.assertGreaterEqual(min(fit["weights"]), -1e-7)
            if constrained:
                self.assertGreaterEqual(sum(fit["weights"][:2]), .1 - 1e-7)
            again = exp.convex_static_fit(x, y, 41, constrained)
            np.testing.assert_allclose(fit["weights"], again["weights"], atol=1e-12)

    def test_native_benchmark_and_overlap_guard(self):
        cfg = TRIAGEConfig(d_model=16, d_pair=8, n_heads=2, n_encoder_layers=1,
                           n_cross_layers=1, topk=3, topm=6, max_pair_len=8)
        model = TRIAGEPPIModel(cfg).eval()
        x = dict(resA=torch.randn(1, 7, cfg.d_res_in), resB=torch.randn(1, 8, cfg.d_res_in))
        with torch.no_grad():
            baseline = model(**x)
            no_removal = exp.support_perturbed_forward(model, x, "top", 0., 41, torch)
            self.assertTrue(torch.allclose(baseline["logit_triage"], no_removal["logit_triage"], atol=1e-5))
            for mode in ("top", "random", "bottom"):
                changed = exp.support_perturbed_forward(model, x, mode, .2, 41, torch)
                self.assertTrue(torch.equal(baseline["logit_res_A"], changed["logit_res_A"]))
                self.assertTrue(torch.equal(baseline["logit_pair_raw"], changed["logit_pair_raw"]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("pair", "site", "esm", "out"):
                (root / name).mkdir()
            rng = np.random.RandomState(5)
            for p in ("A", "B", "C", "D"):
                np.save(str(root / "esm" / (p + ".npy")), rng.normal(size=(12, 1280)).astype(np.float32))
            pd.DataFrame(dict(pair_id=["pair1", "pair2"], A_uid=["A", "C"], B_uid=["B", "D"], label=[1, 0])).to_csv(root / "pair/tuna.shapes.tsv", sep="\t", index=False)
            (root / "pair/tuna.seq.fasta").write_text("".join(">" + p + "\n" + "A" * 12 + "\n" for p in ("A", "B", "C", "D")))
            pd.DataFrame(dict(pair_id=["site1", "site1"], A_chain=["A", "A"], B_chain=["B", "B"], i=[2, 3], j=[2, 10])).to_csv(root / "site/contacts.tsv", sep="\t", index=False)
            pd.DataFrame(dict(unit_id=["pair1", "pair2"], protein_A=["A", "C"], protein_B=["B", "D"])).to_csv(root / "pair_ids.tsv", sep="\t", index=False)
            pd.DataFrame(dict(unit_id=["site1"], protein_A=["A"], protein_B=["B"])).to_csv(root / "site_ids.tsv", sep="\t", index=False)
            pd.DataFrame(dict(protein_id=["EXPOSED"], unit_id=["train1"])).to_csv(root / "seen.tsv", sep="\t", index=False)
            pd.DataFrame(dict(protein_id=["A", "B", "C", "D", "EXPOSED"], canonical_id=["A", "B", "C", "D", "EXPOSED"])).to_csv(root / "ids.tsv", sep="\t", index=False)
            phases = ["original_training", "original_selection", "esm_adaptation_training", "esm_adaptation_selection"]
            lock = dict(exposure_audit_complete=True, exposure_audit_note="SYNTHETIC TEST ONLY",
                original_checkpoint="not_a_real_checkpoint.pt", l2_support_sensitivity=True,
                exposure_manifests=[dict(phase=p, path="seen.tsv") for p in phases], datasets=[
                    dict(name="pair_test", type="pair_fourpack", directory="pair", esm_dir="esm", test_ids_tsv="pair_ids.tsv", canonical_id_map_tsv="ids.tsv", canonical_id_audit_note="synthetic"),
                    dict(name="site_test", type="site_global", directory="site", esm_dir="esm", test_ids_tsv="site_ids.tsv", canonical_id_map_tsv="ids.tsv", canonical_id_audit_note="synthetic", contact_index_base=1)])
            exp.write_json(root / "lock.json", lock)
            args = SimpleNamespace(benchmark_lock="lock.json", device="cpu", force=False, smoke=True)
            with patch("osa_triage_adapter.load_frozen_checkpoint", return_value=(copy.deepcopy(model), cfg, {})):
                exp.run_benchmarks(args, root, root / "out", model, cfg, torch)
                pred = pd.read_csv(root / "out/site_test/predictions.tsv.gz", sep="\t")
                self.assertTrue((pred.loc[(pred.side == "A") & (pred.residue_index == 3), "label"] == 1).all())
                self.assertEqual(len(pd.read_csv(root / "out/benchmark_metrics.tsv", sep="\t")), 10)
                pd.DataFrame(dict(protein_id=["A"], unit_id=["train1"])).to_csv(root / "seen.tsv", sep="\t", index=False)
                with self.assertRaisesRegex(RuntimeError, "overlaps"):
                    exp.run_benchmarks(args, root, root / "out", model, cfg, torch)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
