"""Synthetic endpoint fixtures only; no training or real external reanalysis."""
import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts/validation"))
sys.path.insert(0, str(PROJECT / "model" / "osa_adapter"))
import osa_external_strong_controls as external
import osa_cpap_reversal_endpoint as cp


def synthetic_inputs(root, methods):
    """Build tiny synthetic files in a temporary root, never in real results."""
    base = root / external.BASE
    ids = ["SYNTH_%02d" % i for i in range(6)]
    rng = np.random.RandomState(19)
    candidates = pd.DataFrame({
        "UniProt_accession": ids,
        "protein_detection_fraction": np.linspace(.2, .9, len(ids)),
        "log2_UniProt_length": np.linspace(7, 10, len(ids)),
        "OSA_beta": [1, -1, 1, -1, 1, -1],
    })
    candidate_path = root / cp.CANDIDATE_REL
    candidate_path.parent.mkdir(parents=True)
    candidates.to_csv(candidate_path, sep="\t", index=False)
    config = root / "config/osa_cpap_validation_lock.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"permutation": {"seed": 19}}), encoding="utf-8")
    for cohort in cp.COHORTS:
        out = base / "10_cpap_validation_preprocess_v1" / cohort
        out.mkdir(parents=True)
        evidence = pd.DataFrame({
            "UniProt_accession": ids,
            "validation_available": ["yes"] * len(ids),
            "discovery_OSA_beta_sign": ["positive", "negative"] * 3,
            "CPAP_beta_post_minus_pre": [-1, .4, -.3, .7, .1, -.2],
            "moderated_t": [-3, 1.4, -1.2, 2.2, .3, -.7],
        })
        evidence.to_csv(out / "candidate_validation_evidence.tsv", sep="\t", index=False)
        values = pd.DataFrame(rng.normal(size=(len(ids), 4)),
                              columns=["subject_%d" % i for i in range(4)])
        values.insert(0, "UniProt_accession", ids)
        values.to_csv(out / "candidate_participant_deltas.tsv.gz", sep="\t", index=False)
    independent = pd.DataFrame({"UniProt_accession": ids})
    for dataset in ("GSE38792", "GSE293433"):
        available = dataset + ("_primary_filter_available" if dataset == "GSE293433" else "_available")
        independent[available] = "yes"
        independent[dataset + "_beta_OSA_minus_control"] = rng.normal(size=len(ids))
        independent[dataset + "_moderated_t"] = rng.normal(size=len(ids))
    out = base / "20_independent_validation_preprocess_v1"
    out.mkdir(parents=True)
    independent.to_csv(out / "candidate_independent_evidence_500.tsv", sep="\t", index=False)
    adipose = pd.DataFrame({"UniProt_accession": ids,
                            "adipose_available": ["yes"] * len(ids),
                            "discovery_OSA_beta_sign": ["positive", "negative"] * 3})
    for name in ("state", "state_adjusted", "CPAP"):
        adipose[name + "_beta"] = rng.normal(size=len(ids))
        adipose[name + "_moderated_t"] = rng.normal(size=len(ids))
    out = base / "13_adipose_validation_preprocess_v1"
    out.mkdir(parents=True)
    adipose.to_csv(out / "candidate_adipose_evidence.tsv", sep="\t", index=False)
    ranks = pd.DataFrame(list(itertools.combinations(ids, 2)), columns=["protein_A", "protein_B"])
    ranks.insert(0, "pair_id", ranks.protein_A + "__" + ranks.protein_B)
    score = np.linspace(.9, .1, len(ranks))
    for method in methods:
        ranks[method] = score  # Deliberately duplicate all methods for zero-delta tests.
        ranks["rank_" + method] = np.arange(1, len(ranks) + 1)
    return ranks


class ExternalMethodsTests(unittest.TestCase):
    def test_two_three_four_methods_share_null_and_bootstrap(self):
        for count in (2, 3, 4):
            methods = tuple(["Full"] + ["Generic_%d" % i for i in range(1, count)])
            with self.subTest(methods=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ranks = synthetic_inputs(root, methods)
                out = root / "synthetic_endpoint_outputs"
                out.mkdir()
                with contextlib.redirect_stdout(io.StringIO()):
                    external.cpap(root, out, ranks, 20, 12, methods=methods)
                    external.tissue_controls(root, out, ranks, 20, methods=methods)
                cpap = pd.read_csv(out / "cpap_paired_node_randomization.tsv", sep="\t")
                tissue = pd.read_csv(out / "tissue_paired_node_randomization.tsv", sep="\t")
                bootstrap = pd.read_csv(out / "cpap_paired_participant_bootstrap.tsv", sep="\t")
                self.assertEqual(len(cpap), 4 * (count - 1))
                self.assertEqual(len(tissue), 10 * (count - 1))
                for frame in (cpap, tissue):
                    self.assertTrue((frame.difference == 0).all())
                    self.assertTrue((frame.two_sided_randomization_P == 1).all())
                    self.assertTrue((frame.BH_family_size == len(frame)).all())
                    self.assertTrue((frame.common_reference == "Full").all())
                self.assertIn("BH_FDR_CPAP_comparisons", cpap)
                self.assertNotIn("BH_FDR_8_CPAP_comparisons", cpap)
                self.assertIn("BH_FDR_tissue_comparisons", tissue)
                self.assertTrue((bootstrap.difference == 0).all())
                self.assertTrue((bootstrap.CI95_low == 0).all())
                self.assertTrue((bootstrap.CI95_high == 0).all())
                for cohort in cp.COHORTS:
                    null = pd.read_csv(out / (cohort + "_shared_node_null.tsv.gz"), sep="\t")
                    draws = pd.read_csv(out / (cohort + "_participant_bootstrap.tsv.gz"), sep="\t")
                    self.assertEqual(draws.shape, (12, count * 2))
                    for method in methods[1:]:
                        for metric in ("ratio", "AUEC"):
                            np.testing.assert_array_equal(null["Full_" + metric], null[method + "_" + metric])
                            np.testing.assert_array_equal(draws["Full_" + metric], draws[method + "_" + metric])

    def test_default_and_explicit_legacy_methods_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ranks = synthetic_inputs(root, external.METHODS)
            implicit, explicit = root / "implicit", root / "explicit"
            implicit.mkdir(); explicit.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                external.cpap(root, implicit, ranks, 20, 12)
                external.cpap(root, explicit, ranks, 20, 12, methods=external.METHODS)
                external.tissue_controls(root, implicit, ranks, 20)
                external.tissue_controls(root, explicit, ranks, 20, methods=external.METHODS)
            for path in implicit.iterdir():
                pd.testing.assert_frame_equal(pd.read_csv(path, sep="\t"),
                                               pd.read_csv(explicit / path.name, sep="\t"))
            cpap = pd.read_csv(implicit / "cpap_paired_node_randomization.tsv", sep="\t")
            tissue = pd.read_csv(implicit / "tissue_paired_node_randomization.tsv", sep="\t")
            bootstrap = pd.read_csv(implicit / "cpap_paired_participant_bootstrap.tsv", sep="\t")
            self.assertIn("BH_FDR_8_CPAP_comparisons", cpap)
            self.assertIn("BH_FDR_tissue_family", tissue)
            self.assertNotIn("common_reference", cpap)
            self.assertNotIn("common_reference", tissue)
            self.assertNotIn("common_reference", bootstrap)

    def test_invalid_methods_and_rankings_fail_before_reading_data(self):
        ranks = pd.DataFrame({"pair_id": ["a", "b"], "protein_A": ["A", "A"],
                              "protein_B": ["B", "C"], "Full": [.8, .2],
                              "Generic": [.6, .4], "rank_Full": [1, 2], "rank_Generic": [1, 2]})
        for methods in (("Full",), ("Generic", "Full"), ("Full", "Full"), ("Full", "")):
            with self.subTest(methods=methods), self.assertRaises(RuntimeError):
                external._validated_external_methods(ranks, methods)
        for field, values in (("Full", [np.nan, .2]), ("rank_Generic", [1, 1]),
                              ("rank_Generic", [2, 1]), ("pair_id", ["a", "a"])):
            changed = ranks.copy()
            changed[field] = values
            with self.subTest(field=field, values=values), self.assertRaises(RuntimeError):
                external._validated_external_methods(changed, ("Full", "Generic"))
        with self.assertRaisesRegex(RuntimeError, "Missing external ranking fields"):
            external._validated_external_methods(ranks.drop(columns="Generic"), ("Full", "Generic"))


if __name__ == "__main__":
    unittest.main()
