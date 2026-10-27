"""Synthetic tests only: never open the real GSE283703 expression file."""
import ast
import csv
import gzip
import importlib.util
import itertools
import json
import os
import random
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('OSA_DATA_ROOT', str(ROOT))).resolve()
SCRIPT = ROOT / "scripts/validation/osa_independent_validation.py"
SPEC = importlib.util.spec_from_file_location("osa_longitudinal_test_driver", str(SCRIPT))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class LongitudinalTests(unittest.TestCase):
    def test_legacy_source_lock_is_preserved(self):
        lock = json.loads((ROOT/'config/osa_independent_validation_implementation_lock_v4.json').read_text())
        self.assertTrue(M.longitudinal_legacy_source_matches(lock['script_sha256']))
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary)/'altered.py'
            altered.write_text(SCRIPT.read_text(encoding='utf-8').replace('SCRIPT_VERSION = "1.3.0"','SCRIPT_VERSION = "different"'),encoding='utf-8')
            self.assertFalse(M.longitudinal_legacy_source_matches(lock['script_sha256'],altered))

    def test_python37_syntax(self):
        ast.parse(SCRIPT.read_text(encoding="utf-8-sig"), feature_version=(3, 7))

    def test_metadata_subject_units(self):
        metadata = DATA_ROOT / 'geo/GSE283703_family.soft.gz'
        if not metadata.is_file():
            self.skipTest('Optional integration data absent: set OSA_DATA_ROOT to the archived data workspace.')
        rows, _ = M.parse_longitudinal_soft(metadata)
        self.assertEqual(len(rows), 71)
        self.assertEqual(len({r['subject_id'] for r in rows}), 19)
        self.assertEqual(len({r['subject_id'] for r in rows if r['disease']=='OSA'}), 9)
        self.assertTrue(all(not r['BMI'] for r in rows))
        pairs = M.longitudinal_contrasts(rows)
        counts = Counter(r['contrast'] for r in pairs if r['eligible']=='yes')
        self.assertEqual(counts['month4_AM'],8)
        self.assertEqual(counts['month4_PM'],9)
        self.assertEqual(counts['month24_DAY_MEAN'],8)

    def test_rewire_preserves_degrees(self):
        edges = [('A','B'),('B','C'),('C','D'),('D','E'),('E','F'),('F','A'),('A','D')]
        rewired, swaps = M.longitudinal_rewire(edges, random.Random(3))
        self.assertGreater(swaps,0)
        self.assertEqual(Counter(itertools.chain.from_iterable(edges)),Counter(itertools.chain.from_iterable(rewired)))
        self.assertEqual(len(rewired),len(set(rewired)))
        self.assertTrue(all(a!=b for a,b in rewired))

    def test_star_rewire_not_fabricated(self):
        edges = [('A',x) for x in 'BCDEF']
        rewired, swaps = M.longitudinal_rewire(edges,random.Random(1))
        self.assertEqual(swaps,0)
        self.assertEqual(rewired,edges)

    def test_signflip_zero_and_positive(self):
        self.assertEqual(M.longitudinal_signflip(np,[0.0]*8),(0.0,1.0))
        effect,p = M.longitudinal_signflip(np,[1.0]*8)
        self.assertEqual(effect,1.0)
        self.assertEqual(p,2.0/256)

    def test_graph_score_is_odd_under_visit_swap(self):
        values = {'A':2.,'B':1.,'C':-3.,'D':-2.}
        edges = [('A','B'),('B','C'),('C','D')]
        score = M.longitudinal_graph_score(np,edges,values)
        inverse = M.longitudinal_graph_score(np,edges,{k:-v for k,v in values.items()})
        self.assertAlmostEqual(score,-inverse)
        self.assertAlmostEqual(score,-1./3.)

    def test_neighbor_reconstruction_distinguishes_coherent_topology(self):
        values = {'A':1.0,'B':0.8,'C':0.6,'D':-1.0,'E':-0.8,'F':-0.6}
        coherent = M.longitudinal_neighbor_reconstruction(
            np,[('A','B'),('A','C'),('B','C'),('D','E'),('D','F'),('E','F')],values
        )
        incoherent = M.longitudinal_neighbor_reconstruction(
            np,[('A','D'),('A','E'),('B','D'),('B','F'),('C','E'),('C','F')],values
        )
        self.assertGreater(coherent['Spearman'],incoherent['Spearman'])
        self.assertGreater(coherent['sign_accuracy'],incoherent['sign_accuracy'])
        self.assertLess(coherent['MAE'],incoherent['MAE'])

    def test_header_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='osa_long_test_') as temporary:
            root = Path(temporary)
            design = self.copy_design(root)
            count_path = root/'geo/GSE283703_Normalized_counts.csv.gz'
            with gzip.open(str(count_path),'wt',encoding='utf-8',newline='') as h:
                csv.writer(h).writerow(['gene.ID','wrong_sample'])
            with self.assertRaisesRegex(RuntimeError,'Expression columns'):
                M.run_longitudinal(SimpleNamespace(check_env=False,smoke=False,force=False),root)

    def copy_design(self, root):
        original = DATA_ROOT/'results/discovery_preprocess_v2'/M.LONG_STAGE
        if not (original/'analysis_contract.json').is_file():
            self.skipTest('Optional longitudinal design absent: set OSA_DATA_ROOT to the archived data workspace.')
        contract = json.loads((original/'analysis_contract.json').read_text(encoding='utf-8'))
        for relative in contract['frozen_files']:
            dst = root/relative
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(str(DATA_ROOT/relative),str(dst))
        design = root/'results/discovery_preprocess_v2'/M.LONG_STAGE
        shutil.copyfile(str(original/'analysis_contract.json'),str(design/'analysis_contract.json'))
        return design

    def test_end_to_end_synthetic_expression(self):
        with tempfile.TemporaryDirectory(prefix='osa_long_test_') as temporary:
            root = Path(temporary)
            design = self.copy_design(root)
            samples = M.longitudinal_rows(design/'sample_manifest.tsv')
            annotations = M.longitudinal_rows(design/'candidate_axis_annotation.tsv')
            genes = {r['symbol'] for r in annotations}
            genes.update(r['symbol'] for r in M.longitudinal_rows(design/'pathway_gene_sets.tsv'))
            hgnc = M.longitudinal_rows(root/'annotations/hgnc_complete_set_20260714.tsv')
            genes.update(r['symbol'] for r in hgnc[:200])
            rng = np.random.RandomState(7)
            count_path = root/'geo/GSE283703_Normalized_counts.csv.gz'
            with gzip.open(str(count_path),'wt',encoding='utf-8',newline='') as handle:
                writer = csv.writer(handle)
                # Deliberately reverse columns to test identity matching, not order.
                samples = list(reversed(samples))
                writer.writerow(['gene.ID']+[r['library_name'] for r in samples])
                for gene in sorted(genes):
                    values = np.exp(rng.normal(5,0.3,len(samples)))
                    writer.writerow([gene]+list(values))
            args = SimpleNamespace(check_env=False,smoke=False,force=False)
            self.assertEqual(M.run_longitudinal(args,root),0)
            out = root/'results/discovery_preprocess_v2'/M.LONG_RESULT
            summary = json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['status'],'PASS_FROZEN_LONGITUDINAL_EXECUTION')
            self.assertEqual(summary['primary']['n_subjects'],8)
            self.assertFalse(summary['model_retrained'])
            tests = M.longitudinal_rows(out/'endpoint_tests.tsv')
            self.assertEqual(sum(r['analysis_role']=='primary' for r in tests),1)
            self.assertTrue(any(r['panel']=='hypoxia_HIF_ROS' and r['status']=='NOT_EVALUABLE' for r in tests))
            self.assertTrue(any(r['comparison']=='full_pathway_competitive_expression_change' and r['status']=='EVALUATED' for r in tests))
            for row in tests:
                if row['status']=='EVALUATED':
                    self.assertTrue(0 <= float(row['exact_signflip_P']) <= 1)
                    if row['analysis_role']=='secondary':
                        self.assertGreaterEqual(float(row['BH_FDR_secondary'])+1e-12,float(row['exact_signflip_P']))
            # Reusing or replacing the design does not circumvent a changed input.
            annotation = design/'candidate_axis_annotation.tsv'
            annotation.write_text(annotation.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'Frozen longitudinal input changed'):
                M.run_longitudinal(SimpleNamespace(check_env=True,smoke=False,force=False),root)


if __name__ == '__main__':
    unittest.main()
