import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/candidates'))
from osa_public_evidence_closeout import checked_pareto, component_sizes


class CloseoutTests(unittest.TestCase):
    def test_pareto_matches_bruteforce(self):
        rng=np.random.RandomState(7); x=rng.randint(0,6,(80,3)).astype(float)
        actual=checked_pareto(ROOT)(x)
        expected=np.array([not np.any(np.all(x>=v,axis=1)&np.any(x>v,axis=1)) for v in x])
        np.testing.assert_array_equal(actual,expected)

    def test_components_and_empty(self):
        x=pd.DataFrame({'protein_A':['a','b','d'],'protein_B':['b','c','e']})
        self.assertEqual(component_sizes(x),[3,2])
        self.assertEqual(component_sizes(x.iloc[:0]),[])

if __name__=='__main__': unittest.main()
