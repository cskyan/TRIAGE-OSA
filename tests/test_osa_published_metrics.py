import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model/osa_adapter'))
from osa_published_comparison import _roc_auc, _average_precision


class MetricsTests(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(_roc_auc([0, 1, 0, 1], [.5]*4), .5)
        self.assertEqual(_average_precision([0, 1, 0, 1], [.5]*4), .5)

    def test_ties_permutation_invariance(self):
        y=np.array([1,0,1,0,1]); p=np.array([.7,.7,.3,.3,.7])
        for order in [np.arange(5), np.arange(5)[::-1], [2,4,1,3,0]]:
            self.assertAlmostEqual(_roc_auc(y[order],p[order]),_roc_auc(y,p))
            self.assertAlmostEqual(_average_precision(y[order],p[order]),_average_precision(y,p))

    def test_ordered(self):
        self.assertEqual(_roc_auc([0,0,1,1],[.1,.2,.8,.9]),1.)
        self.assertEqual(_average_precision([0,0,1,1],[.1,.2,.8,.9]),1.)
        self.assertEqual(_roc_auc([0,0,1,1],[.9,.8,.2,.1]),0.)

if __name__=='__main__': unittest.main()
