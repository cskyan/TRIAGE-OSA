"""Protocol tests. Toy fixtures are not experimental results."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model/osa_adapter'))
from osa_cross_participant_adaptation import disjoint_halves


class ParticipantProtocolTests(unittest.TestCase):
    def test_static_initialization_reproduces_experts(self):
        import numpy as np
        import torch
        from triage_osa_model import TRIAGEOSAModel, INPUT_FEATURE_COLUMNS
        from osa_cross_participant_adaptation import initialize_from_static, expert_matrix
        x=np.random.RandomState(17).uniform(.1,.8,(8,len(INPUT_FEATURE_COLUMNS))).astype('float32')
        for w in ([.05,.05,.90],[.01,.98,.01],[.2,.3,.5]):
            model=TRIAGEOSAModel(input_mean=np.zeros(x.shape[1],dtype='float32'),
                                input_std=np.ones(x.shape[1],dtype='float32'))
            initialize_from_static(model,w,torch)
            model.eval()
            with torch.no_grad():
                z=model(torch.as_tensor(x))['adapted_logit'].numpy()
            np.testing.assert_allclose(z,expert_matrix(x).dot(w),atol=5e-6,rtol=0)

    def test_disjoint_deterministic_complete(self):
        m = pd.DataFrame({'sample_id':['p%d'%i for i in range(12)],
                          'study_group':['PS']*4+['MSO']*4+['VSO']*4})
        first = disjoint_halves(m,m.sample_id.tolist(),['a','b','c'],42)
        self.assertEqual(first,disjoint_halves(m,m.sample_id.tolist(),['a','b','c'],42))
        a,b = first
        self.assertFalse(set(a[0])&set(b[0]))
        self.assertEqual(set(a[0])|set(b[0]),set(m.sample_id))
        self.assertFalse(set(a[1])&set(b[1]))
        for arm in (a,b):
            self.assertEqual(set(m[m.sample_id.isin(arm[0])].study_group),{'PS','MSO','VSO'})

    def test_reject_unsplittable_stratum(self):
        m=pd.DataFrame({'sample_id':['p','q','r'], 'study_group':['PS','MSO','VSO']})
        with self.assertRaises(RuntimeError):
            disjoint_halves(m,m.sample_id.tolist(),['a','b'],42)


if __name__ == '__main__':
    unittest.main()
