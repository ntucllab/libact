import unittest

from numpy.testing import assert_array_equal
import numpy as np

from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import VarianceReduction

def run_qs(trn_ds, qs, truth, quota):
    ret = []
    for _ in range(quota):
        ask_id = qs.make_query(n_jobs=1)
        trn_ds.update(ask_id, truth[ask_id])

        ret.append(ask_id)
    return np.array(ret)

class VarianceReductionTestCase(unittest.TestCase):

    def setUp(self):
        self.X = [[-2, -1], [1, 1], [-1, -2], [-1, -1], [1, 2], [2, 1]]
        self.y = [0, 1, 0, 1, 0, 1]
        self.quota = 4

    def test_VarianceReduction(self):
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:2], [None]*(len(self.y)-2)]))
        qs = VarianceReduction(trn_ds, model=LogisticRegression(), sigma=0.1)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq, np.array([3, 4, 2, 5]))


if __name__ == '__main__':
    unittest.main()
