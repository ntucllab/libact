import unittest

from numpy.testing import assert_array_equal
import numpy as np

from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import VarianceReduction
from .utils import run_qs


class VarianceReductionTestCase(unittest.TestCase):
    """Variance reduction test case using artifitial dataset"""
    def setUp(self):
        self.X = [[-2, -1], [1, 1], [-1, -2], [-1, -1], [1, 2], [2, 1]]
        self.y = [0, 1, 0, 1, 0, 1]
        self.quota = 4

    def test_variance_reduction(self):
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:2],
                                         [None] * (len(self.y) - 2)]))
        qs = VarianceReduction(
                trn_ds,
                model=LogisticRegression(solver='liblinear', multi_class="ovr"),
                sigma=0.1
            )
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq, np.array([4, 5, 2, 3]))


if __name__ == '__main__':
    unittest.main()
