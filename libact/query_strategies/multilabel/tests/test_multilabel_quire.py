import unittest

from numpy.testing import assert_array_equal
import numpy as np

from libact.base.dataset import Dataset
from libact.query_strategies.multilabel import MultilabelQUIRE
from libact.utils import run_qs


class MultilabelQUIRETestCase(unittest.TestCase):
    """Variance reduction test case using artifitial dataset"""
    def setUp(self):
        self.X = [[-2, -1], [1, 1], [-1, -2], [-1, -1], [1, 2], [2, 1]]
        self.y = [[0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1]]
        self.quota = 4

    def test_multilabel_quire(self):
        trn_ds = Dataset(self.X, (self.y[:2] + [None] * (len(self.y) - 2)))
        qs = MultilabelQUIRE(trn_ds)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq, np.array([2, 3, 4, 5]))


if __name__ == '__main__':
    unittest.main()
