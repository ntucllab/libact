import unittest

from numpy.testing import assert_array_equal
import numpy as np

from libact.base.dataset import Dataset
from libact.query_strategies import QUIRE
from .utils import run_qs


class QUIRETestCase(unittest.TestCase):
    """QUIRE test case using artifitial dataset"""
    def setUp(self):
        self.X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [0, 1],
                  [0, -2], [1.5, 1.5], [-2, -2]]
        self.y = [-1, -1, -1, 1, 1, 1, -1, -1, 1, 1]
        self.quota = 4

    def test_quire(self):
        trn_ds = Dataset(self.X, np.concatenate([self.y[:6], [None] * 4]))
        qs = QUIRE(trn_ds)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq, np.array([6, 7, 9, 8]))


if __name__ == '__main__':
    unittest.main()

