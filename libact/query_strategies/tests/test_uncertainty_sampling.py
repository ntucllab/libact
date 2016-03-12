import unittest

from numpy.testing import assert_array_equal
import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler


def init_toyexample(X, y):
    trn_ds = Dataset(X, np.concatenate([y[:6], [None] * 4]))
    return trn_ds


def run_qs(trn_ds, lbr, model, qs, quota):
    qseq = []
    for i in range(quota) :
        ask_id = qs.make_query()
        X, y = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)
        qseq.append(ask_id)
    return np.array(qseq)


class UncertaintySamplingTestCase(unittest.TestCase):

    def setUp(self):
        self.X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [0, 1],
                  [0, -2], [1.5, 1.5], [-2, -2]]
        self.y = [-1, -1, -1, 1, 1, 1, -1, -1, 1, 1]
        self.quota = 4
        self.fully_labeled_trn_ds = Dataset(self.X, self.y)
        self.lbr = IdealLabeler(self.fully_labeled_trn_ds)

    def test_uncertainty_lc(self):
        trn_ds = init_toyexample(self.X, self.y)
        qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
        model = LogisticRegression()
        qseq = run_qs(trn_ds, self.lbr, model, qs, self.quota)
        assert_array_equal(qseq, np.array([6,7,8,9]))

    def test_uncertainty_sm(self):
        trn_ds = init_toyexample(self.X, self.y)
        qs = UncertaintySampling(trn_ds, method='sm', model=LogisticRegression())
        model = LogisticRegression()
        qseq = run_qs(trn_ds, self.lbr, model, qs, self.quota)
        assert_array_equal(qseq, np.array([6,7,8,9]))

    def test_quire(self):
        trn_ds = init_toyexample(self.X, self.y)
        qs = QUIRE(trn_ds)
        model = LogisticRegression()
        qseq = run_qs(trn_ds, self.lbr, model, qs, self.quota)
        assert_array_equal(qseq, np.array([6,7,9,8]))


if __name__ == '__main__':
    unittest.main()
