import unittest

import numpy as np
from numpy.testing import assert_array_equal

from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SVM, Perceptron
from libact.query_strategies import UncertaintySampling, QUIRE
from libact.labelers import IdealLabeler


def init_toyexample(X, y):
    trn_ds = Dataset(X, np.concatenate([y[:6], [None] * 4]))
    return trn_ds


def run_qs(trn_ds, lbr, model, qs, quota):
    qseq = []
    for _ in range(quota):
        ask_id, score = qs.make_query(return_score=True)
        X, _ = zip(*trn_ds.data)
        lbl = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lbl)
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
        qs = UncertaintySampling(
            trn_ds, method='lc',
            model=LogisticRegression(solver='liblinear', multi_class="ovr"))
        model = LogisticRegression(solver='liblinear', multi_class="ovr")
        qseq = run_qs(trn_ds, self.lbr, model, qs, self.quota)
        assert_array_equal(qseq, np.array([6, 7, 8, 9]))

    def test_uncertainty_sm(self):
        trn_ds = init_toyexample(self.X, self.y)
        qs = UncertaintySampling(
            trn_ds, method='sm',
            model=LogisticRegression(solver='liblinear', multi_class="ovr"))
        model = LogisticRegression(solver='liblinear', multi_class="ovr")
        qseq = run_qs(trn_ds, self.lbr, model, qs, self.quota)
        assert_array_equal(qseq, np.array([6, 7, 8, 9]))

    def test_uncertainty_entropy(self):
        trn_ds = init_toyexample(self.X, self.y)
        qs = UncertaintySampling(
            trn_ds, method='entropy',
            model=LogisticRegression(solver='liblinear', multi_class="ovr"))
        model = LogisticRegression(solver='liblinear', multi_class="ovr")
        qseq = run_qs(trn_ds, self.lbr, model, qs, self.quota)
        assert_array_equal(qseq, np.array([6, 7, 8, 9]))

    def test_uncertainty_entropy_exceptions(self):
        trn_ds = init_toyexample(self.X, self.y)

        with self.assertRaises(TypeError):
            qs = UncertaintySampling(trn_ds, method='entropy', model=SVM())

        with self.assertRaises(TypeError):
            qs = UncertaintySampling(
                    trn_ds, method='entropy', model=Perceptron())

        with self.assertRaises(TypeError):
            qs = UncertaintySampling(
                    trn_ds, method='not_exist',
                    model=LogisticRegression(solver='liblinear', multi_class="ovr"))


if __name__ == '__main__':
    unittest.main()
