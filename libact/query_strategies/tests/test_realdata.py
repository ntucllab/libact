"""
This module uses real world dataset to test the active learning algorithms.
Since creating own artificial dataset for test is too time comsuming so we would
use real world data, fix the random seed, and compare the query sequence each
active learning algorithm produces.
"""
import random
import os
import unittest

from numpy.testing import assert_array_equal
import numpy as np

from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import LogisticRegression
from libact.query_strategies import *


def run_qs(trn_ds, qs, truth, quota):
    ret = []
    for _ in range(quota):
        ask_id = qs.make_query()
        trn_ds.update(ask_id, truth[ask_id])

        ret.append(ask_id)
    return np.array(ret)

class RealdataTestCase(unittest.TestCase):

    def setUp(self):
        dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'datasets/heart_scale')
        self.X, self.y = import_libsvm_sparse(dataset_filepath).format_sklearn()
        self.quota = 10

    def test_quire(self):
        np.random.seed(1126)
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:5], [None]*(len(self.y)-5)]))
        qs = QUIRE(trn_ds)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([117, 175, 256, 64, 103, 118, 180, 159, 129, 235]))

    def test_RandomSampling(self):
        random.seed(1126)
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:5], [None]*(len(self.y)-5)]))
        qs = RandomSampling(trn_ds)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([141, 37, 129, 15, 151, 149, 237, 17, 146, 91]))

    def test_HintSVM(self):
        np.random.seed(1126)
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:5], [None]*(len(self.y)-5)]))
        qs = HintSVM(trn_ds)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([24, 235, 228, 209, 18, 143, 119, 90, 149, 207]))

    def test_QueryByCommittee(self):
        random.seed(1126)
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:10], [None]*(len(self.y)-10)]))
        qs = QueryByCommittee(trn_ds,
                              models=[LogisticRegression(C=1.0),
                                      LogisticRegression(C=0.01),
                                      LogisticRegression(C=100)])
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([10, 11, 13, 16, 18, 12, 17, 19, 20, 21]))

    def test_UcertaintySamplingLc(self):
        random.seed(1126)
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:10], [None]*(len(self.y)-10)]))
        qs = UncertaintySampling(trn_ds, method='lc',
                                 model=LogisticRegression())
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([145, 66, 82, 37, 194, 60, 191, 211, 245, 131]))

    def test_UcertaintySamplingSm(self):
        random.seed(1126)
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:10], [None]*(len(self.y)-10)]))
        qs = UncertaintySampling(trn_ds, method='sm',
                                 model=LogisticRegression())
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([145, 66, 82, 37, 194, 60, 191, 211, 245, 131]))

    def test_ActiveLearningByLearning(self):
        np.random.seed(1126)
        trn_ds = Dataset(self.X,
                         np.concatenate([self.y[:10], [None]*(len(self.y)-10)]))
        qs = ActiveLearningByLearning(trn_ds, T=self.quota,
                query_strategies=[
                    UncertaintySampling(trn_ds, model=LogisticRegression()),
                    HintSVM(trn_ds)],
                model=LogisticRegression()
            )
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([103, 220, 118,  75, 176,  50, 247, 199,  46,  55]))


if __name__ == '__main__':
    unittest.main()
