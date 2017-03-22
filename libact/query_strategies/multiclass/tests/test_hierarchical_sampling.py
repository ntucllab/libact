""" UncertaintySampling test
"""
import unittest

import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle

from libact.base.dataset import Dataset
from libact.query_strategies.multiclass import HierarchicalSampling as HS


def run_qs(ds, qs, y_truth, quota):
    ret = []
    for _ in range(quota):
        ask_id = qs.make_query()
        ds.update(ask_id, y_truth[ask_id])
        y_pred = qs.report_all_label()
        score = sum(y_pred == y_truth) / len(y_truth)
        ret.append(score)
    return ret


class HierarchicalSamplingTestCase(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        X, y = shuffle(iris.data, iris.target, random_state=1126)
        self.X = X.tolist()
        self.y = y.tolist()
        self.classes = list(set(self.y))

    def test_hs_random_selecting(self):
        ds = Dataset(self.X, self.y[:10] + [None] * (len(self.y) - 10))
        qs = HS(ds, self.classes, active_selecting=False, random_state=1126)
        qseq = run_qs(ds, qs, self.y, len(self.y)-10)
        self.assertEqual(qseq[-1], 1.)

    def test_hs_active_selecting(self):
        ds = Dataset(self.X, self.y[:10] + [None] * (len(self.y) - 10))
        qs = HS(ds, self.classes, active_selecting=True, random_state=1126)
        qseq = run_qs(ds, qs, self.y, len(self.y)-10)
        self.assertEqual(qseq[-1], 1.)

    # def test_unexpected_label(self):
        # ds = Dataset(self.X, self.y[:10] + [None] * (len(self.y) - 10))
        # qs = HS(ds, self.classes[:1], active_selecting=True, random_state=1126)
        # with self.assertRaises(ValueError):
            # qseq = run_qs(ds, qs, self.y, len(self.y)-10)
