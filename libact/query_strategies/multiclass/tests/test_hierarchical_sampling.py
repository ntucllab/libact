""" HierarchicalSampling test
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.utils import shuffle

from libact.base.dataset import Dataset
from libact.models import SVM
from libact.query_strategies import UncertaintySampling
from libact.query_strategies.multiclass import HierarchicalSampling as HS
from ...tests.utils import run_qs


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
        assert_array_equal(
            np.concatenate([qseq[:10], qseq[-10:]]),
            np.array([48, 143, 13, 142, 88, 130, 29, 87, 36, 28,
                      58, 137, 49, 105, 76, 71, 63, 47, 64, 55])
            )

    def test_hs_active_selecting(self):
        ds = Dataset(self.X, self.y[:10] + [None] * (len(self.y) - 10))
        qs = HS(ds, self.classes, active_selecting=True, random_state=1126)
        qseq = run_qs(ds, qs, self.y, len(self.y)-10)
        assert_array_equal(
            np.concatenate([qseq[:10], qseq[-10:]]),
            np.array([48, 143, 13, 64, 101, 108, 51, 87, 36, 28,
                      43, 118, 47, 25, 81, 82, 95, 40, 67, 120])
            )

    def test_hs_subsampling(self):
        ds = Dataset(self.X, self.y[:10] + [None] * (len(self.y) - 10))
        sub_qs = UncertaintySampling(ds,
                    model=SVM(gamma='auto', decision_function_shape='ovr'))
        qs = HS(ds, self.classes, subsample_qs=sub_qs, random_state=1126)
        qseq = run_qs(ds, qs, self.y, len(self.y)-10)
        assert_array_equal(
            np.concatenate([qseq[:10], qseq[-10:]]),
            np.array([120, 50, 33, 28, 78, 133, 52, 124, 102, 109,
                      81, 108, 12, 10, 89, 114, 92, 126, 48, 25])
            )

    def test_hs_report_all_label(self):
        ds = Dataset(self.X, self.y)
        qs = HS(ds, self.classes, random_state=1126)
        y_report = qs.report_all_label()
        assert_array_equal(y_report, self.y)

    def test_hs_report_entry_label(self):
        ds = Dataset(self.X, self.y)
        qs = HS(ds, self.classes, random_state=1126)
        y_report = []
        for i in range(len(self.y)):
            y_report.append(qs.report_entry_label(i))
        assert_array_equal(y_report, self.y)
