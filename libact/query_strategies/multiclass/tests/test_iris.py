""" Multiclass test
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies.multiclass import ActiveLearningWithCostEmbedding as ALCE
from libact.query_strategies.multiclass import EER
from ...tests.utils import run_qs


class IrisTestCase(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        self.quota = 10
        self.n_classes = 3

        self.X, self.y, self.X_pool, self.y_truth = [], [], [], []
        for i in range(self.n_classes):
            self.X.append(X[y == i][0].tolist())
            self.y.append(i)
            self.X_pool += X[y == i][1:].tolist()
            self.y_truth += y[y == i].tolist()

    def test_alce_lr(self):
        cost_matrix = np.random.RandomState(1126).rand(3, 3)
        np.fill_diagonal(cost_matrix, 0)
        ds = Dataset(self.X + self.X_pool,
                     self.y[:3] + [None for _ in range(len(self.X_pool))])
        qs = ALCE(ds, cost_matrix, LinearRegression(), random_state=1126)
        qseq = run_qs(ds, qs, self.y_truth, self.quota)
        assert_array_equal(
            qseq, np.array([58, 118, 134, 43, 60, 139, 87, 78, 67, 146]))

    def test_alce_lr_embed5(self):
        cost_matrix = np.random.RandomState(1126).rand(3, 3)
        np.fill_diagonal(cost_matrix, 0)
        ds = Dataset(self.X + self.X_pool,
                     self.y[:3] + [None for _ in range(len(self.X_pool))])
        qs = ALCE(ds, cost_matrix, LinearRegression(), embed_dim=5,
                random_state=1126)
        qseq = run_qs(ds, qs, self.y_truth, self.quota)
        assert_array_equal(
            qseq, np.array([106, 118, 141, 43, 63, 99, 65, 89, 26, 52]))

    def test_eer(self):
        ds = Dataset(self.X + self.X_pool,
                     self.y[:3] + [None for _ in range(len(self.X_pool))])
        qs = EER(ds,
                 LogisticRegression(solver='liblinear', multi_class="ovr"),
                 random_state=1126)
        qseq = run_qs(ds, qs, self.y_truth, self.quota)
        assert_array_equal(
            qseq, np.array([131, 20, 129, 78, 22, 139, 88, 43, 141, 133]))

    def test_eer_01(self):
        ds = Dataset(self.X + self.X_pool,
                     self.y[:3] + [None for _ in range(len(self.X_pool))])
        qs = EER(ds,
                 LogisticRegression(solver='liblinear', multi_class="ovr"),
                 loss='01', random_state=1126)
        qseq = run_qs(ds, qs, self.y_truth, self.quota)
        assert_array_equal(
            qseq, np.array([105, 16, 131, 117, 109, 148, 136, 115, 144, 121]))
