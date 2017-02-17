""" Multiclass test
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from libact.base.dataset import Dataset
from libact.query_strategies.multiclass import ActiveLearningWithCostEmbedding as ALCE
from .utils import run_qs


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
            self.y_truth += y[y == i][1:].tolist()

    def test_alce_lr(self):
        cost_matrix = np.random.RandomState(1126).rand(3, 3)
        np.fill_diagonal(cost_matrix, 0)
        ds = Dataset(self.X + self.X_pool,
                     self.y[:3] + [None for _ in range(len(self.X_pool))])
        qs = ALCE(ds, cost_matrix, LinearRegression(), random_state=1126)
        qseq = run_qs(ds, qs, self.y_truth, self.quota)
        assert_array_equal(
            qseq, np.array([106, 118, 141, 43, 66, 34, 98, 99, 80, 105]))

    def test_alce_lr_embed5(self):
        cost_matrix = np.random.RandomState(1126).rand(3, 3)
        np.fill_diagonal(cost_matrix, 0)
        ds = Dataset(self.X + self.X_pool,
                     self.y[:3] + [None for _ in range(len(self.X_pool))])
        qs = ALCE(ds, cost_matrix, LinearRegression(), embed_dim=5,
                random_state=1126)
        qseq = run_qs(ds, qs, self.y_truth, self.quota)
        assert_array_equal(
            qseq, np.array([16, 63, 34, 122, 38, 35, 17, 24, 43, 18]))
