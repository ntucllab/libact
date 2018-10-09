""" Test Binary Relevance Model
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn import datasets
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
import sklearn.linear_model

from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.models.multilabel import BinaryRelevance


class BinaryRelevanceTestCase(unittest.TestCase):

    def setUp(self):
        X, Y = datasets.make_multilabel_classification(random_state=1126)
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(X, Y, test_size=0.3, random_state=1126)

    def test_binary_relevance_lr(self):
        br = BinaryRelevance(
            base_clf=LogisticRegression(solver='liblinear', multi_class="ovr",
                                        random_state=1126))
        br.train(Dataset(self.X_train, self.Y_train))

        br_pred_train = br.predict(self.X_train).astype(int)
        br_pred_test = br.predict(self.X_test).astype(int)

        br_pred_proba_train = br.predict_proba(self.X_train).astype(float)
        br_pred_proba_test = br.predict_proba(self.X_test).astype(float)

        for i in range(np.shape(self.Y_train)[1]):
            clf = sklearn.linear_model.LogisticRegression(
                solver='liblinear', multi_class="ovr", random_state=1126)
            clf.fit(self.X_train, self.Y_train[:, i])

            assert_array_almost_equal(clf.predict(self.X_train).astype(int),
                                      br_pred_train[:, i])
            assert_array_almost_equal(clf.predict(self.X_test).astype(int),
                                      br_pred_test[:, i])

            assert_array_almost_equal(clf.predict_proba(self.X_train)[:, 1].astype(float),
                                      br_pred_proba_train[:, i].astype(float))
            assert_array_almost_equal(clf.predict_proba(self.X_test)[:, 1].astype(float),
                                      br_pred_proba_test[:, i].astype(float))

        self.assertEqual(
            np.mean(np.abs(self.Y_test - br_pred_test).mean(axis=1)),
            br.score(Dataset(self.X_test, self.Y_test), 'hamming'))

        self.assertRaises(NotImplementedError,
                lambda: br.score(Dataset(self.X_test, self.Y_test),
                                 criterion='not_exist'))

    def test_binary_relevance_parallel(self):
        br = BinaryRelevance(base_clf=LogisticRegression(solver='liblinear',
                                    multi_class="ovr", random_state=1126),
                             n_jobs=1)
        br.train(Dataset(self.X_train, self.Y_train))
        br_par = BinaryRelevance(
                base_clf=LogisticRegression(solver='liblinear', random_state=1126),
                n_jobs=2)
        br_par.train(Dataset(self.X_train, self.Y_train))

        assert_array_equal(br.predict(self.X_test).astype(int),
                           br_par.predict(self.X_test).astype(int))


if __name__ == '__main__':
    unittest.main()
