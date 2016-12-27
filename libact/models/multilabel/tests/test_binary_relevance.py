""" Test Binary Relevance Model
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn import datasets
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
        print(self.X_test.shape, self.Y_test.shape)
        br = BinaryRelevance(base_clf=LogisticRegression(random_state=1126))
        br.train(Dataset(self.X_train, self.Y_train))

        br_pred_train = br.predict(self.X_train).astype(int)
        br_pred_test = br.predict(self.X_test).astype(int)

        for i in range(np.shape(self.Y_train)[1]):
            clf = sklearn.linear_model.LogisticRegression(random_state=1126)
            clf.fit(self.X_train, self.Y_train[:, i])

            assert_array_equal(clf.predict(self.X_train).astype(int),
                    br_pred_train[:, i])
            assert_array_equal(clf.predict(self.X_test).astype(int),
                    br_pred_test[:, i])


if __name__ == '__main__':
    unittest.main()
