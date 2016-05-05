""" Test Perceptron Model

Since Perceptron model is from scikit-learn, we test it by checking if it has
the same result as the sklearn model in scikit-learn on iris dataset.
"""
import unittest

from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import sklearn.linear_model

from libact.base.dataset import Dataset
from libact.models import Perceptron


class SVMIrisTestCase(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=1126)

    def test_perceptron(self):
        clf = sklearn.linear_model.Perceptron()
        clf.fit(self.X_train, self.y_train)
        perceptron = Perceptron()
        perceptron.train(Dataset(self.X_train, self.y_train))

        assert_array_equal(
            clf.predict(self.X_train), perceptron.predict(self.X_train))
        assert_array_equal(
            clf.predict(self.X_test), perceptron.predict(self.X_test))
        self.assertEqual(
            clf.score(self.X_train, self.y_train),
            perceptron.score(Dataset(self.X_train, self.y_train)))
        self.assertEqual(
            clf.score(self.X_test, self.y_test),
            perceptron.score(Dataset(self.X_test, self.y_test)))


if __name__ == '__main__':
    unittest.main()
