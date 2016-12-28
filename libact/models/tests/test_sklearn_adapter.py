""" Test sklearn adapter Model
"""
import unittest

from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from libact.base.dataset import Dataset
from libact.models import SklearnAdapter, SklearnProbaAdapter


class IrisTestCase(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=1126)

    def check_functions(self, adapter, clf):
        adapter.train(Dataset(self.X_train, self.y_train))
        clf.fit(self.X_train, self.y_train)

        assert_array_equal(
            adapter.predict(self.X_train), clf.predict(self.X_train))
        assert_array_equal(
            adapter.predict(self.X_test), clf.predict(self.X_test))
        self.assertEqual(
            adapter.score(Dataset(self.X_train, self.y_train)),
            clf.score(self.X_train, self.y_train))
        self.assertEqual(
            adapter.score(Dataset(self.X_test, self.y_test)),
            clf.score(self.X_test, self.y_test))

    def check_proba(self, adapter, clf):
        adapter.train(Dataset(self.X_train, self.y_train))
        clf.fit(self.X_train, self.y_train)

        assert_array_equal(adapter.predict_proba(self.X_train),
                           clf.predict_proba(self.X_train))
        assert_array_equal(adapter.predict_real(self.X_train),
                           clf.predict_proba(self.X_train))

    def test_adapt_logistic_regression(self):
        adapter = SklearnProbaAdapter(LogisticRegression(random_state=1126))
        clf = LogisticRegression(random_state=1126)
        self.check_functions(adapter, clf)

    def test_adapt_linear_svc(self):
        adapter = SklearnAdapter(LinearSVC(random_state=1126))
        clf = LinearSVC(random_state=1126)
        self.check_functions(adapter, clf)

    def test_adapt_knn(self):
        adapter = SklearnAdapter(KNeighborsClassifier())
        clf = KNeighborsClassifier()
        self.check_functions(adapter, clf)


if __name__ == '__main__':
    unittest.main()
