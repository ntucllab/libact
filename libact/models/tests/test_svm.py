""" Test SVM Model

Since SVM model is from scikit-learn, we test it by checking if it has the same
result as the SVC model in scikit-learn on iris dataset.
"""
import unittest

from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

from libact.base.dataset import Dataset
from libact.models import SVM


class SVMIrisTestCase(unittest.TestCase):

    def setUp(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.3, random_state=1126)

    def test_svm(self):
        svc_clf = SVC()
        svc_clf.fit(self.X_train, self.y_train)
        svm = SVM()
        svm.train(Dataset(self.X_train, self.y_train))

        assert_array_equal(
            svc_clf.predict(self.X_train), svm.predict(self.X_train))
        assert_array_equal(
            svc_clf.predict(self.X_test), svm.predict(self.X_test))
        self.assertEqual(
            svc_clf.score(self.X_train, self.y_train),
            svm.score(Dataset(self.X_train, self.y_train)))
        self.assertEqual(
            svc_clf.score(self.X_test, self.y_test),
            svm.score(Dataset(self.X_test, self.y_test)))


if __name__ == '__main__':
    unittest.main()
