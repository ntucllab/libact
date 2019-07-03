"""This module provides a dummy classifier, since in multi-label active learning
problem, it is common to see label being all zero in training set. We will let
this classifier handles this condition.
"""
import numpy as np

class DummyClf():
    """This classifier handles training sets with only 0s or 1s to unify the
    interface.

    """

    def __init__(self):
        self.classes_ = [0, 1]

    def fit(self, X, y):
        self.cls = int(y[0]) # 1 or 0

    def train(self, dataset):
        _, y = dataset.get_labeled_entries()
        self.cls = int(y[0])

    def predict(self, X):
        return self.cls * np.ones(len(X))

    def predict_real(self, X):
        return self.predict_proba(X) * 2 - 1

    def predict_proba(self, X):
        ret = np.zeros((len(X), 2))
        ret[:, self.cls] = 1.
        return ret
