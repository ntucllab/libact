"""SVM

An interface for scikit-learn's C-Support Vector Classifier model.
"""

import sklearn.svm

from libact.base.interfaces import ContinuousModel

class SVM(ContinuousModel):
    """C-Support Vector Machine Classifier

    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self, *args, **kwargs):
        self.model = sklearn.svm.SVC(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)
    def predict_real(self, feature, *args, **kwargs):
        return self.model.decision_function(feature, *args, **kwargs)
