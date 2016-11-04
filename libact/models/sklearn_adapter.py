"""scikit-learn classifier adapter
"""
from libact.base.interfaces import Model


class SklearnAdapter(Model):
    """Implementation of the scikit-learn classifier to libact model interface.

    Parameters
    ----------
    clf : scikit-learn classifier object instance
        The classifier object that is intended to be use with libact
    """

    def __init__(self, clf):
        self._model = clf

    def train(self, dataset, *args, **kwargs):
        return self._model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self._model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self._model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
