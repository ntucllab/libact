"""scikit-learn classifier adapter
"""
from libact.base.interfaces import Model, ContinuousModel, ProbabilisticModel


class SklearnAdapter(Model):
    """Implementation of the scikit-learn classifier to libact model interface.

    Parameters
    ----------
    clf : scikit-learn classifier object instance
        The classifier object that is intended to be use with libact

    Examples
    --------
    Here is an example of using SklearnAdapter to classify the iris dataset:

    .. code-block:: python

       from sklearn import datasets
       from sklearn.cross_validation import train_test_split
       from sklearn.linear_model import LogisticRegression

       from libact.base.dataset import Dataset
       from libact.models import SklearnAdapter

       iris = datasets.load_iris()
       X = iris.data
       y = iris.target
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

       adapter = SklearnAdapter(LogisticRegression(random_state=1126))

       adapter.train(Dataset(X_train, y_train))
       adapter.predict(X_test)
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


class SklearnProbaAdapter(ProbabilisticModel):
    """Implementation of the scikit-learn classifier to libact model interface.
    It should support predict_proba method and predict_real is default to return
    predict_proba.

    Parameters
    ----------
    clf : scikit-learn classifier object instance
        The classifier object that is intended to be use with libact

    Examples
    --------
    Here is an example of using SklearnAdapter to classify the iris dataset:

    .. code-block:: python

       from sklearn import datasets
       from sklearn.cross_validation import train_test_split
       from sklearn.linear_model import LogisticRegression

       from libact.base.dataset import Dataset
       from libact.models import SklearnProbaAdapter

       iris = datasets.load_iris()
       X = iris.data
       y = iris.target
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

       adapter = SklearnProbaAdapter(LogisticRegression(random_state=1126))

       adapter.train(Dataset(X_train, y_train))
       adapter.predict(X_test)
       adapter.predict_proba(X_test)
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

    def predict_real(self, feature, *args, **kwargs):
        return self._model.predict_proba(feature, *args, **kwargs) * 2 - 1

    def predict_proba(self, feature, *args, **kwargs):
        return self._model.predict_proba(feature, *args, **kwargs)
