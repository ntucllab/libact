from libact.base.interfaces import Model
import sklearn.linear_model


class LogisticRegression(Model):

    def __init__(self, *args, **kwargs):
        self.model = sklearn.linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    """Logistic regression features"""

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)

    def predict_log_proba(self, feature, *args, **kwargs):
        return self.model.predict_log_proba(feature, *args, **kwargs)
