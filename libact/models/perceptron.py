from libact.base.interfaces import Model
import sklearn.linear_model

"""
A interface for scikit-learn's perceptron model
"""
class Perceptron(Model):

    def __init__(self, *args, **kwargs):
        self.model = sklearn.linear_model.Perceptron(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)
