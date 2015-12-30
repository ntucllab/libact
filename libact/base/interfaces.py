"""
Base interfaces for use in the package.
The package works according to the interfaces defined below.
"""

from abc import ABCMeta, abstractmethod


class QueryStrategy(metaclass=ABCMeta):
    """Pool-based query strategy

    A QueryStrategy advices on which unlabeled data to be queried next given 
    a pool of labeled and unlabeled data.
    """

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        dataset.on_update(self.update)

    @property
    def dataset(self):
        return self._dataset

    def update(self, entry_id, label):
        pass

    @abstractmethod
    def make_query(self):
        pass


class Labeler(metaclass=ABCMeta):
    """Label the queries made by QueryStrategies
    
    A Labeler assigns labels to the features queried by QueryStrategies.
    """

    @abstractmethod
    def label(self, feature):
        pass


class Model(metaclass=ABCMeta):
    """Classification Model

    A Model is trained on a training dataset and produces a class-predicting
    function for future features.
    """

    @abstractmethod
    def train(self, dataset, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, feature, *args, **kwargs):
        pass

    @abstractmethod
    def score(self, testing_dataset, *args, **kwargs):
        pass


class ContinuousModel(Model):
    """Classification Model with intermediate continuous output
    
    A continuous classification model is able to output a real-valued vector
    for each features provided. The output vector is of shape (n_samples, n_classs)
    for an input feature matrix X of shape (n_samples, n_features). The larger the 
    kth-column value is, the more likely a feature x belongs the class k.
    """

    @abstractmethod
    def predict_real(self, feature, *args, **kwargs):
        pass
