"""
Base interfaces for use in the package.
The package works according to the interfaces defined below.
"""

from abc import ABCMeta, abstractmethod


class QueryStrategy(metaclass=ABCMeta):
    #TODO: documentation

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        update_callback = kwargs.pop('update_callback', False)
        if update_callback:
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
    #TODO: documentation

    @abstractmethod
    def label(self, feature):
        pass


class Model(metaclass=ABCMeta):
    #TODO: documentation

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
    #TODO: documentation

    @abstractmethod
    def predict_real(self, feature, *args, **kwargs):
        pass
