"""
Base interfaces for use in the package.
The package works according to the interfaces defined below.
"""

from abc import ABCMeta, abstractmethod


class QueryStrategy(metaclass=ABCMeta):
    #TODO: documentation

    @abstractmethod
    def make_query(self, dataset, n_queries=1):
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
