"""This module contains implementation of binary relevance for multi-label
classification problems
"""
import copy

import numpy as np

from .dummy_clf import DummyClf
from libact.base.dataset import Dataset


class BinaryRelevance():
    r"""Binary Relevance

    base_clf : :py:mod:`libact.models` object instances

    References
    ----------
    """
    def __init__(self, base_clf):
        self.base_clf = copy.copy(base_clf)
        self.clfs_ = None

    def train(self, dataset):
        r"""Train model with given feature.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Train feature vector.

        Y : array-like, shape=(n_samples, n_labels)
            Target labels.

        Attributes
        ----------
        clfs_ : list of :py:mod:`libact.models` object instances
            Classifier instances.

        Returns
        -------
        self : object
            Retuen self.
        """
        X, Y = dataset.format_sklearn()
        X = np.array(X)
        Y = np.array(Y)

        self.n_labels_ = np.shape(Y)[1]
        self.n_features_ = np.shape(X)[1]

        self.clfs_ = []
        for i in range(self.n_labels_):
            # TODO should we handle it here or we should handle it before calling
            if len(np.unique(Y[:, i])) == 1:
                clf = DummyClf()
            else:
                clf = copy.deepcopy(self.base_clf)
            clf.train(Dataset(X, Y[:, i]))
            self.clfs_.append(clf)

        return self

    def predict(self, X):
        r"""Predict labels.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Feature vector.

        Returns
        -------
        pred : numpy array, shape=(n_samples, n_labels)
            Predicted labels of given feature vector.
        """
        X = np.asarray(X)
        if self.clfs_ is None:
            raise ValueError("Train before prediction")
        if X.shape[1] != self.n_features_:
            raise ValueError('Given feature size does not match')

        pred = np.zeros((X.shape[0], self.n_labels_))
        for i in range(self.n_labels_):
            pred[:, i] = self.clfs_[i].predict(X)
        return pred

    def predict_proba(self, X):
        r"""Predict the probability of being 1 for each label.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Feature vector.

        Returns
        -------
        pred : numpy array, shape=(n_samples, n_labels)
            Predicted probability of each label.
        """
        X = np.asarray(X)
        if self.clfs_ is None:
            raise ValueError("Train before prediction")
        if X.shape[1] != self.n_features_:
            raise ValueError('given feature size does not match')

        pred = np.zeros((X.shape[0], self.n_labels_))
        for i in range(self.n_labels_):
            pred[:, i] = self.clfs_[i].predict_proba(X)[:, 1]
        return pred
