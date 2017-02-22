"""This module contains implementation of binary relevance for multi-label
classification problems
"""
import copy

import numpy as np
from joblib import Parallel, delayed

from .dummy_clf import DummyClf
from libact.base.dataset import Dataset
from libact.base.interfaces import MultilabelModel

def _fit_model(model, X, y):
    model.train(Dataset(X, y))

class BinaryRelevance(MultilabelModel):
    r"""Binary Relevance

    base_clf : :py:mod:`libact.models` object instances
        If wanting to use predict_proba, base_clf are required to support
        predict_proba method.

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are
        used. If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    References
    ----------
    """
    def __init__(self, base_clf, n_jobs=1):
        self.base_clf = copy.copy(base_clf)
        self.clfs_ = None
        self.n_jobs = n_jobs

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
        clfs\_ : list of :py:mod:`libact.models` object instances
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
            self.clfs_.append(clf)

        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_fit_model)(self.clfs_[i], X, Y[:, i])
            for i in range(self.n_labels_))
        #clf.train(Dataset(X, Y[:, i]))

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
        return pred.astype(int)

    def predict_real(self, X):
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
            pred[:, i] = self.clfs_[i].predict_real(X)[:, 1]
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

    def score(self, testing_dataset, criterion='hamming'):
        """Return the mean accuracy on the test dataset

        Parameters
        ----------
        testing_dataset : Dataset object
            The testing dataset used to measure the perforance of the trained
            model.
        criterion : ['hamming', 'f1']
            instance-wise criterion.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        # TODO check if data in dataset are all correct
        X, Y = testing_dataset.format_sklearn()
        if criterion == 'hamming':
            return np.mean(np.abs(self.predict(X) - Y).mean(axis=1))
        elif criterion == 'f1':
            Z = self.predict(X)
            Z = Z.astype(int)
            Y = Y.astype(int)
            up = 2*np.sum(Z & Y, axis=1).astype(float)
            down1 = np.sum(Z, axis=1)
            down2 = np.sum(Y, axis=1)

            down = (down1 + down2)
            down[down==0] = 1.
            up[down==0] = 1.
            return np.mean(up / down)
        else:
            raise NotImplementedError(
                "criterion '%s' not implemented" % criterion)
