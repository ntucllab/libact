"""
Base interfaces for use in the package.
The package works according to the interfaces defined below.
"""
import numbers

from six import with_metaclass

from abc import ABCMeta, abstractmethod

import numpy as np


class QueryStrategy(with_metaclass(ABCMeta, object)):

    """Pool-based query strategy

    A QueryStrategy advices on which unlabeled data to be queried next given
    a pool of labeled and unlabeled data.
    """

    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        dataset.on_update(self.update)
        dataset.on_update_batch(self.update_batch)

    @property
    def dataset(self):
        """The Dataset object that is associated with this QueryStrategy."""
        return self._dataset

    def update(self, entry_id, label):
        """Update the internal states of the QueryStrategy after each queried
        sample being labeled.

        Parameters
        ----------
        entry_id : int
            The index of the newly labeled sample.

        label : float
            The label of the queried sample.
        """

    def update_batch(self, entry_ids, labels):
        """Update the internal states of the QueryStrategy after a batch
        of queried samples has been labeled.

        Called exactly once by
        :py:meth:`libact.base.dataset.Dataset.update_batch`, after all
        labels in the batch have been applied to the dataset. The default
        implementation replays the per-entry :py:meth:`update` hook once
        per ``(entry_id, label)`` pair, in order, preserving the
        semantics of strategies that do per-entry bookkeeping.
        Strategies whose update hook retrains a model should override
        this method to train once on the fully updated dataset instead
        of once per entry.

        Parameters
        ----------
        entry_ids : array-like of int, shape (n_updates,)
            Entry ids of the newly labeled samples.

        labels : sequence, shape (n_updates,)
            The label of each newly labeled sample.
        """
        for entry_id, label in zip(entry_ids, labels):
            self.update(entry_id, label)

    def _get_scores(self):
        """Return acquisition scores for all unlabeled samples.

        Subclasses should override this method to enable batch mode queries
        and score-based strategy composition.

        Returns
        -------
        entry_ids : np.ndarray, shape (n_unlabeled,)
            Global entry IDs of unlabeled samples.
        scores : np.ndarray, shape (n_unlabeled,)
            Acquisition scores. Higher = more informative.

        Raises
        ------
        NotImplementedError
            If the strategy does not support per-sample scoring.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _get_scores(). "
            "This is required for batch mode and score-based composition."
        )

    @staticmethod
    def _check_batch_size(batch_size, n_unlabeled):
        """Validate make_query_batch arguments.

        Parameters
        ----------
        batch_size : int
            The requested batch size.

        n_unlabeled : int
            Number of unlabeled samples currently in the pool.

        Raises
        ------
        TypeError
            If batch_size is not an integer (bool is rejected).

        ValueError
            If batch_size < 1, the pool is empty, or batch_size exceeds the
            pool size.
        """
        if isinstance(batch_size, bool) or \
                not isinstance(batch_size, numbers.Integral):
            raise TypeError(
                "batch_size must be an integer, got %r" % (batch_size,))
        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1, got %d" % batch_size)
        if n_unlabeled == 0:
            raise ValueError("No unlabeled samples available")
        if batch_size > n_unlabeled:
            raise ValueError(
                "batch_size (%d) exceeds the number of unlabeled samples "
                "(%d)" % (batch_size, n_unlabeled))

    def make_query_batch(self, batch_size):
        """Return a batch of distinct unlabeled samples to be queried.

        The default implementation ranks the unlabeled pool by the
        acquisition scores from :py:meth:`_get_scores` and returns the
        ``batch_size`` highest-scoring entry ids. Strategies override this
        method when a faithful batch generalization differs from top-k
        (e.g. iterative k-center for CoreSet, sampling without replacement
        for RandomSampling).

        Unlike :py:meth:`make_query`, ties are broken deterministically
        (stable sort, original pool order), so ``make_query_batch(1)`` may
        differ from ``make_query()`` for strategies that randomize
        tie-breaking.

        Parameters
        ----------
        batch_size : int
            Number of samples to query. Must satisfy
            ``1 <= batch_size <= n_unlabeled``. No silent clamping is
            performed.

        Returns
        -------
        entry_ids : np.ndarray of int, shape (batch_size,)
            Distinct entry ids of the samples to be queried, most preferred
            first.

        Raises
        ------
        TypeError
            If batch_size is not an integer.

        ValueError
            If batch_size < 1, batch_size exceeds the number of unlabeled
            samples, or there are no unlabeled samples.

        NotImplementedError
            If the strategy does not support per-sample scoring through
            :py:meth:`_get_scores`.
        """
        entry_ids, scores = self._get_scores()
        self._check_batch_size(batch_size, len(entry_ids))

        order = np.argsort(-np.asarray(scores, dtype=float), kind='stable')
        return np.asarray(entry_ids)[order[:batch_size]]

    @abstractmethod
    def make_query(self):
        """Return the index of the sample to be queried and labeled. Read-only.

        No modification to the internal states.

        Returns
        -------
        ask_id : int
            The index of the next unlabeled sample to be queried and labeled.
        """


class Labeler(with_metaclass(ABCMeta, object)):

    """Label the queries made by QueryStrategies

    Assign labels to the samples queried by QueryStrategies.
    """
    @abstractmethod
    def label(self, feature):
        """Return the class labels for the input feature array.

        Parameters
        ----------
        feature : array-like, shape (n_features,)
            The feature vector whose label is to queried.

        Returns
        -------
        label : int
            The class label of the queried feature.
        """


class Model(with_metaclass(ABCMeta, object)):

    """Classification Model

    A Model returns a class-predicting function for future samples after
    trained on a training dataset.
    """
    @abstractmethod
    def train(self, dataset, *args, **kwargs):
        """Train a model according to the given training dataset.

        Parameters
        ----------
        dataset : Dataset object
             The training dataset the model is to be trained on.

        Returns
        -------
        self : object
            Returns self.
        """

    @abstractmethod
    def predict(self, feature, *args, **kwargs):
        """Predict the class labels for the input samples

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The unlabeled samples whose labels are to be predicted.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            The class labels for samples in the feature array.
        """

    @abstractmethod
    def score(self, testing_dataset, *args, **kwargs):
        """Return the mean accuracy on the test dataset

        Parameters
        ----------
        testing_dataset : Dataset object
            The testing dataset used to measure the perforance of the trained
            model.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """


class MultilabelModel(Model):
    """Multilabel Classification Model

    A Model returns a multilabel-predicting function for future samples after
    trained on a training dataset.
    """


class ContinuousModel(Model):

    """Classification Model with intermediate continuous output

    A continuous classification model is able to output a real-valued vector
    for each features provided.
    """
    @abstractmethod
    def predict_real(self, feature, *args, **kwargs):
        """Predict confidence scores for samples.

        Returns the confidence score for each (sample, class) combination.

        The larger the value for entry (sample=x, class=k) is, the more
        confident the model is about the sample x belonging to the class k.

        Take Logistic Regression as example, the return value is the signed
        distance of that sample to the hyperplane.

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The samples whose confidence scores are to be predicted.

        Returns
        -------
        X : array-like, shape (n_samples, n_classes)
            Each entry is the confidence scores per (sample, class)
            combination.
        """


class ProbabilisticModel(ContinuousModel):

    """Classification Model with probability output

    A probabilistic classification model is able to output a real-valued vector
    for each features provided.
    """
    def predict_real(self, feature, *args, **kwargs):
        return self.predict_proba(feature, *args, **kwargs)

    @abstractmethod
    def predict_proba(self, feature, *args, **kwargs):
        """Predict probability estimate for samples.

        Parameters
        ----------
        feature : array-like, shape (n_samples, n_features)
            The samples whose probability estimation are to be predicted.

        Returns
        -------
        X : array-like, shape (n_samples, n_classes)
            Each entry is the prabablity estimate for each class.
        """
