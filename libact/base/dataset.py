"""
The dataset class used in this package.
Datasets consists of data used for training, represented by a list of
(feature, label) tuples.
May be exported in different formats for application on other libraries.
"""
from __future__ import unicode_literals

import random
import numpy as np
import scipy.sparse as sp

from libact.utils import zip


class Dataset(object):

    """libact dataset object

    Parameters
    ----------
    X : {array-like}, shape = (n_samples, n_features)
        Feature of sample set.

    y : list of {int, None}, shape = (n_samples)
        The ground truth (label) for corresponding sample. Unlabeled data
        should be given a label None.

    Attributes
    ----------
    data : list, shape = (n_samples)
        List of all sample feature and label tuple.

    """

    def __init__(self, X=None, y=None):
        if X is None: X = np.array([])
        elif not isinstance(X, sp.csr_matrix):
            X = np.array(X)

        if y is None: y = []
        y = np.array(y, dtype=object)

        self._X = X
        self._y = y
        self.modified = True
        self._update_callback = set()
        self._batch_update_callback = set()

    def __len__(self):
        """
        Number of all sample entries in this object.

        Returns
        -------
        n_samples : int
        """
        return self._X.shape[0]

    def __getitem__(self, idx):
        # still provide the interface to direct access the data by index
        return self._X[idx], self._y[idx]

    @property
    def data(self): return self

    def get_labeled_mask(self):
        """
        Get the mask of labeled entries.

        Returns
        -------
        mask: numpy array of bool, shape = (n_sample, )
        """
        return ~np.fromiter((e is None for e in self._y), dtype=bool)

    def len_labeled(self):
        """
        Number of labeled data entries in this object.

        Returns
        -------
        n_samples : int
        """
        return self.get_labeled_mask().sum()

    def len_unlabeled(self):
        """
        Number of unlabeled data entries in this object.

        Returns
        -------
        n_samples : int
        """
        return (~self.get_labeled_mask()).sum()

    def get_num_of_labels(self):
        """
        Number of distinct lebels in this object.

        Returns
        -------
        n_labels : int
        """
        return np.unique(self._y[self.get_labeled_mask()]).size

    def append(self, feature, label=None):
        """
        Add a (feature, label) entry into the dataset.
        A None label indicates an unlabeled entry.

        Parameters
        ----------
        feature : {array-like}, shape = (n_features)
            Feature of the sample to append to dataset.

        label : {int, None}
            Label of the sample to append to dataset. None if unlabeled.

        Returns
        -------
        entry_id : {int}
            entry_id for the appened sample.
        """
        if isinstance(self._X, np.ndarray):
            self._X = np.vstack([self._X, feature])
        else: # sp.csr_matrix
            self._X = sp.vstack([self._X, feature])
        self._y = np.append(self._y, label)

        self.modified = True
        return len(self) - 1

    def update(self, entry_id, new_label):
        """
        Updates an entry with entry_id with the given label

        Parameters
        ----------
        entry_id : int
            entry id of the sample to update.

        label : {int, None}
            Label of the sample to be update.
        """
        self._y[entry_id] = new_label
        self.modified = True
        for callback in self._update_callback:
            callback(entry_id, new_label)

    def update_batch(self, entry_ids, labels):
        """Update multiple entries with their labels in a single call.

        Labels are applied one entry at a time, in the given order.
        Observers registered through :py:meth:`on_update_batch` (every
        :py:class:`libact.base.interfaces.QueryStrategy` registers
        itself there) are notified exactly once, with the whole batch,
        after all labels have been applied — so strategies that retrain
        a model in their update hook can train once per batch instead of
        once per entry. Callbacks registered only through
        :py:meth:`on_update` observe the same incremental per-entry
        ``(entry_id, label)`` stream as the equivalent series of
        individual ``update()`` calls.

        Note that some observers impose assumptions of their own on the
        update stream; for instance ActiveLearningByLearning assumes each
        update corresponds to an entry it has itself queried via
        ``make_query()``, and updating other entries is unsupported —
        exactly as with individual ``update()`` calls.

        Parameters
        ----------
        entry_ids : array-like of int, shape (n_updates,)
            Distinct entry ids of the samples to update.

        labels : sequence, shape (n_updates,)
            Label for each entry. None marks an entry as unlabeled.

        Raises
        ------
        ValueError
            If entry_ids is not one-dimensional, labels is not a sequence,
            their lengths differ, entry_ids contains duplicate entries, or
            any entry id is out of range (not in ``[0, len(dataset))``).
        """
        entry_ids = np.asarray(entry_ids)
        if entry_ids.ndim != 1:
            raise ValueError(
                "entry_ids must be a one-dimensional array-like; for a "
                "single entry use update(entry_id, label)")
        try:
            n_labels = len(labels)
        except TypeError:
            raise ValueError(
                "labels must be a sequence of the same length as entry_ids")
        if entry_ids.shape[0] != n_labels:
            raise ValueError(
                "entry_ids and labels must have the same length, got "
                "%d and %d" % (entry_ids.shape[0], n_labels))
        if len(np.unique(entry_ids)) != entry_ids.shape[0]:
            raise ValueError("entry_ids contains duplicate entries")
        # Validate the ids are all in range *before* applying any update, so
        # a bad id fails cleanly instead of leaving a partial update behind
        # (and so negative ids are rejected rather than silently wrapping
        # around to the wrong entry via numpy indexing).
        if entry_ids.size and (entry_ids.min() < 0 or
                               entry_ids.max() >= len(self)):
            raise ValueError(
                "entry_ids must all be in [0, %d), got values in [%s, %s]"
                % (len(self), entry_ids.min(), entry_ids.max()))

        if entry_ids.shape[0] == 0:
            return

        # Batch-aware observers are notified once after the whole batch
        # is applied; skip their per-entry callback so they are not
        # notified twice for the same labels.
        batch_owners = set(
            id(getattr(callback, '__self__', callback))
            for callback in self._batch_update_callback
        )
        per_entry_callbacks = [
            callback for callback in self._update_callback
            if id(getattr(callback, '__self__', callback)) not in batch_owners
        ]

        self.modified = True
        for entry_id, label in zip(entry_ids, labels):
            self._y[entry_id] = label
            for callback in per_entry_callbacks:
                callback(entry_id, label)

        for callback in self._batch_update_callback:
            callback(entry_ids, labels)

    def on_update(self, callback):
        """
        Add callback function to call when dataset updated.

        Parameters
        ----------
        callback : callable
            The function to be called when dataset is updated.
        """
        self._update_callback.add(callback)

    def on_update_batch(self, callback):
        """
        Add callback function to call once per :py:meth:`update_batch`.

        During :py:meth:`update_batch`, an observer registered here is
        notified exactly once with the full batch, after all labels have
        been applied, instead of receiving the per-entry callbacks it
        registered via :py:meth:`on_update`. Individual
        :py:meth:`update` calls still notify only the per-entry
        callbacks.

        Parameters
        ----------
        callback : callable
            Called as ``callback(entry_ids, labels)`` where ``entry_ids``
            is an np.ndarray of the updated entry ids and ``labels`` the
            matching labels.
        """
        self._batch_update_callback.add(callback)

    def format_sklearn(self):
        """
        Returns dataset in (X, y) format for use in scikit-learn.
        Unlabeled entries are ignored.

        Returns
        -------
        X : numpy array, shape = (n_samples, n_features)
            Sample feature set.

        y : numpy array, shape = (n_samples)
            Sample labels.
        """
        # becomes the same as get_labled_entries
        X, y = self.get_labeled_entries()
        return X, np.array(y)

    def get_entries(self):
        """
        Return the list of all sample feature and ground truth tuple.

        Returns
        -------
        X: numpy array or scipy matrix, shape = ( n_sample, n_features )
        y: numpy array, shape = (n_samples)
        """
        return self._X, self._y

    def get_labeled_entries(self):
        """
        Returns list of labeled feature and their label

        Returns
        -------
        X: numpy array or scipy matrix, shape = ( n_sample labeled, n_features )
        y: list, shape = (n_samples lebaled)
        """
        return self._X[self.get_labeled_mask()], self._y[self.get_labeled_mask()].tolist()

    def get_unlabeled_entries(self):
        """
        Returns list of unlabeled features, along with their entry_ids

        Returns
        -------
        idx: numpy array, shape = (n_samples unlebaled)
        X: numpy array or scipy matrix, shape = ( n_sample unlabeled, n_features )
        """
        return np.where(~self.get_labeled_mask())[0], self._X[~self.get_labeled_mask()]

    def labeled_uniform_sample(self, sample_size, replace=True):
        """Returns a Dataset object with labeled data only, which is
        resampled uniformly with given sample size.
        Parameter `replace` decides whether sampling with replacement or not.

        Parameters
        ----------
        sample_size
        """
        idx = np.random.choice(np.where(self.get_labeled_mask())[0],
                               size=sample_size, replace=replace)
        return Dataset(self._X[idx], self._y[idx])


def import_libsvm_sparse(filename):
    """Imports dataset file in libsvm sparse format"""
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(filename)
    return Dataset(X.toarray(), y)


def import_scipy_mat(filename):
    from scipy.io import loadmat
    data = loadmat(filename)
    X = data['X']
    y = data['y']
    zipper = list(zip(X, y))
    np.random.shuffle(zipper)
    X, y = zip(*zipper)
    X, y = np.array(X), np.array(y).reshape(-1)
    return Dataset(X, y)
