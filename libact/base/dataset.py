"""
The dataset class used in this package.
Datasets consists of data used for training, represented by a list of
(feature, label) tuples.
May be exported in different formats for application on other libraries.
"""
from __future__ import unicode_literals

import random
import numpy as np

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
        if X is None: X = []
        if y is None: y = []
        self.data = list(zip(X, y))
        self.modified = True
        self._update_callback = set()

    def __len__(self):
        """
        Number of all sample entries in this object.

        Returns
        -------
        n_samples : int
        """
        return len(self.data)

    def len_labeled(self):
        """
        Number of labeled data entries in this object.

        Returns
        -------
        n_samples : int
        """
        return len(self.get_labeled_entries())

    def len_unlabeled(self):
        """
        Number of unlabeled data entries in this object.

        Returns
        -------
        n_samples : int
        """
        return len(list(filter(lambda entry: entry[1] is None, self.data)))

    def get_num_of_labels(self):
        """
        Number of distinct lebels in this object.

        Returns
        -------
        n_labels : int
        """
        return len({entry[1] for entry in self.data if entry[1] is not None})

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
        self.data.append((feature, label))
        self.modified = True
        return len(self.data) - 1

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
        self.data[entry_id] = (self.data[entry_id][0], new_label)
        self.modified = True
        for callback in self._update_callback:
            callback(entry_id, new_label)

    def on_update(self, callback):
        """
        Add callback function to call when dataset updated.

        Parameters
        ----------
        callback : callable
            The function to be called when dataset is updated.
        """
        self._update_callback.add(callback)

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
        X, y = zip(*self.get_labeled_entries())
        return np.array(X), np.array(y)

    def get_entries(self):
        """
        Return the list of all sample feature and ground truth tuple.

        Returns
        -------
        data : list, shape = (n_samples)
            List of all sample feature and label tuple.
        """
        return self.data

    def get_labeled_entries(self):
        """
        Returns list of labeled feature and their label

        Returns
        -------
        labeled_entries : list of (feature, label) tuple
            Labeled entries
        """
        return list(filter(lambda entry: entry[1] is not None, self.data))

    def get_unlabeled_entries(self):
        """
        Returns list of unlabeled features, along with their entry_ids

        Returns
        -------
        unlabeled_entries : list of (entry_id, feature) tuple
            Labeled entries
        """
        return [
            (idx, entry[0]) for idx, entry in enumerate(self.data)
            if entry[1] is None
        ]

    def labeled_uniform_sample(self, sample_size, replace=True):
        """Returns a Dataset object with labeled data only, which is
        resampled uniformly with given sample size.
        Parameter `replace` decides whether sampling with replacement or not.

        Parameters
        ----------
        sample_size
        """
        if replace:
            samples = [
                random.choice(self.get_labeled_entries())
                for _ in range(sample_size)
            ]
        else:
            samples = random.sample(self.get_labeled_entries(), sample_size)
        return Dataset(*zip(*samples))


def import_libsvm_sparse(filename):
    """Imports dataset file in libsvm sparse format"""
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(filename)
    return Dataset(X.toarray().tolist(), y.tolist())


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
