"""
The dataset class used in this package.
Datasets consists of data used for training, represented by a list of
(feature, label) tuples.
May be exported in different formats for application on other libraries.
"""

import random
import numpy as np


class Dataset(object):

    def __init__(self, X=[], y=[]):
        """Constructor with scikit-learn style (X, y) data"""
        self.labeled, self.unlabeled = [], []
        for feature, label in zip(X, y):
            if label is None:
                self.unlabeled.append((feature, -1))
            else:
                self.labeled.append((feature, label))
        self.modified = True

    def __len__(self):
        """Return the number of all data entries in this object."""
        return len(self.labeled) + len(self.unlabeled)

    def len_labeled(self):
        """Return the number of labeled data entries in this object."""
        return len(self.labeled)

    def len_unlabeled(self):
        """Return the number of unlabeled data entries in this object."""
        return len(self.unlabeled)

    def get_num_of_labels(self):
        s = set()
        for d in self.labeled:
            s.add(d[1])
        return len(s)

    def append(self, feature, label=None):
        """Add a (feature, label) entry into the dataset.
        A None label indicates an unlabeled entry.
        If an unlabeled entry in inserted, returns entry_id for updating labels.
        """
        if label is None:
            self.unlabeled.append((feature, -1))
            return len(self.unlabeled) - 1
        else:
            self.labeled.append((feature, label))
            self.modified = True

    def update(self, entry_id, label):
        """Updates an entry at entry_id with the given new label"""
        entry = self.unlabeled[entry_id]
        if entry[1] == -1:  # not inserted yet
            self.labeled.append((entry[0], label))
            self.unlabeled[entry_id] = (entry[0], len(self.labeled) - 1)
        else:  # update existing entry in labeled pool
            self.labeled[entry[1]] = (entry[0], label)
        self.modified = True

    def format_sklearn(self):
        """Returns dataset in (X, y) format for use in scikit-learn.
        Unlabeled entries are ignored.
        """
        if self.modified:
            l = list(zip(*self.labeled))
            self.cache = (np.array(l[0]), np.array(l[1]))
            self.modified = False
        return self.cache

    def get_labeled_entries(self):
        """Returns list of labeled features and their labels
        Format: [(feature, label), ...]
        """
        return self.labeled

    def get_unlabeled_entries(self):
        """Returns list of unlabeled features, along with their entry_ids
        Format: [(entry_id, feature), ...]
        """
        return [
            (l[0], l[1][0])
            for l in zip(range(len(self.unlabeled)), self.unlabeled)
            if l[1][1] == -1
            ]

    def labeled_uniform_sample(self, samplesize, replace=True):
        """Returns a Dataset object with labeled data only, which is
        resampled uniformly with given sample size.
        Parameter `replace` decides whether sampling with replacement or not.
        """
        ret = Dataset()
        if replace:
            ret.labeled = [
                random.choice(self.labeled) for _ in range(samplesize)
                ]
        else:
            ret.labeled = random.sample(self.labeled, samplesize)
        return ret


def import_libsvm_sparse(filename):
    """Imports dataset file in libsvm sparse format"""
    entries = list()
    dim = 0
    with open(filename, 'r') as f:
        for line in f:
            cols = line.split()
            entry = dict()
            entry['label'] = int(cols[0])
            for col in cols[1:]:
                n_component = int(col.split(':')[0]) - 1  # start from 0
                value = float(col.split(':')[1])
                entry[n_component] = value
                if n_component > dim: dim = n_component + 1
            entries.append(entry)
    dataset = Dataset()
    for entry in entries:
        vec = np.zeros(dim)
        for n_component in entry:
            if type(n_component) is not str:
                vec[n_component] = entry[n_component]
        dataset.append(vec, entry['label'])
    return dataset
