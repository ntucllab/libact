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
        """Constructor with scikit-learn style (X, y) data.
        A None label indicates an unlabeled entry.
        """
        self.data = list(zip(X, y))
        self.modified = True
        self._update_callback = set()

    def __len__(self):
        """Return the number of all data entries in this object."""
        return len(self.data)

    def len_labeled(self):
        """Return the number of labeled data entries in this object."""
        return len(self.get_labeled_entries())

    def len_unlabeled(self):
        """Return the number of unlabeled data entries in this object."""
        return len(list(filter(lambda entry: entry[1] is None, self.data)))

    def get_num_of_labels(self):
        return len({entry[1] for entry in self.data if entry[1] is not None})

    def append(self, feature, label=None):
        """Add a (feature, label) entry into the dataset.
        A None label indicates an unlabeled entry.
        Returns entry_id for updating labels.
        """
        self.data.append((feature, label))
        self.modified = True
        return len(self.data) - 1

    def update(self, entry_id, new_label):
        """Updates an entry at entry_id with the given new label"""
        entry = self.data[entry_id]
        self.data[entry_id] = (entry[0], new_label)
        self.modified = True
        for callback in self._update_callback:
            callback(entry_id, new_label)

    def on_update(self, callback):
        self._update_callback.add(callback)

    def format_sklearn(self):
        """Returns dataset in (X, y) format for use in scikit-learn.
        Unlabeled entries are ignored.
        """
        X, y = zip(*self.get_labeled_entries())
        return np.array(X), np.array(y)

    def get_labeled_entries(self):
        """Returns list of labeled features and their labels
        Format: [(feature, label), ...]
        """
        return list(filter(lambda entry: entry[1] is not None, self.data))

    def get_unlabeled_entries(self):
        """Returns list of unlabeled features, along with their entry_ids
        Format: [(entry_id, feature), ...]
        """
        return [
            (idx, entry[0]) for idx, entry in enumerate(self.data)
            if entry[1] is None
            ]

    def labeled_uniform_sample(self, samplesize, replace=True):
        """Returns a Dataset object with labeled data only, which is
        resampled uniformly with given sample size.
        Parameter `replace` decides whether sampling with replacement or not.
        """
        if replace:
            samples = [
                random.choice(self.get_labeled_entries())
                for _ in range(samplesize)
                ]
        else:
            samples = random.sample(self.get_labeled_entries(), samplesize)
        return Dataset(*zip(*samples))


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
