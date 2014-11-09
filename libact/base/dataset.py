"""
The dataset class used in this package.
Datasets consists of data used for training, represented by a list of
(feature, label) tuples.
May be exported in different formats for application on other libraries.
"""
import random

class Dataset(object):

    def __init__(self, X=[], y=[]):
        """Constructor with scikit-learn style (X, y) data"""
        self.data = list(zip(X, y))

    def __getitem__(self, key):
        """Allow list-like access: dataset[key]"""
        return self.data[key]

    def __len__(self):
        """Return the number of all data entries in this object."""
        return len(self.data)

    def len_labeled(self):
        """Return the number of labeled data entries in this object."""
        ret = 0
        for ent in self.data:
            if ent[1] != None:
                ret += 1
        return ret

    def add(self, feature, label):
        """Add a (feature, label) entry into the dataset.
        A None label indicates an unlabeled entry.
        Returns entry_id for updating labels.
        """
        self.data.append((feature, label))
        return len(self.data) - 1

    def update(self, entry_id, label):
        """Updates an entry at entry_id with the given new label"""
        self.data[entry_id] = (self.data[entry_id][0], label)

    def format_sklearn(self):
        """Returns dataset in (X, y) format for use in scikit-learn.
        Unlabeled entries are ignored.
        """
        l = list(zip(*[entry for entry in self.data if entry[1] is not None]))
        return (list(l[0]), list(l[1]))

    def get_unlabeled(self):
        """Returns list of entry_ids of unlabeled features"""
        return [entry_id for entry_id, entry in enumerate(self.data)
            if entry[1] == None]

    def uniform_sample(self, samplesize, replace=True):
        """Returns a Dataset object with is resampled uniformly with given
        samplsize. Parameter replace decides whether sampling with replacement
        or not"""
        ret = Dataset()
        if replace:
            for i in xrange(samplesize):
                ran = random.choice(self.data)
                ret.add(ran[0], ran[1])
            return ret
        else:
            ret.data = random.sample(self.data, samplesize)
            return ret
            

