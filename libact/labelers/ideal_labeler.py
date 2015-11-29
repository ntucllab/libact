"""
Ideal/Noiseless labeler that returns true label
"""
from libact.base.interfaces import Labeler

class IdealLabeler(Labeler):
    def __init__(self, dataset, true_y, **kwargs):
        X, y = zip(*dataset.data)
        feature = [list(x) for x in X]
        self.feature = feature
        self.true_y = true_y

    def label(self, feature):
        return self.true_y[self.feature.index(list(feature))]
