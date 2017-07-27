"""
jSRE labeler that returns true label

"""
import numpy as np

from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from


class JSRELabeler(Labeler):

    """
    Provide the errorless label to any feature vectors being queried.

    Parameters
    ----------
    dataset: Dataset object
        Dataset object with the ground-truth label for each sample.

    """

    def __init__(self, dataset, **kwargs):
        X, y = zip(*dataset.get_entries())
        # make sure the input dataset is fully labeled
        assert (np.array(y) != np.array(None)).all()
        self.X = X
        self.y = y

    @inherit_docstring_from(Labeler)
    def label(self, feature):
        idx = self.X.index(feature)
        return self.y[idx] 
