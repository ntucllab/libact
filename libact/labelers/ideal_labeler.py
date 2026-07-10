"""
Ideal/Noiseless labeler that returns true label

"""
import numpy as np

from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from, check_random_state


class IdealLabeler(Labeler):

    """
    Provide the errorless/noiseless label to any feature vectors being queried.

    Parameters
    ----------
    dataset: Dataset object
        Dataset object with the ground-truth label for each sample.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate. If None, the global
        numpy random state is used, keeping the previous behavior. Only used
        to break ties when the queried feature matches multiple samples with
        different labels.

    """

    def __init__(self, dataset, random_state=None, **kwargs):
        X, y = dataset.get_entries()
        # make sure the input dataset is fully labeled
        assert (np.array(y) != np.array(None)).all()
        self.X = X
        self.y = y
        self.random_state_ = check_random_state(random_state)

    @inherit_docstring_from(Labeler)
    def label(self, feature):
        yy = self.y[np.where([np.array_equal(x, feature)
                              for x in self.X])[0]]
        ind = np.arange(len(yy))
        return yy[self.random_state_.choice(ind, 1)[0]]
