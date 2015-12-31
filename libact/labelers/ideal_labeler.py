"""
Ideal/Noiseless labeler that returns true label

"""
import numpy as np

from libact.base.interfaces import Labeler

class IdealLabeler(Labeler):
    """    
    Provide the errorless/noiseless label to any feature vectors being queried.

    Parameters
    ----------
    dataset: Dataset object
        Dataset object with the ground-truth label for each sample.
        
    """ 

    def __init__(self, dataset, **kwargs):
        X, y = zip(*dataset.get_entries())
        #make sure the input dataset is fully labeled
        assert((np.array(y) != np.array(None)).all())
        self.X = X
        self.y = y

    def label(self, feature):
        return self.y[np.where([np.array_equal(x, feature) for x in self.X])[0]]
