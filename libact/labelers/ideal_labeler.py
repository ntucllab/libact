"""
Ideal/Noiseless labeler that returns true label

"""
from libact.base.interfaces import Labeler
import numpy as np

class IdealLabeler(Labeler):
    """    
    Provide the errorless/noiseless label to any feature vectors being queried.

    Attributes
    ----------
    features: numpy array
        an array of features used as the search keys for labels
    
    label: numpy array
        an array of noiesless labels corresponding to the features
    """ 

    def __init__(self, dataset, **kwargs):
        X, y = zip(*dataset.get_entries())
        #make sure the input dataset is fully labeled
        assert((np.array(y) != np.array(None)).all())
        self.X = X
        self.y = y

    def label(self, feature):
        return self.y[np.where([np.array_equal(x, feature) for x in self.X])[0]]
