import unittest

import numpy as np
from numpy.testing import assert_array_equal

from libact.base.dataset import Dataset
from libact.query_strategies import HintSVM

class UncertaintySamplingTestCase(unittest.TestCase):

    def setUp(self):
        self.X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [0, 1],
                  [0, -2], [1.5, 1.5], [-2, -2]]
        self.y = [2, 3, 4, 1, 2, 4]

    def test_hintsvm_multiclass_error(self):
        dataset = Dataset(self.X, np.concatenate([self.y[:6], [None] * 4]))
        qs = HintSVM(dataset)
        with self.assertRaises(ValueError):
            qs.make_query()



if __name__ == '__main__':
    unittest.main()
