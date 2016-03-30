import unittest

import numpy as np

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler


class TestDatasetMethods(unittest.TestCase):

    initial_X = np.arange(15).reshape((5, 3))
    initial_y = np.array([1, 2, 3, 1, 4])

    def setup_dataset(self):
        return Dataset(self.initial_X, self.initial_y)

    def test_label(self):
        dataset = self.setup_dataset()
        lbr = IdealLabeler(dataset)
        ask_id = lbr.label(np.array([0, 1, 2]))
        self.assertEqual(ask_id, 1)
        ask_id = lbr.label(np.array([6, 7, 8]))
        self.assertEqual(ask_id, 3)
        ask_id = lbr.label([12, 13, 14])
        self.assertEqual(ask_id, 4)

if __name__ == '__main__':
    unittest.main()
