import unittest

import numpy as np
from sklearn import datasets

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler


class TestDatasetMethods(unittest.TestCase):

    initial_X = np.arange(15).reshape((5, 3))
    initial_y = np.array([1, 2, 3, 1, 4])

    def setup_dataset(self):
        return Dataset(self.initial_X, self.initial_y)

    def setup_mlc_dataset(self):
        X, Y = datasets.make_multilabel_classification(
                n_features=5, random_state=1126)
        return Dataset(X, Y)

    def test_label(self):
        dataset = self.setup_dataset()
        lbr = IdealLabeler(dataset)
        ask_id = lbr.label(np.array([0, 1, 2]))
        self.assertEqual(ask_id, 1)
        ask_id = lbr.label(np.array([6, 7, 8]))
        self.assertEqual(ask_id, 3)
        ask_id = lbr.label([12, 13, 14])
        self.assertEqual(ask_id, 4)

    def test_mlc_label(self):
        """test multi-label case"""
        dataset = self.setup_mlc_dataset()
        lbr = IdealLabeler(dataset)
        ask_id = lbr.label(np.array([12., 5., 2., 11., 14.]))
        np.testing.assert_array_equal(ask_id, [0, 1, 0, 0, 1])
        ask_id = lbr.label(np.array([ 6.,  2., 21., 20.,  5.]))
        np.testing.assert_array_equal(ask_id, [0, 0, 1, 0, 1])

if __name__ == '__main__':
    unittest.main()
