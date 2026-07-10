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

    def test_label_random_state(self):
        """same random_state gives the same label for duplicate features
        carrying conflicting labels"""
        X = np.vstack([np.zeros((2, 3)), np.ones((2, 3))])
        y = np.array([1, 2, 3, 4])
        dataset = Dataset(X, y)
        lbr1 = IdealLabeler(dataset, random_state=1126)
        lbr2 = IdealLabeler(dataset, random_state=1126)
        labels1 = [lbr1.label(np.zeros(3)) for _ in range(20)]
        labels2 = [lbr2.label(np.zeros(3)) for _ in range(20)]
        self.assertEqual(labels1, labels2)
        self.assertTrue(set(labels1) <= {1, 2})

if __name__ == '__main__':
    unittest.main()
