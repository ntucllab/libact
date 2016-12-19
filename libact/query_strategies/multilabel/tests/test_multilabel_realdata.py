"""
This module uses real world dataset to test the active learning algorithms.
Since creating own artificial dataset for test is too time comsuming so we would
use real world data, fix the random seed, and compare the query sequence each
active learning algorithm produces.

The dataset yeast_train.svm is downloaded from
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html
"""
import random
import os
import unittest

from numpy.testing import assert_array_equal
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import LogisticRegression
from libact.query_strategies.multilabel import MMC
from ...tests.utils import run_qs


class MultilabelRealdataTestCase(unittest.TestCase):

    def setUp(self):
        dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'datasets/yeast_train.svm')
        X, y = load_svmlight_file(dataset_filepath, multilabel=True)
        self.X = X.todense().tolist()
        self.y = MultiLabelBinarizer().fit_transform(y).tolist()
        self.quota = 10

    def test_mmc(self):
        trn_ds = Dataset(self.X,
                         self.y[:5] + [None] * (len(self.y) - 5))
        qs = MMC(trn_ds, random_state=1126)
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(
            qseq, np.array([26, 178, 309, 717, 934, 854, 1430, 1222, 739, 1205]))


if __name__ == '__main__':
    unittest.main()
