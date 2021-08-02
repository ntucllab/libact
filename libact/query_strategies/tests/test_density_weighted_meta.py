import unittest
import os

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling
from libact.labelers import IdealLabeler
from ..density_weighted_meta import DensityWeightedMeta
from .utils import run_qs


class DensityWeightedMetaTestCase(unittest.TestCase):

    def setUp(self):
        dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'datasets/heart_scale')
        self.X, self.y = import_libsvm_sparse(
            dataset_filepath).format_sklearn()
        self.quota = 10

    def test_density_weighted_meta_uncertainty_lc(self):
        trn_ds = Dataset(self.X[:20], np.concatenate([self.y[:6], [None] * 14]))
        base_qs = UncertaintySampling(
            trn_ds, method='lc',
            model=LogisticRegression(solver='liblinear', multi_class="ovr"))
        similarity_metric = cosine_similarity
        clustering_method = KMeans(n_clusters=3, random_state=1126)
        qs = DensityWeightedMeta(
            dataset=trn_ds, base_query_strategy=base_qs,
            similarity_metric=similarity_metric,
            clustering_method=clustering_method,
            beta=1.0, random_state=1126)
        model = LogisticRegression(solver='liblinear', multi_class="ovr")
        qseq = run_qs(trn_ds, qs, self.y, self.quota)
        assert_array_equal(qseq, np.array([13, 18,  9, 12,  8, 16, 10, 19, 15, 17]))


if __name__ == '__main__':
    unittest.main()
