"""Tests for DiversityWeightedMeta.

The fixtures pin two properties the wrapper exists to provide:
(1) within-batch diversity — near-duplicate points are not selected
together, with a control assertion proving the plain top-k of the base
strategy WOULD select them; and (2) rank-faithfulness — the base
strategy's preference ranking is followed exactly through a monotone
normalization, whatever the magnitude or sign of the raw scores
(HintSVM-style scores must never be re-interpreted).
"""
import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy
from libact.models import SklearnProbaAdapter
from libact.query_strategies import DiversityWeightedMeta, UncertaintySampling
from libact.query_strategies.diversity_weighted_meta import _minmax_normalize


class MockScoreStrategy(QueryStrategy):
    """Deterministic strategy with canned scores (unlabeled-pool order)."""

    def __init__(self, dataset, scores):
        super(MockScoreStrategy, self).__init__(dataset)
        self.scores = np.asarray(scores, dtype=float)

    def _get_scores(self):
        entry_ids, _ = self.dataset.get_unlabeled_entries()
        return np.asarray(entry_ids), self.scores[:len(entry_ids)].copy()

    def make_query(self):
        entry_ids, scores = self._get_scores()
        return entry_ids[int(np.argmax(scores))]


def _duplicate_cluster_fixture(sparse=False):
    """Two near-duplicate top-scoring points plus a far third candidate.

    Entry 0 (10, 0) and entry 1 (10.01, 0) are near-duplicates holding the
    two highest scores; entry 2 (0, 10) is high-scoring but far away;
    entry 3 (0, 0) is a low-scoring filler.
    """
    X = np.array([[10., 0.], [10.01, 0.], [0., 10.], [0., 0.]])
    if sparse:
        X = sp.csr_matrix(X)
    dataset = Dataset(X, [None, None, None, None])
    base = MockScoreStrategy(dataset, [10., 9.99, 9., 1.])
    return dataset, base


class TestDiversityWeightedMetaValidation(unittest.TestCase):

    def setUp(self):
        X = np.arange(8).reshape(4, 2)
        self.dataset = Dataset(X, [0, None, None, None])
        self.base = MockScoreStrategy(self.dataset, [3., 2., 1.])

    def test_base_must_be_query_strategy(self):
        with self.assertRaises(TypeError):
            DiversityWeightedMeta(self.dataset, base_query_strategy=object())

    def test_base_must_share_dataset(self):
        other = Dataset(np.arange(8).reshape(4, 2), [0, None, None, None])
        with self.assertRaises(ValueError):
            DiversityWeightedMeta(other, base_query_strategy=self.base)

    def test_lmbda_range(self):
        for bad in [-0.1, 1.5]:
            with self.assertRaises(ValueError):
                DiversityWeightedMeta(
                    self.dataset, base_query_strategy=self.base, lmbda=bad)

    def test_transformer_must_have_transform(self):
        with self.assertRaises(TypeError):
            DiversityWeightedMeta(
                self.dataset, base_query_strategy=self.base,
                transformer=object())

    def test_candidate_pool_size_type(self):
        for bad in ['3', True, 2.0]:
            with self.assertRaises(TypeError):
                DiversityWeightedMeta(
                    self.dataset, base_query_strategy=self.base,
                    candidate_pool_size=bad)
        with self.assertRaises(ValueError):
            DiversityWeightedMeta(
                self.dataset, base_query_strategy=self.base,
                candidate_pool_size=0)


class TestRankFaithfulness(unittest.TestCase):
    """The normalization must be monotone and direction-preserving."""

    def test_minmax_preserves_ranking(self):
        for scores in ([1000., 1., 0.5, 0.4],
                       [-5., -1., -0.5],
                       [3., 1., 2.],
                       [0.9, 0.1, 0.5]):
            scores = np.asarray(scores)
            np.testing.assert_array_equal(
                np.argsort(_minmax_normalize(scores)), np.argsort(scores))

    def test_minmax_constant_input(self):
        np.testing.assert_array_equal(
            _minmax_normalize([2., 2., 2.]), [0.5, 0.5, 0.5])

    def test_inverted_semantics_scores_followed(self):
        # HintSVM stand-in: raw scores are confidence-flavored, but the
        # strategy's preference is still its argmax. The wrapper must
        # follow that ranking without re-interpretation.
        X = np.array([[0., 0.], [5., 5.], [9., 1.]])
        dataset = Dataset(X, [None, None, None])
        base = MockScoreStrategy(dataset, [0.9, 0.1, 0.5])
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.0, random_state=0)

        self.assertEqual(qs.make_query(), base.make_query())
        np.testing.assert_array_equal(qs.make_query_batch(3), [0, 2, 1])

    def test_lmbda_zero_equals_base_topk(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        dataset = Dataset(X, [0, 1, 0, None, None, None, None, None])
        base = MockScoreStrategy(dataset, [5., 1., 4., 2., 3.])
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.0, random_state=0)
        np.testing.assert_array_equal(
            qs.make_query_batch(3), base.make_query_batch(3))


class TestDiversity(unittest.TestCase):

    def test_avoids_near_duplicates_with_control(self):
        dataset, base = _duplicate_cluster_fixture()

        # CONTROL: the plain top-k of the base strategy picks both
        # near-duplicates.
        np.testing.assert_array_equal(base.make_query_batch(2), [0, 1])

        # The wrapper keeps the base argmax but replaces the duplicate
        # with the far high-scoring point.
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.5, random_state=0)
        batch = qs.make_query_batch(2)
        np.testing.assert_array_equal(batch, [0, 2])

    def test_sparse_input(self):
        dataset, base = _duplicate_cluster_fixture(sparse=True)
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.5, random_state=0)
        np.testing.assert_array_equal(qs.make_query_batch(2), [0, 2])

    def test_constant_scores_spread_geometrically(self):
        # With a constant-score base (e.g. RandomSampling), selection is
        # driven purely by geometry: whatever the tie-randomized first
        # pick, the second pick must be far from it. Looping over seeds
        # exercises different tie-randomized first picks.
        X = np.array([[0.], [0.01], [10.], [5.]])
        dataset = Dataset(X, [None] * 4)
        for seed in range(8):
            base = MockScoreStrategy(dataset, [1., 1., 1., 1.])
            qs = DiversityWeightedMeta(
                dataset, base_query_strategy=base, lmbda=0.5,
                random_state=seed)
            batch = qs.make_query_batch(2)
            self.assertGreater(abs(X[batch[0], 0] - X[batch[1], 0]), 4.9)

    def test_transformer_honored(self):
        # A transformer collapsing all points to one location removes all
        # geometric information, so the wrapper falls back to score order
        # and selects the near-duplicate it would otherwise avoid.
        class Collapse:
            def transform(self, X):
                n = X.shape[0]
                return np.zeros((n, 2))

        dataset, base = _duplicate_cluster_fixture()
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.5,
            transformer=Collapse(), random_state=0)
        np.testing.assert_array_equal(qs.make_query_batch(2), [0, 1])

    def test_candidate_pool_size_caps_greedy(self):
        X = np.array([[0., 0.], [0.01, 0.], [0.02, 0.], [100., 0.]])
        dataset = Dataset(X, [None] * 4)
        base = MockScoreStrategy(dataset, [10., 9., 8., 1.])

        full = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.9, random_state=0)
        np.testing.assert_array_equal(full.make_query_batch(2), [0, 3])

        capped = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.9,
            candidate_pool_size=3, random_state=0)
        np.testing.assert_array_equal(capped.make_query_batch(2), [0, 2])


class TestApiContract(unittest.TestCase):

    def test_make_query_returns_base_argmax_int(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        dataset = Dataset(X, [0, 1, None, None, None])
        base = MockScoreStrategy(dataset, [1., 5., 3.])
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, random_state=0)
        ask_id = qs.make_query()
        self.assertIsInstance(ask_id, (int, np.integer))
        self.assertEqual(ask_id, 3)

    def test_get_scores_is_passthrough(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        dataset = Dataset(X, [0, 1, None, None, None])
        base = MockScoreStrategy(dataset, [1., 5., 3.])
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, random_state=0)
        ids_w, scores_w = qs._get_scores()
        ids_b, scores_b = base._get_scores()
        np.testing.assert_array_equal(ids_w, ids_b)
        np.testing.assert_array_equal(scores_w, scores_b)

    def test_full_pool_batch_is_score_ordered(self):
        dataset, base = _duplicate_cluster_fixture()
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, lmbda=0.5, random_state=0)
        np.testing.assert_array_equal(qs.make_query_batch(4), [0, 1, 2, 3])

    def test_error_paths(self):
        dataset, base = _duplicate_cluster_fixture()
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, random_state=0)
        with self.assertRaises(ValueError):
            qs.make_query_batch(0)
        with self.assertRaises(ValueError):
            qs.make_query_batch(5)
        with self.assertRaises(TypeError):
            qs.make_query_batch(2.0)

    def test_empty_pool(self):
        dataset = Dataset(np.arange(6).reshape(3, 2), [0, 1, 0])
        base = MockScoreStrategy(dataset, [])
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, random_state=0)
        with self.assertRaises(ValueError):
            qs.make_query()
        with self.assertRaises(ValueError):
            qs.make_query_batch(1)

    def test_with_real_base_strategy(self):
        np.random.seed(1126)
        X = np.random.randn(30, 5)
        y = np.random.choice([0, 1], size=30)
        dataset = Dataset(X, list(y[:10]) + [None] * 20)
        base = UncertaintySampling(
            dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        qs = DiversityWeightedMeta(
            dataset, base_query_strategy=base, random_state=42)
        batch = qs.make_query_batch(5)
        self.assertEqual(len(set(batch.tolist())), 5)
        for eid in batch:
            self.assertIsNone(dataset[eid][1])
        self.assertIsInstance(qs.make_query(), (int, np.integer))


if __name__ == '__main__':
    unittest.main()
