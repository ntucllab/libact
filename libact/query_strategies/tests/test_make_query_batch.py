"""Tests for the make_query_batch() contract across all query strategies.

Covers the default top-k implementation on the QueryStrategy base class,
the strategy-specific overrides (RandomSampling, CoreSet,
EpsilonUncertaintySampling), the explicit NotImplementedError overrides
(ActiveLearningByLearning, VarianceReduction), and the argument
validation / error paths.
"""
import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy
from libact.models import SklearnProbaAdapter
from libact.query_strategies import (
    UncertaintySampling,
    BALD,
    CoreSet,
    EpsilonUncertaintySampling,
    InformationDensity,
    DensityWeightedMeta,
    DiversityWeightedMeta,
    QueryByCommittee,
    QUIRE,
    RandomSampling,
    ActiveLearningByLearning,
)

# Try importing C-extension strategies
try:
    from libact.query_strategies import HintSVM
    HAS_HINTSVM = True
except (ImportError, ModuleNotFoundError):
    HAS_HINTSVM = False

try:
    from libact.query_strategies import VarianceReduction
    HAS_VARIANCE_REDUCTION = True
except (ImportError, ModuleNotFoundError):
    HAS_VARIANCE_REDUCTION = False


class MockScoreStrategy(QueryStrategy):
    """Deterministic strategy with canned scores for contract tests.

    Scores are given in unlabeled-pool order. make_query is a plain
    argmax with no tie randomization, so make_query_batch(1) must equal
    make_query() exactly.
    """

    def __init__(self, dataset, scores):
        super(MockScoreStrategy, self).__init__(dataset)
        self.scores = np.asarray(scores, dtype=float)

    def _get_scores(self):
        entry_ids, _ = self.dataset.get_unlabeled_entries()
        return np.asarray(entry_ids), self.scores[:len(entry_ids)].copy()

    def make_query(self):
        entry_ids, scores = self._get_scores()
        return entry_ids[int(np.argmax(scores))]


class TestMakeQueryBatchContract(unittest.TestCase):
    """Verify the make_query_batch() contract across all strategies."""

    def setUp(self):
        np.random.seed(1126)
        self.X = np.random.randn(30, 5)
        self.y = np.random.choice([0, 1], size=30)
        # First 10 labeled, rest unlabeled
        y_partial = list(self.y[:10]) + [None] * 20
        self.dataset = Dataset(self.X, y_partial)
        self.n_unlabeled = 20

    def _make_dataset(self):
        """Create a fresh dataset for strategies that need their own copy."""
        np.random.seed(1126)
        X = np.random.randn(30, 5)
        y = np.random.choice([0, 1], size=30)
        y_partial = list(y[:10]) + [None] * 20
        return Dataset(X, y_partial)

    def _check_batch_contract(self, qs, batch_size=5):
        """Verify the make_query_batch return format contract."""
        batch = qs.make_query_batch(batch_size)

        self.assertIsInstance(batch, np.ndarray)
        self.assertTrue(np.issubdtype(batch.dtype, np.integer))
        self.assertEqual(len(batch), batch_size)
        # All ids pairwise distinct
        self.assertEqual(len(set(batch.tolist())), batch_size)
        # All ids valid and unlabeled
        for eid in batch:
            self.assertTrue(0 <= eid < len(qs.dataset))
            self.assertIsNone(qs.dataset[eid][1])

        # make_query must keep returning a single int (public contract)
        ask_id = qs.make_query()
        self.assertIsInstance(ask_id, (int, np.integer))
        self.assertNotIsInstance(ask_id, np.ndarray)

    def test_uncertainty_sampling(self):
        qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        self._check_batch_contract(qs)

    def test_uncertainty_sampling_entropy(self):
        qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            method='entropy'
        )
        self._check_batch_contract(qs)

    def test_bald(self):
        qs = BALD(
            self.dataset,
            models=[
                SklearnProbaAdapter(
                    LogisticRegression(C=c, max_iter=200, solver='liblinear')
                )
                for c in [0.01, 0.1, 1.0]
            ],
            random_state=42
        )
        self._check_batch_contract(qs)

    def test_coreset(self):
        qs = CoreSet(self.dataset, random_state=42)
        self._check_batch_contract(qs)

    def test_epsilon_uncertainty_sampling(self):
        qs = EpsilonUncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            epsilon=0.2,
            random_state=42
        )
        self._check_batch_contract(qs)

    def test_information_density(self):
        qs = InformationDensity(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=42
        )
        self._check_batch_contract(qs)

    def test_density_weighted_meta(self):
        base_qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        qs = DensityWeightedMeta(self.dataset, base_qs, beta=1.0,
                                 random_state=42)
        self._check_batch_contract(qs)

    def test_diversity_weighted_meta(self):
        base_qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        qs = DiversityWeightedMeta(self.dataset, base_qs, random_state=42)
        self._check_batch_contract(qs)

    def test_query_by_committee(self):
        qs = QueryByCommittee(
            self.dataset,
            models=[
                SklearnProbaAdapter(
                    LogisticRegression(C=c, max_iter=200, solver='liblinear')
                )
                for c in [0.01, 0.1, 1.0]
            ],
            random_state=42
        )
        self._check_batch_contract(qs)

    def test_quire(self):
        qs = QUIRE(self.dataset)
        self._check_batch_contract(qs)

    def test_random_sampling(self):
        qs = RandomSampling(self.dataset, random_state=42)
        self._check_batch_contract(qs)

    @unittest.skipUnless(HAS_HINTSVM, "HintSVM C extension not compiled")
    def test_hintsvm(self):
        qs = HintSVM(self.dataset, random_state=42)
        self._check_batch_contract(qs)

    def test_albl_raises(self):
        """ALBL is inherently sequential and must reject batch queries."""
        ds = self._make_dataset()
        qs1 = UncertaintySampling(
            ds,
            model=SklearnProbaAdapter(
                LogisticRegression(C=1., max_iter=200, solver='liblinear')
            )
        )
        albl = ActiveLearningByLearning(
            ds,
            query_strategies=[qs1],
            T=20,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=42
        )
        with self.assertRaises(NotImplementedError) as cm:
            albl.make_query_batch(5)
        self.assertIn("sequential", str(cm.exception))

    @unittest.skipUnless(HAS_VARIANCE_REDUCTION,
                         "VarianceReduction C extension not compiled")
    def test_variance_reduction_raises(self):
        qs = VarianceReduction(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        with self.assertRaises(NotImplementedError) as cm:
            qs.make_query_batch(5)
        self.assertIn("batch", str(cm.exception))


class TestMakeQueryBatchDefaultImpl(unittest.TestCase):
    """Pin the default implementation's exact ordering semantics with a
    deterministic mock (no model training, no randomness)."""

    def setUp(self):
        X = np.arange(16).reshape(8, 2)
        # entries 0-2 labeled, 3-7 unlabeled
        y = [0, 1, 0, None, None, None, None, None]
        self.dataset = Dataset(X, y)
        self.unlabeled_ids = [3, 4, 5, 6, 7]

    def test_descending_score_order(self):
        qs = MockScoreStrategy(self.dataset, [5., 1., 4., 2., 3.])
        batch = qs.make_query_batch(3)
        # ids by descending score: 5.0->3, 4.0->5, 3.0->7
        np.testing.assert_array_equal(batch, [3, 5, 7])

    def test_stable_tie_order(self):
        # Equal scores must keep original pool order (stable sort).
        qs = MockScoreStrategy(self.dataset, [5., 5., 3., 3., 1.])
        batch = qs.make_query_batch(4)
        np.testing.assert_array_equal(batch, [3, 4, 5, 6])

    def test_negative_scores_rank_preserved(self):
        # Direction-preserving for any magnitude/sign (HintSVM-style raw
        # scores must never be re-interpreted).
        qs = MockScoreStrategy(self.dataset, [-5., -1., -0.5, -2., -3.])
        batch = qs.make_query_batch(3)
        # descending: -0.5->5, -1->4, -2->6
        np.testing.assert_array_equal(batch, [5, 4, 6])

    def test_adversarial_magnitudes_rank_preserved(self):
        qs = MockScoreStrategy(self.dataset, [1000., 1., 0.5, 0.4, 2.])
        batch = qs.make_query_batch(5)
        np.testing.assert_array_equal(batch, [3, 7, 4, 5, 6])

    def test_batch_of_one_equals_make_query(self):
        # Holds exactly for strategies whose make_query is a plain argmax.
        qs = MockScoreStrategy(self.dataset, [1., 9., 4., 2., 3.])
        self.assertEqual(qs.make_query_batch(1)[0], qs.make_query())

    def test_full_pool_batch(self):
        qs = MockScoreStrategy(self.dataset, [5., 1., 4., 2., 3.])
        batch = qs.make_query_batch(5)
        np.testing.assert_array_equal(batch, [3, 5, 7, 6, 4])
        self.assertEqual(sorted(batch.tolist()), self.unlabeled_ids)


class TestMakeQueryBatchErrors(unittest.TestCase):
    """Argument validation and error paths."""

    def setUp(self):
        X = np.arange(16).reshape(8, 2)
        y = [0, 1, 0, None, None, None, None, None]
        self.dataset = Dataset(X, y)
        self.qs = MockScoreStrategy(self.dataset, [5., 1., 4., 2., 3.])

    def test_zero_batch_size(self):
        with self.assertRaises(ValueError):
            self.qs.make_query_batch(0)

    def test_negative_batch_size(self):
        with self.assertRaises(ValueError):
            self.qs.make_query_batch(-3)

    def test_float_batch_size(self):
        with self.assertRaises(TypeError):
            self.qs.make_query_batch(2.0)

    def test_bool_batch_size(self):
        with self.assertRaises(TypeError):
            self.qs.make_query_batch(True)

    def test_string_batch_size(self):
        with self.assertRaises(TypeError):
            self.qs.make_query_batch('2')

    def test_none_batch_size(self):
        with self.assertRaises(TypeError):
            self.qs.make_query_batch(None)

    def test_numpy_integer_batch_size_accepted(self):
        batch = self.qs.make_query_batch(np.int64(2))
        self.assertEqual(len(batch), 2)

    def test_batch_size_equal_to_pool_allowed(self):
        batch = self.qs.make_query_batch(5)
        self.assertEqual(len(batch), 5)

    def test_batch_size_exceeds_pool(self):
        with self.assertRaises(ValueError) as cm:
            self.qs.make_query_batch(6)
        self.assertIn("exceeds", str(cm.exception))

    def test_empty_pool(self):
        full_ds = Dataset(np.arange(6).reshape(3, 2), [0, 1, 0])
        qs = MockScoreStrategy(full_ds, [])
        with self.assertRaises(ValueError):
            qs.make_query_batch(1)

    def test_no_get_scores_raises_not_implemented(self):
        class NoScores(QueryStrategy):
            def make_query(self):
                return 0

        qs = NoScores(self.dataset)
        with self.assertRaises(NotImplementedError):
            qs.make_query_batch(2)


class TestRandomSamplingBatch(unittest.TestCase):
    """RandomSampling override: uniform sampling without replacement."""

    def setUp(self):
        X = np.arange(16).reshape(8, 2)
        y = [0, 1, 0, None, None, None, None, None]
        self.dataset = Dataset(X, y)

    def test_full_pool_is_permutation(self):
        qs = RandomSampling(self.dataset, random_state=3)
        batch = qs.make_query_batch(5)
        self.assertEqual(sorted(batch.tolist()), [3, 4, 5, 6, 7])

    def test_reproducible_with_seed(self):
        qs1 = RandomSampling(self.dataset, random_state=7)
        qs2 = RandomSampling(self.dataset, random_state=7)
        np.testing.assert_array_equal(
            qs1.make_query_batch(3), qs2.make_query_batch(3))

    def test_not_always_first_k(self):
        # A stable top-k of the uniform scores would always return the
        # first pool entries; the override must actually randomize.
        first_k = [3, 4]
        differs = False
        for seed in range(10):
            qs = RandomSampling(self.dataset, random_state=seed)
            if qs.make_query_batch(2).tolist() != first_k:
                differs = True
                break
        self.assertTrue(differs)


class TestCoreSetBatch(unittest.TestCase):
    """CoreSet override: true iterative k-center greedy."""

    def test_greedy_diverges_from_static_topk(self):
        # Labeled point at 0; unlabeled at 10, 10.1 (near-duplicates far
        # from the label) and 5. Static top-2 of the min-distances picks
        # both near-duplicates; true greedy covers 10.1 after picking
        # 10.1's neighbor and takes 5 instead.
        X = np.array([[0.], [10.], [10.1], [5.]])
        y = [0, None, None, None]
        ds = Dataset(X, y)
        qs = CoreSet(ds, random_state=0)

        entry_ids, scores = qs._get_scores()
        static_top2 = np.asarray(entry_ids)[
            np.argsort(-scores, kind='stable')[:2]]
        np.testing.assert_array_equal(static_top2, [2, 1])

        batch = qs.make_query_batch(2)
        np.testing.assert_array_equal(batch, [2, 3])

    def test_greedy_sparse_input(self):
        X = sp.csr_matrix(np.array([[0.], [10.], [10.1], [5.]]))
        y = [0, None, None, None]
        ds = Dataset(X, y)
        qs = CoreSet(ds, random_state=0)

        batch = qs.make_query_batch(2)
        np.testing.assert_array_equal(batch, [2, 3])

    def test_get_scores_sparse_input(self):
        # Regression: _get_scores used scipy cdist, which crashes on
        # sparse input. pairwise_distances must give identical dense
        # results.
        X_dense = np.array([[0., 1.], [10., 0.], [3., 4.]])
        y = [0, None, None]
        scores_dense = CoreSet(Dataset(X_dense, y))._get_scores()[1]
        scores_sparse = CoreSet(
            Dataset(sp.csr_matrix(X_dense), y))._get_scores()[1]
        np.testing.assert_allclose(scores_dense, scores_sparse)

    def test_no_labeled_data_random_seed_then_greedy(self):
        X = np.array([[0.], [1.], [100.]])
        ds = Dataset(X, [None, None, None])
        qs = CoreSet(ds, random_state=0)

        expected_first = np.random.RandomState(0).randint(0, 3)
        batch = qs.make_query_batch(2)
        self.assertEqual(batch[0], expected_first)
        # The second pick maximizes the distance to the first.
        distances = np.abs(X.ravel() - X[expected_first, 0])
        distances[expected_first] = -np.inf
        self.assertEqual(batch[1], int(np.argmax(distances)))

    def test_transformer_honored_in_greedy_loop(self):
        # Without the transformer the farthest point (euclidean, 2-D) is
        # entry 1; projecting onto the first feature makes it entry 2.
        X = np.array([[0., 0.], [1., 9.], [8., 0.], [7.5, 0.2]])
        y = [0, None, None, None]

        qs_plain = CoreSet(Dataset(X, y), random_state=0)
        self.assertEqual(qs_plain.make_query_batch(1)[0], 1)

        class Project0:
            def transform(self, X):
                return np.asarray(X)[:, :1]

        qs_proj = CoreSet(Dataset(X, y), transformer=Project0(),
                          random_state=0)
        self.assertEqual(qs_proj.make_query_batch(1)[0], 2)

    def test_metric_honored_in_greedy_loop(self):
        # Euclidean-far but cosine-identical vs euclidean-near but
        # cosine-orthogonal.
        X = np.array([[1., 0.], [10., 0.], [0., 0.5]])
        y = [0, None, None]

        qs_euclid = CoreSet(Dataset(X, y), random_state=0)
        self.assertEqual(qs_euclid.make_query_batch(1)[0], 1)

        qs_cosine = CoreSet(Dataset(X, y), metric='cosine', random_state=0)
        self.assertEqual(qs_cosine.make_query_batch(1)[0], 2)


class TestEpsilonUncertaintySamplingBatch(unittest.TestCase):
    """EpsilonUncertaintySampling override: binomial explore/exploit mix."""

    def setUp(self):
        np.random.seed(1126)
        X = np.random.randn(30, 5)
        y = np.random.choice([0, 1], size=30)
        y_partial = list(y[:10]) + [None] * 20
        self.dataset = Dataset(X, y_partial)

    def _make_qs(self, epsilon, seed=42):
        return EpsilonUncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            epsilon=epsilon,
            random_state=seed,
        )

    def test_epsilon_zero_is_exact_topk(self):
        qs = self._make_qs(epsilon=0.0)
        batch = qs.make_query_batch(5)
        entry_ids, scores = qs._get_scores()
        expected = np.asarray(entry_ids)[
            np.argsort(-scores, kind='stable')[:5]]
        np.testing.assert_array_equal(batch, expected)

    def test_epsilon_one_is_valid_random_sample(self):
        qs = self._make_qs(epsilon=1.0)
        batch = qs.make_query_batch(5)
        self.assertEqual(len(set(batch.tolist())), 5)
        for eid in batch:
            self.assertIsNone(self.dataset[eid][1])

    def test_epsilon_one_reproducible(self):
        b1 = self._make_qs(epsilon=1.0, seed=7).make_query_batch(5)
        b2 = self._make_qs(epsilon=1.0, seed=7).make_query_batch(5)
        np.testing.assert_array_equal(b1, b2)

    def test_always_exactly_batch_size_distinct(self):
        for epsilon in [0.0, 0.3, 0.7, 1.0]:
            qs = self._make_qs(epsilon=epsilon)
            batch = qs.make_query_batch(8)
            self.assertEqual(len(batch), 8)
            self.assertEqual(len(set(batch.tolist())), 8)

    def test_full_pool_any_epsilon(self):
        for epsilon in [0.0, 0.5, 1.0]:
            qs = self._make_qs(epsilon=epsilon)
            batch = qs.make_query_batch(20)
            self.assertEqual(sorted(batch.tolist()), list(range(10, 30)))


if __name__ == '__main__':
    unittest.main()
