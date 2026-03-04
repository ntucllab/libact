"""Tests for _get_scores() contract across all query strategies.

Verifies that every strategy implementing _get_scores() returns a consistent
format: a tuple of two numpy arrays (entry_ids, scores).
"""
import unittest

import numpy as np
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


class TestGetScoresContract(unittest.TestCase):
    """Verify _get_scores() contract across all strategies."""

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

    def _check_contract(self, qs):
        """Verify the _get_scores return format contract."""
        result = qs._get_scores()

        # Must return a tuple of two elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        entry_ids, scores = result

        # Both must be numpy arrays
        self.assertIsInstance(entry_ids, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)

        # Same length
        self.assertEqual(len(entry_ids), len(scores))

        # Length matches number of unlabeled samples
        self.assertEqual(len(entry_ids), self.n_unlabeled)

        # entry_ids should be valid indices into the dataset
        for eid in entry_ids:
            self.assertTrue(0 <= eid < len(qs.dataset))
            # and they should be unlabeled
            self.assertIsNone(qs.dataset[eid][1])

        # scores should be finite
        self.assertTrue(np.all(np.isfinite(scores)))

        # Consistency: make_query should return one of the entry_ids
        ask_id = qs.make_query()
        self.assertIn(ask_id, entry_ids)

    def test_uncertainty_sampling(self):
        qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        self._check_contract(qs)

    def test_uncertainty_sampling_sm(self):
        qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            method='sm'
        )
        self._check_contract(qs)

    def test_uncertainty_sampling_entropy(self):
        qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            method='entropy'
        )
        self._check_contract(qs)

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
        self._check_contract(qs)

    def test_coreset(self):
        qs = CoreSet(self.dataset, random_state=42)
        self._check_contract(qs)

    def test_coreset_cosine(self):
        # Use non-zero data for cosine metric
        ds = self._make_dataset()
        X_nonzero = np.abs(np.random.randn(30, 5)) + 0.1
        y_partial = list(self.y[:10]) + [None] * 20
        ds = Dataset(X_nonzero, y_partial)
        qs = CoreSet(ds, metric='cosine', random_state=42)
        result = qs._get_scores()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_epsilon_uncertainty_sampling(self):
        qs = EpsilonUncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            epsilon=0.2,
            random_state=42
        )
        self._check_contract(qs)

    def test_information_density(self):
        qs = InformationDensity(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=42
        )
        self._check_contract(qs)

    def test_density_weighted_meta(self):
        base_qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        qs = DensityWeightedMeta(self.dataset, base_qs, beta=1.0,
                                 random_state=42)
        self._check_contract(qs)

    def test_query_by_committee_vote(self):
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
        self._check_contract(qs)

    def test_query_by_committee_kl(self):
        qs = QueryByCommittee(
            self.dataset,
            models=[
                SklearnProbaAdapter(
                    LogisticRegression(C=c, max_iter=200, solver='liblinear')
                )
                for c in [0.01, 0.1, 1.0]
            ],
            disagreement='kl_divergence',
            random_state=42
        )
        self._check_contract(qs)

    def test_quire(self):
        qs = QUIRE(self.dataset)
        result = qs._get_scores()

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        entry_ids, scores = result
        self.assertIsInstance(entry_ids, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(entry_ids), len(scores))
        self.assertEqual(len(entry_ids), self.n_unlabeled)
        self.assertTrue(np.all(np.isfinite(scores)))

        ask_id = qs.make_query()
        self.assertIn(ask_id, entry_ids)

    def test_random_sampling(self):
        qs = RandomSampling(self.dataset, random_state=42)
        self._check_contract(qs)
        # Random sampling should return uniform scores
        entry_ids, scores = qs._get_scores()
        self.assertTrue(np.allclose(scores, scores[0]))
        self.assertTrue(np.allclose(scores, 1.0))

    @unittest.skipUnless(HAS_HINTSVM, "HintSVM C extension not compiled")
    def test_hintsvm(self):
        qs = HintSVM(self.dataset, random_state=42)
        self._check_contract(qs)

    @unittest.skipUnless(HAS_VARIANCE_REDUCTION,
                         "VarianceReduction C extension not compiled")
    def test_variance_reduction_raises(self):
        qs = VarianceReduction(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        with self.assertRaises(NotImplementedError):
            qs._get_scores()

    def test_albl_raises(self):
        """ALBL is a meta-strategy and does not implement _get_scores."""
        ds = self._make_dataset()
        qs1 = UncertaintySampling(
            ds,
            model=SklearnProbaAdapter(
                LogisticRegression(C=1., max_iter=200, solver='liblinear')
            )
        )
        qs2 = UncertaintySampling(
            ds,
            model=SklearnProbaAdapter(
                LogisticRegression(C=0.01, max_iter=200, solver='liblinear')
            ),
            method='entropy'
        )
        albl = ActiveLearningByLearning(
            ds,
            query_strategies=[qs1, qs2],
            T=20,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=42
        )
        with self.assertRaises(NotImplementedError):
            albl._get_scores()


class TestGetScoresEmptyPool(unittest.TestCase):
    """_get_scores on fully labeled dataset returns empty arrays."""

    def test_uncertainty_sampling_empty(self):
        np.random.seed(1126)
        X = np.random.randn(10, 5)
        y = np.random.choice([0, 1], size=10)
        full_ds = Dataset(X, y)
        qs = UncertaintySampling(
            full_ds,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        entry_ids, scores = qs._get_scores()
        self.assertEqual(len(entry_ids), 0)
        self.assertEqual(len(scores), 0)
        self.assertIsInstance(entry_ids, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)

    def test_bald_empty(self):
        np.random.seed(1126)
        X = np.random.randn(10, 5)
        y = np.random.choice([0, 1], size=10)
        full_ds = Dataset(X, y)
        qs = BALD(
            full_ds,
            models=[
                SklearnProbaAdapter(
                    LogisticRegression(C=c, max_iter=200, solver='liblinear')
                )
                for c in [0.01, 0.1, 1.0]
            ],
            random_state=42
        )
        entry_ids, scores = qs._get_scores()
        self.assertEqual(len(entry_ids), 0)
        self.assertEqual(len(scores), 0)
        self.assertIsInstance(entry_ids, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)

    def test_coreset_empty(self):
        np.random.seed(1126)
        X = np.random.randn(10, 5)
        y = np.random.choice([0, 1], size=10)
        full_ds = Dataset(X, y)
        qs = CoreSet(full_ds, random_state=42)
        entry_ids, scores = qs._get_scores()
        self.assertEqual(len(entry_ids), 0)
        self.assertEqual(len(scores), 0)
        self.assertIsInstance(entry_ids, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)

    def test_random_sampling_empty(self):
        np.random.seed(1126)
        X = np.random.randn(10, 5)
        y = np.random.choice([0, 1], size=10)
        full_ds = Dataset(X, y)
        qs = RandomSampling(full_ds, random_state=42)
        entry_ids, scores = qs._get_scores()
        self.assertEqual(len(entry_ids), 0)
        self.assertEqual(len(scores), 0)
        self.assertIsInstance(entry_ids, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)

    def test_density_weighted_meta_empty(self):
        np.random.seed(1126)
        X = np.random.randn(10, 5)
        y = np.random.choice([0, 1], size=10)
        full_ds = Dataset(X, y)
        base_qs = UncertaintySampling(
            full_ds,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        qs = DensityWeightedMeta(full_ds, base_qs, beta=1.0, random_state=42)
        entry_ids, scores = qs._get_scores()
        self.assertEqual(len(entry_ids), 0)
        self.assertEqual(len(scores), 0)
        self.assertIsInstance(entry_ids, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)


class TestGetScoresReturnScore(unittest.TestCase):
    """Verify return_score=True backward compatibility."""

    def setUp(self):
        np.random.seed(1126)
        X = np.random.randn(30, 5)
        y = np.random.choice([0, 1], size=30)
        y_partial = list(y[:10]) + [None] * 20
        self.dataset = Dataset(X, y_partial)

    def test_uncertainty_sampling_return_score(self):
        qs = UncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            )
        )
        ask_id, score_list = qs.make_query(return_score=True)
        self.assertIsInstance(ask_id, (int, np.integer))
        self.assertIsInstance(score_list, list)
        # Each element should be a tuple of (id, score)
        for item in score_list:
            self.assertEqual(len(item), 2)

    def test_epsilon_us_return_score(self):
        qs = EpsilonUncertaintySampling(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            epsilon=0.2,
            random_state=42
        )
        ask_id, score_list = qs.make_query(return_score=True)
        self.assertIsInstance(ask_id, (int, np.integer))
        self.assertIsInstance(score_list, list)
        for item in score_list:
            self.assertEqual(len(item), 2)

    def test_information_density_return_score(self):
        qs = InformationDensity(
            self.dataset,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=42
        )
        ask_id, score_list = qs.make_query(return_score=True)
        self.assertIsInstance(ask_id, (int, np.integer))
        self.assertIsInstance(score_list, list)
        for item in score_list:
            self.assertEqual(len(item), 2)


if __name__ == '__main__':
    unittest.main()
