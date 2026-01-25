"""Test Information Density Query Strategy"""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from libact.base.dataset import Dataset
from libact.base.interfaces import ProbabilisticModel, ContinuousModel
from libact.query_strategies import InformationDensity


def init_dataset(X, y, n_labeled=4):
    """Initialize dataset with some labeled and some unlabeled samples."""
    labels = list(y[:n_labeled]) + [None] * (len(y) - n_labeled)
    return Dataset(X, labels)


class MockProbModel(ProbabilisticModel):
    """Mock probabilistic model for testing."""

    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self._trained = False

    def train(self, dataset):
        self._trained = True

    def predict(self, feature):
        return np.zeros(len(feature), dtype=int)

    def score(self, testing_dataset):
        return 0.5

    def predict_proba(self, feature):
        n = len(feature)
        # Return probabilities that vary by sample index
        proba = np.zeros((n, self.n_classes))
        for i in range(n):
            # First sample very uncertain, last very confident
            p = 0.5 + 0.4 * (i / max(n - 1, 1))
            proba[i, 0] = p
            proba[i, 1] = 1.0 - p
        return proba


class MockContinuousModel(ContinuousModel):
    """Mock continuous model for testing."""

    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    def train(self, dataset):
        pass

    def predict(self, feature):
        return np.zeros(len(feature), dtype=int)

    def score(self, testing_dataset):
        return 0.5

    def predict_real(self, feature):
        n = len(feature)
        dvalue = np.zeros((n, self.n_classes))
        for i in range(n):
            dvalue[i, 0] = 0.5 + 0.4 * (i / max(n - 1, 1))
            dvalue[i, 1] = -(0.5 + 0.4 * (i / max(n - 1, 1)))
        return dvalue


class InformationDensityTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1126)
        self.X = np.array([
            # Labeled (cluster near origin)
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
            # Unlabeled - dense cluster
            [1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.1, 1.1],
            # Unlabeled - outliers
            [5.0, 5.0], [8.0, 8.0],
        ])
        self.y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])

    def test_returns_valid_entry_id(self):
        """Query should return a valid unlabeled entry ID."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()

        self.assertIn(ask_id, unlabeled_ids)

    def test_prefers_dense_regions(self):
        """Should prefer uncertain points in dense regions over outliers."""
        # Create a dataset where outlier is most uncertain but isolated
        X = np.array([
            [0.0, 0.0], [0.1, 0.1],  # labeled
            [1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.1, 1.1],  # dense cluster
            [10.0, 10.0],  # isolated outlier
        ])
        y = np.array([0, 1, 0, 1, 0, 1, 0])
        trn_ds = init_dataset(X, y, n_labeled=2)

        # Model that makes the outlier (index 6) most uncertain
        model = Mock(spec=ProbabilisticModel)
        model.train = Mock()
        proba = np.array([
            [0.9, 0.1],  # idx 2: confident
            [0.85, 0.15],  # idx 3: confident
            [0.8, 0.2],  # idx 4: somewhat confident
            [0.75, 0.25],  # idx 5: somewhat uncertain
            [0.5, 0.5],  # idx 6: maximally uncertain (outlier)
        ])
        model.predict_proba = Mock(return_value=proba)

        qs = InformationDensity(trn_ds, model=model, beta=2.0, random_state=42)
        ask_id = qs.make_query()

        # Should NOT pick the outlier (index 6) despite highest uncertainty
        self.assertNotEqual(ask_id, 6)
        # Should pick from the dense cluster (indices 2-5)
        self.assertIn(ask_id, [2, 3, 4, 5])

    def test_beta_zero_equals_uncertainty(self):
        """With beta=0, density has no effect (pure uncertainty)."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()

        qs = InformationDensity(trn_ds, model=model, beta=0.0, random_state=42)
        scores = qs._get_scores()

        # With beta=0, density^0 = 1 for all, so scores = uncertainty only
        # The first unlabeled point (most uncertain in MockProbModel) should score highest
        entry_ids, score_values = zip(*scores)
        score_values = list(score_values)
        max_idx = np.argmax(score_values)
        # First unlabeled has p=0.5 (max entropy)
        self.assertEqual(entry_ids[max_idx], 4)

    def test_method_lc(self):
        """Should work with least-confident method."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, method='lc',
                                random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()
        self.assertIn(ask_id, unlabeled_ids)

    def test_method_sm(self):
        """Should work with smallest-margin method."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, method='sm',
                                random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()
        self.assertIn(ask_id, unlabeled_ids)

    def test_method_entropy(self):
        """Should work with entropy method."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, method='entropy',
                                random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()
        self.assertIn(ask_id, unlabeled_ids)

    def test_continuous_model(self):
        """Should work with ContinuousModel using lc or sm."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockContinuousModel()
        qs = InformationDensity(trn_ds, model=model, method='lc',
                                random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()
        self.assertIn(ask_id, unlabeled_ids)

    def test_entropy_requires_probabilistic_model(self):
        """Should raise TypeError if entropy method used with ContinuousModel."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockContinuousModel()

        with self.assertRaises(TypeError):
            InformationDensity(trn_ds, model=model, method='entropy')

    def test_missing_model(self):
        """Should raise TypeError if model not provided."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)

        with self.assertRaises(TypeError):
            InformationDensity(trn_ds)

    def test_invalid_model_type(self):
        """Should raise TypeError for non-compatible model."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)

        with self.assertRaises(TypeError):
            InformationDensity(trn_ds, model=object())

    def test_invalid_method(self):
        """Should raise TypeError for unsupported method."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()

        with self.assertRaises(TypeError):
            InformationDensity(trn_ds, model=model, method='invalid')

    def test_cosine_metric(self):
        """Should work with cosine distance metric."""
        X_cos = np.array([
            [1.0, 0.1], [0.9, 0.2], [1.0, 0.3], [0.8, 0.1],
            [0.1, 1.0], [0.5, 0.5], [0.2, 0.9], [1.0, 1.0],
            [0.3, 0.1], [0.1, 0.5],
        ])
        y_cos = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        trn_ds = init_dataset(X_cos, y_cos, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, metric='cosine',
                                random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()
        self.assertIn(ask_id, unlabeled_ids)

    def test_reproducibility(self):
        """Same random_state should produce same queries."""
        trn_ds1 = init_dataset(self.X, self.y, n_labeled=4)
        trn_ds2 = init_dataset(self.X, self.y, n_labeled=4)
        model1 = MockProbModel()
        model2 = MockProbModel()

        qs1 = InformationDensity(trn_ds1, model=model1, random_state=42)
        qs2 = InformationDensity(trn_ds2, model=model2, random_state=42)

        self.assertEqual(qs1.make_query(), qs2.make_query())

    def test_return_score(self):
        """make_query(return_score=True) should return scores."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, random_state=42)

        result = qs.make_query(return_score=True)
        self.assertEqual(len(result), 2)
        ask_id, scores = result
        self.assertIsInstance(ask_id, (int, np.integer))
        self.assertEqual(len(scores), 6)  # 6 unlabeled

    def test_get_scores(self):
        """_get_scores should return density-weighted uncertainty."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, random_state=42)

        scores = qs._get_scores()
        unlabeled_ids = trn_ds.get_unlabeled_entries()[0]
        self.assertEqual(len(scores), len(unlabeled_ids))

        # All scores should be non-negative
        for entry_id, score in scores:
            self.assertGreaterEqual(score, 0.0)

    def test_empty_pool_error(self):
        """Should raise error when no unlabeled samples available."""
        trn_ds = Dataset(self.X, self.y)  # All labeled
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, random_state=42)

        with self.assertRaises(ValueError):
            qs.make_query()

    def test_multiple_queries(self):
        """Should handle multiple queries with dataset updates."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, random_state=42)

        queries = []
        for _ in range(3):
            ask_id = qs.make_query()
            queries.append(ask_id)
            trn_ds.update(ask_id, self.y[ask_id])

        # All queries should be unique
        self.assertEqual(len(queries), len(set(queries)))

    def test_density_scores_single_point(self):
        """Density with single unlabeled point should return 1.0."""
        X = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]])
        y = np.array([0, 1, 0])
        trn_ds = init_dataset(X, y, n_labeled=2)
        model = MockProbModel()
        qs = InformationDensity(trn_ds, model=model, random_state=42)

        density = qs._density_scores(np.array([[1.0, 1.0]]))
        self.assertEqual(density[0], 1.0)

    def test_continuous_model_large_decision_values(self):
        """Scores should be non-negative even when predict_real returns large values.

        ContinuousModel.predict_real can return unbounded decision values
        (e.g., SVM decision function), causing 1-max(dvalue) to go negative.
        The implementation must clamp uncertainty to >=0 for the multiplicative
        density combination to work correctly.
        """
        # Model that returns large decision values (like SVM far from boundary)
        class LargeDvalueModel(ContinuousModel):
            def train(self, dataset):
                pass

            def predict(self, feature):
                return np.zeros(len(feature), dtype=int)

            def score(self, testing_dataset):
                return 0.5

            def predict_real(self, feature):
                n = len(feature)
                dvalue = np.zeros((n, 2))
                for i in range(n):
                    # Decision values >> 1.0 (confident predictions)
                    d = 3.0 + i * 0.5
                    dvalue[i] = [-d, d]
                return dvalue

        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        model = LargeDvalueModel()
        qs = InformationDensity(trn_ds, model=model, method='lc',
                                random_state=42)

        scores = qs._get_scores()
        # All scores should be non-negative (uncertainty clamped to 0)
        for entry_id, score in scores:
            self.assertGreaterEqual(score, 0.0)

    def test_density_favors_dense_with_continuous_model(self):
        """With ContinuousModel, density should still favor dense regions.

        Even when uncertainty from predict_real is near zero for all points
        (all far from boundary), the algorithm should not invert density
        preference.
        """
        # Points: dense cluster + outlier
        X = np.array([
            [0.0, 0.0], [0.1, 0.1],  # labeled
            [1.0, 1.0], [1.1, 1.0], [1.0, 1.1],  # dense unlabeled cluster
            [10.0, 10.0],  # outlier
        ])
        y = np.array([0, 1, 0, 1, 0, 1])

        # Model where one dense point is slightly uncertain
        class MixedModel(ContinuousModel):
            def train(self, dataset):
                pass

            def predict(self, feature):
                return np.zeros(len(feature), dtype=int)

            def score(self, testing_dataset):
                return 0.5

            def predict_real(self, feature):
                n = len(feature)
                dvalue = np.zeros((n, 2))
                for i in range(n):
                    # First point near boundary, others far
                    d = 0.1 if i == 0 else 5.0
                    dvalue[i] = [-d, d]
                return dvalue

        trn_ds = init_dataset(X, y, n_labeled=2)
        model = MixedModel()
        qs = InformationDensity(trn_ds, model=model, method='lc',
                                beta=1.0, random_state=42)

        ask_id = qs.make_query()
        # Should pick from dense cluster (index 2 — the uncertain one)
        self.assertEqual(ask_id, 2)


if __name__ == '__main__':
    unittest.main()
