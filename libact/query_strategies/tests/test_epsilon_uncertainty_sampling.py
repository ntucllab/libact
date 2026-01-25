"""Test Epsilon Uncertainty Sampling (ε-US) Query Strategy"""
import unittest
from unittest.mock import Mock

import numpy as np

from libact.base.dataset import Dataset
from libact.base.interfaces import ProbabilisticModel
from libact.query_strategies import EpsilonUncertaintySampling
from libact.labelers import IdealLabeler


def make_mock_model(proba_values):
    """Create a mock ProbabilisticModel returning fixed probabilities."""
    model = Mock(spec=ProbabilisticModel)
    model.predict_proba = Mock(return_value=np.array(proba_values))
    model.train = Mock()
    return model


def init_dataset(X, y, n_labeled=4):
    """Initialize dataset with some labeled and some unlabeled samples."""
    labels = list(y[:n_labeled]) + [None] * (len(y) - n_labeled)
    return Dataset(X, labels)


class EpsilonUncertaintySamplingTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1126)
        self.X = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],  # labeled
            [1.0, 0.0], [0.5, 0.5], [3.0, 3.0], [2.0, 0.0],  # unlabeled
            [0.3, 0.1], [5.0, 5.0],
        ])
        self.y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        self.fully_labeled_ds = Dataset(self.X, self.y)
        self.labeler = IdealLabeler(self.fully_labeled_ds)

    def test_returns_valid_entry_id(self):
        """Query should return a valid unlabeled entry ID."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        # Mock model with uniform probabilities
        mock_model = make_mock_model([[0.5, 0.5]] * 6)

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.5, random_state=42
        )

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()

        self.assertIn(ask_id, unlabeled_ids)

    def test_epsilon_zero_is_pure_uncertainty(self):
        """With epsilon=0, should always use uncertainty sampling."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        # Model predicts one point as most uncertain (0.5, 0.5)
        mock_model = make_mock_model([
            [0.9, 0.1],  # idx 4 - confident
            [0.5, 0.5],  # idx 5 - most uncertain
            [0.8, 0.2],  # idx 6
            [0.7, 0.3],  # idx 7
            [0.6, 0.4],  # idx 8
            [0.95, 0.05],  # idx 9 - very confident
        ])

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.0, random_state=42
        )

        # With epsilon=0, should always pick the most uncertain (idx 5)
        ask_id = qs.make_query()
        self.assertEqual(ask_id, 5)

    def test_epsilon_one_is_pure_random(self):
        """With epsilon=1, should always use random sampling."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([[0.5, 0.5]] * 6)

        # Run many queries and check distribution is roughly uniform
        selections = []
        for seed in range(100):
            trn_ds_copy = init_dataset(self.X, self.y, n_labeled=4)
            qs = EpsilonUncertaintySampling(
                trn_ds_copy, model=mock_model, epsilon=1.0, random_state=seed
            )
            selections.append(qs.make_query())

        # All unlabeled IDs should appear (with high probability)
        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        selected_ids = set(selections)
        # At least 4 of 6 unlabeled IDs should be selected in 100 trials
        self.assertGreaterEqual(len(selected_ids & unlabeled_ids), 4)

    def test_epsilon_exploration_rate(self):
        """Exploration should happen approximately epsilon fraction of time."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        # Model has one clearly most uncertain point
        mock_model = make_mock_model([
            [0.99, 0.01],  # idx 4 - very confident
            [0.5, 0.5],    # idx 5 - most uncertain
            [0.99, 0.01],  # idx 6
            [0.99, 0.01],  # idx 7
            [0.99, 0.01],  # idx 8
            [0.99, 0.01],  # idx 9
        ])

        epsilon = 0.3
        n_trials = 200
        uncertain_selections = 0

        for seed in range(n_trials):
            trn_ds_copy = init_dataset(self.X, self.y, n_labeled=4)
            qs = EpsilonUncertaintySampling(
                trn_ds_copy, model=mock_model, epsilon=epsilon, random_state=seed
            )
            ask_id = qs.make_query()
            if ask_id == 5:  # Most uncertain point
                uncertain_selections += 1

        # With epsilon=0.3, exploitation happens 70% of time
        # Expected uncertainty selections ≈ 70% + (30% * 1/6) ≈ 75%
        # Allow reasonable variance
        exploitation_rate = uncertain_selections / n_trials
        self.assertGreater(exploitation_rate, 0.5)  # Should be mostly exploitation
        self.assertLess(exploitation_rate, 0.95)    # But some exploration

    def test_method_lc(self):
        """Least confident method should work."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([
            [0.9, 0.1], [0.5, 0.5], [0.8, 0.2],
            [0.7, 0.3], [0.6, 0.4], [0.95, 0.05],
        ])

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.0, method='lc', random_state=42
        )
        ask_id = qs.make_query()
        self.assertEqual(ask_id, 5)  # 0.5 is least confident

    def test_method_sm(self):
        """Smallest margin method should work."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([
            [0.9, 0.1], [0.51, 0.49], [0.8, 0.2],  # idx 5 has smallest margin
            [0.7, 0.3], [0.6, 0.4], [0.95, 0.05],
        ])

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.0, method='sm', random_state=42
        )
        ask_id = qs.make_query()
        self.assertEqual(ask_id, 5)  # 0.51-0.49=0.02 smallest margin

    def test_method_entropy(self):
        """Entropy method should work with ProbabilisticModel."""
        from libact.base.interfaces import ProbabilisticModel

        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = Mock(spec=ProbabilisticModel)
        mock_model.predict_proba = Mock(return_value=np.array([
            [0.9, 0.1], [0.5, 0.5], [0.8, 0.2],  # idx 5 has max entropy
            [0.7, 0.3], [0.6, 0.4], [0.95, 0.05],
        ]))
        mock_model.train = Mock()

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.0, method='entropy', random_state=42
        )
        ask_id = qs.make_query()
        self.assertEqual(ask_id, 5)  # 0.5, 0.5 has max entropy

    def test_invalid_epsilon(self):
        """Should raise error for epsilon outside [0, 1]."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([[0.5, 0.5]] * 6)

        with self.assertRaises(ValueError):
            EpsilonUncertaintySampling(
                trn_ds, model=mock_model, epsilon=-0.1, random_state=42
            )

        with self.assertRaises(ValueError):
            EpsilonUncertaintySampling(
                trn_ds, model=mock_model, epsilon=1.5, random_state=42
            )

    def test_missing_model(self):
        """Should raise error when model is not provided."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)

        with self.assertRaises(TypeError):
            EpsilonUncertaintySampling(trn_ds, epsilon=0.1, random_state=42)

    def test_invalid_model_type(self):
        """Should raise error for invalid model type."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        invalid_model = object()

        with self.assertRaises(TypeError):
            EpsilonUncertaintySampling(
                trn_ds, model=invalid_model, epsilon=0.1, random_state=42
            )

    def test_invalid_method(self):
        """Should raise error for invalid method."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([[0.5, 0.5]] * 6)

        with self.assertRaises(TypeError):
            EpsilonUncertaintySampling(
                trn_ds, model=mock_model, epsilon=0.1, method='invalid',
                random_state=42
            )

    def test_entropy_requires_probabilistic_model(self):
        """Entropy method should require ProbabilisticModel."""
        from libact.base.interfaces import ContinuousModel

        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = Mock(spec=ContinuousModel)
        mock_model.predict_real = Mock(return_value=np.array([[0.5, 0.5]] * 6))
        mock_model.train = Mock()

        with self.assertRaises(TypeError):
            EpsilonUncertaintySampling(
                trn_ds, model=mock_model, epsilon=0.1, method='entropy',
                random_state=42
            )

    def test_reproducibility(self):
        """Same random_state should produce same queries."""
        trn_ds1 = init_dataset(self.X, self.y, n_labeled=4)
        trn_ds2 = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([[0.5, 0.5]] * 6)

        qs1 = EpsilonUncertaintySampling(
            trn_ds1, model=mock_model, epsilon=0.5, random_state=42
        )
        qs2 = EpsilonUncertaintySampling(
            trn_ds2, model=mock_model, epsilon=0.5, random_state=42
        )

        ask_id1 = qs1.make_query()
        ask_id2 = qs2.make_query()

        self.assertEqual(ask_id1, ask_id2)

    def test_empty_pool_error(self):
        """Should raise error when no unlabeled samples available."""
        trn_ds = Dataset(self.X, self.y)  # All labeled
        mock_model = make_mock_model([[0.5, 0.5]] * 10)

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.5, random_state=42
        )

        with self.assertRaises(ValueError):
            qs.make_query()

    def test_return_score(self):
        """make_query with return_score=True should return scores."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([
            [0.9, 0.1], [0.5, 0.5], [0.8, 0.2],
            [0.7, 0.3], [0.6, 0.4], [0.95, 0.05],
        ])

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.0, random_state=42
        )
        ask_id, scores = qs.make_query(return_score=True)

        self.assertEqual(ask_id, 5)
        self.assertEqual(len(scores), 6)
        # Scores should be (entry_id, score) tuples
        for entry_id, score in scores:
            self.assertIsInstance(entry_id, (int, np.integer))
            self.assertIsInstance(score, (float, np.floating))

    def test_multiple_queries(self):
        """Should handle multiple queries correctly."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        mock_model = make_mock_model([[0.5, 0.5]] * 6)

        qs = EpsilonUncertaintySampling(
            trn_ds, model=mock_model, epsilon=0.5, random_state=42
        )

        queries = []
        for _ in range(4):
            ask_id = qs.make_query()
            queries.append(ask_id)
            lbl = self.labeler.label(self.X[ask_id])
            trn_ds.update(ask_id, lbl)

        # All queries should be from originally unlabeled set
        original_unlabeled = {4, 5, 6, 7, 8, 9}
        for q in queries:
            self.assertIn(q, original_unlabeled)


if __name__ == '__main__':
    unittest.main()
