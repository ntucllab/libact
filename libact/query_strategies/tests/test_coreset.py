"""Test Core-Set (k-Center Greedy) Query Strategy"""
import unittest
from unittest.mock import Mock

import numpy as np

from libact.base.dataset import Dataset
from libact.query_strategies import CoreSet
from libact.labelers import IdealLabeler


def init_dataset(X, y, n_labeled=4):
    """Initialize dataset with some labeled and some unlabeled samples."""
    labels = list(y[:n_labeled]) + [None] * (len(y) - n_labeled)
    return Dataset(X, labels)


class CoreSetTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1126)
        self.X = np.array([
            # Labeled points (cluster near origin)
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
            # Unlabeled points at varying distances
            [1.0, 0.0],   # distance ~1.0 from labeled
            [0.5, 0.5],   # distance ~0.5 from labeled
            [3.0, 3.0],   # distance ~4.24 from labeled (farthest)
            [2.0, 0.0],   # distance ~1.8 from labeled
            [0.3, 0.1],   # distance ~0.1 from labeled (closest)
            [5.0, 5.0],   # distance ~7.07 from labeled (very far)
        ])
        self.y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        self.fully_labeled_ds = Dataset(self.X, self.y)
        self.labeler = IdealLabeler(self.fully_labeled_ds)

    def test_returns_valid_entry_id(self):
        """Query should return a valid unlabeled entry ID."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        qs = CoreSet(trn_ds, random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()

        self.assertIn(ask_id, unlabeled_ids)

    def test_selects_farthest_point(self):
        """Should select the unlabeled point farthest from all labeled points."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        qs = CoreSet(trn_ds, random_state=42)

        ask_id = qs.make_query()

        # Point at [5.0, 5.0] (index 9) is farthest from labeled cluster
        self.assertEqual(ask_id, 9)

    def test_with_transformer(self):
        """Should use transformer for distance computation if provided."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)

        # Create mock transformer that doubles features
        mock_transformer = Mock()
        mock_transformer.transform.side_effect = lambda X: np.asarray(X) * 2

        qs = CoreSet(
            trn_ds,
            transformer=mock_transformer,
            random_state=42
        )
        ask_id = qs.make_query()

        # Verify transformer was called (twice: once for pool, once for labeled)
        self.assertEqual(mock_transformer.transform.call_count, 2)
        # Should still select the farthest point
        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        self.assertIn(ask_id, unlabeled_ids)

    def test_invalid_transformer(self):
        """Should raise TypeError if transformer lacks transform method."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)

        invalid_transformer = object()  # No transform method

        with self.assertRaises(TypeError):
            CoreSet(trn_ds, transformer=invalid_transformer)

    def test_cosine_metric(self):
        """Should work with cosine distance metric."""
        # Use data with no zero vectors (cosine is undefined for zero vectors)
        X_cos = np.array([
            [1.0, 0.1], [0.9, 0.2], [1.0, 0.3], [0.8, 0.1],  # labeled
            [0.1, 1.0], [0.5, 0.5], [0.2, 0.9], [1.0, 1.0],   # unlabeled
            [0.3, 0.1], [0.1, 0.5],
        ])
        y_cos = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        trn_ds = init_dataset(X_cos, y_cos, n_labeled=4)
        qs = CoreSet(trn_ds, metric='cosine', random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()

        self.assertIn(ask_id, unlabeled_ids)

    def test_reproducibility(self):
        """Same random_state should produce same queries."""
        trn_ds1 = init_dataset(self.X, self.y, n_labeled=4)
        trn_ds2 = init_dataset(self.X, self.y, n_labeled=4)

        qs1 = CoreSet(trn_ds1, random_state=42)
        qs2 = CoreSet(trn_ds2, random_state=42)

        ask_id1 = qs1.make_query()
        ask_id2 = qs2.make_query()

        self.assertEqual(ask_id1, ask_id2)

    def test_multiple_queries(self):
        """Should handle multiple queries correctly, covering the space."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        qs = CoreSet(trn_ds, random_state=42)

        queries = []
        for _ in range(6):
            ask_id = qs.make_query()
            queries.append(ask_id)
            lbl = self.labeler.label(self.X[ask_id])
            trn_ds.update(ask_id, lbl)

        # All queries should be unique
        self.assertEqual(len(queries), len(set(queries)))

    def test_empty_pool_error(self):
        """Should raise error when no unlabeled samples available."""
        trn_ds = Dataset(self.X, self.y)  # All labeled
        qs = CoreSet(trn_ds, random_state=42)

        with self.assertRaises(ValueError):
            qs.make_query()

    def test_no_labeled_data_fallback(self):
        """Should fall back to random selection when no labeled data exists."""
        # All unlabeled
        labels = [None] * len(self.y)
        trn_ds = Dataset(self.X, labels)
        qs = CoreSet(trn_ds, random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()

        self.assertIn(ask_id, unlabeled_ids)

    def test_get_scores(self):
        """_get_scores should return min-distances for all unlabeled points."""
        trn_ds = init_dataset(self.X, self.y, n_labeled=4)
        qs = CoreSet(trn_ds, random_state=42)

        scores = qs._get_scores()

        # Should have one score per unlabeled point
        unlabeled_ids = trn_ds.get_unlabeled_entries()[0]
        self.assertEqual(len(scores), len(unlabeled_ids))

        # Scores should be non-negative
        for entry_id, score in scores:
            self.assertGreaterEqual(score, 0.0)

        # The farthest point should have the highest score
        scores_dict = dict(scores)
        max_id = max(scores_dict, key=scores_dict.get)
        self.assertEqual(max_id, 9)  # [5.0, 5.0] is farthest


if __name__ == '__main__':
    unittest.main()
