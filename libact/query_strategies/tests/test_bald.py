"""Test BALD Query Strategy"""
import unittest
from unittest.mock import Mock, MagicMock

import numpy as np

from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SklearnProbaAdapter
from libact.query_strategies import BALD
from libact.labelers import IdealLabeler


def init_dataset(X, y, n_labeled=6):
    """Initialize dataset with some labeled and some unlabeled samples."""
    labels = list(y[:n_labeled]) + [None] * (len(y) - n_labeled)
    return Dataset(X, labels)


class BALDTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(1126)
        self.X = np.array([
            [-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1],
            [0, 1], [0, -2], [1.5, 1.5], [-2, -2]
        ])
        self.y = np.array([-1, -1, -1, 1, 1, 1, -1, -1, 1, 1])
        self.fully_labeled_ds = Dataset(self.X, self.y)
        self.labeler = IdealLabeler(self.fully_labeled_ds)

    def test_with_provided_models(self):
        """Should work with pre-provided list of models."""
        trn_ds = init_dataset(self.X, self.y)

        models = [
            LogisticRegression(solver='liblinear', C=0.1),
            LogisticRegression(solver='liblinear', C=1.0),
            LogisticRegression(solver='liblinear', C=10.0),
        ]

        qs = BALD(trn_ds, models=models, random_state=42)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()

        self.assertIn(ask_id, unlabeled_ids)

    def test_with_base_model_bagging(self):
        """Should create ensemble via bagging from base model."""
        from sklearn.linear_model import LogisticRegression as SklearnLR

        trn_ds = init_dataset(self.X, self.y)

        base_model = SklearnProbaAdapter(
            SklearnLR(solver='liblinear', random_state=42)
        )

        qs = BALD(
            trn_ds,
            base_model=base_model,
            n_models=5,
            random_state=42
        )

        self.assertEqual(len(qs.models), 5)

        unlabeled_ids = set(trn_ds.get_unlabeled_entries()[0])
        ask_id = qs.make_query()

        self.assertIn(ask_id, unlabeled_ids)

    def test_bald_score_computation(self):
        """Verify BALD score is computed correctly: H[mean(P)] - mean(H[P])."""
        trn_ds = init_dataset(self.X, self.y)

        models = [
            LogisticRegression(solver='liblinear', C=1.0),
            LogisticRegression(solver='liblinear', C=1.0),
        ]

        qs = BALD(trn_ds, models=models, random_state=42)

        # Get scores
        scores = qs._get_scores()
        self.assertGreater(len(scores), 0)

        # All BALD scores should be non-negative (MI is non-negative)
        for entry_id, score in scores:
            self.assertGreaterEqual(score, -1e-10)  # Allow small numerical errors

    def test_update_retrains_ensemble(self):
        """Update should retrain the ensemble with new data."""
        from sklearn.linear_model import LogisticRegression as SklearnLR

        trn_ds = init_dataset(self.X, self.y)

        base_model = SklearnProbaAdapter(
            SklearnLR(solver='liblinear', random_state=42)
        )

        qs = BALD(
            trn_ds,
            base_model=base_model,
            n_models=3,
            random_state=42
        )

        # Make a query and update
        ask_id = qs.make_query()
        lbl = self.labeler.label(self.X[ask_id])
        trn_ds.update(ask_id, lbl)

        # After update, models should have been retrained
        # Make another query - should still work
        ask_id2 = qs.make_query()
        remaining_unlabeled = set(trn_ds.get_unlabeled_entries()[0])
        self.assertIn(ask_id2, remaining_unlabeled)

    def test_empty_models_list(self):
        """Should raise ValueError for empty models list."""
        trn_ds = init_dataset(self.X, self.y)

        with self.assertRaises(ValueError):
            BALD(trn_ds, models=[])

    def test_non_probabilistic_model(self):
        """Should raise TypeError for non-ProbabilisticModel."""
        from libact.models import SVM

        trn_ds = init_dataset(self.X, self.y)

        with self.assertRaises(TypeError):
            BALD(trn_ds, models=[SVM()])

    def test_missing_required_args(self):
        """Should raise TypeError if neither models nor base_model provided."""
        trn_ds = init_dataset(self.X, self.y)

        with self.assertRaises(TypeError):
            BALD(trn_ds)

    def test_base_model_without_clone(self):
        """Should raise TypeError if base_model lacks clone method."""
        trn_ds = init_dataset(self.X, self.y)

        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.random.rand(4, 2))
        # Remove clone method
        del mock_model.clone

        with self.assertRaises(TypeError):
            BALD(trn_ds, base_model=mock_model)

    def test_reproducibility(self):
        """Same random_state should produce same queries."""
        from sklearn.linear_model import LogisticRegression as SklearnLR

        base_model = SklearnProbaAdapter(
            SklearnLR(solver='liblinear', random_state=42)
        )

        trn_ds1 = init_dataset(self.X, self.y)
        qs1 = BALD(
            trn_ds1,
            base_model=base_model,
            n_models=3,
            random_state=42
        )

        trn_ds2 = init_dataset(self.X, self.y)
        qs2 = BALD(
            trn_ds2,
            base_model=base_model,
            n_models=3,
            random_state=42
        )

        # Run several queries and compare
        queries1, queries2 = [], []
        for _ in range(3):
            ask_id1 = qs1.make_query()
            ask_id2 = qs2.make_query()
            queries1.append(ask_id1)
            queries2.append(ask_id2)

            lbl1 = self.labeler.label(self.X[ask_id1])
            lbl2 = self.labeler.label(self.X[ask_id2])
            trn_ds1.update(ask_id1, lbl1)
            trn_ds2.update(ask_id2, lbl2)

        self.assertEqual(queries1, queries2)

    def test_multiple_queries(self):
        """Should handle multiple queries correctly."""
        trn_ds = init_dataset(self.X, self.y)

        models = [
            LogisticRegression(solver='liblinear', C=c)
            for c in [0.1, 1.0, 10.0]
        ]

        qs = BALD(trn_ds, models=models, random_state=42)

        # Make multiple queries
        queries = []
        for _ in range(4):
            ask_id = qs.make_query()
            queries.append(ask_id)
            lbl = self.labeler.label(self.X[ask_id])
            trn_ds.update(ask_id, lbl)

        # All queries should be unique (no repetition)
        self.assertEqual(len(queries), len(set(queries)))

    def test_entropy_calculation(self):
        """Test the entropy helper function."""
        trn_ds = init_dataset(self.X, self.y)
        models = [LogisticRegression(solver='liblinear')]
        qs = BALD(trn_ds, models=models, random_state=42)

        # Uniform distribution should have maximum entropy
        uniform = np.array([[0.5, 0.5]])
        entropy_uniform = qs._entropy(uniform)

        # Certain distribution should have zero entropy
        certain = np.array([[1.0, 0.0]])
        entropy_certain = qs._entropy(certain)

        self.assertGreater(entropy_uniform[0], entropy_certain[0])
        self.assertAlmostEqual(entropy_certain[0], 0.0, places=5)

    def test_empty_unlabeled_pool(self):
        """Should raise error when no unlabeled samples available."""
        trn_ds = Dataset(self.X, self.y)  # All labeled
        models = [LogisticRegression(solver='liblinear')]
        qs = BALD(trn_ds, models=models, random_state=42)

        with self.assertRaises(ValueError):
            qs.make_query()


if __name__ == '__main__':
    unittest.main()
