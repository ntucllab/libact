"""Integration tests for Dataset.update_batch with the real, stateful
observers. Bookkeeping strategies (QUIRE index bookkeeping, ALBL bandit
bookkeeping with preconditions) still observe the per-entry
(entry_id, label) stream through the default update_batch replay, while
model-retraining strategies (QueryByCommittee, BALD, InformationDensity,
EpsilonUncertaintySampling) retrain exactly once per batch instead of
once per entry.
"""
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from libact.base.dataset import Dataset
from libact.models import SklearnProbaAdapter
from libact.query_strategies import (
    ActiveLearningByLearning,
    BALD,
    EpsilonUncertaintySampling,
    InformationDensity,
    QueryByCommittee,
    QUIRE,
    UncertaintySampling,
)


def _make_dataset():
    np.random.seed(1126)
    X = np.random.randn(30, 5)
    y = np.random.choice([0, 1], size=30)
    return Dataset(X, list(y[:10]) + [None] * 20), y


class TestUpdateBatchWithQuire(unittest.TestCase):

    def test_end_state_matches_sequential(self):
        ds_batch, y = _make_dataset()
        ds_seq, _ = _make_dataset()
        quire_batch = QUIRE(ds_batch)
        quire_seq = QUIRE(ds_seq)

        ids = [12, 15, 20]
        labels = [int(y[i]) for i in ids]

        ds_batch.update_batch(ids, labels)
        for i, label in zip(ids, labels):
            ds_seq.update(i, label)

        self.assertEqual(quire_batch.Uindex, quire_seq.Uindex)
        self.assertEqual(quire_batch.Lindex, quire_seq.Lindex)
        self.assertEqual(list(quire_batch.y), list(quire_seq.y))
        # The newly labeled ids moved from Uindex to Lindex.
        for i in ids:
            self.assertIn(i, quire_batch.Lindex)
            self.assertNotIn(i, quire_batch.Uindex)


def _spy(obj, method_name, calls):
    """Wrap obj.method_name so each call appends to calls."""
    original = getattr(obj, method_name)

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    setattr(obj, method_name, wrapper)


class TestUpdateBatchWithQueryByCommittee(unittest.TestCase):

    def _make_qbc(self, ds):
        return QueryByCommittee(
            ds,
            models=[
                SklearnProbaAdapter(
                    LogisticRegression(C=c, max_iter=200, solver='liblinear')
                )
                for c in [0.1, 1.0]
            ],
            random_state=0,
        )

    def test_committee_retrains_once_per_batch(self):
        ds, y = _make_dataset()
        np.random.seed(0)
        qbc = self._make_qbc(ds)

        teach_calls = []
        _spy(qbc, 'teach_students', teach_calls)

        ids = [11, 14, 17]
        ds.update_batch(ids, [int(y[i]) for i in ids])
        # One retrain for the whole batch, after all labels are applied —
        # not one per entry (intermediate committees are never queried).
        # A per-entry notification leaking through would also call
        # teach_students via the update hook and fail this count.
        self.assertEqual(len(teach_calls), 1)
        self.assertEqual(ds.len_labeled(), 13)

    def test_sequential_updates_still_retrain_per_entry(self):
        ds, y = _make_dataset()
        np.random.seed(0)
        qbc = self._make_qbc(ds)

        calls = []
        _spy(qbc, 'teach_students', calls)

        for i in [11, 14, 17]:
            ds.update(i, int(y[i]))
        self.assertEqual(len(calls), 3)


class TestUpdateBatchWithBALD(unittest.TestCase):

    def test_ensemble_retrains_once_per_batch(self):
        ds, y = _make_dataset()
        qs = BALD(
            ds,
            models=[
                SklearnProbaAdapter(
                    LogisticRegression(C=c, max_iter=200, solver='liblinear')
                )
                for c in [0.1, 1.0]
            ],
            random_state=0,
        )

        calls = []
        _spy(qs, '_train_ensemble', calls)

        ids = [11, 14, 17]
        ds.update_batch(ids, [int(y[i]) for i in ids])
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(qs.make_query(), (int, np.integer))


class TestUpdateBatchWithInformationDensity(unittest.TestCase):

    def test_model_retrains_once_per_batch(self):
        ds, y = _make_dataset()
        qs = InformationDensity(
            ds,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=0,
        )

        calls = []
        _spy(qs.model, 'train', calls)

        ids = [13, 18]
        ds.update_batch(ids, [int(y[i]) for i in ids])
        self.assertEqual(len(calls), 1)


class TestUpdateBatchWithEpsilonUS(unittest.TestCase):

    def test_model_retrains_once_per_batch(self):
        ds, y = _make_dataset()
        qs = EpsilonUncertaintySampling(
            ds,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            epsilon=0.1,
            random_state=0,
        )

        calls = []
        _spy(qs.model, 'train', calls)

        ids = [13, 18]
        ds.update_batch(ids, [int(y[i]) for i in ids])
        self.assertEqual(len(calls), 1)
        # The strategy stays usable after the bulk update.
        self.assertIsInstance(qs.make_query(), (int, np.integer))


class TestUpdateBatchWithALBL(unittest.TestCase):

    def _make_albl(self):
        ds, y = _make_dataset()
        albl = ActiveLearningByLearning(
            ds,
            query_strategies=[
                UncertaintySampling(
                    ds,
                    model=SklearnProbaAdapter(
                        LogisticRegression(
                            C=1., max_iter=200, solver='liblinear')
                    )
                ),
            ],
            T=10,
            model=SklearnProbaAdapter(
                LogisticRegression(max_iter=200, solver='liblinear')
            ),
            random_state=42,
        )
        return ds, y, albl

    def test_queried_id_through_update_batch(self):
        # The supported flow: ids obtained from ALBL's own make_query may
        # be applied through update_batch, and ALBL's bookkeeping matches
        # the sequential path.
        ds_b, y_b, albl_b = self._make_albl()
        ds_s, y_s, albl_s = self._make_albl()

        ask_b = albl_b.make_query()
        ask_s = albl_s.make_query()
        # Identical fixtures and seeds: both instances pick the same id.
        self.assertEqual(ask_b, ask_s)

        ds_b.update_batch([ask_b], [int(y_b[ask_b])])
        ds_s.update(ask_s, int(y_s[ask_s]))

        self.assertEqual(albl_b.queried_hist_, albl_s.queried_hist_)
        self.assertEqual(albl_b.W, albl_s.W)

    def test_update_before_make_query_fails_like_sequential(self):
        # ALBL assumes each update corresponds to an entry it has itself
        # queried via make_query() — before any query its query
        # distribution is uninitialized and update() raises TypeError.
        # ALBL does NOT otherwise guard against unqueried ids (see the
        # parity test below); what update_batch must guarantee is that it
        # fails exactly like the sequential update() path does.
        ds_b, y_b, _albl_b = self._make_albl()
        ds_s, y_s, _albl_s = self._make_albl()
        with self.assertRaises(TypeError):
            ds_b.update_batch([15], [int(y_b[15])])
        with self.assertRaises(TypeError):
            ds_s.update(15, int(y_s[15]))

    def test_arbitrary_id_parity_with_sequential(self):
        # Pre-existing ALBL behavior: after make_query(), updating an id
        # ALBL did NOT query is accepted silently (the bandit bookkeeping
        # uses whatever query distribution is current). That behavior is
        # out of scope to change here — this test pins that update_batch
        # reproduces the sequential path exactly, corrupt bookkeeping
        # included.
        ds_b, y_b, albl_b = self._make_albl()
        ds_s, y_s, albl_s = self._make_albl()

        ask_b = albl_b.make_query()
        ask_s = albl_s.make_query()
        self.assertEqual(ask_b, ask_s)

        # Pick an unlabeled id that ALBL did not query.
        unlabeled_ids = ds_b.get_unlabeled_entries()[0].tolist()
        other = next(i for i in unlabeled_ids if i != ask_b)

        ds_b.update_batch([other], [int(y_b[other])])
        ds_s.update(other, int(y_s[other]))

        self.assertEqual(albl_b.queried_hist_, albl_s.queried_hist_)
        self.assertEqual(albl_b.W, albl_s.W)


if __name__ == '__main__':
    unittest.main()
