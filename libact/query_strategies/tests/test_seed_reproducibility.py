"""Seed reproducibility tests.

Two full active-learning runs configured with the same random_state must
produce an identical query sequence. Each run rebuilds the Dataset, the query
strategy and its models from scratch, so any randomness not controlled by the
passed random_state (e.g. use of the global numpy RNG) makes the sequences
diverge between the two runs.
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.models import LogisticRegression
from libact.query_strategies import (
    ActiveLearningByLearning,
    BALD,
    CoreSet,
    DensityWeightedMeta,
    DWUS,
    EpsilonUncertaintySampling,
    InformationDensity,
    QueryByCommittee,
    QUIRE,
    RandomSampling,
    UncertaintySampling,
)
from libact.query_strategies.multiclass import EER, HierarchicalSampling
from .utils import run_qs

try:
    from libact.query_strategies import HintSVM
except ImportError:  # HintSVM C-extension not compiled
    HintSVM = None

SEED = 1126
N_INIT = 6
QUOTA = 5


def make_interleaved_blobs(pos_label=1, neg_label=0, n_per_class=20):
    """Deterministically generate two interleaved gaussian blobs."""
    rs = np.random.RandomState(0)
    X = np.empty((2 * n_per_class, 2))
    X[0::2] = rs.randn(n_per_class, 2) + [2., 2.]
    X[1::2] = rs.randn(n_per_class, 2) + [-2., -2.]
    y = np.empty(2 * n_per_class, dtype=int)
    y[0::2] = pos_label
    y[1::2] = neg_label
    return X, y


def init_dataset(X, y, n_labeled=N_INIT):
    """Dataset with the first n_labeled entries labeled, the rest unlabeled."""
    return Dataset(X, np.concatenate(
        [y[:n_labeled], [None] * (len(y) - n_labeled)]))


class SeedReproducibilityTestCase(unittest.TestCase):

    def setUp(self):
        self.X, self.y = make_interleaved_blobs()

    def run_once(self, qs_factory, X=None, y=None, quota=QUOTA):
        X = self.X if X is None else X
        y = self.y if y is None else y
        trn_ds = init_dataset(X, y)
        qs = qs_factory(trn_ds)
        return run_qs(trn_ds, qs, y, quota)

    def assert_reproducible(self, qs_factory, X=None, y=None, quota=QUOTA):
        qseq1 = self.run_once(qs_factory, X=X, y=y, quota=quota)
        qseq2 = self.run_once(qs_factory, X=X, y=y, quota=quota)
        assert_array_equal(qseq1, qseq2)

    def test_random_sampling(self):
        self.assert_reproducible(
            lambda ds: RandomSampling(ds, random_state=SEED))

    def test_random_sampling_different_seeds_differ(self):
        qseq1 = self.run_once(
            lambda ds: RandomSampling(ds, random_state=SEED))
        qseq2 = self.run_once(
            lambda ds: RandomSampling(ds, random_state=9527))
        self.assertFalse(np.array_equal(qseq1, qseq2))

    def test_uncertainty_sampling(self):
        self.assert_reproducible(
            lambda ds: UncertaintySampling(ds, model=LogisticRegression()))

    def test_query_by_committee_vote(self):
        self.assert_reproducible(
            lambda ds: QueryByCommittee(
                ds,
                models=[LogisticRegression(C=1.0),
                        LogisticRegression(C=0.01)],
                random_state=SEED))

    def test_query_by_committee_kl_divergence(self):
        self.assert_reproducible(
            lambda ds: QueryByCommittee(
                ds,
                disagreement='kl_divergence',
                models=[LogisticRegression(C=1.0),
                        LogisticRegression(C=0.01)],
                random_state=SEED))

    def test_bald(self):
        self.assert_reproducible(
            lambda ds: BALD(
                ds,
                models=[LogisticRegression(C=0.1),
                        LogisticRegression(C=1.0),
                        LogisticRegression(C=10.0)],
                random_state=SEED))

    def test_bald_base_model_stochastic(self):
        # The base_model + n_models path manufactures the committee via
        # clone(); the BALD seed must control the (stochastic) clones too,
        # even though the base estimator itself is left unseeded.
        from sklearn.ensemble import RandomForestClassifier
        from libact.models import SklearnProbaAdapter
        self.assert_reproducible(
            lambda ds: BALD(
                ds,
                base_model=SklearnProbaAdapter(
                    RandomForestClassifier(n_estimators=8)),
                n_models=5,
                random_state=SEED))

    def test_coreset(self):
        self.assert_reproducible(lambda ds: CoreSet(ds, random_state=SEED))

    def test_epsilon_uncertainty_sampling(self):
        self.assert_reproducible(
            lambda ds: EpsilonUncertaintySampling(
                ds, model=LogisticRegression(), epsilon=0.5,
                random_state=SEED))

    def test_information_density(self):
        self.assert_reproducible(
            lambda ds: InformationDensity(
                ds, model=LogisticRegression(), random_state=SEED))

    def test_dwus(self):
        self.assert_reproducible(lambda ds: DWUS(ds, random_state=SEED))

    def test_density_weighted_meta(self):
        self.assert_reproducible(
            lambda ds: DensityWeightedMeta(
                ds,
                base_query_strategy=UncertaintySampling(
                    ds, model=LogisticRegression()),
                random_state=SEED))

    def test_active_learning_by_learning(self):
        self.assert_reproducible(
            lambda ds: ActiveLearningByLearning(
                ds,
                T=QUOTA,
                query_strategies=[
                    UncertaintySampling(ds, model=LogisticRegression(C=1.0)),
                    RandomSampling(ds, random_state=SEED),
                ],
                model=LogisticRegression(),
                random_state=SEED))

    def test_hierarchical_sampling(self):
        self.assert_reproducible(
            lambda ds: HierarchicalSampling(ds, [0, 1], random_state=SEED))

    def test_eer(self):
        self.assert_reproducible(
            lambda ds: EER(ds, model=LogisticRegression(), random_state=SEED),
            quota=3)

    def test_quire(self):
        self.assert_reproducible(lambda ds: QUIRE(ds))

    @unittest.skipIf(HintSVM is None, "HintSVM C-extension not compiled")
    def test_hintsvm(self):
        # HintSVM labels the hint samples 0 internally, so use -1/+1 labels.
        X, y = make_interleaved_blobs(pos_label=1, neg_label=-1)
        self.assert_reproducible(
            lambda ds: HintSVM(ds, random_state=SEED), X=X, y=y)


class SeedReproducibilityWithLabelerTestCase(unittest.TestCase):
    """Reproducibility of the canonical run loop with an IdealLabeler in it.

    The pool is small enough that every entry gets queried, including
    duplicated feature rows carrying conflicting labels, which exercises the
    labeler's random tie-breaking.
    """

    def setUp(self):
        X, y = make_interleaved_blobs(n_per_class=5)
        # Duplicate feature rows with conflicting labels in the unlabeled pool.
        X[7] = X[6]
        y[6], y[7] = 0, 1
        self.X, self.y = X, y
        self.n_init = 4
        self.quota = len(y) - self.n_init

    def run_loop(self):
        full_ds = Dataset(self.X, self.y)
        lbr = IdealLabeler(full_ds, random_state=SEED)
        trn_ds = init_dataset(self.X, self.y, n_labeled=self.n_init)
        qs = UncertaintySampling(trn_ds, model=LogisticRegression())
        qseq, labels = [], []
        for _ in range(self.quota):
            ask_id = qs.make_query()
            lb = lbr.label(trn_ds.data[ask_id][0])
            trn_ds.update(ask_id, lb)
            qseq.append(ask_id)
            labels.append(lb)
        return qseq, labels

    def test_full_run_with_labeler(self):
        qseq1, labels1 = self.run_loop()
        qseq2, labels2 = self.run_loop()
        assert_array_equal(qseq1, qseq2)
        assert_array_equal(labels1, labels2)


if __name__ == '__main__':
    unittest.main()
