"""
"""
import copy

import numpy as np
from sklearn.metrics.pairwise import paired_distances
from scipy.spatial.distance import hamming
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from libact.base.dataset import Dataset
from ...base.interfaces import QueryStrategy, ContinuousModel
from ...utils import inherit_docstring_from, zip, seed_random_state


class CostSensitiveReferencePairEncoding(QueryStrategy):
    """Cost Sensitive Reference Pair Encoding (CSRPE)

    Parameters
    ----------
    scoring_fn : function
        scoring_fn(truth label, prediction label) returns a real number,
        the higher the better

    model : multilabel model

    base_clf :
        classifier support train (with sample_weight), predict methods and
        can be cloned by builtin copy.deepcopy function.

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are
        used. If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    Attributes
    ----------
    csrpe_ :
        Internal CSRPE classifier


    Examples
    --------
    Here is an example of how to declare a CostSensitiveReferencePairEncoding
    query_strategy object:

    .. code-block:: python

       from libact.query_strategies import CostSensitiveReferencePairEncoding
       from libact.models.multilabel import BinaryRelevance
       from libact.models import LogisticRegression
       from libact.utils.multilabel import pairwise_f1_score

       base_model = LogisticRegression(
               solver='liblinear', multi_class="ovr")
       model = BinaryRelevance(LogisticRegression(solver='liblinear',
                                                  multi_class="ovr"))
       qs = CostSensitiveReferencePairEncoding(
               dataset,
               scoring_fn=pairwise_f1_score,
               model=model,
               base_model=base_model,
               n_models=100,
               n_jobs=1)

    References
    ----------
    .. [1] Yang, Yao-Yuan, et al. "Cost-Sensitive Reference Pair Encoding for
           Multi-Label Learning." Pacific-Asia Conference on Knowledge Discovery
           and Data Mining. Springer, Cham, 2018.
    """

    def __init__(self, dataset, scoring_fn, model, base_model, n_models=100,
                 n_jobs=1, random_state=None):
        super(CostSensitiveReferencePairEncoding, self).__init__(dataset=dataset)

        self.model_ = model
        self.csrpe_ = CSRPE(scoring_fn=scoring_fn, base_clf=base_model,
                            n_clfs=n_models, n_jobs=n_jobs, random_state=random_state)

        self.random_state_ = seed_random_state(random_state)

    def make_query(self):
        dataset = self.dataset

        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.asarray(X_pool)

        self.csrpe_.train(dataset)
        self.model_.train(dataset)

        predY = self.model_.predict(X_pool)
        Z = self.csrpe_.predicted_code(X_pool)
        predZ = self.csrpe_.encode(predY)

        dist = paired_distances(Z, predZ, metric=hamming) # z1 z2
        dist2 = self.csrpe_.predict_dist(X_pool) # z1 zt
        #dist3 = self.csrpe.distance(predZ) # z2 zt

        dist = dist + dist2
        #dist = dist + dist3

        ask_id = self.random_state_.choice(
            np.where(np.isclose(dist, np.max(dist)))[0])

        return unlabeled_entry_ids[ask_id]


class BinaryCLF():
    def __init__(self, base_clf, scoring_fn, rep_label, random_state=None):
        self.base_clf = base_clf
        self.scoring_fn = scoring_fn
        self.random_state_ = seed_random_state(random_state)
        self.label = None
        self.rep_label = rep_label

    def enc(self, Y):
        if self.rep_label is None:
            raise ValueError
        score0 = self.scoring_fn(Y, np.tile(self.rep_label[0], (len(Y), 1)))
        score1 = self.scoring_fn(Y, np.tile(self.rep_label[1], (len(Y), 1)))
        lbl = (((score1 - score0) > 0) + 0.0)
        return lbl

    def train(self, X, y):
        self.n_samples = np.shape(X)[0]
        self.n_labels = np.shape(y)[1]

        score0 = self.scoring_fn(y, np.tile(self.rep_label[0], (self.n_samples, 1)))
        score1 = self.scoring_fn(y, np.tile(self.rep_label[1], (self.n_samples, 1)))
        lbl = (((score1 - score0) > 0) + 0.0)

        weight = np.abs(score1 - score0)
        if np.sum(weight) > 0:
            weight = weight / np.sum(weight) * len(X)

        if len(np.unique(lbl)) == 1:
            self.label = np.unique(lbl)[0]
            self.base_clf_ = None
        else:
            self.base_clf_ = copy.deepcopy(self.base_clf)
            self.base_clf_.train(Dataset(X, lbl), sample_weight=weight)

    def predict(self, X):
        if self.label is not None:
            return np.ones(len(X)) * self.label
        return self.base_clf_.predict(X)


class CSRPE():
    def __init__(self, scoring_fn, base_clf, n_clfs, n_jobs,
                 metric='euclidean', random_state=None):
        self.scoring_fn = scoring_fn
        self.base_clf = base_clf
        self.nn_ = NearestNeighbors(1, algorithm='ball_tree',
                metric=metric, n_jobs=n_jobs)
        self.n_clfs = n_clfs
        self.random_state_ = seed_random_state(random_state)

        self.n_labels = None
        self.clfs = None
        self.n_jobs = n_jobs

    def _build_clfs(self, Y):
        self.n_labels = np.shape(Y)[1]
        self.clfs = [BinaryCLF(self.base_clf, self.scoring_fn,
                     rep_label=self.random_state_.randint(0, 2, (2, self.n_labels)))
                     for i in range(self.n_clfs)]

    def encode(self, Y):
        Y = np.asarray(Y)
        if self.clfs is None:
            self._build_clfs(Y)
        if Y.shape[1] != self.n_labels:
            raise ValueError("The given label size does not match"
                             " number of labels. Expect %d but get %d"
                             % (self.n_labels, Y.shape[1]))
        encoded = np.zeros((Y.shape[0], self.n_clfs))
        for i, clf in enumerate(self.clfs):
            encoded[:, i] = clf.enc(Y)
        return encoded

    def predicted_code(self, X):
        if self.clfs is None:
            raise ValueError("CSRPE should be trained before calling"
                             "`predicted_code` method.")

        X = np.asarray(X)
        encoded = np.zeros((X.shape[0], self.n_clfs))
        for i, clf in enumerate(self.clfs):
            encoded[:, i] = clf.predict(X)
        return encoded

    def train(self, dataset):
        X, Y = dataset.format_sklearn()
        X, Y = np.asarray(X), np.asarray(Y)

        if self.clfs is None:
            self._build_clfs(Y)

        if Y.shape[1] != self.n_labels:
            raise ValueError("The given label size does not match "
                             " number of labels. Expect %d but get %d"
                             % (self.n_labels, Y.shape[1]))

        self.tokens = Y

        def train_single_clf_helper(clf, X, Y):
            clf.train(X, Y)
        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(train_single_clf_helper)(self.clfs[i], X, Y)
            for i in range(self.n_clfs)
        )
        self.nn_.fit(self.predicted_code(X))

    def predict(self, X):
        encoded = self.predicted_code(X)
        ind = self.nn_.kneighbors(encoded, 1, return_distance=False)
        ind = ind.reshape(-1)
        return self.tokens[ind]

    def predict_dist(self, X):
        encoded = self.predicted_code(X)
        dist, _ = self.nn_.kneighbors(encoded, 1, return_distance=True)
        dist = dist.reshape(-1)
        return dist