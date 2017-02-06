"""
Active Learning with Cost Embedding (ALCE)
"""
import copy

import numpy as np
from sklearn.neighbors import NearestNeighbors

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip
from .mdsp import MDSP


class ActiveLearningWithCostEmbedding(QueryStrategy):
    """Active Learning with Cost Embedding (ALCE)

    Cost sensitive multi-class algorithm.
    Assume each class has at least one sample in the labeled pool.

    Parameters
    ----------
    cost_matrix : array-like, shape=(n_classes, n_classes)
        The ith row, jth column represents the cost of the ground truth being
        ith class and prediction as jth class.

    mds_params : dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html

    nn_params : dict, optional
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

    embed_dim : int, optional (default: None)
        if is None, embed_dim = n_classes

    base_regressor : sklearn regressor

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    nn_ : sklearn.neighbors.NearestNeighbors object instance

    Examples
    --------

    References
    ----------
    .. [1] Kuan-Hao, and Hsuan-Tien Lin. "A Novel Uncertainty Sampling Algorithm
           for Cost-sensitive Multiclass Active Learning", In Proceedings of the
           IEEE International Conference on Data Mining (ICDM), 2016
    """

    def __init__(self,
                 dataset,
                 cost_matrix,
                 base_regressor,
                 embed_dim=None,
                 mds_params={},
                 nn_params={},
                 random_state=None):
        super(ActiveLearningWithCostEmbedding, self).__init__(dataset)

        self.cost_matrix = cost_matrix
        self.base_regressor = base_regressor

        self.n_classes = len(cost_matrix)
        if embed_dim is None:
            self.embed_dim = self.n_classes
        else:
            self.embed_dim = embed_dim
        self.regressors = [
            copy.deepcopy(self.base_regressor) for _ in range(self.embed_dim)
        ]

        self.random_state_ = seed_random_state(random_state)

        self.mds_params = {
            'metric': False,
            'n_components': self.embed_dim,
            'n_uq': self.n_classes,
            'max_iter': 300,
            'eps': 1e-6,
            'dissimilarity': "precomputed",
            'n_init': 8,
            'n_jobs': 1,
            'random_state': self.random_state_
        }
        self.mds_params.update(mds_params)

        self.nn_params = {}
        self.nn_params.update(nn_params)
        self.nn_ = NearestNeighbors(n_neighbors=1, **self.nn_params)

        dissimilarity = np.zeros((2 * self.n_classes, 2 * self.n_classes))
        dissimilarity[:self.n_classes, self.n_classes:] = self.cost_matrix
        dissimilarity[self.n_classes:, :self.n_classes] = self.cost_matrix.T
        mds_ = MDSP(**self.mds_params)
        embedding = mds_.fit(dissimilarity).embedding_

        self.class_embed = embedding[:self.n_classes, :]
        self.nn_.fit(embedding[self.n_classes:, :])


    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, pool_X = zip(*dataset.get_unlabeled_entries())
        # The input class should be 0-n_classes
        X, y = zip(*dataset.get_labeled_entries())

        pred_embed = np.zeros((len(pool_X), self.embed_dim))
        for i in range(self.embed_dim):
            self.regressors[i].fit(X, self.class_embed[y, i])
            pred_embed[:, i] = self.regressors[i].predict(pool_X)

        dist, _ = self.nn_.kneighbors(pred_embed)
        dist = dist[:, 0]

        ask_idx = self.random_state_.choice(
            np.where(np.isclose(dist, np.max(dist)))[0])
        return unlabeled_entry_ids[ask_idx]
