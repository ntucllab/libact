"""Density Weighted
"""
from __future__ import division

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip


class DensityWeightedMeta(QueryStrategy):
    """Density Weighted Meta Algorithm

    :math:`\phi_A` represents the output of some base query strategy :math:`A`.

    The instance to query is given as follows:

    :math:`argmax_{\mathbf{x}} \phi_A(\mathbf{x}) \times (\frac{1}{U} \Sigma^{U}_{u=1} sim(\mathbf{x}, \mathbf{x}^{(u)}))^{\beta}`

    The :math:`sim` function in this implementation is by first ran a
    clustering algorithm. The clustering algorithm will output :math:`U`
    centers. Then we use the similarity metric to calculate the average
    similarity of the given instance :math:`\mathbf{X}` to each cluster
    center.

    Parameters
    ----------
    base_query_strategy:
        The query_strategy has to support _get_score() method.

    similarity_metric: sklearn.metrics.pairwise class instance, optional (default=cosine_similarity)
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html


    clustering_method: sklearn.cluster class instance, optional (default=Kmeans())
        should support method fit and transform and attribute cluster_centers_.
        (reference: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

    beta : float
        Scaling factor for the similarity term.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------

    Examples
    --------
    Here is an example of how to declare a DensityWeightedMeta query_strategy object:

    .. code-block:: python

       from libact.query_strategies import DensityWeightedMeta
       from libact.models import LogisticRegression

       qs = DensityWeightedMeta(dataset)

    References
    ----------
    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """

    def __init__(self, dataset, base_query_strategy, similarity_metric=None,
                 clustering_method=None, beta=1.0, random_state=None):
        super(DensityWeightedMeta, self).__init__(dataset=dataset)
        if not isinstance(base_query_strategy, QueryStrategy):
            raise TypeError(
                "'base_query_strategy' has to be an instance of 'QueryStrategy'"
            )
        if base_query_strategy.dataset != self.dataset:
            raise ValueError("base_query_strategy should share the same"
                             "dataset instance with DensityWeightedMeta")

        self.base_query_strategy = base_query_strategy
        self.beta = beta
        self.random_state_ = seed_random_state(random_state)

        if clustering_method is not None:
            self.clustering_method = clustering_method
        else:
            self.clustering_method = KMeans(
                n_clusters=5, random_state=self.random_state_)
        
        if similarity_metric is not None:
            self.similarity_metric = similarity_metric
        else:
            self.similarity_metric = cosine_similarity


    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        pass

    @inherit_docstring_from(QueryStrategy)
    def _get_scores(self):
        dataset = self.dataset
        X, _ = zip(*dataset.data)
        scores = self.base_query_strategy._get_scores()
        _, X_pool = dataset.get_unlabeled_entries()
        unlabeled_entry_ids, base_scores = zip(*scores)
        
        self.clustering_method.fit(X)
        pool_cluster = self.clustering_method.predict(X_pool)
        cluster_center = self.clustering_method.cluster_centers_
        similarity = []
        for i in range(len(X_pool)):
            similarity.append(
                self.similarity_metric(
                    X_pool[i].reshape(1, -1),
                    cluster_center[pool_cluster[i]].reshape(1, -1)
                )[0][0]
            )
        similarity = np.asarray(similarity)

        scores = base_scores * similarity**self.beta
        return zip(unlabeled_entry_ids, scores)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset

        unlabeled_entry_ids, scores = zip(*self._get_scores())
        ask_id = self.random_state_.choice(np.where(scores == np.max(scores))[0])

        return unlabeled_entry_ids[ask_id]