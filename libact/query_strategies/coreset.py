"""Core-Set (k-Center Greedy) Query Strategy

This module implements the Core-Set approach for active learning, which selects
the unlabeled point farthest from all labeled points (greedy k-Center).
"""
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state


class CoreSet(QueryStrategy):
    """Core-Set (k-Center Greedy) Query Strategy

    This strategy selects samples that maximize the minimum distance to any
    already-labeled point. It greedily builds a coreset by always picking the
    unlabeled point farthest from the current labeled set, ensuring geometric
    coverage of the feature space.

    Parameters
    ----------
    dataset : Dataset object
        The dataset to query from.

    metric : str, optional (default='euclidean')
        Distance metric passed to ``sklearn.metrics.pairwise_distances``.
        Common options: 'euclidean', 'cosine', 'cityblock', 'minkowski'.

    transformer : object with transform method, optional (default=None)
        Optional feature transformer (e.g., encoder, embedding model).
        If provided, distances are computed in the transformed space.
        Must have a ``transform(X)`` method.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        Random state for tie-breaking reproducibility.

    Attributes
    ----------
    metric : str
        The distance metric used.

    transformer : object or None
        The feature transformer if provided.

    random_state_ : np.random.RandomState instance
        The random number generator.

    Examples
    --------
    .. code-block:: python

       from libact.query_strategies import CoreSet

       # Basic usage with Euclidean distance
       qs = CoreSet(dataset)

       # With cosine distance
       qs = CoreSet(dataset, metric='cosine')

       # With a feature transformer
       qs = CoreSet(dataset, transformer=my_encoder)

    References
    ----------
    .. [1] Sener, Ozan, and Silvio Savarese. "Active learning for convolutional
           neural networks: A core-set approach." ICLR 2018.
    """

    def __init__(self, dataset, **kwargs):
        super(CoreSet, self).__init__(dataset, **kwargs)

        self.metric = kwargs.pop('metric', 'euclidean')

        self.transformer = kwargs.pop('transformer', None)
        if self.transformer is not None and not hasattr(self.transformer, 'transform'):
            raise TypeError("transformer must have a 'transform' method")

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

    def _transform(self, X):
        """Apply the optional feature transformer."""
        if self.transformer is not None:
            X = self.transformer.transform(X)
            if isinstance(X, (list, tuple)):
                X = np.asarray(X)
        return X

    def _get_scores(self):
        """Return min-distances to labeled set for all unlabeled samples.

        Returns
        -------
        entry_ids : np.ndarray, shape (n_unlabeled,)
            Global entry IDs of unlabeled samples.
        scores : np.ndarray, shape (n_unlabeled,)
            Min-distance from each unlabeled point to any labeled point.
            Higher score means more informative.
        """
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        if len(unlabeled_entry_ids) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        X_labeled, _ = dataset.get_labeled_entries()

        if X_labeled.shape[0] == 0:
            return np.asarray(unlabeled_entry_ids), \
                np.full(len(unlabeled_entry_ids), float('inf'))

        X_pool_t = self._transform(X_pool)
        X_labeled_t = self._transform(X_labeled)

        # pairwise_distances handles both dense and sparse feature matrices
        dist_matrix = pairwise_distances(
            X_pool_t, X_labeled_t, metric=self.metric)
        min_distances = np.min(dist_matrix, axis=1)

        return np.asarray(unlabeled_entry_ids), min_distances

    def make_query_batch(self, batch_size):
        """Select a batch with the true greedy k-Center algorithm.

        Unlike the default top-k of :py:meth:`_get_scores` (which can
        return a cluster of mutually close points that are all far from
        the labeled set), this recomputes the min-distance after every
        pick: each selected point joins the covered set, so the next pick
        maximizes the distance to the union of the labeled set and the
        already-selected batch. This is the batch algorithm of Sener &
        Savarese (2018).

        Parameters
        ----------
        batch_size : int
            Number of samples to query. Must satisfy
            ``1 <= batch_size <= n_unlabeled``.

        Returns
        -------
        entry_ids : np.ndarray of int, shape (batch_size,)
            Distinct entry ids in selection order.
        """
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        n_unlabeled = len(unlabeled_entry_ids)
        self._check_batch_size(batch_size, n_unlabeled)

        X_labeled, _ = dataset.get_labeled_entries()
        X_pool_t = self._transform(X_pool)

        selected = []
        if X_labeled.shape[0] == 0:
            # No labeled data: seed the batch with a random pick, mirroring
            # make_query's random fallback, then proceed greedily.
            first = self.random_state_.randint(0, n_unlabeled)
            selected.append(first)
            min_distances = pairwise_distances(
                X_pool_t, X_pool_t[[first]], metric=self.metric).ravel()
        else:
            X_labeled_t = self._transform(X_labeled)
            min_distances = np.min(pairwise_distances(
                X_pool_t, X_labeled_t, metric=self.metric), axis=1)

        while len(selected) < batch_size:
            masked = min_distances.copy()
            masked[selected] = -np.inf
            candidates = np.where(np.isclose(masked, np.max(masked)))[0]
            pick = self.random_state_.choice(candidates)
            selected.append(pick)
            # The picked point now covers its neighborhood.
            new_distances = pairwise_distances(
                X_pool_t, X_pool_t[[pick]], metric=self.metric).ravel()
            min_distances = np.minimum(min_distances, new_distances)

        return np.asarray(unlabeled_entry_ids)[selected]

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        unlabeled_entry_ids, min_distances = self._get_scores()

        if len(unlabeled_entry_ids) == 0:
            raise ValueError("No unlabeled samples available")

        # Fallback to random if no labeled data (scores are all inf)
        if np.all(np.isinf(min_distances)):
            idx = self.random_state_.randint(0, len(unlabeled_entry_ids))
            return unlabeled_entry_ids[idx]

        # Select the unlabeled point with maximum min-distance (farthest)
        max_dist = np.max(min_distances)
        candidates = np.where(np.isclose(min_distances, max_dist))[0]
        selected_idx = self.random_state_.choice(candidates)

        return unlabeled_entry_ids[selected_idx]
