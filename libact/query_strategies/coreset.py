"""Core-Set (k-Center Greedy) Query Strategy

This module implements the Core-Set approach for active learning, which selects
the unlabeled point farthest from all labeled points (greedy k-Center).
"""
import numpy as np
from scipy.spatial.distance import cdist

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
        Distance metric passed to ``scipy.spatial.distance.cdist``.
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

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.asarray(X_pool)

        if len(unlabeled_entry_ids) == 0:
            raise ValueError("No unlabeled samples available")

        # Get labeled data
        labeled_entries = dataset.get_labeled_entries()
        X_labeled = np.asarray(labeled_entries[0])

        # Fallback to random if no labeled data
        if len(X_labeled) == 0:
            idx = self.random_state_.randint(0, len(unlabeled_entry_ids))
            return unlabeled_entry_ids[idx]

        # Transform features if transformer is provided
        if self.transformer is not None:
            X_pool_t = np.asarray(self.transformer.transform(X_pool))
            X_labeled_t = np.asarray(self.transformer.transform(X_labeled))
        else:
            X_pool_t = X_pool
            X_labeled_t = X_labeled

        # Compute pairwise distances: (n_unlabeled, n_labeled)
        dist_matrix = cdist(X_pool_t, X_labeled_t, metric=self.metric)

        # For each unlabeled point, find minimum distance to any labeled point
        min_distances = np.min(dist_matrix, axis=1)

        # Select the unlabeled point with maximum min-distance (farthest)
        max_dist = np.max(min_distances)
        candidates = np.where(np.isclose(min_distances, max_dist))[0]
        selected_idx = self.random_state_.choice(candidates)

        return unlabeled_entry_ids[selected_idx]

    def _get_scores(self):
        """Return min-distances to labeled set for all unlabeled samples.

        Returns
        -------
        scores : list of (entry_id, score) tuples
            Each score is the minimum distance from that unlabeled point
            to any labeled point. Higher score means more informative.
        """
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.asarray(X_pool)

        if len(unlabeled_entry_ids) == 0:
            return []

        labeled_entries = dataset.get_labeled_entries()
        X_labeled = np.asarray(labeled_entries[0])

        if len(X_labeled) == 0:
            return list(zip(unlabeled_entry_ids,
                            [float('inf')] * len(unlabeled_entry_ids)))

        if self.transformer is not None:
            X_pool_t = np.asarray(self.transformer.transform(X_pool))
            X_labeled_t = np.asarray(self.transformer.transform(X_labeled))
        else:
            X_pool_t = X_pool
            X_labeled_t = X_labeled

        dist_matrix = cdist(X_pool_t, X_labeled_t, metric=self.metric)
        min_distances = np.min(dist_matrix, axis=1)

        return list(zip(unlabeled_entry_ids, min_distances))
