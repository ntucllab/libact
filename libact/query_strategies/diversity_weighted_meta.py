"""Diversity Weighted Meta-Strategy

This module implements a meta query strategy that turns any score-based
base strategy into a diversity-aware batch strategy: batches balance the
base strategy's own preference with within-batch diversity, so a batch is
not just the top-k points with near-duplicate redundancy.
"""
import numbers

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state


def _minmax_normalize(values):
    """Monotone (direction-preserving) min-max rescale to [0, 1].

    Constant input maps to all 0.5 (neutral). Non-finite values are
    clipped into the finite range beforehand; if no finite value exists,
    everything maps to 0.5. The transform never inverts or re-signs
    values, so the input's ranking is preserved exactly.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 0.5)
    lo = values[finite].min()
    hi = values[finite].max()
    if np.isclose(hi, lo):
        normalized = np.full(len(values), 0.5)
        # keep -inf placeholders at the bottom of the scale
        normalized[values == -np.inf] = 0.0
        return normalized
    clipped = np.clip(values, lo, hi)
    clipped = np.nan_to_num(clipped, nan=lo)
    return (clipped - lo) / (hi - lo)


class DiversityWeightedMeta(QueryStrategy):

    r"""Diversity Weighted Meta-Strategy

    A meta algorithm that makes the batch queries of any score-based base
    strategy diversity-aware. A plain top-k over acquisition scores tends
    to select clusters of near-duplicate points that carry redundant
    information; this wrapper greedily builds the batch so that each pick
    balances the base strategy's preference against the distance to the
    points already selected:

    .. math::

        \mathbf{x}_{next} = \operatorname{argmax}_{\mathbf{x}}
            (1 - \lambda) \cdot s(\mathbf{x}) +
            \lambda \cdot d(\mathbf{x})

    where :math:`s(\mathbf{x})` is the min-max normalized acquisition
    score from the base strategy's ``_get_scores()`` and
    :math:`d(\mathbf{x})` is the min-max normalized distance from
    :math:`\mathbf{x}` to the nearest already-selected batch member. The
    first pick is always the base strategy's argmax, so a batch of one
    matches sequential behavior.

    The score normalization is monotone and direction-preserving: the
    wrapper follows the base strategy's own preference ranking (higher
    score = preferred, i.e. what ``make_query()`` would pick), whatever
    the semantics of the raw scores are. In particular, strategies whose
    scores are not uncertainty-flavored (e.g. HintSVM's absolute decision
    values) are handled correctly by construction — scores are never
    re-interpreted, re-signed, or inverted.

    Parameters
    ----------
    dataset : Dataset object
        The dataset to query from. Must be the same instance the base
        strategy operates on.

    base_query_strategy : :py:mod:`libact.query_strategies` object instance
        The base strategy whose scores are diversified. Has to support the
        ``_get_scores()`` method and share this dataset instance.

    lmbda : float, optional (default=0.5)
        Trade-off between the base score and diversity, in [0, 1].
        ``lmbda=0`` reproduces the plain top-k ranking of the base scores;
        ``lmbda=1`` performs a pure farthest-point traversal after the
        first pick.

    metric : str, optional (default='euclidean')
        Distance metric passed to ``sklearn.metrics.pairwise_distances``
        (handles both dense and sparse feature matrices).

    transformer : object with transform method, optional (default=None)
        Optional feature transformer (e.g. encoder, embedding model). If
        provided, distances are computed in the transformed space. Must
        have a ``transform(X)`` method.

    candidate_pool_size : int, optional (default=None)
        Performance cap: if set, the greedy selection is restricted to the
        ``candidate_pool_size`` highest-scoring candidates (at least
        ``batch_size``). ``None`` uses the full unlabeled pool (exact).
        Setting it trades diversity coverage for speed on very large
        pools.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        Random state for tie-breaking reproducibility.

    Attributes
    ----------
    base_query_strategy : QueryStrategy
        The wrapped base strategy.

    random_state\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of how to use DiversityWeightedMeta to query a
    diverse batch from an uncertainty sampling base:

    .. code-block:: python

       from libact.query_strategies import (
           DiversityWeightedMeta, UncertaintySampling)
       from libact.models import LogisticRegression

       qs = DiversityWeightedMeta(
           dataset,
           base_query_strategy=UncertaintySampling(
               dataset, model=LogisticRegression()),
           lmbda=0.5,
       )
       ask_ids = qs.make_query_batch(10)

    References
    ----------
    .. [1] Brinker, Klaus. "Incorporating diversity in active learning
           with support vector machines." ICML 2003.

    .. [2] Sener, Ozan, and Silvio Savarese. "Active learning for
           convolutional neural networks: A core-set approach." ICLR 2018.
    """

    def __init__(self, dataset, base_query_strategy, lmbda=0.5,
                 metric='euclidean', transformer=None,
                 candidate_pool_size=None, random_state=None):
        super(DiversityWeightedMeta, self).__init__(dataset=dataset)
        if not isinstance(base_query_strategy, QueryStrategy):
            raise TypeError(
                "'base_query_strategy' has to be an instance of "
                "'QueryStrategy'"
            )
        if base_query_strategy.dataset != self.dataset:
            raise ValueError("base_query_strategy should share the same"
                             "dataset instance with DiversityWeightedMeta")
        if not 0 <= lmbda <= 1:
            raise ValueError("lmbda must be in [0, 1], got %r" % (lmbda,))
        if transformer is not None and not hasattr(transformer, 'transform'):
            raise TypeError("transformer must have a 'transform' method")
        if candidate_pool_size is not None:
            if isinstance(candidate_pool_size, bool) or \
                    not isinstance(candidate_pool_size, numbers.Integral):
                raise TypeError(
                    "candidate_pool_size must be an integer or None, got %r"
                    % (candidate_pool_size,))
            if candidate_pool_size < 1:
                raise ValueError(
                    "candidate_pool_size must be at least 1, got %d"
                    % candidate_pool_size)

        self.base_query_strategy = base_query_strategy
        self.lmbda = lmbda
        self.metric = metric
        self.transformer = transformer
        self.candidate_pool_size = candidate_pool_size
        self.random_state_ = seed_random_state(random_state)

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        # Stateless wrapper; the base strategy receives its own
        # notification through the dataset's callback registry.
        pass

    def _get_scores(self):
        """Delegate to the base strategy (rank-faithful passthrough).

        Returns
        -------
        entry_ids : np.ndarray, shape (n_unlabeled,)
            Global entry IDs of unlabeled samples.
        scores : np.ndarray, shape (n_unlabeled,)
            The base strategy's acquisition scores, untouched.
        """
        return self.base_query_strategy._get_scores()

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        entry_ids, scores = self._get_scores()

        if len(entry_ids) == 0:
            raise ValueError("No unlabeled samples available")

        candidates = np.where(np.isclose(scores, np.max(scores)))[0]
        return entry_ids[self.random_state_.choice(candidates)]

    def make_query_batch(self, batch_size):
        """Select a batch balancing base-strategy preference and diversity.

        Greedy selection: the first pick is the base strategy's argmax
        (ties randomized); each subsequent pick maximizes
        ``(1 - lmbda) * s + lmbda * d`` where ``s`` is the min-max
        normalized base score and ``d`` the min-max normalized distance to
        the nearest already-selected batch member.

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
        entry_ids, scores = self._get_scores()
        entry_ids = np.asarray(entry_ids)
        scores = np.asarray(scores, dtype=float)
        self._check_batch_size(batch_size, len(entry_ids))

        # Optional performance cap: restrict the greedy selection to the
        # highest-scoring candidates.
        if self.candidate_pool_size is not None and \
                self.candidate_pool_size < len(entry_ids):
            n_candidates = max(self.candidate_pool_size, batch_size)
            keep = np.argsort(-scores, kind='stable')[:n_candidates]
            entry_ids = entry_ids[keep]
            scores = scores[keep]

        n_candidates = len(entry_ids)
        if batch_size == n_candidates:
            # Every candidate is selected; diversity cannot exclude anyone.
            return entry_ids[np.argsort(-scores, kind='stable')]

        X, _ = self.dataset.get_entries()
        X_candidates = X[entry_ids]
        if self.transformer is not None:
            X_candidates = self.transformer.transform(X_candidates)
            if isinstance(X_candidates, (list, tuple)):
                X_candidates = np.asarray(X_candidates)

        s_norm = _minmax_normalize(scores)

        # First pick: the base strategy's argmax, ties randomized
        # (matches make_query).
        candidates = np.where(np.isclose(scores, np.max(scores)))[0]
        first = self.random_state_.choice(candidates)
        selected = [first]
        remaining = np.ones(n_candidates, dtype=bool)
        remaining[first] = False
        min_distances = self._distances_to(X_candidates, first)

        for _ in range(batch_size - 1):
            d_norm = _minmax_normalize(
                np.where(remaining, min_distances, -np.inf))
            utility = (1 - self.lmbda) * s_norm + self.lmbda * d_norm
            utility[~remaining] = -np.inf
            candidates = np.where(
                remaining & np.isclose(utility, np.max(utility)))[0]
            pick = self.random_state_.choice(candidates)
            selected.append(pick)
            remaining[pick] = False
            min_distances = np.minimum(
                min_distances, self._distances_to(X_candidates, pick))

        return entry_ids[np.asarray(selected)]

    def _distances_to(self, X, idx):
        """Distances from every candidate row of X to row idx."""
        distances = pairwise_distances(
            X, X[[idx]], metric=self.metric).ravel()
        # NaN can occur for degenerate inputs (e.g. cosine with a zero
        # vector); treat it as zero distance so such points are never
        # rewarded for being "far".
        return np.nan_to_num(distances, nan=0.0,
                             posinf=np.inf, neginf=-np.inf)
