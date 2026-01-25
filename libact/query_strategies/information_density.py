"""Information Density Query Strategy

This module implements the Information Density approach for active learning,
which weights uncertainty scores by the average similarity to other unlabeled
instances, preferring samples that are both uncertain and representative.
"""
import numpy as np
from scipy.spatial.distance import cdist

from libact.base.interfaces import QueryStrategy, ProbabilisticModel, \
    ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip


class InformationDensity(QueryStrategy):
    r"""Information Density Query Strategy

    This strategy combines model uncertainty with instance density to avoid
    querying outliers. Each unlabeled sample is scored by its uncertainty
    weighted by its average similarity to all other unlabeled samples:

    .. math::

        \text{ID}(x) = \text{uncertainty}(x) \times
        \left( \frac{1}{|U|} \sum_{u \in U} \text{sim}(x, u) \right)^\beta

    Parameters
    ----------
    dataset : Dataset object
        The dataset to query from.

    model : :py:class:`libact.base.interfaces.ProbabilisticModel` or \
            :py:class:`libact.base.interfaces.ContinuousModel`
        The base model for uncertainty estimation.

    method : str, optional (default='entropy')
        Uncertainty measure to use:

        - ``'lc'``: least confident (1 - max probability)
        - ``'sm'``: smallest margin (difference between top two probabilities)
        - ``'entropy'``: predictive entropy

        ``'entropy'`` requires a ``ProbabilisticModel``.

    metric : str, optional (default='euclidean')
        Distance metric for density calculation, passed to
        ``scipy.spatial.distance.cdist``.

    beta : float, optional (default=1.0)
        Exponent controlling density influence.
        ``beta=0`` gives pure uncertainty; larger values increase
        the density weight.

    random_state : {int, np.random.RandomState instance, None}, optional
        Random state for tie-breaking reproducibility.

    Attributes
    ----------
    model : ProbabilisticModel or ContinuousModel
        The uncertainty model.

    method : str
        The uncertainty method.

    metric : str
        The distance metric for density.

    beta : float
        The density exponent.

    Examples
    --------
    .. code-block:: python

       from libact.query_strategies import InformationDensity

       # Basic usage with entropy uncertainty
       qs = InformationDensity(dataset, model=my_model)

       # With least-confident uncertainty and cosine similarity
       qs = InformationDensity(dataset, model=my_model, method='lc',
                               metric='cosine')

       # Strong density preference
       qs = InformationDensity(dataset, model=my_model, beta=2.0)

    References
    ----------
    .. [1] Settles, Burr, and Mark Craven. "An analysis of active learning
           strategies for sequence labeling tasks." EMNLP 2008.
           https://aclanthology.org/D08-1112.pdf
    """

    def __init__(self, *args, **kwargs):
        super(InformationDensity, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel) and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ContinuousModel or ProbabilisticModel"
            )

        self.method = kwargs.pop('method', 'entropy')
        if self.method not in ['lc', 'sm', 'entropy']:
            raise TypeError(
                "supported methods are ['lc', 'sm', 'entropy'], "
                "the given one is: " + self.method
            )
        if self.method == 'entropy' and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "method 'entropy' requires model to be a ProbabilisticModel"
            )

        self.metric = kwargs.pop('metric', 'euclidean')
        self.beta = kwargs.pop('beta', 1.0)

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        self.model.train(self.dataset)

    def _uncertainty_scores(self, X_pool):
        """Compute uncertainty scores for unlabeled samples.

        Parameters
        ----------
        X_pool : array-like, shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray, shape (n_samples,)
            Uncertainty scores (higher = more uncertain).
        """
        if isinstance(self.model, ProbabilisticModel):
            dvalue = np.asarray(self.model.predict_proba(X_pool))
        elif isinstance(self.model, ContinuousModel):
            dvalue = np.asarray(self.model.predict_real(X_pool))

        if self.method == 'lc':
            return 1.0 - np.max(dvalue, axis=1)
        elif self.method == 'sm':
            # Get top 2 values via partition
            if dvalue.shape[1] == 2:
                top2 = np.sort(dvalue, axis=1)[:, ::-1][:, :2]
            else:
                top2 = -(np.partition(-dvalue, 1, axis=1)[:, :2])
            return 1.0 - np.abs(top2[:, 0] - top2[:, 1])
        elif self.method == 'entropy':
            dvalue = np.clip(dvalue, 1e-10, 1.0)
            return np.sum(-dvalue * np.log(dvalue), axis=1)

    def _density_scores(self, X_pool):
        """Compute density scores for unlabeled samples.

        Density of each sample is the average similarity to all other
        unlabeled samples, where similarity = 1 / (1 + distance).

        Parameters
        ----------
        X_pool : array-like, shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray, shape (n_samples,)
            Density scores (higher = more representative).
        """
        n = len(X_pool)
        if n <= 1:
            return np.ones(n)

        dist_matrix = cdist(X_pool, X_pool, metric=self.metric)
        sim_matrix = 1.0 / (1.0 + dist_matrix)

        # Exclude self-similarity (diagonal) from average
        np.fill_diagonal(sim_matrix, 0.0)
        density = np.sum(sim_matrix, axis=1) / (n - 1)

        return density

    def _get_scores(self):
        """Return information density scores for all unlabeled samples.

        Returns
        -------
        scores : list of (entry_id, score) tuples
            Each score is uncertainty × density^beta. Higher = more informative.
        """
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.asarray(X_pool)

        if len(unlabeled_entry_ids) == 0:
            return []

        uncertainty = self._uncertainty_scores(X_pool)
        # Ensure non-negative uncertainty (ContinuousModel predict_real can
        # produce values > 1.0, causing 1-max or 1-margin to go negative).
        # The Settles formulation requires non-negative uncertainty for the
        # multiplicative combination with density to work correctly.
        uncertainty = np.maximum(uncertainty, 0.0)
        density = self._density_scores(X_pool)

        scores = uncertainty * (density ** self.beta)

        return list(zip(unlabeled_entry_ids, scores))

    @inherit_docstring_from(QueryStrategy)
    def make_query(self, return_score=False):
        dataset = self.dataset
        unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()

        if len(unlabeled_entry_ids) == 0:
            raise ValueError("No unlabeled samples available")

        scores = self._get_scores()
        entry_ids, score_values = zip(*scores)
        score_values = np.asarray(list(score_values))

        max_score = np.max(score_values)
        candidates = np.where(np.isclose(score_values, max_score))[0]
        selected_idx = self.random_state_.choice(candidates)

        if return_score:
            return entry_ids[selected_idx], scores
        else:
            return entry_ids[selected_idx]

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        self.model.train(self.dataset)
