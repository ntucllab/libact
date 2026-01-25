"""Epsilon Uncertainty Sampling (ε-US)

This module implements epsilon-greedy uncertainty sampling, which balances
exploration (random sampling) with exploitation (uncertainty sampling).
"""
import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel, \
    ProbabilisticModel
from libact.utils import inherit_docstring_from, seed_random_state, zip


class EpsilonUncertaintySampling(QueryStrategy):
    r"""Epsilon Uncertainty Sampling (ε-US)

    This strategy implements epsilon-greedy active learning, balancing
    exploration and exploitation:

    - With probability ε: select a random unlabeled sample (exploration)
    - With probability 1-ε: select by uncertainty sampling (exploitation)

    This simple approach provides a tunable exploration rate, which can be
    useful when the model's uncertainty estimates are unreliable early in
    training.

    Parameters
    ----------
    dataset : Dataset object
        The dataset to query from.

    model : :py:class:`libact.base.interfaces.ProbabilisticModel` or \
            :py:class:`libact.base.interfaces.ContinuousModel`
        The base model for uncertainty estimation.

    epsilon : float, optional (default=0.1)
        Probability of random exploration. Must be in [0, 1].
        - ``epsilon=0``: pure uncertainty sampling (no exploration)
        - ``epsilon=1``: pure random sampling (no exploitation)

    method : str, optional (default='lc')
        Uncertainty measure to use when exploiting:

        - ``'lc'``: least confident (1 - max probability)
        - ``'sm'``: smallest margin (difference between top two probabilities)
        - ``'entropy'``: predictive entropy

        ``'entropy'`` requires a ``ProbabilisticModel``.

    random_state : {int, np.random.RandomState instance, None}, optional
        Random state for reproducibility.

    Attributes
    ----------
    model : ProbabilisticModel or ContinuousModel
        The uncertainty model.

    epsilon : float
        The exploration probability.

    method : str
        The uncertainty method.

    Examples
    --------
    .. code-block:: python

       from libact.query_strategies import EpsilonUncertaintySampling
       from libact.models import LogisticRegression

       # 10% exploration, 90% uncertainty sampling
       qs = EpsilonUncertaintySampling(
           dataset,
           model=LogisticRegression(),
           epsilon=0.1
       )

       # Higher exploration early in training
       qs = EpsilonUncertaintySampling(
           dataset,
           model=LogisticRegression(),
           epsilon=0.3,
           method='entropy'
       )

    Notes
    -----
    When using with ALBL (Active Learning by Learning), note that ALBL's
    ``uniform_sampler=True`` parameter already adds random sampling as one
    of the bandit arms. In that context, ε-US may be redundant since ALBL
    learns the optimal exploration/exploitation balance adaptively.

    ε-US is most useful as:
    - A standalone strategy outside ALBL
    - A simple baseline for comparison
    - When you want a fixed (non-adaptive) exploration rate

    See Also
    --------
    UncertaintySampling : Pure uncertainty sampling without exploration.
    RandomSampling : Pure random sampling.
    """

    def __init__(self, *args, **kwargs):
        super(EpsilonUncertaintySampling, self).__init__(*args, **kwargs)

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

        self.epsilon = kwargs.pop('epsilon', 0.1)
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")

        self.method = kwargs.pop('method', 'lc')
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

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        self.model.train(self.dataset)

    def _get_uncertainty_scores(self, X_pool):
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
        else:
            # ContinuousModel
            dvalue = np.asarray(self.model.predict_real(X_pool))

        if self.method == 'lc':
            return 1.0 - np.max(dvalue, axis=1)
        elif self.method == 'sm':
            if dvalue.shape[1] == 2:
                top2 = np.sort(dvalue, axis=1)[:, ::-1][:, :2]
            else:
                top2 = -(np.partition(-dvalue, 1, axis=1)[:, :2])
            return 1.0 - np.abs(top2[:, 0] - top2[:, 1])
        else:  # entropy
            dvalue = np.clip(dvalue, 1e-10, 1.0)
            return np.sum(-dvalue * np.log(dvalue), axis=1)

    def _get_scores(self):
        """Return uncertainty scores for all unlabeled samples.

        Returns
        -------
        scores : list of (entry_id, score) tuples
        """
        dataset = self.dataset
        self.model.train(dataset)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.asarray(X_pool)

        if len(unlabeled_entry_ids) == 0:
            return []

        scores = self._get_uncertainty_scores(X_pool)
        return list(zip(unlabeled_entry_ids, scores))

    @inherit_docstring_from(QueryStrategy)
    def make_query(self, return_score=False):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        if len(unlabeled_entry_ids) == 0:
            raise ValueError("No unlabeled samples available")

        # Epsilon-greedy: explore with probability epsilon
        if self.random_state_.random() < self.epsilon:
            # Exploration: random selection
            ask_id = self.random_state_.choice(unlabeled_entry_ids)
        else:
            # Exploitation: uncertainty sampling
            self.model.train(dataset)
            X_pool = np.asarray(X_pool)
            scores = self._get_uncertainty_scores(X_pool)

            max_score = np.max(scores)
            candidates = np.where(np.isclose(scores, max_score))[0]
            selected_idx = self.random_state_.choice(candidates)
            ask_id = unlabeled_entry_ids[selected_idx]

        if return_score:
            return ask_id, self._get_scores()
        else:
            return ask_id

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        self.model.train(self.dataset)
