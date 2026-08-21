"""Random Sampling
"""
import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip


class RandomSampling(QueryStrategy):

    r"""Random sampling

    This class implements the random query strategy. A random entry from the
    unlabeled pool is returned for each query.

    Parameters
    ----------
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    random_states\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of declaring a RandomSampling query_strategy object:

    .. code-block:: python

       from libact.query_strategies import RandomSampling

       qs = RandomSampling(
                dataset, # Dataset object
            )
    """

    def __init__(self, dataset, **kwargs):
        super(RandomSampling, self).__init__(dataset, **kwargs)

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

    def _get_scores(self):
        """Return uniform scores for all unlabeled samples.

        Returns
        -------
        entry_ids : np.ndarray, shape (n_unlabeled,)
            Global entry IDs of unlabeled samples.
        scores : np.ndarray, shape (n_unlabeled,)
            Uniform scores (all ones).
        """
        unlabeled_entry_ids, _ = self.dataset.get_unlabeled_entries()
        scores = np.ones(len(unlabeled_entry_ids), dtype=float)
        return unlabeled_entry_ids, scores

    def make_query_batch(self, batch_size):
        """Return a uniformly random batch of distinct unlabeled samples.

        Overrides the default top-k behavior: with uniform scores a stable
        top-k would always return the first ``batch_size`` pool entries,
        which is not random sampling.

        Parameters
        ----------
        batch_size : int
            Number of samples to query. Must satisfy
            ``1 <= batch_size <= n_unlabeled``.

        Returns
        -------
        entry_ids : np.ndarray of int, shape (batch_size,)
            Distinct entry ids sampled uniformly without replacement.
        """
        unlabeled_entry_ids, _ = self.dataset.get_unlabeled_entries()
        self._check_batch_size(batch_size, len(unlabeled_entry_ids))
        return self.random_state_.choice(
            unlabeled_entry_ids, size=batch_size, replace=False)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()
        entry_id = unlabeled_entry_ids[
            self.random_state_.randint(0, len(unlabeled_entry_ids))]
        return entry_id
