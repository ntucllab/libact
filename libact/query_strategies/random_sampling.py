import random

from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from


class RandomSampling(QueryStrategy):
    """Random sampling

    This class implements the random query strategy. A random entry from the
    unlabeled pool is returned for each query.

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
        # TODO random state as parameter

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        entry_id, feature = random.choice(self.dataset.get_unlabeled_entries())
        return entry_id
