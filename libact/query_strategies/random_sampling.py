import random

from libact.base.interfaces import QueryStrategy


class RandomSampling(QueryStrategy):
    """Random sampling

    This class implements the random query strategy. A random entry from the
    unlabeled pool is returned for each query.
    """

    def __init__(self, dataset, **kwargs):
        super(RandomSampling, self).__init__(dataset, **kwargs)

    def make_query(self):
        entry_id, feature = random.choice(self.dataset.get_unlabeled_entries())
        return entry_id
