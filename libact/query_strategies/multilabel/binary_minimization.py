"""Binary Minimization
"""
import copy

import numpy as np

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.models.multilabel import BinaryRelevance, DummyClf


class BinaryMinimization(QueryStrategy):
    r"""Binary Version Space Minimization (BinMin)


    Parameters
    ----------
    base_clf : ContinuousModel object instance
        The base learner for binary relavance.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------

    Examples
    --------
    Here is an example of declaring a BinaryMinimization query_strategy object:

    .. code-block:: python

       from libact.query_strategies.multilabel import BinaryMinimization
       from sklearn.linear_model import LogisticRegression

       qs = BinaryMinimization(
                dataset, # Dataset object
                br_base=LogisticRegression()
            )

    References
    ----------
    .. [1] Brinker, Klaus. "On active learning in multi-label classification."
           From Data and Information Analysis to Knowledge Engineering. Springer
           Berlin Heidelberg, 2006. 206-213.
    """

    def __init__(self, dataset, base_clf, random_state=None):
        super(BinaryMinimization, self).__init__(dataset)

        self.n_labels = len(self.dataset.data[0][1])

        self.base_clf = base_clf

        self.random_state_ = seed_random_state(random_state)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        X, Y = dataset.get_labeled_entries()
        Y = np.array(Y)
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.array(X_pool)

        clfs = []
        boundaries = []
        for i in range(self.n_labels):
            if len(np.unique(Y[:, i])) == 1:
                clf = DummyClf()
            else:
                clf = copy.deepcopy(self.base_clf)
            clf.train(Dataset(X, Y[:, i]))
            boundaries.append(np.abs(clf.predict_real(X_pool)[:, 1]))
            clfs.append(clf)

        choices = np.where(np.array(boundaries) == np.min(boundaries))[1]
        ask_id = self.random_state_.choice(choices)

        return unlabeled_entry_ids[ask_id]
