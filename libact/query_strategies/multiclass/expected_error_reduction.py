"""Expected Error Reduction
"""
import copy

import numpy as np

from libact.base.interfaces import QueryStrategy, ProbabilisticModel
from libact.base.dataset import Dataset
from libact.utils import inherit_docstring_from, seed_random_state

class EER(QueryStrategy):
    """Expected Error Reduction(EER)

    This class implements EER active learning algorithm [1]_.

    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The base model used for training.

    loss: {'01', 'log'}, optional (default='log')
        The loss function expected to reduce

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The model trained in last query.

    Examples
    --------
    Here is an example of declaring a UncertaintySampling query_strategy
    object:

    .. code-block:: python

       from libact.query_strategies import EER
       from libact.models import LogisticRegression

       qs = EER(dataset, model=LogisticRegression(C=0.1))

    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.


    References
    ----------
    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """

    def __init__(self, dataset, model=None, loss='log', random_state=None):
        super(EER, self).__init__(dataset)

        self.model = model
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ProbabilisticModel"
            )
        self.loss = loss
        if self.loss not in ['01', 'log']:
            raise TypeError(
                "supported methods are ['01', 'log'], the given one "
                "is: " + self.loss
            )

        self.random_state_ = seed_random_state(random_state)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        X, y = dataset.get_labeled_entries()
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        classes = np.unique(y)
        n_classes = len(classes)

        self.model.train(dataset)
        proba = self.model.predict_proba(X_pool)

        scores = []
        for i, x in enumerate(X_pool):
            score = []
            for yi in range(n_classes):
                m = copy.deepcopy(self.model)
                m.train(Dataset(np.vstack((X, [x])), y + [yi]))
                p = m.predict_proba(X_pool)

                if self.loss == '01':  # 0/1 loss
                    score.append(proba[i, yi] * np.sum(1-np.max(p, axis=1)))
                elif self.loss == 'log': # log loss
                    score.append(proba[i, yi] * -np.sum(p * np.log(p)))
            scores.append(np.sum(score))

        choices = np.where(np.array(scores) == np.min(scores))[0]
        ask_idx = self.random_state_.choice(choices)

        return unlabeled_entry_ids[ask_idx]
