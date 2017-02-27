""" Expected Error Reduction

This module contains a class that implements Expected Error Reduction active learning
algorithm.

"""
import copy

import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel, \
    ProbabilisticModel
from libact.utils import inherit_docstring_from, zip


class ExpectedErrorReduction(QueryStrategy):

    """ Expected Error Reduction

    This class implements Expected Error Reduction active learning algosithm [1]_.

    Parameters
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` or :py:class:`libact.base.interfaces.ProbabilisticModel` object instance
        The base model used for training.


    Attributes
    ----------
    model: :py:class:`libact.base.interfaces.ContinuousModel` object instance
        The model trained in last query.


    Examples
    --------
    Here is an example of declaring a ExpectedErrorReduction query_strategy
    object:

    .. code-block:: python

       from libact.query_strategies import ExpectedErrorReduction
       from libact.models import LogisticRegression

       qs = ExpectedErrorReduction(
                dataset, # Dataset object
                model=LogisticRegression(C=0.1)
            )

    Note that the model given in the :code:`model` parameter must be a
    :py:class:`ContinuousModel` which supports predict_real method.


    References
    ----------

    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.

    """

    def __init__(self):
        super(ExpectedErrorReduction, self).__init__(*args, **kwargs)

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

        self.model.train(self.dataset)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        self.model.train(dataset)

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        X_own, y_own = zip(*dataset.get_labeled_entries())

        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_proba(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)

        ask_id, min_E = -1, np.inf
        model_t = copy.deepcopy(self.model)
        for u in range(len(X_pool)):
            cur_E = 0
            for j in range(len(dvalue[u])):
                model_t.train(np.append(X_own, X_pool[u].reshape(1, -1), axis = 0), np.append(y_own, dvalue[u][j]))
                cur_E = cur_E + dvalue[u][j] * sum(1 - np.max(model_t.predict_real(np.delete(X_pool, u, 0)), 1))
            if cur_E < min_E :
                min_E = cur_E
                ask_id = u

        return unlabeled_entry_ids[ask_id]
