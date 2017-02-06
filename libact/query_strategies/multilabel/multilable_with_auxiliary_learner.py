"""Multi-label Active Learning with Auxiliary Learner
"""
import copy

import numpy as np
from sklearn.svm import SVC

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.models import LogisticRegression, SVM
from libact.models.multilabel import BinaryRelevance, DummyClf


class MultilabelWithAuxiliaryLearner(QueryStrategy):
    r"""Multi-label Active Learning with Auxiliary Learner

    Parameters
    ----------
    main_learner : :py:mod:`libact.models.multilabel` object instance
        The base learner for binary relavance, should support predict_proba

    auxiliary_learner : :py:mod:`libact.models.multilabel` object instance
        The base learner for the binary relevance in MMC.
        Should support predict_proba.

    criterion : ['hlr'], optional(default='hlr')
        hlr, hamming loss reduction

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------

    Examples
    --------
    Here is an example of declaring a multilabel with auxiliary learner
    query_strategy object:

    .. code-block:: python

       from libact.query_strategies.multilabel import MultilabelWithAuxiliaryLearner
       from libact.models.multilabel import BinaryRelevance
       from libact.models import LogisticRegression, SVM

       qs = MultilabelWithAuxiliaryLearner(
                dataset,
                main_learner=BinaryRelevance(LogisticRegression())
                auxiliary_learner=BinaryRelevance(SVM())
            )

    References
    ----------
    .. [1] Hung, Chen-Wei, and Hsuan-Tien Lin. "Multi-label Active Learning
	   with Auxiliary Learner." ACML. 2011.
    """

    def __init__(self, dataset, main_learner, auxiliary_learner,
            criterion='hlr', random_state=None):
        super(MultilabelWithAuxiliaryLearner, self).__init__(*args, **kwargs)

        self.n_labels = len(self.dataset.data[0][1])

        self.main_learner = main_learner
        self.auxiliary_learner = auxiliary_learner

        self.random_state_ = seed_random_state(random_state)

        self.criterion = criterion

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        labeled_pool, Y = zip(*dataset.get_labeled_entries())
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        main_clf = copy.deepcopy(self.main_learner)
        main_clf.train(labeled_pool)
        aux_clf = copy.deepcopy(self.auxiliary_learner)
        aux_clf.train(labeled_pool)

        main_pred = main_clf.predict(X_pool)
        aux_pred = auxiliary_clf.predict(X_pool)

        if self.criterion == 'hlr':
            score = np.abs(main_pred - aux_pred).mean(axis=1)

        ask_id = self.random_state_.choice(
            np.where(score == np.max(score))[0], self.random_state_)

        return unlabeled_entry_ids[ask_id]
