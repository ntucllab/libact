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
    major_learner : :py:mod:`libact.base.interfaces.Model` object instance
        The major multilabel learner. This learner should be the model to be
        used to solve the problem.

    auxiliary_learner : :py:mod:`libact.models.multilabel` object instance
        The auxiliary multilabel learner.
        For criterion 'shlr' and 'mmr', it is required to support predict_real
        or predict_proba.

    criterion : ['hlr', 'shlr', 'mmr'], optional(default='hlr')
        The criterion for estimating the difference between major_learner and
        auxiliary_learner.
        hlr, hamming loss reduction
        shlr, soft hamming loss reduction
        mmr, maximum margin reduction

    b : float
        parameter for criterion shlr.
        It sets the score to be clipped between [-b, b] to remove influence of
        extreme margin values.

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
                major_learner=BinaryRelevance(LogisticRegression())
                auxiliary_learner=BinaryRelevance(SVM())
            )

    References
    ----------
    .. [1] Hung, Chen-Wei, and Hsuan-Tien Lin. "Multi-label Active Learning
	   with Auxiliary Learner." ACML. 2011.
    """

    def __init__(self, dataset, major_learner, auxiliary_learner,
            criterion='hlr', b=1., random_state=None):
        super(MultilabelWithAuxiliaryLearner, self).__init__(dataset)

        self.n_labels = len(self.dataset.data[0][1])

        self.major_learner = major_learner
        self.auxiliary_learner = auxiliary_learner

        self.b = b

        self.random_state_ = seed_random_state(random_state)

        self.criterion = criterion
        if self.criterion not in ['hlr', 'shlr', 'mmr']:
            raise TypeError(
                "supported criterion are ['hlr', 'shlr', 'mmr'], the given "
                "one is: " + self.criterion
            )

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        labeled_pool, Y = zip(*dataset.get_labeled_entries())
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        major_clf = copy.deepcopy(self.major_learner)
        major_clf.train(dataset)
        aux_clf = copy.deepcopy(self.auxiliary_learner)
        aux_clf.train(dataset)

        if self.criterion == 'hlr':
            major_pred = major_clf.predict(X_pool)
            aux_pred = aux_clf.predict(X_pool)
            score = np.abs(major_pred - aux_pred).mean(axis=1)
        elif self.criterion in ['mmr', 'shlr']:
            major_pred = major_clf.predict(X_pool) * 2 - 1

            if 'predict_real' in dir(aux_clf):
                aux_pred = aux_clf.predict_real(X_pool)
            elif 'predict_proba' in dir(aux_clf):
                aux_pred = aux_clf.predict_proba(X_pool) * 2 - 1
            else:
                raise AttributeError("aux_learner did not support either"
                                     "'predict_real' or 'predict_proba'"
                                     "method")

            loss = (major_pred * aux_pred).mean(axis=1)
            if self.criterion == 'mmr':
                score = (1. - major_pred * aux_pred) / 2.
                score = np.sum(score, axis=1)
            elif self.criterion == 'shlr':
                b = self.b
                score = (b - np.clip(major_pred * aux_pred, -b, b)) / 2. / b
                score = np.sum(score, axis=1)
            else:
                raise TypeError(
                    "supported criterion are ['hlr', 'shlr', 'mmr'], the given "
                    "one is: " + self.criterion
                )

        ask_id = self.random_state_.choice(np.where(score == np.max(score))[0])

        return unlabeled_entry_ids[ask_id]
