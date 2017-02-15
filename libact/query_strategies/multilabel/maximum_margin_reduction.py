"""Maximum loss reduction with Maximal Confidence (MMC)
"""
import copy

import numpy as np
from sklearn.svm import SVC

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.models import LogisticRegression, SklearnProbaAdapter
from libact.models.multilabel import BinaryRelevance, DummyClf


class MaximumLossReductionMaximalConfidence(QueryStrategy):
    r"""Maximum loss reduction with Maximal Confidence (MMC)

    This algorithm is designed to use binary relavance with SVM as base model.

    Parameters
    ----------
    base_learner : :py:mod:`libact.query_strategies` object instance
        The base learner for binary relavance, should support predict_proba

    br_base : ProbabilisticModel object instance
        The base learner for the binary relevance in MMC.
        Should support predict_proba.

    logreg_param : dict, optional (default={})
        Setting the parameter for the logistic regression that are used to
        predict the number of labels for a given feature vector. Parameter
        detail please refer to:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    logistic_regression\_ : :py:mod:`libact.models.LogisticRegression` object instance
        The model used to predict the number of label in each instance.
        Should support multi-class classification.

    Examples
    --------
    Here is an example of declaring a MMC query_strategy object:

    .. code-block:: python

       from libact.query_strategies.multilabel import MMC
       from sklearn.linear_model import LogisticRegression

       qs = MMC(
                dataset, # Dataset object
                br_base=LogisticRegression()
            )

    References
    ----------
    .. [1] Yang, Bishan, et al. "Effective multi-label active learning for text
		   classification." Proceedings of the 15th ACM SIGKDD international
		   conference on Knowledge discovery and data mining. ACM, 2009.
    """

    def __init__(self, *args, **kwargs):
        super(MaximumLossReductionMaximalConfidence, self).__init__(*args, **kwargs)

        self.n_labels = len(self.dataset.data[0][1])

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        self.logreg_param = kwargs.pop('logreg_param',
                {'multi_class': 'multinomial', 'solver': 'newton-cg',
                 'random_state': random_state})
        self.logistic_regression_ = LogisticRegression(**self.logreg_param)

        self.br_base = kwargs.pop('br_base',
              SklearnProbaAdapter(SVC(kernel='linear', probability=True,
                                      random_state=random_state)))

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        labeled_pool, Y = zip(*dataset.get_labeled_entries())
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        labeled_pool = np.array(labeled_pool)
        Y = np.array(Y)
        X_pool = np.array(X_pool)

        br = BinaryRelevance(self.br_base)
        br.train(Dataset(labeled_pool, Y))

        trnf = br.predict_proba(labeled_pool)
        poolf = br.predict_proba(X_pool)
        f = poolf * 2 - 1

        trnf = np.sort(trnf, axis=1)[:, ::-1]
        trnf /= np.tile(trnf.sum(axis=1).reshape(-1, 1), (1, trnf.shape[1]))
        if len(np.unique(Y.sum(axis=1))) == 1:
            lr = DummyClf()
        else:
            lr = self.logistic_regression_
        lr.train(Dataset(trnf, Y.sum(axis=1)))

        idx_poolf = np.argsort(poolf, axis=1)[:, ::-1]
        poolf = np.sort(poolf, axis=1)[:, ::-1]
        poolf /= np.tile(poolf.sum(axis=1).reshape(-1, 1), (1, poolf.shape[1]))
        pred_num_lbl = lr.predict(poolf).astype(int)

        yhat = -1 * np.ones((len(X_pool), self.n_labels), dtype=int)
        for i, p in enumerate(pred_num_lbl):
            yhat[i, idx_poolf[i, :p]] = 1

        score = ((1 - yhat * f) / 2).sum(axis=1)
        ask_id = self.random_state_.choice(np.where(score == np.max(score))[0])
        return unlabeled_entry_ids[ask_id]
