"""Maximum loss reduction with Maximal Confidence (MMC)
"""
import copy

import numpy as np
from sklearn.svm import SVC

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.models import LogisticRegression, SVM
from libact.models.multilabel import BinaryRelevance, DummyClf


class MaximumLossReductionMaximalConfidence(QueryStrategy):
    """Maximum loss reduction with Maximal Confidence (MMC)

    This algorithm is designed to use binary relavance with SVM as base model.

    Parameters
    ----------
    base_learner : :py:mod:`libact.query_strategies` object instance
        The base learner for binary relavance, should support predict_proba

    br_base : sklearn classifier, optional (default=sklearn.svm.SVC(kernel='linear', probability=True))
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
    logistic_regression_ : :py:mod:`libact.models.LogisticRegression` object instance
        The model used to predict the number of label in each instance.
        Should support multi-class classification.

    Examples
    --------

    References
    ----------
    .. [1] Yang, Bishan, et al. "Effective multi-label active learning for text
		   classification." Proceedings of the 15th ACM SIGKDD international
		   conference on Knowledge discovery and data mining. ACM, 2009.
    """

    def __init__(self, *args, **kwargs):
        super(MaximumLossReductionMaximalConfidence, self).__init__(*args, **kwargs)

        self.n_labels = len(self.dataset.data[0][1])

        self.logreg_param = kwargs.pop('logreg_param', {})
        self.logistic_regression_ = LogisticRegression(**self.logreg_param)

        self.br_base = kwargs.pop('br_base',
                                  SVC(kernel='linear', probability=True))

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        labeled_pool, Y = zip(*dataset.get_labeled_entries())
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        labeled_pool = np.array(labeled_pool)
        Y = np.array(Y)
        X_pool = np.array(X_pool)

        br = BinaryRelevance(SVM(kernel='linear'))
        br.train(Dataset(labeled_pool, Y))
        #lbl_proba = br.predict_proba(X_pool) # shape=(pool size, labels)
        f = br.predict(X_pool) * 2 - 1 # to (1, -1)

        # predicting number of labels in each label vector with OVA
        labeled_proba = []
        unlabeled_proba = []
        for i in range(1, self.n_labels+1): # shouldn't have 0 label
            lbl_num = (Y.sum(axis=1)==i)
            # if there is only one kind of label
            if len(np.unique(lbl_num)) == 1:
                clf = DummyClf()
            else:
                clf = copy.deepcopy(self.br_base)
            clf.fit(labeled_pool, lbl_num)
            labeled_proba.append(clf.predict_proba(labeled_pool)[:, 1])
            unlabeled_proba.append(clf.predict_proba(X_pool)[:, 1])

        trnf = np.array(labeled_proba).T # shape=(len(labeled_proba), n_labels)
        trnf = np.sort(trnf, axis=1)[:, ::-1]
        if len(np.unique(Y.sum(axis=1))) == 1:
            lr = DummyClf()
        else:
            lr = self.logistic_regression_
        lr.train(Dataset(trnf, Y.sum(axis=1)))

        poolf = np.array(unlabeled_proba).T
        idx_poolf = np.argsort(poolf, axis=1)[:, ::-1]
        pred_num_lbl = lr.predict(
                    poolf[np.arange(poolf.shape[0]).reshape(-1, 1), idx_poolf]
                    ).astype(int)

        yhat = -1 * np.ones((len(X_pool), self.n_labels))
        for i, p in enumerate(pred_num_lbl):
            yhat[i, idx_poolf[:p]] = 1

        score = ((1 - yhat * f) / 2).sum(axis=1)
        ask_id = self.random_state_.choice(np.where(score == np.max(score))[0])
        return unlabeled_entry_ids[ask_id]
